#!/usr/bin/env python3
"""Why does a component cost more replayed alone than inside the real pass?

    python diagnose_replay.py --profile kaggle-t4 --quant float16
    python diagnose_replay.py --profile kaggle-t4 --quant nf4 --components lm_head

CLEAR's replay check proves a replayed call returns the same *values* as the
call decoding made. It does not prove the call does the same *work*: the
first fp16 pass replayed the target MLP at rows=1 in 2.29 ms, inside a layer
that takes 1.31 ms in total, and lm_head at rows=1 in 10.9 ms against 3.7 ms
at rows=2. Energy follows time, so those rows came out 2x over-attributed.

This script times one component several ways, for each row count asked for,
so the cause can be read off instead of guessed:

    replay        exactly what run_clear.py amplifies (captured input, tight loop)
    fresh input   the same values in a newly allocated, contiguous tensor
    random input  same shape and dtype, different values
    one at a time the replay, synchronised around every single call
    after forward the replay, each call preceded by a whole forward pass
    in the pass   the module timed by hooks while a whole forward pass runs

How to read it:

    replay >> in the pass, fresh input ~ in the pass
        the captured tensor's layout (strides, offset) sends the call down a
        slower kernel. Fix: replay on a fresh contiguous copy.
    replay >> in the pass, after forward ~ in the pass
        the component behaves differently in a tight loop of itself than
        between other work (clock / cache / allocator state). Fix: measure
        it interleaved or by in-pass differencing, not in isolation.
    replay ~ random ~ one at a time >> in the pass
        the shape itself is slow when called standalone but not in the pass;
        look at what the pass feeds it (the "in the pass" input line below).
    replay ~ in the pass
        the replay is honest; the additivity failure is somewhere else.

Timing only, no energy counter, so it runs in a couple of minutes.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import encode, load_pair  # noqa: E402
from specdec import clear  # noqa: E402
from specdec import datasets as ds  # noqa: E402
from specdec.configs import DEFAULT_DATASET, DEFAULT_PAIR, MODEL_PAIRS, PROFILES  # noqa: E402
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402


def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def describe(x: torch.Tensor) -> str:
    return (f"shape {tuple(x.shape)}  {str(x.dtype).replace('torch.', '')}  "
            f"strides {tuple(x.stride())}  offset {x.storage_offset()}  "
            f"{'contiguous' if x.is_contiguous() else 'NOT contiguous'}")


def loop_ms(fn, device, seconds: float) -> float:
    """ms per call in a tight loop, synchronised once per batch of calls."""
    for _ in range(10):
        fn()
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    one = max(time.perf_counter() - t0, 1e-5)
    n = max(5, min(20000, int(seconds / 3 / one)))
    out = []
    for _ in range(3):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync(device)
        out.append((time.perf_counter() - t0) / n * 1000)
    return statistics.median(out)


def single_ms(fn, device, seconds: float, before=None) -> float:
    """ms per call with a sync around every call; ``before`` runs untimed first."""
    for _ in range(5):
        fn()
    out = []
    end = time.perf_counter() + seconds
    while len(out) < 10 or time.perf_counter() < end:
        if before is not None:
            before()
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - t0) * 1000)
        if len(out) >= 5000:
            break
    return statistics.median(out)


def in_pass_ms(module, forward, device, seconds: float) -> tuple[float, str]:
    """Time ``module`` by hooks while ``forward`` runs the whole model."""
    acc = {"t": 0.0, "n": 0, "t0": 0.0, "seen": ""}

    def pre(mod, args, kwargs):
        if not acc["seen"]:
            x = next((a for a in list(args) + list(kwargs.values())
                      if isinstance(a, torch.Tensor)), None)
            acc["seen"] = describe(x) if x is not None else "?"
        sync(device)
        acc["t0"] = time.perf_counter()

    def post(mod, args, kwargs, output):
        sync(device)
        acc["t"] += time.perf_counter() - acc["t0"]
        acc["n"] += 1

    handles = [module.register_forward_pre_hook(pre, with_kwargs=True),
               module.register_forward_hook(post, with_kwargs=True)]
    try:
        for _ in range(3):
            forward()
        acc["t"], acc["n"] = 0.0, 0
        end = time.perf_counter() + seconds
        runs = 0
        while runs < 10 or time.perf_counter() < end:
            forward()
            runs += 1
    finally:
        for h in handles:
            h.remove()
    return acc["t"] / max(acc["n"], 1) * 1000, acc["seen"]


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES))
    known, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--config", default=DEFAULT_PAIR, choices=sorted(MODEL_PAIRS))
    ap.add_argument("--quant", default="float16",
                    choices=["nf4", "float16", "bfloat16", "float32"])
    ap.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--draft-precision", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(ds.PAPER_SIZES))
    ap.add_argument("--seed", type=int, default=ds.PAPER_SEED)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--role", default="target", choices=["draft", "target"])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2],
                    help="row counts to compare; 1 is the one that misbehaved")
    ap.add_argument("--components", nargs="+", default=["mlp", "lm_head"],
                    choices=["mlp", "lm_head", "attn", "norm_in", "norm_post", "layer"])
    ap.add_argument("--layer", type=int, default=None, help="default: the middle layer")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=1.5, help="per timing")
    if known.profile:
        p = PROFILES[known.profile]
        ap.set_defaults(config=p.config, dataset=p.dataset)
    args = ap.parse_args()

    pair = MODEL_PAIRS[args.config]
    tok, draft, target = load_pair(pair, args)
    dev = args.device
    model = {"draft": draft, "target": target}[args.role]
    n_layers = model.config.num_hidden_layers
    layer = n_layers // 2 if args.layer is None else args.layer

    # ---- capture one prompt at the K values that produce the rows asked for.
    # Target rows: 1 = autoregressive, K+1 = verify. Draft rows: 1, or 2 right
    # after a fully accepted round, which any K produces.
    ks = sorted({max(r - 1, 1) for r in args.rows}) if args.role == "target" else [4]
    sample = ds.load(args.dataset, limit=1, seed=args.seed, offline=args.offline)[0]
    ids = encode(tok, sample.prompt, dev, not args.no_chat_template)
    store = clear.ActivationStore(per_key=1, per_key_forward=1, seed=args.seed)
    for role, m in (("draft", draft), ("target", target)):
        store.attach(m, role, [layer if role == args.role
                               else m.config.num_hidden_layers // 2])
    gen = dict(max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_token_id)
    with torch.no_grad():
        for k in ks:
            speculative_generate(draft, target, ids, k=k, **gen)
        autoregressive_generate(target, ids, **gen)
    store.detach()

    def cap_for(comp: str, rows: int):
        for key, caps in store.items():
            if caps and (key.role, key.component, key.rows) == (args.role, comp, rows):
                return caps[0]
        return None

    print(f"\n{args.role} of {args.config} ({args.quant if args.role == 'target' else args.draft_precision}), "
          f"layer {layer} of {n_layers}, device "
          f"{torch.cuda.get_device_name(0) if dev.startswith('cuda') else dev}")

    with torch.no_grad():
        for comp in args.components:
            idx = None if comp == "lm_head" else layer
            module = store.modules[(args.role, comp, idx)]
            w = getattr(module, "weight", None)
            print(f"\n=== {args.role}/{comp}" + (f"/L{idx}" if idx is not None else "")
                  + (f"   weight {tuple(w.shape)} {str(w.dtype).replace('torch.', '')}"
                     if isinstance(w, torch.Tensor) else ""))
            for rows in args.rows:
                cap = cap_for(comp, rows)
                fwd_cap = cap_for("forward", rows)
                if cap is None:
                    print(f"\n  q{rows}: never captured (try a longer --max-new-tokens)")
                    continue
                x_cpu = next((a for a in cap.args if isinstance(a, torch.Tensor)), None)
                replay = clear.replay_fn(store, cap, dev)
                results: list[tuple[str, float]] = []
                print(f"\n  q{rows}")
                if x_cpu is not None:
                    x_dev = x_cpu.to(dev)
                    print(f"    captured input: {describe(x_dev)}  "
                          f"|x| max {x_dev.abs().max().item():.3g}")
                results.append(("replay (what CLEAR amplifies)",
                                loop_ms(replay, dev, args.seconds)))

                # The variants below swap the first positional tensor only,
                # so they apply to modules called as module(x): mlp, lm_head, norms.
                simple = (x_cpu is not None and len(cap.args) == 1 and not cap.kwargs)
                if simple:
                    fresh = x_cpu.to(dev).contiguous().clone()
                    rand = torch.randn_like(fresh)
                    results.append(("fresh contiguous input",
                                    loop_ms(lambda: module(fresh), dev, args.seconds)))
                    results.append(("random input, same shape",
                                    loop_ms(lambda: module(rand), dev, args.seconds)))
                results.append(("one at a time (sync per call)",
                                single_ms(replay, dev, args.seconds)))

                if fwd_cap is not None:
                    forward = clear.replay_fn(store, fwd_cap, dev)
                    results.append(("after a forward pass",
                                    single_ms(replay, dev, args.seconds, before=forward)))
                    inside, seen = in_pass_ms(module, forward, dev, args.seconds)
                    results.append(("in the pass (hooks)", inside))
                    whole = loop_ms(forward, dev, args.seconds)
                else:
                    seen, inside, whole = "", None, None
                    print("    no forward capture at this row count; in-pass timing skipped")

                for name, ms in results:
                    print(f"    {name:<32} {ms:>9.3f} ms")
                if whole is not None:
                    print(f"    {'whole forward pass':<32} {whole:>9.3f} ms")
                    print(f"    input seen in the pass: {seen}")
                if inside:
                    ratio = results[0][1] / inside
                    verdict = ("replay matches the pass" if ratio < 1.25 else
                               f"replay is {ratio:.1f}x the in-pass cost: compare the rows "
                               f"above to see which variant brings it back down")
                    print(f"    -> {verdict}")

    print("\nBring this output back with the gate's output; the header of this "
          "script says what each pattern means.")


if __name__ == "__main__":
    main()
