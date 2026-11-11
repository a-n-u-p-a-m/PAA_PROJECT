#!/usr/bin/env python3
"""Phase 3: component-level energy of both models, by CLEAR.

    python run_clear.py --synthetic --device cpu --offline --quick   # smoke test
    python run_clear.py --profile kaggle-t4 --quick                  # ~10 min check
    python run_clear.py --profile kaggle-t4 --deadline 600m --resume # the real pass

Three stages, in order:

  1. capture   decode a few prompts at every K with hooks on both models,
               keeping a bounded sample of each component's real inputs
  2. check     replay every capture once; it must reproduce its output
               exactly, or the energy would describe different work
  3. amplify   replay each component N times per trial, N sized to the
               energy counter's update period, and divide

The output is per-call cost by (model, component, rows, context bin). It is
not an energy-vs-K curve by itself -- ``analyze.py SWEEP --clear THIS``
multiplies it by the call counts in a sweep's round_detail to get one.

Draft components are fp16 whichever precision the target runs at, so a
second pass at another target precision only needs ``--roles target``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# cuBLAS picks a kernel partly from how much workspace it may use, and caches
# the pick for the life of the process. Fixing the workspace makes that choice
# reproducible from one run to the next, so a bad pick becomes a stable fact
# the anchor check can catch instead of a lottery between processes. This has
# to be set before torch initialises CUDA, hence its place here.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import encode, load_pair, load_synthetic  # noqa: E402
from specdec import clear  # noqa: E402
from specdec import datasets as ds  # noqa: E402
from specdec.checkpoint import parse_deadline  # noqa: E402
from specdec.configs import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_PAIR,
    MODEL_PAIRS,
    PROFILES,
)
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402
from specdec.meters import EnergyProbe  # noqa: E402

#: measured first, because compose() cannot produce a total without them
PRIORITY = ("forward", "prefill", "layer", "attn", "mlp", "norm_in",
            "norm_post", "lm_head", "embed", "rope", "norm")

QUICK = dict(limit=2, max_new_tokens=64, k=[1, 4], layers="mid", trials=3,
             window_factor=10.0, per_key=2)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES))
    pre.add_argument("--quick", action="store_true")
    known, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--quick", action="store_true",
                    help="2 prompts, 64 tokens, K in {1,4}, one layer, 3 trials: "
                         "checks the whole path end to end in minutes")
    ap.add_argument("--config", default=DEFAULT_PAIR, choices=sorted(MODEL_PAIRS))
    ap.add_argument("--quant", default="nf4", choices=["nf4", "float16", "bfloat16", "float32"])
    ap.add_argument("--compute-dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--draft-precision", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--synthetic", action="store_true", help="tiny random models, no downloads")

    g = ap.add_argument_group("capture")
    g.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(ds.PAPER_SIZES))
    g.add_argument("--limit", type=int, default=8,
                   help="prompts to capture from (default 8). Capture needs "
                        "coverage of shapes and context lengths, not many prompts")
    g.add_argument("--seed", type=int, default=ds.PAPER_SEED)
    g.add_argument("--offline", action="store_true")
    g.add_argument("--no-chat-template", action="store_true")
    g.add_argument("--max-new-tokens", type=int, default=256,
                   help="match the sweep, so captured context lengths span its range")
    g.add_argument("--k", type=int, nargs="+", default=list(range(1, 9)),
                   help="draft lengths to capture at; verify rows = K+1 (default 1..8)")
    g.add_argument("--no-ar", action="store_true",
                   help="skip capturing autoregressive decoding (the vs-AR baseline)")
    g.add_argument("--layers", default="rep",
                   help="decoder layers to profile: rep (first/mid/last), mid, "
                        "all, or a list like 0,5,11")
    g.add_argument("--kv-bin", type=int, default=128,
                   help="context-length bin width for attention-bearing components")
    g.add_argument("--per-key", type=int, default=4,
                   help="captured inputs kept per measurement key")
    g.add_argument("--per-key-forward", type=int, default=1,
                   help="kept per whole-model key; each carries every layer's cache")

    g = ap.add_argument_group("amplification")
    g.add_argument("--roles", nargs="+", default=["draft", "target"],
                   choices=["draft", "target"],
                   help="which models to measure; draft is fp16 regardless of "
                        "--quant, so a second precision needs only target")
    g.add_argument("--trials", type=int, default=5)
    g.add_argument("--window-factor", type=float, default=20.0,
                   help="each trial spans at least this many sensor periods")
    g.add_argument("--min-window-ms", type=float, default=300.0)
    g.add_argument("--settle-ms", type=float, default=1500.0,
                   help="untimed running before each component's trials, so "
                        "the GPU power state is this component's, not the last one's")
    g.add_argument("--flush-l2", action="store_true",
                   help="flush L2 before every call (and subtract the flush), so "
                        "small components do not run on artificially hot caches")
    g.add_argument("--no-energy", action="store_true", help="timing only")
    g.add_argument("--allow-mismatch", action="store_true",
                   help="measure even if some replays did not reproduce their output")
    g.add_argument("--no-anchor", action="store_true",
                   help="skip stage 1b. Without it nothing checks that a replay "
                        "runs the kernel the real forward pass runs, which is how "
                        "the first fp16 pass over-attributed the target MLP 4x")
    g.add_argument("--in-pass-prompts", type=int, default=1,
                   help="prompts to time components in place on (default 1)")
    g.add_argument("--in-pass-spans", type=int, default=24,
                   help="timed calls kept per measurement key (default 24)")

    g = ap.add_argument_group("output")
    g.add_argument("--out", default=None)
    g.add_argument("--resume", action="store_true",
                   help="skip keys already measured in --out")
    g.add_argument("--gate", action="store_true",
                   help="after measuring, judge the file just written (replay, "
                        "noise, additivity, part-vs-whole) and exit 3 if it fails")
    g.add_argument("--gate-max-w", type=float, default=None, metavar="W",
                   help="the gate also fails any measurement above this power, "
                        "e.g. 75 on a 70 W T4")
    g.add_argument("--repeat", type=int, default=1, metavar="N",
                   help="measure N times over, each in a fresh process, and record "
                        "how far each component moves between them. A repeat inside "
                        "one process cannot see this: the kernel pick is made once "
                        "per process and then cached. Cheap with --quick, as a "
                        "pre-flight before spending a slot on the full pass")
    g.add_argument("--check", metavar="JSON",
                   help="gate an existing output file and exit; loads no model")
    g.add_argument("--deadline", metavar="SPEC",
                   help="stop cleanly before this: '90m', '1h45m', '23:30'")
    g.add_argument("--reserve", type=float, default=2.0,
                   help="minutes held back from the deadline (default 2)")

    if known.profile:
        p = PROFILES[known.profile]
        ap.set_defaults(config=p.config, quant=p.quant, dataset=p.dataset,
                        max_new_tokens=p.max_new_tokens)
    if known.quick:
        ap.set_defaults(**QUICK)
    return ap.parse_args()


def write(path: Path, blob: dict) -> None:
    """Atomically, so a kill mid-write never leaves a truncated file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(blob, fh, indent=1)
    os.replace(tmp, path)


def run_gate(blob: dict, name, roles=None, max_power_w=None) -> bool:
    """Print the gate's verdict on one output; True if it passed."""
    g = clear.gate(blob, roles=roles, max_power_w=max_power_w)
    print(f"\ngate: {name}")
    for line in g["lines"]:
        print(f"  {line}")
    for w in g["warnings"]:
        print(f"  !! {w}")
    if g["problems"]:
        print("\n  GATE FAILED -- do not compose from this file:")
        for prob in g["problems"]:
            print(f"    - {prob}")
        return False
    print("\n  gate passed" + (" (with warnings)" if g["warnings"] else ""))
    return True


def spread_across(blobs: list[dict]) -> dict:
    """Per measurement, how far its per-call time moved between passes.

    Reported as a percentage of the smallest reading, which is the one most
    likely to be the component's real cost: a bad kernel pick only ever
    makes a call slower than it needs to be.
    """
    by: dict[str, list[float]] = {}
    for b in blobs:
        for m in b.get("measurements", []):
            if m.get("time_ms"):
                by.setdefault(m["label"], []).append(m["time_ms"])
    seen = {lab: v for lab, v in by.items() if len(v) > 1 and min(v) > 0}
    return {"passes": len(blobs),
            "by_label": {lab: (max(v) - min(v)) / min(v) * 100
                         for lab, v in seen.items()},
            # so the gate can ignore calls too short for a percentage to mean
            # anything
            "min_ms": {lab: min(v) for lab, v in seen.items()}}


def repeat_in_fresh_processes(n: int, out: Path) -> list[Path]:
    """Run this same command again, n-1 times, each in its own process.

    The point is the process boundary, so this re-execs rather than looping:
    cuBLAS chooses a kernel once per process and caches it, so re-measuring
    inside this one would return the same number however many times it ran.
    """
    flags = {"--gate", "--resume"}                 # take no value
    valued = {"--repeat", "--out"}                 # take one
    argv: list[str] = []
    skip = False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        head = a.split("=", 1)[0]
        if head in flags:
            continue
        if head in valued:
            skip = "=" not in a
            continue
        argv.append(a)

    done: list[Path] = []
    for i in range(2, n + 1):
        p = out.with_name(f"{out.stem}-pass{i}{out.suffix}")
        print(f"\n--- repeat {i}/{n}, fresh process -> {p}", flush=True)
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             *argv, "--out", str(p)]).returncode
        if rc == 0:
            done.append(p)
        else:
            print(f"  repeat {i} exited {rc}; leaving it out of the spread")
    return done


def sensor_period(probe: EnergyProbe):
    from probe_env import sensor_period_ms

    try:
        return sensor_period_ms(probe._handle)
    except Exception as exc:
        print(f"  could not measure the sensor period: {exc}")
        return None


def main() -> None:
    args = parse_args()
    if args.check:
        old = json.loads(Path(args.check).read_text())
        if old.get("kind") != "clear":
            sys.exit(f"{args.check} is not a run_clear.py output")
        sys.exit(0 if run_gate(old, args.check, max_power_w=args.gate_max_w) else 3)
    if args.device == "cpu" and args.quant in ("float16", "nf4"):
        args.quant = "float32"
        args.draft_precision = "float32"
    pair = MODEL_PAIRS[args.config]
    tag = "synthetic" if args.synthetic else f"{args.config}-{args.quant}"
    out = Path(args.out or f"results/clear-{tag}{'-quick' if args.quick else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = parse_deadline(args.deadline) if args.deadline else None

    # ---- models and the energy counter -----------------------------------
    if args.synthetic:
        tok, draft, target = load_synthetic(args.device, "float32")
    else:
        tok, draft, target = load_pair(pair, args)

    probe, period, idle_w = None, None, None
    if args.device.startswith("cuda") and not args.no_energy:
        probe = EnergyProbe()
        if probe.supported:
            print(f"NVML energy counter on physical GPU {probe.nvml_index}")
            period = sensor_period(probe)
            idle_w = probe.measure_idle_w()
            print(f"  counter updates every {period or float('nan'):.0f} ms; "
                  f"idle draw {idle_w or float('nan'):.1f} W")
        else:
            print(f"energy unavailable, timing only: {probe.reason}")
            probe = None

    # ---- 1. capture --------------------------------------------------------
    layers = {
        "draft": clear.representative_layers(draft.config.num_hidden_layers, args.layers),
        "target": clear.representative_layers(target.config.num_hidden_layers, args.layers),
    }
    samples = ds.load(args.dataset, limit=args.limit, seed=args.seed, offline=args.offline)
    print(f"\n1. capture: {len(samples)} {args.dataset} prompts x K={args.k}"
          f"{'' if args.no_ar else ' + autoregressive'}, {args.max_new_tokens} tokens; "
          f"layers draft {layers['draft']}, target {layers['target']}")
    store = clear.ActivationStore(kv_bin=args.kv_bin, per_key=args.per_key,
                                  per_key_forward=args.per_key_forward, seed=args.seed)
    store.attach(draft, "draft", layers["draft"])
    store.attach(target, "target", layers["target"])
    gen = dict(max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_token_id)
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        ids = encode(tok, s.prompt, args.device, not args.no_chat_template)
        for k in args.k:
            speculative_generate(draft, target, ids, k=k, **gen)
        if not args.no_ar:
            autoregressive_generate(target, ids, **gen)
        print(f"  prompt {i + 1}/{len(samples)} ({ids.shape[1]} tokens) "
              f"-> {len(store.buckets)} keys, {time.perf_counter() - t0:.0f} s", flush=True)
    store.detach()

    # ---- 1b. what each component costs where it really runs ----------------
    # The replay check proves a replayed call returns the same values. It
    # cannot prove it does the same work: cuBLAS may pick a different kernel
    # for a tight loop of one component than for that component inside a
    # forward pass, and the pick is cached, so the wrong one reads precise
    # and steady. These timings are what gate() checks every amplified
    # number against.
    in_pass: dict[str, dict] = {}
    if not args.no_anchor:
        timer = clear.InPassTimer(kv_bin=args.kv_bin, max_spans=args.in_pass_spans,
                                  cuda=args.device.startswith("cuda"))
        timer.attach(draft, "draft", layers["draft"])
        timer.attach(target, "target", layers["target"])
        n_pass = max(1, min(args.in_pass_prompts, len(samples)))
        print(f"\n1b. in-pass timing: {n_pass} prompt(s) at K={args.k}, every "
              f"component timed by events where it actually runs")
        t0 = time.perf_counter()
        with timer:
            for s in samples[:n_pass]:
                ids = encode(tok, s.prompt, args.device, not args.no_chat_template)
                for k in args.k:
                    speculative_generate(draft, target, ids, k=k, **gen)
                if not args.no_ar:
                    autoregressive_generate(target, ids, **gen)
        in_pass = timer.results()
        print(f"  {len(in_pass)} keys timed in place, {time.perf_counter() - t0:.0f} s")

    # ---- 2. replay check ---------------------------------------------------
    print("\n2. replay check")
    report = clear.check_replay(store, args.device)
    bad = [r for r in report if not (r["exact"] and r["cache_restored"])]
    print(f"  {len(report) - len(bad)}/{len(report)} replays reproduce their output exactly")
    for r in bad[:10]:
        print(f"  MISMATCH {r['key']}  kv={r['kv']}  max|diff|={r['max_abs_diff']:.3g}"
              f"{'' if r['cache_restored'] else '  cache NOT restored'}")
    if bad and not args.allow_mismatch:
        sys.exit("\nreplay does not reproduce decoding, so its energy would describe\n"
                 "different work. Investigate, or pass --allow-mismatch if the\n"
                 "differences are rounding-level and you have decided that is fine.")

    # ---- 3. amplify ----------------------------------------------------------
    settings = {k: getattr(args, k) for k in (
        "config", "quant", "draft_precision", "compute_dtype", "dataset", "limit",
        "seed", "max_new_tokens", "k", "layers", "kv_bin", "per_key", "trials",
        "window_factor", "min_window_ms", "settle_ms", "flush_l2", "synthetic")}
    # Bumped whenever the measurement procedure changes, so --resume refuses
    # to mix numbers made two different ways in one file. Rev 2: empty_cache
    # before each pilot, and a second pilot after the settle window.
    settings["method_rev"] = 2
    blob = {
        "kind": "clear",
        "meta": {
            "model_pair": {"key": pair.key, "target": pair.target_id, "draft": pair.draft_id},
            "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
            "energy": probe is not None, "sensor_period_ms": period, "idle_watts": idle_w,
            "n_layers": dict(store.n_layers), "layers": layers,
            # Labels carry no precision, so analyze.py needs this to keep two
            # files' measurements of the same model apart. "roles" is filled
            # in from what was actually measured.
            "precision": {"draft": args.draft_precision, "target": args.quant},
            "roles": [],
            # how many keys stage 1b managed to time in place; 0 means the
            # anchor check cannot run on this file. Only CUDA-event timings
            # are sharp enough for the gate to fail a run on, so it checks
            # in_pass_events before treating a miss as a problem.
            "in_pass_timed": len(in_pass),
            "in_pass_events": bool(in_pass) and args.device.startswith("cuda"),
            "torch": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "settings": settings,
        "replay_check": {"total": len(report), "mismatched": len(bad)},
        "measurements": [],
        "complete": False,
    }
    if probe is not None:
        print("\n2b. meter check: does reading the counter change the reading?")
        print(f"  {'reference':<28}{'poll':>7}{'reads/s':>9}{'idle W':>8}"
              f"{'W':>7}{'mJ/call':>11}{'ms/call':>9}{'rsd':>7}")
        cfg0 = clear.AmplifyConfig(window_factor=args.window_factor,
                                   min_window_ms=args.min_window_ms,
                                   settle_ms=args.settle_ms)
        blob["meter_check"] = []
        for want in (("draft", "forward"), ("target", "attn")):
            ref = next(((k, c) for k, c in store.items()
                        if (k.role, k.component) == want and c), None)
            if ref is None:
                continue
            key, caps = ref
            fns = [clear.replay_fn(store, c, args.device) for c in caps]
            for r in clear.meter_check(fns, args.device, probe, period, idle_w, cfg=cfg0):
                r["reference"] = key.label
                blob["meter_check"].append(r)
                idle = f"{r['idle_w']:.1f}" if r["idle_w"] else "-"
                print(f"  {key.label:<28}{r['poll_ms']:>5.0f}ms{r['reads_per_s']:>9.0f}"
                      f"{idle:>8}{r['power_w']:>7.1f}{r['energy_mj']:>11.3f}"
                      f"{r['time_ms']:>9.3f}{r['rel_std_pct']:>6.1f}%", flush=True)
            del fns

    done: set[str] = set()
    if args.resume and out.exists():
        old = json.loads(out.read_text())
        if old.get("settings") != settings:
            diff = [k for k in settings if old.get("settings", {}).get(k) != settings[k]]
            sys.exit(f"refusing to resume {out}: settings changed ({', '.join(diff)})")
        # Keep every earlier measurement, including roles not asked for this
        # time: resuming with --roles target must not drop the draft's.
        blob["measurements"] = list(old["measurements"])
        done = {m["label"] for m in blob["measurements"]}
    blob["meta"]["roles"] = sorted({m["role"] for m in blob["measurements"]})

    keys = [(k, caps) for k, caps in store.items()
            if caps and k.role in args.roles and k.label not in done]
    keys.sort(key=lambda kc: (kc[0].role, PRIORITY.index(kc[0].component),
                              kc[0].layer or 0, kc[0].rows or 0, kc[0].kv_bin or 0))
    cfg = clear.AmplifyConfig(trials=args.trials, window_factor=args.window_factor,
                              min_window_ms=args.min_window_ms,
                              settle_ms=args.settle_ms)
    window_s = max(cfg.min_window_ms, cfg.window_factor * (period or 0)) / 1000
    settle_s = cfg.settle_ms / 1000 if probe else 0
    guess = len(keys) * (settle_s + cfg.trials * (window_s + (period or 0) / 1000) + 0.5)
    print(f"\n3. amplify {len(keys)} keys"
          f"{f' ({len(done)} already done)' if done else ''}, {settle_s:.1f} s settle + "
          f"{cfg.trials} tick-aligned trials of >={window_s * 1000:.0f} ms each: "
          f"roughly {guess / 60:.0f} min"
          f"{', L2 flushed per call' if args.flush_l2 else ''}")
    flush = clear.l2_flusher(args.device) if args.flush_l2 else None

    spent: list[float] = []
    stop = None
    for i, (key, caps) in enumerate(keys):
        if deadline:
            per = statistics.mean(spent) if spent else guess / max(len(keys), 1)
            if time.time() + per + args.reserve * 60 > deadline:
                stop = f"deadline: next key would not finish ({per:.0f} s each)"
                break
        t_key = time.perf_counter()
        fns = [clear.replay_fn(store, c, args.device) for c in caps]
        m = clear.amplify(fns, args.device, probe, period, idle_w, cfg, overhead=flush)
        del fns
        kvs = [c.kv for c in caps]
        blob["measurements"].append({
            "label": key.label, "role": key.role, "component": key.component,
            "layer": key.layer, "rows": key.rows, "kv_bin": key.kv_bin,
            "kv_mean": statistics.mean(kvs) if key.kv_bin is not None else None,
            "rows_mean": statistics.mean(c.rows for c in caps),
            # in_pass_ms / in_pass_n, when stage 1b ran: this key's cost
            # measured inside a real forward pass, for gate()'s anchor check.
            "n_inputs": len(caps), **m, **in_pass.get(key.label, {}),
        })
        if key.role not in blob["meta"]["roles"]:
            blob["meta"]["roles"] = sorted(blob["meta"]["roles"] + [key.role])
        write(out, blob)
        spent.append(time.perf_counter() - t_key)

        spread = f"{m['rel_std_pct']:.1f}%" if m["rel_std_pct"] is not None else "-"
        if m["energy_mj"] is not None:
            cost = (f"{m['energy_mj']:>10.3f} mJ/call  {m['time_ms']:>8.3f} ms  "
                    f"{m['power_w']:>5.1f} W  meter {m['meter_overhead_pct']:+5.1f}%")
        else:
            cost = f"{m['time_ms']:>10.4f} ms/call"
        ip = (in_pass.get(key.label) or {}).get("in_pass_ms")
        anchor = "" if not ip else (
            f"  anchor {m['time_ms'] / ip:4.2f}x"
            + ("" if 0.8 <= m["time_ms"] / ip <= 1.2 else " !!"))
        print(f"  [{i + 1:>4}/{len(keys)}] {key.label:<34} {cost}  "
              f"N={m['calls_per_trial']:<6} rsd {spread:<6}{anchor}"
              f"{'' if m['ok'] else '  !! ' + m['why']}", flush=True)
        if args.device.startswith("cuda") and i % 20 == 19:
            torch.cuda.empty_cache()

    # "complete" covers the roles this run was asked for; meta.roles says
    # which roles the file holds, so a target-only file cannot pass for a full one.
    uncovered = [r for r in args.roles if r not in blob["meta"]["roles"]]
    if uncovered and stop is None:
        stop = f"no measurements for {', '.join(uncovered)}"
    blob["complete"] = stop is None
    blob["stopped_because"] = stop
    write(out, blob)

    # ---- summary -----------------------------------------------------------
    ms = blob["measurements"]
    unusable = [m for m in ms if not m["ok"]]
    print(f"\nwrote {out}: {len(ms)} measurements, {len(unusable)} outside CLEAR's bar")
    metered = [m for m in ms if m.get("meter_overhead_pct") is not None]
    if metered:
        over = sorted(m["meter_overhead_pct"] for m in metered)
        print(f"metering overhead vs unmetered timing: median {statistics.median(over):+.1f}%, "
              f"range {over[0]:+.1f}% .. {over[-1]:+.1f}%; one NVML read takes "
              f"{statistics.median(m['nvml_read_ms'] for m in metered):.2f} ms")
    if stop:
        print(f"stopped early -- {stop}. Re-run the same command with --resume.")
    try:
        costs = clear.Costs(blob)
        checks = clear.additivity(costs)
    except KeyError as exc:
        print(f"additivity skipped: {exc}")
        checks = []
    if checks:
        print(f"\nadditivity: sum of parts / whole, in {costs.unit}")
        for c in (c for c in checks if c["check"].endswith("forward")):
            print(f"  {c['check']:<16} q{c['rows']:<3} kv {c['kv']:>6.0f}  "
                  f"{c['ratio']:.3f}")
        layer = [c["ratio"] for c in checks if c["check"].endswith("layer") and c["ratio"]]
        if layer:
            print(f"  decoder layers   {len(layer)} checks, ratio "
                  f"{min(layer):.3f} .. {max(layer):.3f}")
    have = blob["meta"]["roles"]
    if set(have) != {"draft", "target"}:
        print(f"\nthis file holds {', '.join(have) or 'no'} measurements only; "
              f"pass the file with the other model before it:")
        print(f"next:  python analyze.py <sweep.json> --clear <other-role.json> {out}")
    else:
        print(f"\nnext:  python analyze.py <sweep.json> --clear {out}")
    # ---- 4. the same measurement again, in fresh processes ------------------
    if args.repeat > 1:
        extra = repeat_in_fresh_processes(args.repeat, out)
        if extra:
            blob["repeat_spread"] = spread_across(
                [blob] + [json.loads(p.read_text()) for p in extra])
            write(out, blob)
            by_label = blob["repeat_spread"]["by_label"]
            if by_label:
                print(f"\nbetween-process spread over "
                      f"{blob['repeat_spread']['passes']} passes, widest first "
                      f"(a component's cost should not depend on the process "
                      f"that measured it):")
                for lab, pct in sorted(by_label.items(), key=lambda kv: -kv[1])[:8]:
                    print(f"  {lab:<34} {pct:>7.1f}%")

    if args.gate and not run_gate(blob, out, roles=args.roles,
                                  max_power_w=args.gate_max_w):
        sys.exit(3)


if __name__ == "__main__":
    main()
