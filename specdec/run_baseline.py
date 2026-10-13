#!/usr/bin/env python3
"""Sweep constant draft lengths to produce the points of an energy-vs-K curve.

Phase-2 data collection: acceptance rate, latency, throughput and the
draft/verify phase split at each K, for one model pair on one dataset. These
are the numbers the component-level energy results get read against.

Every protocol knob is held at the value used by

    Dutta et al., "Benchmarking the Energy Savings with Speculative Decoding
    Strategies", Findings of the ACL: EACL 2026.  arXiv:2602.09113

-- NF4 target quantization with double quant, greedy decoding, batch size 1,
model-specific chat templates. Anything set away from those is listed at
startup and recorded in the output file under ``protocol_deviations``.

    python run_baseline.py --list               # pairs, fit, and the protocol
    python run_baseline.py --synthetic --device cpu --offline --limit 3

    # the sweep: eight constant draft lengths, one pair, one dataset
    python run_baseline.py --config qwen2.5-3b --dataset gsm8k
    python analyze.py results/qwen2.5-3b-nf4-gsm8k.json

Only COGA appears in the default sweep. DYGA changes K *within* a run, so its
mean draft length is an outcome rather than a setting -- it is a scatter point
to mark against the curve, never a point on it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specdec import datasets as ds  # noqa: E402
from specdec.checkpoint import Checkpoint, Unit, parse_deadline  # noqa: E402
from specdec.configs import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_PAIR,
    DEFAULT_SWEEP,
    DUTTA_PROTOCOL,
    MODEL_PAIRS,
    PAPER_STRATEGIES,
    PROFILES,
    PROJECT_PAIRS,
    ModelPair,
    feasibility_table,
    parse_strategy,
    protocol_summary,
)
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402
from specdec.kv import logits_for_row  # noqa: E402
from specdec.meters import EnergyProbe, PhaseMeter  # noqa: E402

#: Vicuna v1.3's tokenizer ships no chat template; this is the official one.
VICUNA_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions. USER: {prompt} ASSISTANT:"
)


# ---------------------------------------------------------------- loading --


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def build_quant_config(quant: str, compute_dtype: str):
    """NF4 exactly as the paper used it (QLoRA-style, Dettmers et al. 2023).

    Compute dtype must be float16 on a V100: Volta has no native bf16, and
    bitsandbytes will either fall back silently or throw.
    """
    if quant != "nf4":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype(compute_dtype),
    )


def load_pair(pair: ModelPair, args) -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not pair.supported:
        raise SystemExit(
            f"{pair.key} is {pair.architecture}. {pair.notes}\n"
            "Our engine assumes a decoder-only KV cache; supporting Flan-T5 "
            "means a separate path that keeps encoder outputs fixed and caches "
            "only the decoder. Out of scope for the baseline."
        )

    target_kw: dict = {}
    if args.quant == "nf4":
        target_kw["quantization_config"] = build_quant_config("nf4", args.compute_dtype)
    else:
        target_kw["dtype"] = _dtype(args.quant)

    print(f"loading target {pair.target_id}  ({args.quant})")
    tok = AutoTokenizer.from_pretrained(pair.target_id)
    target = AutoModelForCausalLM.from_pretrained(pair.target_id, **target_kw)
    if args.quant != "nf4":
        target = target.to(args.device)
    target.eval()

    # The draft model is small; the paper does not state its precision, so we
    # keep it unquantized. Quantizing a 68M draft would save ~100 MB and
    # change its energy profile out of proportion to the memory won.
    print(f"loading draft  {pair.draft_id}  ({args.draft_precision})")
    draft = (
        AutoModelForCausalLM.from_pretrained(
            pair.draft_id, dtype=_dtype(args.draft_precision)
        )
        .to(args.device)
        .eval()
    )

    draft_tok = AutoTokenizer.from_pretrained(pair.draft_id)
    if tok.get_vocab() != draft_tok.get_vocab():
        raise SystemExit(
            f"tokenizer mismatch between {pair.target_id} and {pair.draft_id}.\n"
            "Two-model speculative decoding verifies the draft's token *ids*; "
            "a vocabulary mismatch produces silent garbage, not an error."
        )
    return tok, draft, target


class _ByteTokenizer:
    """Stand-in tokenizer for ``--synthetic``."""

    eos_token_id = 0
    chat_template = None
    name_or_path = "synthetic"

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [min(b, 255) for b in text.encode()[:96]] or [1]

        class _Out:
            input_ids = torch.tensor([ids])

        return _Out()

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def decode(self, ids, **kw):
        return bytes(int(i) % 256 for i in ids).decode(errors="replace")


def load_synthetic(device: str, dtype: str):
    """Two tiny random Qwen2 models. Smoke-tests the whole driver offline."""
    import copy

    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    target = Qwen2ForCausalLM(cfg).to(device=device, dtype=_dtype(dtype)).eval()
    draft = copy.deepcopy(target)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in draft.parameters():
            p.add_(torch.randn_like(p) * 0.05 * p.float().std().clamp(min=1e-6).to(p.dtype))
    print("synthetic mode: 2 tiny random Qwen2 models, no downloads")
    return _ByteTokenizer(), draft.eval(), target


# ---------------------------------------------------------------- prompts --


def encode(tok, prompt: str, device: str, use_chat_template: bool) -> torch.Tensor:
    if not use_chat_template:
        return tok(prompt, return_tensors="pt").input_ids.to(device)

    name = str(getattr(tok, "name_or_path", "")).lower()
    if getattr(tok, "chat_template", None):
        kw: dict = {}
        if "qwen3" in name:
            # Qwen3 emits a <think> block by default, which would dominate
            # generation length and make the datasets incomparable.
            kw["enable_thinking"] = False
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, **kw,
            )
        except TypeError:
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
    elif "vicuna" in name:
        text = VICUNA_TEMPLATE.format(prompt=prompt)
    else:
        text = prompt
    return tok(text, return_tensors="pt").input_ids.to(device)


# ------------------------------------------------------------------- main --


def protocol_deviations(args) -> list[str]:
    """Anything set away from the paper's value, so the report can say so."""
    out: list[str] = []
    if args.quant != DUTTA_PROTOCOL["target_quantization"]:
        out.append(
            f"target precision {args.quant}, not {DUTTA_PROTOCOL['target_quantization']} "
            "-- deliberate: unquantized energy describes the model rather than "
            "the dequantization kernel"
        )
    if args.sample:
        out.append("sampling enabled; the paper decoded greedily")
    if args.no_chat_template:
        out.append("chat template disabled; the paper used model-specific templates")
    if args.compute_dtype != DUTTA_PROTOCOL["nf4_compute_dtype"]:
        out.append(f"NF4 compute dtype {args.compute_dtype}")
    if args.draft_precision != DUTTA_PROTOCOL["draft_precision"]:
        out.append(f"draft precision {args.draft_precision}")
    return out


def selftest(tok, draft, target, args, n: int = 3) -> bool:
    """Re-prove losslessness on the real models after moving machines.

    The unit tests check the algorithm on toy models; this checks the actual
    pair on the actual device. It is the gate to run after any port, because
    the things that break on a move -- a quantization path that changes
    numerics, a transformers version with a different cache API, a dtype
    silently downcast -- all surface as speculative output diverging from
    plain greedy on the *same* machine.

    Note the invariant being checked is per-machine. Greedy output is NOT
    expected to be token-identical across two different GPUs: fp16 matmuls
    reduce in a different order, so an occasional argmax flips. What must
    hold everywhere is spec == autoregressive locally.
    """
    from specdec import datasets as _ds

    samples = _ds.load(args.dataset, limit=n, seed=args.seed, offline=args.offline)
    print(f"self-test: speculative vs autoregressive on {len(samples)} prompts")
    ok = True
    near_ties = 0
    for s in samples:
        ids = encode(tok, s.prompt, args.device, not args.no_chat_template)
        ref = autoregressive_generate(
            target, ids, max_new_tokens=32, eos_token_id=tok.eos_token_id
        )
        for k in (1, 2, 4):
            got = speculative_generate(
                draft, target, ids, k=k, max_new_tokens=32,
                eos_token_id=tok.eos_token_id,
            )
            seq_a, seq_b = ref.sequences[0].tolist(), got.sequences[0].tolist()
            if seq_a == seq_b:
                print(f"  ok   sample {s.index} k={k}  "
                      f"accept {got.acceptance_rate:5.1%}")
                continue

            # Divergence alone does not mean a bug. Verification feeds the
            # target K+1 tokens in one matmul where decoding feeds one, and
            # the two shapes sum in a different order, so a near-tie can flip.
            # Measure the disagreement against the top-2 gap before judging.
            d = next(i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y)
            verdict = "UNEXPLAINED"
            detail = ""
            try:
                la = logits_for_row(target, ref.sequences[:, : d + 1], d - 1, 1)
                lb = logits_for_row(target, ref.sequences[:, : d + 1], d - 1, k + 1)
                top = torch.topk(la, 2).values
                gap, delta = float(top[0] - top[1]), float((la - lb).abs().max())
                detail = f"top-2 gap {gap:.5f} vs numerical spread {delta:.5f}"
                if gap <= delta:
                    verdict = "near-tie"
            except Exception as exc:  # too early in the sequence to probe
                detail = f"could not probe: {exc}"

            if verdict == "near-tie":
                near_ties += 1
                print(f"  tie  sample {s.index} k={k}: token "
                      f"{d - ref.prompt_len} flipped on a near-tie "
                      f"({detail}); accept {got.acceptance_rate:5.1%}")
            else:
                ok = False
                a = seq_a[ref.prompt_len:]
                b = seq_b[got.prompt_len:]
                i = d - ref.prompt_len
                print(f"  FAIL sample {s.index} k={k}: diverged at token {i}"
                      f"  [{detail}]")
                print(f"       greedy {a[max(0, i - 3):i + 3]}")
                print(f"       spec   {b[max(0, i - 3):i + 3]}")
    if ok and near_ties:
        print(f"\nself-test passed with {near_ties} near-tie flip(s).\n"
              "  Those are inherent, not defects: greedy speculative decoding is\n"
              "  lossless in exact arithmetic, and on a GPU the verify pass sums\n"
              "  in a different order than the decode pass. Where two candidates\n"
              "  are closer together than that difference, either may win.\n"
              "  Acceptance, timing and energy do not depend on which one did.\n")
    elif ok:
        print("self-test passed\n")
    else:
        print("\nself-test FAILED -- a divergence too large to be rounding.\n"
              "  Do not trust any measurement from this build; run\n"
              "  diagnose_divergence.py for the per-position detail.\n")
    return ok


def main() -> None:
    # profile first, so it can supply defaults that explicit flags override
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=sorted(PROFILES))
    preknown, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profile", choices=sorted(PROFILES),
                    help="machine preset: " + ", ".join(sorted(PROFILES)))
    ap.add_argument("--selftest", action="store_true",
                    help="check spec == greedy on the real models, then exit; "
                         "run this first on any new machine")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the .jsonl log beside --out")
    ap.add_argument("--force-resume", action="store_true",
                    help="resume even if the settings no longer match the log")
    ap.add_argument("--deadline", metavar="SPEC",
                    help="stop cleanly before this much time is up: '90m', "
                         "'1h45m', '5400s', or a clock time like '23:30'. "
                         "Essential inside a fixed GPU slot.")
    ap.add_argument("--reserve", type=float, default=3.0,
                    help="minutes held back from the deadline to write "
                         "results (default 3)")
    ap.add_argument("--order", default="prompt-major",
                    choices=["prompt-major", "strategy-major"],
                    help="prompt-major (default) finishes every strategy on "
                         "each prompt in turn, so an interrupted run still "
                         "covers all K; strategy-major completes one K at a time")
    ap.add_argument("--list", action="store_true",
                    help="print model pairs, what fits here, and the protocol; exit")
    ap.add_argument("--config", default=DEFAULT_PAIR, choices=sorted(MODEL_PAIRS),
                    help=f"model pair (default: {DEFAULT_PAIR}, the pair that fits "
                         "every device here in both fp16 and NF4)")
    ap.add_argument("--strategy", nargs="+", default=None,
                    help="e.g. coga-1 coga-2 ...; default is COGA-1..8, which is "
                         "what an energy-vs-K curve needs")
    ap.add_argument("--paper-strategies", action="store_true",
                    help="use exactly the strategies the paper ran for this family")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(ds.PAPER_SIZES),
                    help=f"one dataset (default: {DEFAULT_DATASET})")
    ap.add_argument("--limit", type=int, default=None,
                    help="prompts (default: the paper's count for this dataset)")
    ap.add_argument("--seed", type=int, default=ds.PAPER_SEED,
                    help="sampling seed; the paper's is not published")
    ap.add_argument("--offline", action="store_true",
                    help="bundled sample instead of the real datasets (debug only)")

    ap.add_argument("--quant", default="nf4",
                    choices=["nf4", "float16", "bfloat16", "float32"],
                    help="target precision; the paper used nf4")
    ap.add_argument("--compute-dtype", default="float16",
                    choices=["float16", "bfloat16"],
                    help="NF4 compute dtype; keep float16 unless every device in "
                         "the comparison has native bf16")
    ap.add_argument("--draft-precision", default="float16",
                    choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="the paper does not state its value")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample", action="store_true",
                    help="rejection sampling; the paper used greedy")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--energy", action="store_true", help="attach the NVML counter")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed repeats per (prompt, strategy); median reported")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="tiny random models, no downloads")
    ap.add_argument("--out", default=None)

    if preknown.profile:
        p = PROFILES[preknown.profile]
        ap.set_defaults(config=p.config, quant=p.quant, dataset=p.dataset,
                        limit=p.limit, max_new_tokens=p.max_new_tokens,
                        repeats=p.repeats, strategy=p.strategies)
    args = ap.parse_args()

    if args.profile:
        p = PROFILES[args.profile]
        # report the EFFECTIVE values, not the profile's: an explicit flag
        # overrides the profile, and a banner that disagrees with the run is
        # worse than no banner
        over = [f"{k}={v}" for k, v in (("quant", args.quant),
                                        ("config", args.config),
                                        ("dataset", args.dataset))
                if v != getattr(p, k)]
        print(f"profile {p.name}: {args.config} at {args.quant}, {args.dataset}"
              f"{'' if args.limit is None else f' x{args.limit} prompts'}"
              + (f"   [overridden: {', '.join(over)}]" if over else ""))
        print(f"  {p.note}\n")

    if args.list:
        vram = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3 - 0.8
            if torch.cuda.is_available()
            else 15.0
        )
        print(feasibility_table(vram))
        print(f"\nour sweep (COGA only -- the points of an energy-vs-K curve):"
              f"\n  {' '.join(DEFAULT_SWEEP)}")
        print("\nstrategies the paper ran, by family (reproduction only):")
        for fam, strat in PAPER_STRATEGIES.items():
            print(f"  {fam:<10}{' '.join(strat)}")
        print()
        print(protocol_summary())
        return

    pair = MODEL_PAIRS[args.config]
    if args.device == "cpu" and args.quant in ("float16", "nf4"):
        args.quant = "float32"
        args.draft_precision = "float32"

    strategies = args.strategy
    if args.paper_strategies:
        strategies = PAPER_STRATEGIES[pair.family]
    strategies = [parse_strategy(s) for s in (strategies or DEFAULT_SWEEP)]

    # ---- fit check before spending minutes on a download that will OOM
    if args.device == "cuda" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        need = pair.estimated_vram_gb(args.quant, args.draft_precision)
        print(f"{pair.label}: ~{need:.1f} GB estimated, {total:.1f} GB on device")
        if need > total:
            raise SystemExit(
                f"\n{pair.key} at {args.quant} will not fit.\n"
                f"{'' if pair.key != 'llama-70b' else 'The paper needed an A6000 48GB for this pair. '}"
                f"Try --quant nf4, or --config qwen3-4b.\n\n"
                + feasibility_table(total - 0.8)
            )

    if args.synthetic:
        tok, draft, target = load_synthetic(args.device, "float32")
    else:
        tok, draft, target = load_pair(pair, args)
    eos = tok.eos_token_id

    if args.selftest:
        sys.exit(0 if selftest(tok, draft, target, args) else 1)

    out = Path(args.out or f"results/{args.config}-{args.quant}-{args.dataset}.json")

    # Settings that must match for a resume to be meaningful. Anything here
    # changes which rows are comparable; anything omitted (device, deadline,
    # --energy) does not change the generated tokens.
    ckpt = Checkpoint(
        out=out,
        settings={
            "config": args.config, "quant": args.quant,
            "draft_precision": args.draft_precision,
            "compute_dtype": args.compute_dtype,
            "dataset": args.dataset, "limit": args.limit, "seed": args.seed,
            "max_new_tokens": args.max_new_tokens, "sample": args.sample,
            "temperature": args.temperature, "top_p": args.top_p,
            "chat_template": not args.no_chat_template,
            "strategies": [s.name for s in strategies],
            "baseline": not args.no_baseline,
            "repeats": args.repeats,
        },
        resume=args.resume,
        force=args.force_resume,
        deadline=parse_deadline(args.deadline) if args.deadline else None,
        reserve_s=args.reserve * 60,
    )
    ckpt.open()

    samples = ds.load(args.dataset, limit=args.limit, seed=args.seed,
                      offline=args.offline)
    if any(s.synthetic for s in samples):
        print("\n!! offline fallback data -- debug only, not reportable\n")

    deviations = protocol_deviations(args)
    if deviations:
        print("deviations from the Dutta et al. protocol (recorded in the output):")
        for d in deviations:
            print(f"  - {d}")
        print()
    print(f"{args.dataset}: {len(samples)} prompts x {len(strategies)} strategies "
          f"on {args.device}\n")

    probe = None
    idle_w = None
    if args.energy:
        probe = EnergyProbe()
        print(f"NVML energy counter attached to physical GPU "
              f"{probe.nvml_index} (whole-phase, coarse)"
              if probe.supported else
              f"energy requested but unavailable: {probe.reason}")
        probe = probe if probe.supported else None
        if probe:
            idle_w = probe.measure_idle_w()
            if idle_w:
                print(f"idle draw {idle_w:.1f} W -- recorded so the analysis "
                      f"can report energy above idle, not just energy during "
                      f"the phase")
        print()

    gen_kw = dict(max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                  temperature=args.temperature, top_p=args.top_p, eos_token_id=eos)

    if args.warmup and samples:
        warm = encode(tok, samples[0].prompt, args.device, not args.no_chat_template)
        for _ in range(args.warmup):
            speculative_generate(draft, target, warm, k=4,
                                 **{**gen_kw, "max_new_tokens": 8})
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # ---- build the work list ------------------------------------------
    units: list[Unit] = []
    for s in samples:
        if not args.no_baseline:
            units.append(Unit("autoregressive", s.dataset, s.index, s, None))
        for strat in strategies:
            units.append(Unit(strat.name, s.dataset, s.index, s, strat))

    # Prompt-major by default: run every strategy on prompt 1, then prompt 2.
    # If the slot ends early you are left with every K over fewer prompts,
    # which still gives a complete (noisier) energy-vs-K curve. Strategy-major
    # would instead leave a precise curve over only the K values that
    # finished, with nothing at all for the rest -- far less useful.
    if args.order == "strategy-major":
        rank = {u: i for i, u in enumerate(
            ["autoregressive"] + [s.name for s in strategies])}
        units.sort(key=lambda u: (rank[u.strategy], u.sample_index))

    pending = [u for u in units if u.key not in ckpt.done]
    print(f"{len(units)} units total, {len(pending)} to run "
          f"({args.order} order)")
    if ckpt.deadline:
        print(f"deadline in {(ckpt.deadline - time.time()) / 60:.0f} min, "
              f"holding back {args.reserve:.0f} min to write results")
    print()

    # ---- the loop ------------------------------------------------------
    completed_now = 0
    for i, unit in enumerate(pending):
        reason = ckpt.should_stop()
        if reason:
            print(f"\nstopping early -- {reason}")
            break

        ids = encode(tok, unit.sample.prompt, args.device,
                     not args.no_chat_template)
        t0 = time.perf_counter()
        runs = []
        for _ in range(args.repeats):
            meter = PhaseMeter(energy=probe)
            if unit.runner is None:
                runs.append(autoregressive_generate(target, ids, meter=meter,
                                                    **gen_kw))
            else:
                runs.append(speculative_generate(draft, target, ids,
                                                 k=unit.runner, meter=meter,
                                                 **gen_kw))
        best = sorted(runs, key=lambda r: r.wall_ms)[len(runs) // 2]
        elapsed = time.perf_counter() - t0

        row = {
            "strategy": unit.strategy, "dataset": unit.dataset,
            "sample_index": unit.sample_index,
            "new_tokens": best.new_tokens, "wall_ms": best.wall_ms,
            "tokens_per_s": best.tokens_per_second,
            "acceptance_rate": (best.acceptance_rate
                                if best.rounds and best.proposed_total else None),
            "tokens_per_round": best.mean_accepted_per_round,
            "mean_draft_length": best.mean_draft_length,
            "rounds": len(best.rounds), "proposed": best.proposed_total,
            "accepted": best.accepted_total, "meter": best.meter,
            "round_detail": [asdict(r) for r in best.rounds],
        }
        ckpt.record(row, elapsed)
        completed_now += 1

        every = max(1, len(pending) // 40)
        if completed_now % every == 0 or i == len(pending) - 1:
            print(ckpt.progress(len(ckpt.done), len(units),
                                len(pending) - i - 1), flush=True)

    still_pending = [u for u in units if u.key not in ckpt.done]

    peak = (torch.cuda.max_memory_allocated() / 1024**3
            if args.device == "cuda" and torch.cuda.is_available() else None)
    if peak:
        print(f"peak VRAM actually used: {peak:.2f} GB "
              f"(estimate was {pair.estimated_vram_gb(args.quant, args.draft_precision):.1f} GB)\n")

    # the log is the source of truth: it carries resumed rows too
    all_rows = ckpt.all_rows()

    with out.open("w") as fh:
        json.dump({
            "config": vars(args),
            "model_pair": {
                "key": pair.key, "paper_name": pair.paper_name,
                "target": pair.target_id, "draft": pair.draft_id,
            },
            "reference": "Dutta et al., Findings of EACL 2026, arXiv:2602.09113",
            "protocol": DUTTA_PROTOCOL,
            "protocol_deviations": deviations,
            "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "peak_vram_gb": peak,
            "idle_watts": idle_w,
            "dataset": args.dataset,
            "dataset_seed": args.seed,
            "offline_data": any(s.synthetic for s in samples),
            "profile": args.profile,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "complete": not still_pending,
            "units_done": len(ckpt.done),
            "units_total": len(units),
            "stopped_because": ckpt.stop_reason,
            "rows": all_rows,
        }, fh, indent=2)
    print(f"\nwrote {out}  ({len(all_rows)} rows, "
          f"{len(ckpt.done)}/{len(units)} units)")

    strategy_table(all_rows)
    summarise(all_rows)
    ckpt.report_remaining(still_pending, sys.argv)


def strategy_table(rows: list[dict]) -> None:
    """Per-strategy means. Printed at the end, since prompt-major ordering
    means no strategy is finished until almost all of them are."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r)
    if not by:
        return
    order = sorted(by, key=lambda s: (s != "autoregressive", s))
    print(f"\n  {'strategy':<14}{'n':>5}{'accept':>9}{'tok/round':>11}"
          f"{'tok/s':>9}{'draft%':>8}")
    print("  " + "-" * 56)
    for name in order:
        g = by[name]
        tps = statistics.mean(r["tokens_per_s"] for r in g)
        if name == "autoregressive":
            print(f"  {name:<14}{len(g):>5}{'-':>9}{1.0:>11.2f}{tps:>9.1f}{'-':>8}")
            continue
        acc = statistics.mean(r["acceptance_rate"] or 0 for r in g)
        tpr = statistics.mean(r["tokens_per_round"] for r in g)
        d = statistics.mean(r["meter"]["time_ms"].get("draft", 0) for r in g)
        v = statistics.mean(r["meter"]["time_ms"].get("verify", 0) for r in g)
        share = d / (d + v) * 100 if (d + v) else 0
        print(f"  {name:<14}{len(g):>5}{acc:>8.1%}{tpr:>11.2f}"
              f"{tps:>9.1f}{share:>7.1f}%")


def summarise(rows: list[dict]) -> None:
    base = {}
    for r in rows:
        if r["strategy"] == "autoregressive":
            base.setdefault(r["dataset"], []).append(r["tokens_per_s"])
    if not base:
        return
    print("\nspeedup over autoregressive (x)")
    dsets = sorted({r["dataset"] for r in rows})
    strats = [s for s in dict.fromkeys(r["strategy"] for r in rows)
              if s != "autoregressive"]
    print("  " + "strategy".ljust(12) + "".join(d.ljust(12) for d in dsets))
    for s in strats:
        cells = []
        for d in dsets:
            vals = [r["tokens_per_s"] for r in rows
                    if r["strategy"] == s and r["dataset"] == d]
            ref = base.get(d)
            cells.append(f"{statistics.mean(vals) / statistics.mean(ref):.2f}x".ljust(12)
                         if vals and ref else "-".ljust(12))
        print("  " + s.ljust(12) + "".join(cells))
    print("\nDutta et al. found latency speedup does not imply proportional energy\n"
          "savings. These are the latency numbers; the energy curve comes next,\n"
          "and the gap between their minima is the result we are after.")


if __name__ == "__main__":
    main()
