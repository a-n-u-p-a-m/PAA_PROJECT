#!/usr/bin/env python3
"""Will this sweep fit in the slot? Plan before you burn GPU time.

    # before you have the machine -- rough, from a bandwidth model
    python plan_run.py --vram 48 --bandwidth 1344 --slot-minutes 120

    # on the machine -- 3 minutes of measurement makes the rest exact
    python plan_run.py --calibrate --config qwen2.5-14b --quant float16

A sweep's cost is driven by three things, and the one people forget is the
third:

1. **Decode throughput.** At batch 1 this is memory-bandwidth bound, so time
   per forward pass is roughly ``weight_bytes / bandwidth``. A bigger target
   costs linearly more per verify step.

2. **Expected tokens per round.** With per-token acceptance probability a,
   a round of K drafts emits ``(1 - a^(K+1)) / (1 - a)`` tokens on average
   (Leviathan et al.). Throughput is that divided by
   ``K * t_draft + t_verify``, which is why large K stops paying.

3. **Getting the weights there at all.** NF4 does not shrink the download:
   bitsandbytes quantizes on load, so a 32B target still pulls ~65 GB of
   fp16 safetensors first. On a shared campus link that alone can exceed a
   two-hour slot. Pre-cache before the slot starts.

The pre-calibration numbers are a planning aid, not a measurement -- treat
them as +/-50% until ``--calibrate`` has run on the actual hardware.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specdec.memory import ARCHS, BYTES, GIB, budget, usable_vram  # noqa: E402

# Candidate pairs, largest target last.
CANDIDATES: list[tuple[str, str, str]] = [
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3-4b", "Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"),
    ("qwen3-8b", "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"),
    ("qwen3-14b", "Qwen/Qwen3-14B", "Qwen/Qwen3-0.6B"),
    ("qwen3-32b", "Qwen/Qwen3-32B", "Qwen/Qwen3-0.6B"),
    ("vicuna-7b", "lmsys/vicuna-7b-v1.3", "double7/vicuna-68m"),
    ("vicuna-13b", "lmsys/vicuna-13b-v1.3", "double7/vicuna-68m"),
    ("llama-8b", "meta-llama/Llama-3.1-8B-Instruct",
     "meta-llama/Llama-3.2-1B-Instruct"),
]

@dataclass(frozen=True)
class Device:
    """A machine we can actually get time on, with its real limits."""

    key: str
    name: str
    vram_gb: float          # usable, after whatever the platform reserves
    bandwidth_gbs: float
    cc: tuple[int, int]
    tdp_w: int
    drives_display: bool
    session_minutes: float | None   # None = no hard cap
    total_minutes: float | None     # budget per week, or in total
    notes: str

    @property
    def has_energy_counter(self) -> bool:
        """nvmlDeviceGetTotalEnergyConsumption needs Volta (CC 7.0) or newer."""
        return self.cc[0] >= 7

    @property
    def usable_session(self) -> float:
        """Session length minus load, warm-up and writing results."""
        cap = self.session_minutes or 600.0
        return max(cap - 20, 10)


DEVICES: dict[str, Device] = {
    "local-4050": Device(
        key="local-4050", name="RTX 4050 Laptop (local)",
        vram_gb=6.0, bandwidth_gbs=192, cc=(8, 9), tdp_w=115,
        drives_display=True, session_minutes=None, total_minutes=None,
        notes="Always available, but drives a display and throttles on a "
              "35-115 W laptop power budget. Has an energy counter (CC 8.9) "
              "yet will not hold a steady clock -- development, not data.",
    ),
    "colab-t4": Device(
        key="colab-t4", name="Colab free (T4)",
        vram_gb=14.7, bandwidth_gbs=320, cc=(7, 5), tdp_w=70,
        drives_display=False, session_minutes=720, total_minutes=None,
        notes="12 h ceiling, unpublished idle timeout, GPU type not "
              "guaranteed. 70 W cap means sustained load clock-throttles.",
    ),
    "kaggle-t4": Device(
        key="kaggle-t4", name="Kaggle (T4)",
        vram_gb=14.7, bandwidth_gbs=320, cc=(7, 5), tdp_w=70,
        drives_display=False, session_minutes=720, total_minutes=1800,
        notes="12 h sessions and ~30 h/week: by far the largest compute "
              "budget available here. Energy counter works.",
    ),
    "kaggle-p100": Device(
        key="kaggle-p100", name="Kaggle (P100)",
        vram_gb=15.9, bandwidth_gbs=732, cc=(6, 0), tdp_w=250,
        drives_display=False, session_minutes=720, total_minutes=1800,
        notes="Faster than the T4 and fits more, but CC 6.0 means NO energy "
              "counter at all. Fine for acceptance and latency, useless for "
              "the measurement this project exists to make.",
    ),
    "rtxpro-48": Device(
        key="rtxpro-48", name="RTX PRO 5000 Blackwell 48GB",
        vram_gb=47.0, bandwidth_gbs=1344, cc=(12, 0), tdp_w=300,
        drives_display=False, session_minutes=120, total_minutes=360,
        notes="Dedicated, 300 W, clock-lockable if you have admin. The only "
              "credible measurement device here -- and only 6 h of it. "
              "Needs torch >= 2.7 / cu128 for sm_120.",
    ),
}


#: Fraction of peak bandwidth HF eager achieves at batch 1. Calibrated against
#: a real Kaggle T4 sweep running with --energy, which adds a CUDA sync and
#: two NVML reads at every phase boundary; without --energy expect a little
#: better. The Python loop, kernel launches and that instrumentation together
#: cost more than the 0.40 originally assumed.
DEFAULT_EFFICIENCY = 0.30

#: Per-layer floor from kernel launches and Python dispatch. Small drafts are
#: bound by this, not by bandwidth -- which is exactly why a tiny draft model
#: does not make drafting free.
LAUNCH_MS_PER_LAYER = 0.09

#: NF4 pays a dequantization kernel on every matmul. Measured against a live
#: Kaggle T4 sweep: NF4 came out 1.29x SLOWER in wall time than fp16, not
#: faster, so the dequant cost more than cancels the 3x saving in bytes read.
#: bitsandbytes' NF4 path is compute-bound and Turing has little compute to
#: spare. Expect this to be gentler on Ampere and later -- re-measure with
#: --calibrate rather than trusting this constant on a different architecture.
NF4_SLOWDOWN = 3.6


def forward_ms(model_id: str, precision: str, bandwidth_gbs: float,
               efficiency: float = DEFAULT_EFFICIENCY) -> float:
    """Time for one batch-1 forward pass, in milliseconds."""
    a = ARCHS[model_id]
    weight_gib = a.weights_gib(precision)
    bandwidth_term = weight_gib * GIB / (bandwidth_gbs * 1e9 * efficiency) * 1000
    launch_term = a.layers * LAUNCH_MS_PER_LAYER
    t = max(bandwidth_term, launch_term)
    return t * (NF4_SLOWDOWN if precision == "nf4" else 1.0)


def tokens_per_round(k: int, acceptance: float) -> float:
    """Expected tokens emitted by one draft-verify round.

    Sum of a^i for i in 0..K: the bonus token always lands, each draft token
    survives only if every token before it did.
    """
    a = acceptance
    if a >= 1.0:
        return k + 1.0
    return (1.0 - a ** (k + 1)) / (1.0 - a)


@dataclass
class Timing:
    t_draft_ms: float
    t_verify_ms: float
    calibrated: bool = False

    def tok_per_s(self, k: int, acceptance: float) -> float:
        per_round_ms = k * self.t_draft_ms + self.t_verify_ms
        return tokens_per_round(k, acceptance) / per_round_ms * 1000

    def baseline_tok_per_s(self) -> float:
        return 1000.0 / self.t_verify_ms


def estimate_timing(target_id: str, draft_id: str, precision: str,
                    bandwidth_gbs: float,
                    efficiency: float = DEFAULT_EFFICIENCY) -> Timing:
    return Timing(
        t_draft_ms=forward_ms(draft_id, "float16", bandwidth_gbs, efficiency),
        t_verify_ms=forward_ms(target_id, precision, bandwidth_gbs, efficiency),
    )


def sweep_minutes(timing: Timing, ks: list[int], n_prompts: int,
                  max_new_tokens: int, acceptance: float,
                  include_baseline: bool = True, repeats: int = 1) -> float:
    total_s = 0.0
    if include_baseline:
        total_s += n_prompts * max_new_tokens / timing.baseline_tok_per_s()
    for k in ks:
        total_s += n_prompts * max_new_tokens / timing.tok_per_s(k, acceptance)
    return total_s * repeats / 60.0


def download_gib(target_id: str, draft_id: str) -> float:
    """Bytes pulled from the hub. Always fp16: NF4 quantizes after download."""
    return (ARCHS[target_id].weights_gib("float16")
            + ARCHS[draft_id].weights_gib("float16"))


def max_prompts(timing: Timing, ks: list[int], minutes: float,
                max_new_tokens: int, acceptance: float,
                repeats: int = 1) -> int:
    per_prompt = sweep_minutes(timing, ks, 1, max_new_tokens, acceptance,
                               repeats=repeats)
    return int(minutes / per_prompt) if per_prompt > 0 else 0


# ---------------------------------------------------------------- calibrate --


def calibrate(target_id: str, draft_id: str, precision: str,
              device: str = "cuda", warmup: int = 5, iters: int = 30) -> Timing:
    """Measure t_draft and t_verify on the real hardware.

    Three minutes here removes all the guesswork from the rest of the plan,
    which is worth it when slots are two hours and non-refundable.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from run_baseline import build_quant_config

    def load(model_id: str, prec: str):
        kw: dict = {}
        if prec == "nf4":
            kw["quantization_config"] = build_quant_config("nf4", "float16")
        else:
            kw["dtype"] = torch.float16
        m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        return (m if prec == "nf4" else m.to(device)).eval()

    def time_forward(model) -> float:
        ids = torch.randint(0, 1000, (1, 1), device=device)
        past = None
        prompt = torch.randint(0, 1000, (1, 256), device=device)
        with torch.no_grad():
            out = model(input_ids=prompt, use_cache=True)
            past = out.past_key_values
            for _ in range(warmup):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    print(f"loading draft {draft_id}")
    d = load(draft_id, "float16")
    t_draft = time_forward(d)
    print(f"  draft forward  {t_draft:.2f} ms  ({1000 / t_draft:.0f} tok/s solo)")
    del d
    torch.cuda.empty_cache()

    print(f"loading target {target_id} ({precision})")
    t = load(target_id, precision)
    t_verify = time_forward(t)
    print(f"  target forward {t_verify:.2f} ms  ({1000 / t_verify:.0f} tok/s solo)")
    del t
    torch.cuda.empty_cache()

    print(f"  draft is {t_verify / t_draft:.1f}x cheaper than the target\n")
    return Timing(t_draft, t_verify, calibrated=True)


# ------------------------------------------------------------------ report --


def device_report(dev: Device, ks: list[int], n_prompts: int,
                  max_new_tokens: int, acceptance: float, efficiency: float,
                  timing: Timing | None = None) -> None:
    """One device: what fits, how fast, and how it maps onto its session cap."""
    usable = usable_vram(dev.vram_gb, dev.drives_display)
    cc = f"{dev.cc[0]}.{dev.cc[1]}"
    limit = ("no session cap" if dev.session_minutes is None
             else f"{dev.session_minutes:.0f} min/session")
    budget_s = ("" if dev.total_minutes is None
                else f", {dev.total_minutes / 60:.0f} h budget")
    print(f"\n{dev.name}")
    print(f"  {dev.vram_gb:.1f} GB"
          + (f" (-{usable_vram(0, True) * -1:.1f} display = {usable:.1f} usable)"
             if dev.drives_display else "")
          + f", {dev.bandwidth_gbs:.0f} GB/s, CC {cc}, {dev.tdp_w} W, "
          + limit + budget_s)
    print(f"  energy counter: "
          + ("yes" if dev.has_energy_counter else "NO -- CC < 7.0"))
    print(f"  {dev.notes}")
    print()

    head = (f"    {'pair':<14}"
            f"{'fp16':>7}{'tok/s':>7}{'sweep':>8}{'/sess':>7}   "
            f"{'NF4':>6}{'tok/s':>7}{'sweep':>8}{'/sess':>7}")
    print(head)
    print("    " + "-" * (len(head) - 4))

    for key, tgt, drf in CANDIDATES:
        cells = []
        for prec in ("float16", "nf4"):
            used = budget(tgt, drf, prec, "float16", 256, max_new_tokens).total
            if used > usable:
                cells.append(f"{used:>6.1f}G{'--':>7}{'--':>8}{'--':>7}")
                continue
            tm = timing or estimate_timing(tgt, drf, prec, dev.bandwidth_gbs,
                                           efficiency)
            tps = tm.tok_per_s(ks[len(ks) // 2], acceptance)
            mins = sweep_minutes(tm, ks, n_prompts, max_new_tokens, acceptance)
            sessions = mins / dev.usable_session
            sess = f"{sessions:.1f}x" if sessions > 1 else "fits"
            cells.append(f"{used:>6.1f}G{tps:>7.0f}{_hm(mins):>8}{sess:>7}")
        print(f"    {key:<14}{cells[0]}   {cells[1]}")


def _hm(minutes: float) -> str:
    return f"{minutes:.0f}m" if minutes < 90 else f"{minutes / 60:.1f}h"


def plan(vram: float, bandwidth: float, slot_minutes: float, slots: int,
         ks: list[int], max_new_tokens: int, acceptance: float,
         efficiency: float, timing: Timing | None = None,
         only: str | None = None) -> None:
    usable = usable_vram(vram)
    print(f"{vram:.0f} GB, {bandwidth:.0f} GB/s, "
          f"{slots} x {slot_minutes:.0f} min slots "
          f"({slots * slot_minutes / 60:.0f} h total)")
    print(f"sweep = baseline + COGA-{min(ks)}..{max(ks)} "
          f"({len(ks)} values), {max_new_tokens} new tokens, "
          f"assumed acceptance {acceptance:.0%}")
    if timing and timing.calibrated:
        print("timings CALIBRATED on this machine")
    else:
        print("timings ESTIMATED from a bandwidth model -- treat as +/-50%")
    print()

    head = (f"  {'pair':<14}{'fp16':>7}{'NF4':>7}  {'prec':<8}"
            f"{'tok/s':>7}{'256 pr.':>9}{'64 pr.':>8}{'dl':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    for key, tgt, drf in CANDIDATES:
        if only and key != only:
            continue
        fp16 = budget(tgt, drf, "float16", "float16", 256, max_new_tokens).total
        nf4 = budget(tgt, drf, "nf4", "float16", 256, max_new_tokens).total
        if fp16 <= usable:
            prec, used = "float16", fp16
        elif nf4 <= usable:
            prec, used = "nf4", nf4
        else:
            print(f"  {key:<14}{fp16:>6.1f}G{nf4:>6.1f}G  "
                  f"{'--':<8}{'--':>7}{'does not fit':>17}")
            continue

        tm = timing or estimate_timing(tgt, drf, prec, bandwidth, efficiency)
        mid_k = ks[len(ks) // 2]
        tps = tm.tok_per_s(mid_k, acceptance)
        m256 = sweep_minutes(tm, ks, 256, max_new_tokens, acceptance)
        m64 = sweep_minutes(tm, ks, 64, max_new_tokens, acceptance)
        dl = download_gib(tgt, drf)

        def mark(m: float) -> str:
            return f"{m:.0f}m" + ("" if m <= slot_minutes else "!")

        print(f"  {key:<14}{fp16:>6.1f}G{nf4:>6.1f}G  {prec:<8}{tps:>7.0f}"
              f"{mark(m256):>9}{mark(m64):>8}{dl:>6.0f}G")

    print("\n  '!' = longer than one slot. 'dl' = hub download, fp16 even for")
    print("  NF4, since bitsandbytes quantizes after download.")

    # what fits a single slot at full 256 prompts
    print(f"\n  largest prompt count fitting one {slot_minutes:.0f}-minute slot:")
    for key, tgt, drf in CANDIDATES:
        if only and key != only:
            continue
        fp16 = budget(tgt, drf, "float16", "float16", 256, max_new_tokens).total
        nf4 = budget(tgt, drf, "nf4", "float16", 256, max_new_tokens).total
        prec = "float16" if fp16 <= usable else ("nf4" if nf4 <= usable else None)
        if prec is None:
            continue
        tm = timing or estimate_timing(tgt, drf, prec, bandwidth, efficiency)
        # leave 15 min for load, warmup and writing results
        n = max_prompts(tm, ks, slot_minutes - 15, max_new_tokens, acceptance)
        print(f"    {key:<14}{min(n, 256):>4} prompts"
              + ("  (full 256)" if n >= 256 else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None,
                    metavar="KEY",
                    help="per-device report; 'all' for every device. "
                         + ", ".join(DEVICES))
    ap.add_argument("--prompts", type=int, default=256,
                    help="prompts per sweep when sizing (default: GSM8K's 256)")
    ap.add_argument("--vram", type=float, default=48.0)
    ap.add_argument("--bandwidth", type=float, default=1344.0,
                    help="GB/s; 1344 = RTX PRO 5000 Blackwell, 900 = V100 SXM2")
    ap.add_argument("--slot-minutes", type=float, default=120.0)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--acceptance", type=float, default=0.7,
                    help="assumed per-token acceptance; measure it and re-plan")
    ap.add_argument("--efficiency", type=float, default=DEFAULT_EFFICIENCY)
    ap.add_argument("--config", help="restrict the table to one pair")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure real forward times on this machine first")
    ap.add_argument("--quant", default="float16", choices=["float16", "nf4"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.devices:
        keys = list(DEVICES) if "all" in args.devices else args.devices
        unknown = [k for k in keys if k not in DEVICES]
        if unknown:
            sys.exit(f"unknown device(s): {', '.join(unknown)}")
        print(f"sweep = baseline + COGA-{min(args.k)}..{max(args.k)} "
              f"({len(args.k)} values) x {args.prompts} prompts x "
              f"{args.max_new_tokens} new tokens")
        print(f"assumed acceptance {args.acceptance:.0%}; "
              f"'/sess' = sessions of that device needed for one sweep")
        print("timings are modelled, not measured -- run --calibrate on each "
              "machine to replace them")
        for k in keys:
            device_report(DEVICES[k], args.k, args.prompts,
                          args.max_new_tokens, args.acceptance, args.efficiency)
        print()
        return

    timing = None
    if args.calibrate:
        if not args.config:
            sys.exit("--calibrate needs --config")
        match = [c for c in CANDIDATES if c[0] == args.config]
        if not match:
            sys.exit(f"unknown config {args.config}; "
                     f"have {', '.join(c[0] for c in CANDIDATES)}")
        _, tgt, drf = match[0]
        timing = calibrate(tgt, drf, args.quant, args.device)

    plan(args.vram, args.bandwidth, args.slot_minutes, args.slots, args.k,
         args.max_new_tokens, args.acceptance, args.efficiency, timing,
         only=args.config if not args.calibrate else args.config)


if __name__ == "__main__":
    main()
