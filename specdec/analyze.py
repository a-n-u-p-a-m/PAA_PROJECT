#!/usr/bin/env python3
"""Turn a COGA sweep into the energy-vs-K curve.

    python analyze.py results/qwen2.5-3b-nf4-gsm8k.json
    python analyze.py results/*.json --csv curve.csv

Reports, per constant draft length K:

    accept      fraction of drafted tokens the target kept
    tok/round   block efficiency, in [1, K+1]
    tok/s       throughput
    speedup     against autoregressive decoding on the same target
    draft%      share of phase time spent in the draft model
    E/token     energy per generated token, once --energy data exists

The point of the exercise is the gap between two minima. Latency is optimised
by drafting aggressively: a larger K amortises the target's forward pass over
more tokens. Energy is not, because every rejected draft token is work that
produced nothing, and the draft model's cost grows linearly in K while the
acceptance rate falls. So E(K) is expected to turn upward well before
throughput does, and K* < K_latency. Locating that gap on real hardware is
the result; this script is where it becomes visible.

Without ``--energy`` in the sweep there is no energy column, and the script
reports the latency curve plus the per-token draft-work term that the energy
curve will be dominated by.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def load(paths: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    meta: dict = {}
    for p in paths:
        with open(p) as fh:
            blob = json.load(fh)
        meta = meta or {
            k: blob.get(k)
            for k in ("model_pair", "device", "dataset", "protocol_deviations",
                      "peak_vram_gb", "offline_data", "idle_watts")
        }
        rows.extend(blob["rows"])
    return rows, meta


def aggregate(rows: list[dict], idle_w: float | None = None) -> tuple[list[dict], dict | None]:
    """Collapse per-prompt rows into one record per strategy."""
    by_strategy: dict[str, list[dict]] = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r)

    baseline = None
    if "autoregressive" in by_strategy:
        g = by_strategy.pop("autoregressive")
        baseline = {
            "tokens_per_s": mean(r["tokens_per_s"] for r in g),
            "energy_mj": mean(_energy(r) for r in g),
            "energy_above_idle_mj": mean(_energy_above_idle(r, idle_w) for r in g),
            "new_tokens": mean(r["new_tokens"] for r in g),
        }

    out = []
    for name, g in by_strategy.items():
        is_coga = name.startswith("coga-")
        energy = mean(_energy(r) for r in g)
        dynamic = mean(_energy_above_idle(r, idle_w) for r in g)
        tokens = mean(r["new_tokens"] for r in g)
        out.append({
            "strategy": name,
            "k": int(name.split("-")[1]) if is_coga else None,
            "constant_k": is_coga,
            "mean_draft_length": mean(r["mean_draft_length"] for r in g),
            "acceptance_rate": mean(r["acceptance_rate"] for r in g),
            "tokens_per_round": mean(r["tokens_per_round"] for r in g),
            "tokens_per_s": mean(r["tokens_per_s"] for r in g),
            "draft_share": _draft_share(g),
            "draft_tokens_per_output_token": _draft_work(g),
            "energy_mj": energy,
            "energy_per_token_mj": (energy / tokens) if energy and tokens else None,
            "dynamic_mj": dynamic,
            "dynamic_per_token_mj": (dynamic / tokens) if dynamic and tokens else None,
            "n_prompts": len(g),
        })
    out.sort(key=lambda r: (r["k"] is None, r["k"] or 0, r["strategy"]))
    return out, baseline


def _energy(row: dict):
    e = (row.get("meter") or {}).get("energy_mj")
    return sum(e.values()) if e else None


def _energy_above_idle(row: dict, idle_w: float | None):
    """Phase energy minus what the card would have drawn sitting idle.

    Raw phase energy charges the work for the GPU merely being powered on
    for that long. Since draft phases lengthen as K grows, the static term
    grows with K too and would masquerade as rising draft cost. Subtracting
    idle x elapsed leaves the part attributable to the work itself.
    """
    e = _energy(row)
    if e is None or not idle_w:
        return None
    t_ms = sum((row.get("meter") or {}).get("time_ms", {}).values())
    return max(0.0, e - idle_w * t_ms)  # W x ms = mJ


def _draft_share(group: list[dict]):
    d = mean((r["meter"]["time_ms"].get("draft", 0)) for r in group)
    v = mean((r["meter"]["time_ms"].get("verify", 0)) for r in group)
    return d / (d + v) * 100 if d is not None and v is not None and (d + v) else None


def _draft_work(group: list[dict]):
    """Draft forward passes per emitted token.

    The clearest latency-free proxy for wasted draft energy: it rises with K
    whenever acceptance falls, and it is what the energy curve tracks.
    """
    proposed = sum(r["proposed"] for r in group)
    emitted = sum(r["new_tokens"] for r in group)
    return proposed / emitted if emitted else None


def fmt(value, spec: str, dash: str = "   -  ") -> str:
    return dash if value is None else format(value, spec)


def report(records: list[dict], baseline: dict | None, meta: dict,
           rows: list[dict] | None = None) -> None:
    pair = meta.get("model_pair") or {}
    print(f"{pair.get('target', '?')}  <-  {pair.get('draft', '?')}")
    print(f"{meta.get('dataset', '?')} on {meta.get('device', '?')}"
          + (f", peak {meta['peak_vram_gb']:.1f} GB" if meta.get("peak_vram_gb") else ""))
    if meta.get("offline_data"):
        print("!! offline fallback data -- not reportable")
    for d in meta.get("protocol_deviations") or []:
        print(f"deviation: {d}")
    print()

    have_energy = any(r["energy_per_token_mj"] is not None for r in records)
    _sanity_check_idle(records, baseline, meta.get("idle_watts"))
    have_dyn = any(r["dynamic_per_token_mj"] is not None for r in records)
    idle_w = meta.get("idle_watts")
    head = (f"  {'K':>3}  {'accept':>7}  {'tok/round':>9}  {'tok/s':>8}  "
            f"{'speedup':>8}  {'draft%':>7}  {'draft fwd/tok':>13}")
    if have_energy:
        head += f"  {'mJ/token':>9}  {'vs AR':>7}"
    if have_dyn:
        head += f"  {'dyn mJ/tok':>11}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    base_tps = baseline["tokens_per_s"] if baseline else None
    base_ept = None
    if baseline and baseline.get("energy_mj") and baseline.get("new_tokens"):
        base_ept = baseline["energy_mj"] / baseline["new_tokens"]

    if baseline:
        line = (f"  {'AR':>3}  {'-':>7}  {1.0:>9.2f}  {base_tps:>8.1f}  "
                f"{1.00:>8.2f}  {'-':>7}  {0.0:>13.2f}")
        if have_energy:
            line += f"  {fmt(base_ept, '>9.2f')}  {1.00:>7.2f}"
        if have_dyn:
            bd = (baseline.get("energy_above_idle_mj") or 0) / baseline["new_tokens"] \
                if baseline.get("energy_above_idle_mj") and baseline.get("new_tokens") else None
            line += f"  {fmt(bd, '>11.2f')}"
        print(line)

    for r in records:
        k = r["k"] if r["constant_k"] else None
        klabel = str(k) if k is not None else "~"
        speedup = r["tokens_per_s"] / base_tps if base_tps else None
        line = (f"  {klabel:>3}  {fmt(r['acceptance_rate'], '>7.1%')}  "
                f"{fmt(r['tokens_per_round'], '>9.2f')}  "
                f"{fmt(r['tokens_per_s'], '>8.1f')}  "
                f"{fmt(speedup, '>8.2f')}  {fmt(r['draft_share'], '>7.1f')}  "
                f"{fmt(r['draft_tokens_per_output_token'], '>13.2f')}")
        if have_energy:
            ept = r["energy_per_token_mj"]
            ratio = (ept / base_ept) if (ept and base_ept) else None
            line += f"  {fmt(ept, '>9.2f')}  {fmt(ratio, '>7.2f')}"
        if have_dyn:
            line += f"  {fmt(r['dynamic_per_token_mj'], '>11.2f')}"
        if not r["constant_k"]:
            line += f"   <- {r['strategy']}, mean K={r['mean_draft_length']:.1f}"
        print(line)

    # ---- the headline: where the two optima sit
    curve = [r for r in records if r["constant_k"]]
    if not curve:
        return
    print()
    _verdict(curve, rows or [], have_dyn, meta.get('idle_watts'))


def _paired(rows, a, b, key):
    """Mean per-prompt (a - b) with a 95% CI.

    Every K is run on the same prompts, so these are paired samples and
    prompt difficulty cancels. Comparing two independent means instead --
    or eyeballing whether error bars overlap -- is far less sensitive and
    will call a real difference a tie.
    """
    import statistics as _st

    A = {r["sample_index"]: r for r in rows if r["strategy"] == a}
    B = {r["sample_index"]: r for r in rows if r["strategy"] == b}
    ids = sorted(set(A) & set(B))
    diff = []
    for i in ids:
        x, y = key(A[i]), key(B[i])
        if x is None or y is None:
            continue
        diff.append(x - y)
    if len(diff) < 3:
        return None
    m = _st.mean(diff)
    half = 1.96 * _st.stdev(diff) / len(diff) ** 0.5
    return m, half, (m - half > 0 or m + half < 0)


def _verdict(curve, rows, have_dyn, idle_w=None) -> None:
    ekey = "dynamic_per_token_mj" if have_dyn else "energy_per_token_mj"
    fastest = max(curve, key=lambda r: r["tokens_per_s"] or 0)
    with_e = [r for r in curve if r[ekey]]
    if not with_e:
        print(f"  latency-optimal   K = {fastest['k']}  "
              f"({fastest['tokens_per_s']:.1f} tok/s)")
        return
    leanest = min(with_e, key=lambda r: r[ekey])

    def tps(r):
        return r["tokens_per_s"]

    def ene(r):
        # must measure the SAME quantity the verdict ranks on, or the paired
        # delta and the percentage can point in opposite directions -- they
        # genuinely do when static power and work energy rank K differently
        m = r.get("meter") or {}
        e = m.get("energy_mj")
        if not e or not r.get("new_tokens"):
            return None
        total = sum(e.values())
        if have_dyn and idle_w:
            total = max(0.0, total - idle_w * sum(m.get("time_ms", {}).values()))
        return total / r["new_tokens"]

    fast_set = {fastest["k"]}
    lean_set = {leanest["k"]}
    for r in curve:
        s = _paired(rows, f"coga-{fastest['k']}", f"coga-{r['k']}", tps)
        if s and not s[2]:
            fast_set.add(r["k"])
        s = _paired(rows, f"coga-{r['k']}", f"coga-{leanest['k']}", ene)
        if s and not s[2]:
            lean_set.add(r["k"])

    print(f"  latency-optimal   K = {fastest['k']}  "
          f"({fastest['tokens_per_s']:.1f} tok/s)"
          f"   statistically tied: K={sorted(fast_set)}")
    print(f"  energy-optimal    K = {leanest['k']}  ({leanest[ekey]:.0f} "
          f"mJ/token{' above idle' if have_dyn else ''})"
          f"   statistically tied: K={sorted(lean_set)}")

    ks = [r["k"] for r in curve]
    if leanest["k"] in (min(ks), max(ks)):
        print(f"\n  !! the energy minimum sits at K={leanest['k']}, the edge of the\n"
              f"     swept range. The true optimum may lie beyond it -- extend the\n"
              f"     sweep past K={max(ks)} before quoting this as K*.")

    both = sorted(fast_set & lean_set)
    if both:
        print(f"\n  No trade-off within noise: K={both} is both fastest and\n"
              f"  lowest-energy. Pick the smallest and stop.")
    else:
        s = _paired(rows, f"coga-{fastest['k']}", f"coga-{leanest['k']}", ene)
        extra = f" ({s[0]:+.0f} +/- {s[1]:.0f} mJ/token, paired)" if s else ""
        pen = (fastest[ekey] / leanest[ekey] - 1) * 100
        print(f"\n  A real trade-off: the fastest and lowest-energy K are\n"
              f"  statistically disjoint. Running at K={fastest['k']} for speed costs\n"
              f"  {pen:.1f}% more energy per token than K={leanest['k']}{extra}.")


def _sanity_check_idle(records, baseline, idle_w) -> None:
    """A GPU cannot draw more at rest than it draws working.

    If it appears to, the counter was not reading the device that did the
    work -- the classic cause being that NVML ignores CUDA_VISIBLE_DEVICES,
    so two processes pinned to different GPUs both read physical GPU 0.
    """
    if not idle_w or not baseline:
        return
    e, tps = baseline.get("energy_mj"), baseline.get("tokens_per_s")
    if not e or not tps or not baseline.get("new_tokens"):
        return
    total_w = e / baseline["new_tokens"] * tps / 1000
    if idle_w >= total_w:
        print(f"\n  !! idle draw {idle_w:.1f} W exceeds measured total power "
              f"{total_w:.1f} W.\n     That is impossible, so the energy "
              f"columns below are NOT trustworthy:\n     the counter was "
              f"almost certainly reading a different GPU. Check the\n     "
              f"'attached to physical GPU N' line in the run log.\n")


def write_csv(records: list[dict], path: str) -> None:
    fields = ["strategy", "k", "constant_k", "mean_draft_length", "acceptance_rate",
              "tokens_per_round", "tokens_per_s", "draft_share",
              "draft_tokens_per_output_token", "energy_mj",
              "energy_per_token_mj", "dynamic_mj", "dynamic_per_token_mj",
              "n_prompts"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in records)
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("results", nargs="+", help="JSON file(s) from run_baseline.py")
    ap.add_argument("--csv", help="also write the aggregated curve as CSV")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.results:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or [pattern])
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        sys.exit(f"no such file: {', '.join(missing)}")
    rows, meta = load(paths)
    if not rows:
        sys.exit("no rows found")
    records, baseline = aggregate(rows, meta.get('idle_watts'))
    report(records, baseline, meta, rows)
    if args.csv:
        write_csv(records, args.csv)


if __name__ == "__main__":
    main()
