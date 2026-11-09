#!/bin/bash
# Everything on Kaggle in one go: tests, probe, CLEAR quick check, and --
# only if the quick check passes its own gate -- the full pass and analysis.
#
#   !bash /kaggle/input/<dataset>/specdec/kaggle_run.sh           # quick, then full
#   !bash /kaggle/input/<dataset>/specdec/kaggle_run.sh quick     # stop after quick
#
# Whatever happens, it ends by packaging results/ into
# /kaggle/working/clear-results-<time>.tar.gz, so there is always one file to
# bring back. Safe to re-run: the full pass resumes where it stopped.

MODE="${1:-full}"
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK=/kaggle/working/specdec
STAMP=$(date +%Y%m%d-%H%M)
export HF_HOME=/kaggle/working/hf

say() { printf '\n======== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

package() {
    cd /kaggle/working || return
    local out="clear-results-$STAMP.tar.gz"
    tar czf "$out" --exclude='3b-nf4.json' --exclude='3b-fp16.json' -C specdec results
    say "packaged /kaggle/working/$out  ($(du -h "$out" | cut -f1)) -- download it from Output"
}
trap package EXIT

# ---- fresh copy of the code; keep earlier results so --resume still works
say "copying code from $SRC"
mkdir -p /kaggle/working/keep
[ -d "$WORK/results" ] && cp -n "$WORK"/results/clear-* /kaggle/working/keep/ 2>/dev/null
rm -rf "$WORK" && cp -r "$SRC" "$WORK" && cd "$WORK" || exit 1
cp -n /kaggle/working/keep/clear-* results/ 2>/dev/null
LOG=results/kaggle-run-$STAMP.log
exec > >(tee -a "$LOG") 2>&1

say "installing requirements"
pip install -q -r requirements.txt || exit 1

say "tests"
python tests/test_lossless.py > results/test-lossless.log 2>&1 \
    && echo "  test_lossless: passed" || { echo "  test_lossless FAILED"; tail -20 results/test-lossless.log; exit 1; }
python tests/test_clear.py > results/test-clear.log 2>&1 \
    && echo "  test_clear: passed" || { echo "  test_clear FAILED"; tail -20 results/test-clear.log; exit 1; }

say "environment probe"
python probe_env.py --json results/env.json --text results/env.txt | tail -12

say "CLEAR quick check"
python run_clear.py --profile kaggle-t4 --quick 2>&1 | tee results/clear-quick.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "quick run failed"; exit 1; }

say "gate: is the measurement good enough for the full pass?"
python - <<'EOF'
import json, statistics, sys
sys.path.insert(0, ".")
from specdec.clear import Costs, additivity

b = json.load(open("results/clear-qwen2.5-3b-nf4-quick.json"))
ms = b["measurements"]
problems = []
if b["replay_check"]["mismatched"]:
    problems.append(f"{b['replay_check']['mismatched']} replays did not reproduce")
bad = [m["label"] for m in ms if not m["ok"]]
if len(bad) > len(ms) * 0.1:
    problems.append(f"{len(bad)}/{len(ms)} measurements outside the 10% bar")
over = statistics.median(m.get("meter_overhead_pct", 0.0) for m in ms)
if abs(over) > 5:
    problems.append(f"metering adds {over:+.1f}% to call time")
hot = [m["label"] for m in ms if (m.get("power_w") or 0) > 75]
if hot:
    problems.append(f"impossible power (>75 W on a 70 W card): {hot[:3]}")
fwd = [c["ratio"] for c in additivity(Costs(b)) if c["check"].endswith("forward")]
if not all(0.8 <= r <= 1.25 for r in fwd):
    problems.append("forward additivity " + ", ".join(f"{r:.2f}" for r in fwd))

print(f"  replays exact:          {b['replay_check']['total'] - b['replay_check']['mismatched']}/{b['replay_check']['total']}")
print(f"  inside 10% bar:         {len(ms) - len(bad)}/{len(ms)}")
print(f"  metering overhead:      {over:+.1f}% median")
print(f"  forward additivity:     {min(fwd):.2f} .. {max(fwd):.2f}")
if problems:
    print("\n  GATE FAILED -- not starting the full pass:")
    for p in problems:
        print(f"    - {p}")
    sys.exit(1)
print("\n  gate passed")
EOF
GATE=$?
[ "$GATE" -eq 0 ] || { echo "bring the archive back for a look"; exit 1; }
[ "$MODE" = "quick" ] && { say "quick mode: stopping here"; exit 0; }

say "CLEAR full pass (~1.5-2 h; resumes if re-run)"
python run_clear.py --profile kaggle-t4 --deadline 600m --resume 2>&1 | tee results/clear-full.log

say "composing E(K) against the existing T4 sweep"
python analyze.py results/3b-nf4.json --clear results/clear-qwen2.5-3b-nf4.json \
    --csv results/clear-curve.csv 2>&1 | tee results/clear-analysis.txt
