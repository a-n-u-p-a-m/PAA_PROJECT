#!/bin/bash
# Everything on Kaggle in one go: tests, probe, CLEAR quick check, and --
# only if the quick check passes its own gate -- the full pass and analysis.
#
#   !bash /kaggle/input/<dataset>/specdec/kaggle_run.sh           # quick, then full
#   !bash /kaggle/input/<dataset>/specdec/kaggle_run.sh quick     # stop after quick
#   !QUANT=float16 bash .../kaggle_run.sh                         # fp16 target
#
# QUANT picks the target precision (nf4, the default, or float16). The gate
# always judges the file that run just wrote, whatever its precision. With
# QUANT=float16 only the target is measured if an NF4 pass is already in
# results/ (the draft is fp16 in both, so it is taken from there); otherwise
# both models are measured.
#
# Whatever happens, it ends by packaging results/ into
# /kaggle/working/clear-results-<time>.tar.gz, so there is always one file to
# bring back. Safe to re-run: the full pass resumes where it stopped.

MODE="${1:-full}"
QUANT="${QUANT:-nf4}"
REPEAT="${REPEAT:-2}"
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK=/kaggle/working/specdec
STAMP=$(date +%Y%m%d-%H%M)
export HF_HOME=/kaggle/working/hf
# cuBLAS chooses a kernel partly from the workspace it may use and caches the
# choice per process. Fixing it makes the choice the same from run to run, so
# a bad one is a stable fact the gate can catch instead of a lottery.
# run_clear.py sets this too; here so every step in this script shares it.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

case "$QUANT" in
    nf4)     SWEEP=results/3b-nf4.json;  ROLES="draft target" ;;
    float16) SWEEP=results/3b-fp16.json; ROLES="draft target" ;;
    *) echo "QUANT must be nf4 or float16, not '$QUANT'"; exit 1 ;;
esac
OUT=results/clear-qwen2.5-3b-$QUANT.json
CLEAR_FILES="$OUT"
NF4_PASS=results/clear-qwen2.5-3b-nf4.json
if [ "$QUANT" = float16 ] && [ -f "$NF4_PASS" ]; then
    ROLES="target"
    CLEAR_FILES="$NF4_PASS $OUT"
    echo "fp16 target only; draft measurements come from $NF4_PASS"
fi
# 75 W: nothing on a 70 W T4 can draw more, so a reading above it is wrong.
COMMON="--profile kaggle-t4 --quant $QUANT --roles $ROLES --gate --gate-max-w 75"

say "CLEAR quick check ($QUANT, roles: $ROLES, $REPEAT fresh processes)"
# --gate judges the file this run wrote and exits 3 if it is not good enough.
# --repeat measures the whole thing again in a fresh process: a kernel pick is
# made once per process and cached, so this is the only way to see one that is
# not reproducible. Cheap here, which is why it belongs on the quick check
# rather than the full pass. REPEAT=1 to skip, REPEAT=3 if the slot allows.
python run_clear.py $COMMON --quick --repeat "$REPEAT" 2>&1 \
    | tee "results/clear-quick-$QUANT.log"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 3 ] && { echo "quick gate failed -- bring the archive back for a look"; exit 1; }
[ "$RC" -eq 0 ] || { echo "quick run failed"; exit 1; }
[ "$MODE" = "quick" ] && { say "quick mode: stopping here"; exit 0; }

say "CLEAR full pass (~1.5-2 h; resumes if re-run)"
python run_clear.py $COMMON --deadline 600m --resume 2>&1 | tee "results/clear-full-$QUANT.log"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] || [ "$RC" -eq 3 ] || { echo "full pass failed"; exit 1; }

say "audit: does every replay cost what the component costs in the real pass?"
# run_clear.py already anchors each measurement as it takes it, so for a file
# written just now this is a second opinion in its own file -- and the only way
# to screen a file measured before the anchor existed.
python diagnose_replay.py --profile kaggle-t4 --quant "$QUANT" --audit "$OUT" \
    2>&1 | tee "results/clear-audit-$QUANT.txt"

say "composing E(K) against the existing T4 sweep"
if [ "$RC" -eq 3 ]; then
    echo "!! THE FULL PASS FAILED ITS GATE. The tables below are for diagnosis only;"
    echo "!! do not quote numbers from them."
fi
python analyze.py "$SWEEP" --clear $CLEAR_FILES \
    --csv "results/clear-curve-$QUANT.csv" 2>&1 | tee "results/clear-analysis-$QUANT.txt"
python analyze.py "$SWEEP" --clear $CLEAR_FILES --clear-metric time_ms 2>&1 \
    | sed -n '/^CLEAR composition/,$p' > "results/clear-time-analysis-$QUANT.txt"
echo "time composition written to results/clear-time-analysis-$QUANT.txt"
[ "$RC" -eq 0 ] || exit 1
