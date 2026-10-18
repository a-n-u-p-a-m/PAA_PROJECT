#!/bin/bash
# Unattended slot runner: finish the 7B sweep, fetch and run 3B, trace both,
# analyse everything, package it.
#
#   ./run_slot.sh 120        # 120 = minutes available in this slot
#
# Safe to re-run: every sweep uses --resume, the download resumes per shard,
# and the engine patch is idempotent. Nothing here re-does finished work.
#
# The 3B weights download in the background over curl while the GPU works on
# the 7B remainder, because one is network-bound and the other is not.
# curl ignores HF_HUB_OFFLINE, so the 7B cache (a hand-built snapshot under a
# fake revision) stays safe from being re-downloaded.

SLOT_MIN="${1:-120}"
TAIL_MIN=8                      # held back for traces, analysis, packaging
SEVEN_CAP=30                    # the 7B remainder needs ~8 min; never let it
                                # eat the slot if something goes wrong

WORK=/data/cs3406/team09
PROJ=$WORK/PAA_PROJECT/specdec
REV=0000000000000000000000000000000000000001

export HF_HOME=$WORK/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

END=$(( $(date +%s) + SLOT_MIN * 60 ))
STAMP=$(date +%Y%m%d-%H%M)
LOG=$WORK/slot-$STAMP.log
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n======== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
left() { echo $(( (END - TAIL_MIN * 60 - $(date +%s)) / 60 )); }
cap()  { local l; l=$(left); [ "$l" -gt "$1" ] && echo "$1" || echo "$l"; }

source "$WORK/.venv/bin/activate" || { echo "no venv"; exit 1; }
cd "$PROJ" || { echo "no project dir"; exit 1; }
mkdir -p results

say "slot = $SLOT_MIN min, hard stop $(date -d "@$END" +%H:%M), log $LOG"
python -c "import torch;print(torch.__version__, torch.cuda.get_arch_list()[-1], torch.cuda.get_device_name(0))" || exit 1

link_model() {                  # $1 = Qwen2.5-3B-Instruct
    local NAME=$1
    local SRC=$WORK/models/$NAME CACHE=$WORK/hf/hub/models--Qwen--$NAME
    rm -rf "$CACHE"
    mkdir -p "$CACHE/snapshots/$REV" "$CACHE/refs"
    for f in "$SRC"/*; do ln -s "$f" "$CACHE/snapshots/$REV/$(basename "$f")"; done
    printf '%s' "$REV" > "$CACHE/refs/main"
    echo "linked $NAME: $(ls "$CACHE/snapshots/$REV" | wc -l) files"
}

trace_pair() {                  # $1 = config key, $2 = short tag
    local CFG=$1 TAG=$2 K
    for K in 4 8; do
        if [ "$(left)" -lt 2 ]; then echo "  no time for $TAG K=$K"; return; fi
        python trace_spec.py --config "$CFG" --quant float16 --k "$K" \
            --max-new-tokens 64 --no-compare \
            --html "results/trace-$TAG-k$K.html" \
            --json "results/trace-$TAG-k$K.json"
    done
}

done_frac() {                   # $1 = results json -> "1278/1440 (incomplete)"
    python - "$1" <<'EOS' 2>/dev/null || echo "not run"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d.get('units_done')}/{d.get('units_total')} "
      f"({'COMPLETE' if d.get('complete') else 'incomplete'})")
EOS
}

# ---------------------------------------------------------------- A. 3B pull
say "A. starting 3B download in the background"
cat > "$WORK/get3b.sh" <<'EOS'
#!/bin/bash
BASE=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main
DST=/data/cs3406/team09/models/Qwen2.5-3B-Instruct
PY=/data/cs3406/team09/.venv/bin/python
mkdir -p "$DST"; cd "$DST" || exit 1
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  curl -fsSL -C - -o "$f" "$BASE/$f" || rm -f "$f"
done
if curl -fsSL -o model.safetensors.index.json "$BASE/model.safetensors.index.json"; then
  FILES=$("$PY" -c "import json;print(' '.join(sorted(set(json.load(open('model.safetensors.index.json'))['weight_map'].values()))))")
else
  rm -f model.safetensors.index.json; FILES=model.safetensors
fi
echo "shards: $FILES"
for f in $FILES; do
  ( until curl -fsSL -C - --retry 100 --retry-all-errors --retry-delay 5 -o "$f" "$BASE/$f"
    do sleep 5; done; echo "done $f" ) &
done
wait
echo "3B ALL DONE"
EOS
chmod +x "$WORK/get3b.sh"
setsid nohup "$WORK/get3b.sh" > "$WORK/get3b.log" 2>&1 < /dev/null &

# ------------------------------------------------------------ B. engine patch
say "B. token capture patch"
python patch_trace.py

# ------------------------------------------------------- C. finish 7B sweep
say "C. 7B sweep, resuming (allowing $(cap $SEVEN_CAP) min of $(left) left)"
if [ "$(left)" -ge 5 ]; then
  python run_baseline.py \
    --profile rtxpro-48 --config qwen2.5-7b --quant float16 \
    --dataset gsm8k --limit 160 --max-new-tokens 256 \
    --order prompt-major --energy \
    --deadline "$(cap $SEVEN_CAP)m" --reserve 3 \
    --out results/7b-fp16.json --resume
else
  echo "skipped: not enough time"
fi
echo "7B now at: $(done_frac results/7b-fp16.json)"

say "D. 7B traces"
trace_pair qwen2.5-7b 7b

# --------------------------------------------------------- E. wait for 3B
say "E. waiting for the 3B download ($(left) min left)"
while ! grep -q "3B ALL DONE" "$WORK/get3b.log" 2>/dev/null; do
  if [ "$(left)" -lt 12 ]; then
    echo "out of time; the download continues and resumes next slot"; break
  fi
  sleep 30
  echo "  $(du -sh "$WORK/models/Qwen2.5-3B-Instruct" 2>/dev/null | cut -f1) so far, $(left) min left"
done

# --------------------------------------------------------- F. 3B pipeline
THREE_OK=no
if grep -q "3B ALL DONE" "$WORK/get3b.log" 2>/dev/null; then
  say "F. verifying and linking 3B"
  python - <<'EOS'
import json, pathlib, sys
from safetensors import safe_open
d = pathlib.Path("/data/cs3406/team09/models/Qwen2.5-3B-Instruct")
tot = 0
for p in sorted(d.glob("*.safetensors")):
    with safe_open(p, framework="pt") as f:
        n = len(f.keys())
    tot += n
    print(f"  ok {p.name} {p.stat().st_size/2**30:.2f} GiB {n} tensors")
idx = d / "model.safetensors.index.json"
if idx.exists():
    wm = json.loads(idx.read_text())["weight_map"]
    print(f"  shards {len(list(d.glob('*.safetensors')))}/{len(set(wm.values()))}  "
          f"tensors {tot}/{len(wm)}")
    sys.exit(0 if tot == len(wm) else 1)
sys.exit(0 if tot else 1)
EOS
  if [ $? -eq 0 ]; then
    link_model Qwen2.5-3B-Instruct
    say "G. 3B correctness gate"
    if python run_baseline.py --profile rtxpro-48 --config qwen2.5-3b \
         --quant float16 --selftest 2>&1 | tee results/selftest-3b.txt; then
      THREE_OK=yes
    else
      echo "3B selftest FAILED -- not sweeping a decoder we cannot vouch for"
    fi
  else
    echo "3B weights incomplete -- skipping, rerun this script next slot"
  fi
fi

if [ "$THREE_OK" = yes ]; then
  say "H. 3B sweep ($(left) min allowed)"
  if [ "$(left)" -ge 10 ]; then
    python run_baseline.py \
      --profile rtxpro-48 --config qwen2.5-3b --quant float16 \
      --dataset gsm8k --limit 160 --max-new-tokens 256 \
      --order prompt-major --energy \
      --deadline "$(left)m" --reserve 3 \
      --out results/3b-fp16-blackwell.json --resume
  else
    echo "skipped: not enough time"
  fi
  say "I. 3B traces"
  trace_pair qwen2.5-3b 3b
fi

# ------------------------------------------------------------- J. analyse
say "J. analysis"
for f in results/7b-fp16.json results/3b-fp16-blackwell.json; do
  [ -f "$f" ] || continue
  echo "---- $f"
  python analyze.py "$f" --csv "${f%.json}.csv"
done

# ------------------------------------------------------------- K. package
say "K. packaging"
cp "$LOG" results/ 2>/dev/null
TAR=$WORK/rtxpro-$STAMP.tar.gz
tar czf "$TAR" results/
ls -lh "$TAR"
echo
echo "contents:"
tar tzf "$TAR" | sed 's/^/  /'

say "SUMMARY"
echo "  7B sweep : $(done_frac results/7b-fp16.json)"
echo "  3B sweep : $(done_frac results/3b-fp16-blackwell.json)"
echo "  traces   : $(ls results/trace-*.html 2>/dev/null | wc -l) html files"
echo "  tarball  : $TAR"
echo
echo "  Any run marked 'incomplete' continues from where it stopped if you"
echo "  re-run this script in the next slot."
say "finished at $(date +%H:%M:%S)"
