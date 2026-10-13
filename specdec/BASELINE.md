# Baseline runbook

Phases 1–2 only: get the draft-verify pipeline running and produce the
acceptance / tokens-per-round / latency curve over COGA-1…8. No energy, no
CLEAR, no NVML.

That exclusion decides the hardware. The baseline measures **acceptance rate**
and **tokens per round**, which are properties of the model pair and the
prompts, not of the GPU — the same pair gives the same acceptance on a laptop
and on an A100. Only `tok/s` is device-bound. So nothing here needs an energy
counter, a locked clock, or a quiet GPU, and the devices ruled out for later
phases are perfectly good now.

---

## Step 0 — what downloads, and what you install

**The code fetches models and data itself.** `run_baseline.py` pulls the
checkpoints through `transformers` and the prompts through `datasets` on
first use, then caches both. Qwen2.5 is ungated, so no HuggingFace token and
no licence click-through — unlike the Llama pairs, which need an approved
account.

| what | how much | handled by |
|---|---|---|
| pip packages (torch, transformers, bitsandbytes, datasets) | 2–3 GB | **you**, once |
| Qwen2.5-3B + 0.5B weights | ~6.7 GB | code, on first run |
| GSM8K | ~5 MB | code, on first run |
| HumanEval | <1 MB | code, on first run |
| CNN-DM, only if you use it | ~1.3 GB | code, on first run |

So the only required manual step is `pip install -r requirements.txt`.

Downloading *during* a run is still worth avoiding, because it hides inside
what looks like a hang and costs metered time. `prefetch.py` separates it:

```bash
python prefetch.py --check            # what's cached, what's missing
python prefetch.py                    # pull the default pair + gsm8k
```

Three practical traps:

- **Kaggle disables internet by default.** Turn on *Internet* in the notebook
  settings sidebar (it requires a phone-verified account) or every download
  fails. This catches people out because the error looks like a network
  problem rather than a setting.
- **Hosted caches are per-session.** On Kaggle and Colab `~/.cache/huggingface`
  dies with the session, so each fresh session re-pulls ~6.7 GB. Point
  `HF_HOME` somewhere that persists first:
  `export HF_HOME=/kaggle/working/hf`, or a mounted Drive folder on Colab.
- **NF4 does not shrink the download.** bitsandbytes quantizes *after* loading
  fp16 safetensors, so the 3B pull is ~6.7 GB whichever precision you run.

## Step 1 — local smoke test (~10 min, RTX 4050)

```bash
pip install -r requirements.txt
python tests/test_lossless.py          # CPU, no downloads, ~1 min
```

That must pass before anything else: it proves speculative decoding is
emitting exactly what plain greedy would.

```bash
python run_baseline.py --profile local-4050 --selftest
```

Downloads Qwen2.5-3B + 0.5B (~7.2 GB, ungated — no HF token needed), loads the
target at NF4 to fit 6 GB, and re-checks losslessness on the *real* models.

```bash
python run_baseline.py --profile local-4050
python analyze.py results/qwen2.5-3b-nf4-gsm8k.json
```

16 prompts × {COGA-1,2,4,8} × 128 tokens, roughly 6 minutes at ~30 tok/s.
Enough to see a real acceptance curve and confirm the whole path works.

On Windows, `bitsandbytes>=0.43` ships CUDA wheels, so NF4 works — but if the
import fails, run `--quant float16 --config qwen2.5-1.5b` instead (4.7 G,
still fits) and move the 3B run to Kaggle.

## Step 2 — the real sweep (~85 min, Kaggle P100)

```bash
python run_baseline.py --profile kaggle-p100 --out results/baseline.json --resume
python analyze.py results/baseline.json --csv baseline.csv
```

Full 256 GSM8K prompts × COGA-1…8 × 256 new tokens.

**Use the P100, not the T4.** It is the device to avoid for energy work — CC
6.0 is pre-Volta and has no energy counter at all — but at 732 GB/s it is more
than twice the T4's bandwidth, and the baseline does not measure energy. The
same sweep is 85 minutes on a P100 against 3.3 hours on a T4. Pick it
deliberately for this phase and avoid it later.

NF4 is the right precision here too, and not only for memory: batch-1 decode
is bandwidth-bound, so reading ~3× fewer weight bytes is a straight speedup.
It is also the precision Dutta et al. used.

`--resume` appends each row to `results/baseline.jsonl` as it completes, so a
dropped session costs one row rather than the run. Re-issue the identical
command to continue. See **Checkpointing** below for fixed-length slots.

## Checkpointing against a hard time limit

Inside a 2 h GPU slot, pass a deadline and the run lands rather than crashes:

```bash
python run_baseline.py --profile rtxpro-48 --energy \
    --deadline 110m --out results/f-3b.json --resume
```

It finishes the unit in flight, writes the merged JSON, prints what is left
and the exact command to continue. Three things make that work:

**The deadline is predictive.** It stops when the *next* unit would not fit,
using the observed mean unit time plus `--reserve` (3 min default), rather
than when the clock has already run out. Being killed mid-write leaves a
truncated line and no merged result.

**Signals are caught.** Slots and notebooks end with SIGTERM, then SIGKILL a
few seconds later. The first signal becomes a clean stop after the current
unit; a second exits immediately.

**Resuming into the wrong run is refused.** Each log carries a `.meta.json`
fingerprint of the settings that produced it. Resume with a different
precision, prompt limit or token budget and it stops and names the field that
changed, rather than silently appending rows that are not comparable:

```
refusing to resume run.jsonl: the settings changed.
      max_new_tokens: log has 128, now 64
```

Override with `--force-resume` only if you are certain. A truncated last line
from a hard kill is detected and dropped on resume, so the log stays parseable.

### Ordering matters more than it looks

By default the sweep is **prompt-major**: every strategy on prompt 1, then
every strategy on prompt 2. If the slot ends at 60%, you are left with all
eight K values over fewer prompts — a complete, noisier energy-vs-K curve.
Strategy-major (`--order strategy-major`) would instead give a precise curve
over K=1…5 and nothing at all for 6…8, which cannot be plotted. Same total
work either way; only the shape of a partial result differs.

## Step 3 — read the curve

`analyze.py` prints one row per K:

```
    K   accept  tok/round     tok/s   speedup   draft%  draft fwd/tok
   AR        -       1.00      30.1      1.00        -           0.00
    1    72.4%       1.72      34.8      1.16     27.8           0.58
    ...
  latency-optimal   K = 4
  least draft work  K = 1
```

What to look for, and what each column is for later:

- **accept** should fall monotonically as K rises. If it does not, something
  is wrong with the rollback.
- **tok/round** must stay inside [1, K+1]. The tests enforce this, but it is
  worth eyeballing on real models.
- **latency-optimal K** is the headline baseline result.
- **draft fwd/tok** is draft forward passes per emitted token. It is the term
  the energy curve will be dominated by, so it is the best preview you get of
  where E(K) turns upward — and it comes free from this phase.

## What this leaves ready

The sweep writes per-round detail (`round_detail`) for every prompt: how many
tokens were proposed and accepted in each round. That is exactly the
`N_component(K)` execution count the energy phase needs, which is why CLEAR
later does not have to be re-run per K — component energy gets measured once
and composed against these counts.

So the baseline is not just a warm-up. Done properly at 256 prompts, it is
half of the final result.

## If something looks wrong

| symptom | cause |
|---|---|
| `--selftest` fails | the port is broken; do not trust any number from it |
| acceptance ~0% at every K | tokenizer mismatch, or the draft loaded in the wrong dtype |
| acceptance 100% at every K | draft and target are the same checkpoint |
| CUDA OOM on the 4050 | close the browser; or drop to `--config qwen2.5-1.5b` |
| `no kernel image is available` | torch too old for the GPU; only an issue on Blackwell |
| speedup < 1 everywhere | expected on a bandwidth-starved card at small K — check `tok/round` is above 1 before suspecting a bug |
