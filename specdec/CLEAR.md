# CLEAR runbook

Phase 3: per-component energy for **both** models, then E(K) composed from it.
The baseline sweep already holds the call counts (`round_detail`); this phase
supplies the cost per call.

```
run_clear.py          capture -> replay check -> amplify  ->  clear-*.json
analyze.py SWEEP --clear clear-*.json                      ->  E(K), split by component
```

## What gets measured

Each model is cut into the same components:

| component | what it is | keyed by |
|---|---|---|
| `embed` | token embedding | rows |
| `rope` | rotary cos/sin tables | rows |
| `norm_in`, `norm_post` | the two RMSNorms in each layer | rows |
| `attn` | QKV + RoPE + KV concat + SDPA + o_proj | rows, **context bin** |
| `mlp` | gate/up/down + SiLU | rows |
| `norm` | final RMSNorm | rows |
| `lm_head` | vocabulary projection, logits rows kept only | rows |
| `layer` | one whole decoder layer, for the additivity check | rows, context bin |
| `forward` | one whole model call; the total E(K) is built from these | rows, context bin |
| `prefill` | the first call of a prompt, charged per prompt token | — |

"rows" is how many positions the call processes: 1 for a draft step (2 right
after a fully accepted round), K+1 for verify, 1 for autoregressive. Per-layer
components are measured on the first, middle and last layer (`--layers rep`)
and multiplied by the layer count.

The total per token comes from the `forward` measurements, since they are
what the hardware actually ran. The component split is CLEAR's breakdown of
that total, and `unattributed` is whatever the components do not cover:
kernel launches, residual adds, mask construction.

## Step 0: CPU tests (here, before any GPU time)

```bash
python tests/test_clear.py        # ~20 s, no downloads
```

These tests prove that capture sees the engine's real shapes, that every
replay reproduces its captured output exactly, that replaying attention does
not grow the KV cache, and that the call counts rebuilt from `round_detail`
are exactly the calls the models received. If any of them fails, no energy
number is worth collecting.

## Step 1: Kaggle quick check (~10 min, T4)

Notebook settings: **GPU T4**, **Internet on**. Then:

```bash
export HF_HOME=/kaggle/working/hf
cd specdec && pip install -q -r requirements.txt
python tests/test_lossless.py && python tests/test_clear.py
python probe_env.py --json results/env.json          # sensor period, the 10% bar
python run_clear.py --profile kaggle-t4 --quick
```

`--quick` uses 2 prompts, 64 tokens, K ∈ {1, 4}, the middle layer only, and
3 trials. It exercises every stage on the real NF4 models, which the CPU tests
cannot. What to check in its output:

- **replay check** must say `N/N replays reproduce their output exactly`.
  If it doesn't, stop and bring the output back.
- every amplify line should be free of `!!`, and rsd should mostly be a few
  percent. Each line shows mJ/call, ms/call and the average **W** during the
  trials; the T4 is capped at 70 W, so anything near or above that is wrong.
- **additivity**: sum of parts / whole for each forward pass. Time already
  adds up to ~1.02–1.09 on the T4, so energy should land in a similar range.
  Far below 1 means a component is missing; far above means separately
  replayed parts each pay launch overhead that the real pass overlaps.

How amplification works, since it is the part most easily got wrong: each
component first runs untimed for 1.5 s, so the GPU's power state belongs to
it rather than to whatever ran before. A T4 stays high-power for a while
after heavy work, and without settling, the first component after a heavy
one read up to 10x high. Trials then run back to back and are cut exactly at
the energy counter's ~100 ms update ticks. Reading the counter at arbitrary
times instead gave ±20% quantisation in a 500 ms window.

## Step 2: the real pass (~1.5–2 h, T4)

```bash
python run_clear.py --profile kaggle-t4 --deadline 600m --resume
python analyze.py results/3b-nf4.json --clear results/clear-qwen2.5-3b-nf4.json --csv results/clear-curve.csv
```

The default is 8 prompts × K=1…8 + autoregressive, 256 tokens, 3 layers,
a 1.5 s settle, then 5 tick-aligned trials of ≥20 sensor periods (~2 s). The script prints its own time estimate before
amplifying. Each measurement is written as it finishes, so a session that ends
costs only the key in flight: rerun the same command with `--resume`.

`results/3b-nf4.json` is the existing Kaggle T4 sweep, so the call counts
**and** the NVML whole-run energy come from the same device. That makes the
`CLEAR/meas` column a real validation: it compares composed energy against
measured energy.

### fp16 target

The draft is fp16 whichever precision the target uses, so a second pass needs
only the target:

```bash
python run_clear.py --profile kaggle-t4 --quant float16 --roles target --resume
python analyze.py results/3b-fp16.json \
    --clear results/clear-qwen2.5-3b-nf4.json results/clear-qwen2.5-3b-float16.json
```

Later files override earlier ones key by key, so this takes the draft from the
NF4 file and the target from the fp16 one.

### Packaging the results

```bash
tar czf /kaggle/working/clear-$(date +%Y%m%d-%H%M).tar.gz results/
```

## Reading `analyze.py --clear`

```
CLEAR composition (mJ above idle per emitted token)
    K      CLEAR   vs AR   draft%   measured  CLEAR/meas
   AR       ...
    1       ...
  CLEAR energy-optimal K = ...

  per-component split (mJ above idle/token)
                         AR    K=1    K=2 ...
  draft.attn            ...
  target.mlp            ...
```

- **CLEAR/meas** near 1 means the composition accounts for the energy. Well
  below 1 means real work happens outside the forward passes (accept/reject,
  rollback, host-side glue), and the gap is a result worth reporting.
- The **draft.*** rows should grow with K and the **target.*** rows should
  shrink per token. The point where their sum turns upward is K\*.
- A `nearest used` warning means the sweep made a call shape the capture
  never saw, usually a K outside `--k`. Capture with the same K list as the
  sweep.

## If something looks wrong

| symptom | cause |
|---|---|
| replay mismatch on GPU, tiny diffs | nondeterministic kernel choice; check the size, then `--allow-mismatch` if it is rounding-level |
| replay mismatch, large diffs | a component input was not captured; do not measure |
| `read 0 mJ` / window warnings | sensor period not measured; rerun `probe_env.py` |
| `rsd >= 10%` on many keys | another process on the GPU, or the card is throttling; never run two energy jobs at once, even on separate T4s, since they share the host CPU |
| first component after a heavy one reads high | power state carried over; raise `--settle-ms` |
| `energy counter stopped updating` | NVML stalled; rerun with `--resume` |
| small components look implausibly cheap | weights hot in L2 from back-to-back replay; compare a `--flush-l2` run |
| CLEAR/meas well above 1 | parts replayed alone each pay launch overhead; compare the additivity ratios |
