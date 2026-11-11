# CLEAR runbook

Phase 3: per-component energy for **both** models, then E(K) composed from it.
The baseline sweep already holds the call counts (`round_detail`); this phase
supplies the cost per call.

```
run_clear.py          capture -> in-pass timing -> replay check -> amplify
                                                            ->  clear-*.json
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

`--gate` makes the run judge the file it just wrote and exit 3 if it is not
fit to compose from; `kaggle_run.sh` passes it for both the quick check and
the full pass, at whatever precision. To gate a file that already exists,
with no model or GPU:

```bash
python run_clear.py --check results/clear-qwen2.5-3b-float16.json
```

The gate fails on: a replay mismatch, more than 10% of measurements outside
the bar, metering overhead above 5%, forward additivity outside 0.8–1.25, a
missing role, **a part that takes longer replayed alone than the whole that
contains it**, **an amplified cost that does not match what the component
costs inside the real pass** (0.7–1.4), and **a component whose cost is not
reproducible in a fresh process** (over 25%, when `--repeat` was used). The
last three all work in ms, so they need no counter. It warns, without
failing, when layer additivity leaves 0.8–1.25, when fewer rows cost more
than more rows, or when a file carries no in-pass times at all. The total is
built from forward measurements, so a warning does not corrupt E(K), but it
does mean the component split is not yet trustworthy at that level.

### The anchor, and why the other checks were not enough

The replay check proves a replayed call returns the same **values**. It does
not prove it does the same **work**. cuBLAS picks a kernel partly from the
workspace available, caches the pick for the life of the process, and may
pick differently for a tight loop of one component than for that component
inside a forward pass. A bad pick is then *stable*: it survives the settle
window, reads a steady rsd, and keeps W plausible, because time and energy
inflate together.

Both real failures were this, and neither was caught by the gates CLEAR
publishes:

| | what happened | what saw it |
|---|---|---|
| fp16 `target/mlp` q1 | 2.30 ms amplified, 0.55 ms in a fresh process, inside a 1.31 ms layer | part-vs-whole, and additivity at 2.11 |
| NF4 `target/lm_head` q1 | 10.89 ms amplified against 2.50 ms in the pass | **nothing** — the file passed |

`lm_head` sits in no layer, so part-vs-whole has no whole to compare it
with, and 8 ms of excess against a 75 ms forward pass leaves additivity at
1.02. So stage 1b times every component **where it actually runs**, by hooks
with CUDA events and no synchronisation in the loop, and the gate compares
every amplified measurement against that. The band is 0.7–1.4 rather than
tight around 1: a part replayed alone legitimately pays launch and sync
costs that overlap in the real pass (a sound T4 `lm_head` read 1.33×), and
an in-pass span carries the hook's own overhead. The failures it exists to
catch were 4.2× and 4.4×.

Each amplify line prints `anchor N.NNx`, with `!!` when it leaves the band.
`--no-anchor` skips the stage; nothing then checks this.

Two things only warn rather than fail, because in both the ratio is an
artefact of how the span is taken:

- **components under 0.2 ms amplified.** An in-pass span is bracketed on the
  stream, so for a launch-bound component it also covers the gaps where the
  GPU waits for the CPU, and the replay looks faster than the pass. The
  floor is on the amplified time, not the in-pass time — an inflated in-pass
  reading is what this is protecting against, so letting it clear its own
  floor would exempt the cases worth catching. `embed`, `rope` and the norms
  sit here, and together they are a few percent of a forward pass.
- **in-pass times taken without CUDA events** (any CPU run), which charge
  Python dispatch to the component.

To screen a file measured **before** the anchor existed, with no GPU slot and
in a couple of minutes:

```bash
python diagnose_replay.py --profile kaggle-t4 --quant nf4 \
    --audit results/clear-qwen2.5-3b-nf4.json
```

That times every component in place and lists each measurement whose
amplified cost does not match, widest ratio first. Those cells have to be
**re-measured, not rescaled**: the GPU really spent that energy, it just
spent it running a kernel the real pass does not use.

To explain one component rather than screen all of them:

```bash
python diagnose_replay.py --profile kaggle-t4 --quant float16
```

### Repeatability across processes

`rel_std_pct` is trial-to-trial spread *within* one process, which is the
small variance component — 1.4% median on the T4. The large one is between
processes, where the same unquantised `lm_head` matmul read 3.66 ms in one
pass and 16.89 ms in another. `--repeat N` measures everything again, each
time in a fresh process, and records how far each component moved:

```bash
python run_clear.py --profile kaggle-t4 --quick --repeat 3 --gate
```

It re-execs rather than looping, because a repeat inside one process cannot
see the thing worth seeing. It belongs on the quick check, where it is
cheap; `kaggle_run.sh` runs it there by default (`REPEAT=1` to skip,
`REPEAT=3` if the slot allows). Components below 0.05 ms are exempt — a call
of a few microseconds swings for reasons unrelated to kernel choice, and no
composed total leans on one.

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

Files merge per model. The fp16 file's target runs at a different precision
from the NF4 file's, so it replaces the NF4 target wholesale, not key by key:
a key the fp16 pass lacks is never filled in from NF4. The draft is fp16 in
both and comes from the NF4 file. The report prints which file and precision
each model was taken from. A target-only file records `meta.roles:
["target"]` and cannot be composed alone; `analyze.py` says so instead of
quietly producing a table.

On Kaggle, `QUANT=float16 bash kaggle_run.sh` does all of this, gate
included. If no NF4 pass is in `results/`, it measures both models.

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

- **+/-** is one standard error on the composed total, carried through from
  each forward measurement's trial-to-trial spread. Totals at different K
  share measurements, so the line under the table tests the *difference*
  from the best K and lists every K within 2 s.e. of it. Quote that set, not
  the single K, when the curve is flat. It is a lower bound: it leaves out
  systematic error and prompt sampling.
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
| gate: a part is slower alone than its whole | the replay runs a different kernel or memory path from the real pass; run `diagnose_replay.py` and do not use that row count until it is explained |
| gate: amplified cost does not match the in-pass cost | the replay is running a kernel the pass does not. Re-measure that cell; do not rescale it. `--audit` lists every cell with the problem, `diagnose_replay.py` on one component says which variant brings it back down |
| gate: not reproducible in a fresh process | the kernel pick differs between processes. `CUBLAS_WORKSPACE_CONFIG` is set to pin it; if the spread survives that, quote the spread as the error bar rather than one pass's number |
| `no in-pass timings in this file` | measured with `--no-anchor`, or before stage 1b existed. Screen it with `--audit` before quoting any component split |
| `the sweep left NF4's unquantized modules in ... and the components were measured in ...` | bitsandbytes leaves some modules alone and, unless told a dtype, they keep the checkpoint's — bf16 for Qwen2.5. `lm_head` is one, and a bf16 `lm_head` GEMM is 4.6x the fp16 one on a T4 from two logit rows up, which every verify call pays and the one-row AR baseline does not. `results/3b-nf4.json` ran that way; measure its CLEAR pass with `--nf4-unquantized-dtype checkpoint` to match, or re-sweep with the default |
| CLEAR/meas well above 1 | parts replayed alone each pay launch overhead; compare the additivity ratios |
