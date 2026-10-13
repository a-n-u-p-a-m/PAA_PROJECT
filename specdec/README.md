# Speculative decoding + CLEAR energy profiling — baseline pipeline

Phase 1–2 of the project: a correct, instrumented draft-verify pipeline that
sweeps constant draft lengths to produce the points of an energy-vs-K curve,
running our own Qwen pair under the protocol of Dutta et al. (Findings of
EACL 2026).

```
specdec/
  configs.py     model pairs, COGA/DYGA, the fixed Dutta protocol, VRAM fit
  memory.py      exact VRAM accounting from real config.json architectures
  datasets.py    HumanEval / GSM8K / CNN-DM at the paper's sample counts
  kv.py          cache length + rollback, isolating HF cache-API churn
  engine.py      draft -> verify -> accept/reject -> rollback; greedy + sampling
  meters.py      per-phase timing and NVML energy (hook point for CLEAR)
probe_env.py     can this machine measure energy, and which pairs fit?
run_baseline.py  the K sweep
analyze.py       sweep JSON -> energy-vs-K curve, and the two optima
tests/test_lossless.py   correctness gate
```

**Running the baseline only, right now? → [BASELINE.md](BASELINE.md).** This
file covers all five phases and four machines; that one is the step-by-step
for phases 1–2.

## Quick start

```bash
pip install -r requirements.txt

python tests/test_lossless.py                   # CPU, no downloads, ~1 min
python run_baseline.py --list                   # pairs, fit, and the protocol
python probe_env.py --json env.json             # can this GPU measure energy?

# smoke test the whole thing offline
python run_baseline.py --synthetic --device cpu --offline --limit 3
python analyze.py results/qwen2.5-3b-nf4-gsm8k.json

# the real sweep: one pair, one dataset, eight draft lengths
python run_baseline.py --config qwen2.5-3b --dataset gsm8k
python analyze.py results/qwen2.5-3b-nf4-gsm8k.json --csv curve.csv
```

## Scope

**One model pair.** Default `qwen2.5-3b` (Qwen2.5-3B-Instruct ← Qwen2.5-0.5B-Instruct).

| config | target ← draft | NF4 | fp16 | on a 14.7 GB T4 |
|---|---|---|---|---|
| `qwen2.5-3b` | Qwen2.5-3B-Instruct ← Qwen2.5-0.5B-Instruct | 4.1 G | 8.6 G | **fp16 or NF4** |
| `qwen2.5-7b` | Qwen2.5-7B-Instruct ← Qwen2.5-0.5B-Instruct | 7.1 G | 18.2 G | NF4 only |
| `qwen3-4b` | Qwen3-4B ← Qwen3-0.6B | 5.0 G | 10.8 G | fp16 or NF4 |
| `qwen3-8b` | Qwen3-8B ← Qwen3-0.6B | 7.8 G | 19.6 G | NF4 only |

VRAM figures for the Qwen rows are exact, computed from each model's real
`config.json` by `specdec/memory.py` (`python -m specdec.memory 15` prints the
full breakdown). A bytes-per-parameter heuristic gets this wrong, because
bitsandbytes quantizes `nn.Linear` only — Qwen's 151,936-row embedding table
stays in fp16, and it is 28% of all parameters in the 0.5B draft.

**Qwen2.5-3B ← Qwen2.5-0.5B, 256-token prompt, K=8:**

| | fp16 | NF4 target |
|---|---|---|
| target weights | 5.75 G | 1.91 G |
| draft weights (fp16 either way) | 0.92 G | 0.92 G |
| KV caches, both models | 0.02 G | 0.02 G |
| logits | 0.01 G | 0.01 G |
| activations | 0.12 G | 0.12 G |
| CUDA context + workspace | 0.75 G | 0.75 G |
| **total** | **7.57 G** | **3.73 G** |

Both fit a 15 GB T4 with room to spare, and NF4 fits the 6 GB laptop card
beside a display. Context length
is nearly free: Qwen2.5-3B uses 2 KV heads across 36 layers, so 36 KiB/token,
and going from a 128-token prompt to 4096 moves the total only 7.55 → 8.06 G.

The term that would have bitten is the logits tensor: `[1, P, 151936]` upcast
to fp32 is 0.58 MiB *per position*, so a 1000-token CNN-DM prefill would
materialise ~560 MB of logits we immediately discard — eight times the KV
cache. The engine passes `logits_to_keep` (K+1 rows for the target, 1 for the
draft), which cuts it to 0.01 G. The correctness tests cover that path.

Two things about the model ladder, because they constrain what pairs are even
possible: **Qwen2.5 has no 8B** (it goes 0.5 / 1.5 / 3 / 7 / 14 B) and **Qwen3
has no 3B and no 0.5B** (0.6 / 1.7 / 4 / 8 B). And draft and target must share
a tokenizer, since speculative decoding verifies the draft's token *ids* — a
vocabulary mismatch produces silent garbage rather than an error, so
`run_baseline.py` checks it and refuses. So a 0.5B draft means Qwen2.5 and a
3B or 7B target; an 8B target means Qwen3 and a 0.6B draft. `qwen3-4b` and
`qwen3-8b` are additionally Dutta's own pairs, if a closer comparison is
wanted later.

**One dataset.** Default GSM8K, 256 prompts (the paper's count). GSM8K is the
right one to sweep on: HumanEval is code, where acceptance is so high the
optimum sits at the top of the K range, and CNN-DM's ~800-token articles mean
prefill energy — which does not depend on K — swamps the signal we are
looking for. GSM8K sits in between with bounded outputs. `--dataset humaneval`
or `--dataset cnndm` if you want to check that intuition; `--limit 128` halves
the runtime.

**COGA only.** The default sweep is COGA-1 … COGA-8: eight constant draft
lengths, eight points on E(K). DYGA is implemented and tested but kept out of
the sweep on purpose — it changes K *within* a run, so its mean draft length
is an outcome rather than a setting. A DYGA run is a scatter point to mark
against the curve, never a point on it. `--strategy dyga-20` adds it if you
want that marker.

## The protocol, held at Dutta et al.'s values

> Dutta, Koley, Poddar, Misra, Podder, Balani, Ghosh, Ganguly.
> *Benchmarking the Energy Savings with Speculative Decoding Strategies.*
> Findings of the ACL: EACL 2026, 4737–4748. [arXiv:2602.09113](https://arxiv.org/abs/2602.09113)

`DUTTA_PROTOCOL` in `configs.py` is the single source of truth, and every run
records it:

| knob | value | source |
|---|---|---|
| target quantization | NF4, double quant | paper §4 (QLoRA-style, Dettmers et al. 2023) |
| NF4 compute dtype | `float16` | ours — keeps one dtype across a T4 and a Blackwell card |
| draft precision | `float16`, unquantized | paper does not state it |
| decoding | greedy | paper |
| batch size | 1, sequential | paper |
| chat template | model-specific | paper |
| max new tokens | 256 | **not stated in the paper**; recorded per run |
| dataset seed | 42 | **not published**; recorded per run |

Anything set away from these is printed at startup and written into the
output under `protocol_deviations`. Holding the protocol fixed means a
difference we observe is attributable to the draft length, not to a decoding
setting drifting underneath us.

What it does *not* buy: our absolute joules are not comparable to theirs.
Different models, and our own mix of devices rather than their A5000 (24 GB,
CC 8.6) — different power envelopes throughout. What carries across is the
methodology and the qualitative claim.

One live tension worth deciding deliberately. NF4 is the paper's setting, but
every quantized matmul carries a dequantization kernel, so a component-level
breakdown of an NF4 model describes *quantized inference*, not the model.
`qwen2.5-3b` is the pair that fits every device here both ways, so you can run
`--quant nf4` and `--quant float16` and report both. The delta between them is
a result in its own right, and one CodeCarbon's model-level view could not
have produced.

## Reading the output

`analyze.py` collapses the sweep into one row per K:

```
    K   accept  tok/round     tok/s   speedup   draft%  draft fwd/tok   mJ/token    vs AR
   AR        -       1.00     895.9      1.00        -           0.00       0.37     1.00
    1    65.8%       1.64     600.7      0.67     47.4           0.62       0.58     1.56
    ...
  latency-optimal   K = 1
  energy-optimal    K = 1
```

The expectation this is built to test: latency improves with aggressive
drafting, because a larger K amortises the target's forward pass over more
tokens. Energy does not, because every rejected draft token is work that
produced nothing, and the draft model's cost grows linearly in K while
acceptance falls. So E(K) should turn upward before throughput does, and
K\* < K_latency. Without `--energy`, the `draft fwd/tok` column (draft forward
passes per emitted token) is the best available proxy for the term that drives
the turn.

## The correctness gate

Greedy speculative decoding is **lossless**: it must emit exactly the tokens
the target model would have produced alone. Any mistake in the logit row
offset or the KV-cache rollback breaks that, and the resulting energy numbers
would describe a decoder that isn't the one we claim to be studying.

`tests/test_lossless.py` asserts token-identical output against plain greedy
for COGA K ∈ {1,2,3,4,5,8} across four draft qualities — including a draft
identical to the target (must accept 100%, catching off-by-one errors a weaker
draft would hide) and an independently initialised one (accepts 0%, exercising
the reject path every round) — and for DYGA-{1,4,20}, where the draft length
changes between rounds. Losslessness must hold for *any* schedule; if it only
held at constant K, the rollback would be silently depending on K not changing.
Sampling mode is checked separately: modified rejection sampling must
reproduce the target's distribution, verified to TV distance 0.023 over 1200
draws. All currently pass.

## Index bookkeeping (where this algorithm usually goes wrong)

With prefix length `n` and drafts `d₁…d_K`, the target runs **one** forward
pass over `n+K` tokens. Logit row at absolute position `p` predicts token
`p+1`, so rows `n-1 … n+K-1` — exactly `K+1` rows — are the judges:

- row `n-1+i` judges `dᵢ`
- row `n+K-1` supplies the bonus token when all K are accepted

A round therefore emits between 1 and K+1 tokens, never zero. After accepting
`m` drafts, both caches are cropped to `n+m`: entries beyond that were
computed from a continuation just discarded. The draft cache holds only
`n+K-1` positions, because `d_K` was produced but never fed back in.

## Where CLEAR attaches

`PhaseMeter.phase("draft")` / `.phase("verify")` in `engine.py` are the seams.
Phase 3 replaces the NVML whole-phase read with CLEAR's three-stage pipeline:
hook the component boundaries, cache activations, re-execute each component
N≈10³–10⁴ times on those cached activations, and divide. The whole-phase NVML
read in `meters.py` is deliberately coarse and labelled as such: a draft phase
lasting tens of microseconds sits below the NVML sensor's millisecond cadence,
which is exactly the mismatch CLEAR's amplification exists to defeat. Once
those numbers exist, `analyze.py` already has the column for them.


# Four machines, and what each is for

```bash
python plan_run.py --devices all --prompts 256     # the whole table
python plan_run.py --calibrate --config qwen2.5-3b --quant nf4   # on the machine
```

| device | VRAM | GB/s | CC | energy counter | budget |
|---|---|---|---|---|---|
| RTX 4050 Laptop | 6 (5.2 usable) | 192 | 8.9 | yes | unlimited |
| Colab free T4 | 14.7 | 320 | 7.5 | yes | 12 h/session |
| Kaggle T4 | 14.7 | 320 | 7.5 | yes | 12 h/session, ~30 h/week |
| Kaggle P100 | 15.9 | 732 | 6.0 | **no** | 12 h/session, ~30 h/week |
| RTX PRO 5000 | 47 | 1344 | 12.0 | yes | 3 × 2 h = 6 h total |

Two results fall out of this that drive everything else.

**NF4 is faster than fp16 on every one of these cards, not just smaller.**
Batch-1 decode is memory-bandwidth bound, so reading ~3× fewer weight bytes
per forward pass is a straight speedup, and the dequantization kernel does not
eat it. On a T4 the 3B pair goes 34 → 50 tok/s; on the RTX PRO, 132 → 189.
Dutta's NF4 choice was not only about fitting 24 GB. It does still change what
a component-level energy breakdown describes, so fp16 remains the scientific
reference wherever it fits — but NF4 is the sensible default everywhere else.

**A Kaggle T4 is a valid measurement device — measured, not assumed.**
We expected the 70 W cap and the inability to lock clocks to put a hosted T4
out of reach. Run properly it reaches **2.3% relative standard deviation**,
well inside CLEAR's 10% bar. The first attempt said 207%, but that was an
artefact: the counter advances only about every 100 ms and the window was
27 ms, so most trials read zero and a few caught a whole tick. `probe_env.py`
now measures the sensor period and sizes the window to it.

The power cap may be *why* it is stable: a card pinned at its limit does not
boost and throttle around a setpoint, it simply sits there. That leaves the
4050 (drives a display) and the P100 (no energy counter at CC 6.0) as the
machines genuinely unfit for measurement, and it makes Kaggle's ~30 h/week —
five times the RTX PRO budget — the main place energy work can happen.

## Sweep times, 256 prompts, COGA-1…8, 256 new tokens

Modelled at 70% acceptance; `--calibrate` replaces these with measurements.

| pair | 4050 NF4 | T4 fp16 | T4 NF4 | P100 NF4 | RTXPRO fp16 | RTXPRO NF4 |
|---|---|---|---|---|---|---|
| qwen2.5-1.5b | 4.3 h | 3.6 h | 2.6 h | 68 m | 55 m | 44 m |
| qwen2.5-3b | 5.4 h | 5.4 h | 3.3 h | 85 m | **81 m** | 51 m |
| qwen2.5-7b | — | — | 5.8 h | 2.5 h | 2.6 h | 87 m |
| qwen2.5-14b | — | — | 9.0 h | 3.9 h | 4.6 h | 2.2 h |
| qwen3-4b | — | 6.8 h | 4.1 h | 1.8 h | 1.7 h | 62 m |
| qwen3-8b | — | — | 6.5 h | 2.9 h | 2.8 h | 1.6 h |
| vicuna-7b | — | 8.1 h | 3.1 h | 80 m | 1.9 h | 44 m |
| vicuna-13b | — | — | 5.5 h | 2.4 h | 3.7 h | 78 m |

At 64 prompts, everything fits a single 2 h RTX PRO slot, including
`qwen2.5-14b` fp16 at 69 m.

## The plan

**Local 4050** — `--profile local-4050`. Code, correctness tests, and
`qwen2.5-3b` at NF4 (3.7 G) for functional runs. The energy counter works
here, which is useful for checking the plumbing, but a display-driving laptop
GPU will not hold a clock steady enough to mean anything.

**Kaggle, ~30 h/week** — `--profile kaggle-t4`. This is five times the total
compute of the RTX PRO slots, and it should carry the bulk work: the full
256-prompt acceptance and tokens-per-round curves, prompt-category
stratification, and the CLEAR activation capture. Acceptance rate is a
property of the model pair, not the GPU, so that data is fully reportable —
only timings and energy are device-bound. Use `kaggle-p100` for the same work
when a T4 is unavailable; it is faster, and the missing energy counter does
not matter for acceptance data. Never pass `--energy` on the P100.

**Colab** — `--profile colab-t4`, as overflow beside Kaggle. Always `--resume`.

**RTX PRO, 6 h** — `--profile rtxpro-48`. Energy only.

One structural point makes those 6 hours sufficient: **CLEAR does not have to
be re-run per K.** Component energy is a property of the component, and what
changes with draft length is only how many times each component executes. So
measure E_component once, take the execution counts N_component(K) from the
Kaggle sweep, and compose E(K) = Σ E_component × N_component(K). The energy
sweep collapses from "8 draft lengths × components" to "components".

A workable three-slot split:

- **Slot 1** — environment and trust. Install torch ≥ 2.7/cu128, run
  `probe_env.py` (confirm sm_120 kernels, energy counter, and the <10%
  std-dev bar), `--selftest`, then the full 256-prompt `qwen2.5-3b` fp16
  latency sweep at ~81 m.
- **Slot 2** — CLEAR capture and amplification for `qwen2.5-3b`, fp16 and
  NF4. The quantization delta in the component split is a result Dutta's
  model-level CodeCarbon view could not have produced.
- **Slot 3** — one larger target for size scaling (`qwen2.5-7b` fp16, 64
  prompts, 39 m) plus its CLEAR pass, giving a 3B→7B curve.

Pre-cache every checkpoint before a slot opens. NF4 does not shrink the
download — bitsandbytes quantizes after loading fp16 safetensors — so a 7B
pull is ~15 GB and a 14B is ~28 GB, which on a shared link can outlast the
slot itself.

## Things that will bite

**sm_120 needs torch ≥ 2.7 / cu128.** Blackwell is compute capability 12.0,
and a PyTorch without those kernels fails at the first matmul with `no kernel
image is available for execution on the device` — *after* the model loads.
`probe_env.py` checks `torch.cuda.get_arch_list()` against the device and says
so up front. The 2.6 build Dutta used predates Blackwell entirely.

**Kaggle P100 has no energy counter.** CC 6.0 is pre-Volta, so
`nvmlDeviceGetTotalEnergyConsumption` does not exist. NF4 still works (it only
needs Pascal), so the pipeline runs perfectly and produces no energy data at
all. The most expensive machine to pick by accident.

**Size the measurement window to the sensor, not to a guess.** The energy
counter advances in discrete ticks — about every 100 ms on a T4. Measure a
shorter window than that and most trials read zero while a few catch a whole
tick, which looks like catastrophic instability and is really aliasing. A
27 ms window gave 207% relative standard deviation on a card that reaches
2.3% at 1.7 s. `probe_env.py` measures the period and auto-sizes; never pass
a hand-picked `--reps` without checking the window against it.

**Clock locking is still unavailable on hosted notebooks**, and
`nvmlDeviceSetGpuLockedClocks` will fail there. It turns out to matter less
than expected for a power-capped card, but it is one more reason to re-run
the probe on any machine before trusting its numbers rather than assuming
this result transfers.

**One device per table.** A 4050, a T4 and a Blackwell card differ in process
node, power envelope and memory technology. Energy numbers from different
devices cannot share an axis. All reported energy comes from the RTX PRO.

**Always `--resume`.** Every row is appended to a `.jsonl` beside `--out`, so
a dropped Colab session or an expiring slot costs one row rather than a run.

## References

- Leviathan, Kalman & Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv 2023.
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. (NF4)
- Jain et al. *CLEAR: Component-Level Energy Assessment via Repetitions.* 2025.
- Dutta et al. *Benchmarking the Energy Savings with Speculative Decoding Strategies.* Findings of EACL 2026. [arXiv:2602.09113](https://arxiv.org/abs/2602.09113)
