# A16 gate-failed CLEAR passes — do not quote

These files failed the gate (`--gate --gate-max-w 65`) when measured on a rented
Vultr A16 instance. Per the project's own validation protocol (%Capture /
anchor check, step 4 of the CLEAR methodology): if the amplified replay does
not reproduce what the real pass costs, the decomposition is incomplete and
we report that instead of the number, rather than quoting a bad composition.

**What's wrong:** fine-grained per-layer replay measurements
(`draft/layer/L.../qN/kvN`, `draft/prefill/prefill`, etc.) come back
systematically ~0.5-0.7x of what the real pass takes — the replay is not
running the same kernel the real pass runs. This is reproducible across every
dataset and quantization tried on this instance (gsm8k K1-16 extension,
humaneval, cnndm, and the 7B NF4 pass), so it looks like an A16-specific
kernel-selection quirk in CLEAR's fine-layer replay step, not a data or
dtype bug (contrast with the T4 dtype bug fixed in `621ec53`, which these
files don't exhibit).

The two cnndm files are additionally `"complete": false` — they hit the
200-minute deadline before finishing the full amplification pass.

**What's still trustworthy:** the original T4 CLEAR passes at
`clear-results/fp16/clear-qwen2.5-3b-float16.json` and
`clear-results/nf4/clear-qwen2.5-3b-nf4.json` (gsm8k, K=1-8) both pass gate
cleanly and are the basis for any component-attribution numbers in the
report/slides. The raw sweep energy numbers for humaneval/cnndm/the K9-16
extension (in `results/`) are unaffected by this — they come from the real
NVML-metered pass, not CLEAR's replay, and are fine to use.
