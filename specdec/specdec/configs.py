"""Model pairs, decoding strategies, and the fixed protocol from Dutta et al.

    Rohit Dutta, Paramita Koley, Soham Poddar, Janardan Misra, Sanjay Podder,
    Naveen Balani, Saptarshi Ghosh, Niloy Ganguly.
    "Benchmarking the Energy Savings with Speculative Decoding Strategies."
    Findings of the ACL: EACL 2026, pp. 4737-4748.  arXiv:2602.09113

We run our own Qwen model pair, but hold every *protocol* knob at the paper's
value -- NF4 target quantization with double quant, greedy decoding, batch
size 1, model-specific chat templates. ``DUTTA_PROTOCOL`` below is the single
source of truth for those, and ``run_baseline.py`` records it in every result
file alongside any deviation.

What that buys and what it costs: holding the protocol fixed means a
difference we observe is attributable to the model pair or the draft length,
not to a decoding setting drifting underneath us, and it keeps our pipeline
auditable against a published one. It does *not* make our absolute joules
comparable to theirs -- different models on different silicon. What carries
across is the methodology and the qualitative claim, that latency speedup and
energy saving peak at different draft lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Architecture = Literal["decoder-only", "encoder-decoder"]

# --------------------------------------------------------------------------
# the protocol we hold fixed at the paper's values
# --------------------------------------------------------------------------

DUTTA_PROTOCOL: dict[str, object] = {
    # precision -- their Sec. 4: targets quantized to NF4 (QLoRA-style,
    # Dettmers et al. 2023). Assistant precision is not stated; we leave the
    # draft unquantized, since quantizing a 0.5B draft saves a few hundred MB
    # and distorts its energy profile out of proportion to the memory won.
    "target_quantization": "nf4",
    "nf4_double_quant": True,
    "nf4_compute_dtype": "float16",  # V100 has no native bf16
    "draft_precision": "float16",
    # decoding
    "do_sample": False,  # greedy
    "batch_size": 1,  # samples processed sequentially
    "chat_template": True,  # model-specific templates
    # their energy tooling, replaced by ours
    "their_energy_tool": "CodeCarbon, 1 s sampling, PUE 1.0",
    # stated nowhere in the paper -- our choice, recorded for honesty
    "max_new_tokens": "not stated in paper; ours is recorded per run",
    "dataset_seed": "not stated in paper; ours is 42",
}

PAPER_HARDWARE = "NVIDIA A5000 24GB (A6000 48GB for the 70B pair)"
PAPER_SOFTWARE = "PyTorch 2.6 / CUDA 12.8 / Transformers 4.51"

# Effective bytes per parameter. NF4 is 4 bits plus blockwise absmax and
# double quantization, giving ~4.13 bits/param; rounded up for headroom.
BYTES_PER_PARAM: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "nf4": 0.55,
    "int8": 1.0,
}

CUDA_CONTEXT_GB = 0.6


@dataclass(frozen=True)
class ModelPair:
    key: str
    label: str
    family: str
    target_id: str
    draft_id: str
    target_params_b: float
    draft_params_b: float
    architecture: Architecture = "decoder-only"
    #: "project" = what we run; "paper" = Dutta's, kept for provenance
    source: Literal["project", "paper"] = "paper"
    paper_name: str = ""
    notes: str = ""

    def weights_gb(self, target_precision: str, draft_precision: str) -> float:
        return (
            self.target_params_b * BYTES_PER_PARAM[target_precision]
            + self.draft_params_b * BYTES_PER_PARAM[draft_precision]
        )

    def estimated_vram_gb(
        self,
        target_precision: str = "nf4",
        draft_precision: str = "float16",
        seq_len: int = 2048,
    ) -> float:
        """Total VRAM at peak.

        Exact for the Qwen pairs, computed from their real configs by
        ``specdec.memory`` -- which matters, because a bytes-per-parameter
        heuristic misses that NF4 leaves the 151,936-row embedding table in
        fp16. For pairs without a registered arch spec this falls back to a
        parameter-count approximation. Either way ``run_baseline.py`` reports
        the measured peak from ``torch.cuda.max_memory_allocated``; trust that.
        """
        try:
            from .memory import ARCHS, budget
        except ImportError:  # run as a bare script rather than a module
            from memory import ARCHS, budget  # type: ignore[no-redef]

        if self.target_id in ARCHS and self.draft_id in ARCHS:
            return budget(
                self.target_id, self.draft_id,
                target_precision=target_precision,
                draft_precision=draft_precision,
                prompt_len=seq_len // 4,
                max_new_tokens=seq_len // 4,
            ).total

        kv_and_act = 0.12 * (self.target_params_b + self.draft_params_b) * (
            seq_len / 2048
        ) + 0.4
        return (
            self.weights_gb(target_precision, draft_precision)
            + kv_and_act
            + CUDA_CONTEXT_GB
        )

    @property
    def exact(self) -> bool:
        """True when the VRAM figure comes from the real config, not a guess."""
        try:
            from .memory import ARCHS
        except ImportError:
            from memory import ARCHS  # type: ignore[no-redef]

        return self.target_id in ARCHS and self.draft_id in ARCHS

    def fits(
        self,
        vram_gb: float,
        target_precision: str = "nf4",
        draft_precision: str = "float16",
        seq_len: int = 2048,
    ) -> bool:
        return self.estimated_vram_gb(target_precision, draft_precision, seq_len) <= vram_gb

    @property
    def supported(self) -> bool:
        """Our engine is decoder-only; T5 would need a separate code path."""
        return self.architecture == "decoder-only"


# --------------------------------------------------------------------------
# the pairs we actually run
# --------------------------------------------------------------------------
#
# Qwen2.5 ships 0.5B / 1.5B / 3B / 7B / 14B / ... and Qwen3 ships
# 0.6B / 1.7B / 4B / 8B / ... -- so there is NO Qwen2.5-8B and NO Qwen3-3B.
# A 0.5B draft therefore belongs to Qwen2.5 and an 8B target to Qwen3, and
# they cannot be mixed: two-model speculative decoding verifies the draft's
# token *ids*, so draft and target must share a tokenizer. Hence two separate
# ladders below, each self-consistent.

_PROJECT = [
    ModelPair(
        key="qwen2.5-1.5b",
        label="Qwen2.5-1.5B <- Qwen2.5-0.5B",
        family="qwen2.5",
        target_id="Qwen/Qwen2.5-1.5B-Instruct",
        draft_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_params_b=1.54,
        draft_params_b=0.49,
        source="project",
        notes="Smallest useful pair: fits a 6GB laptop card unquantized. "
              "Development only -- a 3x size ratio is a weak speculative "
              "setup, so its K* will not match the 3B pair's.",
    ),
    ModelPair(
        key="qwen2.5-3b",
        label="Qwen2.5-3B <- Qwen2.5-0.5B",
        family="qwen2.5",
        target_id="Qwen/Qwen2.5-3B-Instruct",
        draft_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_params_b=3.09,
        draft_params_b=0.49,
        source="project",
        notes="Primary config: fits a 16GB V100 unquantized, so component "
              "energies describe the model rather than the dequantizer.",
    ),
    ModelPair(
        key="qwen2.5-7b",
        label="Qwen2.5-7B <- Qwen2.5-0.5B",
        family="qwen2.5",
        target_id="Qwen/Qwen2.5-7B-Instruct",
        draft_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_params_b=7.62,
        draft_params_b=0.49,
        source="project",
        notes="Qwen2.5 has no 8B; 7B is the nearest rung. NF4 only on a V100.",
    ),
    ModelPair(
        key="qwen3-8b",
        label="Qwen3-8B <- Qwen3-0.6B",
        family="qwen3",
        target_id="Qwen/Qwen3-8B",
        draft_id="Qwen/Qwen3-0.6B",
        target_params_b=8.19,
        draft_params_b=0.60,
        source="paper",
        paper_name="QWEN3-8B",
        notes="Dutta's exact pair. Qwen3 has no 3B and no 0.5B draft. "
              "Thinking mode is disabled in the chat template.",
    ),
    ModelPair(
        key="qwen2.5-14b",
        label="Qwen2.5-14B <- Qwen2.5-0.5B",
        family="qwen2.5",
        target_id="Qwen/Qwen2.5-14B-Instruct",
        draft_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_params_b=14.77,
        draft_params_b=0.49,
        source="project",
        notes="Needs ~30 GB fp16: a 48 GB card only. The third point on a "
              "3B/7B/14B size-scaling curve, which a 16 GB V100 cannot give.",
    ),
    ModelPair(
        key="qwen2.5-32b",
        label="Qwen2.5-32B <- Qwen2.5-0.5B",
        family="qwen2.5",
        target_id="Qwen/Qwen2.5-32B-Instruct",
        draft_id="Qwen/Qwen2.5-0.5B-Instruct",
        target_params_b=32.76,
        draft_params_b=0.49,
        source="project",
        notes="NF4 only even at 48 GB, and a ~62 GB download. Not comparable "
              "to the fp16 points, so only worth it for a quantization study.",
    ),
    ModelPair(
        key="qwen3-14b",
        label="Qwen3-14B <- Qwen3-0.6B",
        family="qwen3",
        target_id="Qwen/Qwen3-14B",
        draft_id="Qwen/Qwen3-0.6B",
        target_params_b=14.77,
        draft_params_b=0.60,
        source="project",
    ),
    ModelPair(
        key="qwen3-32b",
        label="Qwen3-32B <- Qwen3-0.6B",
        family="qwen3",
        target_id="Qwen/Qwen3-32B",
        draft_id="Qwen/Qwen3-0.6B",
        target_params_b=32.76,
        draft_params_b=0.60,
        source="project",
        notes="NF4 only even at 48 GB; ~62 GB download.",
    ),
    ModelPair(
        key="qwen3-4b",
        label="Qwen3-4B <- Qwen3-0.6B",
        family="qwen3",
        target_id="Qwen/Qwen3-4B",
        draft_id="Qwen/Qwen3-0.6B",
        target_params_b=4.02,
        draft_params_b=0.60,
        source="paper",
        paper_name="QWEN3-4B",
        notes="Dutta's exact pair, and fits a V100 unquantized.",
    ),
]

# Kept so --list can show what the paper covered and why we excluded the rest.
_PAPER_ONLY = [
    ModelPair("vicuna-7b", "Vicuna-7B <- Vicuna-68M", "vicuna",
              "lmsys/vicuna-7b-v1.3", "double7/vicuna-68m", 6.74, 0.068,
              source="paper", paper_name="VICUNA-7B",
              notes="Their headline pair; COGA/DYGA can increase energy on it."),
    ModelPair("vicuna-13b", "Vicuna-13B <- Vicuna-68M", "vicuna",
              "lmsys/vicuna-13b-v1.3", "double7/vicuna-68m", 13.0, 0.068,
              source="paper", paper_name="VICUNA-13B"),
    ModelPair("llama-8b", "Llama-3.1-8B <- Llama-3.2-1B", "llama",
              "meta-llama/Llama-3.1-8B-Instruct",
              "meta-llama/Llama-3.2-1B-Instruct", 8.03, 1.24,
              source="paper", paper_name="LLAMA-8B",
              notes="Gated repo: needs a HuggingFace token."),
    ModelPair("llama-70b", "Llama-3.3-70B <- Llama-3.2-1B", "llama",
              "meta-llama/Llama-3.3-70B-Instruct",
              "meta-llama/Llama-3.2-1B-Instruct", 70.6, 1.24,
              source="paper", paper_name="LLAMA-70B",
              notes="Needed an A6000 48GB even at NF4. Out of reach."),
    ModelPair("flan-t5-large", "Flan-T5-Large <- Flan-T5-Base", "flan-t5",
              "google/flan-t5-large", "google/flan-t5-base", 0.78, 0.25,
              architecture="encoder-decoder", source="paper",
              paper_name="FLAN-T5-L",
              notes="Encoder-decoder: our cache handling is decoder-only."),
    ModelPair("flan-t5-xl", "Flan-T5-XL <- Flan-T5-Base", "flan-t5",
              "google/flan-t5-xl", "google/flan-t5-base", 2.85, 0.25,
              architecture="encoder-decoder", source="paper",
              paper_name="FLAN-T5-XL",
              notes="Encoder-decoder: our cache handling is decoder-only."),
]

MODEL_PAIRS: dict[str, ModelPair] = {p.key: p for p in _PROJECT + _PAPER_ONLY}
PROJECT_PAIRS: list[str] = [p.key for p in _PROJECT]

DEFAULT_PAIR = "qwen2.5-3b"
DEFAULT_DATASET = "gsm8k"


# --------------------------------------------------------------------------
# machine profiles
# --------------------------------------------------------------------------
#
# One flag per machine, so moving a phase between them is not a hand-edit.
# The deliberate choice here: the *model pair stays the same* across local and
# the measurement GPU, and only precision changes. Keeping the pair fixed
# means local runs rehearse the real thing rather than being a different
# experiment, and the only variable to re-validate after a move is precision.

@dataclass(frozen=True)
class Profile:
    name: str
    config: str
    quant: str
    dataset: str
    limit: int | None
    max_new_tokens: int
    repeats: int
    strategies: list[str] | None
    note: str


PROFILES: dict[str, Profile] = {
    "local-4050": Profile(
        name="local-4050",
        config="qwen2.5-3b",
        quant="nf4",            # 3.7 G: fp16 will not fit 6 GB beside a display
        dataset="gsm8k",
        limit=16,               # enough to shake out the pipeline, not to report
        max_new_tokens=128,
        repeats=1,
        strategies=["coga-1", "coga-2", "coga-4", "coga-8"],
        note="Development only. The pair matches the measurement runs so the "
             "port is one flag, but nothing timed here is reportable: the GPU "
             "drives a display and throttles on a laptop power budget.",
    ),
    "kaggle-t4": Profile(
        name="kaggle-t4",
        config="qwen2.5-3b",
        quant="nf4",            # ~1.5x faster than fp16 on a bandwidth-bound T4
        dataset="gsm8k",
        limit=None,             # the full 256 fits one 12 h session
        max_new_tokens=256,
        repeats=1,
        strategies=None,
        note="The bulk-data machine AND, measured rather than assumed, a valid "
             "measurement device: a T4 clears CLEAR's 10% bar at 2.3% rel "
             "std dev once the window is sized to its ~100 ms sensor period. "
             "The 70 W cap seems to help -- a pinned card does not oscillate. "
             "~30 h/week dwarfs every other budget here. Pass --energy.",
    ),
    "colab-t4": Profile(
        name="colab-t4",
        config="qwen2.5-3b",
        quant="nf4",
        dataset="gsm8k",
        limit=128,
        max_new_tokens=256,
        repeats=1,
        strategies=None,
        note="Overflow capacity beside Kaggle. 12 h ceiling, unpublished idle "
             "timeout, GPU type not guaranteed -- always --resume.",
    ),
    "kaggle-p100": Profile(
        name="kaggle-p100",
        config="qwen2.5-3b",
        quant="nf4",
        dataset="gsm8k",
        limit=None,
        max_new_tokens=256,
        repeats=1,
        strategies=None,
        note="Faster than the T4 (732 GB/s) and fine for acceptance and "
             "latency, but CC 6.0 has NO energy counter. Never use --energy "
             "here; the numbers would silently be absent.",
    ),
    "rtxpro-48": Profile(
        name="rtxpro-48",
        config="qwen2.5-3b",
        quant="float16",
        dataset="gsm8k",
        limit=None,             # full 256 fits one 2 h slot at ~81 min
        max_new_tokens=256,
        repeats=1,
        strategies=None,
        note="Dedicated, 300 W, clock-lockable, and the only device here that "
             "holds 7B+ in fp16. Since Kaggle now carries the main energy "
             "work, spend these 6 h on what Kaggle cannot do: larger targets, "
             "and an independent cross-check of a few components. Needs "
             "torch >= 2.7 / cu128 for sm_120, and pre-cached weights.",
    ),
}


# --------------------------------------------------------------------------
# draft-length strategies
# --------------------------------------------------------------------------


@dataclass
class DraftStrategy:
    """How many tokens to draft in the next round.

    COGA-x  constant draft length x.
    DYGA-x  adaptive, starting at x: +2 after a fully accepted draft,
            -1 otherwise (Dutta et al. Sec. 2).

    Only COGA yields points on an energy-vs-K curve. K is the independent
    variable there, and DYGA varies it *within* a single run -- its mean draft
    length is an outcome, not a setting, so a DYGA run is one scatter point
    against the curve, never a point on it. Hence the default sweep is COGA
    only and DYGA is opt-in as a reference marker.
    """

    name: str
    kind: Literal["coga", "dyga"]
    initial_k: int
    increment: int = 2
    decrement: int = 1
    min_k: int = 1
    max_k: int = 64
    _k: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._k = self.initial_k

    def reset(self) -> None:
        self._k = self.initial_k

    def next_k(self) -> int:
        return self._k

    def update(self, proposed: int, accepted: int) -> None:
        if self.kind != "dyga":
            return
        if accepted >= proposed:
            self._k = min(self.max_k, self._k + self.increment)
        else:
            self._k = max(self.min_k, self._k - self.decrement)

    @property
    def sweepable_k(self) -> int | None:
        """The K this run holds fixed, or None if K varied within the run."""
        return self.initial_k if self.kind == "coga" else None


def parse_strategy(spec: str) -> DraftStrategy:
    """``"coga-5"`` / ``"dyga-20"`` / plain ``"4"`` (treated as COGA-4)."""
    spec = spec.strip().lower()
    if spec.isdigit():
        return DraftStrategy(name=f"coga-{spec}", kind="coga", initial_k=int(spec))
    if "-" not in spec:
        raise ValueError(f"bad strategy {spec!r}; expected e.g. 'coga-5' or 'dyga-20'")
    kind, _, num = spec.partition("-")
    if kind not in ("coga", "dyga"):
        raise ValueError(f"unknown strategy kind {kind!r}; expected 'coga' or 'dyga'")
    if not num.isdigit():
        raise ValueError(f"bad draft length in {spec!r}")
    return DraftStrategy(name=spec, kind=kind, initial_k=int(num))


#: Our sweep: constant draft lengths 1..8, giving eight points on E(K).
#: The paper's own grid was COGA-{5,10,20}, too coarse to locate a minimum.
DEFAULT_SWEEP = [f"coga-{k}" for k in range(1, 9)]

#: What the paper ran, by family -- for reproduction, not for the curve.
PAPER_STRATEGIES: dict[str, list[str]] = {
    "vicuna": ["coga-5", "coga-10", "coga-20", "dyga-20"],
    "llama": ["coga-5", "coga-10", "coga-20", "dyga-20"],
    "flan-t5": ["coga-5", "coga-10", "coga-20", "dyga-20"],
    "qwen3": ["dyga-20"],
    "qwen2.5": ["dyga-20"],
}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def feasibility_table(vram_gb: float = 15.0, seq_len: int = 2048) -> str:
    lines = [
        f"Model pairs on {vram_gb:.1f} GB (seq_len={seq_len}, batch 1)",
        "",
        f"  {'config':<15}{'target <- draft':<34}{'NF4':>7}{'fp16':>8}  verdict",
        "  " + "-" * 78,
    ]

    def row(pair: ModelPair) -> str:
        nf4 = pair.estimated_vram_gb("nf4", "float16", seq_len)
        fp16 = pair.estimated_vram_gb("float16", "float16", seq_len)
        if not pair.supported:
            verdict = "unsupported (encoder-decoder)"
        elif fp16 <= vram_gb:
            verdict = "fp16 or NF4"
        elif nf4 <= vram_gb:
            verdict = "NF4 only"
        else:
            verdict = "does not fit"
        mark = "" if pair.exact else " ~"
        return (f"  {pair.key:<15}{pair.label:<34}"
                f"{nf4:>6.1f}G{fp16:>7.1f}G  {verdict}{mark}")

    lines.append("  OURS")
    lines += [row(MODEL_PAIRS[k]) for k in PROJECT_PAIRS]
    lines += ["", "  PAPER ONLY (not run; listed for provenance)"]
    lines += [
        row(p) for p in MODEL_PAIRS.values()
        if p.source == "paper" and p.key not in PROJECT_PAIRS
    ]
    lines += [
        "",
        "  Qwen figures are exact, from each model's config.json (see",
        "  specdec/memory.py); rows marked ~ are parameter-count estimates.",
        "  Qwen2.5 has no 8B and Qwen3 has no 3B, and a draft must share its",
        "  target's tokenizer -- so 0.5B pairs with 3B/7B, 0.6B pairs with 4B/8B.",
        "  NF4 needs Pascal or newer, so a V100 (CC 7.0) is fine, but its compute",
        "  dtype must be float16: Volta has no native bf16.",
    ]
    return "\n".join(lines)


def protocol_summary() -> str:
    lines = ["Protocol held at Dutta et al. (arXiv:2602.09113) values:", ""]
    for key, value in DUTTA_PROTOCOL.items():
        lines.append(f"  {key:<24}{value}")
    lines += [
        "",
        f"  their hardware          {PAPER_HARDWARE}",
        f"  their software          {PAPER_SOFTWARE}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    vram = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    print(feasibility_table(vram))
    print()
    print(protocol_summary())
