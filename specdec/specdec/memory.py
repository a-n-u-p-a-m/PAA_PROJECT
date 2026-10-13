"""Exact VRAM accounting, from real architecture configs.

Parameter-count heuristics get this wrong in two ways that matter at our
model sizes, both because of Qwen's 151,936-token vocabulary:

1. **bitsandbytes quantizes ``nn.Linear`` only.** Embeddings stay in fp16,
   and ``lm_head`` is in the default skip list. For Qwen2.5-0.5B the
   embedding table is 28% of all parameters, so "NF4 = 0.5 bytes/param"
   understates the real footprint badly. With tied embeddings there is one
   such table; Qwen3-8B and Qwen2.5-7B untie them and pay for two.

2. **KV cache is far smaller than rules of thumb suggest**, because these
   models use aggressive GQA. Qwen2.5-3B has 2 KV heads across 36 layers:
   36 KiB per token, so a 2048-token context costs 75 MB, not gigabytes.

The term that actually bites instead is the **logits tensor**. A forward pass
over P positions produces [1, P, 151936], which HF upcasts to fp32 -- 0.58 MiB
*per position*. A 1000-token CNN-DM prefill therefore costs ~580 MB in logits
alone, more than the KV cache by a factor of eight. ``logits_to_keep`` fixes
this and the engine uses it; ``logits_full`` below shows what it saves.

Figures are GiB (1024^3). Run ``python -m specdec.memory`` for the table.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3

#: NF4 with double quantization: 4 bits per weight, plus an 8-bit absmax per
#: block of 64, plus an fp32 scale per 256 absmax values.
NF4_BITS_PER_PARAM = 4 + 8 / 64 + 32 / (64 * 256)
NF4_BYTES_PER_PARAM = NF4_BITS_PER_PARAM / 8  # ~0.5159

BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "nf4": NF4_BYTES_PER_PARAM}


@dataclass(frozen=True)
class ArchSpec:
    """Everything needed to count parameters and cache bytes exactly."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tie_embeddings: bool
    attn_bias: bool = False   # Qwen2.5 has q/k/v biases; Qwen3 does not
    qk_norm: bool = False     # Qwen3 adds per-head RMSNorm on q and k

    # -- parameter counts ---------------------------------------------------

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def lm_head_params(self) -> int:
        """Zero when tied: the embedding table is reused as the output head."""
        return 0 if self.tie_embeddings else self.vocab * self.hidden

    @property
    def attn_linear_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        k = self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        return q + 2 * k + o

    @property
    def mlp_linear_params(self) -> int:
        return 3 * self.hidden * self.intermediate

    @property
    def per_layer_linear(self) -> int:
        return self.attn_linear_params + self.mlp_linear_params

    @property
    def per_layer_other(self) -> int:
        """Biases and norms -- never quantized, and negligible, but counted."""
        other = 2 * self.hidden  # input_layernorm + post_attention_layernorm
        if self.attn_bias:
            other += (self.heads + 2 * self.kv_heads) * self.head_dim
        if self.qk_norm:
            other += 2 * self.head_dim
        return other

    @property
    def linear_params(self) -> int:
        """The quantizable part: every nn.Linear in the decoder stack."""
        return self.layers * self.per_layer_linear

    @property
    def other_params(self) -> int:
        return self.layers * self.per_layer_other + self.hidden  # + final norm

    @property
    def total_params(self) -> int:
        return (
            self.embedding_params
            + self.lm_head_params
            + self.linear_params
            + self.other_params
        )

    # -- memory -------------------------------------------------------------

    def weights_gib(self, precision: str) -> float:
        """Weight bytes. Under NF4 only the decoder Linears are quantized;
        embeddings and lm_head stay at fp16, as transformers' default skip
        list leaves them alone."""
        if precision == "nf4":
            quantized = self.linear_params * NF4_BYTES_PER_PARAM
            kept = (
                self.embedding_params + self.lm_head_params + self.other_params
            ) * BYTES["float16"]
            return (quantized + kept) / GIB
        return self.total_params * BYTES[precision] / GIB

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, every layer, fp16."""
        return 2 * self.layers * self.kv_heads * self.head_dim * 2

    def kv_gib(self, seq_len: int) -> float:
        return self.kv_bytes_per_token * seq_len / GIB

    def logits_gib(self, positions: int, dtype: str = "float32") -> float:
        """[1, positions, vocab]. HF upcasts to fp32 before the softmax."""
        return positions * self.vocab * BYTES[dtype] / GIB


# ---------------------------------------------------------------------------
# verified against each model's published config.json
# ---------------------------------------------------------------------------

ARCHS: dict[str, ArchSpec] = {
    "Qwen/Qwen2.5-0.5B-Instruct": ArchSpec(
        hidden=896, layers=24, heads=14, kv_heads=2, head_dim=64,
        intermediate=4864, vocab=151936, tie_embeddings=True, attn_bias=True),
    "Qwen/Qwen2.5-1.5B-Instruct": ArchSpec(
        hidden=1536, layers=28, heads=12, kv_heads=2, head_dim=128,
        intermediate=8960, vocab=151936, tie_embeddings=True, attn_bias=True),
    "Qwen/Qwen2.5-3B-Instruct": ArchSpec(
        hidden=2048, layers=36, heads=16, kv_heads=2, head_dim=128,
        intermediate=11008, vocab=151936, tie_embeddings=True, attn_bias=True),
    "Qwen/Qwen2.5-7B-Instruct": ArchSpec(
        hidden=3584, layers=28, heads=28, kv_heads=4, head_dim=128,
        intermediate=18944, vocab=152064, tie_embeddings=False, attn_bias=True),
    "Qwen/Qwen3-0.6B": ArchSpec(
        hidden=1024, layers=28, heads=16, kv_heads=8, head_dim=128,
        intermediate=3072, vocab=151936, tie_embeddings=True, qk_norm=True),
    "Qwen/Qwen3-4B": ArchSpec(
        hidden=2560, layers=36, heads=32, kv_heads=8, head_dim=128,
        intermediate=9728, vocab=151936, tie_embeddings=True, qk_norm=True),
    "Qwen/Qwen3-8B": ArchSpec(
        hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128,
        intermediate=12288, vocab=151936, tie_embeddings=False, qk_norm=True),
    # -- reachable only on a 48 GB card ------------------------------------
    "Qwen/Qwen2.5-14B-Instruct": ArchSpec(
        hidden=5120, layers=48, heads=40, kv_heads=8, head_dim=128,
        intermediate=13824, vocab=152064, tie_embeddings=False, attn_bias=True),
    "Qwen/Qwen2.5-32B-Instruct": ArchSpec(
        hidden=5120, layers=64, heads=40, kv_heads=8, head_dim=128,
        intermediate=27648, vocab=152064, tie_embeddings=False, attn_bias=True),
    "Qwen/Qwen3-14B": ArchSpec(
        hidden=5120, layers=40, heads=40, kv_heads=8, head_dim=128,
        intermediate=17408, vocab=151936, tie_embeddings=False, qk_norm=True),
    "Qwen/Qwen3-32B": ArchSpec(
        hidden=5120, layers=64, heads=64, kv_heads=8, head_dim=128,
        intermediate=25600, vocab=151936, tie_embeddings=False, qk_norm=True),
    # -- Dutta's own non-Qwen pairs, which 48 GB puts in reach -------------
    # Llama configs are behind a gated repo; these are reconstructed and
    # validated against the published parameter counts (8.03B / 1.24B).
    "lmsys/vicuna-7b-v1.3": ArchSpec(
        hidden=4096, layers=32, heads=32, kv_heads=32, head_dim=128,
        intermediate=11008, vocab=32000, tie_embeddings=False),
    "lmsys/vicuna-13b-v1.3": ArchSpec(
        hidden=5120, layers=40, heads=40, kv_heads=40, head_dim=128,
        intermediate=13824, vocab=32000, tie_embeddings=False),
    "double7/vicuna-68m": ArchSpec(
        hidden=768, layers=2, heads=12, kv_heads=12, head_dim=64,
        intermediate=3072, vocab=32000, tie_embeddings=False),
    "meta-llama/Llama-3.1-8B-Instruct": ArchSpec(
        hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128,
        intermediate=14336, vocab=128256, tie_embeddings=False),
    "meta-llama/Llama-3.2-1B-Instruct": ArchSpec(
        hidden=2048, layers=16, heads=32, kv_heads=8, head_dim=64,
        intermediate=8192, vocab=128256, tie_embeddings=True),
}

#: Measured on a V100: CUDA context, cuBLAS/cuDNN workspaces, allocator slack.
RUNTIME_OVERHEAD_GIB = 0.75

#: What a desktop environment holds on the GPU before you start. Subtract this
#: on a machine whose GPU also drives a display -- a laptop or a workstation
#: running a compositor -- or the budget will look better than it is.
DISPLAY_RESERVE_GIB = 0.8


def usable_vram(total_gib: float, drives_display: bool = False) -> float:
    return total_gib - (DISPLAY_RESERVE_GIB if drives_display else 0.0)


def amplification_gib(target_id: str, component: str = "decoder_layer",
                      seq_len: int = 256, batch: int = 1) -> float:
    """VRAM to re-execute ONE component on cached activations (CLEAR phase 4).

    This is the measurement step, and it is tiny: a single decoder layer's
    weights plus one cached activation tensor. The full model only has to be
    resident for phase 3, when activations are captured. So CLEAR's
    amplification loop is not what a small card runs out of memory on --
    which matters, because it means a 6 GB machine is limited by measurement
    *stability*, not capacity.
    """
    a = ARCHS[target_id]
    per_layer = {
        "decoder_layer": a.per_layer_linear,
        "attention": a.attn_linear_params,
        "mlp": a.mlp_linear_params,
    }[component]
    weights = per_layer * BYTES["float16"] / GIB
    # input activation, plus the MLP intermediate as the largest temporary
    act = batch * seq_len * (a.hidden + a.intermediate) * 2 * BYTES["float16"] / GIB
    return weights + act + RUNTIME_OVERHEAD_GIB


@dataclass
class Budget:
    """A full VRAM breakdown for one speculative-decoding configuration."""

    target_weights: float
    draft_weights: float
    kv_cache: float
    logits: float
    activations: float
    overhead: float
    logits_full: float  # what logits would cost without logits_to_keep

    @property
    def total(self) -> float:
        return (
            self.target_weights + self.draft_weights + self.kv_cache
            + self.logits + self.activations + self.overhead
        )

    def rows(self) -> list[tuple[str, float]]:
        return [
            ("target weights", self.target_weights),
            ("draft weights", self.draft_weights),
            ("KV caches (both models)", self.kv_cache),
            ("logits", self.logits),
            ("activations", self.activations),
            ("CUDA context + workspace", self.overhead),
        ]


def budget(
    target_id: str,
    draft_id: str,
    target_precision: str = "nf4",
    draft_precision: str = "float16",
    prompt_len: int = 256,
    max_new_tokens: int = 256,
    k: int = 8,
) -> Budget:
    """VRAM for one draft/verify configuration at peak.

    Peak occurs during the prefill forward over the prompt, where the logits
    tensor is largest; the per-round verify pass over K+1 positions is tiny
    by comparison.
    """
    t, d = ARCHS[target_id], ARCHS[draft_id]
    seq = prompt_len + max_new_tokens + k

    # both caches are live at once, and during drafting the draft cache
    # carries K speculative positions beyond the accepted prefix
    kv = t.kv_gib(seq) + d.kv_gib(seq)

    # with logits_to_keep the target materialises K+1 rows and the draft 1
    logits = t.logits_gib(k + 1) + d.logits_gib(1)
    logits_full = t.logits_gib(seq) + d.logits_gib(seq)

    # transient per-layer activations: the MLP intermediate dominates, and
    # the allocator keeps roughly a couple of layers' worth in flight
    act = 4 * prompt_len * t.intermediate * 2 / GIB + 0.1

    return Budget(
        target_weights=t.weights_gib(target_precision),
        draft_weights=d.weights_gib(draft_precision),
        kv_cache=kv,
        logits=logits,
        activations=act,
        overhead=RUNTIME_OVERHEAD_GIB,
        logits_full=logits_full,
    )


# ---------------------------------------------------------------------------


def _params_table() -> str:
    lines = ["Parameter counts (from each model's published config.json)", "",
             f"  {'model':<28}{'total':>12}{'embed':>12}{'lm_head':>10}"
             f"{'linear':>14}  tied"]
    lines.append("  " + "-" * 80)
    for name, a in ARCHS.items():
        lines.append(
            f"  {name.split('/')[-1]:<28}{a.total_params / 1e9:>11.3f}B"
            f"{a.embedding_params / 1e6:>11.1f}M"
            f"{a.lm_head_params / 1e6:>9.1f}M"
            f"{a.linear_params / 1e9:>13.3f}B"
            f"  {'yes' if a.tie_embeddings else 'no'}"
        )
    return "\n".join(lines)


def report(target_id: str, draft_id: str, prompt_len: int = 256,
           max_new_tokens: int = 256, k: int = 8, vram: float = 15.0) -> str:
    t = ARCHS[target_id]
    lines = [
        f"{target_id}  <-  {draft_id}",
        f"prompt {prompt_len} + {max_new_tokens} new tokens, K={k}",
        "",
        f"  {'component':<30}{'fp16':>10}{'NF4 target':>14}",
        "  " + "-" * 54,
    ]
    fp16 = budget(target_id, draft_id, "float16", "float16",
                  prompt_len, max_new_tokens, k)
    nf4 = budget(target_id, draft_id, "nf4", "float16",
                 prompt_len, max_new_tokens, k)
    for (label, a), (_, b) in zip(fp16.rows(), nf4.rows()):
        lines.append(f"  {label:<30}{a:>9.2f}G{b:>13.2f}G")
    lines.append("  " + "-" * 54)
    lines.append(f"  {'TOTAL':<30}{fp16.total:>9.2f}G{nf4.total:>13.2f}G")
    lines.append("")
    for label, b in (("fp16", fp16), ("NF4", nf4)):
        head = max(0.0, vram - b.total)
        verdict = "fits" if b.total <= vram else "DOES NOT FIT"
        lines.append(f"  {label:<5}{verdict:<14}{head:.1f} G free on a {vram:.0f} G card")
    lines += [
        "",
        f"  KV cache is {t.kv_bytes_per_token / 1024:.0f} KiB/token "
        f"({t.kv_heads} KV heads x {t.layers} layers), so context length is",
        "  nearly free here. Without logits_to_keep the prefill logits would",
        f"  cost {fp16.logits_full:.2f} G instead of {fp16.logits:.2f} G -- the engine passes it.",
    ]
    return "\n".join(lines)


PAIRS: list[tuple[str, str, str]] = [
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3-4b", "Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"),
    ("qwen3-8b", "Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B"),
]


def fit_report(vram: float, drives_display: bool = False,
               prompt_len: int = 256, max_new_tokens: int = 256,
               k: int = 8) -> str:
    """What runs on a card of this size, and which phases it unlocks."""
    usable = usable_vram(vram, drives_display)
    lines = [
        f"Fitting {vram:.1f} GB"
        + (f" minus {DISPLAY_RESERVE_GIB:.1f} G for the display = "
           f"{usable:.1f} G usable" if drives_display else " (headless)"),
        f"prompt {prompt_len} + {max_new_tokens} new tokens, K={k}",
        "",
        f"  {'pair':<16}{'fp16':>9}{'NF4':>9}   runs as",
        "  " + "-" * 56,
    ]
    best: tuple[str, str] | None = None
    for key, tgt, drf in PAIRS:
        f = budget(tgt, drf, "float16", "float16", prompt_len, max_new_tokens, k).total
        n = budget(tgt, drf, "nf4", "float16", prompt_len, max_new_tokens, k).total
        if f <= usable:
            how, cand = "fp16 or NF4", "float16"
        elif n <= usable:
            how, cand = "NF4 only", "nf4"
        else:
            how, cand = "does not fit", None
        if cand:
            best = (key, cand)
        lines.append(f"  {key:<16}{f:>8.2f}G{n:>8.2f}G   {how}")

    lines.append("")
    if best:
        lines.append(f"  largest that fits: {best[0]} at {best[1]}")
    else:
        lines.append("  nothing fits; use --synthetic on CPU to develop")

    # CLEAR phase 4 needs one component resident, not the whole model
    amp = amplification_gib("Qwen/Qwen2.5-3B-Instruct", "decoder_layer")
    lines += [
        "",
        f"  CLEAR amplification of one Qwen2.5-3B decoder layer: {amp:.2f} G",
        "  -- so phase 4 is not capacity-bound on a small card. What rules a",
        "  laptop GPU out is measurement stability, not memory: a display",
        "  attached, no clock locking and thermal throttling will miss CLEAR's",
        "  <10% std-dev bar. Run probe_env.py to see the actual number.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vram", nargs="?", type=float, default=15.0,
                    help="GPU memory in GB (default 15, a 16GB V100)")
    ap.add_argument("--display", action="store_true",
                    help="this GPU also drives a monitor; reserve for it")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--detail", action="store_true",
                    help="full per-component breakdown for every pair")
    args = ap.parse_args()

    print(_params_table())
    print()
    print(fit_report(args.vram, args.display, args.prompt_len,
                     args.max_new_tokens, args.k))
    if args.detail:
        print()
        for _, tgt, drf in PAIRS:
            print(report(tgt, drf, args.prompt_len, args.max_new_tokens,
                         args.k, usable_vram(args.vram, args.display)))
            print()
