"""Baseline speculative decoding: draft, verify, accept/reject, rollback.

Reference implementation of the two-model scheme from Leviathan et al. (ICML
2023) and Chen et al. (arXiv 2023), written for *measurement* rather than for
speed records: every phase boundary is instrumented, and the draft and verify
model calls are kept structurally separate so CLEAR's activation hooks can be
attached to each independently in a later phase of the project.

Index bookkeeping, which is where this algorithm is usually got wrong:

    let n = len(prefix) at the start of a round, and d_1..d_K the drafts.
    The target runs one forward pass over prefix + drafts (length n+K).
    Logit row at absolute position p predicts the token at position p+1, so
    rows n-1 .. n+K-1 -- that is K+1 rows -- are the ones we need:
        row n-1+i  judges d_i          (i = 1..K)
        row n+K-1  supplies the bonus token if all K are accepted
    Hence a round emits between 1 and K+1 tokens, and never zero.

After accepting m of K drafts, both caches are cropped to n+m: entries at
positions >= n+m were computed from a continuation that was just thrown away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .configs import DraftStrategy, parse_strategy
from .kv import cache_length, crop_cache, forward_with_cache, new_cache
from .meters import PhaseMeter


@dataclass
class RoundRecord:
    """One draft-verify round."""

    index: int
    prefix_len: int
    proposed: int
    accepted: int
    emitted: int
    draft_ms: float
    verify_ms: float


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    prompt_len: int
    new_tokens: int
    rounds: list[RoundRecord] = field(default_factory=list)
    wall_ms: float = 0.0
    meter: Optional[dict] = None
    stopped_on_eos: bool = False
    strategy: str = ""

    @property
    def mean_draft_length(self) -> float:
        """Mean K actually used. Constant for COGA; the quantity of interest
        for DYGA, whose whole premise is that it finds a good K by itself."""
        return (
            sum(r.proposed for r in self.rounds) / len(self.rounds)
            if self.rounds
            else 0.0
        )

    @property
    def proposed_total(self) -> int:
        return sum(r.proposed for r in self.rounds)

    @property
    def accepted_total(self) -> int:
        return sum(r.accepted for r in self.rounds)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target kept. The headline quality
        number: it sets how much work the draft model did for nothing."""
        total = self.proposed_total
        return self.accepted_total / total if total else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        """Mean tokens emitted per verify pass (the 'block efficiency', in
        [1, K+1]). This, not acceptance rate, is what drives speedup."""
        return self.new_tokens / len(self.rounds) if self.rounds else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / (self.wall_ms / 1000.0) if self.wall_ms else 0.0


# --------------------------------------------------------------------------
# logit processing
# --------------------------------------------------------------------------


def _process_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Map raw logits to a probability distribution.

    The draft and target must be filtered *identically*, otherwise the
    rejection-sampling correction below no longer targets the target model's
    distribution and the scheme stops being lossless.
    """
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        # drop a token once the mass *strictly before* it already covers top_p,
        # which always keeps at least the argmax
        remove = (probs.cumsum(dim=-1) - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    return logits.softmax(dim=-1)


def _sample(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())


# --------------------------------------------------------------------------
# baseline: plain autoregressive decoding
# --------------------------------------------------------------------------


@torch.no_grad()
def autoregressive_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    meter: Optional[PhaseMeter] = None,
    seed: Optional[int] = None,
) -> GenerationResult:
    """Standard one-token-at-a-time decoding. The control condition.

    Shares cache handling with the speculative path so latency and energy
    comparisons are not contaminated by implementation differences.
    """
    if seed is not None:
        torch.manual_seed(seed)
    meter = meter or PhaseMeter()
    device = next(model.parameters()).device
    full_ids = input_ids.to(device)
    prompt_len = full_ids.shape[1]

    cache = new_cache()
    cached = 0
    rounds: list[RoundRecord] = []
    stopped = False

    t_start = time.perf_counter()
    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        with meter.phase("autoregressive"):
            logits, cache, cached, _ = forward_with_cache(
                model, full_ids, cache, cached, logits_to_keep=1
            )
        last = logits[:, -1, :]
        if do_sample:
            tok = _sample(_process_logits(last, temperature, top_k, top_p)[0])
        else:
            tok = int(last.argmax(dim=-1).item())
        dt = (time.perf_counter() - t0) * 1000.0

        full_ids = torch.cat(
            [full_ids, torch.tensor([[tok]], device=device, dtype=full_ids.dtype)],
            dim=1,
        )
        rounds.append(
            RoundRecord(
                index=step,
                prefix_len=full_ids.shape[1] - 1,
                proposed=0,
                accepted=0,
                emitted=1,
                draft_ms=0.0,
                verify_ms=dt,
            )
        )
        if eos_token_id is not None and tok == eos_token_id:
            stopped = True
            break

    wall = (time.perf_counter() - t_start) * 1000.0
    return GenerationResult(
        sequences=full_ids,
        prompt_len=prompt_len,
        new_tokens=full_ids.shape[1] - prompt_len,
        rounds=rounds,
        wall_ms=wall,
        meter=meter.summary(),
        stopped_on_eos=stopped,
        strategy="autoregressive",
    )


# --------------------------------------------------------------------------
# speculative decoding
# --------------------------------------------------------------------------


@torch.no_grad()
def speculative_generate(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,
    k: int | str | DraftStrategy = 4,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    meter: Optional[PhaseMeter] = None,
    seed: Optional[int] = None,
) -> GenerationResult:
    """Draft K tokens with the small model, verify all K in one target pass.

    ``k`` is an int (constant draft length, i.e. COGA-k), a strategy string
    such as ``"coga-5"`` or ``"dyga-20"``, or a ``DraftStrategy``. DYGA varies
    the draft length between rounds, so ``K`` below is per-round.

    With ``do_sample=False`` the output is token-identical to
    ``autoregressive_generate`` on the target model, for *any* draft-length
    schedule -- that equality is the correctness test, and
    ``tests/test_lossless.py`` asserts it for both strategies.

    With ``do_sample=True`` the modified rejection sampling of Leviathan et
    al. makes the output distribution identical to sampling from the target
    alone; equality then holds in distribution, not per-token.
    """
    if isinstance(k, DraftStrategy):
        strategy = k
        strategy.reset()
    elif isinstance(k, str):
        strategy = parse_strategy(k)
    else:
        if k < 1:
            raise ValueError("k must be >= 1; use autoregressive_generate for k=0")
        strategy = parse_strategy(str(int(k)))

    if seed is not None:
        torch.manual_seed(seed)

    meter = meter or PhaseMeter()
    device = next(target_model.parameters()).device
    full_ids = input_ids.to(device)
    prompt_len = full_ids.shape[1]

    draft_cache, target_cache = new_cache(), new_cache()
    draft_cached = target_cached = 0

    rounds: list[RoundRecord] = []
    stopped = False

    t_start = time.perf_counter()
    round_idx = 0

    while full_ids.shape[1] - prompt_len < max_new_tokens:
        n = full_ids.shape[1]
        budget = max_new_tokens - (n - prompt_len)
        # DYGA re-chooses this every round; COGA keeps it fixed
        k = strategy.next_k()

        # ---------------- draft phase: K sequential forward passes ---------
        t0 = time.perf_counter()
        drafts: list[int] = []
        draft_probs: list[torch.Tensor] = []
        speculative_ids = full_ids

        with meter.phase("draft"):
            for _ in range(k):
                logits, draft_cache, draft_cached, _ = forward_with_cache(
                    draft_model, speculative_ids, draft_cache, draft_cached,
                    logits_to_keep=1,
                )
                last = logits[:, -1, :]
                if do_sample:
                    q = _process_logits(last, temperature, top_k, top_p)[0]
                    tok = _sample(q)
                    draft_probs.append(q)
                else:
                    tok = int(last.argmax(dim=-1).item())
                drafts.append(tok)
                speculative_ids = torch.cat(
                    [
                        speculative_ids,
                        torch.tensor(
                            [[tok]], device=device, dtype=full_ids.dtype
                        ),
                    ],
                    dim=1,
                )
        draft_ms = (time.perf_counter() - t0) * 1000.0
        # draft cache now holds n+K-1 positions: d_K was produced but never fed back in.

        # ---------------- verify phase: ONE target forward pass ------------
        t0 = time.perf_counter()
        with meter.phase("verify"):
            # only the last K+1 rows are ever read; see kv.forward_with_cache
            tlogits, target_cache, target_cached, first_pos = forward_with_cache(
                target_model, speculative_ids, target_cache, target_cached,
                logits_to_keep=k + 1,
            )

        # rows for absolute positions n-1 .. n+K-1 inclusive
        offset = (n - 1) - first_pos
        if offset < 0:
            raise RuntimeError(
                f"target logits start at position {first_pos}, past the "
                f"accepted prefix end {n - 1}; rollback is broken"
            )
        judge = tlogits[0, offset:, :]
        assert judge.shape[0] == k + 1, (
            f"expected {k + 1} judging rows, got {judge.shape[0]}"
        )

        # ---------------- accept / reject ----------------------------------
        if do_sample:
            accepted, extra = _reject_sample(
                judge, drafts, draft_probs, temperature, top_k, top_p
            )
        else:
            accepted, extra = _reject_greedy(judge, drafts)
        verify_ms = (time.perf_counter() - t0) * 1000.0

        emitted = drafts[:accepted] + [extra]
        if len(emitted) > budget:
            emitted = emitted[:budget]

        hit_eos = eos_token_id is not None and eos_token_id in emitted
        if hit_eos:
            emitted = emitted[: emitted.index(eos_token_id) + 1]
            stopped = True

        full_ids = torch.cat(
            [
                full_ids,
                torch.tensor([emitted], device=device, dtype=full_ids.dtype),
            ],
            dim=1,
        )

        # ---------------- rollback -----------------------------------------
        # Positions >= n + accepted were computed from a continuation we just
        # discarded; everything before is still exactly right.
        keep = n + min(accepted, len(emitted) - 1 if emitted else 0)
        target_cache = crop_cache(target_cache, keep)
        draft_cache = crop_cache(draft_cache, keep)
        target_cached = min(target_cached, cache_length(target_cache))
        draft_cached = min(draft_cached, cache_length(draft_cache))

        rounds.append(
            RoundRecord(
                index=round_idx,
                prefix_len=n,
                proposed=k,
                accepted=accepted,
                emitted=len(emitted),
                draft_ms=draft_ms,
                verify_ms=verify_ms,
            )
        )
        strategy.update(proposed=k, accepted=accepted)
        round_idx += 1
        if stopped:
            break

    wall = (time.perf_counter() - t_start) * 1000.0
    return GenerationResult(
        sequences=full_ids,
        prompt_len=prompt_len,
        new_tokens=full_ids.shape[1] - prompt_len,
        rounds=rounds,
        wall_ms=wall,
        meter=meter.summary(),
        stopped_on_eos=stopped,
        strategy=strategy.name,
    )


def _reject_greedy(judge: torch.Tensor, drafts: list[int]) -> tuple[int, int]:
    """Accept the longest prefix of drafts the target would itself have picked.

    Returns ``(n_accepted, next_token)``. ``next_token`` is the target's own
    choice at the first divergence, or the bonus token when all K match --
    which is why a round can never emit zero tokens, and why speculative
    decoding cannot be slower than autoregressive by more than the wasted
    draft work.
    """
    target_choice = judge.argmax(dim=-1)
    accepted = 0
    for i, d in enumerate(drafts):
        if int(target_choice[i].item()) != d:
            break
        accepted += 1
    return accepted, int(target_choice[accepted].item())


def _reject_sample(
    judge: torch.Tensor,
    drafts: list[int],
    draft_probs: list[torch.Tensor],
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[int, int]:
    """Modified rejection sampling (Leviathan et al. Alg. 1, Chen et al. §2).

    Accept d_i with probability min(1, p(d_i)/q(d_i)). On rejection, resample
    from the normalised residual max(0, p - q), which is what makes the
    overall output distribution exactly the target's.
    """
    accepted = 0
    for i, d in enumerate(drafts):
        p = _process_logits(judge[i : i + 1], temperature, top_k, top_p)[0]
        q = draft_probs[i]
        p_d, q_d = float(p[d]), float(q[d])
        if q_d <= 0.0 or torch.rand(1).item() < min(1.0, p_d / q_d):
            accepted += 1
            continue
        residual = torch.clamp(p - q, min=0.0)
        total = float(residual.sum())
        if total <= 0.0:
            return accepted, _sample(p)
        return accepted, _sample(residual / total)

    bonus = _process_logits(judge[len(drafts) : len(drafts) + 1], temperature, top_k, top_p)[0]
    return accepted, _sample(bonus)
