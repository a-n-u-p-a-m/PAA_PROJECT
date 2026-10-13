"""Correctness tests for the draft-verify-rollback loop.

The defining property of greedy speculative decoding is that it is *lossless*:
it must emit exactly the token sequence the target model would have produced
on its own. Any bug in the logit indexing or the KV-cache rollback breaks that
equality, so these tests are the gate the pipeline has to pass before any
energy number from it means anything.

Runs on CPU with small randomly-initialised models -- no downloads, no GPU.

    python -m pytest tests/ -q     (or just: python tests/test_lossless.py)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specdec.configs import parse_strategy  # noqa: E402
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402


def _tiny_model(seed: int, vocab: int = 128, layers: int = 2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _perturbed(model, scale: float = 0.05, seed: int = 99):
    """A near-copy of ``model`` -- stands in for a well-aligned draft model,
    so the accept path and the mid-sequence reject path both get exercised."""
    torch.manual_seed(seed)
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for p in clone.parameters():
            p.add_(torch.randn_like(p) * scale * p.std().clamp(min=1e-6))
    return clone.eval()


def _prompt(vocab: int = 128, length: int = 8, seed: int = 7):
    torch.manual_seed(seed)
    return torch.randint(0, vocab, (1, length))


def test_greedy_is_lossless():
    """Spec decoding must reproduce plain greedy decoding, token for token."""
    target = _tiny_model(seed=0)
    prompt = _prompt()
    max_new = 24

    reference = autoregressive_generate(target, prompt, max_new_tokens=max_new)
    ref_ids = reference.sequences[0].tolist()

    drafts = {
        "identical": target,
        "near-copy": _perturbed(target, scale=0.05),
        "far-copy": _perturbed(target, scale=0.8, seed=5),
        "independent": _tiny_model(seed=1),
    }

    for name, draft in drafts.items():
        for k in (1, 2, 3, 4, 5, 8):
            out = speculative_generate(
                draft, target, prompt, k=k, max_new_tokens=max_new
            )
            got = out.sequences[0].tolist()
            assert got == ref_ids, (
                f"draft={name} k={k}: output diverged from greedy\n"
                f"  expected {ref_ids}\n  got      {got}"
            )
            assert out.new_tokens == max_new, (
                f"draft={name} k={k}: emitted {out.new_tokens}, want {max_new}"
            )
            print(
                f"  ok  draft={name:<12} k={k}  "
                f"accept={out.acceptance_rate:5.1%}  "
                f"tokens/round={out.mean_accepted_per_round:4.2f}  "
                f"rounds={len(out.rounds)}"
            )


def test_round_emits_between_one_and_k_plus_one():
    """A round always emits >= 1 token (so the loop cannot stall) and
    <= K+1 (the bonus-token bound)."""
    target = _tiny_model(seed=0)
    draft = _perturbed(target, scale=0.05)
    prompt = _prompt()
    for k in (1, 4, 8):
        out = speculative_generate(draft, target, prompt, k=k, max_new_tokens=32)
        for r in out.rounds:
            assert 1 <= r.emitted <= k + 1, f"k={k}: round emitted {r.emitted}"
            assert 0 <= r.accepted <= k, f"k={k}: round accepted {r.accepted}"
        print(f"  ok  k={k}: all {len(out.rounds)} rounds within [1, {k + 1}]")


def test_identical_models_accept_everything():
    """When draft == target, greedy verification must accept all K drafts,
    so every round emits exactly K+1 tokens. Catches off-by-one errors in the
    logit row offset that a weaker draft would hide."""
    target = _tiny_model(seed=0)
    prompt = _prompt()
    for k in (1, 3, 6):
        out = speculative_generate(target, target, prompt, k=k, max_new_tokens=5 * (k + 1))
        assert out.acceptance_rate == 1.0, (
            f"k={k}: self-draft accepted only {out.acceptance_rate:.1%}"
        )
        assert all(r.emitted == k + 1 for r in out.rounds)
        print(f"  ok  k={k}: self-draft accepted 100%, {k + 1} tokens/round")


def test_eos_stops_generation():
    target = _tiny_model(seed=0)
    draft = _perturbed(target, scale=0.05)
    prompt = _prompt()
    ref = autoregressive_generate(target, prompt, max_new_tokens=40)
    eos = int(ref.sequences[0, ref.prompt_len + 6].item())

    ref_eos = autoregressive_generate(
        target, prompt, max_new_tokens=40, eos_token_id=eos
    )
    out = speculative_generate(
        draft, target, prompt, k=4, max_new_tokens=40, eos_token_id=eos
    )
    assert out.sequences[0].tolist() == ref_eos.sequences[0].tolist(), (
        "EOS truncation differs between speculative and autoregressive paths"
    )
    assert int(out.sequences[0, -1].item()) == eos
    print(f"  ok  stopped at eos={eos} after {out.new_tokens} tokens")


def test_dyga_is_lossless_too():
    """DYGA varies the draft length between rounds. Losslessness must hold
    for any schedule -- if it only held for constant K, the rollback would be
    depending on K not changing, which is exactly the bug to rule out."""
    target = _tiny_model(seed=0)
    draft = _perturbed(target, scale=0.05)
    prompt = _prompt()
    max_new = 24
    ref = autoregressive_generate(target, prompt, max_new_tokens=max_new)

    for spec in ("dyga-1", "dyga-4", "dyga-20"):
        out = speculative_generate(draft, target, prompt, k=spec, max_new_tokens=max_new)
        assert out.sequences[0].tolist() == ref.sequences[0].tolist(), (
            f"{spec}: output diverged from greedy"
        )
        ks = [r.proposed for r in out.rounds]
        print(f"  ok  {spec:<9} K schedule {ks[:8]}{'...' if len(ks) > 8 else ''}  "
              f"mean K={out.mean_draft_length:.2f}")


def test_dyga_update_rule_matches_paper():
    """Dutta et al. Sec. 2: +2 after a fully accepted draft, -1 otherwise,
    floored at 1."""
    s = parse_strategy("dyga-20")
    assert s.next_k() == 20
    s.update(proposed=20, accepted=20)
    assert s.next_k() == 22, "full acceptance should add 2"
    s.update(proposed=22, accepted=5)
    assert s.next_k() == 21, "partial acceptance should subtract 1"
    s.update(proposed=21, accepted=0)
    assert s.next_k() == 20, "zero acceptance should subtract 1"

    floor = parse_strategy("dyga-1")
    for _ in range(5):
        floor.update(proposed=1, accepted=0)
    assert floor.next_k() == 1, "draft length must not fall below 1"

    const = parse_strategy("coga-5")
    for acc in (5, 0, 3):
        const.update(proposed=5, accepted=acc)
    assert const.next_k() == 5, "COGA must stay constant"
    print("  ok  +2 on full accept, -1 otherwise, floored at 1; COGA constant")


def test_sampling_matches_target_distribution():
    """Modified rejection sampling must target the *target* model's
    distribution. Checks the empirical first-token distribution of
    speculative sampling against sampling from the target alone."""
    target = _tiny_model(seed=0, vocab=32)
    draft = _perturbed(target, scale=0.4, seed=3)
    prompt = _prompt(vocab=32)
    trials = 1200

    def histogram(fn):
        counts = torch.zeros(32)
        for t in range(trials):
            out = fn(seed=t)
            counts[int(out.sequences[0, out.prompt_len].item())] += 1
        return counts / counts.sum()

    spec = histogram(
        lambda seed: speculative_generate(
            draft, target, prompt, k=4, max_new_tokens=1,
            do_sample=True, temperature=1.0, seed=seed,
        )
    )
    base = histogram(
        lambda seed: autoregressive_generate(
            target, prompt, max_new_tokens=1,
            do_sample=True, temperature=1.0, seed=seed,
        )
    )

    tv = 0.5 * float((spec - base).abs().sum())
    # sampling noise at n=1200 per arm gives a TV floor around 0.02-0.03
    assert tv < 0.06, f"total-variation distance {tv:.3f} is too large"
    print(f"  ok  TV(speculative, target) = {tv:.3f} over {trials} draws")


if __name__ == "__main__":
    tests = [
        ("greedy losslessness", test_greedy_is_lossless),
        ("emitted-token bounds", test_round_emits_between_one_and_k_plus_one),
        ("self-draft accepts all", test_identical_models_accept_everything),
        ("eos handling", test_eos_stops_generation),
        ("dyga losslessness", test_dyga_is_lossless_too),
        ("dyga update rule", test_dyga_update_rule_matches_paper),
        ("sampling distribution", test_sampling_matches_target_distribution),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {exc}")
    print("\n" + ("all tests passed" if not failed else f"{failed} test(s) failed"))
    sys.exit(1 if failed else 0)
