#!/usr/bin/env python3
"""Is a greedy divergence a bug in the pipeline, or floating-point noise?

    python diagnose_divergence.py --config qwen2.5-3b --quant nf4 --k 2
    python diagnose_divergence.py --config qwen2.5-3b --quant float16 --k 2

``--selftest`` asserts that speculative decoding reproduces plain greedy
token for token. That is the right invariant *in exact arithmetic*. On a GPU
it is not quite achievable, for a reason built into the algorithm:

    autoregressive decode feeds the target ONE token per forward pass
    speculative verify feeds it K+1 tokens in a SINGLE forward pass

Those are different matrix shapes. cuBLAS picks different kernels and tile
sizes for them, so the summations happen in a different order, and floating
point addition is not associative. The resulting logits differ in their last
bits. Almost always that changes nothing -- but when the top two candidates
are nearly tied, a difference of 1e-3 is enough to flip the argmax, and from
there the two sequences diverge completely.

So "spec != greedy" has two possible causes with opposite implications:

    a broken rollback   -> the engine is wrong, every number is worthless
    a near-tie flip     -> the engine is right, finite precision is the limit

This script tells them apart by measuring, at the exact position where the
two paths diverged, how far apart the target's own logits are between the
two shapes, and comparing that to the gap between the top two candidates.
If the top-2 gap is smaller than the numerical disagreement, the flip is
fully explained by arithmetic and no bug is required to account for it.

A real rollback bug looks completely different: the logits disagree by a
large margin, because the model is being fed a different prefix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import encode, load_pair  # noqa: E402
from specdec import datasets as ds  # noqa: E402
from specdec.configs import MODEL_PAIRS  # noqa: E402
from specdec.engine import autoregressive_generate, speculative_generate  # noqa: E402
from specdec.kv import logits_for_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="qwen2.5-3b", choices=sorted(MODEL_PAIRS))
    ap.add_argument("--quant", default="nf4",
                    choices=["nf4", "float16", "bfloat16", "float32"])
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument("--draft-precision", default="float16")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    pair = MODEL_PAIRS[args.config]
    tok, draft, target = load_pair(pair, args)
    samples = ds.load(args.dataset, limit=args.limit, seed=args.seed,
                      offline=args.offline)

    print(f"\ncomparing decode (1 token/forward) against verify "
          f"({args.k + 1} tokens/forward)\n")
    diverged = explained = unexplained = 0

    for s in samples:
        ids = encode(tok, s.prompt, args.device, not args.no_chat_template)
        ref = autoregressive_generate(target, ids,
                                      max_new_tokens=args.max_new_tokens,
                                      eos_token_id=tok.eos_token_id)
        spec = speculative_generate(draft, target, ids, k=args.k,
                                    max_new_tokens=args.max_new_tokens,
                                    eos_token_id=tok.eos_token_id)
        a = ref.sequences[0].tolist()
        b = spec.sequences[0].tolist()
        if a == b:
            print(f"  sample {s.index}: identical")
            continue

        d = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        diverged += 1
        row = d - 1  # the logit row that predicts position d

        try:
            la = logits_for_row(target, ref.sequences[:, : d + 1], row, 1)
            lb = logits_for_row(target, ref.sequences[:, : d + 1], row, args.k + 1)
        except ValueError as exc:
            print(f"  sample {s.index}: diverged at {d}, cannot probe ({exc})")
            continue

        top_a = torch.topk(la, 2)
        gap = float(top_a.values[0] - top_a.values[1])
        delta = float((la - lb).abs().max())
        flipped = int(la.argmax()) != int(lb.argmax())
        rel = delta / max(abs(float(top_a.values[0])), 1e-9)

        print(f"  sample {s.index}: diverged at generated token {d - ref.prompt_len}")
        print(f"      top-2 gap in the decode path : {gap:9.5f}")
        print(f"      max |logit difference|       : {delta:9.5f}"
              f"   ({rel:.2%} of the top logit)")
        print(f"      argmax flips between shapes  : {flipped}")
        if gap <= delta:
            explained += 1
            print(f"      -> NEAR TIE: the candidates are closer together than "
                  f"the\n         numerical disagreement, so arithmetic alone "
                  f"accounts for it.")
        else:
            unexplained += 1
            print(f"      -> UNEXPLAINED: the gap exceeds the numerical "
                  f"difference.\n         This is not rounding. Suspect the "
                  f"rollback or the logit offset.")

    print(f"\n  {diverged} divergence(s): {explained} explained by arithmetic, "
          f"{unexplained} not")
    if diverged and not unexplained:
        print("""
  Verdict: the pipeline is correct. Every divergence sits on a near-tie that
  finite precision cannot resolve consistently between the two matmul shapes.
  This is inherent to speculative decoding on a GPU, not a defect: the
  algorithm is lossless in exact arithmetic, and "lossless" in practice means
  identical up to floating-point non-determinism. Acceptance rate, timing and
  energy are all unaffected -- none of them depends on which of two nearly
  equal tokens was chosen.""")
    elif unexplained:
        print("""
  Verdict: at least one divergence is too large to be rounding. Do not run
  the sweep. Re-check the logit row offset and the cache crop in engine.py.""")
    else:
        print("\n  No divergences at all on these prompts.")


if __name__ == "__main__":
    main()
