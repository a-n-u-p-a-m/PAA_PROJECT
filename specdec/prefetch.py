#!/usr/bin/env python3
"""Download weights and data ahead of time, so a run is pure compute.

    python prefetch.py                              # default pair + gsm8k
    python prefetch.py --config qwen2.5-7b --dataset humaneval gsm8k
    python prefetch.py --all-datasets --check       # what's cached already?

Nothing here is strictly required -- ``run_baseline.py`` pulls what it needs
on first use. It exists because *when* the download happens matters:

- On a metered slot, minutes spent downloading are minutes not measuring.
- On Kaggle and Colab the cache is per-session, so every fresh session
  re-pulls ~7 GB unless HF_HOME points somewhere that persists.
- A download that dies halfway through a sweep looks like a crash.

Weights land in HF_HOME (default ~/.cache/huggingface). Point it at a
persistent directory to reuse them:

    export HF_HOME=/kaggle/working/hf      # Kaggle
    export HF_HOME=/content/drive/MyDrive/hf   # Colab, with Drive mounted
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from specdec import datasets as ds  # noqa: E402
from specdec.configs import DEFAULT_DATASET, DEFAULT_PAIR, MODEL_PAIRS  # noqa: E402
from specdec.memory import ARCHS  # noqa: E402


def cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def is_cached(model_id: str) -> bool:
    folder = "models--" + model_id.replace("/", "--")
    hub = cache_root() / "hub" / folder
    if not hub.exists():
        return False
    # a partial download leaves the folder but no complete snapshot
    snapshots = hub / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def download_gb(model_id: str) -> float:
    """fp16 weight size -- what actually comes off the hub, NF4 or not."""
    if model_id in ARCHS:
        return ARCHS[model_id].weights_gib("float16")
    return float("nan")


def fetch_model(model_id: str, check_only: bool) -> None:
    size = download_gb(model_id)
    size_s = "unknown size" if size != size else f"~{size:.1f} GB"
    if is_cached(model_id):
        print(f"  cached   {model_id}  ({size_s})")
        return
    if check_only:
        print(f"  MISSING  {model_id}  ({size_s})")
        return
    print(f"  fetching {model_id}  ({size_s}) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(model_id)
    # weights only: no device placement, no dtype conversion, no VRAM touched
    AutoModelForCausalLM.from_pretrained(model_id, device_map=None)
    print(f"  done     {model_id}")


def fetch_dataset(name: str, check_only: bool) -> None:
    if check_only:
        print(f"  (cannot check {name} without loading it)")
        return
    print(f"  fetching dataset {name} ...")
    try:
        rows = ds.load(name, limit=4)
        print(f"  done     {name}: {len(rows)} sample(s) read back")
    except Exception as exc:
        print(f"  FAILED   {name}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_PAIR, choices=sorted(MODEL_PAIRS))
    ap.add_argument("--dataset", nargs="+", default=[DEFAULT_DATASET],
                    choices=list(ds.PAPER_SIZES))
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="report what is cached without downloading")
    args = ap.parse_args()

    pair = MODEL_PAIRS[args.config]
    names = list(ds.PAPER_SIZES) if args.all_datasets else args.dataset

    print(f"HF_HOME = {cache_root()}")
    if not os.environ.get("HF_HOME"):
        print("  (not set -- on Kaggle or Colab this cache dies with the "
              "session; export HF_HOME to somewhere persistent)")
    print(f"\nmodels for {pair.key}:")
    total = sum(download_gb(m) for m in (pair.target_id, pair.draft_id)
                if not is_cached(m))
    for model_id in (pair.target_id, pair.draft_id):
        fetch_model(model_id, args.check)
    if args.check and total > 0:
        print(f"\n  ~{total:.1f} GB still to download")

    if not args.models_only:
        print("\ndatasets:")
        for name in names:
            fetch_dataset(name, args.check)

    if not args.check:
        print("\nready -- run_baseline.py will now start without a download.")


if __name__ == "__main__":
    main()
