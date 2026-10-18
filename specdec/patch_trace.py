#!/usr/bin/env python3
"""Add opt-in token capture to the engine, so a run can be replayed as text.

``round_detail`` records how many tokens were proposed and accepted, but not
*which* ones, so a finished sweep cannot be turned back into a readable
trace. This adds two optional fields to RoundRecord and fills them only when
``engine.TRACE_TOKENS`` is True, so normal sweeps are byte-for-byte unchanged.

Idempotent: run it twice and the second run reports "already patched".

    python patch_trace.py            # patch specdec/engine.py in place
    python patch_trace.py --check    # report, change nothing
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ENGINE = pathlib.Path(__file__).resolve().parent / "specdec" / "engine.py"

EDITS = [
    # 1. module-level switch, placed just above the RoundRecord dataclass
    (
        "@dataclass\nclass RoundRecord:",
        "# Set True to have each RoundRecord carry the token ids it handled.\n"
        "# Off by default: a sweep writes one record per round per prompt, and\n"
        "# the ids would multiply the size of the .jsonl checkpoint log.\n"
        "TRACE_TOKENS = False\n"
        "\n"
        "\n"
        "@dataclass\nclass RoundRecord:",
    ),
    # 2. the two new fields, after the last existing one
    (
        "    draft_ms: float\n    verify_ms: float\n",
        "    draft_ms: float\n    verify_ms: float\n"
        "    draft_ids: list[int] = field(default_factory=list)\n"
        "    emitted_ids: list[int] = field(default_factory=list)\n",
    ),
    # 3. autoregressive path records the single token it emitted
    (
        "                proposed=0,\n"
        "                accepted=0,\n"
        "                emitted=1,\n"
        "                draft_ms=0.0,\n"
        "                verify_ms=dt,\n",
        "                proposed=0,\n"
        "                accepted=0,\n"
        "                emitted=1,\n"
        "                draft_ms=0.0,\n"
        "                verify_ms=dt,\n"
        "                emitted_ids=[tok] if TRACE_TOKENS else [],\n",
    ),
    # 4. speculative path records the drafts and what survived
    (
        "                emitted=len(emitted),\n"
        "                draft_ms=draft_ms,\n"
        "                verify_ms=verify_ms,\n",
        "                emitted=len(emitted),\n"
        "                draft_ms=draft_ms,\n"
        "                verify_ms=verify_ms,\n"
        "                draft_ids=list(drafts) if TRACE_TOKENS else [],\n"
        "                emitted_ids=list(emitted) if TRACE_TOKENS else [],\n",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not ENGINE.exists():
        print(f"not found: {ENGINE}", file=sys.stderr)
        return 1

    src = ENGINE.read_text()
    if "TRACE_TOKENS" in src:
        print("already patched -- nothing to do")
        return 0

    for old, new in EDITS:
        if src.count(old) != 1:
            print(
                f"refusing to patch: expected exactly one match for\n---\n{old}---\n"
                f"found {src.count(old)}. The engine has changed; patch by hand.",
                file=sys.stderr,
            )
            return 1
        src = src.replace(old, new, 1)

    if args.check:
        print("would patch cleanly (4 edits)")
        return 0

    backup = ENGINE.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(ENGINE.read_text())
    ENGINE.write_text(src)
    print(f"patched {ENGINE}  (backup at {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
