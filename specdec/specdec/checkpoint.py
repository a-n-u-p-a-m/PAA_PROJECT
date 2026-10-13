"""Checkpointing for runs that will not finish inside one session.

A two-hour slot is a hard wall. The pipeline has to survive meeting it, and
"survive" means three separate things:

1. **Lose nothing.** Every completed unit of work is appended to a JSONL log
   the moment it finishes, so a kill costs at most the unit in flight.

2. **Stop before the wall, not at it.** Being killed mid-write can leave a
   truncated last line, and the merged result file never gets produced. With
   a deadline, the run finishes the current unit, flushes, writes the merged
   JSON and prints the resume command -- a planned landing instead of a crash.
   The deadline check is predictive: it stops when the *next* unit would not
   fit, using the observed mean unit time, rather than when the clock has
   already run out.

3. **Refuse to resume into the wrong run.** The nastiest failure here is
   silent: resuming with a different precision or prompt limit appends rows
   that are not comparable to the ones already in the log, and nothing
   complains until the analysis makes no sense. Every log carries a sidecar
   fingerprint of the settings that produced it, and a mismatched resume is
   refused rather than merged.

Signals matter because that is how slots and notebooks actually end: a
SIGTERM, then a SIGKILL some seconds later. The handler below converts the
first signal into a clean stop after the current unit. A second signal exits
immediately, for when you mean it.
"""

from __future__ import annotations

import hashlib
import json
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.I)


def parse_deadline(spec: str, now: Optional[float] = None) -> float:
    """``"90m"``, ``"1h45m"``, ``"5400s"``, ``"3600"`` or an absolute ``"23:30"``.

    Returns a POSIX timestamp. A bare clock time that has already passed
    today is taken as tomorrow.
    """
    now = now if now is not None else time.time()
    spec = spec.strip().lower()

    if ":" in spec:
        hh, _, mm = spec.partition(":")
        target = datetime.fromtimestamp(now).replace(
            hour=int(hh), minute=int(mm), second=0, microsecond=0
        )
        if target.timestamp() <= now:
            target += timedelta(days=1)
        return target.timestamp()

    if spec.isdigit():  # bare number = seconds
        return now + int(spec)

    m = _DURATION.fullmatch(spec)
    if not m or not any(m.groups()):
        raise ValueError(
            f"cannot read deadline {spec!r}; try '90m', '1h45m', '5400s' or '23:30'"
        )
    h, mi, s = (int(g or 0) for g in m.groups())
    total = h * 3600 + mi * 60 + s
    if total <= 0:
        raise ValueError(f"deadline {spec!r} is not in the future")
    return now + total


def fingerprint(settings: dict[str, Any]) -> str:
    """Stable hash of the settings that must not change across a resume."""
    canonical = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class Unit:
    """One (strategy, prompt) pair -- the granularity of a checkpoint."""

    strategy: str
    dataset: str
    sample_index: int
    sample: Any = None
    runner: Any = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.strategy, self.dataset, self.sample_index)


@dataclass
class Checkpoint:
    out: Path
    settings: dict[str, Any]
    resume: bool = False
    force: bool = False
    deadline: Optional[float] = None
    reserve_s: float = 180.0

    done: set[tuple[str, str, int]] = field(default_factory=set, init=False)
    stop_reason: Optional[str] = field(default=None, init=False)
    _durations: list[float] = field(default_factory=list, init=False)
    _started: float = field(default_factory=time.time, init=False)
    _signalled: bool = field(default=False, init=False)

    # -- paths --------------------------------------------------------------

    @property
    def log(self) -> Path:
        return self.out.with_suffix(".jsonl")

    @property
    def meta(self) -> Path:
        return self.out.with_suffix(".meta.json")

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        fp = fingerprint(self.settings)

        if self.resume and self.log.exists():
            self._check_fingerprint(fp)
            self.done = self._load_done()
            print(f"resuming {self.log.name}: {len(self.done)} units already done")
        else:
            if self.log.exists():
                # a stale log from different settings would corrupt the merge
                self.log.unlink()
            if self.meta.exists():
                self.meta.unlink()

        self.meta.write_text(json.dumps(
            {"fingerprint": fp, "settings": self.settings,
             "created": datetime.now().isoformat(timespec="seconds")},
            indent=2,
        ))
        self._install_signal_handlers()

    def _check_fingerprint(self, fp: str) -> None:
        if not self.meta.exists():
            print(f"  no {self.meta.name} beside the log; cannot verify the "
                  f"settings match. Continuing, but check the results.")
            return
        stored = json.loads(self.meta.read_text())
        if stored.get("fingerprint") == fp:
            return
        diff = [
            f"      {k}: log has {stored['settings'].get(k)!r}, "
            f"now {self.settings.get(k)!r}"
            for k in sorted(set(stored.get("settings", {})) | set(self.settings))
            if stored.get("settings", {}).get(k) != self.settings.get(k)
        ]
        message = (
            f"\nrefusing to resume {self.log.name}: the settings changed.\n"
            + "\n".join(diff)
            + "\n\n  Rows already in the log are not comparable to the ones "
              "this run would add.\n"
              "  Use a different --out, or pass --force-resume if you are "
              "certain they belong together.\n"
        )
        if not self.force:
            raise SystemExit(message)
        print(message + "  --force-resume given; continuing anyway.\n")

    def _load_done(self) -> set[tuple[str, str, int]]:
        done: set[tuple[str, str, int]] = set()
        kept: list[str] = []
        with self.log.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # a kill mid-write leaves one partial line; drop it
                    continue
                done.add((row["strategy"], row["dataset"], row["sample_index"]))
                kept.append(line if line.endswith("\n") else line + "\n")
        # rewrite without the truncated tail so later appends stay parseable
        self.log.write_text("".join(kept))
        return done

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            if self._signalled:
                raise SystemExit(f"second signal {signum}; exiting now")
            self._signalled = True
            self.stop_reason = f"signal {signal.Signals(signum).name}"
            print(f"\n  {self.stop_reason} received -- finishing the current "
                  f"unit, then writing results. Signal again to abort.",
                  flush=True)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported

    # -- during the run -----------------------------------------------------

    def record(self, row: dict, seconds: float) -> None:
        with self.log.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        self.done.add((row["strategy"], row["dataset"], row["sample_index"]))
        self._durations.append(seconds)

    @property
    def mean_unit_s(self) -> float:
        recent = self._durations[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    def should_stop(self) -> Optional[str]:
        """Checked between units. Returns a reason, or None to continue."""
        if self.stop_reason:
            return self.stop_reason
        if self.deadline is None:
            return None
        remaining = self.deadline - time.time()
        needed = self.mean_unit_s + self.reserve_s
        if remaining <= needed:
            self.stop_reason = (
                f"deadline: {remaining / 60:.1f} min left, next unit needs "
                f"~{self.mean_unit_s:.0f}s plus {self.reserve_s / 60:.0f} min reserved"
            )
            return self.stop_reason
        return None

    def eta(self, remaining_units: int) -> str:
        if not self._durations or remaining_units <= 0:
            return "unknown"
        secs = self.mean_unit_s * remaining_units
        return f"{secs / 60:.0f}m" if secs < 5400 else f"{secs / 3600:.1f}h"

    def progress(self, completed: int, total: int, remaining: int) -> str:
        pct = completed / total * 100 if total else 100.0
        elapsed = (time.time() - self._started) / 60
        line = (f"  [{completed}/{total}  {pct:4.1f}%]  elapsed {elapsed:.0f}m  "
                f"eta {self.eta(remaining)}")
        if self.deadline:
            left = (self.deadline - time.time()) / 60
            line += f"  deadline in {left:.0f}m"
        return line

    # -- finishing ----------------------------------------------------------

    def all_rows(self) -> list[dict]:
        """Every row in the log, resumed ones included."""
        if not self.log.exists():
            return []
        rows = []
        with self.log.open() as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    def report_remaining(self, pending: Iterable[Unit], argv: list[str]) -> None:
        pending = list(pending)
        if not pending:
            print("\n  run complete: nothing left.")
            return
        by_strategy: dict[str, int] = {}
        for u in pending:
            by_strategy[u.strategy] = by_strategy.get(u.strategy, 0) + 1
        print(f"\n  {len(pending)} units still to do"
              f" (~{self.eta(len(pending))} at the observed rate):")
        for name, n in sorted(by_strategy.items()):
            print(f"    {name:<16}{n:>5}")
        script = Path(argv[0]).name if argv else "run_baseline.py"
        cmd = " ".join([script, *argv[1:]])
        if "--resume" not in cmd:
            cmd += " --resume"
        print(f"\n  resume with:\n    python {cmd}")
