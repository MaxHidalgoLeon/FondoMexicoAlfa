"""Lightweight pipeline progress tracker.

Emits a single line to stdout each time the pipeline crosses a stage boundary
or makes meaningful sub-progress (throttled to ≤1 line/second). The line is
formatted so that multiple sources running in parallel can be visually
distinguished by their source label prefix:

    [Bloomberg   5%] 00:23
    [Yahoo      12%] 00:31
    [Bloomberg  15%] 00:51
    [Bloomberg ✓] completed in 38:42

Stages are weighted by their empirical share of total runtime so that the
percentage roughly matches wall-clock progress, not stage count. Reports
(stage 8) get 2% even though it's the final phase, because it's quick.

The tracker is a process-local singleton. Each run_pipeline() invocation
re-initialises it. For parallel multi-source runs (run_all.py with
parallel_sources=true), the source label comes from the FMIA_SOURCE_LABEL
environment variable; subprocesses inherit stdout from the parent so the
prefixed lines naturally interleave.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Final


# Stage weights (must sum to 100). Calibrated from observed runtime ratios
# on mock + hedge + reform: forecast_returns dominates (~60%), backtest is
# the next biggest sink (~15%), everything else is overhead.
_STAGE_WEIGHTS: Final[dict[str, int]] = {
    "load_data":   2,
    "features":    3,
    "forecast":   60,
    "bl_views":    5,
    "backtest":   15,
    "hedge":       8,
    "reform":      5,
    "reports":     2,
}


def _stage_offsets() -> dict[str, tuple[int, int]]:
    """Return (start_pct, end_pct) for each stage, ordered by insertion."""
    out: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, w in _STAGE_WEIGHTS.items():
        out[name] = (offset, offset + w)
        offset += w
    return out


_OFFSETS: Final = _stage_offsets()


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as Xm or Xh Ym (no seconds shown)."""
    m = int(seconds) // 60
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h {m % 60}m"


class _Tracker:
    def __init__(self, label: str) -> None:
        self.label = label
        self.start = time.time()
        self.current_pct = 0
        self.current_stage = ""
        self._lock = threading.Lock()
        self._last_print = 0.0

    def _line_prefix(self, pct_str: str) -> str:
        if self.label:
            # Pad label to 10 chars so columns align across sources.
            return f"[{self.label:<10s} {pct_str}]"
        return f"[{pct_str}]"

    def _print(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_print < 30.0:
            return
        self._last_print = now
        pct_str = f"{self.current_pct:3d}%"
        elapsed = _fmt_elapsed(now - self.start)
        print(f"{self._line_prefix(pct_str)} {elapsed}", flush=True)

    def set_stage(self, stage: str, sub_pct: float = 0.0) -> None:
        if stage not in _OFFSETS:
            return
        start_pct, end_pct = _OFFSETS[stage]
        with self._lock:
            new_pct = min(end_pct, int(start_pct + (end_pct - start_pct) * max(0.0, min(1.0, sub_pct)) ))
            changed = new_pct != self.current_pct or stage != self.current_stage
            if changed:
                self.current_pct = new_pct
                self.current_stage = stage
            # Always print on stage/pct change; otherwise throttle to 30s.
            self._print(force=changed)

    def done(self) -> None:
        elapsed = _fmt_elapsed(time.time() - self.start)
        prefix = f"[{self.label:<10s} ✓]" if self.label else "[✓]"
        print(f"{prefix} completed in {elapsed}", flush=True)


_tracker: _Tracker | None = None


def init(label: str | None = None) -> _Tracker:
    """(Re)initialise the global progress tracker.

    Label can be passed explicitly (e.g. for tests) or comes from the
    FMIA_SOURCE_LABEL environment variable, which is set by run_all.py when
    launching parallel subprocesses. When unset, the tracker still works but
    emits unprefixed lines, suitable for a single-source local run.
    """
    global _tracker
    final_label = label if label is not None else os.environ.get("FMIA_SOURCE_LABEL", "")
    _tracker = _Tracker(final_label)
    return _tracker


def set_stage(stage: str, sub_pct: float = 0.0) -> None:
    """Advance the tracker to ``stage`` with optional sub-progress in [0,1].

    No-op when the tracker has not been initialised (e.g. inside unit tests).
    sub_pct represents how far through the current stage we are: 0.0 = just
    entered, 1.0 = about to exit.
    """
    if _tracker is not None:
        _tracker.set_stage(stage, sub_pct)


def done() -> None:
    """Print the completion line and clear the tracker."""
    global _tracker
    if _tracker is not None:
        _tracker.done()
        _tracker = None
