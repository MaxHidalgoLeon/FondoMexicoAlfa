"""PIT (point-in-time) invariant tests for compute_macro_regime_history.

The core invariant: the regime assigned to date t must depend ONLY on macro
data up to and including t, never on future observations. Concretely:

    compute_macro_regime_history(macro_full).loc[t]
    ==
    compute_macro_regime_history(macro[:t]).iloc[-1]

This guards against future regressions where someone replaces a causal
operation (expanding z-score, adjust=False EWMA) with a non-causal one
(rolling with future fill, transform across the full series, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_macro_indexed(n_months: int = 36, seed: int = 42) -> pd.DataFrame:
    """Synthetic macro DataFrame with date as index (mirrors backtest.py usage).

    Uses non-trivial variation in all three signals so that expanding stats
    actually diverge across dates — a flat series would pass even a broken
    non-causal implementation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")

    # Banxico: drifting rate cycle so diff(3) changes sign
    banxico = np.cumsum(rng.normal(0.05, 0.30, n_months)) + 5.0
    # USD/MXN: random walk with occasional jumps
    usd_mxn = np.cumsum(rng.normal(0.003, 0.025, n_months)) + 17.5
    # Industrial production YoY: noisy, crosses zero
    ip_yoy = rng.normal(0.02, 0.05, n_months)

    return pd.DataFrame(
        {
            "banxico_rate": banxico,
            "usd_mxn": usd_mxn,
            "industrial_production_yoy": ip_yoy,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# PIT invariant — regime label
# ---------------------------------------------------------------------------

def test_regime_label_pit_invariant():
    """regime at t must be identical with full vs truncated macro panel."""
    from src.risk import compute_macro_regime_history

    macro_full = _make_macro_indexed(36)
    full_history = compute_macro_regime_history(macro_full)

    # Test every date from month 6 onward (first few have NaN expanding stats)
    test_dates = macro_full.index[5:]

    failures = []
    for date in test_dates:
        macro_trunc = macro_full.loc[macro_full.index <= date]
        trunc_history = compute_macro_regime_history(macro_trunc)

        regime_full = full_history.loc[date, "regime"]
        regime_trunc = trunc_history.iloc[-1]["regime"]

        if regime_full != regime_trunc:
            failures.append(
                f"  date={date.date()}: full={regime_full!r}  trunc={regime_trunc!r}"
            )

    assert not failures, (
        f"PIT violation in compute_macro_regime_history — regime differs at {len(failures)} date(s):\n"
        + "\n".join(failures)
        + "\nA causal operation (expanding z-score or EWMA) was likely replaced "
        "with a non-causal one that leaks future information."
    )


# ---------------------------------------------------------------------------
# PIT invariant — numeric scores (tighter check)
# ---------------------------------------------------------------------------

def test_regime_score_pit_invariant():
    """regime_score and regime_score_smoothed at t must match truncated computation."""
    from src.risk import compute_macro_regime_history

    macro_full = _make_macro_indexed(36)
    full_history = compute_macro_regime_history(macro_full)

    # Sample every 3rd date to keep the test fast while staying sensitive
    test_dates = macro_full.index[5::3]

    score_failures = []
    smoothed_failures = []

    for date in test_dates:
        macro_trunc = macro_full.loc[macro_full.index <= date]
        trunc_history = compute_macro_regime_history(macro_trunc)

        score_full = float(full_history.loc[date, "regime_score"])
        score_trunc = float(trunc_history.iloc[-1]["regime_score"])

        smoothed_full = float(full_history.loc[date, "regime_score_smoothed"])
        smoothed_trunc = float(trunc_history.iloc[-1]["regime_score_smoothed"])

        # NaN vs NaN is OK; NaN vs value is a failure
        both_nan_score = np.isnan(score_full) and np.isnan(score_trunc)
        both_nan_smooth = np.isnan(smoothed_full) and np.isnan(smoothed_trunc)

        if not both_nan_score and not np.isclose(score_full, score_trunc, atol=1e-10):
            score_failures.append(
                f"  date={date.date()}: full={score_full:.6f}  trunc={score_trunc:.6f}"
            )
        if not both_nan_smooth and not np.isclose(smoothed_full, smoothed_trunc, atol=1e-10):
            smoothed_failures.append(
                f"  date={date.date()}: full={smoothed_full:.6f}  trunc={smoothed_trunc:.6f}"
            )

    assert not score_failures, (
        f"PIT violation in regime_score at {len(score_failures)} date(s):\n"
        + "\n".join(score_failures)
    )
    assert not smoothed_failures, (
        f"PIT violation in regime_score_smoothed at {len(smoothed_failures)} date(s):\n"
        + "\n".join(smoothed_failures)
    )


# ---------------------------------------------------------------------------
# Regression guard — future data must NOT change past regimes
# ---------------------------------------------------------------------------

def test_appending_future_rows_does_not_change_past_regimes():
    """Adding rows after date t must not retroactively change t's regime.

    This is the sharpest possible test: take a history, record regimes,
    append one more row, recompute, and verify nothing changed.
    """
    from src.risk import compute_macro_regime_history

    macro_base = _make_macro_indexed(24)
    history_base = compute_macro_regime_history(macro_base)

    # Append a new month with extreme values that would shift any global statistic
    last_date = macro_base.index[-1]
    new_date = last_date + pd.DateOffset(months=1)
    new_row = pd.DataFrame(
        {
            "banxico_rate": [20.0],   # extreme rate hike
            "usd_mxn": [30.0],        # extreme peso depreciation
            "industrial_production_yoy": [-0.5],  # severe contraction
        },
        index=[new_date],
    )
    macro_extended = pd.concat([macro_base, new_row])
    history_extended = compute_macro_regime_history(macro_extended)

    # All regimes up to macro_base.index[-1] must be unchanged
    changed = []
    for date in macro_base.index[5:]:  # skip initial NaN-heavy rows
        r_base = history_base.loc[date, "regime"]
        r_ext = history_extended.loc[date, "regime"]
        if r_base != r_ext:
            changed.append(f"  date={date.date()}: before={r_base!r}  after={r_ext!r}")

    assert not changed, (
        f"Adding a future row retroactively changed {len(changed)} past regime(s):\n"
        + "\n".join(changed)
        + "\nThis indicates a non-causal computation in compute_macro_regime_history."
    )
