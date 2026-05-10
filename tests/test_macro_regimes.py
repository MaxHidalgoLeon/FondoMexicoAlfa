"""Tests for src/macro_regimes.py — regime assignment and performance decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_macro(n_months: int = 24, seed: int = 0) -> pd.DataFrame:
    """Synthetic monthly macro DataFrame with banxico_rate."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rate = np.cumsum(rng.normal(0.05, 0.1, n_months)) + 4.0
    return pd.DataFrame({"date": dates, "banxico_rate": rate})


def _make_prices(n_days: int = 500, n_tickers: int = 5, seed: int = 1) -> pd.DataFrame:
    """Synthetic daily equity price panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    px = np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_tickers)), axis=0) * 100
    tickers = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(px, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# 3.10.1  No-lookahead for rate regime
# ---------------------------------------------------------------------------

def test_regime_no_lookahead():
    """Regime at date t must be identical whether called with full or truncated macro."""
    from src.macro_regimes import assign_rate_regime

    macro = _make_macro(24)
    test_date = pd.Timestamp("2021-07-31")   # 18 months in

    # Truncate to strictly before test_date
    macro_truncated = macro[macro["date"] < test_date].copy()

    regime_full = assign_rate_regime(macro, test_date)
    regime_trunc = assign_rate_regime(macro_truncated, test_date)

    assert regime_full == regime_trunc, (
        f"Regime differs: full={regime_full}, truncated={regime_trunc}. "
        "Lookahead detected — future rows are affecting the result."
    )


# ---------------------------------------------------------------------------
# 3.10.2  Exhaustive labels
# ---------------------------------------------------------------------------

def test_regime_labels_exhaustive():
    """Every rebalance date must receive a non-null rate and stress regime label."""
    from src.macro_regimes import build_regime_table, TIGHTENING, EASING, NEUTRAL, STRESS, CALM

    macro  = _make_macro(24)
    prices = _make_prices(500, 5)
    tickers = [f"T{i}" for i in range(5)]

    # Rebalance dates: last business day of each month within price range
    rebal_dates = pd.date_range("2020-06-30", periods=16, freq="ME")
    rebal_dates = [d for d in rebal_dates if d >= prices.index[0] and d <= prices.index[-1]]

    rt = build_regime_table(rebal_dates, macro, prices, tickers)

    # No NaN labels
    assert rt["rate_regime"].isna().sum() == 0, "NaN in rate_regime"
    assert rt["stress_regime"].isna().sum() == 0, "NaN in stress_regime"

    # Labels are valid
    valid_rate   = {TIGHTENING, EASING, NEUTRAL}
    valid_stress = {STRESS, CALM}
    for rr in rt["rate_regime"]:
        assert rr in valid_rate, f"Unknown rate_regime label: {rr!r}"
    for sr in rt["stress_regime"]:
        assert sr in valid_stress, f"Unknown stress_regime label: {sr!r}"

    # Every input date appears in the output
    assert len(rt) == len(rebal_dates), (
        f"Output has {len(rt)} rows but {len(rebal_dates)} input dates."
    )


# ---------------------------------------------------------------------------
# 3.10.3  Stress threshold reproducible
# ---------------------------------------------------------------------------

def test_stress_threshold_reproducible():
    """compute_stress_threshold returns the same value on repeated calls."""
    from src.macro_regimes import compute_stress_threshold, _ipc_equity_returns

    prices  = _make_prices(500, 5)
    tickers = [f"T{i}" for i in range(5)]
    ipc_ret = _ipc_equity_returns(prices, tickers)

    rebal_dates = pd.date_range("2020-06-30", periods=16, freq="ME")
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    t1 = compute_stress_threshold(rebal_dates, ipc_ret, percentile=75)
    t2 = compute_stress_threshold(rebal_dates, ipc_ret, percentile=75)

    assert t1 == t2, f"Threshold not reproducible: {t1} vs {t2}"
    assert np.isfinite(t1), f"Threshold is not finite: {t1}"
    assert t1 > 0, f"Threshold must be positive: {t1}"


# ---------------------------------------------------------------------------
# 3.10.4  Performance table schema
# ---------------------------------------------------------------------------

def test_performance_table_schema(tmp_path):
    """regime_performance_table.csv must have all required columns."""
    csv_path = Path("reports") / "regime_performance_table.csv"
    if not csv_path.exists():
        pytest.skip("regime_performance_table.csv not yet generated — run render_step3_report.py first")

    df = pd.read_csv(csv_path)
    required = {
        "rate_regime", "stress_regime", "n_rebalances",
        "ic_mean", "ic_std", "icir", "hit_rate",
        "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "cvar_95", "mean_turnover",
    }
    missing = required - set(df.columns)
    assert not missing, f"Performance table missing columns: {missing}"

    # Rate regime column contains valid labels
    from src.macro_regimes import TIGHTENING, EASING, NEUTRAL
    valid = {TIGHTENING, EASING, NEUTRAL}
    bad = set(df["rate_regime"]) - valid
    assert not bad, f"Unexpected rate_regime values: {bad}"
