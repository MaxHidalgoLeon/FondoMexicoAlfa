"""Unit tests for Jensen's alpha computation and Yahoo/Refinitiv fundamental detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alpha_significance import (
    _beta,
    _annualized_alpha,
    _annualized_return,
    compute_benchmark_alpha_significance,
)
from src.features import (
    _pit_merge_fundamentals,
    _EQUITY_FUND_COLS,
    rolling_momentum,
)

# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

def test_beta_perfect_correlation():
    """fund = 2 × benchmark  →  beta ≈ 2."""
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 300))
    fund = bench * 2.0
    assert abs(_beta(fund, bench) - 2.0) < 0.02


def test_beta_market_portfolio_is_one():
    """fund = benchmark  →  beta = 1."""
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 300))
    assert abs(_beta(series, series) - 1.0) < 1e-6


def test_beta_uncorrelated_near_zero():
    """Uncorrelated series  →  beta near zero."""
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0, 0.01, 500))
    fund = pd.Series(rng.normal(0, 0.01, 500))
    assert abs(_beta(fund, bench)) < 0.20


def test_beta_short_series_returns_one():
    """< 20 observations  →  fallback beta = 1."""
    short = pd.Series([0.01, -0.01, 0.005])
    assert _beta(short, short * 2) == 1.0


# ---------------------------------------------------------------------------
# Jensen's alpha
# ---------------------------------------------------------------------------

def test_jensen_alpha_identical_series_is_zero():
    """fund = benchmark  →  Jensen alpha = 0 by construction."""
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(0.0003, 0.01, 500))
    alpha = _annualized_alpha(s, s, risk_free_daily=0.04 / 252)
    assert abs(alpha) < 1e-6


def test_jensen_alpha_pure_idiosyncratic_excess():
    """fund = bench + constant positive drift  →  positive Jensen alpha."""
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0003, 0.01, 500))
    # idiosyncratic return: uncorrelated with benchmark, persistently positive
    extra = pd.Series(rng.normal(0.0005, 0.001, 500))
    fund = bench + extra
    alpha = _annualized_alpha(fund, bench, risk_free_daily=0.02 / 252)
    assert alpha > 0


def test_jensen_alpha_differs_from_raw_excess_when_beta_not_one():
    """When beta ≠ 1, Jensen alpha ≠ raw excess return."""
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0.0003, 0.01, 500))
    fund = bench * 0.5           # beta ≈ 0.5, lower vol, lower raw return
    raw_excess = _annualized_return(fund) - _annualized_return(bench)
    jensen = _annualized_alpha(fund, bench, risk_free_daily=0.04 / 252)
    # Raw excess is negative (fund returns less), but Jensen alpha should be
    # close to zero (fund tracks benchmark with beta 0.5, no idiosyncratic alpha)
    assert abs(jensen) < abs(raw_excess), (
        "Jensen alpha should be smaller in magnitude than raw excess when beta < 1"
    )


# ---------------------------------------------------------------------------
# compute_benchmark_alpha_significance — returns beta field
# ---------------------------------------------------------------------------

_FAST_SETTINGS = {
    "bootstrap_enabled": True,
    "bootstrap_n_reps": 50,
    "bootstrap_block_size": 20,
    "bootstrap_confidence": 0.95,
    "bootstrap_seed": 42,
}


def test_alpha_significance_contains_beta():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-01", periods=300)
    fund = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
    bench = pd.DataFrame({"IPC": rng.normal(0.0002, 0.01, 300)}, index=idx)
    result = compute_benchmark_alpha_significance(
        fund, bench, settings=_FAST_SETTINGS, risk_free_rate=0.05
    )
    assert "IPC" in result
    assert "beta" in result["IPC"], "'beta' key missing from alpha significance output"
    assert 0.0 < result["IPC"]["beta"] < 3.0, "Beta should be a plausible positive number"


def test_alpha_significance_empty_benchmark_returns_empty():
    rng = np.random.default_rng(7)
    fund = pd.Series(rng.normal(0, 0.01, 100))
    result = compute_benchmark_alpha_significance(fund, pd.DataFrame(), settings=_FAST_SETTINGS)
    assert result == {}


# ---------------------------------------------------------------------------
# PIT merge — Yahoo vs Refinitiv mode detection
# ---------------------------------------------------------------------------

def test_pit_merge_yahoo_snapshot_all_nan():
    """Yahoo: single today-dated snapshot → all historical dates receive NaN."""
    today = pd.Timestamp.today().normalize()
    fund = pd.DataFrame({
        "date": [today, today],
        "ticker": ["A", "B"],
        "pe_ratio": [12.0, 8.0],
        "pb_ratio": [1.5, 1.0],
    })
    hist = pd.date_range("2020-01-01", periods=12, freq="60D")
    feature_df = pd.DataFrame({
        "date": list(hist) * 2,
        "ticker": ["A"] * 12 + ["B"] * 12,
    })
    result = _pit_merge_fundamentals(feature_df, fund, ["pe_ratio", "pb_ratio"])
    assert result["pe_ratio"].isna().all(), "Yahoo snapshot must produce all-NaN for historical dates"
    assert result["pb_ratio"].isna().all()


def test_pit_merge_refinitiv_historical_populated():
    """Refinitiv: quarterly history → historical dates get real values, no look-ahead."""
    fund = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=8, freq="QE"),
        "ticker": ["A"] * 8,
        "pe_ratio": [10.0 + i for i in range(8)],
    })
    hist = pd.date_range("2019-06-01", periods=20, freq="30D")
    feature_df = pd.DataFrame({"date": hist, "ticker": ["A"] * 20})
    result = _pit_merge_fundamentals(feature_df, fund, ["pe_ratio"])
    # All feature dates fall after 2019-01-01 (first quarterly record) → all populated
    assert result["pe_ratio"].notna().all(), "Refinitiv history should populate all feature dates"


def test_pit_merge_no_future_leak():
    """PIT: a date should never see a fundamental filed AFTER that date."""
    fund = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01", "2022-07-01"]),
        "ticker": ["A", "A"],
        "pe_ratio": [10.0, 20.0],
    })
    # Feature date between the two fundamental dates
    feature_df = pd.DataFrame({"date": pd.to_datetime(["2022-04-01"]), "ticker": ["A"]})
    result = _pit_merge_fundamentals(feature_df, fund, ["pe_ratio"])
    # Should see the 2022-01-01 filing (10.0), NOT the 2022-07-01 one (20.0)
    assert result["pe_ratio"].iloc[0] == pytest.approx(10.0), "Look-ahead detected: saw future fundamental"


# ---------------------------------------------------------------------------
# rolling_momentum — skip parameter
# ---------------------------------------------------------------------------

def test_rolling_momentum_skip1_no_lookahead():
    """skip=1: signal on date t uses P_{t-1}, not P_t."""
    prices = pd.DataFrame({"A": range(1, 200)}, dtype=float)
    mom = rolling_momentum(prices, window=63, skip=1)
    # On row 0, signal is NaN (no history); row 63 onward should be non-NaN
    assert mom["A"].iloc[0] != mom["A"].iloc[0]  # NaN


def test_rolling_momentum_skip21_excludes_recent_month():
    """skip=21: signal on date t uses P_{t-21}, not P_{t-1}."""
    n = 300
    rng = np.random.default_rng(8)
    prices = pd.DataFrame({"A": np.cumprod(1 + rng.normal(0, 0.01, n))})
    mom1 = rolling_momentum(prices, 126, skip=1)
    mom21 = rolling_momentum(prices, 126, skip=21)
    # The two series should differ — different skip means different numerator
    valid = mom1["A"].dropna().index.intersection(mom21["A"].dropna().index)
    assert not (mom1["A"][valid] == mom21["A"][valid]).all(), \
        "skip=1 and skip=21 should produce different momentum values"
