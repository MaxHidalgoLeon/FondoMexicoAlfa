#!/usr/bin/env python
"""Pytest smoke tests for hedge_overlay module (migrated from print-only script)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genextreme

from src.hedge_overlay import (
    long_short_portfolio,
    dynamic_leverage,
    fx_directional_overlay,
    tail_risk_hedge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 0.01, 150))


@pytest.fixture(scope="module")
def sample_signal_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.Timestamp("2024-01-01"),
        "ticker": [f"T{i}" for i in range(10)],
        "expected_return": rng.standard_normal(10),
        "sector": ["A"] * 5 + ["B"] * 5,
    })


@pytest.fixture(scope="module")
def sample_macro_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "date": dates,
        "banxico_rate": 5.5 + rng.normal(0, 0.05, 60),
        "us_fed_rate": 5.0 + rng.normal(0, 0.05, 60),
        "usd_mxn": 19.0 + np.cumsum(rng.normal(0, 0.05, 60)),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_long_short_portfolio_returns_positions(sample_signal_df):
    portfolio = long_short_portfolio(sample_signal_df, top_n=3, bottom_n=2)
    assert not portfolio.empty, "Expected non-empty portfolio"
    assert "net_weight" in portfolio.columns
    assert set(portfolio["side"].unique()).issubset({"long", "short"})


def test_long_short_gross_sum_is_one(sample_signal_df):
    portfolio = long_short_portfolio(sample_signal_df, top_n=3, bottom_n=2)
    if portfolio.empty:
        pytest.skip("Portfolio is empty")
    gross = portfolio["net_weight"].abs().sum()
    assert abs(gross - 1.0) < 0.01, f"Gross should be ≈1.0, got {gross:.4f}"


def test_dynamic_leverage_bounds(sample_returns):
    leverage = dynamic_leverage(sample_returns, max_leverage=1.5, cvar_limit=0.02, window=30)
    assert len(leverage) == len(sample_returns)
    assert leverage.min() >= 0.5 - 1e-9
    assert leverage.max() <= 1.5 + 1e-9
    assert leverage.isna().sum() == 0, "Leverage series should have no NaNs"


def test_fx_overlay_hedge_ratio_within_bounds(sample_macro_df):
    signal_df = pd.DataFrame({
        "date": sample_macro_df["date"],
        "ticker": "TEST",
        "sector": "X",
        "expected_return": 0.0,
    })
    usd_exposure = pd.Series([0.3])
    result = fx_directional_overlay(
        sample_macro_df, signal_df, usd_exposure,
        min_hedge_ratio=0.10, max_hedge_ratio=0.95,
    )
    assert "hedge_ratio" in result.columns
    assert result["hedge_ratio"].min() >= 0.10 - 1e-9
    assert result["hedge_ratio"].max() <= 0.95 + 1e-9


def test_fx_overlay_no_estimated_fx_pnl_column(sample_macro_df):
    """Ensure estimated_fx_pnl is NOT in output (look-ahead removed)."""
    signal_df = pd.DataFrame({
        "date": sample_macro_df["date"],
        "ticker": "TEST",
        "sector": "X",
        "expected_return": 0.0,
    })
    result = fx_directional_overlay(sample_macro_df, signal_df, pd.Series([0.3]))
    assert "estimated_fx_pnl" not in result.columns


def test_tail_risk_hedge_all_keys(sample_returns):
    gev_params = genextreme.fit(-sample_returns[sample_returns < 0])
    result = tail_risk_hedge(sample_returns, gev_params, protection_level=0.99, cost_bps=30.0)
    expected_keys = {"unhedged_loss_at_99", "hedge_payoff", "daily_cost_drag", "net_benefit", "recommended"}
    assert set(result.keys()) == expected_keys
    assert isinstance(result["recommended"], bool)
    for k in ("unhedged_loss_at_99", "hedge_payoff", "daily_cost_drag", "net_benefit"):
        assert np.isfinite(result[k]), f"{k} must be finite"
