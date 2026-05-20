"""Unit tests for src/backtest.py — turnover math, returns lag, basic shapes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import get_rebalance_dates, run_backtest


def _synthetic_universe(tickers: list[str]) -> pd.DataFrame:
    """Build a 2-equity, 0-FI universe (no liquidity sleeve interference)."""
    return pd.DataFrame({
        "ticker": tickers,
        "asset_class": ["equity"] * len(tickers),
        "sector": ["Industrial"] * len(tickers),
        "investable": [True] * len(tickers),
        "liquidity_score": [0.8] * len(tickers),
        "market_cap_mxn": [1e9] * len(tickers),
        "usd_exposure": [0.2] * len(tickers),
    })


def _synthetic_prices(tickers: list[str], n_days: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    data = {}
    for t in tickers:
        data[t] = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days)))
    return pd.DataFrame(data, index=idx)


def _flat_signal(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """A constant cross-sectional signal triggers a stable target portfolio."""
    eom = prices.resample("ME").last().index
    rows = []
    for d in eom:
        for i, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "expected_return": 0.05 + 0.001 * i})
    return pd.DataFrame(rows)


def test_rebalance_dates_are_monthly_eom():
    prices = _synthetic_prices(["A", "B"], n_days=400)
    dates = get_rebalance_dates(prices, freq="ME")
    assert len(dates) >= 12
    # Each interior rebalance date must be month-end (or its last trading day);
    # the final entry can be the data-set cutoff, not a true EOM.
    for d in dates[:-1]:
        assert d.day >= 24, f"Unexpected non-EOM rebalance date: {d}"


def test_run_backtest_returns_have_expected_shape():
    tickers = ["A", "B"]
    prices = _synthetic_prices(tickers, n_days=600)
    universe = _synthetic_universe(tickers)
    signal = _flat_signal(prices, tickers)
    out = run_backtest(
        prices, signal, universe,
        optimizer="mv",
        risk_free_rate=0.05,
        settings={"bootstrap_enabled": False},
    )
    assert "metrics" in out
    assert "weights" in out
    assert "returns" in out
    assert "turnover" in out
    metrics = out["metrics"]
    for key in ("sharpe", "annualized_return", "annualized_vol", "max_drawdown", "turnover"):
        assert key in metrics
        assert np.isfinite(metrics[key]) or metrics[key] == 0.0


def test_turnover_metric_drops_initial_allocation():
    """The reported turnover must exclude the first rebalance (zero -> target)."""
    tickers = ["A", "B"]
    prices = _synthetic_prices(tickers, n_days=600, seed=1)
    universe = _synthetic_universe(tickers)
    signal = _flat_signal(prices, tickers)
    out = run_backtest(
        prices, signal, universe,
        optimizer="mv",
        risk_free_rate=0.05,
        settings={"bootstrap_enabled": False, "mv_turnover_penalty": 0.05},
    )
    tv = out["turnover"]
    nonzero = tv[tv > 0]
    assert nonzero.iloc[0] >= 0.5, "First rebalance should still be visible in the raw series."
    reported = out["metrics"]["turnover"]
    # The reported steady-state turnover must be < first-allocation spike / 2.
    assert reported < 0.5 * float(nonzero.iloc[0])


def test_zero_weight_when_signal_empty():
    """run_backtest must not crash when signal_df has no entries for some rebalances."""
    tickers = ["A", "B"]
    prices = _synthetic_prices(tickers, n_days=600)
    universe = _synthetic_universe(tickers)
    empty_signal = pd.DataFrame(columns=["date", "ticker", "expected_return"])
    out = run_backtest(
        prices, empty_signal, universe,
        optimizer="mv",
        risk_free_rate=0.05,
        settings={"bootstrap_enabled": False},
    )
    # Weights must be all-zero because no rebalance fired.
    assert (out["weights"].abs().sum().sum()) == 0.0
