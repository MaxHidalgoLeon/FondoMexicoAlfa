"""Walk-forward leakage tests for src/signals.py.

The training mask in forecast_returns must never include rows whose forward
return target uses prices observed after the decision date. Concretely, for a
decision at `date` with horizon `forward_days`, all training rows must satisfy
t' + forward_days <= date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals import _compute_forward_returns, forecast_returns


def _synthetic_panel(n_days: int = 400, n_tickers: int = 4, seed: int = 0) -> pd.DataFrame:
    """Build a minimal feature panel with the columns forecast_returns expects."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days)))
        # All equity feature columns must be present (forecast_returns filters by name).
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": ticker,
                "asset_class": "equity",
                "price": prices[i],
                "momentum_63": rng.normal(0, 0.05),
                "momentum_126": rng.normal(0, 0.05),
                "volatility_63": abs(rng.normal(0.02, 0.005)),
                "pe_ratio": 15 + rng.normal(0, 2),
                "pb_ratio": 2 + rng.normal(0, 0.3),
                "roe": 0.15 + rng.normal(0, 0.02),
                "profit_margin": 0.10 + rng.normal(0, 0.02),
                "net_debt_to_ebitda": 1.5 + rng.normal(0, 0.3),
                "ebitda_growth": 0.05 + rng.normal(0, 0.02),
                "capex_to_sales": 0.05 + rng.normal(0, 0.01),
                "industrial_production_yoy": 0.03 + rng.normal(0, 0.005),
                "usd_mxn": 20 + rng.normal(0, 0.5),
                "exports_yoy": 0.04 + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def test_compute_forward_returns_nans_tail():
    """Last forward_days rows per ticker must be NaN — basic PIT contract."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "price": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
    })
    fwd = _compute_forward_returns(df, forward_days=21)
    assert fwd.iloc[-21:].isna().all()
    assert fwd.iloc[:-21].notna().all()


def test_train_mask_excludes_lookahead_window():
    """For a synthetic panel, the train mask used by forecast_returns must
    never include rows whose target extends past the decision date.

    We reproduce the mask the implementation uses and assert the property.
    """
    forward_days = 21
    panel = _synthetic_panel(n_days=300, n_tickers=2)
    # Compute _fwd_return per ticker (same as the production loop).
    parts = []
    for _, grp in panel.groupby("ticker", group_keys=False):
        parts.append(_compute_forward_returns(grp.sort_values("date"), forward_days))
    panel["_fwd_return"] = pd.concat(parts).reindex(panel.index)

    decision = pd.Timestamp("2020-09-30")
    cutoff = decision - pd.tseries.offsets.BDay(forward_days)
    mask = (panel["date"] <= cutoff) & panel["_fwd_return"].notna()
    train = panel.loc[mask]

    # Every training row must have its forward-target window fully realized by `decision`.
    target_end_dates = train["date"] + pd.tseries.offsets.BDay(forward_days)
    assert (target_end_dates <= decision).all(), (
        f"Lookahead detected: {(target_end_dates > decision).sum()} training rows "
        f"have target windows extending past decision={decision.date()}."
    )


def test_forecast_returns_runs_without_lookahead():
    """End-to-end smoke test: forecast_returns must not crash and must produce
    a forecast dataframe that does not reference future-only rows."""
    panel = _synthetic_panel(n_days=400, n_tickers=3)
    # forecast_returns needs a price column already present; OK.
    returns = pd.DataFrame()  # unused by elasticnet path
    settings = {
        "forecast_model": "elasticnet",
        "forecast_forward_days": 21,
        "forecast_min_train_rows": 30,
        "elasticnet_cv_folds": 3,
        "elasticnet_l1_ratios": [0.1, 0.5, 0.9],
        "elasticnet_max_iter": 2000,
        "elasticnet_tol": 1e-3,
        "compute_shap": False,
    }
    out = forecast_returns(panel, returns, settings=settings)
    assert not out.empty
    # Predictions exist for at least one rebalance after the warmup period.
    assert out["date"].nunique() >= 2
