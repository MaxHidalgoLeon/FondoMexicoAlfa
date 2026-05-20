"""Smoke tests for src/signal_diagnostics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal_diagnostics import SIGNAL_COLUMNS, compute_signal_ic_diagnostics


def _synthetic_feature_df(n_tickers: int = 5, n_dates: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for t in tickers:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_dates)))
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": t,
                "price": prices[i],
                "momentum_63": rng.normal(0, 0.05),
                "momentum_126": rng.normal(0, 0.05),
                "volatility_63": abs(rng.normal(0.02, 0.005)),
                "value_score": rng.normal(0, 1),
                "quality_score": rng.normal(0, 1),
                "liquidity_score": rng.uniform(0, 1),
            })
    return pd.DataFrame(rows)


def test_signal_ic_diagnostics_empty_input():
    out = compute_signal_ic_diagnostics(pd.DataFrame())
    assert out == {}


def test_signal_ic_diagnostics_returns_per_signal_dict():
    feature_df = _synthetic_feature_df()
    out = compute_signal_ic_diagnostics(
        feature_df,
        settings={"ic_diagnostics_enabled": True, "bootstrap_n_reps": 50, "ic_bootstrap_block_size": 4},
    )
    assert isinstance(out, dict)
    # At least one of the known signal columns should produce an IC entry.
    intersect = set(out.keys()) & set(SIGNAL_COLUMNS)
    assert intersect, f"No diagnostics produced for any known signal column: {out.keys()}"
    # Each entry contains the headline IC fields.
    sample = next(iter(out.values()))
    assert "ic_mean" in sample
    assert np.isfinite(sample["ic_mean"]) or np.isnan(sample["ic_mean"])


def test_signal_ic_diagnostics_disabled_returns_empty():
    feature_df = _synthetic_feature_df()
    out = compute_signal_ic_diagnostics(feature_df, settings={"ic_diagnostics_enabled": False})
    assert out == {}
