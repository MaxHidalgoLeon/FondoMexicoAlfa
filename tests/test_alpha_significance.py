"""Smoke tests for src/alpha_significance.py public API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_significance import compute_benchmark_alpha_significance


def _two_correlated_series(n: int = 400, alpha_per_day: float = 0.0003, beta: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    benchmark = pd.Series(rng.normal(0, 0.01, n))
    noise = pd.Series(rng.normal(0, 0.005, n))
    fund = alpha_per_day + beta * benchmark + noise
    return fund, benchmark


def test_compute_benchmark_alpha_significance_returns_per_benchmark():
    fund, bench = _two_correlated_series(beta=1.2, alpha_per_day=0.0005)
    bench_df = pd.DataFrame({"IPC": bench})
    out = compute_benchmark_alpha_significance(
        fund, bench_df,
        settings={
            "bootstrap_enabled": True,
            "bootstrap_n_reps": 50,
            "bootstrap_block_size": 5,
            "bootstrap_confidence": 0.95,
            "bootstrap_seed": 42,
        },
        risk_free_rate=0.04,
    )
    assert "IPC" in out
    entry = out["IPC"]
    assert "alpha_annualized" in entry
    assert "information_ratio" in entry
    assert "tracking_error" in entry
    assert isinstance(entry["beta"], float)
    assert 0.5 < entry["beta"] < 2.0  # beta=1.2 with noise should land in this range


def test_compute_benchmark_alpha_significance_empty_benchmark_returns_empty():
    fund, _ = _two_correlated_series()
    out = compute_benchmark_alpha_significance(
        fund, pd.DataFrame(),
        settings={"bootstrap_enabled": True},
    )
    assert out == {}


def test_compute_benchmark_alpha_significance_disabled_returns_empty():
    fund, bench = _two_correlated_series()
    out = compute_benchmark_alpha_significance(
        fund, pd.DataFrame({"IPC": bench}),
        settings={"bootstrap_enabled": False},
    )
    assert out == {}
