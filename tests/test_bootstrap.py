"""Unit tests for src/bootstrap.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_block_size_selector, bootstrap_metric


def _ar1_series(n: int = 1000, phi: float = 0.6, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0, 0.01, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return pd.Series(x)


def test_block_size_selector_positive_and_capped():
    """Block size must be a positive int within the [5, 60] cap or the fallback 20."""
    iid = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    b = bootstrap_block_size_selector(iid)
    assert isinstance(b, int)
    assert 1 <= b <= 60


def test_block_size_selector_short_series_fallback():
    """< 10 observations falls back to the default of 20."""
    short = pd.Series(np.linspace(0, 0.05, 5))
    assert bootstrap_block_size_selector(short) == 20


def test_block_size_grows_with_autocorrelation():
    """Higher AR(1) coefficient ⇒ longer optimal block (or hits the cap)."""
    low = bootstrap_block_size_selector(_ar1_series(phi=0.05, seed=1))
    high = bootstrap_block_size_selector(_ar1_series(phi=0.9, seed=1))
    assert high >= low


def test_bootstrap_metric_is_deterministic_with_seed():
    """Same seed ⇒ identical CI, distribution, point."""
    s = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    a = bootstrap_metric(s, lambda x: float(x.mean()), block_size=10, n_reps=200, seed=42)
    b = bootstrap_metric(s, lambda x: float(x.mean()), block_size=10, n_reps=200, seed=42)
    assert a["point"] == b["point"]
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]
    np.testing.assert_array_equal(a["distribution"], b["distribution"])


def test_bootstrap_metric_ci_brackets_point():
    """The point estimate falls inside (or at the boundary of) the 95% CI."""
    s = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    res = bootstrap_metric(s, lambda x: float(x.mean()), block_size=10, n_reps=300, seed=42)
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert res["se"] >= 0
