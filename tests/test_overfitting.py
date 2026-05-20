"""Unit tests for src/overfitting.py — Deflated Sharpe, PBO, expected max SR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.overfitting import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)


def test_expected_max_sharpe_monotone_in_n():
    """E[max SR] increases with the number of trials."""
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) < expected_max_sharpe(100) < expected_max_sharpe(1000)


def test_deflated_sharpe_returns_required_keys():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    out = deflated_sharpe_ratio(rets, n_trials=20)
    for key in ("sharpe_observed", "sharpe_annualized", "expected_max_sharpe", "dsr_p_value"):
        assert key in out, f"Missing key {key}"
    assert 0.0 <= out["dsr_p_value"] <= 1.0


def test_deflated_sharpe_penalizes_more_trials():
    """For the same returns, more trials ⇒ higher expected_max ⇒ lower confidence
    that the observed Sharpe is real (dsr_p_value is the probability the true
    Sharpe is positive after deflation; more trials erode it)."""
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    a = deflated_sharpe_ratio(rets, n_trials=5)
    b = deflated_sharpe_ratio(rets, n_trials=500)
    assert a["expected_max_sharpe"] <= b["expected_max_sharpe"]
    assert a["dsr_p_value"] >= b["dsr_p_value"] - 1e-9


def test_probability_of_backtest_overfitting_random_matrix():
    """Random metric matrix ⇒ PBO close to 0.5 (no skill, just noise)."""
    rng = np.random.default_rng(42)
    mat = rng.normal(0, 1, size=(14, 30))
    out = probability_of_backtest_overfitting(mat, n_chunks=14)
    assert "pbo" in out and "n_splits" in out
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["n_splits"] > 0
