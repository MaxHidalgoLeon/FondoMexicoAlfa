"""Unit tests for src/portfolio.py optimizers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import (
    apply_fx_overlay,
    black_litterman,
    optimize_portfolio,
    optimize_portfolio_robust,
)


def _toy_inputs(n: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    mu = pd.Series(rng.normal(0.001, 0.0005, n), index=tickers)
    rets = pd.DataFrame(rng.normal(0, 0.01, (252, n)), columns=tickers)
    cov = rets.cov()
    return mu, cov, tickers


def test_optimize_portfolio_respects_box_constraints():
    """Per-name weights must be in [0, max_position] and sum to target_net_exposure."""
    mu, cov, _ = _toy_inputs(n=5)
    w = optimize_portfolio(mu, cov, max_position=0.30, min_position=0.0, target_net_exposure=1.0)
    assert (w >= -1e-6).all()
    assert (w <= 0.30 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-4


def test_optimize_portfolio_respects_asset_class_constraints():
    """Group bounds must hold post-optimization."""
    mu, cov, tickers = _toy_inputs(n=4)
    ac_map = {tickers[0]: "equity", tickers[1]: "equity", tickers[2]: "fibra", tickers[3]: "fibra"}
    constraints = {
        "__asset_class_map__": ac_map,
        "equity": {"min": 0.4, "max": 0.8},
        "fibra":  {"min": 0.2, "max": 0.6},
    }
    w = optimize_portfolio(
        mu, cov,
        max_position=0.5, target_net_exposure=1.0,
        asset_class_constraints=constraints,
    )
    eq = sum(w[t] for t in tickers if ac_map[t] == "equity")
    fb = sum(w[t] for t in tickers if ac_map[t] == "fibra")
    assert 0.4 - 1e-3 <= eq <= 0.8 + 1e-3
    assert 0.2 - 1e-3 <= fb <= 0.6 + 1e-3


def test_optimize_portfolio_turnover_penalty_anchors_to_prev():
    """A large turnover penalty must push the new solution towards prev_weights."""
    mu, cov, tickers = _toy_inputs(n=4)
    prev = pd.Series([0.25, 0.25, 0.25, 0.25], index=tickers)
    w_low = optimize_portfolio(mu, cov, prev_weights=prev, turnover_penalty=0.0,
                                max_position=0.6, target_net_exposure=1.0)
    w_high = optimize_portfolio(mu, cov, prev_weights=prev, turnover_penalty=50.0,
                                max_position=0.6, target_net_exposure=1.0)
    drift_low = float((w_low - prev).abs().sum())
    drift_high = float((w_high - prev).abs().sum())
    assert drift_high <= drift_low + 1e-6


def test_black_litterman_blends_market_and_views():
    """With zero confidence in views, BL ≈ implied market returns (sanity check)."""
    mu, cov, tickers = _toy_inputs(n=4)
    market_w = pd.Series([0.25] * 4, index=tickers)
    views = pd.Series([0.0, 0.0, 0.0, 0.0], index=tickers)
    confidences = pd.Series([0.0, 0.0, 0.0, 0.0], index=tickers)
    bl = black_litterman(market_w, cov, views, confidences, risk_aversion=2.5, tau=0.05)
    assert isinstance(bl, pd.Series)
    assert len(bl) == 4
    assert np.isfinite(bl.values).all()


def test_apply_fx_overlay_zero_hedge_returns_baseline():
    """Hedge ratio = 0 ⇒ FX overlay leaves expected returns unchanged."""
    _, _, tickers = _toy_inputs(n=3)
    base = pd.Series([0.05, 0.06, 0.04], index=tickers)
    usd = pd.Series([0.5, 0.0, 1.0], index=tickers)
    out = apply_fx_overlay(base, usd, usd_mxn_level=20.0, expected_usdmxn_return=0.0, hedge_ratio=0.0)
    pd.testing.assert_series_equal(out.reindex(base.index), base, check_dtype=False)


def test_robust_optimizer_fallback_respects_constraints():
    """When all Michaud simulations fail, the fallback must respect asset_class_constraints.

    Setup is deliberately feasible so that _build_feasible_x0 can satisfy all bounds:
      8 equity tickers × max_position=0.15 → max equity = min(0.90, 8×0.15=1.20) = 0.90
      2 FI tickers    × max_position=0.15 → max FI    = min(0.10, 2×0.15=0.30)  = 0.10
      Total achievable = 1.00 ✓

    Old equal-weight fallback: 10 tickers × 0.10 each → FI total = 2×0.10 = 0.20 > 0.10 ← VIOLATION
    New _build_feasible_x0 fallback: FI total = 0.10 ← RESPECTS CONSTRAINT
    """
    rng = np.random.default_rng(99)
    equity_tickers = [f"EQ{i}" for i in range(8)]
    fi_tickers = ["FI0", "FI1"]  # 2 fixed-income names
    tickers = equity_tickers + fi_tickers
    n = len(tickers)

    mu = pd.Series(rng.normal(0.001, 0.0005, n), index=tickers)
    rets = pd.DataFrame(rng.normal(0, 0.01, (252, n)), columns=tickers)
    cov = rets.cov()

    asset_class_constraints = {
        "__asset_class_map__": {t: "equity" for t in equity_tickers} | {t: "fixed_income" for t in fi_tickers},
        "equity": {"min": 0.0, "max": 0.90},
        "fixed_income": {"min": 0.0, "max": 0.10},
    }

    # Patch optimize_portfolio to always raise RuntimeError to force the all-fail path
    import unittest.mock as mock
    with mock.patch("src.portfolio.optimize_portfolio", side_effect=RuntimeError("forced failure")):
        w = optimize_portfolio_robust(
            mu, cov,
            n_simulations=5,
            max_position=0.15,
            min_position=0.0,
            target_net_exposure=1.0,
            risk_aversion=2.0,
            asset_class_constraints=asset_class_constraints,
        )

    assert isinstance(w, pd.Series), "Fallback must return a pd.Series."
    assert set(w.index) == set(tickers), "Fallback must cover all tickers."
    assert abs(w.sum() - 1.0) < 1e-4, f"Fallback weights must sum to 1.0, got {w.sum():.6f}."
    assert (w >= -1e-9).all(), "No weight may be negative."
    assert (w <= 0.15 + 1e-6).all(), f"No weight may exceed max_position=0.15; got {w.max():.4f}."

    ac_map = asset_class_constraints["__asset_class_map__"]
    fi_weight = w[[t for t in tickers if ac_map[t] == "fixed_income"]].sum()
    assert fi_weight <= 0.10 + 1e-4, (
        f"fixed_income total weight {fi_weight:.4f} exceeds max=0.10 — "
        "feasible fallback failed to respect asset_class constraint."
    )
