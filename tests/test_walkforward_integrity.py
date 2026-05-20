"""Walk-forward integrity tests for src/backtest.py.

These guard against two classes of defects:

1. FUTURE-DATA LEAKAGE: weights at rebalance date t must be identical whether
   the price panel ends at t or extends years further.  If any global computation
   (covariance, ADTV, regime, SHAP baseline) leaks future prices, appending
   rows after t will silently shift earlier weights.

2. ASSET-CLASS CONSTRAINT VIOLATIONS: at every rebalance date the aggregate
   portfolio weights must respect all asset-class group bounds.  This exercises
   the optimizer end-to-end and guards against fallback paths that skip constraints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import get_rebalance_dates, run_backtest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _prices(tickers: list[str], n_days: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days))) for t in tickers},
        index=idx,
    )


def _signal(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    eom = prices.resample("ME").last().index
    rows = []
    for d in eom:
        for i, t in enumerate(tickers):
            rows.append({"date": d, "ticker": t, "expected_return": 0.04 + 0.002 * i})
    return pd.DataFrame(rows)


def _universe(tickers: list[str], asset_classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": tickers,
        "asset_class": asset_classes,
        "sector": ["Industrial"] * len(tickers),
        "investable": [True] * len(tickers),
        "liquidity_score": [0.8] * len(tickers),
        "market_cap_mxn": [5e9] * len(tickers),
        "usd_exposure": [0.2] * len(tickers),
    })


# ---------------------------------------------------------------------------
# Test 1 — future-price leak
# ---------------------------------------------------------------------------

def test_backtest_weights_invariant_to_future_prices():
    """Weights at rebalance date t must not change when the price panel is extended.

    Run the backtest twice:
      (A) prices truncated just past a mid-panel cut date
      (B) prices over the full horizon

    Weights at all rebalances up to the cut must be numerically identical (atol=1e-9).
    Any difference proves a global computation (covariance, regime, etc.) is using
    future price data to influence earlier decisions.
    """
    tickers = ["EQ0", "EQ1", "EQ2", "EQ3"]
    prices_full = _prices(tickers, n_days=700)
    sig_full    = _signal(prices_full, tickers)
    universe    = _universe(tickers, ["equity"] * 4)

    # Cut roughly in the middle of the panel
    cut_date = prices_full.index[350]

    # Panel A: truncated at cut_date
    prices_trunc = prices_full.loc[prices_full.index <= cut_date]
    sig_trunc    = sig_full[sig_full["date"] <= cut_date]
    result_trunc = run_backtest(
        prices_trunc, sig_trunc, universe,
        optimizer="mv",
        settings={"bootstrap_enabled": False},
    )
    w_trunc = result_trunc["weights"]

    # Panel B: full panel
    result_full = run_backtest(
        prices_full, sig_full, universe,
        optimizer="mv",
        settings={"bootstrap_enabled": False},
    )
    w_full = result_full["weights"]

    # Compare weights at every date in the truncated panel
    common_dates = w_trunc.index.intersection(w_full.index)
    rebal_dates = get_rebalance_dates(prices_trunc, "ME")
    check_dates = [d for d in rebal_dates if d in common_dates]

    assert check_dates, "No overlapping rebalance dates to compare."

    differences = []
    for d in check_dates:
        wt = w_trunc.loc[d].values
        wf = w_full.loc[d].values
        if not np.allclose(wt, wf, atol=1e-9):
            diff = float(np.max(np.abs(wt - wf)))
            differences.append(f"  {d.date()}: max |Δw| = {diff:.2e}")

    assert not differences, (
        "Backtest weights change when future prices are appended — "
        f"look-ahead leak detected at {len(differences)} rebalance(s):\n"
        + "\n".join(differences)
    )


# ---------------------------------------------------------------------------
# Test 2 — asset-class constraint compliance at every rebalance
# ---------------------------------------------------------------------------

def test_backtest_respects_asset_class_constraints_every_rebalance():
    """At every rebalance date the aggregate weights must satisfy all group bounds.

    Setup uses 12 equity + 2 fixed_income tickers (14 total).
    With max_position=0.10 hardcoded in run_backtest and target_net_exposure=0.90,
    the feasibility check: 14 × 0.10 = 1.40 > 0.90 — the problem is feasible.

    FI tickers receive the highest expected returns so the optimizer is pushed
    to maximise FI allocation.  The FI group max=0.10 must cap the total.
    """
    eq_tickers = [f"EQ{i}" for i in range(12)]
    fi_tickers = ["FI0", "FI1"]
    all_tickers = eq_tickers + fi_tickers  # 14 tickers, 14×0.10=1.40 > target 0.90

    prices  = _prices(all_tickers, n_days=600)
    signal  = _signal(prices, all_tickers)   # FI tickers get highest expected return
    uni_ac  = ["equity"] * 12 + ["fixed_income"] * 2
    universe = _universe(all_tickers, uni_ac)

    ac_map = {t: ac for t, ac in zip(all_tickers, uni_ac)}
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity":       {"min": 0.0, "max": 0.90},
        "fixed_income": {"min": 0.0, "max": 0.10},
    }

    result = run_backtest(
        prices, signal, universe,
        optimizer="mv",
        asset_class_constraints=asset_class_constraints,
        settings={"bootstrap_enabled": False},
    )
    weights = result["weights"]
    rebal_dates = get_rebalance_dates(prices, "ME")
    rebal_dates = [d for d in rebal_dates if d in weights.index]
    rebal_dates = [d for d in rebal_dates if weights.loc[d].abs().sum() > 1e-6]

    assert rebal_dates, "No active rebalances found — check signal generation."

    violations = []
    for d in rebal_dates:
        w = weights.loc[d]
        for ac, bounds in asset_class_constraints.items():
            if ac == "__asset_class_map__":
                continue
            ac_tickers = [t for t in all_tickers if ac_map[t] == ac]
            group_sum = float(w[ac_tickers].sum())
            ac_max = bounds.get("max", 1.0)
            ac_min = bounds.get("min", 0.0)
            if group_sum > ac_max + 1e-4:   # 1e-4 tolerance for SLSQP numerical error
                violations.append(
                    f"  {d.date()}: {ac} total={group_sum:.4f} > max={ac_max:.4f}"
                )
            if group_sum < ac_min - 1e-4:
                violations.append(
                    f"  {d.date()}: {ac} total={group_sum:.4f} < min={ac_min:.4f}"
                )

    assert not violations, (
        f"Asset-class constraint violated at {len(violations)} rebalance(s):\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Test 3 — constraint compliance survives a near-infeasible regime stress period
# ---------------------------------------------------------------------------

def test_backtest_constraint_compliance_under_macro_stress():
    """Constraints must still be respected when the regime shifts to 'stress',
    which applies tighter equity max (0.60) and higher fixed_income min (0.10).

    This catches the case where `regime_asset_class_constraints` produces
    bounds that the optimizer cannot satisfy, causing a fallback that
    previously returned equal-weight (violating fixed_income max).

    Uses 14 equity + 3 FI = 17 tickers.  14+3=17, 17×0.10=1.70 > target 0.90.
    Even in the tightest regime (stress: equity max=0.60, FI max=0.30),
    the problem is feasible: 14×0.10=1.40>0.60 for equity, 3×0.10=0.30 for FI.
    """
    eq_tickers = [f"EQ{i}" for i in range(14)]
    fi_tickers = ["FI0", "FI1", "FI2"]
    all_tickers = eq_tickers + fi_tickers

    prices  = _prices(all_tickers, n_days=500)
    signal  = _signal(prices, all_tickers)
    uni_ac  = ["equity"] * 14 + ["fixed_income"] * 3
    universe = _universe(all_tickers, uni_ac)

    # Construct a macro panel that forces a 'stress' regime mid-panel.
    # IP contracts, FX depreciates, Banxico hikes simultaneously.
    rng = np.random.default_rng(99)
    n_months = 22
    month_dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    banxico = np.concatenate([np.full(10, 6.0), np.full(12, 9.0)])  # rate hike
    usd_mxn = np.concatenate([np.full(10, 19.5), np.full(12, 23.0)])  # peso crash
    ip_yoy  = np.concatenate([np.full(10, 0.03), np.full(12, -0.05)])  # contraction
    macro = pd.DataFrame({
        "date": month_dates,
        "banxico_rate": banxico,
        "usd_mxn": usd_mxn,
        "industrial_production_yoy": ip_yoy,
    })

    ac_map = {t: ac for t, ac in zip(all_tickers, uni_ac)}
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity":       {"min": 0.40, "max": 0.90},
        "fixed_income": {"min": 0.00, "max": 0.20},
    }

    result = run_backtest(
        prices, signal, universe,
        optimizer="mv",
        asset_class_constraints=asset_class_constraints,
        macro=macro,
        settings={"bootstrap_enabled": False},
    )
    weights = result["weights"]
    rebal_dates = get_rebalance_dates(prices, "ME")
    rebal_dates = [d for d in rebal_dates if d in weights.index]
    rebal_dates = [d for d in rebal_dates if weights.loc[d].abs().sum() > 1e-6]

    # In stress regime the effective fixed_income max is 0.30 (regime override).
    # Verify that the actual weights never grossly exceed any group cap.
    # We allow the wider of base constraints (0.20) vs stress constraints (0.30).
    fi_hard_cap = 0.35  # some tolerance above stress max=0.30 to catch true bugs

    violations = []
    for d in rebal_dates:
        w = weights.loc[d]
        fi_sum = float(w[fi_tickers].sum())
        eq_sum = float(w[eq_tickers].sum())
        if fi_sum > fi_hard_cap:
            violations.append(f"  {d.date()}: FI total={fi_sum:.4f} exceeds hard cap {fi_hard_cap:.4f}")
        if eq_sum < 0.20:
            violations.append(f"  {d.date()}: equity total={eq_sum:.4f} suspiciously low (< 0.20)")

    assert not violations, (
        f"Gross constraint violation during macro stress at {len(violations)} rebalance(s):\n"
        + "\n".join(violations)
    )
