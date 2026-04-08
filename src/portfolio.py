from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _portfolio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    turnover_penalty: float,
    prev_weights: np.ndarray,
) -> float:
    expected_cost = -weights.dot(expected_returns)
    variance = risk_aversion * weights.dot(cov_matrix).dot(weights)
    # Scale turnover penalty by expected return magnitude so it doesn't dominate
    er_scale = max(np.abs(expected_returns).mean(), 1e-4)
    turnover = turnover_penalty * er_scale * np.sum(np.abs(weights - prev_weights))
    return expected_cost + variance + turnover


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.15,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.1,
    asset_class_constraints: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    tickers = expected_returns.index.tolist()
    if prev_weights is None:
        prev_weights = pd.Series(0.0, index=tickers)

    x0 = np.repeat(target_net_exposure / len(tickers), len(tickers))
    bounds = [(min_position, max_position)] * len(tickers)
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - target_net_exposure,
        }
    ]

    # Asset-class group constraints
    if asset_class_constraints:
        # asset_class_constraints: {"equity": {"min": 0.4, "max": 0.7}, ...}
        # Requires expected_returns.index to have an associated asset_class_map passed
        # via the name attribute or a separate mapping.  We accept a parallel dict
        # under the special key "__asset_class_map__" if present.
        ac_map: Dict[str, str] = asset_class_constraints.pop("__asset_class_map__", {})
        if ac_map:
            for ac, bounds_dict in asset_class_constraints.items():
                ac_tickers = [t for t in tickers if ac_map.get(t) == ac]
                if not ac_tickers:
                    continue
                idx = [tickers.index(t) for t in ac_tickers]
                ac_min = bounds_dict.get("min", 0.0)
                ac_max = bounds_dict.get("max", 1.0)
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mn=ac_min: np.sum(x[i]) - mn}
                )
                constraints.append(
                    {"type": "ineq", "fun": lambda x, i=idx, mx=ac_max: mx - np.sum(x[i])}
                )

    result = minimize(
        _portfolio_objective,
        x0,
        args=(expected_returns.values, cov_matrix.values, risk_aversion, turnover_penalty, prev_weights.values),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        logger.warning("Portfolio optimization did not converge: %s", result.message)
        if np.any(np.isnan(result.x)):
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return pd.Series(result.x, index=tickers)


def black_litterman(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    views: Dict[str, float],
    view_confidences: Dict[str, float],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected returns."""
    pi = risk_aversion * cov_matrix.dot(market_weights)  # CAPM equilibrium returns
    P = np.zeros((len(views), len(market_weights)))
    Q = np.zeros(len(views))
    omega = np.zeros((len(views), len(views)))

    for i, (ticker, view) in enumerate(views.items()):
        if ticker not in market_weights.index:
            continue
        P[i, market_weights.index.get_loc(ticker)] = 1
        Q[i] = view
        conf = max(view_confidences.get(ticker, 0.5), 1e-6)
        omega[i, i] = (1.0 / conf) * tau

    # Robustly regularize omega before solving the linear system
    omega += np.eye(len(views)) * 1e-6

    cov_arr = cov_matrix.values
    # BL formula using scipy.linalg.solve for numerical stability
    # Posterior = pi + tau*Sigma*P' * inv(P*tau*Sigma*P' + Omega) * (Q - P*pi)
    M = P.dot(tau * cov_arr).dot(P.T) + omega  # (K x K)
    rhs = Q - P.dot(pi.values)
    try:
        adjustment = tau * cov_arr.dot(P.T).dot(linalg.solve(M, rhs))
    except linalg.LinAlgError:
        logger.warning("Black-Litterman linear solve failed; returning CAPM prior.")
        return pi

    pi_bl = pi + adjustment
    return pi_bl


def apply_fx_overlay(
    expected_returns: pd.Series,
    usd_exposure: pd.Series,
    usd_mxn_level: float,
    expected_usdmxn_return: float,
    hedge_ratio: float = 0.5,
) -> pd.Series:
    """Apply FX overlay to expected returns."""
    fx_adjustment = usd_exposure * (1 - hedge_ratio) * expected_usdmxn_return
    adjusted = expected_returns + fx_adjustment
    return adjusted
