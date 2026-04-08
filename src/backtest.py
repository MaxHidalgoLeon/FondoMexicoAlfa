from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.covariance import LedoitWolf

from .portfolio import optimize_portfolio
from .risk import compute_cvar, max_drawdown, compute_sortino, compute_sharpe

logger = logging.getLogger(__name__)


def get_rebalance_dates(prices: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    """Return the last available trading date in each rebalancing period."""
    resampled = prices.resample(freq).last().dropna(how="all")
    # Snap to the nearest actual trading day that exists in prices.index
    snapped = []
    for dt in resampled.index:
        # Find the last trading day on or before dt
        candidates = prices.index[prices.index <= dt]
        if len(candidates):
            snapped.append(candidates[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def build_covariance_matrix(
    returns: pd.DataFrame, date: pd.Timestamp, window: int = 63
) -> pd.DataFrame:
    """Build a Ledoit-Wolf shrunk covariance matrix using at most `window` days."""
    subset = returns.loc[:date].tail(window).fillna(0.0)
    if subset.shape[0] < 10:
        # Fallback to identity-scaled matrix when data is scarce
        n = len(returns.columns)
        avg_var = returns.var().mean() if not returns.empty else 1e-4
        return pd.DataFrame(
            np.eye(n) * avg_var, index=returns.columns, columns=returns.columns
        )
    try:
        lw = LedoitWolf()
        lw.fit(subset)
        cov_array = lw.covariance_
        return pd.DataFrame(cov_array, index=subset.columns, columns=subset.columns).reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)
    except Exception:
        return subset.cov().reindex(
            index=returns.columns, columns=returns.columns
        ).fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    transaction_cost: float = 0.001,
    rebalance_freq: str = "ME",
) -> Dict[str, pd.DataFrame]:
    returns = prices.pct_change().fillna(0.0)
    rebalance_dates = get_rebalance_dates(prices, rebalance_freq)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    portfolio_returns = pd.Series(0.0, index=prices.index)
    prev_weights = pd.Series(0.0, index=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)

    for date in rebalance_dates:
        if date not in prices.index:
            continue
        date_signal = signal_df[signal_df["date"] == date]
        if date_signal.empty:
            logger.warning("No signal for rebalance date %s — carrying previous weights.", date)
            continue
        expected_returns = date_signal.set_index("ticker")["expected_return"].reindex(prices.columns).fillna(0.0)
        cov_matrix = build_covariance_matrix(returns, date)
        try:
            target_weights = optimize_portfolio(
                expected_returns,
                cov_matrix,
                prev_weights=prev_weights,
                max_position=0.15,
                min_position=0.0,
                target_net_exposure=0.9,
                risk_aversion=4.0,
                turnover_penalty=0.05,
            )
        except RuntimeError as exc:
            logger.warning("Optimization failed on %s: %s — carrying previous weights.", date, exc)
            target_weights = prev_weights
        weights.loc[date:, :] = target_weights.values
        turnover.loc[date] = np.sum(np.abs(target_weights - prev_weights))
        prev_weights = target_weights

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    transaction_costs = turnover * transaction_cost
    portfolio_returns = portfolio_returns - transaction_costs
    metrics = {
        "sharpe": compute_sharpe(portfolio_returns),
        "sortino": compute_sortino(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "cvar_95": compute_cvar(portfolio_returns, alpha=0.95),
        "annualized_return": ((1 + portfolio_returns).prod() ** (252 / len(portfolio_returns))) - 1,
        "annualized_vol": portfolio_returns.std() * np.sqrt(252),
        "turnover": turnover.mean(),
    }
    return {
        "weights": weights,
        "returns": portfolio_returns,
        "metrics": metrics,
        "turnover": turnover,
    }
