from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import genextreme, kstest
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / (excess.std(ddof=0) + 1e-9)


def compute_sortino(returns: pd.Series, required_return: float = 0.0) -> float:
    downside = returns[returns < required_return] - required_return  # excess loss below MAR
    # Lower partial moment (standard semideviation)
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 1 else 0.0
    excess_mean = returns.mean() - required_return
    return excess_mean / (downside_std + 1e-9) * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_var(returns: pd.Series, alpha: float = 0.95) -> float:
    return np.percentile(returns.dropna(), 100 * (1 - alpha))


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    var_level = compute_var(returns, alpha)
    tail = returns[returns <= var_level]
    if len(tail) == 0:
        return float(var_level)
    if len(tail) == 1:
        return float(tail.iloc[0])
    return float(tail.mean())


def stress_test(
    portfolio_returns: pd.Series,
    scenario_shocks: dict[str, float],
    exposures: dict[str, float],
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    scenario_results = []
    _base_return = portfolio_returns.mean()  # noqa: F841
    for label, shock in scenario_shocks.items():
        exposure = exposures.get(label, 0.0)
        adjusted = portfolio_returns + shock * exposure
        scenario_results.append(
            {
                "scenario": label,
                "mean_return": adjusted.mean(),
                "volatility": adjusted.std(ddof=0),
                "sharpe": compute_sharpe(adjusted, risk_free_rate=risk_free_rate),
                "max_drawdown": max_drawdown(adjusted),
            }
        )
    return pd.DataFrame(scenario_results)


def fit_garch(returns: pd.Series, model: str = "GJR") -> arch_model:
    """Fit a GARCH model to returns."""
    if model == "GARCH":
        mod = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
    elif model == "GJR":
        mod = arch_model(returns, vol="Garch", p=1, o=1, q=1, rescale=True)
    elif model == "EGARCH":
        mod = arch_model(returns, vol="EGarch", p=1, q=1, rescale=True)
    else:
        raise ValueError("Unsupported model")
    result = mod.fit(disp="off")
    if not result.convergence_flag == 0:
        logger.warning("GARCH (%s) did not converge (flag=%d).", model, result.convergence_flag)
    return result


def garch_forecast_vol(fitted_result, horizon: int = 21) -> float:
    """Forecast annualized volatility for given horizon."""
    forecast = fitted_result.forecast(horizon=horizon)
    vol = np.sqrt(forecast.variance.iloc[-1, -1] * 252 / horizon)
    return vol


def dynamic_var(returns: pd.Series, alpha: float = 0.95, method: str = "garch") -> pd.Series:
    """Rolling 1-day VaR."""
    if method == "garch":
        fitted = fit_garch(returns, "GJR")
        vol = fitted.conditional_volatility
        var = -vol * np.percentile(returns / (returns.std() + 1e-9), (1 - alpha) * 100)
    elif method == "empirical":
        var = returns.rolling(252).quantile(1 - alpha)
    else:
        raise ValueError("Unsupported method")
    return var


def monte_carlo_var(returns: pd.Series, alpha: float = 0.95, n_sim: int = 10000, horizon: int = 1) -> float:
    """Monte Carlo VaR using multivariate normal."""
    np.random.seed(42)
    cov = LedoitWolf().fit(returns.values.reshape(-1, 1)).covariance_
    mean = returns.mean()
    sim_returns = np.random.multivariate_normal([mean], cov, n_sim)
    var = np.percentile(sim_returns, (1 - alpha) * 100)
    return var


def gev_var(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """VaR and CVaR using GEV distribution on left tail."""
    tail = -returns[returns < 0]
    if len(tail) < 20:
        # Not enough tail observations — fall back to empirical
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    params = genextreme.fit(tail)
    # Goodness-of-fit check: reject GEV if K-S p-value < 0.05
    _ks_stat, ks_pval = kstest(tail, "genextreme", args=params)
    if ks_pval < 0.05:
        logger.warning(
            "GEV fit rejected by K-S test (p=%.4f); falling back to empirical quantile.", ks_pval
        )
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    var = float(genextreme.ppf(alpha, *params))
    x_tail = np.linspace(var, genextreme.ppf(0.9999, *params), 1000)
    cvar = float(np.trapz(x_tail * genextreme.pdf(x_tail, *params), x_tail) / (1 - alpha))
    return var, cvar


def duration_var(duration: float, dv01: float, rate_shock_std: float) -> float:
    """Parametric VaR for fixed income position."""
    return dv01 * rate_shock_std * 10000  # DV01 × rate shock in bps
