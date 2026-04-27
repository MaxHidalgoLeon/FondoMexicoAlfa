"""
Overfitting diagnostics for hyperparameter-tuned strategies.

Implements two metrics from Bailey & López de Prado (2014, 2016):

1. Deflated Sharpe Ratio (DSR)
   Adjusts a backtest's Sharpe Ratio for non-normality (skew, kurtosis), sample
   length, and the number of independent strategy configurations tried (N). Returns
   the probability that the observed SR is statistically distinguishable from
   the expected maximum SR under the null hypothesis of zero true Sharpe.

2. Probability of Backtest Overfitting (PBO)
   Combinatorially Symmetric Cross-Validation: split the trial OOS-Sharpe matrix
   into S equally-sized chunks, take every (S/2)-combination as IS, the rest as
   OOS. PBO is the fraction of splits where the IS-best trial ranks below the
   OOS median.

References
----------
Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.
    Journal of Portfolio Management, 40(5), 94–107.

Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2016).
    The Probability of Backtest Overfitting. Journal of Computational Finance,
    20(4), 39–69.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm


_EULER_MASCHERONI = 0.5772156649015328606


def expected_max_sharpe(n_trials: int) -> float:
    """E[max SR over N independent trials] under H0 of zero true Sharpe (per-period)."""
    if n_trials <= 1:
        return 0.0
    inv = 1.0 / n_trials
    z1 = norm.ppf(1.0 - inv)
    z2 = norm.ppf(1.0 - inv * math.exp(-1.0))
    return (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    annualization: int = 252,
    trial_sharpes: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute the Deflated Sharpe Ratio for a return series.

    Parameters
    ----------
    returns
        Daily (or per-period) return series of the selected strategy.
    n_trials
        Number of strategy configurations tried during selection (e.g., Optuna trials).
    annualization
        Periods per year for annualizing Sharpe (default 252 for daily).
    trial_sharpes
        Optional per-period Sharpe estimates for each of the N trials. When
        supplied, V (the variance of trial Sharpes) is set to their sample
        variance — this is Bailey-López de Prado's recommended specification.
        When omitted, V defaults to 1/(T-1) under the assumption that under
        H0 each SR_hat ~ N(0, 1/(T-1)).

    Returns
    -------
    dict with:
        sharpe_observed     : SR per-period (selected strategy)
        sharpe_annualized   : SR scaled to annual
        expected_max_sharpe : E[max SR | n_trials] per-period
        expected_max_sharpe_annual : annualized
        dsr_p_value         : Φ((SR_obs - SR_0) / sigma_SR_obs)
                              high values = low overfitting risk; >0.95 is strong evidence
        skew, kurtosis      : higher moments used in the variance correction
        n_obs               : T
    """
    arr = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = arr.size
    nan_payload = {
        "sharpe_observed": 0.0, "sharpe_annualized": 0.0,
        "expected_max_sharpe": 0.0, "expected_max_sharpe_annual": 0.0,
        "dsr_p_value": float("nan"), "skew": 0.0, "kurtosis": 3.0, "n_obs": n,
    }
    if n < 30:
        return nan_payload

    mean = arr.mean()
    sd = arr.std(ddof=1)
    if sd <= 0:
        return nan_payload

    sr = mean / sd
    z = (arr - mean) / sd
    g3 = (z ** 3).mean()
    g4 = (z ** 4).mean()

    # Variance of the SR estimator (Mertens 2002 / Lo 2002, non-normality corrected)
    var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / (n - 1)
    if var_sr <= 0 or not math.isfinite(var_sr):
        return {**nan_payload, "sharpe_observed": float(sr),
                "sharpe_annualized": float(sr * math.sqrt(annualization)),
                "skew": float(g3), "kurtosis": float(g4)}

    # E[max SR | N] = sqrt(V) * z_quantile, where V is the cross-trial variance
    # of SR estimates. Default V to 1/(T-1) — variance under H0 of zero true SR
    # for one estimator — when the user has not supplied per-trial Sharpes.
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        v_arr = np.asarray(trial_sharpes, dtype=float)
        v_arr = v_arr[np.isfinite(v_arr)]
        v = float(np.var(v_arr, ddof=1)) if v_arr.size >= 2 else 1.0 / max(n - 1, 1)
    else:
        v = 1.0 / max(n - 1, 1)

    z_quantile = expected_max_sharpe(int(n_trials))
    sr0 = math.sqrt(max(v, 0.0)) * z_quantile

    dsr_z = (sr - sr0) / math.sqrt(var_sr)
    p = float(norm.cdf(dsr_z))

    return {
        "sharpe_observed": float(sr),
        "sharpe_annualized": float(sr * math.sqrt(annualization)),
        "expected_max_sharpe": float(sr0),
        "expected_max_sharpe_annual": float(sr0 * math.sqrt(annualization)),
        "dsr_p_value": p,
        "skew": float(g3),
        "kurtosis": float(g4),
        "n_obs": int(n),
    }


def _logit(x: float) -> float:
    eps = 1e-9
    x = min(max(x, eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def probability_of_backtest_overfitting(
    trial_metric_matrix: np.ndarray,
    n_chunks: int = 14,
) -> dict[str, float]:
    """
    Combinatorially Symmetric Cross-Validation (CSCV) PBO.

    Parameters
    ----------
    trial_metric_matrix
        2-D array of shape (T_chunks, N_trials) of OOS performance metrics
        (e.g., per-period Sharpe ratios). Rows are time slices; columns are
        trial configurations. If only fold-level metrics are available
        (F folds × N trials), pass that — n_chunks will be clamped to F.
    n_chunks
        Target number of equally-sized time chunks (S in López de Prado).
        Must be even and ≤ rows of the matrix. Larger S → finer resolution
        but more combinations. Default 14 → C(14,7) = 3432 splits.

    Returns
    -------
    dict with:
        pbo                : fraction of splits where IS-best logit(λ) < 0
        median_logit_lambda: median relative-rank logit across splits
        n_splits           : number of (S/2)-combinations evaluated
        n_trials           : N
    """
    M = np.asarray(trial_metric_matrix, dtype=float)
    if M.ndim != 2 or M.size == 0:
        return {"pbo": float("nan"), "median_logit_lambda": float("nan"),
                "n_splits": 0, "n_trials": 0}

    n_rows, n_trials = M.shape
    if n_trials < 4 or n_rows < 2:
        return {"pbo": float("nan"), "median_logit_lambda": float("nan"),
                "n_splits": 0, "n_trials": int(n_trials)}

    s = min(int(n_chunks), n_rows)
    if s % 2 == 1:
        s -= 1
    if s < 2:
        return {"pbo": float("nan"), "median_logit_lambda": float("nan"),
                "n_splits": 0, "n_trials": int(n_trials)}

    # Aggregate rows into S contiguous chunks if rows > S
    if n_rows > s:
        bin_edges = np.linspace(0, n_rows, s + 1, dtype=int)
        chunked = np.vstack([
            M[bin_edges[i]:bin_edges[i + 1], :].mean(axis=0) for i in range(s)
        ])
    else:
        chunked = M

    half = s // 2
    chunk_idx = list(range(s))
    logits: list[float] = []
    for is_combo in combinations(chunk_idx, half):
        is_set = set(is_combo)
        oos_set = [i for i in chunk_idx if i not in is_set]

        is_perf = chunked[list(is_set), :].mean(axis=0)
        oos_perf = chunked[oos_set, :].mean(axis=0)

        j_star = int(np.argmax(is_perf))
        # Relative rank of j_star in the OOS distribution
        oos_ranks = pd.Series(oos_perf).rank(method="average").to_numpy()
        rel_rank = oos_ranks[j_star] / (n_trials + 1)
        logits.append(_logit(float(rel_rank)))

    arr = np.asarray(logits)
    pbo = float((arr < 0).mean())
    return {
        "pbo": pbo,
        "median_logit_lambda": float(np.median(arr)),
        "n_splits": int(arr.size),
        "n_trials": int(n_trials),
    }


__all__ = [
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
