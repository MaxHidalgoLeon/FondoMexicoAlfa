"""
Hyperparameter optimization for the FMIA pipeline via Bayesian search (Optuna).

Optimizes a configurable subset of pipeline hyperparameters using
walk-forward cross-validation with a purging gap to prevent data leakage.

Regulatory parameters (CNBV 10% position / issuer limits, FX overlay cap,
liquidity sleeve per regime) are intentionally excluded from the search
space and remain fixed at their prospectus-compliant values.

References
----------
Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter
    Optimization Framework*. KDD 2019.
Bergstra & Bengio (2012). *Random Search for Hyper-Parameter
    Optimization*. JMLR 13.
López de Prado (2018). *Advances in Financial Machine Learning*.
    Chapter 12: Cross-Validation in Finance (purged k-fold).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .settings import DEFAULT_SETTINGS, resolve_settings
from .signals import forecast_returns, score_cross_section

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Search space: (distribution, lower, upper, log_scale)
# `categorical` uses the second element as the list of choices; the other
# fields are unused.
# ----------------------------------------------------------------------
DEFAULT_SEARCH_SPACE: dict[str, tuple] = {
    # Black-Litterman
    "bl_risk_aversion":     ("float", 1.0,  5.0,  True),
    "bl_tau":               ("float", 0.01, 0.15, True),
    # MV optimizer
    "mv_risk_aversion":     ("float", 1.0,  10.0, True),
    "mv_turnover_penalty":  ("float", 0.01, 0.5,  True),
    "mv_market_impact_eta": ("float", 0.01, 0.5,  True),
    # CVaR optimizer
    "cvar_risk_aversion":   ("float", 5.0,  50.0, True),
    # EWMA covariance
    "ewma_lambda_cov":      ("float", 0.90, 0.99, False),
    # ElasticNet forecast
    "elasticnet_l1_ratios": ("categorical", [[0.1, 0.5, 0.9], [0.1, 0.9], [0.5], [0.9]], None, None),
    # Regime detector
    "regime_ewma_span":     ("int",   3,    12,   False),
    # Forecast horizon
    "forecast_forward_days": ("int",  10,   42,   False),
    # ADTV
    "adtv_ewma_lambda":     ("float", 0.90, 0.99, False),
}

# Keys forbidden from the search space — fixed by regulation or prospectus.
REGULATORY_FIXED_KEYS = {
    "max_position",
    "max_position_mv",
    "max_position_cvar",
    "max_position_robust",
    "issuer_concentration_limit",
    "fx_overlay_notional_cap",
    "liquidity_sleeve_expansion",
    "liquidity_sleeve_tightening",
    "liquidity_sleeve_stress",
}


@dataclass
class FoldData:
    """Single walk-forward fold."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_prices: pd.DataFrame
    test_prices: pd.DataFrame
    feature_df: pd.DataFrame
    universe: pd.DataFrame
    macro: pd.DataFrame


@dataclass
class OptimResult:
    """Result of a hyperparameter optimization study."""

    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float = float("nan")
    trial_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    n_trials_completed: int = 0
    optimization_time_seconds: float = 0.0
    search_space: dict[str, tuple] = field(default_factory=dict)
    objective_metric: str = "sharpe_adj"
    turnover_penalty: float = 0.5


def build_walk_forward_folds(
    prices: pd.DataFrame,
    feature_df: pd.DataFrame,
    universe: pd.DataFrame,
    macro: pd.DataFrame,
    n_folds: int = 3,
    purge_gap_days: int = 21,
    min_train_days: int = 252,
) -> list[FoldData]:
    """
    Build n_folds walk-forward folds with a purge gap between train and test.

    Each fold's train window is expanding, and the corresponding test window
    starts `purge_gap_days` business days after the train end. This prevents
    leakage of forward-labeled features into the validation slice.
    """
    if prices is None or prices.empty:
        return []

    all_dates = pd.DatetimeIndex(sorted(prices.index.unique()))
    n_dates = len(all_dates)
    if n_dates < min_train_days + purge_gap_days + 30:
        logger.warning(
            "Insufficient price history for %d folds (need ≥%d days, got %d).",
            n_folds, min_train_days + purge_gap_days + 30, n_dates,
        )
        return []

    usable = n_dates - min_train_days - purge_gap_days
    test_block = max(usable // n_folds, 30)
    folds: list[FoldData] = []
    for k in range(n_folds):
        train_end_idx = min_train_days + k * test_block
        test_start_idx = train_end_idx + purge_gap_days
        test_end_idx = min(test_start_idx + test_block, n_dates - 1)
        if test_end_idx <= test_start_idx:
            break
        train_start = all_dates[0]
        train_end = all_dates[train_end_idx - 1]
        test_start = all_dates[test_start_idx]
        test_end = all_dates[test_end_idx]
        train_prices = prices.loc[:train_end]
        test_prices = prices.loc[test_start:test_end]
        feat_slice = (
            feature_df[feature_df["date"] <= test_end].copy()
            if not feature_df.empty and "date" in feature_df.columns
            else feature_df.copy()
        )
        macro_slice = macro.copy() if macro is not None else pd.DataFrame()
        if not macro_slice.empty and "date" in macro_slice.columns:
            macro_slice = macro_slice[macro_slice["date"] <= test_end]
        folds.append(
            FoldData(
                fold_idx=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_prices=train_prices,
                test_prices=test_prices,
                feature_df=feat_slice,
                universe=universe,
                macro=macro_slice,
            )
        )
    return folds


def _suggest_params(trial, search_space: dict[str, tuple]) -> dict[str, Any]:
    """Sample hyperparameters from the search space using the Optuna trial."""
    params: dict[str, Any] = {}
    for key, spec in search_space.items():
        if key in REGULATORY_FIXED_KEYS:
            continue
        kind, low, high, log = spec
        if kind == "float":
            params[key] = trial.suggest_float(key, float(low), float(high), log=bool(log))
        elif kind == "int":
            params[key] = trial.suggest_int(key, int(low), int(high), log=bool(log))
        elif kind == "categorical":
            choice_idx = trial.suggest_categorical(f"{key}__idx", list(range(len(low))))
            params[key] = low[choice_idx]
        else:
            raise ValueError(f"Unknown distribution kind {kind!r} for key {key!r}.")
    return params


def _mini_pipeline_metrics(
    fold: FoldData,
    settings: dict[str, Any],
    optimizer: str = "mv",
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """
    Reduced pipeline: forecast + single optimizer + backtest metrics only.

    Returns a dict with sharpe, sortino, max_drawdown, turnover, cvar_95
    measured on the OOS test window of `fold`.  Bootstrap CI and HTML
    reports are disabled for speed.
    """
    cfg = dict(settings)
    cfg["bootstrap_enabled"] = False

    prices = fold.train_prices.combine_first(fold.test_prices).sort_index()
    feature_df = fold.feature_df
    if feature_df.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "cvar_95": 0.0}

    scored = score_cross_section(feature_df)
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    forecast_df = forecast_returns(scored, returns, settings=cfg)
    if forecast_df.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "cvar_95": 0.0}

    fi_tickers = fold.universe.loc[fold.universe["asset_class"] == "fixed_income", "ticker"].tolist()
    forecast_df_opt = forecast_df[~forecast_df["ticker"].isin(fi_tickers)].copy()
    prices_opt = prices[[c for c in prices.columns if c not in fi_tickers]]

    ac_map = (
        fold.universe[fold.universe["asset_class"].isin(["equity", "fibra"])]
        .set_index("ticker")["asset_class"]
        .to_dict()
    )
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity": {"min": 0.30, "max": 0.95},
        "fibra":  {"min": 0.00, "max": 0.30},
    }

    result = run_backtest(
        prices_opt,
        forecast_df_opt,
        fold.universe[fold.universe["asset_class"].isin(["equity", "fibra"])],
        risk_free_rate=risk_free_rate,
        asset_class_constraints=asset_class_constraints,
        optimizer=optimizer,
        macro=fold.macro,
        settings=cfg,
    )
    # Restrict to the OOS test window
    oos_returns = result["returns"].loc[fold.test_start:fold.test_end]
    if oos_returns.empty:
        oos_returns = result["returns"].tail(len(fold.test_prices))

    oos_turnover_series = result["turnover"].loc[fold.test_start:fold.test_end]
    oos_turnover = float(oos_turnover_series.mean()) if not oos_turnover_series.empty else float(
        result["metrics"].get("turnover", 0.0)
    )

    from .risk import compute_cvar, compute_sharpe, compute_sortino, max_drawdown

    return {
        "sharpe": float(compute_sharpe(oos_returns, risk_free_rate=risk_free_rate)) if len(oos_returns) > 0 else 0.0,
        "sortino": float(compute_sortino(oos_returns, required_return=risk_free_rate / 252)) if len(oos_returns) > 0 else 0.0,
        "max_drawdown": float(max_drawdown(oos_returns)) if len(oos_returns) > 0 else 0.0,
        "turnover": oos_turnover,
        "cvar_95": float(compute_cvar(oos_returns, alpha=0.95)) if len(oos_returns) > 0 else 0.0,
    }


def _objective_score(
    metrics_per_fold: list[dict[str, float]],
    objective_metric: str,
    turnover_penalty: float,
) -> float:
    """
    Aggregate metrics across folds into a scalar objective value.

    sharpe_adj = mean(Sharpe_OOS) - turnover_penalty * mean(turnover)
    sortino    = mean(Sortino_OOS)
    calmar     = mean(CAGR / |MaxDD|) = -mean(Sharpe)*mean(return)/dd (approx)
    """
    if not metrics_per_fold:
        return -1e9
    sharpe_vals = [m["sharpe"] for m in metrics_per_fold]
    turnover_vals = [m["turnover"] for m in metrics_per_fold]
    if objective_metric == "sharpe_adj":
        return float(np.mean(sharpe_vals) - turnover_penalty * np.mean(turnover_vals))
    if objective_metric == "sortino":
        return float(np.mean([m["sortino"] for m in metrics_per_fold]))
    if objective_metric == "calmar":
        mdd = np.mean([abs(m["max_drawdown"]) for m in metrics_per_fold]) + 1e-9
        return float(np.mean(sharpe_vals) / mdd)
    raise ValueError(f"Unknown objective_metric: {objective_metric!r}")


def _build_trial_history(study) -> pd.DataFrame:
    """Turn the Optuna study history into a flat DataFrame."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {"trial_number": trial.number, "value": trial.value}
        for key, val in trial.params.items():
            row[key] = val
        row["state"] = str(trial.state)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def run_hyperopt(
    prices: pd.DataFrame,
    feature_df: pd.DataFrame,
    universe: pd.DataFrame,
    macro: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    purge_gap_days: int = 21,
    objective_metric: str = "sharpe_adj",
    turnover_penalty: float = 0.5,
    search_space: dict[str, tuple] | None = None,
    seed: int = 42,
    settings: dict[str, Any] | None = None,
    optimizer: str = "mv",
    risk_free_rate: float = 0.02,
    search_keys: list[str] | None = None,
) -> OptimResult:
    """
    Run Bayesian hyperparameter optimization via Optuna TPE sampler.

    Parameters
    ----------
    prices, feature_df, universe, macro
        Same objects produced by data_loader.load_data and features.build_signal_matrix.
    n_trials
        Number of Optuna trials.
    n_folds
        Number of walk-forward folds per trial.
    purge_gap_days
        Business-day gap inserted between train-end and test-start in each fold.
    objective_metric
        "sharpe_adj" | "sortino" | "calmar".  See `_objective_score`.
    turnover_penalty
        Coefficient on mean turnover subtracted from mean Sharpe in
        sharpe_adj mode.
    search_space
        Mapping key -> (kind, low, high, log_scale) (categorical uses the list
        in position 1).  Defaults to DEFAULT_SEARCH_SPACE.
    search_keys
        Subset of search_space keys to actually optimize.  The other keys are
        fixed at their current settings value.  None = full search space.
    seed
        Optuna sampler seed for reproducibility.

    Returns
    -------
    OptimResult
    """
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover — surfaced only when optuna missing
        raise ImportError(
            "run_hyperopt requires the 'optuna' package. Install it via `pip install optuna>=3.6.0`."
        ) from exc

    cfg_base = resolve_settings(settings)

    full_space = dict(search_space) if search_space is not None else dict(DEFAULT_SEARCH_SPACE)
    # Remove regulatory keys no matter what the caller asked for
    for k in list(full_space.keys()):
        if k in REGULATORY_FIXED_KEYS:
            logger.warning("Dropping regulatory key %s from search space.", k)
            del full_space[k]
    if search_keys is not None:
        full_space = {k: v for k, v in full_space.items() if k in set(search_keys)}

    folds = build_walk_forward_folds(
        prices=prices,
        feature_df=feature_df,
        universe=universe,
        macro=macro,
        n_folds=int(n_folds),
        purge_gap_days=int(purge_gap_days),
    )
    if not folds:
        logger.warning("build_walk_forward_folds produced 0 folds — returning empty OptimResult.")
        return OptimResult(
            best_params={},
            best_value=float("nan"),
            trial_history=pd.DataFrame(),
            validation_metrics={},
            n_trials_completed=0,
            optimization_time_seconds=0.0,
            search_space=full_space,
            objective_metric=objective_metric,
            turnover_penalty=float(turnover_penalty),
        )

    def _objective(trial) -> float:
        params = _suggest_params(trial, full_space)
        cfg = {**cfg_base, **params}
        metrics_per_fold: list[dict[str, float]] = []
        for fold in folds:
            try:
                m = _mini_pipeline_metrics(fold, cfg, optimizer=optimizer, risk_free_rate=risk_free_rate)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("Fold %d failed: %s", fold.fold_idx, exc)
                m = {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "cvar_95": 0.0}
            metrics_per_fold.append(m)
        return _objective_score(metrics_per_fold, objective_metric, float(turnover_penalty))

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    start = time.time()
    study.optimize(_objective, n_trials=int(n_trials), show_progress_bar=False)
    elapsed = time.time() - start

    best_params_raw: dict[str, Any] = dict(study.best_params) if study.best_trial is not None else {}
    # Translate categorical indices back into their real values
    best_params: dict[str, Any] = {}
    for key, spec in full_space.items():
        kind = spec[0]
        if kind == "categorical":
            idx_key = f"{key}__idx"
            idx = best_params_raw.get(idx_key)
            if idx is not None:
                best_params[key] = spec[1][int(idx)]
        elif key in best_params_raw:
            best_params[key] = best_params_raw[key]

    # Validation metrics at best_params
    validation_metrics: dict[str, float] = {}
    if best_params and folds:
        cfg_best = {**cfg_base, **best_params}
        val_metrics_per_fold: list[dict[str, float]] = []
        for fold in folds:
            try:
                val_metrics_per_fold.append(
                    _mini_pipeline_metrics(fold, cfg_best, optimizer=optimizer, risk_free_rate=risk_free_rate)
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Validation fold %d failed: %s", fold.fold_idx, exc)
        if val_metrics_per_fold:
            validation_metrics = {
                "sharpe_mean": float(np.mean([m["sharpe"] for m in val_metrics_per_fold])),
                "sortino_mean": float(np.mean([m["sortino"] for m in val_metrics_per_fold])),
                "max_drawdown_mean": float(np.mean([m["max_drawdown"] for m in val_metrics_per_fold])),
                "turnover_mean": float(np.mean([m["turnover"] for m in val_metrics_per_fold])),
                "cvar_95_mean": float(np.mean([m["cvar_95"] for m in val_metrics_per_fold])),
            }

    trial_history = _build_trial_history(study)

    return OptimResult(
        best_params=best_params,
        best_value=float(study.best_value) if study.best_trial is not None else float("nan"),
        trial_history=trial_history,
        validation_metrics=validation_metrics,
        n_trials_completed=len(study.trials),
        optimization_time_seconds=float(elapsed),
        search_space=full_space,
        objective_metric=objective_metric,
        turnover_penalty=float(turnover_penalty),
    )


__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "REGULATORY_FIXED_KEYS",
    "FoldData",
    "OptimResult",
    "build_walk_forward_folds",
    "run_hyperopt",
]
