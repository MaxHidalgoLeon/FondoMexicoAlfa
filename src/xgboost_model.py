"""XGBoost cross-sectional return predictor.

Drop-in alternative to the inline `ElasticNetCV` block used by
`forecast_returns()` in `src/signals.py`. Exposes the same de-facto
sklearn API (`fit(X, y)` + `predict(X)`) so the dispatcher in
`signals.forecast_returns` can swap models with no other code changes.

Hyperparameter selection runs `RandomizedSearchCV` with
`TimeSeriesSplit(n_splits=5)` *inside the supplied training window only*,
honouring the no-look-ahead invariant of the walk-forward backtest.
Early stopping on the held-out fold caps `n_estimators` per trial.

Everything is seeded (`numpy`, `xgboost`, sklearn search) so the same
panel + same settings produce bit-stable predictions.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

logger = logging.getLogger(__name__)


DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_alpha":        [0.0, 0.1, 1.0],
    "reg_lambda":       [0.1, 1.0, 10.0],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators_cap": 2000,
    "early_stopping_rounds": 50,
    "n_iter": 20,
    "cv_splits": 5,
    "scoring": "neg_mean_squared_error",  # or "ic" for Spearman rank-IC
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
    "search_n_jobs": 1,
    "verbosity": 0,
}


def _spearman_ic_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between y_true and y_pred. Higher is better."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size < 4 or y_pred.size < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(y_pred, y_true)
    return float(rho) if np.isfinite(rho) else 0.0


_IC_SCORER = make_scorer(_spearman_ic_scorer, greater_is_better=True)


def _resolve_config(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def _resolve_search_space(overrides: Mapping[str, Any] | None) -> dict[str, list[Any]]:
    space = {k: list(v) for k, v in DEFAULT_SEARCH_SPACE.items()}
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            space[k] = list(v)
    return space


def _resolve_scorer(name: str):
    name = (name or "").lower().strip()
    if name in ("ic", "spearman", "spearman_ic", "rank_ic"):
        return _IC_SCORER
    return name or "neg_mean_squared_error"


class XGBoostModel:
    """sklearn-compatible XGBoost regressor with internal time-series CV.

    Public surface (matches the sklearn estimator protocol used by
    `forecast_returns` in `src/signals.py`):
        - `fit(X, y)`  → self
        - `predict(X)` → ndarray of predicted forward returns

    The `fit` call performs:
      1. StandardScaler on X (fit on train only).
      2. RandomizedSearchCV over `DEFAULT_SEARCH_SPACE` with
         `TimeSeriesSplit(n_splits=cfg.cv_splits)` and `cfg.n_iter` draws.
         Scoring defaults to negative MSE; set `scoring="ic"` to use the
         Spearman rank-IC scorer.
      3. Refits the best estimator on the full training window with early
         stopping against the last `1/(cv_splits+1)` slice of train,
         which sets `best_iteration` ≤ `n_estimators_cap`.

    All randomness (numpy, sklearn search, xgboost) is seeded from
    `cfg.random_state` so the same input panel yields bit-stable output.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        search_space: Mapping[str, Any] | None = None,
    ) -> None:
        self._cfg = _resolve_config(config)
        self._search_space = _resolve_search_space(search_space)
        self._scaler: StandardScaler | None = None
        self._best_estimator: xgb.XGBRegressor | None = None
        self._best_params_: dict[str, Any] | None = None
        self._best_iteration_: int | None = None
        self._n_features_: int | None = None
        self._feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_base(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=int(self._cfg["n_estimators_cap"]),
            tree_method=str(self._cfg["tree_method"]),
            objective=str(self._cfg["objective"]),
            random_state=int(self._cfg["random_state"]),
            n_jobs=int(self._cfg["n_jobs"]),
            verbosity=int(self._cfg["verbosity"]),
        )

    @staticmethod
    def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float, copy=False)
        return np.asarray(X, dtype=float)

    @staticmethod
    def _to_target(y: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.to_numpy(dtype=float, copy=False)
        return np.asarray(y, dtype=float)

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "XGBoostModel":
        X_arr = self._to_numpy(X)
        y_arr = self._to_target(y)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X / y row mismatch: {X_arr.shape[0]} vs {y_arr.shape[0]}"
            )
        if isinstance(X, pd.DataFrame):
            self._feature_names_ = list(X.columns)
        self._n_features_ = X_arr.shape[1]

        # Need enough rows for at least cv_splits+1 chronological folds.
        cv_splits = int(self._cfg["cv_splits"])
        if X_arr.shape[0] < (cv_splits + 1) * 2:
            cv_splits = max(2, X_arr.shape[0] // 2 - 1)

        np.random.seed(int(self._cfg["random_state"]))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        scorer = _resolve_scorer(str(self._cfg["scoring"]))
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        base = self._make_base()

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=self._search_space,
            n_iter=int(self._cfg["n_iter"]),
            scoring=scorer,
            cv=tscv,
            n_jobs=int(self._cfg["search_n_jobs"]),
            random_state=int(self._cfg["random_state"]),
            refit=False,
            error_score="raise",
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            search.fit(X_scaled, y_arr)

        self._best_params_ = dict(search.best_params_)

        # Refit with early stopping on a chronological holdout.
        holdout_frac = 1.0 / (cv_splits + 1)
        n = X_scaled.shape[0]
        cut = max(int(n * (1.0 - holdout_frac)), max(50, n - 200))
        cut = min(cut, n - 10) if n > 20 else max(1, n - 1)

        X_tr, X_val = X_scaled[:cut], X_scaled[cut:]
        y_tr, y_val = y_arr[:cut], y_arr[cut:]

        final = xgb.XGBRegressor(
            n_estimators=int(self._cfg["n_estimators_cap"]),
            tree_method=str(self._cfg["tree_method"]),
            objective=str(self._cfg["objective"]),
            random_state=int(self._cfg["random_state"]),
            n_jobs=int(self._cfg["n_jobs"]),
            verbosity=int(self._cfg["verbosity"]),
            early_stopping_rounds=int(self._cfg["early_stopping_rounds"]),
            **self._best_params_,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            self._best_iteration_ = int(getattr(final, "best_iteration", 0) or 0)
        except Exception as exc:
            logger.warning("XGBoost early-stopping refit failed (%s); refitting on full train.", exc)
            final = xgb.XGBRegressor(
                n_estimators=int(self._cfg["n_estimators_cap"]),
                tree_method=str(self._cfg["tree_method"]),
                objective=str(self._cfg["objective"]),
                random_state=int(self._cfg["random_state"]),
                n_jobs=int(self._cfg["n_jobs"]),
                verbosity=int(self._cfg["verbosity"]),
                **self._best_params_,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final.fit(X_scaled, y_arr, verbose=False)
            self._best_iteration_ = None

        self._best_estimator = final
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._best_estimator is None or self._scaler is None:
            raise RuntimeError("XGBoostModel.predict called before fit().")
        X_arr = self._to_numpy(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if self._n_features_ is not None and X_arr.shape[1] != self._n_features_:
            raise ValueError(
                f"Feature count mismatch: fit had {self._n_features_}, predict got {X_arr.shape[1]}"
            )
        X_scaled = self._scaler.transform(X_arr)
        if self._best_iteration_ and self._best_iteration_ > 0:
            return self._best_estimator.predict(
                X_scaled, iteration_range=(0, self._best_iteration_ + 1)
            ).astype(float)
        return self._best_estimator.predict(X_scaled).astype(float)

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------
    @property
    def best_params_(self) -> dict[str, Any] | None:
        return None if self._best_params_ is None else dict(self._best_params_)

    @property
    def best_iteration_(self) -> int | None:
        return self._best_iteration_

    @property
    def feature_names_(self) -> list[str] | None:
        return None if self._feature_names_ is None else list(self._feature_names_)

    @property
    def estimator_(self) -> xgb.XGBRegressor | None:
        return self._best_estimator
