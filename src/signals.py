from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .settings import resolve_settings

logger = logging.getLogger(__name__)

SUPPORTED_FORECAST_MODELS = ("elasticnet", "xgboost")

# Feature sets per asset class (externalized so they're easy to update)
_EQUITY_FEATURES = [
    "momentum_63", "momentum_126", "volatility_63",
    "pe_ratio", "pb_ratio", "roe", "profit_margin", "net_debt_to_ebitda",
    "ebitda_growth", "capex_to_sales",
    "industrial_production_yoy", "usd_mxn", "exports_yoy",
]
_FIBRA_FEATURES = [
    "momentum_63", "volatility_63",
    "cap_rate", "ffo_yield", "dividend_yield", "ltv", "vacancy_rate",
    "industrial_production_yoy", "usd_mxn",
]

_FORWARD_DAYS = 21  # default prediction horizon (runtime value comes from settings.forecast_forward_days)


def _compute_forward_returns(group_df: pd.DataFrame, forward_days: int) -> pd.Series:
    """
    Compute forward_days-ahead return per ticker using only past prices.
    The last forward_days rows per ticker receive NaN (return not yet realized).
    This avoids any look-ahead bias.
    """
    n = len(group_df)
    fwd = np.full(n, np.nan)
    if n > forward_days:
        prices = group_df["price"].values
        ratio = prices[forward_days:] / (prices[: n - forward_days] + 1e-9)
        fwd[: n - forward_days] = np.log(ratio)
    return pd.Series(fwd, index=group_df.index)


def _end_of_month_dates(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return the last available date in each calendar month."""
    s = pd.Series(dates, index=dates)
    return s.resample("ME").last().dropna().values


def score_cross_section(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional scores and the composite score.

    For each date, transforms factors (momentum, value, quality, liquidity)
    into percentile ranks (rank/pct) within the universe — this makes each
    signal comparable across assets regardless of scale differences.

    Composite score = simple average of the available factor ranks.
    A score close to 1 = best in universe; close to 0 = worst.
    """
    scores = feature_df.copy()
    # features.py produces momentum_63 and momentum_126; use momentum_63 as the base momentum signal
    candidate_columns = [
        "momentum_63",
        "value_score",
        "quality_score",
        "liquidity_score",
    ]
    score_columns = [c for c in candidate_columns if c in scores.columns]
    for col in score_columns:
        scores[f"{col}_rank"] = scores.groupby("date")[col].rank(ascending=False, pct=True)
    if score_columns:
        scores["composite_score"] = scores[[f"{col}_rank" for col in score_columns]].mean(axis=1)
    else:
        scores["composite_score"] = np.nan
    return scores


def _fit_predict_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    cfg: dict,
) -> np.ndarray | None:
    """Fit ElasticNetCV on (X_train, y_train) and predict on X_pred.

    Returns predictions as a 1-D ndarray, or None if the fit fails.
    Behaviour bit-identical to the pre-dispatcher inline block.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = ElasticNetCV(
        cv=int(cfg["elasticnet_cv_folds"]),
        l1_ratio=list(cfg["elasticnet_l1_ratios"]),
        max_iter=int(cfg["elasticnet_max_iter"]),
        tol=float(cfg["elasticnet_tol"]),
        random_state=42,
        n_jobs=-1,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train_scaled, y_train)
    except Exception as exc:
        logger.warning("ElasticNetCV fit failed: %s", exc)
        return None
    return model.predict(scaler.transform(X_pred))


def _xgboost_cfg_from_settings(cfg: dict) -> dict:
    return {
        "n_iter": int(cfg["forecast_xgb_n_iter"]),
        "cv_splits": int(cfg["forecast_xgb_cv_splits"]),
        "n_estimators_cap": int(cfg["forecast_xgb_n_estimators_cap"]),
        "early_stopping_rounds": int(cfg["forecast_xgb_early_stopping_rounds"]),
        "scoring": str(cfg["forecast_xgb_scoring"]),
        "random_state": int(cfg["forecast_xgb_random_state"]),
        "n_jobs": int(cfg["forecast_xgb_n_jobs"]),
        "search_n_jobs": int(cfg["forecast_xgb_search_n_jobs"]),
    }


def _fit_predict_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    cfg: dict,
) -> np.ndarray | None:
    """Fit XGBoostModel on (X_train, y_train) and predict on X_pred.

    Returns predictions as a 1-D ndarray, or None if the fit fails.
    XGBoostModel handles its own StandardScaler + RandomizedSearchCV +
    early-stopping refit internally.
    """
    from .xgboost_model import XGBoostModel

    model = XGBoostModel(config=_xgboost_cfg_from_settings(cfg))
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        logger.warning("XGBoostModel fit failed: %s", exc)
        return None
    return model.predict(X_pred)


def forecast_returns(
    feature_df: pd.DataFrame,
    returns: pd.DataFrame,
    settings: dict | None = None,
) -> pd.DataFrame:
    """Forecast returns with an expanding-window cross-sectional model.

    For each asset class (equity, fibra) and each end-of-month rebalancing date:
      1. Builds the training set: all historical (date, ticker) pairs with
         realized forward return available (no look-ahead bias).
      2. Hands (X_train, y_train, X_pred) to the model selected by
         `settings["forecast_model"]`:
            - "elasticnet" → ElasticNetCV with internal KFold(cv_folds).
            - "xgboost"    → XGBoostModel with RandomizedSearchCV inside
                             a TimeSeriesSplit(cv_splits) over the training
                             window only, plus early stopping.
      3. Stores predicted expected return for all tickers active on that date.

    The target return is the log return forecast_forward_days ahead,
    computed with _compute_forward_returns() without look-ahead.

    Finally, standardizes predictions cross-sectionally (z-score per date)
    so they are comparable across assets and dates.

    Hyperparameters (cv folds, l1_ratios, max_iter, tol for ElasticNet;
    n_iter, cv_splits, scoring, etc. for XGBoost) and the forecast horizon
    come from `settings`; see DEFAULT_SETTINGS in src/settings.py.
    """
    cfg = resolve_settings(settings)
    forward_days = int(cfg["forecast_forward_days"])
    min_train_rows = int(cfg["forecast_min_train_rows"])

    model_name = str(cfg.get("forecast_model", "elasticnet")).lower().strip()
    if model_name not in SUPPORTED_FORECAST_MODELS:
        raise ValueError(
            f"Unsupported forecast_model={model_name!r}; "
            f"choose one of {SUPPORTED_FORECAST_MODELS}."
        )

    if model_name == "xgboost":
        fit_predict = _fit_predict_xgboost
    else:
        fit_predict = _fit_predict_elasticnet

    do_shap = (model_name == "xgboost") and bool(cfg.get("compute_shap", True))
    shap_records: list[dict] = []

    forecasts: list[pd.DataFrame] = []
    asset_classes = feature_df["asset_class"].unique()

    for asset_class in asset_classes:
        class_df = feature_df[feature_df["asset_class"] == asset_class].copy()
        class_df = class_df.sort_values(["ticker", "date"])

        if asset_class == "fixed_income":
            logger.debug("Skipping fixed_income for forecast inputs.")
            continue
        elif asset_class == "fibra":
            feature_cols = [c for c in _FIBRA_FEATURES if c in class_df.columns]
        else:
            feature_cols = [c for c in _EQUITY_FEATURES if c in class_df.columns]

        if not feature_cols:
            logger.warning("No feature columns found for asset_class=%s — skipping.", asset_class)
            continue

        _fwd_parts = []
        for ticker, grp in class_df.groupby("ticker", group_keys=False):
            _fwd_parts.append(_compute_forward_returns(grp, forward_days))
        class_df["_fwd_return"] = pd.concat(_fwd_parts).reindex(class_df.index)

        all_dates = pd.DatetimeIndex(sorted(class_df["date"].unique()))
        rebal_dates = _end_of_month_dates(all_dates)

        for date in rebal_dates:
            train_mask = (class_df["date"] <= date) & class_df["_fwd_return"].notna()
            train_data = class_df.loc[train_mask]

            if len(train_data) < min_train_rows:
                logger.debug(
                    "Skipping %s on %s: only %d training rows.", asset_class, date, len(train_data)
                )
                continue

            current_data = class_df[class_df["date"] == date].copy()
            if current_data.empty:
                continue

            X_train = train_data[feature_cols].fillna(0.0)
            y_train = train_data["_fwd_return"]
            X_pred = current_data[feature_cols].fillna(0.0)

            if do_shap:
                # Inline XGBoost fit so we can intercept the model object for SHAP.
                from .xgboost_model import XGBoostModel
                from .shap_attribution import collect_rebalance_shap

                _xgb_model = XGBoostModel(config=_xgboost_cfg_from_settings(cfg))
                try:
                    _xgb_model.fit(X_train, y_train)
                except Exception as exc:
                    logger.warning("XGBoostModel fit failed at %s: %s", date, exc)
                    continue
                # collect_rebalance_shap returns [] on any SHAP error — no try/except needed.
                shap_records.extend(
                    collect_rebalance_shap(
                        _xgb_model,
                        X_pred,
                        date,
                        current_data["ticker"].tolist(),
                        feature_cols,
                    )
                )
                preds = _xgb_model.predict(X_pred)
            else:
                preds = fit_predict(X_train, y_train, X_pred, cfg)

            if preds is None:
                logger.warning(
                    "%s fit failed for %s on %s — skipping rebalance.",
                    model_name, asset_class, date,
                )
                continue
            current_data["expected_return"] = preds
            forecasts.append(current_data)

    if do_shap and shap_records:
        from .shap_attribution import write_shap_parquet
        shap_path = str(cfg.get("shap_output_path", "data/shap_values.parquet"))
        write_shap_parquet(shap_records, shap_path)

    if not forecasts:
        return pd.DataFrame(columns=["date", "ticker", "asset_class", "expected_return"])

    result = pd.concat(forecasts, ignore_index=True)
    result["expected_return"] = result.groupby("date")["expected_return"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    result = result.drop(columns=["_fwd_return"], errors="ignore")
    return result
