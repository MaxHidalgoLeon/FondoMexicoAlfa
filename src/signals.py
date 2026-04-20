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


def forecast_returns(
    feature_df: pd.DataFrame,
    returns: pd.DataFrame,
    settings: dict | None = None,
) -> pd.DataFrame:
    """
    Forecast returns using an expanding-window Elastic Net per asset class.

    Training target: forward forecast_forward_days return computed WITHOUT look-ahead.
    Models are retrained monthly (end-of-month dates only) for efficiency.

    ElasticNetCV hyperparameters (cv folds, l1_ratios, max_iter, tol) and the
    forecast horizon come from `settings`; see DEFAULT_SETTINGS in src/settings.py.
    """
    cfg = resolve_settings(settings)
    forward_days = int(cfg["forecast_forward_days"])
    min_train_rows = int(cfg["forecast_min_train_rows"])
    cv_folds = int(cfg["elasticnet_cv_folds"])
    l1_ratios = list(cfg["elasticnet_l1_ratios"])
    max_iter = int(cfg["elasticnet_max_iter"])
    tol = float(cfg["elasticnet_tol"])

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

            X_train = train_data[feature_cols].fillna(0.0)
            y_train = train_data["_fwd_return"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = ElasticNetCV(
                cv=cv_folds,
                l1_ratio=l1_ratios,
                max_iter=max_iter,
                tol=tol,
                random_state=42,
                n_jobs=-1,
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(X_train_scaled, y_train)
            except Exception as exc:
                logger.warning("ElasticNetCV fit failed for %s on %s: %s", asset_class, date, exc)
                continue

            current_data = class_df[class_df["date"] == date].copy()
            if current_data.empty:
                continue
            X_pred = current_data[feature_cols].fillna(0.0)
            current_data["expected_return"] = model.predict(scaler.transform(X_pred))
            forecasts.append(current_data)

    if not forecasts:
        return pd.DataFrame()

    result = pd.concat(forecasts, ignore_index=True)
    result["expected_return"] = result.groupby("date")["expected_return"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    result = result.drop(columns=["_fwd_return"], errors="ignore")
    return result
