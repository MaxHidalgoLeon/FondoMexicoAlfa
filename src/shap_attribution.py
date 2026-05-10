"""SHAP attribution utilities for the XGBoost walk-forward forecaster.

Public API consumed by `forecast_returns` in `src/signals.py`:
    collect_rebalance_shap(model, X_pred_df, date, tickers, feature_cols)
    write_shap_parquet(records, path)

Analysis functions consumed by `scripts/render_step2_report.py`:
    load_shap_parquet(path)
    compute_stability(shap_df, k_values)
    compute_turnover_drivers(shap_df)
    top_features_table(shap_df, n)
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"date", "ticker", "feature", "shap_value"}


# ---------------------------------------------------------------------------
# Per-rebalance collection
# ---------------------------------------------------------------------------

def collect_rebalance_shap(
    model,
    X_pred_df: pd.DataFrame,
    date,
    tickers: list[str],
    feature_cols: list[str],
) -> list[dict]:
    """Compute SHAP values for one rebalance and return as a list of records.

    Parameters
    ----------
    model:        Fitted XGBoostModel (must have .estimator_ and .scale()).
    X_pred_df:    Raw (unscaled) prediction-set features, shape (n_tickers, n_features).
    date:         Rebalance date (any type accepted by pd.Timestamp).
    tickers:      Ordered list of ticker labels matching X_pred_df rows.
    feature_cols: Ordered list of feature names matching X_pred_df columns.

    Returns
    -------
    List of dicts with keys: date, ticker, feature, shap_value.
    Returns [] on any error so the caller can continue without crashing.
    """
    try:
        import shap as _shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP collection.")
        return []

    estimator = model.estimator_
    if estimator is None:
        logger.warning("collect_rebalance_shap: model.estimator_ is None — skipping.")
        return []

    try:
        X_scaled = model.scale(X_pred_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = _shap.TreeExplainer(estimator)
            sv = explainer.shap_values(X_scaled)
        # sv shape: (n_tickers, n_features)
        sv = np.asarray(sv, dtype=float)
        if sv.ndim != 2 or sv.shape != (len(tickers), len(feature_cols)):
            logger.warning(
                "collect_rebalance_shap: unexpected shap_values shape %s at %s — skipping.",
                sv.shape, date,
            )
            return []
    except Exception as exc:
        logger.warning("collect_rebalance_shap failed at %s: %s", date, exc)
        return []

    ts = pd.Timestamp(date)
    records: list[dict] = []
    for i, ticker in enumerate(tickers):
        for j, feat in enumerate(feature_cols):
            records.append({
                "date": ts,
                "ticker": ticker,
                "feature": feat,
                "shap_value": float(sv[i, j]),
            })
    return records


def write_shap_parquet(records: list[dict], path: str | Path) -> None:
    """Persist accumulated SHAP records to parquet (overwrite)."""
    if not records:
        logger.warning("write_shap_parquet: no records to write.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["feature"] = df["feature"].astype(str)
    df["shap_value"] = df["shap_value"].astype(float)
    df.to_parquet(path, index=False)
    logger.info("SHAP parquet written: %s  (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_shap_parquet(path: str | Path) -> pd.DataFrame:
    """Load and validate the SHAP parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHAP parquet not found: {path}")
    df = pd.read_parquet(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"SHAP parquet missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def top_features_table(shap_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n features by time-averaged mean |SHAP|.

    Columns: rank, feature, mean_abs_shap, std_abs_shap.
    """
    abs_df = shap_df.copy()
    abs_df["abs_shap"] = abs_df["shap_value"].abs()
    per_date = abs_df.groupby(["date", "feature"])["abs_shap"].mean().rename("mean_abs")
    agg = per_date.groupby("feature").agg(["mean", "std"]).reset_index()
    agg.columns = ["feature", "mean_abs_shap", "std_abs_shap"]
    agg = agg.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    return agg


def compute_mean_abs_shap_pivot(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Return pivot: index=date, columns=feature, values=mean(|shap|) across tickers."""
    abs_df = shap_df.copy()
    abs_df["abs_shap"] = abs_df["shap_value"].abs()
    pivot = (
        abs_df.groupby(["date", "feature"])["abs_shap"]
        .mean()
        .unstack("feature")
        .sort_index()
    )
    return pivot


def compute_stability(
    shap_df: pd.DataFrame,
    k_values: Sequence[int | None] = (5, 10, None),
) -> dict[str | int, list[float]]:
    """Compute Spearman rank-correlation stability of feature rankings.

    For each consecutive pair of rebalance dates (t, t+1), rank features by
    mean |SHAP| and compute Spearman correlation of the rankings.

    Parameters
    ----------
    k_values : sequence of int or None.
        For each k, the top-k features at date t are used.
        None means all features.

    Returns
    -------
    dict mapping k (int or "all") → list of per-consecutive-pair correlations.
    """
    pivot = compute_mean_abs_shap_pivot(shap_df)
    dates = sorted(pivot.index)
    if len(dates) < 2:
        return {(k if k is not None else "all"): [] for k in k_values}

    results: dict = {}
    for k in k_values:
        label = k if k is not None else "all"
        corrs: list[float] = []
        for i in range(len(dates) - 1):
            r1 = pivot.loc[dates[i]].dropna()
            r2 = pivot.loc[dates[i + 1]].dropna()
            common = r1.index.intersection(r2.index)
            if len(common) < 2:
                continue
            r1c, r2c = r1[common], r2[common]
            if k is not None and k < len(common):
                top_k = r1c.nlargest(k).index
                r1c = r1c[top_k]
                r2c = r2c.reindex(top_k).fillna(0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(r1c.rank(ascending=False), r2c.rank(ascending=False))
            if np.isfinite(rho):
                corrs.append(float(rho))
        results[label] = corrs
    return results


def stability_summary(stability: dict) -> pd.DataFrame:
    """Convert stability dict to a summary DataFrame (mean ± std per K)."""
    rows = []
    for k, corrs in stability.items():
        label = f"top-{k}" if isinstance(k, int) else "all"
        arr = np.asarray(corrs, dtype=float)
        rows.append({
            "K": label,
            "n_pairs": len(arr),
            "mean_spearman": float(arr.mean()) if len(arr) else float("nan"),
            "std_spearman": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_turnover_drivers(shap_df: pd.DataFrame, top_n: int = 3) -> dict:
    """Decompose SHAP score volatility to identify turnover drivers.

    Total SHAP score per (date, ticker) ≈ f(x) - E[f(x)] = sum of per-feature
    SHAP values. The change in SHAP score from t-1 to t is driven by changes
    in individual feature SHAP contributions.

    Returns
    -------
    dict with:
        "top_drivers": DataFrame (feature, var_contribution, pct_contribution)
        "shap_score_df": long DataFrame (date, ticker, shap_score, delta_shap_score)
    """
    # Total SHAP score per (date, ticker)
    score = (
        shap_df.groupby(["date", "ticker"])["shap_value"]
        .sum()
        .rename("shap_score")
        .reset_index()
        .sort_values(["ticker", "date"])
    )
    score["delta_shap_score"] = score.groupby("ticker")["shap_score"].diff()

    # Per-feature SHAP delta
    feat_pivot = (
        shap_df.pivot_table(index=["date", "ticker"], columns="feature", values="shap_value")
        .sort_index()
    )
    feat_delta = feat_pivot.groupby(level="ticker").diff()

    # Variance of each feature's SHAP delta across all (date, ticker) pairs
    feat_var = feat_delta.var()
    total_var = feat_var.sum()
    if total_var <= 0:
        return {"top_drivers": pd.DataFrame(), "shap_score_df": score}

    pct = (feat_var / total_var * 100).sort_values(ascending=False)
    drivers = pd.DataFrame({
        "feature": pct.index,
        "var_contribution": feat_var[pct.index].values,
        "pct_contribution": pct.values,
    }).head(top_n).reset_index(drop=True)

    return {"top_drivers": drivers, "shap_score_df": score}
