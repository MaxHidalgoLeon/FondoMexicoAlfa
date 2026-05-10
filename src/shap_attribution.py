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

_REQUIRED_COLS = {"date", "ticker", "feature", "shap_value", "feature_value"}


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

    Uses ``shap.TreeExplainer`` on the fitted ``XGBRegressor`` inside ``model``.
    The feature matrix is scaled via ``model.scale()`` before SHAP computation
    so that SHAP values correspond to the same feature space the trees were
    trained on.

    Args:
        model: Fitted ``XGBoostModel`` instance. Must expose ``.estimator_``
            (the underlying ``XGBRegressor``) and ``.scale()`` for transforming
            raw features to the scaled space.
        X_pred_df: Raw (unscaled) prediction-set features, shape
            ``(n_tickers, n_features)``.
        date: Rebalance date; any value accepted by ``pd.Timestamp``.
        tickers: Ordered list of ticker labels corresponding to the rows of
            ``X_pred_df``.
        feature_cols: Ordered list of feature names corresponding to the
            columns of ``X_pred_df``.

    Returns:
        List of dicts, one per (ticker, feature) pair, with keys:
        ``date``, ``ticker``, ``feature``, ``shap_value``, ``feature_value``.
        Returns an empty list on any error so the walk-forward loop can
        continue without crashing.
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
    X_raw = np.asarray(X_pred_df, dtype=float)
    records: list[dict] = []
    for i, ticker in enumerate(tickers):
        for j, feat in enumerate(feature_cols):
            records.append({
                "date": ts,
                "ticker": ticker,
                "feature": feat,
                "shap_value": float(sv[i, j]),
                "feature_value": float(X_raw[i, j]),
            })
    return records


def write_shap_parquet(records: list[dict], path: str | Path) -> None:
    """Persist accumulated SHAP records to parquet, overwriting any existing file.

    Args:
        records: List of dicts produced by ``collect_rebalance_shap``. Each
            dict must contain ``date``, ``ticker``, ``feature``,
            ``shap_value``, and ``feature_value`` keys.
        path: Destination file path. Parent directories are created if absent.
    """
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
    df["feature_value"] = df["feature_value"].astype(float)
    df.to_parquet(path, index=False)
    logger.info("SHAP parquet written: %s  (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_shap_parquet(path: str | Path) -> pd.DataFrame:
    """Load and validate the SHAP parquet file written by ``write_shap_parquet``.

    Args:
        path: Path to the parquet file (``data/shap_values.parquet`` by default).

    Returns:
        DataFrame with columns ``date`` (datetime64), ``ticker`` (str),
        ``feature`` (str), ``shap_value`` (float64), ``feature_value`` (float64).

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If any required column is absent.
    """
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
    """Return the top-n features by time-averaged mean absolute SHAP value.

    Args:
        shap_df: Long-format SHAP DataFrame with at least ``date``,
            ``feature``, and ``shap_value`` columns.
        n: Number of top features to return (default 10).

    Returns:
        DataFrame with columns ``rank``, ``feature``, ``mean_abs_shap``,
        ``std_abs_shap``, sorted descending by ``mean_abs_shap``.
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
    """Build a pivot table of mean absolute SHAP values by date and feature.

    Args:
        shap_df: Long-format SHAP DataFrame with ``date``, ``feature``,
            and ``shap_value`` columns.

    Returns:
        DataFrame with ``date`` as index and feature names as columns;
        values are mean ``|shap_value|`` across tickers for each
        ``(date, feature)`` pair.
    """
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

    For each consecutive pair of rebalance dates ``(t, t+1)``, ranks features
    by mean ``|SHAP|`` and computes the Spearman correlation between the two
    rankings. A score near 1.0 indicates that model feature importance is
    stable across months; near 0.0 indicates high churn.

    Args:
        shap_df: Long-format SHAP DataFrame as returned by ``load_shap_parquet``.
        k_values: For each element ``k``, only the top-``k`` features at
            date ``t`` are used when computing the correlation. ``None`` uses
            all features.

    Returns:
        Dict mapping ``k`` (int) or ``"all"`` (str) to a list of
        per-consecutive-pair Spearman correlations.
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
    """Convert the stability dict to a summary DataFrame with mean and std per K.

    Args:
        stability: Dict as returned by ``compute_stability``, mapping
            ``k`` (int or ``"all"``) to a list of Spearman correlations.

    Returns:
        DataFrame with columns ``K``, ``n_pairs``, ``mean_spearman``,
        ``std_spearman`` — one row per entry in ``stability``.
    """
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
    """Decompose SHAP score volatility to identify the main turnover drivers.

    The total SHAP score per ``(date, ticker)`` approximates the model's
    centred prediction: ``f(x) - E[f(x)] = sum_j SHAP_j``. Month-to-month
    variance of per-feature SHAP deltas across all ``(date, ticker)`` pairs
    quantifies each feature's contribution to prediction instability.

    Args:
        shap_df: Long-format SHAP DataFrame as returned by ``load_shap_parquet``.
        top_n: Number of top driver features to return (default 3).

    Returns:
        Dict with two keys:

        - ``"top_drivers"``: DataFrame with columns ``feature``,
          ``var_contribution``, ``pct_contribution`` for the top-``top_n``
          features sorted by variance contribution descending.
        - ``"shap_score_df"``: Long DataFrame ``(date, ticker, shap_score,
          delta_shap_score)`` for downstream visualisation.
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
