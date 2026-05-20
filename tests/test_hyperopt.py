"""Unit tests for src/hyperopt.py — walk-forward fold construction and purge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.hyperopt import build_walk_forward_folds


def _synthetic_prices(n_days: int = 800, n_tickers: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n_days)
    data = {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days))) for i in range(n_tickers)}
    return pd.DataFrame(data, index=idx)


def _empty_feature_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "asset_class"])


def _universe(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "asset_class": ["equity"] * n,
        "investable": [True] * n,
    })


def test_build_walk_forward_folds_count():
    """Requested n_folds must produce at most n_folds non-empty folds."""
    prices = _synthetic_prices(n_days=1200)
    folds = build_walk_forward_folds(
        prices, _empty_feature_df(), _universe(), pd.DataFrame(),
        n_folds=3, purge_gap_days=21, min_train_days=252,
    )
    assert 1 <= len(folds) <= 3


def test_build_walk_forward_folds_respect_purge_gap():
    """Each fold's test window must start at least purge_gap_days after train end."""
    prices = _synthetic_prices(n_days=1500)
    purge = 21
    folds = build_walk_forward_folds(
        prices, _empty_feature_df(), _universe(), pd.DataFrame(),
        n_folds=3, purge_gap_days=purge, min_train_days=252,
    )
    assert folds, "Expected at least one fold."
    for f in folds:
        gap_days = len(pd.bdate_range(f.train_end, f.test_start)) - 1
        assert gap_days >= purge - 2, (
            f"Fold {f.fold_idx}: train_end={f.train_end}, test_start={f.test_start} — "
            f"only {gap_days} bday gap < purge_gap_days={purge}."
        )


def test_build_walk_forward_folds_no_overlap():
    """Train slice must end strictly before the test slice begins."""
    prices = _synthetic_prices(n_days=1500)
    folds = build_walk_forward_folds(
        prices, _empty_feature_df(), _universe(), pd.DataFrame(),
        n_folds=3, purge_gap_days=21, min_train_days=252,
    )
    for f in folds:
        assert f.train_end < f.test_start, f"Train/test overlap in fold {f.fold_idx}."
        assert f.test_start <= f.test_end, f"Test window is empty in fold {f.fold_idx}."


def test_build_walk_forward_folds_empty_on_short_history():
    """Insufficient history must return an empty fold list, not crash."""
    prices = _synthetic_prices(n_days=100)
    folds = build_walk_forward_folds(
        prices, _empty_feature_df(), _universe(), pd.DataFrame(),
        n_folds=3, purge_gap_days=21, min_train_days=252,
    )
    assert folds == []


def test_build_walk_forward_folds_prior_test_purged_from_train():
    """With purge_prior_test_from_train=True, fold k+1 feature_df must not contain
    any dates from fold k's test window (labeled examples from prior test sets
    must be excluded from subsequent training folds)."""
    prices = _synthetic_prices(n_days=1500)
    n_tickers = prices.shape[1]

    # Build a feature_df with one row per (date, ticker)
    all_dates = prices.index
    rows = [
        {"date": d, "ticker": f"T{i}", "asset_class": "equity", "momentum_63": float(i)}
        for d in all_dates
        for i in range(n_tickers)
    ]
    feature_df = pd.DataFrame(rows)

    folds = build_walk_forward_folds(
        prices, feature_df, _universe(n_tickers), pd.DataFrame(),
        n_folds=3, purge_gap_days=21, min_train_days=252,
        purge_prior_test_from_train=True,
    )
    assert len(folds) >= 2, "Need at least 2 folds to test inter-fold purge."

    for k in range(1, len(folds)):
        fold_k = folds[k]
        for prior in folds[:k]:
            dates_in_feat = set(pd.to_datetime(fold_k.feature_df["date"]))
            test_dates = set(
                d for d in prices.index
                if prior.test_start <= d <= prior.test_end
            )
            leaked = dates_in_feat & test_dates
            assert not leaked, (
                f"Fold {k} feature_df contains {len(leaked)} date(s) from fold "
                f"{prior.fold_idx} test window [{prior.test_start.date()}, "
                f"{prior.test_end.date()}]. Inter-fold purge is broken."
            )


def test_build_walk_forward_folds_prior_test_not_purged_when_disabled():
    """With purge_prior_test_from_train=False (legacy), fold k+1 feature_df
    includes dates from fold k's test window (expanding window behaviour)."""
    prices = _synthetic_prices(n_days=1500)
    n_tickers = prices.shape[1]
    all_dates = prices.index
    rows = [
        {"date": d, "ticker": f"T{i}", "asset_class": "equity", "momentum_63": 0.0}
        for d in all_dates
        for i in range(n_tickers)
    ]
    feature_df = pd.DataFrame(rows)

    folds = build_walk_forward_folds(
        prices, feature_df, _universe(n_tickers), pd.DataFrame(),
        n_folds=3, purge_gap_days=21, min_train_days=252,
        purge_prior_test_from_train=False,
    )
    assert len(folds) >= 2

    # fold 1's feature_df must include at least some of fold 0's test dates
    fold0, fold1 = folds[0], folds[1]
    overlap_dates = set(pd.to_datetime(fold1.feature_df["date"])) & set(
        d for d in prices.index if fold0.test_start <= d <= fold0.test_end
    )
    assert overlap_dates, (
        "Expected fold 1 feature_df to contain fold 0 test dates when "
        "purge_prior_test_from_train=False (expanding window behaviour)."
    )
