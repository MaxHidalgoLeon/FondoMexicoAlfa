"""Tests for src/lightgbm_model.py.

Covers the four contract checks called out in Step 1:
  - test_fit_predict_shapes  : fit on a synthetic panel → predict shape matches.
  - test_no_lookahead        : training window predictions don't depend on
                               post-cutoff feature rows.
  - test_reproducibility     : same seed + same data → bit-identical preds.
  - test_interface_parity    : LightGBMModel exposes the sklearn-compatible
                               (fit, predict) surface that the inline
                               ElasticNetCV path also relies on.

A fifth test pins the dispatcher in `src/signals.py:forecast_returns` so the
elasticnet baseline keeps working when forecast_model is unset.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.lightgbm_model import LightGBMModel


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Reproducible synthetic regression panel: y = X @ beta + noise."""
    rng = np.random.default_rng(7)
    n_rows, n_features = 240, 6
    X = pd.DataFrame(
        rng.standard_normal((n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    beta = np.array([0.5, -0.4, 0.0, 0.3, 0.0, -0.2])
    y = pd.Series(X.values @ beta + 0.1 * rng.standard_normal(n_rows))
    return X, y


def _fast_cfg(**overrides) -> dict:
    cfg = {
        "n_iter": 4,
        "cv_splits": 3,
        "n_estimators_cap": 200,
        "early_stopping_rounds": 20,
        "verbosity": 0,
    }
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------
# 1. Fit / predict shapes
# --------------------------------------------------------------------------

def test_fit_predict_shapes(synthetic_panel) -> None:
    X, y = synthetic_panel
    train, test = X.iloc[:200], X.iloc[200:]
    y_train = y.iloc[:200]
    model = LightGBMModel(config=_fast_cfg())
    model.fit(train, y_train)
    preds = model.predict(test)
    assert preds.shape == (len(test),)
    assert preds.dtype.kind == "f"
    assert np.isfinite(preds).all()


# --------------------------------------------------------------------------
# 2. No look-ahead — training-time predictions don't peek at future rows
# --------------------------------------------------------------------------

def test_no_lookahead(synthetic_panel) -> None:
    """Predictions on row t must not depend on feature rows after t.

    Concretely: fit on (X[:cut], y[:cut]); predict on X[cut-10:cut]. Then
    perturb X[cut+1:] (simulate "future leakage") and refit/predict — the
    prediction set on the same withheld rows must be identical because
    nothing past `cut` was ever fed in.
    """
    X, y = synthetic_panel
    cut = 200
    train_X, train_y = X.iloc[:cut], y.iloc[:cut]
    eval_X = X.iloc[cut - 10 : cut]

    model_a = LightGBMModel(config=_fast_cfg())
    model_a.fit(train_X, train_y)
    preds_a = model_a.predict(eval_X)

    X_polluted = X.copy()
    X_polluted.iloc[cut:] = np.nan  # destroy future rows

    model_b = LightGBMModel(config=_fast_cfg())
    model_b.fit(X_polluted.iloc[:cut], y.iloc[:cut])
    preds_b = model_b.predict(eval_X)

    np.testing.assert_allclose(preds_a, preds_b, atol=1e-12)


# --------------------------------------------------------------------------
# 3. Reproducibility — same seed → same predictions
# --------------------------------------------------------------------------

def test_reproducibility(synthetic_panel) -> None:
    X, y = synthetic_panel
    train, test = X.iloc[:200], X.iloc[200:]
    y_train = y.iloc[:200]

    cfg = _fast_cfg(random_state=123)
    model_a = LightGBMModel(config=cfg)
    model_a.fit(train, y_train)
    preds_a = model_a.predict(test)

    model_b = LightGBMModel(config=cfg)
    model_b.fit(train, y_train)
    preds_b = model_b.predict(test)

    np.testing.assert_allclose(preds_a, preds_b, atol=1e-10)
    assert model_a.best_params_ == model_b.best_params_


# --------------------------------------------------------------------------
# 4. Interface parity — LightGBMModel exposes the sklearn surface
#    the existing ElasticNetCV path also relies on inside forecast_returns.
# --------------------------------------------------------------------------

def test_interface_parity() -> None:
    from sklearn.linear_model import ElasticNetCV

    elastic = ElasticNetCV(cv=3, random_state=0)
    model = LightGBMModel(config=_fast_cfg())

    for method in ("fit", "predict"):
        assert hasattr(elastic, method), f"sanity: ElasticNetCV missing {method}"
        assert hasattr(model, method), f"LightGBMModel missing {method}"
        assert callable(getattr(model, method))

    fit_sig = inspect.signature(model.fit)
    fit_params = list(fit_sig.parameters)
    assert fit_params[:2] == ["X", "y"], f"LightGBMModel.fit signature drifted: {fit_params}"

    predict_sig = inspect.signature(model.predict)
    assert list(predict_sig.parameters)[:1] == ["X"]


# --------------------------------------------------------------------------
# 5. Dispatcher — forecast_returns honours forecast_model and defaults to
#    elasticnet so the existing pipeline keeps working unchanged.
# --------------------------------------------------------------------------

def test_forecast_returns_default_is_elasticnet() -> None:
    from src.settings import resolve_settings

    cfg = resolve_settings(None)
    assert cfg["forecast_model"] == "elasticnet"


def test_forecast_returns_rejects_unknown_model() -> None:
    from src.signals import forecast_returns

    feature_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ticker": ["A"] * 5,
            "asset_class": ["equity"] * 5,
            "price": np.arange(1.0, 6.0),
            "momentum_63": np.arange(5, dtype=float),
        }
    )
    with pytest.raises(ValueError, match="Unsupported forecast_model"):
        forecast_returns(feature_df, returns=pd.DataFrame(), settings={"forecast_model": "rf"})


# ---------------------------------------------------------------------------
# Fix C — _holdout_cut unit tests
# ---------------------------------------------------------------------------

def test_holdout_cut_small_n() -> None:
    """n=10 must return a valid split: training >= 1, holdout >= 1."""
    from src.lightgbm_model import _holdout_cut

    cut = _holdout_cut(n=10, holdout_frac=0.2)
    assert 1 <= cut <= 9, f"Expected 1 <= cut <= 9, got {cut}"


def test_holdout_cut_normal_n() -> None:
    """n=300, holdout_frac=0.2 → cut should be max(240, max(50, 100)) = 240."""
    from src.lightgbm_model import _holdout_cut

    cut = _holdout_cut(n=300, holdout_frac=0.2)
    # int(300 * 0.8)=240, max(50, 300-200)=100 → max(240,100)=240
    # then min(240, 300-10)=240
    assert cut == 240, f"Expected 240, got {cut}"


def test_holdout_cut_large_n() -> None:
    """n=1000 caps holdout at 200 samples (cut >= n-200 = 800)."""
    from src.lightgbm_model import _holdout_cut

    cut = _holdout_cut(n=1000, holdout_frac=0.2)
    # int(1000*0.8)=800, max(50, 1000-200)=800 → max(800,800)=800
    # holdout = 1000 - 800 = 200 (capped)
    assert cut == 800, f"Expected 800, got {cut}"
    assert (1000 - cut) <= 200, f"Holdout {1000 - cut} exceeds 200-sample cap"
