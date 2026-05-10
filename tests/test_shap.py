"""Tests for src/shap_attribution.py and the compute_shap gate in forecast_returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# 2.8.1  SHAP output shape
# ---------------------------------------------------------------------------

def test_shap_output_shape():
    """collect_rebalance_shap returns (n_test_obs * n_features) records."""
    pytest.importorskip("shap")
    from src.xgboost_model import XGBoostModel
    from src.shap_attribution import collect_rebalance_shap

    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 120, 8, 5
    feat_names = [f"f{i}" for i in range(n_feat)]

    X_train = pd.DataFrame(rng.standard_normal((n_train, n_feat)), columns=feat_names)
    y_train = pd.Series(rng.standard_normal(n_train))
    X_test  = pd.DataFrame(rng.standard_normal((n_test,  n_feat)), columns=feat_names)
    tickers = [f"T{i}" for i in range(n_test)]

    model = XGBoostModel(config={"n_iter": 2, "cv_splits": 2, "n_estimators_cap": 50,
                                 "early_stopping_rounds": 5})
    model.fit(X_train, y_train)

    records = collect_rebalance_shap(model, X_test, pd.Timestamp("2021-01-31"), tickers, feat_names)
    assert len(records) == n_test * n_feat
    assert set(records[0].keys()) >= {"date", "ticker", "feature", "shap_value", "feature_value"}

    sv_array = np.array([r["shap_value"] for r in records]).reshape(n_test, n_feat)
    assert sv_array.shape == (n_test, n_feat)
    assert np.all(np.isfinite(sv_array))


# ---------------------------------------------------------------------------
# 2.8.2  Parquet schema
# ---------------------------------------------------------------------------

def test_shap_parquet_schema(tmp_path):
    """write_shap_parquet → load_shap_parquet round-trip preserves schema."""
    from src.shap_attribution import write_shap_parquet, load_shap_parquet

    records = [
        {"date": pd.Timestamp("2021-01-31"), "ticker": "AMXL",
         "feature": "momentum_63", "shap_value": 0.001, "feature_value": 0.5},
        {"date": pd.Timestamp("2021-01-31"), "ticker": "AMXL",
         "feature": "pe_ratio", "shap_value": -0.002, "feature_value": 14.0},
    ]
    pq = tmp_path / "shap_test.parquet"
    write_shap_parquet(records, pq)
    df = load_shap_parquet(pq)

    assert set(df.columns) >= {"date", "ticker", "feature", "shap_value", "feature_value"}
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["shap_value"].dtype == np.float64
    assert df["feature_value"].dtype == np.float64
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 2.8.3  Stability metric range
# ---------------------------------------------------------------------------

def test_stability_metric_range(tmp_path):
    """All Spearman rank-correlation scores must be in [-1, 1]."""
    from src.shap_attribution import compute_stability

    # Build a small synthetic SHAP DataFrame with 10 rebalances, 4 tickers, 5 features
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-31", periods=10, freq="ME")
    tickers = ["A", "B", "C", "D"]
    features = ["f0", "f1", "f2", "f3", "f4"]

    rows = []
    for d in dates:
        for t in tickers:
            for f in features:
                rows.append({
                    "date": d, "ticker": t, "feature": f,
                    "shap_value": rng.standard_normal(),
                    "feature_value": rng.standard_normal(),
                })
    shap_df = pd.DataFrame(rows)
    shap_df["date"] = pd.to_datetime(shap_df["date"])

    stab = compute_stability(shap_df, k_values=(3, 5, None))
    for k, corrs in stab.items():
        assert len(corrs) > 0, f"No pairs for k={k}"
        for rho in corrs:
            assert -1.0 <= rho <= 1.0, f"Out-of-range Spearman {rho} for k={k}"


# ---------------------------------------------------------------------------
# 2.8.4  compute_shap flag — no parquet written when False
# ---------------------------------------------------------------------------

def test_compute_shap_flag(tmp_path):
    """forecast_returns with compute_shap=False must not write any parquet."""
    from src.signals import forecast_returns
    import numpy as np
    import pandas as pd

    shap_path = tmp_path / "should_not_exist.parquet"

    # Minimal synthetic feature_df: enough rows for at least one rebalance fit
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2019-01-31", periods=n, freq="D")
    tickers = ["A", "B", "C", "D"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "asset_class": "equity",
                "price": float(rng.uniform(10, 100)),
                "momentum_63": float(rng.standard_normal()),
                "volatility_63": float(rng.uniform(0.01, 0.3)),
                "pe_ratio": float(rng.uniform(8, 30)),
                "pb_ratio": float(rng.uniform(0.5, 4)),
            })
    feature_df = pd.DataFrame(rows)

    _ = forecast_returns(
        feature_df,
        returns=pd.DataFrame(),
        settings={
            "forecast_model": "xgboost",
            "compute_shap": False,
            "shap_output_path": str(shap_path),
            "forecast_xgb_n_iter": 1,
            "forecast_xgb_cv_splits": 2,
            "forecast_xgb_n_estimators_cap": 20,
            "forecast_xgb_early_stopping_rounds": 3,
        },
    )
    assert not shap_path.exists(), "Parquet must NOT be written when compute_shap=False"
