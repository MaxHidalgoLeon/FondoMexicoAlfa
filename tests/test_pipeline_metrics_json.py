"""Unit tests for pipeline._emit_metrics_json — the single source of truth
that downstream renderers consume."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import _emit_metrics_json, _metrics_path

ROOT = Path(__file__).resolve().parent.parent


def _fake_results() -> dict:
    """Build a minimal results dict shaped like run_pipeline's return value."""
    regime_history = pd.DataFrame({
        "regime": ["expansion"],
        "regime_confidence": [0.78],
    }, index=[pd.Timestamp("2024-12-31")])
    return {
        "backtest": {
            "metrics": {
                "sharpe": 0.31,
                "sortino": 0.34,
                "max_drawdown": -0.28,
                "annualized_return": 0.0612,
                "annualized_vol": 0.1502,
                "cvar_95": -0.018,
                "turnover": 0.0062,
            },
            "metrics_ci": {
                "sharpe": {"point": 0.31, "ci_low": -0.15, "ci_high": 0.77},
            },
            "regime_history": regime_history,
        },
        "data": {
            "liquidity_sleeve": {"regime": "expansion", "total_sleeve": 0.04},
        },
        "hedge_layer": None,
    }


def test_emit_metrics_json_writes_file_with_payload(tmp_path, monkeypatch):
    # Redirect _metrics_path output into tmp by chdir'ing the cwd.
    monkeypatch.chdir(tmp_path)
    results = _fake_results()
    _emit_metrics_json(results, data_source="bloomberg", forecast_model="elasticnet", end_date="2026-03-31")

    out = Path("reports/output/metrics_bloomberg_elasticnet.json")
    assert out.exists(), "Metrics JSON was not emitted."
    payload = json.loads(out.read_text())
    assert payload["source"] == "bloomberg"
    assert payload["model"] == "elasticnet"
    assert payload["as_of"] == "2026-03-31"
    assert abs(payload["metrics"]["sharpe"] - 0.31) < 1e-9
    assert payload["metrics"]["turnover"] == 0.0062
    assert payload["regime_last"]["regime"] == "expansion"
    assert payload["liquidity_sleeve"]["total_sleeve"] == 0.04


def test_emit_metrics_json_survives_missing_optional_keys(tmp_path, monkeypatch):
    """Backtest without regime_history / hedge_layer must still emit a valid JSON."""
    monkeypatch.chdir(tmp_path)
    minimal = {"backtest": {"metrics": {"sharpe": 0.1}}, "data": {}, "hedge_layer": None}
    _emit_metrics_json(minimal, data_source="yahoo", forecast_model="lightgbm", end_date="2026-03-31")
    payload = json.loads(Path("reports/output/metrics_yahoo_lightgbm.json").read_text())
    assert payload["metrics"]["sharpe"] == 0.1
    assert payload["regime_last"] == {}
    assert payload["liquidity_sleeve"] == {}


def test_metrics_path_is_per_source_model():
    p1 = _metrics_path("bloomberg", "elasticnet")
    p2 = _metrics_path("bloomberg", "lightgbm")
    assert p1 != p2
    assert p1.endswith("metrics_bloomberg_elasticnet.json")
