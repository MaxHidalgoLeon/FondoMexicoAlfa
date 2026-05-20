"""Smoke tests for scripts/render_research_report.py.

These verify the renderer reads metrics JSONs, rebuilds Table 1, and substitutes
the headline KPIs in the Abstract — all without invoking the heavy pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "render_research_report.py"
OUTPUT = ROOT / "reports" / "FMIA_Research_Report.md"


def _write_metrics(source: str, model: str, sharpe: float, ret: float, vol: float, mdd: float,
                   sortino: float = 0.45, cvar: float = -0.02, turnover: float = 0.005) -> Path:
    path = ROOT / "reports" / "output" / f"metrics_{source}_{model}.json"
    payload = {
        "source": source, "model": model, "as_of": "2026-03-31",
        "metrics": {
            "sharpe": sharpe, "sortino": sortino,
            "annualized_return": ret, "annualized_vol": vol,
            "max_drawdown": mdd, "cvar_95": cvar, "turnover": turnover,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def md_backup():
    """Snapshot the existing FMIA_Research_Report.md so we can restore after the test."""
    saved = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else None
    yield
    if saved is not None:
        OUTPUT.write_text(saved, encoding="utf-8")


@pytest.fixture
def metrics_cleanup():
    """Remove any test-generated metrics JSON files after the test."""
    created: list[Path] = []
    yield created
    for p in created:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def test_renderer_substitutes_abstract_kpis(md_backup, metrics_cleanup):
    """Abstract KPIs must come from the JSON, not from a hardcoded literal."""
    p = _write_metrics("bloomberg", "elasticnet", sharpe=0.27, ret=0.0612, vol=0.1502, mdd=-0.281)
    metrics_cleanup.append(p)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", "bloomberg", "--model", "elasticnet"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, f"renderer failed: {result.stderr}"
    text = OUTPUT.read_text(encoding="utf-8")
    assert "Sharpe ratio of 0.27" in text
    assert "6.12% at 15.02%" in text


def test_renderer_rebuilds_table_1(md_backup, metrics_cleanup):
    """Table 1 row for the active (source, model) reflects the new JSON values."""
    p = _write_metrics("bloomberg", "elasticnet", sharpe=0.99, ret=0.123, vol=0.456,
                       mdd=-0.789, sortino=1.01, cvar=-0.04, turnover=0.123)
    metrics_cleanup.append(p)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", "bloomberg", "--model", "elasticnet"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, f"renderer failed: {result.stderr}"
    text = OUTPUT.read_text(encoding="utf-8")
    # The new row should appear in Table 1 with the values we just wrote.
    assert "| Bloomberg | ElasticNetCV | 12.30%" in text or "| Bloomberg | ElasticNetCV | 12.30 %" in text


def test_renderer_fails_without_metrics(md_backup, metrics_cleanup):
    """Missing metrics JSON must produce a non-zero exit, not phantom numbers."""
    target = ROOT / "reports" / "output" / "metrics_madeup_provider_elasticnet.json"
    if target.exists():
        target.unlink()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--source", "madeup_provider", "--model", "elasticnet"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode != 0, "renderer should fail loudly on missing JSON"
    assert "Missing" in result.stderr or "Missing" in result.stdout
