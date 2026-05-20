"""Minimal smoke test: pipeline with LightGBM, no hedge/reform, short window.

Validates the LightGBM integration end-to-end without the cost of the full
hedge + reform comparison.  Prints metrics + wall-time so we can compare
against the prior XGBoost baseline.
"""
import time
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from src.pipeline import run_pipeline
from src.settings import resolve_settings

cfg = resolve_settings({"forecast_model": "lightgbm", "bootstrap_enabled": False})

t0 = time.time()
print("Starting mock smoke pipeline (LightGBM, hedge=False, reform=False, 2020-2024)...", flush=True)

results = run_pipeline(
    data_source="mock",
    start_date="2022-01-01",
    end_date="2024-12-31",
    hedge_mode=None,
    hedge_reform=False,
    optimizer="mv",
    settings=cfg,
)

elapsed = time.time() - t0
print(f"\nPipeline done in {elapsed:.1f}s", flush=True)

metrics = results["backtest"]["metrics"]
print(f"  Sharpe:   {metrics['sharpe']:.3f}", flush=True)
print(f"  Return:   {metrics['annualized_return']:.3f}", flush=True)
print(f"  Vol:      {metrics['annualized_vol']:.3f}", flush=True)
print(f"  Max DD:   {metrics['max_drawdown']:.3f}", flush=True)
print(f"  Turnover: {metrics['turnover']:.4f}", flush=True)
sys.exit(0)
