"""Instrumented smoke: shows TIME per pipeline stage to find LightGBM bottleneck."""
import time
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Monkey-patch forecast_returns to time each rebalance fit_predict
from src import signals as _signals
_orig_fit_predict_lightgbm = _signals._fit_predict_lightgbm

_call_count = {"n": 0, "total_t": 0.0}

def _timed_fit_predict_lightgbm(X_train, y_train, X_pred, cfg):
    t0 = time.time()
    out = _orig_fit_predict_lightgbm(X_train, y_train, X_pred, cfg)
    dt = time.time() - t0
    _call_count["n"] += 1
    _call_count["total_t"] += dt
    if _call_count["n"] <= 10 or _call_count["n"] % 10 == 0:
        print(f"[fit_predict #{_call_count['n']:3d}] X_train={X_train.shape}  X_pred={X_pred.shape}  took {dt:.2f}s  (avg {_call_count['total_t']/_call_count['n']:.2f}s)", flush=True)
    return out

_signals._fit_predict_lightgbm = _timed_fit_predict_lightgbm

from src.pipeline import run_pipeline
from src.settings import resolve_settings

cfg = resolve_settings({"forecast_model": "lightgbm", "bootstrap_enabled": False})

t0 = time.time()
print("Starting instrumented smoke (LightGBM, hedge=False, reform=False, 1 year mock)...", flush=True)

results = run_pipeline(
    data_source="mock",
    start_date="2023-01-01",
    end_date="2024-12-31",
    hedge_mode=None,
    hedge_reform=False,
    optimizer="mv",
    settings=cfg,
)

elapsed = time.time() - t0
print(f"\nPipeline done in {elapsed:.1f}s", flush=True)
print(f"Total fit_predict calls: {_call_count['n']}", flush=True)
print(f"Avg time/fit_predict:    {_call_count['total_t']/max(_call_count['n'],1):.2f}s", flush=True)
print(f"Pipeline overhead:       {elapsed - _call_count['total_t']:.1f}s", flush=True)

metrics = results["backtest"]["metrics"]
print(f"\nMetrics:", flush=True)
print(f"  Sharpe:   {metrics['sharpe']:.3f}", flush=True)
print(f"  Turnover: {metrics['turnover']:.4f}", flush=True)
