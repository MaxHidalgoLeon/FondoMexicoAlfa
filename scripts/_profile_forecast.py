"""Profile forecast_returns iteration-by-iteration via module-attribute patch.

CRITICAL: patches src.signals._fit_predict_lightgbm BEFORE forecast_returns is
called, so the binding inside the loop hits the wrapped version.
"""
import time
import warnings
import sys
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Import signals module first to get the original function
import src.signals as _s
_orig_fit_predict = _s._fit_predict_lightgbm

_iter = [0]
_total = [0.0]

def _wrapped(X_tr, y_tr, X_pr, cfg):
    t0 = time.time()
    r = _orig_fit_predict(X_tr, y_tr, X_pr, cfg)
    dt = time.time() - t0
    _iter[0] += 1
    _total[0] += dt
    print(f"iter {_iter[0]:3d}: tr={X_tr.shape}  pr={X_pr.shape}  t={dt:5.2f}s  cum={_total[0]:6.1f}s",
          flush=True)
    return r

# Replace the *module attribute* — forecast_returns binds fit_predict at call time
# from the module namespace, so this needs to happen before forecast_returns is invoked.
_s._fit_predict_lightgbm = _wrapped

# Now import other functions
from src.signals import forecast_returns, score_cross_section
from src.data_loader import load_data
from src.features import build_signal_matrix
from src.settings import resolve_settings

cfg = resolve_settings({
    "forecast_model": "lightgbm",
    "bootstrap_enabled": False,
    "compute_shap": False,  # ensure the else-branch path is taken
})

print("Loading mock + features...", flush=True)
t0 = time.time()
data = load_data(source="mock", start_date="2023-01-01", end_date="2024-12-31")
feat = build_signal_matrix(data["prices"], data["fundamentals"], data["fibra_fundamentals"],
                            data["bonds"], data["macro"], data["universe"])
scored = score_cross_section(feat)
returns = np.log(data["prices"] / data["prices"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
print(f"  setup: {time.time()-t0:.1f}s", flush=True)

print("\nStarting forecast_returns (compute_shap=False)...", flush=True)
t0 = time.time()
forecast = forecast_returns(scored, returns, settings=cfg)
total = time.time() - t0

print(f"\n=== SUMMARY ===", flush=True)
print(f"forecast_returns total:   {total:.1f}s", flush=True)
print(f"Sum of fit_predict times: {_total[0]:.1f}s", flush=True)
print(f"Outer loop overhead:      {total - _total[0]:.1f}s", flush=True)
print(f"Total iterations:         {_iter[0]}", flush=True)
print(f"Avg per iteration:        {total / max(_iter[0],1):.2f}s", flush=True)
