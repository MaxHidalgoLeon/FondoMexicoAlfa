# FMIA — PROGRESS

## Step 1: XGBoost alternative model

Status: DONE
Last updated: 2026-05-10T00:00:00Z
Last commit hash: b145fee

### Checkpoints
- [x] 1.1 Discovery scan complete
- [x] 1.2 xgboost added to dependencies
- [x] 1.3 XGBoostModel class implemented (interface parity with ElasticNet)
- [x] 1.4 Hyperparameter search (RandomizedSearchCV + TimeSeriesSplit) wired in
- [x] 1.5 Config flag for model selection (elasticnet | xgboost)
- [x] 1.6 Backtest runs end-to-end with model=xgboost
- [x] 1.7 Comparison report generated (reports/step1_xgboost_vs_elasticnet.md)
- [x] 1.8 Tests added and all tests passing
- [x] 1.9 Final commit + Step 1 closed

### Notes / decisions
- Existing baseline: `forecast_returns` in `src/signals.py` uses `ElasticNetCV` inline (no class wrapper). Keeping that path unchanged via a thin private helper; adding a parallel `_fit_predict_xgboost` helper plus a new `XGBoostModel` class in `src/xgboost_model.py`.
- Model selector lives in `DEFAULT_SETTINGS["forecast_model"]` (`"elasticnet"` default) and is plumbed via `config.yaml` + a new `--model` flag in `scripts/run_all.py`. The hyperopt-tunable allowlist is NOT extended — model choice is structural, not tunable.
- Inner-CV scheme: `TimeSeriesSplit(n_splits=5)` inside the training window only. Two scorers exposed: `neg_mean_squared_error` (default) and a Spearman IC scorer (set `forecast_xgb_scoring: "ic"`).
- Early stopping: enabled with `early_stopping_rounds=50`; the search picks `n_estimators` implicitly via `best_iteration`.
- Comparison report runs on mock data (108 monthly rebalances). Bloomberg side-by-side is blocked on a pre-existing `KeyError: 'pe_ratio'` in `src/features.py:194` triggered when `load_data(source='bloomberg')` is called outside `run_pipeline`'s preprocessing — out of scope for Step 1; flagged as a Step 2 prerequisite.
- Combined 1.3 and 1.4 in a single commit — they're a tightly coupled unit (the inner CV search lives inside `XGBoostModel.fit`); splitting would require an unusable intermediate state.
- macOS env note: xgboost wheel needs `libomp.dylib`. Installed user-local Homebrew at `~/homebrew` and patched the venv-shipped `libxgboost.dylib` rpath with `install_name_tool -add_rpath ~/homebrew/opt/libomp/lib …`. Re-running this is required if the venv is rebuilt; not part of the FMIA codebase.

### Acceptance check (all green)
- [x] `pytest tests/ -q` → 93 passed (was 87 + 6 new XGBoost tests).
- [x] `python scripts/run_all.py --model elasticnet …` baseline path untouched.
- [x] `python scripts/run_all.py --model xgboost …` runs end-to-end (verified
       via `run_pipeline(settings={'forecast_model':'xgboost', ...})` with
       all 8 standard result keys produced).
- [x] `reports/step1_xgboost_vs_elasticnet.md` exists with full comparison
       table, equity curve PNG, IC time-series PNG, and written interpretation.
- [x] No leakage (`test_no_lookahead`), fixed seeds
       (`test_reproducibility` shows max abs diff = 0), deterministic outputs.

### Next step on resume
Step 2 — see below.

---

## Step 2: SHAP attribution

Status: DONE
Last updated: 2026-05-10T00:00:00Z
Last commit hash: 0767588

### Checkpoints
- [x] 2.1 Discovery scan complete + shap in requirements.txt
- [x] 2.2 Per-rebalance SHAP collection wired into walk-forward loop
- [x] 2.3 data/shap_values.parquet written on a test run
- [x] 2.4 Feature stability metric computed + shap_stability.csv
- [x] 2.5 Turnover driver analysis complete
- [x] 2.6 SHAP vs ElasticNet comparison figure (documented skip — ElasticNet coef_ not stored)
- [x] 2.7 SHAP report (reports/step2_shap_analysis.md + figures)
- [x] 2.8 Tests added + 97 tests passing (93 existing + 4 new SHAP tests)
- [x] 2.9 Final commit + Step 2 closed

### Notes / decisions
- Architecture: SHAP computed inline in `forecast_returns` walk-forward loop via an explicit `XGBoostModel` instantiation when `model_name=="xgboost"` and `compute_shap=True`. The `fit_predict` function-pointer pattern is bypassed for this path so the model object is accessible before being discarded.
- `XGBoostModel` gets a new `scale(X)` method exposing the internal `StandardScaler.transform` so `shap_attribution.py` can produce SHAP values on the correct (scaled) feature space without coupling shap to xgboost_model.py.
- ElasticNet per-rebalance coefficients are not stored anywhere — `_fit_predict_elasticnet` discards the fitted model. The feature-importance comparison panel (§2.4) is therefore XGBoost-only; this is noted in the report.
- `TreeExplainer` is reconstructed each rebalance (not persisted). Acceptable given universe size (~26 equities + FIBRAs).
- SHAP parquet is overwritten on each run (not appended). Gate: `compute_shap: true` in settings.

### Next step on resume
Step 3 — see below.

---

## Step 3: Macro regime performance

Status: DONE
Last updated: 2026-05-10T00:00:00Z
Last commit hash: 44e03ce

### Checkpoints
- [x] 3.1 Discovery scan complete
- [x] 3.2 Macro data located / loaded (banxico_rate in mock macro; IPC proxy = EW equity returns)
- [x] 3.3 src/macro_regimes.py implemented + unit tested
- [x] 3.4 Regime table built and saved (reports/regime_table.csv)
- [x] 3.5 Regime-conditioned performance table computed (reports/regime_performance_table.csv)
- [x] 3.6 Regime-conditioned SHAP stability computed
- [x] 3.7 Momentum SHAP by regime computed
- [x] 3.8 All figures generated (step3_regime_equity_curves.png, step3_ic_by_regime.png)
- [x] 3.9 reports/step3_regime_analysis.md complete
- [x] 3.10 Tests added + 101 passing (97 existing + 4 new regime tests)
- [x] 3.11 Final commit + Step 3 closed

### Notes / decisions
- Banxico rate already in mock macro DataFrame (field: banxico_rate). No lookahead: regime at rebalance t uses macro data up to end of month t-1.
- IPC proxy = equal-weighted daily return of equity sub-universe (no IPC index ticker exists). 60d rolling std × sqrt(252) = annualised IPC vol.
- Stress threshold = 75th percentile of all 108 computed IPC vols. Research-only; documented here.
- Rate regime: 3-month trailing change in banxico_rate. TIGHTENING > 0, EASING < 0, NEUTRAL = 0.
- Per-rebalance IC computed from forecast_df.expected_return vs realized 21d log forward returns from prices.
- Per-regime portfolio metrics from backtest.returns sliced to rebalance windows within that regime.

### Next step on resume
Step 4 — see below.

---

## Step 4: PDF Tearsheet

Status: DONE
Last updated: 2026-05-10T00:00:00Z
Last commit hash: (see final commit)

### Checkpoints
- [x] 4.1 Discovery scan + file inventory complete
- [x] 4.2 weasyprint + fpdf2 added to requirements.txt
- [x] 4.3 HTML structure scaffolded (all 5 pages, placeholder content)
- [x] 4.4 Page 1 (cover) complete
- [x] 4.5 Page 2 (model comparison) complete with figures
- [x] 4.6 Page 3 (SHAP attribution) complete with figures
- [x] 4.7 Page 4 (regime analysis) complete with figures
- [x] 4.8 Page 5 (risk + methodology) complete
- [x] 4.9 PDF renders without errors, file size < 15 MB (534 KB)
- [x] 4.10 Tests added + 104 passing (101 existing + 3 new tearsheet tests)
- [x] 4.11 Final commit + Step 4 closed

### Notes / decisions
- WeasyPrint ≥ 60 still requires pango/gobject system libs (unavailable on this macOS env without Xcode CLT update). fpdf2 2.8.7 used as pure-Python fallback.
- All PNGs embedded as base64 data URIs; no external file refs in HTML.
- Source data: parse step1 MD for comparison table; load regime_performance_table.csv; parse step2 MD for top-10 features and stability tables.
- Page layout: A4 landscape, dark theme, CSS page counters for footer.
- fpdf2 normalize_text override applied to handle Unicode chars (em-dash, rho, Delta, etc.) that latin-1 core fonts cannot encode.

### Next step on resume
Step 4 DONE. No further steps defined.

---

## Step 5: Docstrings and documentation

Status: DONE
Last updated: 2026-05-10T00:00:00Z
Last commit hash: 07cc0a1

### Checkpoints
- [x] 5.1 Discovery scan + file inventory complete
- [x] 5.2 src/xgboost_model.py documented
- [x] 5.3 src/shap_attribution.py documented
- [x] 5.4 src/macro_regimes.py documented
- [x] 5.5 All scripts/ files documented (module-level + non-obvious helpers)
- [x] 5.6 104 tests passing
- [x] 5.7 Final commit + Step 5 closed

### Bugs found — deferred to Step 6
None found.

### Notes / decisions
- Google-style docstrings throughout (Args, Returns, Raises).
- NumPy-style docstrings in shap_attribution.py and macro_regimes.py converted to Google style.
- scripts/ received module-level docstrings (already present) + one-liner docstrings on all non-obvious private helpers and main() entry points.
- Inner fpdf2 class methods in render_tearsheet.py documented inline.
- Zero logic changes.

### Next step on resume
Step 5 DONE. No further steps defined.

---

## Step 6: Code cleanup

Status: DONE
Last updated: 2026-05-13T00:00:00Z
Last commit hash: e3ab9ab

### Checkpoints
- [x] 6.A _render_pdf_fpdf2 refactored into page builders
- [x] 6.B Named constants in render_step3_report.py
- [x] 6.C _holdout_cut extracted + 3 unit tests
- [x] 6.D SHAP error handling consolidated
- [x] 6.E fpdf2 ln= deprecation warnings eliminated
- [x] 6.F Final: 107 tests passing, 0 DeprecationWarnings
- [x] 6.G chore(progress): close Step 6

### Notes / decisions
- Fix A: Colour palette moved to _PDF_BG/_PDF_SURF/etc. module-level constants. Inner closures became _pdf_set_primary/muted/accent, _pdf_section_header, _pdf_add_img_safe. Five page builders: _pdf_page_cover/model_comparison/shap/regime/risk_methodology. _render_pdf_fpdf2 is now a 20-line orchestrator.
- Fix B: SHAP_STABILITY_FLOOR=0.30, SIGNAL_SHRINKAGE_TIGHTENING=0.70 at module level in render_step3_report.py.
- Fix C: _holdout_cut(n, holdout_frac) extracted with guard for n<=1. Three new unit tests added to test_xgboost_model.py.
- Fix D: Split the monolithic try/except in signals.py so fit() failure alone triggers continue; collect_rebalance_shap call is now unwrapped (it never raises).
- Fix E: All ln=True/False replaced with XPos/YPos enums. Zero DeprecationWarning lines confirmed with python -W error::DeprecationWarning.

### Next step on resume
Step 6 DONE. No further steps defined.
