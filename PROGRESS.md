# FMIA — PROGRESS

## Step 1: XGBoost alternative model

Status: IN_PROGRESS
Last updated: 2026-05-09T00:00:00Z
Last commit hash: bd371aa (pre-Step-1 baseline)

### Checkpoints
- [x] 1.1 Discovery scan complete
- [x] 1.2 xgboost added to dependencies
- [x] 1.3 XGBoostModel class implemented (interface parity with ElasticNet)
- [x] 1.4 Hyperparameter search (RandomizedSearchCV + TimeSeriesSplit) wired in
- [ ] 1.5 Config flag for model selection (elasticnet | xgboost)
- [ ] 1.6 Backtest runs end-to-end with model=xgboost
- [ ] 1.7 Comparison report generated (reports/step1_xgboost_vs_elasticnet.md)
- [ ] 1.8 Tests added and all tests passing
- [ ] 1.9 Final commit + Step 1 closed

### Notes / decisions
- Existing baseline: `forecast_returns` in `src/signals.py` uses `ElasticNetCV` inline (no class wrapper). Keeping that path unchanged via a thin private helper; adding a parallel `_fit_predict_xgboost` helper plus a new `XGBoostModel` class in `src/xgboost_model.py`.
- Model selector lives in `DEFAULT_SETTINGS["forecast_model"]` (`"elasticnet"` default) and is plumbed via `config.yaml` + a new `--model` flag in `scripts/run_all.py`. The hyperopt-tunable allowlist is NOT extended — model choice is structural, not tunable.
- Inner-CV scheme: `TimeSeriesSplit(n_splits=5)` inside the training window only. Two scorers exposed: `neg_mean_squared_error` (default) and a Spearman IC scorer (set `forecast_xgb_scoring: "ic"`).
- Early stopping: enabled with `early_stopping_rounds=50`; the search picks `n_estimators` implicitly via `best_iteration`.
- Comparison report uses the locally-cached Bloomberg parquet data (real ~9y panel) so the side-by-side is meaningful.
- Combined 1.3 and 1.4 in a single commit — they're a tightly coupled unit (the inner CV search lives inside `XGBoostModel.fit`); splitting would require an unusable intermediate state.
- macOS env note: xgboost wheel needs `libomp.dylib`. Installed user-local Homebrew at `~/homebrew` and patched the venv-shipped `libxgboost.dylib` rpath with `install_name_tool -add_rpath ~/homebrew/opt/libomp/lib …`. Re-running this is required if the venv is rebuilt; not part of the FMIA codebase.

### Next step on resume
Wire `cfg["forecast_model"]` dispatcher into `src/signals.py:forecast_returns`, add `forecast_model` (+ `forecast_xgb_*` keys) to `DEFAULT_SETTINGS`, surface as `forecast_model` in `config.yaml`, and add `--model` flag to `scripts/run_all.py`.
