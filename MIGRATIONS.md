# FMIA — Migration Log

This file documents breaking changes, model replacements, and data/leakage fixes that
altered backtest results.  Any archive of reports or metrics produced **before** a
migration in this log is not comparable to current output without re-running the pipeline.

---

## M-1 · Walk-forward training leakage fix

**When:** Working-tree change (current branch, not yet tagged).
**Symptom:** `reports/output/_pre_leakage_fix/` contains the last HTML reports
produced with the buggy code.

### What the bug was

`forecast_returns` in `src/signals.py` built the training mask as:

```python
train_mask = class_df["_fwd_return"].notna()
```

This included rows whose **forward-return target** was computed from prices observed
**after** the decision date.  Concretely: a row dated 2022-11-01 with a 21-day forward
return required prices up to 2022-11-22.  If the decision date was 2022-11-15, that row
leaked future price information into the training set.

### What the fix is

Added a PIT cutoff that ensures only rows whose full forward-return window has been
realized are included in training:

```python
cutoff = pd.Timestamp(date) - pd.tseries.offsets.BDay(forward_days)
train_mask = (class_df["date"] <= cutoff) & class_df["_fwd_return"].notna()
```

Now a training row at date `t'` is only included when `t' + forward_days ≤ decision_date`.

### Impact on results

The pre-fix reports (now archived in `reports/output/_pre_leakage_fix/`) overestimate
Sharpe and IC metrics by an unknown amount — the magnitude depends on how strongly
near-future prices were correlated with the features.  **Do not use those reports for
investment decisions or for performance comparisons.**

### Tests that guard against regression

- `tests/test_signals_leakage.py::test_train_mask_excludes_lookahead_window`
- `tests/test_signals_leakage.py::test_forecast_returns_runs_without_lookahead`
- `tests/test_walkforward_integrity.py::test_backtest_weights_invariant_to_future_prices`

---

## M-2 · XGBoost → LightGBM model replacement

**When:** Working-tree change (current branch, not yet tagged).

### Motivation

LightGBM was found to train 3–5× faster on the FMIA cross-section (small-N, wide-feature
regime) and produces comparable or slightly better out-of-sample IC.  Key improvements:

- Native categorical support and built-in `num_leaves` tuning.
- `min_child_samples` replaces `min_child_weight` as the leaf-size regulariser
  (more interpretable on small datasets).
- The Python API allows early-stopping directly in `fit()` without a custom wrapper.

### Files changed

| Before | After |
|--------|-------|
| `src/xgboost_model.py` (deleted) | `src/lightgbm_model.py` |
| `tests/test_xgboost_model.py` | `tests/test_lightgbm_model.py` |
| `reports/step1_xgboost_vs_elasticnet.md` | `reports/step1_lightgbm_vs_elasticnet.md` |
| Config keys `forecast_xgb_*` | Config keys `forecast_lgbm_*` |
| `SUPPORTED_FORECAST_MODELS = ("elasticnet", "xgboost")` | `("elasticnet", "lightgbm")` |

### Breaking changes for existing configs

Any `config.yaml` or settings dict that sets `forecast_xgb_*` keys will be silently
ignored (the old keys are no longer in `DEFAULT_SETTINGS`).  `resolve_settings` now logs
a warning for unknown keys.  **Rename all occurrences**:

| Old key | New key |
|---------|---------|
| `forecast_xgb_n_iter` | `forecast_lgbm_n_iter` |
| `forecast_xgb_cv_splits` | `forecast_lgbm_cv_splits` |
| `forecast_xgb_n_estimators_cap` | `forecast_lgbm_n_estimators_cap` |
| `forecast_xgb_early_stopping_rounds` | `forecast_lgbm_early_stopping_rounds` |
| `forecast_xgb_scoring` | `forecast_lgbm_scoring` |
| `forecast_xgb_random_state` | `forecast_lgbm_random_state` |
| `forecast_xgb_n_jobs` | `forecast_lgbm_n_jobs` |
| `forecast_xgb_search_n_jobs` | `forecast_lgbm_search_n_jobs` |

Also set `forecast_model: lightgbm` (was `xgboost`) in any persisted configs.

### Hyperparameter search space differences

XGBoost used `min_child_weight` and `subsample`; LightGBM uses `min_child_samples`
and `num_leaves` instead.  Any saved Optuna studies from the XGBoost era are **not
compatible** with the LightGBM search space and should be discarded.  Re-run
`scripts/run_hyperopt.py` to build a fresh LightGBM study.

### Archive of pre-migration reports

The XGBoost-era HTML strategy reports are preserved in `reports/output/_pre_leakage_fix/`
(they overlap with the pre-leakage-fix archive — both leakage fix and model swap
happened before the next commit).  File names containing `xgboost` in that folder
are the XGBoost-era outputs.  Do not compare them to current LightGBM outputs.

---

## Reporting any new migration

Add a new `## M-N · Title` section to this file whenever:
- A model is replaced or its training procedure changes materially.
- A data leakage or look-ahead bug is discovered and fixed.
- A config key rename breaks backward compatibility.
- Backtest weights would change for a given input panel.

Include: what changed, why, which files, whether old reports are invalidated, and
the tests that guard against regression.
