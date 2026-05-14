# FondoMéxicoAlfa (FMIA)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-107%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![XGBoost](https://img.shields.io/badge/model-XGBoost%202.0-red)
![SHAP](https://img.shields.io/badge/attribution-SHAP-blueviolet)

Systematic long-short equity and FIBRA pipeline for Mexican public markets.
Walk-forward cross-sectional return forecasting with XGBoost, SHAP attribution,
and Banxico macro-regime conditioning. All results are out-of-sample.

---

## Abstract

FMIA is a quantitative research pipeline that forecasts cross-sectional returns
across Mexican equities (BMV) and real estate investment trusts (FIBRAs) using
gradient-boosted trees with internal time-series cross-validation. The pipeline
covers the full systematic workflow: data ingestion, feature engineering,
walk-forward signal generation, mean-variance portfolio construction, SHAP-based
attribution, and macro-regime performance decomposition. A PDF tearsheet is
generated automatically at the end of each run.

The model is differentiated by three design choices: (1) FIBRA-specific
fundamental features (LTV, FFO yield, cap rate, vacancy rate) that are absent
from standard equity factor libraries; (2) Banxico rate-regime conditioning that
reveals when the model's feature hierarchy is stable and when it is not; and
(3) an explicit bias-variance tradeoff documented via SHAP stability metrics,
which flags the small-cross-section limitation rather than hiding it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FMIA Pipeline                            │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐
  │   Data   │────▶│   Features   │────▶│  Signal / Forecast  │
  │          │     │              │     │                     │
  │ Bloomberg│     │ Equity:      │     │ ElasticNetCV        │
  │ parquet  │     │  pe_ratio    │     │  (baseline)         │
  │          │     │  roe         │     │                     │
  │ Mock     │     │  momentum_63 │     │ XGBoost             │
  │ fallback │     │  ...         │     │  RandomizedSearchCV │
  └──────────┘     │              │     │  TimeSeriesSplit    │
                   │ FIBRA:       │     │  early stopping     │
                   │  ltv         │     └────────┬────────────┘
                   │  ffo_yield   │              │
                   │  cap_rate    │              ▼
                   │  vacancy_rate│     ┌─────────────────────┐
                   └──────────────┘     │  Walk-Forward OOS   │
                                        │  Backtest           │
                                        │  MV optimizer       │
                                        │  10 bp tx costs     │
                                        └────────┬────────────┘
                                                 │
                          ┌──────────────────────┼──────────────────────┐
                          ▼                      ▼                      ▼
                  ┌──────────────┐    ┌──────────────────┐   ┌────────────────┐
                  │ SHAP Report  │    │  Regime Report   │   │ PDF Tearsheet  │
                  │              │    │                  │   │                │
                  │ TreeExplainer│    │ TIGHTENING       │   │ 5-page A4      │
                  │ per rebalance│    │ EASING           │   │ WeasyPrint /   │
                  │ stability    │    │ NEUTRAL          │   │ fpdf2 fallback │
                  │ turnover     │    │ × STRESS / CALM  │   │                │
                  └──────────────┘    └──────────────────┘   └────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/MaxHidalgoLeon/FondoMexicoAlfa.git
cd FondoMexicoAlfa
pip install -r requirements.txt

# 2. Run full pipeline on mock data (no Bloomberg required)
python scripts/run_all.py --source mock

# 3. Generate SHAP attribution report
python scripts/render_step2_report.py

# 4. Generate regime analysis report
python scripts/render_step3_report.py

# 5. Render PDF tearsheet
python scripts/render_tearsheet.py
# → reports/FMIA_Tearsheet.pdf (534 KB)
```

To run with Bloomberg data (requires a live terminal connection):

```bash
python scripts/run_all.py --source bloomberg --model xgboost
```

---

## Results

All metrics are out-of-sample walk-forward (108 monthly rebalances).
Transaction costs: 10 bp per side. Optimizer: mean-variance.

| Metric | ElasticNetCV | XGBoost | Δ |
|---|---|---|---|
| IC mean (Spearman) | +0.079 | +0.339 | +0.260 |
| ICIR | 0.306 | 1.452 | +1.146 |
| Hit rate | 0.516 | 0.573 | +0.056 |
| Annualized return | +9.95% | +30.89% | +20.94 pp |
| Annualized vol | 7.63% | 7.57% | −0.06 pp |
| Sharpe | 0.981 | 3.292 | +2.311 |
| Sortino | 1.025 | 3.626 | +2.601 |
| Max drawdown | −12.7% | −4.6% | +8.1 pp |
| CVaR 95% (daily) | −0.97% | −0.91% | +0.06 pp |
| Turnover / rebalance | 0.062 | 0.268 | +0.206 |
| Forecast wall time | 13 s | 573 s | 44× slower |

> **Note:** Results above use synthetic mock data (9 tickers, 108 periods).
> Bloomberg-sourced results pending full pe_ratio feature availability across
> the complete universe. Mock results establish the OOS validation framework;
> live-data results will be added in a subsequent update.

---

## Key Findings

**FIBRA signals dominate feature importance.**
LTV, FFO yield, and cap rate are the top-3 features by mean |SHAP| —
collectively 2–3× larger than any equity feature. This reflects the
quarterly reporting cadence of FIBRAs and the direct mechanistic link
between these variables and NAV-based valuation in the Mexican REIT market.

**Model reliability is regime-dependent.**
SHAP feature-rank stability (Spearman) is 0.57 in EASING regimes vs 0.41
in TIGHTENING. IC and Sharpe are higher in EASING periods, consistent with
the hypothesis that falling discount rates amplify cross-sectional dispersion
of fundamental signals. The practical implication: the XGBoost signal should
be down-weighted or replaced by the ElasticNet baseline when Banxico enters
a tightening cycle and top-5 stability falls below 0.30.

**Turnover is the primary cost of using XGBoost.**
XGBoost generates 4× the turnover of ElasticNetCV (0.27 vs 0.06 per rebalance).
SHAP decomposition identifies momentum_63 and short-horizon macro features as
the primary drivers of weight churn. Regime-conditional signal scaling
(0.7× in TIGHTENING) is the recommended first mitigation before considering
feature pruning or ensemble blending.

---

## Config Reference

All settings are defined in `src/settings.py` (`DEFAULT_SETTINGS`) and
overridable via `config.yaml`.

| Key | Type | Default | Description |
|---|---|---|---|
| `forecast_model` | `str` | `"elasticnet"` | Model selector. Options: `elasticnet`, `xgboost` |
| `compute_shap` | `bool` | `True` | Compute and persist SHAP values during walk-forward |
| `shap_output_path` | `str` | `"data/shap_values.parquet"` | SHAP parquet output path |
| `forecast_xgb_scoring` | `str` | `"neg_mean_squared_error"` | Inner-CV scorer. Options: `neg_mean_squared_error`, `ic` |
| `forecast_xgb_n_iter` | `int` | `20` | RandomizedSearchCV iterations (use 4 for fast runs) |
| `forecast_xgb_cv_splits` | `int` | `5` | TimeSeriesSplit folds in inner CV |
| `forecast_xgb_n_estimators_cap` | `int` | `2000` | Max trees before early stopping |
| `forecast_xgb_holdout_frac` | `float` | `0.2` | Fraction of training window held out for early stopping |

---

## Project Structure

```
FondoMexicoAlfa/
├── config.yaml                   # Runtime config (overrides DEFAULT_SETTINGS)
├── requirements.txt
├── PROGRESS.md                   # Step-by-step build log
│
├── src/
│   ├── settings.py               # DEFAULT_SETTINGS + config loader
│   ├── signals.py                # Walk-forward loop + forecast dispatcher
│   ├── features.py               # Feature engineering (equity + FIBRA)
│   ├── xgboost_model.py          # XGBoostModel class (Steps 1–3)
│   ├── shap_attribution.py       # SHAP collection + stability metrics (Step 2)
│   └── macro_regimes.py          # Rate + stress regime classifiers (Step 3)
│
├── scripts/
│   ├── run_all.py                # Main entry point (--source, --model flags)
│   ├── run_step1_comparison.py   # ElasticNet vs XGBoost side-by-side
│   ├── render_step2_report.py    # SHAP figures + markdown report
│   ├── render_step3_report.py    # Regime figures + markdown report
│   └── render_tearsheet.py       # 5-page PDF tearsheet
│
├── data/
│   ├── shap_values.parquet       # Per-rebalance SHAP (generated at runtime)
│   └── regime_table.csv          # Per-rebalance regime assignments
│
├── reports/
│   ├── FMIA_Tearsheet.pdf        # Main deliverable
│   ├── FMIA_Tearsheet.html       # Source HTML (images embedded as base64)
│   ├── step1_xgboost_vs_elasticnet.md
│   ├── step2_shap_analysis.md
│   ├── step3_regime_analysis.md
│   ├── regime_performance_table.csv
│   ├── shap_stability.csv
│   └── figures/
│       ├── step1_equity_mock.png
│       ├── step2_shap_beeswarm.png
│       ├── step2_shap_waterfall.png
│       ├── step2_feature_importance_over_time.png
│       ├── step3_regime_equity_curves.png
│       └── step3_ic_by_regime.png
│
└── tests/
    ├── test_xgboost_model.py
    ├── test_shap.py
    ├── test_macro_regimes.py
    └── test_tearsheet.py
```

---

## Environment Notes

### macOS: XGBoost libomp

XGBoost on macOS may fail with a `libomp.dylib` load error if the rpath
is not patched. Fix (no sudo required, assumes Homebrew at `~/homebrew`):

```bash
~/homebrew/bin/brew install libomp
install_name_tool -add_rpath \
  $(~/homebrew/bin/brew --prefix libomp)/lib \
  $(python -c "import xgboost; print(xgboost.__file__.replace('__init__.py',''))")lib/libxgboost.dylib
```

This is a one-time fix per virtual environment. Document the venv path in
PROGRESS.md so it can be reproduced if the env is rebuilt.

### PDF rendering: WeasyPrint vs fpdf2

`render_tearsheet.py` attempts WeasyPrint first (higher fidelity).
WeasyPrint requires `pango` and `gobject`, which are not available on macOS
without Xcode Command Line Tools or a Homebrew install of `pango`.

If WeasyPrint is unavailable, the script automatically falls back to fpdf2
(pure Python, no system dependencies). Both renderers produce a valid PDF;
the fpdf2 output is functionally identical at the cost of slightly reduced
typography quality.

To force the fpdf2 path explicitly:

```bash
# WeasyPrint is tried first; if it raises, fpdf2 runs automatically.
# No flag needed — the fallback is always active.
python scripts/render_tearsheet.py
```

---

## Running Tests

```bash
pytest -q          # 107 tests, ~8 seconds
pytest -v tests/test_xgboost_model.py   # model + holdout-cut tests
pytest -v tests/test_shap.py            # SHAP schema + flag tests
pytest -v tests/test_macro_regimes.py   # regime assignment + no-lookahead
pytest -v tests/test_tearsheet.py       # PDF smoke tests
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*FondoMéxicoAlfa is a research project. All results are based on
walk-forward out-of-sample simulation. Past simulation performance
does not guarantee future results. Mock data is used where live
Bloomberg data is unavailable.*
