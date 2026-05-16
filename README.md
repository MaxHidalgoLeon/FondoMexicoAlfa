# FondoMéxicoAlfa (FMIA)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-107%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Data](https://img.shields.io/badge/data-Bloomberg%20%7C%20Refinitiv%20%7C%20Yahoo-informational)

Systematic long-short equity and FIBRA strategy for the Mexican market (BMV, 2017–2026).
Multi-provider data pipeline, Black–Litterman portfolio construction, Bayesian hyperparameter
optimization, XGBoost ML signal with SHAP attribution, Banxico macro-regime conditioning,
Layer 2 FX hedge overlay, TMEC stress testing, LFI regulatory scenario analysis,
and deflated-Sharpe overfitting diagnostics. All results are walk-forward out-of-sample.

---

## Performance (Bloomberg, 2017–2026)

Primary source: Bloomberg with point-in-time fundamentals. CNBV-compliant constraints
(max 10% per position, issuer concentration limit). MV optimizer, 10 bp/side transaction costs.

| Metric | ElasticNetCV | XGBoost | Note |
|:---|:---:|:---:|:---|
| Annualized return | 8.34% | 8.29% | Regulated NAV |
| Annualized vol | 13.59% | 13.57% | |
| Sharpe ratio | 0.44 | 0.44 | 95% CI: [−0.25, 1.18] |
| Sortino ratio | 0.45 | 0.44 | |
| Max drawdown | −35.50% | −35.38% | |
| CVaR 95% (daily) | −1.95% | −1.95% | |
| Avg turnover | 0.57% | 6.33% | XGBoost 11× higher |

**XGBoost and ElasticNetCV produce statistically indistinguishable performance on real data.**
The value of the XGBoost component is attribution: SHAP values per rebalance identify which
features drive each position, and Banxico regime conditioning reveals when the model is
reliable and when it is not. With a cross-section of ~30 assets, the Mexican market does not
provide sufficient statistical power for a non-linear model to systematically outperform a
well-regularized linear baseline — a finding consistent with Gu, Kelly & Xiu (2020) in
small-panel settings.

**By data source** (ElasticNetCV, MV optimizer):

| Source | Sharpe | Return | Signal set | Note |
|:---|:---:|:---:|:---|:---|
| Bloomberg | 0.44 | 8.34% | PIT fundamentals + price | Production source |
| Yahoo Finance | 0.47 | 9.98% | Price only (momentum, liquidity) | No historical fundamentals |
| Refinitiv | 0.23 | 5.80% | PIT fundamentals + price | Shorter coverage for some tickers |

Yahoo Finance does not provide historical point-in-time fundamental data; its backtest
uses only momentum and liquidity signals and is not directly comparable to Bloomberg.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FMIA Pipeline                                │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐
  │     Data Layer      │
  │                     │
  │  Bloomberg (BLPAPI) │
  │  Refinitiv / LSEG   │──────────┐
  │  Yahoo Finance      │          │
  │  FRED + Banxico SIE │          ▼
  │  (auto-fallback)    │   ┌───────────────────────────────┐
  └─────────────────────┘   │      Feature Engineering      │
                            │                               │
                            │  Equity: pe_ratio, roe,       │
                            │  ebitda_growth, net_debt/     │
                            │  ebitda, capex_to_sales,      │
                            │  dividend_yield, momentum_63  │
                            │                               │
                            │  FIBRA: ltv, ffo_yield,       │
                            │  cap_rate, vacancy_rate       │
                            │                               │
                            │  Macro: Banxico rate, TIIE,   │
                            │  USDMXN, US IP, CPI, exports  │
                            └──────────────┬────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────┐
               ▼                           ▼                       ▼
  ┌────────────────────┐   ┌───────────────────────┐   ┌──────────────────┐
  │   ML Signal Layer  │   │   Black–Litterman     │   │   ETF Anchor     │
  │                    │   │                       │   │                  │
  │  ElasticNetCV      │   │  Per-ticker views     │   │  ETF universe    │
  │  (baseline)        │──▶│  from ElasticNet/XGB  │   │  sector weights  │
  │                    │   │  + macro sector views │   │  as soft         │
  │  XGBoost +         │   │  (confidence = 0.20)  │   │  constraints     │
  │  RandomizedSearchCV│   │  BL posterior → μ     │   │  (±15pp band)    │
  │  TimeSeriesSplit   │   └───────────┬───────────┘   └────────┬─────────┘
  │  SHAP attribution  │              └──────────────┬──────────┘
  │  Regime conditioning│                            ▼
  └────────────────────┘              ┌──────────────────────────┐
                                      │    Portfolio Optimizer   │
                                      │                          │
                                      │  Mean-Variance (SLSQP)   │
                                      │  Min-CVaR                │
                                      │  Michaud Robust          │
                                      │  Ledoit-Wolf shrinkage   │
                                      │  CNBV constraints        │
                                      └────────────┬─────────────┘
                                                   │
         ┌─────────────────────────────────────────┼──────────────────────────────┐
         ▼                                         ▼                              ▼
┌─────────────────────┐             ┌──────────────────────────┐   ┌─────────────────────┐
│   Layer 2 Hedge     │             │    Risk & Scenarios      │   │       Reports       │
│                     │             │                          │   │                     │
│  FX overlay         │             │  TMEC stress test        │   │  Interactive HTML   │
│  (GARCH vol adj)    │             │  LFI reform (4 structs)  │   │  (Plotly, 14 sects) │
│  Dynamic leverage   │             │  Deflated Sharpe (DSR)   │   │  PDF tearsheet      │
│  Analytical ref —   │             │  CSCV overfitting diag   │   │  SHAP report        │
│  not regulatory NAV │             │  Bootstrap CI (N=5000)   │   │  Regime report      │
└─────────────────────┘             └──────────────────────────┘   └─────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/MaxHidalgoLeon/FondoMexicoAlfa.git
cd FondoMexicoAlfa
pip install -r requirements.txt

# 2. Full pipeline — Bloomberg, ElasticNet baseline
python scripts/run_all.py

# 3. XGBoost signal + SHAP attribution
python scripts/run_all.py --model xgboost

# 4. No Bloomberg terminal — Yahoo Finance
python scripts/run_all.py --source yahoo

# 5. Hyperparameter optimization (Optuna TPE, purged walk-forward CV)
python scripts/run_hyperopt.py

# 6. ETF anchor — run before run_all if etf_sector_anchor.enabled: true
python scripts/run_etf.py --source bloomberg

# 7. PDF tearsheet
python scripts/render_tearsheet.py
```

Output: `reports/output/strategy_report_{source}.html`

**Credentials:**
- Bloomberg: local Bloomberg Terminal session (BLPAPI)
- Refinitiv: `lseg-data.config.json` in project root (never committed)
- Yahoo Finance: no credentials required

---

## Key Features

**Multi-provider data pipeline.** Bloomberg (primary, PIT fundamentals), Refinitiv/LSEG,
Yahoo Finance (price signals), FRED and Banxico SIE for macro and rate data. Automatic
fallback chain. `strict_data_mode: true` prevents silent mock injection in production runs.
A 90-day reporting lag is applied to all fundamentals to prevent look-ahead bias.

**Black–Litterman with ML views.** ElasticNetCV or XGBoost generates per-ticker return
views that feed the BL posterior. Macro sector views from industrial production, exports,
Banxico rate, USDMXN momentum, and inflation are blended at low confidence (0.20) to
nudge rather than override the quantitative signal.

**Bayesian hyperparameter optimization.** Optuna TPE search over BL risk aversion,
MV/CVaR optimizer parameters, EWMA covariance lambda, and ElasticNet mixing ratios.
Purged walk-forward CV with a 21-day gap between training and validation. Best OOS
Sharpe (walk-forward): Bloomberg 0.43, Yahoo 0.57, Refinitiv 0.26 (50 trials each).

**XGBoost ML signal with SHAP attribution.** XGBoost cross-sectional return forecaster
with internal RandomizedSearchCV over TimeSeriesSplit — no lookahead at any stage.
TreeExplainer SHAP values are computed per rebalance and accumulated into a
(date, ticker, feature, shap_value) panel. Feature-rank stability across consecutive
rebalances is 0.44 (Spearman, top-5) — below the 0.80 production threshold, consistent
with the small cross-section of the Mexican universe. EASING Banxico regimes produce
higher stability (0.57) than TIGHTENING (0.41).

**FIBRA-specific features.** LTV, FFO yield, capitalization rate, and vacancy rate are
integrated as first-class features alongside equity fundamentals. SHAP attribution shows
these four features dominate return prediction — larger in aggregate than any single
equity factor — reflecting the direct mechanistic link between FIBRA operating metrics
and NAV-based valuation.

**ETF → equity sector bridge.** `run_etf.py` backtests a price-signal ETF universe
(EWW, INDS, IGF, ILF, EMLC) and persists sector weights as soft constraints for the
equity optimizer (±15pp band). Wide band leaves the optimizer unconstrained; narrow
band replicates the ETF allocation. A "band binding" table flags sectors that hit a
constraint edge.

**Layer 2 FX hedge overlay.** FX directional signal (expanding z-score, GARCH vol
adjustment), dynamic leverage, and short borrow cost model. Reported as an analytical
reference — not included in CNBV-regulated NAV. Bloomberg result with hedge: return
45.70%, Sharpe 1.50 (analytical, not regulatory).

**LFI reform scenario analysis.** Comparative backtest across four regulatory structures:
current regulated, 130/30, market-neutral, and 130/30 sector-neutral. Bloomberg result
(hedge basis): regulated Sharpe 1.47 vs 130/30 Sharpe 1.82.

**Overfitting diagnostics.** Deflated Sharpe Ratio (Bailey & López de Prado 2014) and
Probability of Backtest Overfitting via CSCV, rendered alongside hyperopt results in
the main HTML report.

**CNBV compliance.** Max 10% per position, 10% issuer concentration limit, liquidity
sleeve (CETES28/91) sized by macro regime (3–15% of NAV), MBONO3Y optional buffer.
Regulatory parameters are fixed and excluded from hyperopt.

---

## Config Reference

All settings in `config.yaml`. Command-line arguments override the file.

| Key | Type | Default | Description |
|:---|:---|:---:|:---|
| `source` | str / list | `bloomberg` | Data provider(s) |
| `forecast_model` | str | `elasticnet` | `elasticnet` \| `xgboost` |
| `optimizer` | str | `both` | `mv` \| `cvar` \| `robust` \| `both` |
| `hedge` | bool | `true` | Layer 2 FX hedge overlay |
| `reform` | bool | `true` | LFI reform scenario comparison |
| `compute_shap` | bool | `true` | SHAP values (XGBoost only) |
| `forecast_xgb_scoring` | str | `neg_mean_squared_error` | `neg_mean_squared_error` \| `ic` |
| `forecast_xgb_n_iter` | int | `20` | RandomizedSearchCV iterations |
| `forecast_xgb_cv_splits` | int | `5` | TimeSeriesSplit inner CV folds |
| `hyperopt_n_trials` | int | `50` | Optuna trials per source |
| `hyperopt_objective` | str | `sharpe_adj` | `sharpe_adj` \| `sortino` \| `calmar` |
| `bl_views.use_macro` | bool | `true` | Macro sector views in BL |
| `bl_views.macro_view_confidence` | float | `0.20` | Macro view confidence weight |
| `etf_sector_anchor.enabled` | bool | `true` | ETF → equity sector bridge |
| `etf_sector_anchor.band` | float | `0.15` | ±half-width of sector band |
| `strict_data_mode` | bool | `true` | Abort on data failure |
| `fundamentals_lag_days` | int | `90` | Reporting lag (look-ahead prevention) |

---

## Project Structure

```
FondoMexicoAlfa/
├── config.yaml                        # Runtime configuration
├── requirements.txt
├── PROGRESS.md                        # Build log (Steps 1–9)
│
├── src/
│   ├── settings.py                    # DEFAULT_SETTINGS + config loader
│   ├── signals.py                     # Walk-forward loop + forecast dispatcher
│   ├── features.py                    # Feature engineering (equity + FIBRA + macro)
│   ├── xgboost_model.py               # XGBoostModel with internal CV (Step 1)
│   ├── shap_attribution.py            # SHAP collection + stability metrics (Step 2)
│   └── macro_regimes.py               # Banxico rate + IPC stress classifiers (Step 3)
│
├── scripts/
│   ├── run_all.py                     # Main entry point (--source, --model flags)
│   ├── run_hyperopt.py                # Optuna hyperparameter search
│   ├── run_etf.py                     # ETF universe pipeline
│   ├── run_etf_hyperopt.py            # ETF hyperopt
│   ├── render_tearsheet.py            # PDF tearsheet (WeasyPrint / fpdf2 fallback)
│   ├── render_step2_report.py         # SHAP report
│   └── render_step3_report.py         # Regime analysis report
│
├── reports/
│   ├── output/                        # HTML strategy reports
│   ├── FMIA_Tearsheet.pdf
│   ├── step2_shap_analysis.md
│   ├── step3_regime_analysis.md
│   └── figures/
│
├── config/                            # Ticker mappings, universe settings
├── data/                              # Parquet cache (gitignored)
└── tests/                             # 107 unit and integration tests
```

---

## Running Tests

```bash
pytest -q                                    # 107 tests
pytest -v tests/test_xgboost_model.py        # XGBoost + holdout-cut
pytest -v tests/test_shap.py                 # SHAP schema + compute_shap flag
pytest -v tests/test_macro_regimes.py        # Regime assignment + no-lookahead
pytest -v tests/test_tearsheet.py            # PDF smoke tests
```

---

## Environment Notes

**macOS — XGBoost libomp.** XGBoost on macOS may fail with a `libomp.dylib` load
error. One-time fix per virtual environment (no sudo required):

```bash
~/homebrew/bin/brew install libomp
install_name_tool -add_rpath \
  $(~/homebrew/bin/brew --prefix libomp)/lib \
  $(python -c "import xgboost; print(xgboost.__file__.replace('__init__.py',''))")lib/libxgboost.dylib
```

**PDF rendering.** `render_tearsheet.py` attempts WeasyPrint first (requires `pango`
and `gobject`). Falls back automatically to fpdf2 if system libraries are unavailable.

---

## License

MIT — see [LICENSE](LICENSE).

---

*FondoMéxicoAlfa is a research prototype. All results are walk-forward out-of-sample
simulations with CNBV-compliant constraints applied. Past simulation performance does
not guarantee future results. Bloomberg is the primary data source; Yahoo Finance
results use price signals only and are not directly comparable to Bloomberg or Refinitiv.*
