# FondoMéxicoAlfa (FMIA)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-172%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Data](https://img.shields.io/badge/data-Bloomberg%20%7C%20Refinitiv%20%7C%20Yahoo-informational)

Systematic long-only equity and FIBRA strategy for the Mexican market (BMV, 2017–2026).
Multi-provider data pipeline (Bloomberg, Refinitiv/LSEG, Yahoo Finance), Black–Litterman
portfolio construction with machine-learning cross-sectional views, Bayesian hyperparameter
optimization (Optuna TPE), LightGBM ML signal with TreeExplainer SHAP attribution, Banxico
macro-regime conditioning, Layer 2 FX hedge overlay (analytical reference only), LFI
regulatory scenario analysis, and deflated-Sharpe overfitting diagnostics. All results are
walk-forward out-of-sample under CNBV regulatory constraints. The primary finding is that
ElasticNetCV and LightGBM are statistically indistinguishable on real data; the small
effective cross-section of the Mexican universe (~30 assets) prevents non-linear models from
extracting advantage over a well-regularized linear baseline.

---

## Performance (Bloomberg, 2017–2026)

Primary source: Bloomberg with point-in-time fundamentals. CNBV-compliant constraints
(10% per position, 10% issuer concentration limit). Mean-variance optimizer, 10 bp/side
transaction costs. Period: January 2017 – March 2026 (108 monthly rebalances).

| Metric | ElasticNetCV | LightGBM |
|:---|:---:|:---:|
| Annualized return | 8.46% | 5.28% |
| Annualized vol | 13.70% | 14.56% |
| Sharpe ratio | 0.03 | −0.17 |
| Sortino ratio | 0.03 | −0.18 |
| Max drawdown | −37.19% | −39.46% |
| CVaR 95% (daily) | −1.98% | −2.11% |
| Avg monthly turnover | 8.87% | 3.49% |

Bootstrap 95% CI on Bloomberg ElasticNetCV Sharpe (stationary block bootstrap,
N = 5,000): [−0.634, 0.740]. The point estimates differ numerically between models
but are statistically indistinguishable at conventional significance levels.
The value of the LightGBM component is attribution (SHAP per rebalance, feature-rank
stability, regime conditioning), not raw return lift.

**By data source** (ElasticNetCV, MV optimizer):

| Source | Return | Vol | Sharpe | Sortino | Max DD | Turnover | Signal set |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Bloomberg | 8.46% | 13.70% | 0.03 | 0.03 | −37.19% | 8.87% | PIT fundamentals + price |
| Yahoo Finance | 3.32% | 15.49% | −0.31 | −0.31 | −43.37% | 1.01% | Price only |
| Refinitiv | 2.63% | 16.59% | −0.33 | −0.33 | −48.64% | 0.98% | PIT fundamentals + price |

Yahoo Finance does not provide historical point-in-time fundamental data; its backtest
uses only momentum and liquidity signals and is not directly comparable to Bloomberg or
Refinitiv. Performance differences across providers primarily reflect fundamental-data
coverage and look-ahead discipline, not signal quality.

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
  │  (baseline)        │──▶│  from ElasticNet/LGBM │   │  sector weights  │
  │                    │   │  + macro sector views │   │  as soft         │
  │  LightGBM +        │   │  (confidence = 0.20)  │   │  constraints     │
  │  RandomizedSearchCV│   │  BL posterior → μ     │   │  (±15pp band)    │
  │  TimeSeriesSplit   │   └───────────┬───────────┘   └────────┬─────────┘
  │  SHAP attribution  │              └──────────────┬──────────┘
  │  Regime condition. │                             ▼
  └────────────────────┘              ┌──────────────────────────┐
                                      │    Portfolio Optimizer   │
                                      │                          │
                                      │  Mean-Variance (SLSQP)   │
                                      │  Min-CVaR (95%)          │
                                      │  Michaud Robust (N=100)  │
                                      │  EWMA Ledoit-Wolf cov.   │
                                      │  CNBV constraints        │
                                      └────────────┬─────────────┘
                                                   │
         ┌─────────────────────────────────────────┼──────────────────────────────┐
         ▼                                         ▼                              ▼
┌─────────────────────┐             ┌──────────────────────────┐   ┌─────────────────────┐
│   Layer 2 Hedge     │             │    Risk & Scenarios      │   │       Reports       │
│                     │             │                          │   │                     │
│  FX overlay         │             │  LFI reform (4 structs)  │   │  Interactive HTML   │
│  (GARCH vol adj.)   │             │  Deflated Sharpe (DSR)   │   │  (Plotly)           │
│  Dynamic leverage   │             │  CSCV overfitting diag.  │   │  PDF tearsheet      │
│  Analytical ref. —  │             │  Bootstrap CI (N=5,000)  │   │  SHAP report        │
│  not regulatory NAV │             │  Fan-chart (N=1,000)     │   │  Regime report      │
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

# 3. LightGBM signal + SHAP attribution
python scripts/run_all.py --model lightgbm

# 4. No Bloomberg terminal — Yahoo Finance (price signals only)
python scripts/run_all.py --source yahoo

# 5. Hyperparameter optimization (Optuna TPE, purged walk-forward CV)
python scripts/run_hyperopt.py

# 6. ETF anchor — run before run_all if etf_sector_anchor.enabled: true
python scripts/run_etf.py --source bloomberg

# 7. PDF tearsheet (reads reports/output/metrics_<source>_<model>.json)
python scripts/render_tearsheet.py --source bloomberg --model elasticnet

# 8. Research report (regenerates Abstract KPIs + Table 1 from same JSON)
python scripts/render_research_report.py --source bloomberg --model elasticnet
```

Output files:
- `reports/output/strategy_report_{source}_{model}.html` — full interactive report
- `reports/output/metrics_{source}_{model}.json` — single source of truth for
  all downstream renderers; emitted by `run_all.py` at the end of each backtest

**Credentials:**
- Bloomberg: active Bloomberg Terminal session (BLPAPI)
- Refinitiv: `lseg-data.config.json` in project root (never committed; obtain App Key at developers.lseg.com)
- Yahoo Finance: no credentials required

---

## Key Features

**Multi-provider data pipeline.** Bloomberg (primary, point-in-time fundamentals via BLPAPI),
Refinitiv/LSEG (institutional alternative), Yahoo Finance (price signals; no historical
fundamentals), FRED and Banxico SIE for macro and rate data. Automatic fallback chain.
`strict_data_mode: true` aborts on any data failure rather than substituting synthetic
values. A 90-day reporting lag is applied to all fundamentals to prevent look-ahead bias.

**FIBRA-specific features.** Loan-to-value (LTV), FFO yield, capitalization rate, and vacancy
rate are integrated as first-class features alongside standard equity fundamentals. SHAP
attribution (Bloomberg walk-forward sample) shows these four features occupy the top three
positions by mean |SHAP| value — LTV alone is 1.8× larger than the highest equity feature
(P/E ratio). This reflects the mechanistic link between FIBRA operating metrics and
NAV-based valuation; equity-style factors systematically misprice FIBRAs without them.

**Black–Litterman with ML views.** ElasticNetCV or LightGBM generates per-ticker return
views that feed the BL posterior. Macro sector views from industrial production, exports,
Banxico rate, USDMXN momentum, and inflation are blended at low confidence (0.20) to
nudge rather than override the quantitative signal. The BL step is essential in a small
cross-section: it stabilizes the mean-variance optimizer by pulling the expected-return
vector toward the equilibrium prior weighted by view confidence.

**Macro-regime conditioning.** Banxico rate regime (TIGHTENING / EASING / NEUTRAL) and IPC
stress regime (CALM / STRESS) are classified at each rebalance using only lagged data.
Current regime (as of March 2026): EXPANSION with confidence 0.924.
Regime-conditioned metrics: TIGHTENING shows higher IC mean (0.14) and ICIR (0.59) than
EASING (IC 0.04, ICIR 0.18); EASING shows higher SHAP feature-rank stability (0.57 vs 0.41).

**LightGBM ML signal with SHAP attribution.** LightGBM cross-sectional return forecaster
with internal RandomizedSearchCV over TimeSeriesSplit — no lookahead at any stage.
TreeExplainer SHAP values are computed per rebalance and accumulated into a
(date, ticker, feature, shap_value) panel. Feature-rank stability across consecutive
rebalances is 0.44 (Spearman, top-5) — below the 0.80 institutional deployment threshold,
consistent with the small cross-section of the Mexican universe. The gradient-boosted
forecaster does not materially outperform ElasticNetCV on raw return; its value is the
attribution framework (SHAP + regime conditioning).

**Bayesian hyperparameter optimization.** Optuna TPE search (25 trials per source, purged
walk-forward CV with 21-day gap) over BL risk aversion, MV/CVaR optimizer parameters,
EWMA covariance lambda, and ElasticNet mixing ratios. Best walk-forward validated Sharpe:
Bloomberg 0.15, Yahoo 0.22, Refinitiv 0.04 (current run). Deflated Sharpe Ratio and
Probability of Backtest Overfitting (CSCV) are reported alongside each hyperopt run.

**ETF → equity sector bridge.** `run_etf.py` backtests a price-signal ETF universe
(EWW, INDS, IGF, ILF, EMLC) and persists sector weights as soft constraints for the
equity optimizer (±15pp band). Wide band leaves the optimizer unconstrained; narrow
band replicates the ETF allocation.

**Layer 2 FX hedge overlay.** FX directional signal (expanding z-score, GARCH vol
adjustment), dynamic leverage, and short borrow cost model. Reported on an analytical
basis — excluded from CNBV-regulated NAV. Bloomberg ElasticNetCV with hedge overlay
(analytical): Sharpe −0.16, Return 3.88%, Vol 24.1%. Bloomberg LightGBM with hedge
(analytical): Sharpe 0.22, Return 13.41%, Vol 22.8%.

**LFI reform scenario analysis.** Comparative backtest across four regulatory structures:
current regulated (long-only CNBV NAV), 130/30, market-neutral, and 130/30
sector-neutral. See `reports/output/strategy_report_{source}_{model}.html` for
scenario-level results.

**Overfitting diagnostics.** Deflated Sharpe Ratio (Bailey & López de Prado 2014) and
Probability of Backtest Overfitting via CSCV. Bootstrap CI (N = 5,000 block replications)
on Sharpe, Sortino, Max DD, and CAGR. Fan-chart (N = 1,000 paths) on NAV projections.

**CNBV compliance.** Max 10% per position, 10% issuer concentration limit, liquidity
sleeve (CETES28/CETES91) sized by macro regime (3–15% of NAV), optional MBONO3Y buffer
(disabled in production). Regulatory parameters are fixed and excluded from hyperopt.

---

## Config Reference

All settings in `config.yaml`. Command-line arguments override the file.

| Key | Type | Default | Description |
|:---|:---|:---:|:---|
| `source` | str / list | `bloomberg` | Data provider(s): `bloomberg` \| `refinitiv` \| `yahoo` |
| `forecast_model` | str | `elasticnet` | `elasticnet` \| `lightgbm` |
| `optimizer` | str | `both` | `mv` \| `cvar` \| `robust` \| `both` |
| `hedge` | bool | `true` | Layer 2 FX hedge overlay (analytical mode) |
| `reform` | bool | `true` | LFI reform scenario comparison |
| `compute_shap` | bool | `true` | SHAP values per rebalance (LightGBM only) |
| `forecast_lgbm_scoring` | str | `neg_mean_squared_error` | Inner-CV metric; config overrides to `ic` (Spearman IC) |
| `forecast_lgbm_n_iter` | int | `20` | RandomizedSearchCV draws; config overrides to `5` |
| `forecast_lgbm_cv_splits` | int | `5` | TimeSeriesSplit inner folds; config overrides to `3` |
| `forecast_lgbm_n_jobs` | int | `1` | Thread count (1 is faster than >1 on small cross-sections) |
| `forecast_lgbm_n_estimators_cap` | int | `2000` | Early-stopping ceiling for LightGBM |
| `hyperopt_n_trials` | int | `25` | Optuna trials per source |
| `hyperopt_n_folds` | int | `2` | Purged walk-forward folds in hyperopt |
| `hyperopt_objective` | str | `sharpe_adj` | `sharpe_adj` \| `sortino` \| `calmar` |
| `hyperopt_search_keys` | list\|null | `null` | Restrict hyperopt to these keys; null = full search space |
| `bl_views.use_macro` | bool | `true` | Macro sector views in BL posterior |
| `bl_views.macro_view_confidence` | float | `0.20` | Macro view confidence weight |
| `bl_views.macro_view_max_magnitude` | float | `0.015` | Macro view magnitude cap |
| `etf_sector_anchor.enabled` | bool | `true` | ETF → equity sector soft constraint |
| `etf_sector_anchor.band` | float | `0.15` | ±half-width of sector constraint band |
| `strict_data_mode` | bool | `true` | Abort on data failure (no silent mock injection) |
| `fundamentals_lag_days` | int | `90` | Reporting lag applied to all fundamentals |
| `covariance_method` | str | `ewma_ledoit_wolf` | `rolling_ledoit_wolf` \| `ewma_ledoit_wolf` |
| `ewma_lambda_cov` | float | `0.94` | EWMA decay for covariance estimation |
| `bootstrap_n_reps` | int | `5000` | Block bootstrap replications for CI |
| `cvar_scenario_window` | int | `504` | Days of scenarios for CVaR optimizer (~2 years) |
| `max_position_mv` | float | `0.10` | CNBV per-position cap (MV optimizer) |
| `issuer_concentration_limit` | float | `0.10` | CNBV issuer concentration cap |
| `parallel_providers` | bool | `false` | Run multiple providers simultaneously |

---

## Project Structure

```
FondoMexicoAlfa/
├── config.yaml                        # Runtime configuration (all pipeline settings)
├── config_optimized_{source}.yaml     # Optuna best params (auto-generated per source)
├── requirements.txt
│
├── src/
│   ├── settings.py                    # DEFAULT_SETTINGS + config loader
│   ├── pipeline.py                    # Top-level pipeline orchestrator
│   ├── signals.py                     # Walk-forward loop + forecast dispatcher
│   ├── features.py                    # Feature engineering (equity + FIBRA + macro)
│   ├── data_loader.py                 # Data loading and caching
│   ├── data_providers.py              # Bloomberg / Refinitiv / Yahoo backends
│   ├── bl_views.py                    # Black–Litterman posterior construction
│   ├── portfolio.py                   # MV / CVaR / Robust optimizers + CNBV constraints
│   ├── backtest.py                    # Walk-forward backtest engine
│   ├── lightgbm_model.py              # LightGBM with internal CV + SHAP
│   ├── shap_attribution.py            # SHAP collection + stability metrics
│   ├── macro_regimes.py               # Regime classification (rate + stress axes)
│   ├── hedge_overlay.py               # Layer 2 FX hedge overlay
│   ├── hyperopt.py                    # Optuna TPE search
│   ├── overfitting.py                 # DSR + CSCV diagnostics
│   ├── bootstrap.py                   # Block bootstrap CI + fan-chart
│   ├── risk.py                        # GARCH vol, VaR, CVaR, GEV tail
│   ├── alpha_significance.py          # IC significance tests
│   └── signal_diagnostics.py         # Walk-forward signal diagnostics
│
├── scripts/
│   ├── run_all.py                     # Main entry point (--source, --model flags)
│   ├── run_hyperopt.py                # Optuna hyperparameter search
│   ├── run_etf.py                     # ETF universe pipeline
│   ├── render_tearsheet.py            # PDF tearsheet (WeasyPrint / fpdf2 fallback)
│   ├── render_research_report.py      # Research report renderer (metrics_*.json)
│   ├── render_step1_report.py         # ElasticNet vs LightGBM comparison
│   ├── render_step2_report.py         # SHAP attribution report
│   └── render_step3_report.py         # Regime analysis report
│
├── reports/
│   ├── output/                        # HTML strategy reports + metrics JSON
│   ├── hyperopt_data/                 # Per-source hyperopt results JSON
│   ├── figures/                       # Exported chart images
│   ├── FMIA_Research_Report.md        # Academic research report
│   ├── FMIA_Tearsheet.pdf
│   └── regime_performance_table.csv   # Regime-conditioned metrics (pipeline output)
│
├── config/
│   └── ticker_map.yaml                # Ticker → provider symbol mapping
│
├── index/                             # Sector index CSVs (IPC constituents)
├── data/                              # Parquet cache (gitignored)
└── tests/                             # 172 unit and integration tests
```

---

## Running Tests

```bash
pytest -q                                    # 172 tests
pytest -v tests/test_signals_leakage.py      # Walk-forward PIT guarantee
pytest -v tests/test_walkforward_integrity.py # Backtest weight invariance + constraint compliance
pytest -v tests/test_backtest.py             # Turnover math + return shape
pytest -v tests/test_hyperopt.py             # Purged walk-forward folds
pytest -v tests/test_portfolio.py            # Optimizer constraints
pytest -v tests/test_lightgbm_model.py       # LightGBM + holdout-cut
pytest -v tests/test_shap.py                 # SHAP schema + compute_shap flag
pytest -v tests/test_macro_regimes.py        # Regime assignment + no-lookahead
pytest -v tests/test_hedge_overlay.py        # Layer 2 hedge mechanics
pytest -v tests/test_overfitting.py          # DSR + CSCV diagnostics
pytest -v tests/test_bootstrap.py            # Block bootstrap CI
```

---

## Environment Notes

**macOS — LightGBM libomp.** LightGBM on macOS requires a runtime copy of `libomp.dylib`
which is not bundled with the PyPI wheel. If you do not have Homebrew installed,
the repo includes a workaround that points LightGBM at any arm64 `libomp.dylib`
already on your system (a copy from R, scikit-learn, or conda works):

```bash
# Option A: Homebrew (cleanest, requires sudo to install brew)
brew install libomp

# Option B: reuse an existing arm64 libomp.dylib (no sudo)
mkdir -p .venv/lib
cp /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib .venv/lib/
install_name_tool -change @rpath/libomp.dylib \
  "$(pwd)/.venv/lib/libomp.dylib" \
  .venv/lib/python3.13/site-packages/lightgbm/lib/lib_lightgbm.dylib
```

**LightGBM threading.** The config sets `forecast_lgbm_n_jobs: 1`. Despite being counter-
intuitive, a single-threaded LightGBM fit is 5–11× faster than multi-threaded on a small
cross-section (~30 assets) because OpenMP thread-synchronization overhead (psynch_cvwait)
dominates the actual compute. Do not increase this without profiling.

**PDF rendering.** `render_tearsheet.py` attempts WeasyPrint first (requires `pango` and
`gobject`). Falls back automatically to fpdf2 if system libraries are unavailable.

---

## License

MIT — see [LICENSE](LICENSE).

---

*FondoMéxicoAlfa is a research prototype. All results are walk-forward out-of-sample
simulations with CNBV-compliant constraints applied. Performance numbers are derived
from `reports/output/metrics_{source}_{model}.json` emitted by the pipeline; see that
directory for the authoritative values. Past simulation performance does not guarantee
future results. Bloomberg is the primary data source; Yahoo Finance results use price
signals only and are not directly comparable to Bloomberg or Refinitiv.*
