# Fondo Mexico Strategy Research

A Mexico-focused long-biased industrial strategy blending macro, fundamental, and technical signals with institutional risk controls and FX management. Covers Mexican equities and FIBRAs with a Layer 2 analytical hedge overlay.

## Features

- **Multi-provider data** — Yahoo Finance (live), LSEG/Refinitiv (institutional), Bloomberg (institutional), with automatic fallbacks to FRED and Banxico SIE for macro and bond data.
- **Layer 1 portfolio** — Mean-variance, min-CVaR, and Michaud robust optimizers with bootstrap-based statistical significance testing, alpha measurement vs GBMALFA / GBMCRE / GBMMOD / GBMNEAR / IPC, and signal IC diagnostics.
- **Black–Litterman views** — Per-ticker views from the ElasticNetCV forecast plus sector-level macro views (industrial production, exports, banxico rate, USD/MXN momentum, US IP, inflation), confidence-weighted via the BL posterior. Macro confidence defaults to 0.20 so it nudges, not dominates.
- **TMEC stress scenario** — Deterministic and distributional shock simulating USMCA tariff / supply-chain disruption, sized by industrial + USD-export exposure of the live portfolio.
- **ETF → equity bridge** — Optional soft sector anchoring that propagates the ETF run's sector vector into the equity optimizer with a tunable band (default ±15 pp). Wide band keeps the optimizer free; narrow band replicates the ETF allocation. Hyperopt-tunable.
- **Layer 2 hedge overlay** — FX directional overlay (expanding z-score, GARCH vol adjustment), dynamic leverage, short borrow and leverage change cost model.
- **LFI reform scenarios** — Comparative backtest across 4 structures (regulated, 130/30, market-neutral, 130/30 sector-neutral) when `reform: true`.
- **ETF variant** — Parallel pipeline over an EWW / INDS / IGF / ILF / EMLC universe with price-only signals (momentum + volatility).
- **Hyperparameter optimization** — Bayesian search (Optuna TPE) over Black-Litterman, optimizer, EWMA, and forecast parameters via purged walk-forward cross-validation. Generates source-specific calibrated configs.
- **Overfitting diagnostics** — Deflated Sharpe Ratio (Bailey & López de Prado 2014) and Probability of Backtest Overfitting via combinatorially symmetric cross-validation, both rendered alongside the hyperopt results in the main report.
- **Point-in-time fundamentals** — Asof merge eliminates look-ahead bias in equity and FIBRA fundamental signals.
- **Unified HTML report** — Single interactive dashboard (Plotly) covering performance, benchmarks, risk, signal quality, optimizer comparison, stress tests, hedge breakdown, reform scenarios, hyperopt convergence/parallel-coords/importance, and overfitting diagnostics.

## Project structure

```
src/              core modules (data, features, signals, portfolio, backtest, risk, hedge, hyperopt, overfitting)
config/           ticker mapping and universe settings
reports/          report generator and chart builder
scripts/          pipeline runners (run_all, run_hyperopt, run_etf, run_etf_hyperopt)
tests/            unit and integration tests (74 tests)
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Uses sources defined in config.yaml (default: bloomberg + yahoo + refinitiv)
python scripts/run_all.py

# Override source from CLI
python scripts/run_all.py --source yahoo
python scripts/run_all.py --source yahoo,refinitiv,bloomberg
```

Output: `reports/output/strategy_report_{source}.html`

### 3. Optimize hyperparameters (optional)

```bash
# Runs for all sources defined in config.yaml
python scripts/run_hyperopt.py

# Or specify sources explicitly
python scripts/run_hyperopt.py --source yahoo,refinitiv --n-trials 50
```

Generates `config_optimized_{source}.yaml` for each source. The next `run_all.py` call picks them up automatically — no manual copy needed.

Output: `reports/hyperopt_data/hyperopt_results_{source}.json` (consumed by the main strategy report; no standalone HTML is produced).

### 4. ETF variant (optional)

```bash
# Backtest the ETF universe (EWW / INDS / IGF / ILF / EMLC) with price-only signals
python scripts/run_etf.py --source yahoo

# Hyperopt for the ETF variant
python scripts/run_etf_hyperopt.py --source yahoo --n-trials 30
```

Output: `reports/output/output_etf_{source}.html` and `reports/output/hyperopt_report_etf_{source}.html`.

### 5. Credentials

- **Yahoo Finance** — no credentials required.
- **LSEG/Refinitiv** — requires `lseg-data.config.json` in the project root (never committed).
- **Bloomberg** — requires a local Bloomberg Terminal session (BLPAPI). Data extraction lives in `scripts/extract_bloomberg_data.py`.

## Configuration

All pipeline settings live in `config.yaml`. Key parameters:

| Key | Description |
|-----|-------------|
| `source` | Data provider(s): `yahoo`, `refinitiv`, `bloomberg`, or a list |
| `optimizer` | `mv`, `cvar`, `robust`, or `both` |
| `hedge` | Enable Layer 2 FX hedge overlay |
| `reform` | Enable LFI reform scenario comparison (4 structures) |
| `hyperopt_n_trials` | Number of Optuna trials per source |
| `hyperopt_n_folds` | Walk-forward folds with purge gap |
| `bl_views.use_macro` | Toggle macro views in Black–Litterman |
| `bl_views.macro_view_confidence` | Confidence applied to macro views (default 0.20) |
| `etf_sector_anchor.enabled` | Activate the ETF→normal sector anchor |
| `etf_sector_anchor.band` | ±half-width of the soft sector band (default 0.15) |
| `etf_sector_anchor.source` | Which ETF run to read for the sector vector |

### ETF anchor mode

The ETF universe lives in a different basket than the equity universe; the anchor lets you propagate the **sector allocation** of the ETF run into the equity optimizer as soft constraints, without forcing replication.

```bash
# 1. Run the ETF pipeline first — it persists reports/output/etf_sector_weights_{source}.json
python scripts/run_etf.py --source bloomberg

# 2. Enable the anchor in config.yaml (etf_sector_anchor.enabled: true) and run normal mode
python scripts/run_all.py --source bloomberg
```

The new HTML section "Puente ETF → Normal" shows the source ETF weights, the realized normal-mode bucket weights, and a `band binding` table that flags any sector that hit a constraint edge (a sign the band is too tight). All `free` rows means the anchor did not degrade the unanchored optimum.

Regulatory parameters (`max_position`, `issuer_concentration_limit`, `fx_overlay_notional_cap`, `liquidity_sleeve_*`) are fixed at CNBV/prospectus-compliant values and excluded from the hyperopt search space.

## Notes on Yahoo vs Refinitiv performance

Hyperopt results show a consistent Sharpe gap between providers (Yahoo ~0.57, Refinitiv ~0.26 OOS). This is expected: Refinitiv's Mexican universe has more missing data and shorter effective history for several tickers, which dampens signal dispersion and reduces the optimizer's ability to differentiate parameter combinations. It is a data coverage difference, not a model defect.

## Running tests

```bash
python -m pytest tests/ -q
```
