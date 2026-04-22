# Fondo Mexico Strategy Research

A Mexico-focused long-biased industrial strategy blending macro, fundamental, and technical signals with institutional risk controls and FX management. Covers Mexican equities and FIBRAs with a Layer 2 analytical hedge overlay.

## Features

- **Multi-provider data** — Yahoo Finance (live), LSEG/Refinitiv (institutional), with automatic fallbacks to FRED and Banxico SIE for macro and bond data.
- **Layer 1 portfolio** — Mean-variance, min-CVaR, and Michaud robust optimizers with bootstrap-based statistical significance testing, alpha measurement vs GBMALFA / GBMCRE / GBMMOD / GBMNEAR / IPC, and signal IC diagnostics.
- **Layer 2 hedge overlay** — FX directional overlay (expanding z-score, GARCH vol adjustment), dynamic leverage, short borrow and leverage change cost model.
- **Hyperparameter optimization** — Bayesian search (Optuna TPE) over Black-Litterman, optimizer, EWMA, and forecast parameters via purged walk-forward cross-validation. Generates source-specific calibrated configs.
- **Point-in-time fundamentals** — Asof merge eliminates look-ahead bias in equity and FIBRA fundamental signals.
- **HTML report** — Interactive dashboard (Plotly) covering performance, benchmarks, risk, signal quality, optimizer comparison, stress tests, hedge breakdown, and hyperopt results.

## Project structure

```
src/              core modules (data, features, signals, portfolio, backtest, risk, hedge, hyperopt)
config/           ticker mapping and universe settings
reports/          report generator and chart builder
scripts/          pipeline runner (run_all.py) and hyperopt runner (run_hyperopt.py)
tests/            unit and integration tests (74 tests)
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Uses sources defined in config.yaml (default: yahoo + refinitiv)
python scripts/run_all.py

# Override source from CLI
python scripts/run_all.py --source yahoo
python scripts/run_all.py --source yahoo,refinitiv
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

Output: `reports/output/hyperopt_report_{source}.html`

### 4. Credentials

- **Yahoo Finance** — no credentials required.
- **LSEG/Refinitiv** — requires `lseg-data.config.json` in the project root (never committed).

## Configuration

All pipeline settings live in `config.yaml`. Key parameters:

| Key | Description |
|-----|-------------|
| `source` | Data provider(s): `yahoo`, `refinitiv`, `bloomberg`, or a list |
| `optimizer` | `mv`, `cvar`, `robust`, or `both` |
| `hedge` | Enable Layer 2 FX hedge overlay |
| `hyperopt_n_trials` | Number of Optuna trials per source |
| `hyperopt_n_folds` | Walk-forward folds with purge gap |

Regulatory parameters (`max_position`, `issuer_concentration_limit`, `fx_overlay_notional_cap`, `liquidity_sleeve_*`) are fixed at CNBV/prospectus-compliant values and excluded from the hyperopt search space.

## Notes on Yahoo vs Refinitiv performance

Hyperopt results show a consistent Sharpe gap between providers (Yahoo ~0.57, Refinitiv ~0.26 OOS). This is expected: Refinitiv's Mexican universe has more missing data and shorter effective history for several tickers, which dampens signal dispersion and reduces the optimizer's ability to differentiate parameter combinations. It is a data coverage difference, not a model defect.

## Running tests

```bash
python -m pytest tests/ -q
```
