# Fondo Mexico Strategy Research

A Mexico-focused long-biased industrial strategy blending macro, fundamental, and technical signals with institutional risk controls and FX management. Covers Mexican equities and FIBRAs with a Layer 2 analytical hedge overlay.

## Features

- **Multi-provider data** — Yahoo Finance (live), LSEG/Refinitiv (institutional), with automatic fallbacks to FRED and Banxico SIE for macro and bond data.
- **Layer 1 portfolio** — Mean-variance and min-CVaR optimizers with bootstrap-based statistical significance testing, alpha measurement vs GBMALFA / GBMCRE / GBMMOD / GBMNEAR / IPC, and signal IC diagnostics.
- **Layer 2 hedge overlay** — FX directional overlay (expanding z-score, GARCH vol adjustment), dynamic leverage, short borrow and leverage change cost model.
- **Point-in-time fundamentals** — Asof merge eliminates look-ahead bias in equity and FIBRA fundamental signals.
- **HTML report** — 12-section interactive dashboard (Plotly) covering performance, risk, signal quality, optimizer comparison, stress tests, and hedge breakdown.

## Project structure

```
src/              core modules (data, features, signals, portfolio, backtest, risk, hedge)
config/           ticker mapping and universe settings
reports/          report generator and chart builder
scripts/          data exploration and pipeline runner utilities
tests/            unit and integration tests (60 tests)
```

## Generating a report

```bash
# Yahoo Finance (no credentials required)
python reports/generate_report.py --source yahoo --hedge

# LSEG/Refinitiv (requires lseg-data.config.json)
python reports/generate_report.py --source refinitiv --hedge

# Both sources in one run
python reports/generate_report.py --source yahoo,refinitiv --hedge
```

Output is written to `reports/output/strategy_report_<source>.html`.

## Running tests

```bash
python -m pytest tests/ -q
```
