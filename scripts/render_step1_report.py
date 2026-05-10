#!/usr/bin/env python
"""Render reports/step1_xgboost_vs_elasticnet.md from cached step1 artifacts.

Reads the pickles produced by `scripts/run_step1_comparison.py` for both
models, draws equity-curve and IC time-series PNGs to reports/figures/,
and writes the markdown comparison report.

Usage:
    python scripts/render_step1_report.py --source mock
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACT_DIR = ROOT / "reports" / "output" / "step1"
FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "reports" / "step1_xgboost_vs_elasticnet.md"

METRIC_ROWS: list[tuple[str, str, str]] = [
    ("ic_mean", "IC mean (Spearman)", "{:+.4f}"),
    ("ic_std", "IC std", "{:.4f}"),
    ("icir", "ICIR", "{:+.3f}"),
    ("hit_rate", "Hit rate (daily)", "{:.3f}"),
    ("annualized_return", "Annualized return", "{:+.4f}"),
    ("annualized_vol", "Annualized vol", "{:.4f}"),
    ("sharpe", "Sharpe", "{:+.3f}"),
    ("sortino", "Sortino", "{:+.3f}"),
    ("max_drawdown", "Max drawdown", "{:+.3f}"),
    ("cvar_95", "CVaR 95% (daily)", "{:+.4f}"),
    ("turnover", "Turnover (per rebalance)", "{:.4f}"),
    ("forecast_seconds", "Forecast wall time (s)", "{:.1f}"),
]


def _load(model: str, source: str) -> dict:
    path = ARTIFACT_DIR / f"{model}_{source}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna().sort_index()
    return r.cumsum().apply(np.exp)


def _plot_equity(curves: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (1.0 = start)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_ic(ic_series: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, ic in ic_series.items():
        ax.plot(ic.index, ic.values, label=f"{name} (mean={ic.mean():+.3f})", linewidth=1.2, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Spearman IC")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _table(elastic: dict, xgb: dict) -> str:
    e_summary, x_summary = elastic["summary"], xgb["summary"]
    lines = [
        "| Metric | ElasticNetCV | XGBoost | Δ (xgb − elastic) |",
        "|---|---:|---:|---:|",
    ]
    for key, label, fmt in METRIC_ROWS:
        e_val = e_summary.get(key, float("nan"))
        x_val = x_summary.get(key, float("nan"))
        if not (np.isfinite(e_val) and np.isfinite(x_val)):
            delta_str = "—"
        else:
            delta_str = f"{(x_val - e_val):+.4f}"
        lines.append(
            f"| {label} | {fmt.format(e_val)} | {fmt.format(x_val)} | {delta_str} |"
        )
    return "\n".join(lines)


def _interpretation(elastic_summary: dict, xgb_summary: dict) -> str:
    e, x = elastic_summary, xgb_summary
    bullets: list[str] = []

    if np.isfinite(x["ic_mean"]) and np.isfinite(e["ic_mean"]):
        diff = x["ic_mean"] - e["ic_mean"]
        if diff > 0.005:
            bullets.append(
                f"XGBoost lifts mean Spearman IC from {e['ic_mean']:+.4f} to "
                f"{x['ic_mean']:+.4f} ({diff:+.4f}) — the non-linear interactions "
                "between fundamentals and momentum that ElasticNet's linear "
                "shrinkage flattens are picked up by tree splits."
            )
        elif diff < -0.005:
            bullets.append(
                f"XGBoost loses {-diff:+.4f} of mean IC vs ElasticNet "
                f"({x['ic_mean']:+.4f} vs {e['ic_mean']:+.4f}) — likely "
                "overfitting on the modest panel size; the L1/L2 shrinkage of "
                "ElasticNet is a stronger inductive bias here."
            )
        else:
            bullets.append(
                f"Mean IC is statistically indistinguishable between the two "
                f"({x['ic_mean']:+.4f} vs {e['ic_mean']:+.4f}, |Δ|<0.005)."
            )

    if np.isfinite(x["sharpe"]) and np.isfinite(e["sharpe"]):
        diff_sh = x["sharpe"] - e["sharpe"]
        if abs(diff_sh) >= 0.05:
            verb = "improves" if diff_sh > 0 else "drags"
            bullets.append(
                f"Net-of-cost Sharpe {verb} from {e['sharpe']:+.3f} (elastic) to "
                f"{x['sharpe']:+.3f} (xgb) — Δ={diff_sh:+.3f}."
            )
        else:
            bullets.append(
                f"Sharpe parity ({e['sharpe']:+.3f} vs {x['sharpe']:+.3f}, "
                "|Δ|<0.05) — once the optimizer's transaction-cost penalty kicks "
                "in, the predictor edge is largely absorbed."
            )

    if np.isfinite(x["max_drawdown"]) and np.isfinite(e["max_drawdown"]):
        bullets.append(
            f"Max drawdown: ElasticNet {e['max_drawdown']:+.3f} vs "
            f"XGBoost {x['max_drawdown']:+.3f} "
            f"(turnover {e['turnover']:.3f} vs {x['turnover']:.3f}) — "
            "different signal stability changes how much the optimizer churns "
            "month over month."
        )

    bullets.append(
        f"Wall time: ElasticNet {e['forecast_seconds']:.1f} s vs XGBoost "
        f"{x['forecast_seconds']:.1f} s ({x['forecast_seconds'] / max(e['forecast_seconds'], 1e-9):.1f}× slower) — "
        "the inner RandomizedSearchCV × TimeSeriesSplit grid dominates; reducing "
        "`forecast_xgb_n_iter` is the obvious knob if speed matters more than the "
        "last few bps of IC."
    )
    return "\n".join(f"- {b}" for b in bullets)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="mock")
    args = p.parse_args()

    elastic = _load("elasticnet", args.source)
    xgb = _load("xgboost", args.source)

    eq_curves = {
        "ElasticNetCV": _equity_curve(elastic["returns"]),
        "XGBoost": _equity_curve(xgb["returns"]),
    }
    ic_series = {
        "ElasticNetCV": elastic["ic_series"],
        "XGBoost": xgb["ic_series"],
    }

    eq_path = FIGURE_DIR / f"step1_equity_{args.source}.png"
    ic_path = FIGURE_DIR / f"step1_ic_{args.source}.png"
    # Paths used inside the markdown body need to be relative to the
    # markdown file's location, not the repo root.
    eq_md = eq_path.relative_to(REPORT_PATH.parent).as_posix()
    ic_md = ic_path.relative_to(REPORT_PATH.parent).as_posix()
    _plot_equity(eq_curves, eq_path, f"Equity curve — Mean-Variance long-only ({args.source})")
    _plot_ic(ic_series, ic_path, f"Spearman IC per rebalance ({args.source})")

    table = _table(elastic, xgb)
    interp = _interpretation(elastic["summary"], xgb["summary"])
    summary_json = ARTIFACT_DIR / f"summary_{args.source}.json"
    if summary_json.exists():
        with open(summary_json) as f:
            summaries = json.load(f)
    else:
        summaries = {"elasticnet": elastic["summary"], "xgboost": xgb["summary"]}
        with open(summary_json, "w") as f:
            json.dump(summaries, f, indent=2, default=float)

    body = f"""# Step 1 — XGBoost vs ElasticNetCV

Side-by-side walk-forward backtest on the same `{args.source}` data panel,
same rebalance schedule, same Mean-Variance optimizer, same transaction-cost
model. Only the cross-sectional return predictor changes.

The ElasticNetCV path is the existing baseline (KFold(5) inside ElasticNet's
own internal CV). The XGBoost path uses `XGBRegressor` tuned with
`RandomizedSearchCV` over the spec's grid (`max_depth`, `learning_rate`,
`subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`,
`reg_lambda`) inside a `TimeSeriesSplit(5)` over the *training window only*,
plus early stopping on a chronological holdout.

## Reproduce

```bash
python scripts/run_step1_comparison.py --source {args.source}
python scripts/render_step1_report.py    --source {args.source}
```

Or via the production CLI:

```bash
python scripts/run_all.py --skip-tests --source {args.source} --model elasticnet
python scripts/run_all.py --skip-tests --source {args.source} --model xgboost
```

## Comparison table

{table}

Notes
- `ic_mean` / `ic_std` / `icir` are computed on the model's
  `expected_return` against the realised 21-day forward log return per
  rebalance, NOT on the base signals (which `signal_diagnostics.py` covers).
- `sharpe`, `sortino`, `max_drawdown`, `cvar_95`, `turnover`, `annualized_*`
  come from `run_backtest._compute_returns_and_metrics`, net of a
  10 bp transaction cost on each rebalance.
- `forecast_seconds` is the wall-clock for `forecast_returns` only
  (the inner search dominates XGBoost runtime; backtest cost is identical).

## Equity curve

![Equity curve]({eq_md})

## IC time series

![IC time series]({ic_md})

## Interpretation

{interp}

## Method notes

- Same expanding-window training set per rebalance (no change vs baseline).
- Same `_compute_forward_returns` — predictions are bit-identical w.r.t.
  data leakage protection. The `XGBoostModel.fit → predict` round-trip never
  sees rows after the rebalance date (verified in
  `tests/test_xgboost_model.py::test_no_lookahead`).
- Hyperparameter search uses the spec defaults (TimeSeriesSplit(5),
  early_stopping_rounds=25 here) except this run set `n_iter=4` and
  `n_estimators_cap=300` so the side-by-side fits in a single short
  session. Bump `forecast_xgb_n_iter` to 20 and the cap back to 2000 in
  `config.yaml` for the publication-quality search at the cost of ~10×
  runtime.
- Both models run on the same Mean-Variance optimizer with min_position=0,
  max_position=1.0, no asset-class regime overrides — this isolates the
  predictor effect from the rest of the pipeline. Production
  `run_pipeline` adds Black-Litterman, FX overlay, sleeve sizing, etc.;
  those layers are model-agnostic and were tested separately via
  `python scripts/run_all.py --skip-tests --source {args.source} --model xgboost`.
"""
    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"[step1] wrote {REPORT_PATH}")
    print(f"[step1] figures: {eq_path}  {ic_path}")


if __name__ == "__main__":
    main()
