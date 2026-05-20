#!/usr/bin/env python
"""Step 1 — LightGBM vs ElasticNet side-by-side driver.

Runs `forecast_returns` + `run_backtest` twice (one model each) on the
same data panel, writes per-model artifacts under reports/output/step1/,
and emits a metrics summary used by `reports/step1_lightgbm_vs_elasticnet.md`.

Idempotent: cached pickles under reports/output/step1/ are reused unless
`--force` is passed. Defaults to the locally-cached Bloomberg parquet.

Usage:
    python scripts/run_step1_comparison.py
    python scripts/run_step1_comparison.py --source mock --force
    python scripts/run_step1_comparison.py --models lightgbm
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest
from src.data_loader import load_data
from src.features import build_signal_matrix
from src.signals import forecast_returns, score_cross_section

OUT_DIR = ROOT / "reports" / "output" / "step1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _safe_div(num: float, den: float) -> float:
    """Return num/den, or NaN if den is zero or non-finite."""
    return float(num / den) if den != 0 and np.isfinite(den) else float("nan")


def _hit_rate(returns: pd.Series) -> float:
    """Fraction of positive daily returns (NaN if series is empty)."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def _annualized_sortino(returns: pd.Series, risk_free: float = 0.02) -> float:
    """Annualised Sortino ratio using downside deviation of excess returns."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    daily_rf = risk_free / 252
    excess = r - daily_rf
    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")
    return _safe_div(excess.mean() * 252, downside.std(ddof=0) * np.sqrt(252))


def _ic_per_period(forecast_df: pd.DataFrame, prices: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Spearman IC of the model's expected_return against realized H-day forward return."""
    if forecast_df.empty:
        return pd.Series(dtype=float)
    fwd = np.log(prices.shift(-horizon) / prices).replace([np.inf, -np.inf], np.nan)
    rows = []
    for date, group in forecast_df.groupby("date"):
        if date not in fwd.index:
            continue
        fwd_row = fwd.loc[date]
        merged = group[["ticker", "expected_return"]].copy()
        merged["fwd"] = merged["ticker"].map(fwd_row)
        clean = merged.dropna(subset=["expected_return", "fwd"])
        if len(clean) < 4:
            continue
        rho, _ = spearmanr(clean["expected_return"], clean["fwd"])
        if np.isfinite(rho):
            rows.append((pd.Timestamp(date), float(rho)))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.name = "ic"
    return s


def _summarize(forecast_df, prices, returns, metrics, transaction_cost: float) -> dict:
    """Aggregate IC, return, and risk metrics into a single summary dict."""
    ic = _ic_per_period(forecast_df, prices)
    raw_returns = returns.copy()
    # Net-of-cost Sharpe is the per-rebalance net Sharpe — already in metrics["sharpe"]
    return {
        "ic_mean": float(ic.mean()) if not ic.empty else float("nan"),
        "ic_std": float(ic.std(ddof=0)) if not ic.empty else float("nan"),
        "icir": _safe_div(ic.mean(), ic.std(ddof=0)) if not ic.empty else float("nan"),
        "hit_rate": _hit_rate(raw_returns),
        "annualized_return": float(metrics.get("annualized_return", np.nan)),
        "annualized_vol": float(metrics.get("annualized_vol", np.nan)),
        "sharpe": float(metrics.get("sharpe", np.nan)),
        "sortino": float(metrics.get("sortino", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "cvar_95": float(metrics.get("cvar_95", np.nan)),
        "turnover": float(metrics.get("turnover", np.nan)),
        "n_periods_ic": int(len(ic)),
        "n_periods_returns": int(len(raw_returns.dropna())),
        "transaction_cost": float(transaction_cost),
    }


def run_one(
    model_name: str,
    feature_df: pd.DataFrame,
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    log_returns: pd.DataFrame,
    settings: dict,
    transaction_cost: float = 0.001,
    optimizer: str = "mv",
) -> dict:
    """Forecast + backtest with the given model. Returns artifact dict."""
    cfg = dict(settings)
    cfg["forecast_model"] = model_name

    fi_tickers = universe.loc[universe["asset_class"] == "fixed_income", "ticker"].tolist()
    prices_opt = prices[[c for c in prices.columns if c not in fi_tickers]]
    optimizable_universe = universe[universe["asset_class"].isin(["equity", "fibra"])].copy()
    ac_map = optimizable_universe.set_index("ticker")["asset_class"].to_dict()
    asset_class_constraints = {
        "__asset_class_map__": ac_map,
        "equity": {"min": 0.0, "max": 1.0},
        "fibra": {"min": 0.0, "max": 1.0},
    }
    adtv_scores = universe.set_index("ticker")["liquidity_score"].astype(float)

    t0 = time.time()
    forecast_df = forecast_returns(scored, log_returns, settings=cfg)
    forecast_elapsed = time.time() - t0

    forecast_df_opt = (
        forecast_df[~forecast_df["ticker"].isin(fi_tickers)].copy()
        if not forecast_df.empty
        else forecast_df
    )

    t0 = time.time()
    bt = run_backtest(
        prices_opt,
        forecast_df_opt,
        optimizable_universe,
        transaction_cost=transaction_cost,
        optimizer=optimizer,
        adtv_scores=adtv_scores,
        asset_class_constraints=asset_class_constraints,
        settings=cfg,
    )
    backtest_elapsed = time.time() - t0

    summary = _summarize(forecast_df_opt, prices_opt, bt["returns"], bt["metrics"], transaction_cost)
    summary["model"] = model_name
    summary["forecast_seconds"] = forecast_elapsed
    summary["backtest_seconds"] = backtest_elapsed
    return {
        "model": model_name,
        "forecast_df": forecast_df_opt,
        "weights": bt["weights"],
        "returns": bt["returns"],
        "turnover": bt["turnover"],
        "metrics": bt["metrics"],
        "ic_series": _ic_per_period(forecast_df_opt, prices_opt),
        "summary": summary,
    }


def main() -> None:
    """CLI entry point: parse args, load data, run both models, save artifacts."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="bloomberg", choices=["mock", "bloomberg", "yahoo", "refinitiv"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--models", default="elasticnet,lightgbm")
    p.add_argument("--n-iter", type=int, default=10, help="LightGBM RandomizedSearchCV draws")
    p.add_argument("--cv-splits", type=int, default=5, help="LightGBM inner TimeSeriesSplit folds")
    p.add_argument("--n-estimators-cap", type=int, default=2000, help="LightGBM upper bound on boosting rounds")
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--force", action="store_true", help="Re-run even if cached pickles exist")
    args = p.parse_args()

    base_settings = {
        "forecast_lgbm_n_iter": args.n_iter,
        "forecast_lgbm_cv_splits": args.cv_splits,
        "forecast_lgbm_n_estimators_cap": args.n_estimators_cap,
        "forecast_lgbm_early_stopping_rounds": args.early_stopping_rounds,
    }

    print(f"[step1] source={args.source} models={args.models} n_iter={args.n_iter} cv_splits={args.cv_splits}")
    data = load_data(source=args.source, start_date=args.start, end_date=args.end)
    universe = data["universe"]
    prices = data["prices"]

    feature_df = build_signal_matrix(
        prices, data["fundamentals"], data["fibra_fundamentals"], data["bonds"], data["macro"], universe
    )
    scored = score_cross_section(feature_df)
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    artifacts: dict[str, dict] = {}
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        cache_path = OUT_DIR / f"{model_name}_{args.source}.pkl"
        if cache_path.exists() and not args.force:
            print(f"[step1] {model_name}: loading cached → {cache_path}")
            with open(cache_path, "rb") as f:
                artifacts[model_name] = pickle.load(f)
            continue
        print(f"[step1] {model_name}: running forecast + backtest …")
        art = run_one(
            model_name,
            feature_df,
            scored,
            prices,
            universe,
            log_returns,
            settings=base_settings,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(art, f)
        artifacts[model_name] = art
        s = art["summary"]
        print(
            f"[step1] {model_name}: ic_mean={s['ic_mean']:+.4f} ICIR={s['icir']:+.3f} "
            f"sharpe={s['sharpe']:+.3f} ann_ret={s['annualized_return']:+.3f} "
            f"mdd={s['max_drawdown']:+.3f} fc_secs={s['forecast_seconds']:.1f}"
        )

    summary_path = OUT_DIR / f"summary_{args.source}.json"
    summaries = {m: a["summary"] for m, a in artifacts.items()}
    import json
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=float)
    print(f"[step1] wrote summaries → {summary_path}")


if __name__ == "__main__":
    main()
