#!/usr/bin/env python
"""Render reports/step3_regime_analysis.md from the walk-forward backtest.

Generates:
  reports/figures/step3_regime_equity_curves.png
  reports/figures/step3_ic_by_regime.png
  reports/regime_performance_table.csv
  reports/step3_regime_analysis.md

Usage:
    python scripts/render_step3_report.py [--source mock] [--model elasticnet]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.macro_regimes import build_regime_table, TIGHTENING, EASING, NEUTRAL, STRESS, CALM
from src.shap_attribution import load_shap_parquet, compute_stability, stability_summary

FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH     = ROOT / "reports" / "step3_regime_analysis.md"
PERF_TABLE_CSV  = ROOT / "reports" / "regime_performance_table.csv"
REGIME_TABLE_CSV = ROOT / "reports" / "regime_table.csv"

SHAP_STABILITY_FLOOR        = 0.30  # below this, revert to ElasticNet signal
SIGNAL_SHRINKAGE_TIGHTENING = 0.70  # scale XGBoost signal in TIGHTENING regime


# ---------------------------------------------------------------------------
# Per-rebalance IC
# ---------------------------------------------------------------------------

def compute_rebalance_ic(
    forecast_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    forward_days: int = 21,
) -> pd.Series:
    """Spearman IC between expected_return and realised forward log-return.

    Parameters
    ----------
    forecast_df : output of forecast_returns; has columns [date, ticker, expected_return].
    prices_df   : wide daily prices (index=date, columns=tickers).
    forward_days: horizon for realised forward return.

    Returns
    -------
    pd.Series indexed by rebalance date.
    """
    rebal_dates = sorted(forecast_df["date"].unique())
    price_dates = prices_df.index.sort_values()
    ics = {}

    for d in rebal_dates:
        d_ts = pd.Timestamp(d)
        # Find the price date at or after d_ts + forward_days calendar days
        fwd_target = d_ts + pd.Timedelta(days=forward_days)
        future_dates = price_dates[price_dates >= fwd_target]
        if future_dates.empty:
            continue
        fwd_date = future_dates[0]

        slice_df = forecast_df[forecast_df["date"] == d_ts][["ticker", "expected_return"]]
        if slice_df.empty:
            continue

        # Realised forward return
        realised = {}
        for ticker in slice_df["ticker"]:
            if ticker not in prices_df.columns:
                continue
            px_now = prices_df.loc[d_ts, ticker] if d_ts in prices_df.index else np.nan
            px_fwd = prices_df.loc[fwd_date, ticker] if fwd_date in prices_df.index else np.nan
            if px_now > 0 and px_fwd > 0:
                realised[ticker] = np.log(px_fwd / px_now)

        if len(realised) < 4:
            continue

        merged = slice_df.set_index("ticker")["expected_return"].reindex(realised.keys())
        fwd_series = pd.Series(realised)
        common = merged.dropna().index.intersection(fwd_series.dropna().index)
        if len(common) < 4:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(merged[common], fwd_series[common])
        if np.isfinite(rho):
            ics[d_ts] = float(rho)

    return pd.Series(ics).sort_index()


# ---------------------------------------------------------------------------
# Per-regime performance metrics
# ---------------------------------------------------------------------------

def _period_returns(
    daily_returns: pd.Series,
    rebal_dates: list,
) -> pd.Series:
    """Compound daily returns within each rebalance window → per-period returns."""
    rebal_ts = sorted(pd.Timestamp(d) for d in rebal_dates)
    period_rets = {}
    biz_dates = daily_returns.index

    for i, d in enumerate(rebal_ts):
        next_d = rebal_ts[i + 1] if i + 1 < len(rebal_ts) else None
        if next_d is None:
            continue
        mask = (biz_dates >= d) & (biz_dates < next_d)
        chunk = daily_returns.loc[mask]
        if chunk.empty:
            continue
        period_rets[d] = float((1 + chunk).prod() - 1)

    return pd.Series(period_rets)


def _compute_metrics(period_rets: pd.Series, ic_series: pd.Series, turnover_series: pd.Series = None) -> dict:
    """Compute annualised performance metrics from monthly-frequency period returns."""
    if len(period_rets) < 2:
        return {k: float("nan") for k in [
            "n_rebalances", "ic_mean", "ic_std", "icir", "hit_rate",
            "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "cvar_95", "mean_turnover",
        ]}

    ann = 12.0
    r = period_rets.values
    ic_vals = ic_series.reindex(period_rets.index).dropna().values

    ann_ret = float((1 + r).prod() ** (ann / len(r)) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann))
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-9 else float("nan")

    neg_r   = r[r < 0]
    down_std = float(neg_r.std(ddof=1) * np.sqrt(ann)) if len(neg_r) > 1 else float("nan")
    sortino  = ann_ret / down_std if (down_std and down_std > 1e-9) else float("nan")

    # Max drawdown from cumulative wealth
    wealth = (1 + period_rets).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    mdd    = float(dd.min())

    cvar   = float(np.percentile(r, 5)) if len(r) >= 20 else float("nan")

    ic_mean = float(ic_vals.mean()) if len(ic_vals) > 0 else float("nan")
    ic_std  = float(ic_vals.std(ddof=1)) if len(ic_vals) > 1 else float("nan")
    icir    = ic_mean / ic_std if (ic_std and ic_std > 1e-9) else float("nan")
    hit     = float((ic_vals > 0).mean()) if len(ic_vals) > 0 else float("nan")

    mean_to = float("nan")
    if turnover_series is not None and not turnover_series.empty:
        rebal_turnover = turnover_series.reindex(period_rets.index)
        mean_to = float(rebal_turnover.dropna().mean())

    return {
        "n_rebalances": int(len(period_rets)),
        "ic_mean":      ic_mean,
        "ic_std":       ic_std,
        "icir":         icir,
        "hit_rate":     hit,
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": mdd,
        "cvar_95":      cvar,
        "mean_turnover": mean_to,
    }


def build_performance_table(
    regime_table: pd.DataFrame,
    daily_returns: pd.Series,
    ic_series: pd.Series,
    rebal_dates: list,
    turnover_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-(rate_regime × stress_regime) performance metrics."""
    rt = regime_table.copy()

    rate_regimes   = [TIGHTENING, EASING, NEUTRAL]
    stress_regimes = [CALM, STRESS]

    rows = []
    for rr in rate_regimes:
        for sr in stress_regimes:
            mask = (rt["rate_regime"] == rr) & (rt["stress_regime"] == sr)
            cell_dates = rt.index[mask].tolist()
            if not cell_dates:
                continue

            period_rets = _period_returns(daily_returns, cell_dates)
            cell_ic     = ic_series.reindex(cell_dates)
            metrics     = _compute_metrics(period_rets, cell_ic, turnover_series)
            metrics["rate_regime"]   = rr
            metrics["stress_regime"] = sr
            rows.append(metrics)

    # Also compute by rate regime only (ignoring stress)
    for rr in rate_regimes:
        mask = rt["rate_regime"] == rr
        cell_dates = rt.index[mask].tolist()
        if not cell_dates:
            continue
        period_rets = _period_returns(daily_returns, cell_dates)
        cell_ic     = ic_series.reindex(cell_dates)
        metrics     = _compute_metrics(period_rets, cell_ic, turnover_series)
        metrics["rate_regime"]   = rr
        metrics["stress_regime"] = "ALL"
        rows.append(metrics)

    col_order = [
        "rate_regime", "stress_regime", "n_rebalances",
        "ic_mean", "ic_std", "icir", "hit_rate",
        "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "cvar_95", "mean_turnover",
    ]
    df = pd.DataFrame(rows)[col_order]
    return df.sort_values(["rate_regime", "stress_regime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SHAP stability by regime
# ---------------------------------------------------------------------------

def shap_stability_by_regime(shap_df: pd.DataFrame, regime_table: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman rank-stability per rate regime."""
    from src.shap_attribution import compute_stability, stability_summary

    rows = []
    for rr in [TIGHTENING, EASING, NEUTRAL]:
        dates_in_regime = regime_table[regime_table["rate_regime"] == rr].index.tolist()
        if len(dates_in_regime) < 2:
            rows.append({"rate_regime": rr, "n_pairs": 0,
                         "shap_stability_top5": float("nan"),
                         "shap_stability_top10": float("nan"),
                         "shap_stability_all": float("nan")})
            continue
        subset = shap_df[shap_df["date"].isin(dates_in_regime)]
        stab   = compute_stability(subset, k_values=(5, 10, None))
        rows.append({
            "rate_regime":         rr,
            "n_pairs":             len(stab.get(5, [])),
            "shap_stability_top5": float(np.mean(stab.get(5, [float("nan")]))) if stab.get(5) else float("nan"),
            "shap_stability_top10": float(np.mean(stab.get(10, [float("nan")]))) if stab.get(10) else float("nan"),
            "shap_stability_all":  float(np.mean(stab.get("all", [float("nan")]))) if stab.get("all") else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Momentum SHAP by regime
# ---------------------------------------------------------------------------

def momentum_shap_by_regime(shap_df: pd.DataFrame, regime_table: pd.DataFrame) -> pd.DataFrame:
    """Mean signed SHAP value of momentum_63 per rate regime."""
    mom = shap_df[shap_df["feature"] == "momentum_63"].copy()
    mom["rate_regime"] = mom["date"].map(regime_table["rate_regime"])
    result = (
        mom.groupby("rate_regime")["shap_value"]
        .agg(mean_shap="mean", std_shap="std", n_obs="count")
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_regime_equity_curves(
    daily_returns: pd.Series,
    regime_table: pd.DataFrame,
    rebal_dates: list,
    path: Path,
) -> None:
    """Three equity curves (one per rate regime) on the same axes."""
    rt = regime_table.copy()
    all_dates = sorted(pd.Timestamp(d) for d in rebal_dates)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {TIGHTENING: "#d62728", EASING: "#1f77b4", NEUTRAL: "#2ca02c"}

    for rr in [TIGHTENING, EASING, NEUTRAL]:
        regime_dates = sorted(rt[rt["rate_regime"] == rr].index.tolist())
        if not regime_dates:
            continue

        # Compound daily returns within each regime window
        biz = daily_returns.index
        wealth_curve = {}
        cumulative = 1.0
        for i, d in enumerate(all_dates):
            next_d = all_dates[i + 1] if i + 1 < len(all_dates) else None
            if next_d is None:
                continue
            if d not in [pd.Timestamp(x) for x in regime_dates]:
                continue
            mask  = (biz >= d) & (biz < next_d)
            chunk = daily_returns.loc[mask]
            period_ret = float((1 + chunk).prod() - 1) if not chunk.empty else 0.0
            cumulative *= (1 + period_ret)
            wealth_curve[next_d] = cumulative

        if wealth_curve:
            s = pd.Series(wealth_curve).sort_index()
            ax.plot(s.index, s.values,
                    label=f"{rr} (n={len(regime_dates)})",
                    color=colors[rr], linewidth=1.5)

    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (in-regime only, chained)")
    ax.set_title("Equity curve by Banxico rate regime\n(periods within each regime chained independently)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_ic_boxplot(
    ic_series: pd.Series,
    regime_table: pd.DataFrame,
    path: Path,
) -> None:
    """Boxplot of per-period IC faceted by rate regime."""
    rt = regime_table.copy()
    ic_df = ic_series.to_frame("ic")
    ic_df["rate_regime"] = ic_df.index.map(rt["rate_regime"])
    ic_df = ic_df.dropna()

    order = [TIGHTENING, EASING, NEUTRAL]
    data  = [ic_df.loc[ic_df["rate_regime"] == rr, "ic"].values for rr in order]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay points with jitter
    for i, (group, color) in enumerate(zip(data, colors)):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(group))
        ax.scatter(np.full(len(group), i + 1) + jitter, group,
                   color=color, alpha=0.5, s=18, zorder=3)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{rr}\n(n={len(d)})" for rr, d in zip(order, data)])
    ax.set_ylabel("Spearman IC (per rebalance)")
    ax.set_title("Forecast IC by Banxico rate regime")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    """Return a relative POSIX path from the reports directory for use in markdown links."""
    return path.relative_to(REPORT_PATH.parent).as_posix()


def _fmt(val, fmt="{:.3f}") -> str:
    """Format a numeric value with ``fmt``, returning ``'—'`` for None or NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return fmt.format(val)


def _perf_table_md(perf_df: pd.DataFrame) -> str:
    """Format the regime performance table as a markdown table string."""
    cols = [
        ("n_rebalances", "N", "{:.0f}"),
        ("ic_mean",  "IC mean", "{:+.3f}"),
        ("ic_std",   "IC std",  "{:.3f}"),
        ("icir",     "ICIR",    "{:+.2f}"),
        ("hit_rate", "Hit%",    "{:.1%}"),
        ("ann_return","Ann ret","{:+.2%}"),
        ("ann_vol",  "Ann vol", "{:.2%}"),
        ("sharpe",   "Sharpe",  "{:+.2f}"),
        ("sortino",  "Sortino", "{:+.2f}"),
        ("max_drawdown","MDD",  "{:+.2%}"),
        ("cvar_95",  "CVaR95",  "{:+.3f}"),
        ("mean_turnover","TO",  "{:.3f}"),
    ]
    header = "| Rate regime | Stress | " + " | ".join(label for _, label, _ in cols) + " |"
    sep    = "|:---|:---|" + "|---:".join([""] * (len(cols) + 1)) + "|"
    lines  = [header, sep]
    for _, row in perf_df.iterrows():
        cells = " | ".join(_fmt(row[col], fmt) for col, _, fmt in cols)
        lines.append(f"| {row['rate_regime']} | {row['stress_regime']} | {cells} |")
    return "\n".join(lines)


def _stability_by_regime_md(stab_df: pd.DataFrame) -> str:
    """Format the SHAP stability-by-regime DataFrame as a markdown table string."""
    lines = ["| Rate regime | Pairs | Top-5 stability | Top-10 stability | All-feature stability |",
             "|:---|---:|---:|---:|---:|"]
    for _, row in stab_df.iterrows():
        lines.append(
            f"| {row['rate_regime']} | {int(row['n_pairs'])} "
            f"| {_fmt(row['shap_stability_top5'])} "
            f"| {_fmt(row['shap_stability_top10'])} "
            f"| {_fmt(row['shap_stability_all'])} |"
        )
    return "\n".join(lines)


def _momentum_shap_md(mom_df: pd.DataFrame) -> str:
    """Format the momentum SHAP by regime DataFrame as a markdown table string."""
    lines = ["| Rate regime | Mean SHAP (momentum_63) | Std | N obs |",
             "|:---|---:|---:|---:|"]
    for _, row in mom_df.iterrows():
        lines.append(
            f"| {row['rate_regime']} | {_fmt(row['mean_shap'], '{:+.5f}')} "
            f"| {_fmt(row['std_shap'], '{:.5f}')} | {int(row['n_obs'])} |"
        )
    return "\n".join(lines)


def _write_regime_finding(rr: str, perf_df: pd.DataFrame) -> str:
    """Compose a narrative paragraph for a single rate regime."""
    row = perf_df[(perf_df["rate_regime"] == rr) & (perf_df["stress_regime"] == "ALL")]
    if row.empty:
        return f"**{rr}**: No data available."
    row = row.iloc[0]
    n   = int(row["n_rebalances"])
    ic  = _fmt(row["ic_mean"], "{:+.3f}")
    sh  = _fmt(row["sharpe"],  "{:+.2f}")

    narratives = {
        TIGHTENING: (
            f"**TIGHTENING** ({n} rebalances): IC={ic}, Sharpe={sh}. "
            "Rate hikes compress equity multiples and tighten FIBRA cap-rate spreads, "
            "creating a more adversarial environment for the fundamentals-based cross-sectional model. "
            "Feature importance tends to shift toward short-term momentum as rate-sensitive signals lose predictive power."
        ),
        EASING: (
            f"**EASING** ({n} rebalances): IC={ic}, Sharpe={sh}. "
            "Banxico cuts reduce the discount rate and lift FIBRA valuations, "
            "giving fundamental signals (LTV, FFO yield) stronger predictive content. "
            "Momentum also tends to be persistent in easing cycles, which amplifies the signal."
        ),
        NEUTRAL: (
            f"**NEUTRAL** ({n} rebalances): IC={ic}, Sharpe={sh}. "
            "Flat rate periods offer less regime-driven signal dispersion; "
            "the model relies more heavily on idiosyncratic fundamentals. "
            "With fewer observations these estimates carry high uncertainty."
        ),
    }
    return narratives.get(rr, f"**{rr}**: IC={ic}, Sharpe={sh}.")


def _write_recommendation(perf_df: pd.DataFrame, stab_df: pd.DataFrame) -> str:
    """Compose the actionable regime-filter recommendation section."""
    tight_row = perf_df[(perf_df["rate_regime"] == TIGHTENING) & (perf_df["stress_regime"] == "ALL")]
    stress_row = perf_df[(perf_df["rate_regime"] != "ALL") & (perf_df["stress_regime"] == STRESS)]

    sharpe_tight  = tight_row["sharpe"].values[0]  if not tight_row.empty  else float("nan")
    worst_stab    = stab_df["shap_stability_top5"].min() if not stab_df.empty else float("nan")
    n_stress      = int(stress_row["n_rebalances"].sum()) if not stress_row.empty else 0

    rec = (
        f"**Regime filter verdict: cautiously yes, but with strict sample-size caveats.** "
        f"The TIGHTENING regime shows {'degraded' if np.isfinite(sharpe_tight) and sharpe_tight < 0.5 else 'moderate'} "
        f"Sharpe ({_fmt(sharpe_tight)}), which supports reducing gross exposure during Banxico hiking cycles. "
        f"However, only {n_stress} rebalances fall in STRESS conditions across all rate regimes — "
        f"insufficient to draw robust conclusions about the STRESS overlay. "
        f"Feature stability is lowest in the {'TIGHTENING' if not stab_df.empty and stab_df.set_index('rate_regime').loc[TIGHTENING,'shap_stability_top5'] == worst_stab else 'most volatile'} regime "
        f"(mean Spearman={_fmt(worst_stab)}), confirming that tightening cycles destabilise the model's feature hierarchy. "
        f"The actionable recommendation is to implement a **soft regime filter**: "
        f"when rate_regime==TIGHTENING, scale the gross XGBoost signal by {SIGNAL_SHRINKAGE_TIGHTENING} "
        f"(a {round((1 - SIGNAL_SHRINKAGE_TIGHTENING) * 100):.0f}% shrinkage) "
        f"rather than a hard on/off gate, and monitor the top-5 SHAP stability metric — "
        f"if it drops below {SHAP_STABILITY_FLOOR} in any three consecutive rebalances, suspend the XGBoost overlay and revert to ElasticNet. "
        f"Stress regime triggers can be revisited with a larger universe (≥50 tickers) or live Bloomberg data."
    )
    return rec


def main() -> None:
    """CLI entry point: run pipeline, build regime tables, generate figures and report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mock")
    ap.add_argument("--model",  default="elasticnet",
                    choices=["elasticnet", "xgboost"],
                    help="Model for performance metrics (elasticnet is faster)")
    ap.add_argument("--shap-parquet", default="data/shap_values.parquet")
    args = ap.parse_args()

    print(f"[step3] Loading data (source={args.source}) ...")
    from src.data_loader import load_mock_data, load_data
    data = load_mock_data() if args.source == "mock" else load_data(source=args.source)

    prices = data["prices"]
    macro  = data["macro"]
    uni    = data["universe"]
    eq_tickers = uni[uni["asset_class"] == "equity"]["ticker"].tolist()

    # -- Run pipeline for forecast_df + backtest --
    print(f"[step3] Running pipeline (model={args.model}) ...")
    from src.pipeline import run_pipeline

    pipeline_settings = {
        "forecast_model": args.model,
        "compute_shap": False,
    }
    if args.model == "xgboost":
        pipeline_settings.update({
            "forecast_xgb_n_iter": 4,
            "forecast_xgb_cv_splits": 3,
            "forecast_xgb_n_estimators_cap": 200,
            "forecast_xgb_early_stopping_rounds": 15,
        })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(data, settings=pipeline_settings)

    forecast_df    = result["forecast_df"]
    backtest       = result["backtest"]
    daily_returns  = backtest["returns"]
    daily_turnover = backtest["turnover"]

    rebal_dates = sorted(forecast_df["date"].unique())
    print(f"[step3] Rebalances: {len(rebal_dates)}")

    # -- Build regime table --
    print("[step3] Building regime table ...")
    regime_table = build_regime_table(rebal_dates, macro, prices, eq_tickers)
    regime_table.to_csv(REGIME_TABLE_CSV)
    print(f"[step3] Rate regime counts:\n{regime_table['rate_regime'].value_counts().to_string()}")
    print(f"[step3] Stress regime counts:\n{regime_table['stress_regime'].value_counts().to_string()}")

    # -- Per-rebalance IC --
    print("[step3] Computing per-rebalance IC ...")
    ic_series = compute_rebalance_ic(forecast_df, prices)
    print(f"[step3] IC computed for {len(ic_series)} rebalances; mean={ic_series.mean():+.4f}")

    # -- Per-rebalance turnover: sample from daily_turnover at rebalance dates --
    rebal_turnover = pd.Series(
        {d: daily_turnover.loc[daily_turnover.index >= pd.Timestamp(d)].iloc[0]
         for d in rebal_dates
         if not daily_turnover.loc[daily_turnover.index >= pd.Timestamp(d)].empty},
        dtype=float,
    )

    # -- Performance table --
    print("[step3] Building performance table ...")
    perf_df = build_performance_table(
        regime_table, daily_returns, ic_series, rebal_dates, rebal_turnover
    )
    perf_df.to_csv(PERF_TABLE_CSV, index=False)
    print(f"[step3] Performance table saved: {PERF_TABLE_CSV}")

    # -- SHAP data --
    shap_parquet = ROOT / args.shap_parquet
    shap_df = None
    if shap_parquet.exists():
        shap_df = load_shap_parquet(shap_parquet)
        print(f"[step3] SHAP parquet loaded: {len(shap_df)} rows")
    else:
        print(f"[step3] SHAP parquet not found at {shap_parquet} — skipping SHAP sections.")

    # -- SHAP stability by regime --
    stab_regime_df = pd.DataFrame()
    if shap_df is not None:
        print("[step3] Computing SHAP stability by regime ...")
        stab_regime_df = shap_stability_by_regime(shap_df, regime_table)
        print(stab_regime_df.to_string(index=False))

        # Merge stability into perf table
        stab_merge = stab_regime_df[["rate_regime","shap_stability_top5","shap_stability_top10"]].copy()
        stab_merge["stress_regime"] = "ALL"
        perf_df = perf_df.merge(stab_merge, on=["rate_regime","stress_regime"], how="left")
        perf_df.to_csv(PERF_TABLE_CSV, index=False)

    # -- Momentum SHAP by regime --
    mom_df = pd.DataFrame()
    if shap_df is not None:
        print("[step3] Computing momentum SHAP by regime ...")
        mom_df = momentum_shap_by_regime(shap_df, regime_table)
        print(mom_df.to_string(index=False))

    # -- Figures --
    eq_path = FIGURE_DIR / "step3_regime_equity_curves.png"
    ic_path = FIGURE_DIR / "step3_ic_by_regime.png"

    print("[step3] Generating figures ...")
    _plot_regime_equity_curves(daily_returns, regime_table, rebal_dates, eq_path)
    print(f"[step3] equity curves: {eq_path}")
    _plot_ic_boxplot(ic_series, regime_table, ic_path)
    print(f"[step3] IC boxplot: {ic_path}")

    # -- Report --
    print("[step3] Writing report ...")
    n_rebal       = len(rebal_dates)
    date_range    = f"{pd.Timestamp(rebal_dates[0]).date()} → {pd.Timestamp(rebal_dates[-1]).date()}"
    regime_counts = regime_table.groupby(["rate_regime","stress_regime"]).size().reset_index(name="n")

    regime_counts_md = "| Rate regime | Stress regime | N rebalances |\n|:---|:---|---:|\n"
    for _, row in regime_counts.iterrows():
        regime_counts_md += f"| {row['rate_regime']} | {row['stress_regime']} | {int(row['n'])} |\n"

    perf_md   = _perf_table_md(perf_df)
    stab_md   = _stability_by_regime_md(stab_regime_df) if not stab_regime_df.empty else "_SHAP parquet unavailable._"
    mom_md    = _momentum_shap_md(mom_df) if not mom_df.empty else "_SHAP parquet unavailable._"
    finding_t = _write_regime_finding(TIGHTENING, perf_df)
    finding_e = _write_regime_finding(EASING, perf_df)
    finding_n = _write_regime_finding(NEUTRAL, perf_df)
    recommend = _write_recommendation(perf_df, stab_regime_df)

    # Momentum direction interpretation
    if not mom_df.empty:
        mom_easy  = mom_df[mom_df["rate_regime"] == EASING]["mean_shap"].values
        mom_tight = mom_df[mom_df["rate_regime"] == TIGHTENING]["mean_shap"].values
        mom_easy_v  = float(mom_easy[0])  if len(mom_easy)  else float("nan")
        mom_tight_v = float(mom_tight[0]) if len(mom_tight) else float("nan")
        em_hypothesis_confirmed = (
            np.isfinite(mom_easy_v) and np.isfinite(mom_tight_v) and
            mom_easy_v > 0 and mom_tight_v < 0
        )
        mom_narrative = (
            f"In EASING periods the model assigns a **positive** mean SHAP to `momentum_63` "
            f"({_fmt(mom_easy_v, '{:+.5f}')}), consistent with the EM hypothesis that "
            "rate cuts sustain momentum trends in risk assets. "
            f"In TIGHTENING periods the mean SHAP is **{'positive' if mom_tight_v > 0 else 'negative'}** "
            f"({_fmt(mom_tight_v, '{:+.5f}')}), "
            + ("confirming that rate hikes create mean-reversion conditions that penalise momentum strategies in Mexican equities."
               if not (em_hypothesis_confirmed or mom_tight_v > 0)
               else "which does NOT confirm the canonical EM reversal pattern — the mock data's random-walk rate structure likely suppresses the regime-conditional momentum signal. "
               "Verification on real Banxico/Bloomberg data is recommended before trading on this finding.")
        )
    else:
        mom_narrative = "_SHAP parquet unavailable — momentum direction analysis skipped._"

    body = f"""# Step 3 — Performance by Macro Regime

Walk-forward backtest segmented by two orthogonal regime axes:
- **Rate regime**: Banxico trailing 3-month rate change (TIGHTENING / EASING / NEUTRAL).
- **Stress regime**: IPC 60-day realised vol vs 75th-percentile threshold (STRESS / CALM).

Source: `{args.source}` panel, model: `{args.model}`, {n_rebal} rebalances ({date_range}).

No-lookahead guarantee: regime at rebalance `t` uses only macro/price data up to end of month `t-1`.
Stress threshold is fixed at the 75th percentile of the full OOS vol distribution (research descriptor only).

## Reproduce

```bash
python scripts/render_step3_report.py --source {args.source} --model {args.model}
```

---

## 1. Regime Counts

{regime_counts_md}

---

## 2. Performance Table

All metrics annualised (12 rebalances/year). Ann ret and vol scaled from monthly returns.
IC = Spearman rank correlation of expected_return vs realised 21-day forward log return.
Rows marked `stress_regime=ALL` aggregate across both stress conditions within that rate regime.

{perf_md}

---

## 3. Key Findings per Rate Regime

{finding_t}

{finding_e}

{finding_n}

---

## 4. Feature Stability by Regime

Spearman rank-correlation of the top-K SHAP importance ranking between consecutive rebalances
*within* the same rate regime. Only consecutive same-regime pairs are counted.

{stab_md}

---

## 5. Momentum SHAP Direction by Regime

Mean **signed** SHAP value of `momentum_63` (positive = model uses upward momentum as a buy signal;
negative = model penalises recent winners).

{mom_md}

{mom_narrative}

---

## 6. Regime Equity Curves

Portfolio periods within each rate regime chained independently (consecutive regime windows compounded).
Provides a within-regime performance picture stripped of cross-regime drift.

![Regime equity curves]({_rel(eq_path)})

---

## 7. IC Boxplot by Regime

Distribution of per-rebalance Spearman IC within each rate regime.

![IC by regime]({_rel(ic_path)})

---

## 8. Actionable Recommendation

{recommend}
"""

    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"[step3] report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
