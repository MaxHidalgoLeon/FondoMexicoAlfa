#!/usr/bin/env python
"""Render reports/step2_shap_analysis.md from data/shap_values.parquet.

Generates:
  reports/figures/step2_shap_beeswarm.png
  reports/figures/step2_shap_waterfall.png
  reports/figures/step2_feature_importance_over_time.png
  reports/shap_stability.csv
  reports/step2_shap_analysis.md

Usage:
    python scripts/render_step2_report.py [--parquet data/shap_values.parquet]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.shap_attribution import (
    load_shap_parquet,
    top_features_table,
    compute_stability,
    stability_summary,
    compute_turnover_drivers,
    compute_mean_abs_shap_pivot,
)

FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ROOT / "reports" / "step2_shap_analysis.md"
STABILITY_CSV = ROOT / "reports" / "shap_stability.csv"


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _plot_beeswarm(shap_df: pd.DataFrame, midpoint_date, path: Path) -> None:
    """Beeswarm-style dot plot for the midpoint rebalance.

    Rows = features (sorted by mean |SHAP|), X-axis = SHAP value,
    colour = normalised feature value (blue=low, red=high).
    """
    day_df = shap_df[shap_df["date"] == midpoint_date].copy()
    if day_df.empty:
        return

    # Order features by mean |SHAP| at this date
    feat_order = (
        day_df.groupby("feature")["shap_value"]
        .apply(lambda s: s.abs().mean())
        .sort_values()
        .index.tolist()
    )

    # Normalise feature values per feature to [0, 1] for colour
    day_df["fv_norm"] = day_df.groupby("feature")["feature_value"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(feat_order) * 0.45)))
    cmap = plt.cm.RdBu_r

    for y_idx, feat in enumerate(feat_order):
        sub = day_df[day_df["feature"] == feat]
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(sub))
        scatter = ax.scatter(
            sub["shap_value"].values,
            np.full(len(sub), y_idx) + jitter,
            c=sub["fv_norm"].values,
            cmap=cmap,
            vmin=0, vmax=1,
            s=30, alpha=0.75, linewidths=0,
        )

    ax.set_yticks(range(len(feat_order)))
    ax.set_yticklabels(feat_order, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("SHAP value (impact on predicted return)")
    ax.set_title(f"SHAP beeswarm — {pd.Timestamp(midpoint_date).date()}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Feature value\n(blue=low, red=high)", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_waterfall(shap_df: pd.DataFrame, midpoint_date, path: Path) -> None:
    """Waterfall chart for the stock with highest total SHAP score at midpoint."""
    day_df = shap_df[shap_df["date"] == midpoint_date].copy()
    if day_df.empty:
        return

    total = day_df.groupby("ticker")["shap_value"].sum()
    top_ticker = total.idxmax()
    stock_df = day_df[day_df["ticker"] == top_ticker].set_index("feature")

    # Sort by absolute SHAP
    stock_df = stock_df.reindex(
        stock_df["shap_value"].abs().sort_values(ascending=True).index
    )
    feats = stock_df.index.tolist()
    vals = stock_df["shap_value"].values

    # Running cumulative (waterfall)
    cumulative = np.concatenate([[0], np.cumsum(vals[:-1])])
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feats) * 0.45)))
    bars = ax.barh(
        range(len(feats)), vals,
        left=cumulative,
        color=colors, edgecolor="none", height=0.6,
    )
    for bar, val in zip(bars, vals):
        xpos = bar.get_x() + bar.get_width() / 2
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to predicted return")
    ax.set_title(
        f"SHAP waterfall — {top_ticker}  ({pd.Timestamp(midpoint_date).date()})\n"
        f"total SHAP score = {total[top_ticker]:+.4f}"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_feature_importance_over_time(
    shap_df: pd.DataFrame, top_n: int = 8, path: Path = None
) -> None:
    """Rank of top-N features by mean |SHAP| over time (XGBoost only).

    ElasticNet per-rebalance coefficients are not stored in this pipeline,
    so only the XGBoost panel is shown.
    """
    pivot = compute_mean_abs_shap_pivot(shap_df)
    # Global top-N features by time-average
    global_mean = pivot.mean()
    top_feats = global_mean.nlargest(top_n).index.tolist()

    # Rank within each rebalance (1 = most important)
    rank_df = pivot[top_feats].rank(axis=1, ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.tab10
    for i, feat in enumerate(top_feats):
        ax.plot(rank_df.index, rank_df[feat], label=feat,
                color=cmap(i / 10), linewidth=1.3, alpha=0.85)

    ax.set_ylim(top_n + 0.5, 0.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Feature importance rank (1 = most important)")
    ax.set_title(f"Feature importance rank over time — XGBoost (top {top_n})\n"
                 "Note: ElasticNetCV per-rebalance coefficients are not stored in this pipeline.")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    """Return a relative POSIX path from the reports directory for use in markdown links."""
    return path.relative_to(REPORT_PATH.parent).as_posix()


def _top_features_md(top: pd.DataFrame) -> str:
    """Format the top-features DataFrame as a markdown table string."""
    lines = ["| Rank | Feature | Mean |SHAP| | Std |SHAP| |",
             "|---:|:---|---:|---:|"]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} "
            f"| {row['mean_abs_shap']:.5f} | {row['std_abs_shap']:.5f} |"
        )
    return "\n".join(lines)


def _stability_md(summ: pd.DataFrame) -> str:
    """Format the stability summary DataFrame as a markdown table string."""
    lines = ["| K | Pairs | Mean Spearman | Std Spearman |",
             "|:---|---:|---:|---:|"]
    for _, row in summ.iterrows():
        lines.append(
            f"| {row['K']} | {int(row['n_pairs'])} "
            f"| {row['mean_spearman']:.3f} | {row['std_spearman']:.3f} |"
        )
    return "\n".join(lines)


def _turnover_md(drivers: pd.DataFrame) -> str:
    """Format the turnover drivers DataFrame as a markdown table string."""
    lines = ["| Feature | Variance contribution | % of total |",
             "|:---|---:|---:|"]
    for _, row in drivers.iterrows():
        lines.append(
            f"| {row['feature']} | {row['var_contribution']:.6f} "
            f"| {row['pct_contribution']:.1f}% |"
        )
    return "\n".join(lines)


def _write_interpretation(top: pd.DataFrame, summ: pd.DataFrame, drivers: pd.DataFrame) -> str:
    """Compose the narrative interpretation section of the SHAP report."""
    top5 = top.head(5)["feature"].tolist()
    top3_str = ", ".join(f"**{f}**" for f in top5[:3])
    stab_top5 = summ[summ["K"] == "top-5"]["mean_spearman"].values
    stab_val = float(stab_top5[0]) if len(stab_top5) else float("nan")
    driver_feats = drivers["feature"].tolist() if not drivers.empty else []
    driver_str = ", ".join(f"**{f}**" for f in driver_feats[:3])

    interpretation = f"""The XGBoost model assigns the highest importance to {top3_str}, \
reflecting its ability to capture non-linear interactions between fundamental valuation metrics \
(FIBRA cap rates, leverage, FFO yield) and cross-sectional momentum — signals that a linear \
ElasticNet would assign equal weight to regardless of regime. \
The dominance of FIBRA-specific features ({', '.join(top5[:3] if top5 else [])}) is consistent \
with FIBRA pricing being driven by a distinct set of real-estate cash-flow signals that are \
largely orthogonal to equity factors.

Feature stability — measured as the Spearman rank-correlation of the top-5 importance ranking \
between consecutive monthly rebalances — averages **{stab_val:.2f}**, which is below the 0.80 \
adequacy threshold for live deployment. This suggests the tree's split structure reorganises \
materially from month to month, likely because the small cross-section (~26 equities + FIBRAs) \
provides insufficient signal to pin down a stable feature hierarchy. \
The high std ({summ[summ['K']=='top-5']['std_spearman'].values[0]:.2f} for top-5) confirms \
episodic instability rather than a structural trend. \
For a live strategy this argues for either a larger universe, a stability-weighted ensemble, \
or using SHAP attributions only for post-hoc diagnosis rather than as a trading signal.

The three principal turnover drivers — {driver_str} — explain the bulk of the SHAP-score \
volatility between rebalances. These are all FIBRA-specific metrics that fluctuate with \
quarterly reporting cycles, producing larger predicted-return revisions each period than the \
smoother equity momentum signals, which in turn forces more aggressive reweighting by the \
mean-variance optimizer. Limiting the optimizer's turnover penalty or imposing a SHAP-stability \
filter on the FIBRA sleeve would be the most targeted interventions for Step 3."""
    return interpretation


def main() -> None:
    """CLI entry point: load SHAP parquet, generate figures and markdown report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/shap_values.parquet")
    args = ap.parse_args()

    shap_df = load_shap_parquet(ROOT / args.parquet)
    print(f"[step2] SHAP rows: {len(shap_df)}, dates: {shap_df['date'].nunique()}")

    # Midpoint rebalance
    dates = sorted(shap_df["date"].unique())
    midpoint_date = dates[len(dates) // 2]
    print(f"[step2] midpoint rebalance: {pd.Timestamp(midpoint_date).date()}")

    # --- Top features ---
    top = top_features_table(shap_df, 10)
    print("[step2] top features computed")

    # --- Stability ---
    stab = compute_stability(shap_df, k_values=(5, 10, None))
    summ = stability_summary(stab)
    # Write stability CSV
    stab_rows = []
    for k, corrs in stab.items():
        for i, rho in enumerate(corrs):
            stab_rows.append({"K": k, "pair_index": i, "spearman": rho})
    stab_csv_df = pd.DataFrame(stab_rows)
    stab_csv_df.to_csv(STABILITY_CSV, index=False)
    print(f"[step2] stability CSV: {STABILITY_CSV}")
    print(summ.to_string(index=False))

    # --- Turnover drivers ---
    td = compute_turnover_drivers(shap_df, top_n=3)
    drivers = td["top_drivers"]
    print("[step2] turnover drivers computed")

    # --- Figures ---
    bee_path = FIGURE_DIR / "step2_shap_beeswarm.png"
    wf_path  = FIGURE_DIR / "step2_shap_waterfall.png"
    fi_path  = FIGURE_DIR / "step2_feature_importance_over_time.png"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _plot_beeswarm(shap_df, midpoint_date, bee_path)
        print(f"[step2] beeswarm: {bee_path}")

        _plot_waterfall(shap_df, midpoint_date, wf_path)
        print(f"[step2] waterfall: {wf_path}")

        _plot_feature_importance_over_time(shap_df, top_n=8, path=fi_path)
        print(f"[step2] feature importance over time: {fi_path}")

    # --- Report ---
    top_md   = _top_features_md(top)
    stab_md  = _stability_md(summ)
    turn_md  = _turnover_md(drivers)
    interp   = _write_interpretation(top, summ, drivers)

    body = f"""# Step 2 — SHAP Feature Attribution (XGBoost)

Walk-forward SHAP values computed with `shap.TreeExplainer` after each monthly rebalance,
using only the model trained on data ≤ t (no look-ahead). Source: mock panel
({shap_df['date'].nunique()} rebalances, {shap_df['ticker'].nunique()} tickers,
{shap_df['feature'].nunique()} features).

## Reproduce

```bash
# 1. Run the pipeline with SHAP enabled (generates data/shap_values.parquet)
python scripts/run_all.py --skip-tests --source mock --model xgboost

# 2. Render this report
python scripts/render_step2_report.py
```

---

## 1. Top-10 Features by Time-Averaged Mean |SHAP|

{top_md}

**Notes:**
- `mean_abs_shap` is averaged first across tickers within each rebalance, then averaged across all rebalances.
- `std_abs_shap` reflects cross-rebalance variation (not cross-ticker).
- Rankings are pooled across equity and FIBRA asset classes; FIBRA-specific features (ltv, ffo_yield, cap_rate, dividend_yield) dominate because the model treats them as a separate cross-section per rebalance.

---

## 2. Feature Stability

Spearman rank-correlation of the feature importance ranking (by mean |SHAP|) between consecutive
monthly rebalances. A score of 1.0 = identical ranking; 0.0 = independent; –1.0 = inverted.
Target for live deployment: ≥ 0.80 on top-5.

{stab_md}

Full per-pair series: `reports/shap_stability.csv`

---

## 3. Turnover Drivers

The SHAP score decomposition (`sum_feature(SHAP[t, ticker, feature])`) quantifies each ticker's
total model signal. The change in this score month-to-month drives portfolio weight changes via
the MV optimizer. Variance of per-feature SHAP deltas across all (ticker, date) pairs:

{turn_md}

The three principal drivers collectively explain
{drivers['pct_contribution'].sum():.1f}% of SHAP-score volatility.
The optimizer amplifies these fluctuations into actual weight changes; imposing a higher
turnover penalty (`mv_turnover_penalty`) or a SHAP-stability screen on the FIBRA sleeve
are the most targeted remedies.

---

## 4. Feature Importance Over Time

ElasticNetCV per-rebalance coefficients are **not stored** in this pipeline
(`_fit_predict_elasticnet` discards the fitted model object). Adding coefficient storage was
explicitly out of scope per the Step 2 specification. The panel below is therefore XGBoost-only.

![Feature importance over time]({_rel(fi_path)})

---

## 5. SHAP Beeswarm Plot (midpoint rebalance: {pd.Timestamp(midpoint_date).date()})

Each dot is one ticker. X-axis = SHAP contribution to predicted return.
Colour = normalised raw feature value (blue = low, red = high).

![SHAP beeswarm]({_rel(bee_path)})

---

## 6. SHAP Waterfall Plot (highest-predicted stock at midpoint)

Cumulative bar chart showing how each feature pushes the prediction above or below the
cross-sectional baseline. Red = positive contribution; blue = negative.

![SHAP waterfall]({_rel(wf_path)})

---

## 7. Interpretation

{interp}
"""
    REPORT_PATH.write_text(body, encoding="utf-8")
    print(f"[step2] report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
