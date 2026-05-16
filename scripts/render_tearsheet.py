#!/usr/bin/env python
"""Render FMIA_Tearsheet.html and FMIA_Tearsheet.pdf from existing report artefacts.

Loads only pre-computed files from reports/ and reports/figures/.
Does NOT rerun any backtest, SHAP, or regime computation.

Usage:
    python scripts/render_tearsheet.py

Outputs:
    reports/FMIA_Tearsheet.html  (all images embedded as base64 data URIs)
    reports/FMIA_Tearsheet.pdf
"""
from __future__ import annotations

import base64
import re
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(path: Path) -> str:
    """Return base64-encoded PNG as a data URI, or '' if file missing."""
    if not path.exists():
        print(f"  [warn] missing figure: {path.name}")
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _img(path: Path, style: str = "") -> str:
    """Return an <img> tag with embedded data URI, or a placeholder."""
    uri = _b64(path)
    if not uri:
        return f'<div class="missing">Figure not found: {path.name}</div>'
    return f'<img src="{uri}" style="{style}" />'


def _fmt(val, fmt=".3f", color=True) -> str:
    """Format a number; optionally wrap in green/red span."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = f"{val:{fmt}}"
    if color and isinstance(val, (int, float)) and np.isfinite(val):
        cls = "pos" if val > 0 else ("neg" if val < 0 else "")
        if cls:
            return f'<span class="{cls}">{s}</span>'
    return s


def _fmt_pct(val) -> str:
    """Format a fraction (0–1) as a percentage string with one decimal place."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return _fmt(val * 100, ".1f") + "%"


# ---------------------------------------------------------------------------
# Source data parsers
# ---------------------------------------------------------------------------

def _parse_step1_table(md_path: Path) -> list[dict]:
    """Parse the Step 1 comparison markdown table into a list of row dicts."""
    if not md_path.exists():
        print(f"  [warn] missing: {md_path.name}")
        return []
    text = md_path.read_text(encoding="utf-8")
    # Find lines between '## Comparison table' and 'Notes'
    in_table = False
    rows = []
    for line in text.splitlines():
        if "## Comparison table" in line:
            in_table = True
            continue
        if in_table and line.startswith("Notes"):
            break
        if in_table and line.startswith("|") and not line.startswith("|---|"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts[0] == "Metric":
                continue
            if len(parts) >= 4:
                rows.append({
                    "metric":    parts[0],
                    "elastic":   parts[1],
                    "xgboost":   parts[2],
                    "delta":     parts[3],
                })
    return rows


def _parse_step2_tables(md_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse top-10 features table and stability table from step2 markdown."""
    if not md_path.exists():
        print(f"  [warn] missing: {md_path.name}")
        return [], []

    text = md_path.read_text(encoding="utf-8")
    top10, stability = [], []

    # Top-10 table: lines between '## 1.' and '---'
    in_top = False
    in_stab = False
    for line in text.splitlines():
        if "## 1. Top" in line:
            in_top, in_stab = True, False
            continue
        if "## 2." in line and "stability" in line.lower():
            in_stab, in_top = True, False
            continue
        if "## 3." in line:
            in_stab = False
            continue
        if line.strip() == "---":
            in_top = in_stab = False
            continue

        if (in_top or in_stab) and line.startswith("|") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if not parts:
                continue
            if in_top and parts[0] not in ("Rank", "---"):
                if len(parts) >= 4:
                    top10.append({"rank": parts[0], "feature": parts[1],
                                  "mean_abs": parts[2], "std_abs": parts[3]})
            if in_stab and parts[0] not in ("K", "---"):
                if len(parts) >= 4:
                    stability.append({"K": parts[0], "pairs": parts[1],
                                      "mean": parts[2], "std": parts[3]})
    return top10, stability


def _load_regime_perf() -> pd.DataFrame:
    """Load regime_performance_table.csv; returns an empty DataFrame if absent."""
    csv = REPORTS / "regime_performance_table.csv"
    if not csv.exists():
        print(f"  [warn] missing: {csv.name}")
        return pd.DataFrame()
    return pd.read_csv(csv)


def _extract_cover_stats(step1_rows: list[dict]) -> dict:
    """Pull headline XGBoost metrics for the cover stat-strip."""
    lookup = {r["metric"]: r for r in step1_rows}
    def xgb(key):
        """Look up the XGBoost value for a given metric name."""
        row = lookup.get(key, {})
        return row.get("xgboost", "—")
    return {
        "sharpe":  xgb("Sharpe"),
        "icir":    xgb("ICIR"),
        "mdd":     xgb("Max drawdown"),
    }


# ---------------------------------------------------------------------------
# HTML builders (one function per page)
# ---------------------------------------------------------------------------

CSS = """
@page {
  size: A4 landscape;
  margin: 14mm;
  @bottom-left   { content: "FondoMexicoAlfa — Confidential Research"; font-size: 7pt; color: #6B7280; font-family: Arial, sans-serif; }
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 7pt; color: #6B7280; font-family: Arial, sans-serif; }
  @bottom-right  { content: "May 2026"; font-size: 7pt; color: #6B7280; font-family: Arial, sans-serif; }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: Arial, Helvetica, sans-serif;
  background: #F7F8FA;
  color: #1A1A1A;
  font-size: 8.5pt;
  line-height: 1.45;
}
.page {
  page-break-after: always;
  min-height: 180mm;
  padding: 4mm 0;
}
.page:last-child { page-break-after: auto; }

/* Typography */
h1 { font-size: 22pt; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; }
h2 { font-size: 9pt; font-weight: 700; color: #FFFFFF; letter-spacing: 0.5px; text-transform: uppercase; }
h3 { font-size: 8.5pt; font-weight: 700; color: #1B2B4B; margin: 3mm 0 2mm; }
p  { margin-bottom: 2mm; color: #6B7280; }

/* Colours */
.pos { color: #16A34A; font-weight: 600; }
.neg { color: #DC2626; font-weight: 600; }
.muted { color: #6B7280; }
.accent { color: #2563EB; }

/* Section header bar */
.section-header {
  background: #1B2B4B;
  padding: 2mm 4mm;
  margin-bottom: 4mm;
}
.section-header h2 { margin: 0; color: #FFFFFF; }

/* Cover header */
.cover-header {
  background: #1B2B4B;
  padding: 6mm 8mm;
  margin-bottom: 5mm;
}
.cover-header h1 { color: #FFFFFF; margin-bottom: 1mm; }
.cover-sub { font-size: 9pt; color: #D1D5DB; margin-top: 2mm; }

/* Stat strip (cover) */
.stat-strip { display: flex; gap: 6mm; margin: 5mm 0; }
.stat-box {
  flex: 1;
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  padding: 5mm 7mm;
  text-align: center;
}
.stat-box .big { font-size: 24pt; font-weight: 700; }
.stat-box .label { font-size: 7.5pt; color: #6B7280; margin-top: 1mm; }

/* Tagline */
.tagline { font-size: 7.5pt; color: #6B7280; margin-top: 3mm; }
.cover-date { font-size: 7pt; color: #6B7280; margin-top: 2mm; }

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 7.5pt;
  margin-bottom: 3mm;
}
th {
  background: #1B2B4B;
  color: #FFFFFF;
  padding: 2mm 3mm;
  text-align: right;
  font-weight: 700;
  white-space: nowrap;
}
th:first-child { text-align: left; }
td {
  padding: 1.5mm 3mm;
  text-align: right;
  border-bottom: 1px solid #D1D5DB;
  color: #6B7280;
}
td:first-child { text-align: left; color: #1A1A1A; font-weight: 500; }
tr:nth-child(even) td { background: #F7F8FA; }
tr:nth-child(odd)  td { background: #FFFFFF; }

/* Figures */
.fig-frame {
  border: 1px solid #D1D5DB;
  padding: 3mm;
  background: #FFFFFF;
  margin: 2mm 0;
}
.fig-frame img { width: 100%; height: auto; display: block; }
.fig-caption { font-size: 7pt; color: #6B7280; font-style: italic; text-align: center; margin-top: 1mm; }
.fig-row { display: flex; gap: 4mm; margin: 2mm 0; }
.fig-row .fig-frame { flex: 1; }
.missing {
  background: #F7F8FA;
  border: 1px dashed #D1D5DB;
  color: #6B7280;
  padding: 4mm;
  text-align: center;
  font-size: 7.5pt;
}

/* Two-column layout */
.two-col { display: flex; gap: 8mm; }
.two-col > * { flex: 1; }

/* Methodology bullets */
.method-list { list-style: none; }
.method-list li {
  padding: 1.5mm 0;
  border-bottom: 1px solid #D1D5DB;
  color: #6B7280;
  font-size: 7.5pt;
}
.method-list li strong { color: #1B2B4B; }

/* Disclaimer */
.disclaimer {
  margin-top: 4mm;
  padding: 3mm;
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  font-size: 6.5pt;
  color: #6B7280;
  line-height: 1.4;
}
"""


def _page_cover(stats: dict) -> str:
    """Return the HTML string for Page 1 (cover with headline stat strip)."""
    sharpe = stats.get("sharpe", "—")
    icir   = stats.get("icir",   "—")
    mdd    = stats.get("mdd",    "—")

    sharpe_html = f'<span class="pos">{sharpe}</span>' if sharpe != "—" else sharpe
    icir_html   = f'<span class="pos">{icir}</span>'   if icir   != "—" else icir
    mdd_html    = f'<span class="neg">{mdd}</span>'    if mdd    != "—" else mdd

    return f"""
<div class="page">
  <div class="cover-header">
    <h1>FondoMéxicoAlfa (FMIA)</h1>
    <div class="cover-sub">Systematic Long-Only Equity &amp; FIBRA &nbsp;·&nbsp; Quantitative Research Report</div>
  </div>

  <div class="stat-strip">
    <div class="stat-box">
      <div class="big pos">{sharpe_html}</div>
      <div class="label">XGBoost Sharpe Ratio</div>
    </div>
    <div class="stat-box">
      <div class="big pos">{icir_html}</div>
      <div class="label">XGBoost ICIR (Walk-forward)</div>
    </div>
    <div class="stat-box">
      <div class="big neg">{mdd_html}</div>
      <div class="label">Max Drawdown (XGBoost)</div>
    </div>
  </div>

  <p class="tagline">
    Walk-forward validated · Bloomberg data · Mexican equities &amp; FIBRAs (BMV) ·
    Monthly rebalancing · 10 bp transaction cost · Banxico regime conditioning
  </p>
  <p class="cover-date">Report date: {date.today().isoformat()}</p>
</div>
"""


def _page_model_comparison(step1_rows: list[dict]) -> str:
    """Return the HTML string for Page 2 (XGBoost vs ElasticNet model comparison table)."""
    if not step1_rows:
        rows_html = '<tr><td colspan="4" class="missing">Step 1 comparison data not found.</td></tr>'
    else:
        rows_html = ""
        for r in step1_rows:
            rows_html += f"""
        <tr>
          <td>{r['metric']}</td>
          <td>{r['elastic']}</td>
          <td>{r['xgboost']}</td>
          <td>{r['delta']}</td>
        </tr>"""

    eq_img = _img(FIGURES / "step1_equity_mock.png", "width:100%;")
    ic_img = _img(FIGURES / "step1_ic_mock.png", "width:100%;")

    return f"""
<div class="page">
  <div class="section-header"><h2>01 · Model Performance — XGBoost vs ElasticNetCV</h2></div>

  <table>
    <thead>
      <tr><th>Metric</th><th>ElasticNetCV</th><th>XGBoost</th><th>&Delta; (xgb &minus; elastic)</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="fig-row" style="margin-top:3mm;">
    <div class="fig-frame">
      {eq_img}
      <div class="fig-caption">Cumulative equity curves — XGBoost vs ElasticNetCV (108 rebalances)</div>
    </div>
    <div class="fig-frame">
      {ic_img}
      <div class="fig-caption">Rolling information coefficient time series</div>
    </div>
  </div>

  <p style="margin-top:2mm;font-size:6.5pt;color:#6B7280;">
    Same mock panel · same rebalance schedule · same MV optimizer · same 10 bp transaction cost.
    Only the cross-sectional return predictor changes. Walk-forward OOS: 108 monthly rebalances.
  </p>
</div>
"""


def _page_shap(top10: list[dict], stability: list[dict]) -> str:
    """Return the HTML string for Page 3 (SHAP feature attribution with beeswarm/waterfall figures)."""
    if top10:
        top10_rows = ""
        for r in top10:
            top10_rows += f"<tr><td>{r['rank']}</td><td>{r['feature']}</td><td>{r['mean_abs']}</td><td>{r['std_abs']}</td></tr>"
    else:
        top10_rows = '<tr><td colspan="4" class="missing">Top-10 feature data not found.</td></tr>'

    if stability:
        stab_rows = ""
        for r in stability:
            stab_rows += f"<tr><td>{r['K']}</td><td>{r['pairs']}</td><td>{r['mean']}</td><td>{r['std']}</td></tr>"
    else:
        stab_rows = '<tr><td colspan="4" class="missing">Stability data not found.</td></tr>'

    bee_img = _img(FIGURES / "step2_shap_beeswarm.png", "width:100%;")
    wf_img  = _img(FIGURES / "step2_shap_waterfall.png", "width:100%;")

    return f"""
<div class="page">
  <div class="section-header"><h2>02 · Feature Attribution (SHAP) — XGBoost Walk-forward</h2></div>

  <div class="two-col">
    <div>
      <h3>Top-10 Features by Time-Averaged Mean |SHAP|</h3>
      <table>
        <thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th><th>Std |SHAP|</th></tr></thead>
        <tbody>{top10_rows}</tbody>
      </table>

      <h3 style="margin-top:3mm;">Feature Stability (Spearman rank-correlation)</h3>
      <table>
        <thead><tr><th>K</th><th>Pairs</th><th>Mean &rho;</th><th>Std &rho;</th></tr></thead>
        <tbody>{stab_rows}</tbody>
      </table>
      <p style="font-size:6.5pt;margin-top:1mm;">
        Target for live deployment: &ge; 0.80 on top-5. Current: 0.44 (mock data).
        FIBRA fundamental features (ltv, ffo_yield, cap_rate) dominate importance.
      </p>
    </div>
    <div>
      <div class="fig-frame">
        {bee_img}
        <div class="fig-caption">Beeswarm — Midpoint Rebalance (2021-10-29)</div>
      </div>
      <div class="fig-frame" style="margin-top:3mm;">
        {wf_img}
        <div class="fig-caption">Waterfall — Highest-Predicted Stock</div>
      </div>
    </div>
  </div>
</div>
"""


def _page_regime(regime_df: pd.DataFrame) -> str:
    """Return the HTML string for Page 4 (macro regime performance table and equity curves figure)."""
    if not regime_df.empty:
        count_rows = ""
        for _, row in regime_df.iterrows():
            sr = row.get("stress_regime", "")
            rr = row.get("rate_regime", "")
            n  = int(row.get("n_rebalances", 0))
            ic = float(row.get("ic_mean", float("nan")))
            sh = float(row.get("sharpe", float("nan")))
            top5 = row.get("shap_stability_top5", float("nan"))
            count_rows += f"""<tr>
              <td>{rr}</td><td>{sr}</td><td style="text-align:center;">{n}</td>
              <td>{_fmt(ic, '+.3f')}</td>
              <td>{_fmt(sh, '+.2f')}</td>
              <td>{_fmt(top5 if pd.notna(top5) else float('nan'), '.3f', color=False)}</td>
            </tr>"""
    else:
        count_rows = '<tr><td colspan="6" class="missing">Regime performance data not found.</td></tr>'

    eq_img = _img(FIGURES / "step3_regime_equity_curves.png", "width:100%;")
    ic_img = _img(FIGURES / "step3_ic_by_regime.png", "width:100%;")

    return f"""
<div class="page">
  <div class="section-header"><h2>03 · Performance by Macro Regime</h2></div>

  <h3>Regime Performance Table (Banxico rate &times; IPC vol stress)</h3>
  <table>
    <thead>
      <tr><th>Rate regime</th><th>Stress</th><th>N</th>
          <th>IC mean</th><th>Sharpe</th><th>SHAP top-5 stability</th></tr>
    </thead>
    <tbody>{count_rows}</tbody>
  </table>
  <p style="font-size:6.5pt;margin-bottom:2mm;">
    Rate regime: Banxico trailing 3-month rate change. Stress: IPC-proxy 60d vol vs p75 threshold.
    No-lookahead: regime at rebalance t uses data strictly before t.
  </p>

  <div class="fig-row">
    <div class="fig-frame" style="flex:3;">
      {eq_img}
      <div class="fig-caption">Regime-conditioned equity curves</div>
    </div>
    <div class="fig-frame" style="flex:2;">
      {ic_img}
      <div class="fig-caption">IC distribution by regime</div>
    </div>
  </div>
</div>
"""


def _page_risk_methodology(step1_rows: list[dict]) -> str:
    """Return the HTML string for Page 5 (risk metrics, factor exposures, methodology summary)."""
    lookup = {r["metric"]: r.get("xgboost", "—") for r in step1_rows}
    sharpe  = lookup.get("Sharpe", "—")
    mdd     = lookup.get("Max drawdown", "—")
    cvar    = lookup.get("CVaR 95% (daily)", "—")
    vol     = lookup.get("Annualized vol", "—")
    to      = lookup.get("Turnover (per rebalance)", "—")
    ann_ret = lookup.get("Annualized return", "—")
    sortino = lookup.get("Sortino", "—")

    return f"""
<div class="page">
  <div class="section-header"><h2>04 · Risk Profile &amp; Methodology</h2></div>

  <div class="two-col">
    <div>
      <h3>XGBoost Risk Metrics (Walk-forward OOS)</h3>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Annualized return</td><td>{ann_ret}</td></tr>
          <tr><td>Annualized vol</td><td>{vol}</td></tr>
          <tr><td>Sharpe ratio</td><td>{sharpe}</td></tr>
          <tr><td>Sortino ratio</td><td>{sortino}</td></tr>
          <tr><td>Max drawdown</td><td><span class="neg">{mdd}</span></td></tr>
          <tr><td>CVaR 95% (daily)</td><td>{cvar}</td></tr>
          <tr><td>Avg turnover / rebalance</td><td>{to}</td></tr>
        </tbody>
      </table>

      <h3 style="margin-top:3mm;">Regime Filter Recommendation</h3>
      <ul class="method-list">
        <li><strong>EASING:</strong> Full XGBoost signal (stability 0.57)</li>
        <li><strong>TIGHTENING:</strong> Scale signal &times;0.7 (stability 0.41)</li>
        <li><strong>STRESS trigger:</strong> IPC vol &gt; p75 &rarr; reduce exposure</li>
        <li><strong>Safety valve:</strong> if top-5 stability &lt; 0.30 for 3 periods, revert to ElasticNet</li>
      </ul>
    </div>
    <div>
      <h3>Methodology</h3>
      <ul class="method-list">
        <li><strong>Universe:</strong> Mexican equities + FIBRAs (BMV), ~26 instruments</li>
        <li><strong>Signal:</strong> XGBoost cross-sectional return forecast (RandomizedSearchCV + TimeSeriesSplit)</li>
        <li><strong>Alternative signal:</strong> ElasticNetCV (baseline, interchangeable)</li>
        <li><strong>Optimizer:</strong> Mean-variance (SLSQP) with market-impact penalty</li>
        <li><strong>Rebalance frequency:</strong> Monthly (end-of-month)</li>
        <li><strong>Validation:</strong> Walk-forward OOS, 108 rebalances (2017–2026)</li>
        <li><strong>Transaction costs:</strong> 10 bp per side, applied each rebalance</li>
        <li><strong>Regime conditioning:</strong> Banxico rate (TIGHTENING / EASING / NEUTRAL) &times; IPC vol stress</li>
        <li><strong>Feature attribution:</strong> Per-rebalance SHAP values via TreeExplainer</li>
        <li><strong>Data sources:</strong> Bloomberg (production) · mock data (research)</li>
        <li><strong>Regulatory framework:</strong> CNBV position limits, LFI structure</li>
      </ul>

      <h3 style="margin-top:3mm;">SHAP Feature Stability</h3>
      <ul class="method-list">
        <li><strong>Overall top-5 stability:</strong> 0.44 mean Spearman (target &ge; 0.80)</li>
        <li><strong>Best regime (EASING):</strong> 0.57 — model most reliable</li>
        <li><strong>Top turnover drivers:</strong> ffo_yield (18%), ltv (15%), momentum_63 (14%)</li>
      </ul>
    </div>
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong> For research purposes only. Past performance does not guarantee future results.
    All results based on walk-forward out-of-sample simulation on mock data unless stated otherwise.
    Mock data is a statistical approximation of real market behaviour and should not be used for investment decisions.
    Bloomberg data results require live terminal access. This document is confidential and intended solely for internal research.
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def build_html(
    step1_rows: list[dict],
    top10: list[dict],
    stability: list[dict],
    regime_df: pd.DataFrame,
) -> str:
    """Assemble all 5 pages into a complete HTML document string."""
    stats = _extract_cover_stats(step1_rows)

    pages = "".join([
        _page_cover(stats),
        _page_model_comparison(step1_rows),
        _page_shap(top10, stability),
        _page_regime(regime_df),
        _page_risk_methodology(step1_rows),
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FMIA Tearsheet — FondoMéxicoAlfa</title>
<style>
{CSS}
</style>
</head>
<body>
{pages}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# PDF rendering (WeasyPrint first, fpdf2 fallback)
# ---------------------------------------------------------------------------

def _render_pdf_weasyprint(html_path: Path, pdf_path: Path) -> bool:
    """Attempt to render the HTML file to PDF using WeasyPrint; returns True on success."""
    try:
        import weasyprint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return True
    except Exception as exc:
        print(f"  [warn] WeasyPrint failed ({exc}); falling back to fpdf2.")
        return False


def _ascii(text: str) -> str:
    """Replace common non-latin-1 chars so fpdf2 core fonts (latin-1) don't error."""
    return (str(text)
            .replace("—", "-").replace("–", "-")  # em/en dash
            .replace("→", "->").replace("≥", ">=").replace("≤", "<=")
            .replace("Δ", "D").replace("−", "-")  # Delta, minus sign
            .replace("ρ", "r")                     # rho
            .replace("é", "e").replace("á", "a")  # accented vowels
            .encode("latin-1", errors="replace").decode("latin-1"))


# ---------------------------------------------------------------------------
# fpdf2 colour palette — institutional navy/white design system
# ---------------------------------------------------------------------------

_PDF_NAVY     = (27,  43,  75)   # #1B2B4B
_PDF_WHITE    = (255, 255, 255)  # #FFFFFF
_PDF_OFFWHITE = (247, 248, 250)  # #F7F8FA
_PDF_TEXT     = (26,  26,  26)   # #1A1A1A
_PDF_MUTED    = (107, 114, 128)  # #6B7280
_PDF_GREEN    = (22,  163, 74)   # #16A34A
_PDF_RED      = (220, 38,  38)   # #DC2626
_PDF_ACCENT   = (37,  99,  235)  # #2563EB
_PDF_RULE     = (209, 213, 219)  # #D1D5DB


def _pdf_set_text(pdf) -> None:
    """Set fpdf2 text colour to body TEXT."""
    pdf.set_text_color(*_PDF_TEXT)


def _pdf_set_muted(pdf) -> None:
    """Set fpdf2 text colour to MUTED grey."""
    pdf.set_text_color(*_PDF_MUTED)


def _pdf_set_white(pdf) -> None:
    """Set fpdf2 text colour to WHITE."""
    pdf.set_text_color(*_PDF_WHITE)


def _pdf_section_header(pdf, title: str) -> None:
    """Render a full-width NAVY strip with WHITE bold text (11pt, 7mm tall)."""
    from fpdf.enums import XPos, YPos
    pdf.set_fill_color(*_PDF_NAVY)
    _pdf_set_white(pdf)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _ascii(title), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)


def _pdf_add_img_safe(
    pdf,
    path: Path,
    x: float | None = None,
    y: float | None = None,
    w: float | None = None,
    h: float | None = None,
    caption: str = "",
) -> float:
    """Insert a framed image with a RULE-colored border and optional caption.

    The frame is ``w`` mm wide total; the image inside has 3mm padding on each
    side.  Returns the total height consumed (frame + caption) in mm, or 0 if
    the file is missing or an error occurs.
    """
    if not path.exists():
        return 0
    PAD = 3.0
    try:
        cx = x if x is not None else pdf.get_x()
        cy = y if y is not None else pdf.get_y()
        cw = w if w is not None else pdf.epw
        img_w = cw - 2 * PAD

        # Place image; retrieve rendered height from fpdf2
        info = pdf.image(str(path), x=cx + PAD, y=cy + PAD, w=img_w, h=h or 0)
        disp_h = info.rendered_height
        frame_h = disp_h + 2 * PAD

        # Draw RULE border over image edges
        pdf.set_draw_color(*_PDF_RULE)
        pdf.set_line_width(0.3)
        pdf.rect(cx, cy, cw, frame_h)

        total = frame_h
        if caption:
            pdf.set_xy(cx, cy + frame_h + 1)
            pdf.set_font("Helvetica", "I", 7.5)
            _pdf_set_muted(pdf)
            pdf.cell(cw, 4, _ascii(caption), align="C")
            total += 5

        return total
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# PDF page builders (new institutional design)
# ---------------------------------------------------------------------------

def _pdf_page_cover(pdf, step1_rows: list[dict]) -> None:
    """Render Page 1 — institutional cover with NAVY header and stat strip."""
    from fpdf.enums import XPos, YPos
    stats = _extract_cover_stats(step1_rows)
    pdf.add_page()

    # NAVY header block
    header_h = 22
    pdf.set_fill_color(*_PDF_NAVY)
    pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, header_h, style="F")
    pdf.set_xy(pdf.l_margin + 3, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 20)
    _pdf_set_white(pdf)
    pdf.cell(0, 10, "FondoMexicoAlfa (FMIA)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(pdf.l_margin + 3, pdf.get_y())
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_PDF_RULE)
    pdf.cell(0, 5, "Systematic Long-Only Equity & FIBRA  |  Quantitative Research Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(pdf.get_y() + (header_h - (pdf.get_y() - pdf.t_margin - 0)) + 6)
    pdf.set_y(pdf.t_margin + header_h + 5)

    # Stat strip — three boxes
    labels = [
        ("XGBoost Sharpe", stats.get("sharpe", "-"), _PDF_GREEN),
        ("XGBoost ICIR",   stats.get("icir",   "-"), _PDF_GREEN),
        ("Max Drawdown",   stats.get("mdd",    "-"), _PDF_RED),
    ]
    col_w = pdf.epw / 3
    box_h = 22
    y_stat = pdf.get_y()

    for i, (label, val, color) in enumerate(labels):
        bx = pdf.l_margin + i * col_w
        # Box fill
        pdf.set_fill_color(*_PDF_WHITE)
        pdf.set_draw_color(*_PDF_RULE)
        pdf.set_line_width(0.3)
        pdf.rect(bx, y_stat, col_w - 2, box_h, style="FD")
        # Value
        pdf.set_xy(bx + 2, y_stat + 3)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*color)
        pdf.cell(col_w - 6, 10, str(val), align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        # Label
        pdf.set_xy(bx + 2, y_stat + 14)
        pdf.set_font("Helvetica", "", 7)
        _pdf_set_muted(pdf)
        pdf.cell(col_w - 6, 4, label, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_xy(pdf.l_margin, y_stat + box_h + 5)
    pdf.set_font("Helvetica", "I", 7.5)
    _pdf_set_muted(pdf)
    pdf.cell(0, 5,
             "Walk-forward validated  *  Bloomberg data  *  Mexican equities & FIBRAs (BMV)"
             "  *  Monthly rebalancing  *  10 bp transaction cost",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 5, f"Report date: {date.today().isoformat()}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _pdf_page_model_comparison(pdf, step1_rows: list[dict]) -> None:
    """Render Page 2 — XGBoost vs ElasticNet comparison table and framed figures."""
    from fpdf.enums import XPos, YPos
    pdf.add_page()
    _pdf_section_header(pdf, "01  MODEL PERFORMANCE - XGBoost vs ElasticNetCV")

    # Table header
    col_widths = [72, 38, 38, 40]
    headers = ["Metric", "ElasticNetCV", "XGBoost", "D (xgb - elastic)"]
    pdf.set_fill_color(*_PDF_NAVY)
    _pdf_set_white(pdf)
    pdf.set_font("Helvetica", "B", 7)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 5.5, h, fill=True, align="R" if h != "Metric" else "L")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 7)
    for i, row in enumerate(step1_rows):
        bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
        pdf.set_fill_color(*bg)
        _pdf_set_text(pdf)
        pdf.cell(col_widths[0], 4.5, _ascii(row["metric"]), fill=True)
        _pdf_set_muted(pdf)
        for val, cw in [(row["elastic"], col_widths[1]),
                        (row["xgboost"], col_widths[2]),
                        (row["delta"],   col_widths[3])]:
            pdf.cell(cw, 4.5, _ascii(str(val)), fill=True, align="R")
        pdf.ln()

    # RULE line below table
    pdf.set_draw_color(*_PDF_RULE)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + sum(col_widths), pdf.get_y())

    pdf.ln(4)
    y_fig = pdf.get_y()
    w_fig = pdf.epw / 2 - 2

    _pdf_add_img_safe(pdf, FIGURES / "step1_equity_mock.png",
                      x=pdf.l_margin, y=y_fig, w=w_fig,
                      caption="Cumulative equity curves (108 rebalances)")
    _pdf_add_img_safe(pdf, FIGURES / "step1_ic_mock.png",
                      x=pdf.l_margin + w_fig + 4, y=y_fig, w=w_fig,
                      caption="Rolling information coefficient (IC) time series")


def _pdf_page_shap(pdf, top10: list[dict], stability: list[dict]) -> None:
    """Render Page 3 — SHAP feature attribution tables and framed figures."""
    from fpdf.enums import XPos, YPos
    pdf.add_page()
    _pdf_section_header(pdf, "02  FEATURE ATTRIBUTION (SHAP) - XGBoost Walk-forward")

    half = pdf.epw / 2 - 3
    y0 = pdf.get_y()

    # Left column: top-10 table
    pdf.set_font("Helvetica", "B", 8)
    _pdf_set_text(pdf)
    pdf.cell(half, 5, "Top-10 Features by Mean |SHAP|",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*_PDF_NAVY)
    _pdf_set_white(pdf)
    pdf.set_font("Helvetica", "B", 7)
    for hdr, cw in [("#", 10), ("Feature", 45), ("Mean |SHAP|", 30), ("Std", 27)]:
        pdf.cell(cw, 5, hdr, fill=True, align="C" if hdr != "Feature" else "L")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for i, r in enumerate(top10[:10]):
        bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
        pdf.set_fill_color(*bg)
        _pdf_set_text(pdf)
        pdf.cell(10, 4.5, r["rank"],    fill=True, align="C")
        pdf.cell(45, 4.5, r["feature"], fill=True)
        _pdf_set_muted(pdf)
        pdf.cell(30, 4.5, r["mean_abs"], fill=True, align="R")
        pdf.cell(27, 4.5, r["std_abs"],  fill=True, align="R")
        pdf.ln()

    pdf.ln(3)
    # Stability table
    pdf.set_font("Helvetica", "B", 8)
    _pdf_set_text(pdf)
    pdf.cell(half, 5, "Feature Stability (Spearman rank-correlation)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*_PDF_NAVY)
    _pdf_set_white(pdf)
    pdf.set_font("Helvetica", "B", 7)
    for hdr, cw in [("K", 20), ("Pairs", 22), ("Mean r", 30), ("Std r", 30)]:
        pdf.cell(cw, 5, hdr, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for i, r in enumerate(stability):
        bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
        pdf.set_fill_color(*bg)
        _pdf_set_text(pdf)
        pdf.cell(20, 4.5, r["K"],     fill=True)
        _pdf_set_muted(pdf)
        pdf.cell(22, 4.5, r["pairs"], fill=True, align="C")
        pdf.cell(30, 4.5, r["mean"],  fill=True, align="R")
        pdf.cell(30, 4.5, r["std"],   fill=True, align="R")
        pdf.ln()

    # Right column: two stacked framed figures
    x_right = pdf.l_margin + half + 6
    fig_w = half
    _pdf_add_img_safe(pdf, FIGURES / "step2_shap_beeswarm.png",
                      x=x_right, y=y0, w=fig_w, h=48,
                      caption="Beeswarm - Midpoint Rebalance (2021-10-29)")
    _pdf_add_img_safe(pdf, FIGURES / "step2_shap_waterfall.png",
                      x=x_right, y=y0 + 60, w=fig_w, h=44,
                      caption="Waterfall - Highest-Predicted Stock")


def _pdf_page_regime(pdf, regime_df: pd.DataFrame) -> None:
    """Render Page 4 — macro regime performance table and framed equity curves."""
    from fpdf.enums import XPos, YPos
    pdf.add_page()
    _pdf_section_header(pdf, "03  PERFORMANCE BY MACRO REGIME")

    if not regime_df.empty:
        cols4 = [("Rate regime", 48), ("Stress", 28), ("N", 14),
                 ("IC mean", 28), ("Sharpe", 26), ("SHAP top-5 stab.", 30)]
        pdf.set_fill_color(*_PDF_NAVY)
        _pdf_set_white(pdf)
        pdf.set_font("Helvetica", "B", 7)
        for hdr, cw in cols4:
            pdf.cell(cw, 5.5, hdr, fill=True, align="C" if hdr != "Rate regime" else "L")
        pdf.ln()

        pdf.set_font("Helvetica", "", 7)
        for i, (_, row) in enumerate(regime_df.iterrows()):
            bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
            pdf.set_fill_color(*bg)
            _pdf_set_text(pdf)
            pdf.cell(48, 4.5, _ascii(str(row.get("rate_regime",  ""))), fill=True)
            pdf.cell(28, 4.5, _ascii(str(row.get("stress_regime", ""))), fill=True)
            pdf.cell(14, 4.5, str(int(row.get("n_rebalances", 0))), fill=True, align="C")
            _pdf_set_muted(pdf)
            ic   = row.get("ic_mean",             float("nan"))
            sh   = row.get("sharpe",               float("nan"))
            top5 = row.get("shap_stability_top5",  float("nan"))
            pdf.cell(28, 4.5, f"{ic:+.3f}"  if pd.notna(ic)   else "-", fill=True, align="R")
            pdf.cell(26, 4.5, f"{sh:+.2f}"  if pd.notna(sh)   else "-", fill=True, align="R")
            pdf.cell(30, 4.5, f"{top5:.3f}" if pd.notna(top5) else "-", fill=True, align="R")
            pdf.ln()

        pdf.set_draw_color(*_PDF_RULE)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + sum(c for _, c in cols4), pdf.get_y())

    pdf.ln(4)
    y_fig4 = pdf.get_y()
    w3 = pdf.epw * 0.58
    w4 = pdf.epw * 0.40

    _pdf_add_img_safe(pdf, FIGURES / "step3_regime_equity_curves.png",
                      x=pdf.l_margin, y=y_fig4, w=w3,
                      caption="Regime-conditioned cumulative equity curves")
    _pdf_add_img_safe(pdf, FIGURES / "step3_ic_by_regime.png",
                      x=pdf.l_margin + w3 + 4, y=y_fig4, w=w4,
                      caption="IC distribution by macro regime")


def _pdf_page_risk_methodology(pdf, step1_rows: list[dict]) -> None:
    """Render Page 5 — risk metrics table and methodology summary."""
    from fpdf.enums import XPos, YPos
    pdf.add_page()
    _pdf_section_header(pdf, "04  RISK PROFILE & METHODOLOGY")

    lookup = {r["metric"]: r.get("xgboost", "-") for r in step1_rows}
    risk_items = [
        ("Annualized return",      lookup.get("Annualized return",        "-")),
        ("Annualized vol",         lookup.get("Annualized vol",           "-")),
        ("Sharpe ratio",           lookup.get("Sharpe",                   "-")),
        ("Sortino ratio",          lookup.get("Sortino",                  "-")),
        ("Max drawdown",           lookup.get("Max drawdown",             "-")),
        ("CVaR 95% (daily)",       lookup.get("CVaR 95% (daily)",         "-")),
        ("Avg turnover/rebalance", lookup.get("Turnover (per rebalance)", "-")),
    ]
    method_items = [
        ("Universe",        "Mexican equities + FIBRAs (BMV), ~26 instruments"),
        ("Signal",          "XGBoost cross-sectional return forecast (RandomizedSearchCV + TimeSeriesSplit)"),
        ("Baseline signal", "ElasticNetCV (interchangeable via config flag)"),
        ("Optimizer",       "Mean-variance (SLSQP) with market-impact penalty"),
        ("Rebalance",       "Monthly (end-of-month), walk-forward OOS"),
        ("Validation",      "108 rebalances, 2017-2026"),
        ("Costs",           "10 bp per side applied each rebalance"),
        ("Regimes",         "Banxico rate (TIGHTENING/EASING/NEUTRAL) x IPC vol stress"),
        ("Attribution",     "Per-rebalance SHAP via TreeExplainer (shap 0.51)"),
        ("Data",            "Bloomberg (production) * mock data (research)"),
        ("Framework",       "CNBV position limits, LFI structure"),
    ]

    half5 = pdf.epw / 2 - 3
    y5 = pdf.get_y()

    # Left: risk table
    pdf.set_font("Helvetica", "B", 8)
    _pdf_set_text(pdf)
    pdf.cell(half5, 5, "XGBoost Risk Metrics (Walk-forward OOS)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(*_PDF_NAVY)
    _pdf_set_white(pdf)
    pdf.set_font("Helvetica", "B", 7)
    m_w, v_w = half5 * 0.58, half5 * 0.42
    pdf.cell(m_w, 5, "Metric", fill=True)
    pdf.cell(v_w, 5, "Value",  fill=True, align="R")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for i, (label, val) in enumerate(risk_items):
        bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
        pdf.set_fill_color(*bg)
        _pdf_set_text(pdf)
        pdf.cell(m_w, 4.5, label,         fill=True)
        _pdf_set_muted(pdf)
        pdf.cell(v_w, 4.5, _ascii(str(val)), fill=True, align="R")
        pdf.ln()

    # Right: methodology list
    pdf.set_xy(pdf.l_margin + half5 + 6, y5)
    pdf.set_font("Helvetica", "B", 8)
    _pdf_set_text(pdf)
    pdf.cell(half5, 5, "Methodology", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_xy(pdf.l_margin + half5 + 6, y5 + 7)
    pdf.set_font("Helvetica", "", 7)
    for i, (label, desc) in enumerate(method_items):
        pdf.set_xy(pdf.l_margin + half5 + 6, pdf.get_y())
        bg = _PDF_OFFWHITE if i % 2 == 0 else _PDF_WHITE
        pdf.set_fill_color(*bg)
        pdf.set_font("Helvetica", "B", 7)
        _pdf_set_text(pdf)
        pdf.cell(30, 4.5, label + ":", fill=True)
        pdf.set_font("Helvetica", "", 7)
        _pdf_set_muted(pdf)
        pdf.cell(half5 - 30, 4.5, _ascii(desc[:68]), fill=True)
        pdf.ln()

    # Disclaimer at bottom
    pdf.set_y(-40)
    pdf.set_fill_color(*_PDF_OFFWHITE)
    pdf.set_draw_color(*_PDF_RULE)
    pdf.set_line_width(0.3)
    pdf.set_font("Helvetica", "I", 6)
    _pdf_set_muted(pdf)
    pdf.multi_cell(0, 3.5,
        "DISCLAIMER: For research purposes only. Past performance does not guarantee future results. "
        "All results based on walk-forward out-of-sample simulation. Mock data used where Bloomberg "
        "data unavailable. This document is confidential and intended solely for internal research use.",
        fill=True, border=1,
    )


def _render_pdf_fpdf2(html_path: Path, pdf_path: Path, step1_rows, top10, stability, regime_df) -> None:
    """Build a compact multi-page PDF using fpdf2 (no system library deps)."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    class TearsheetPDF(FPDF):
        def normalize_text(self, text: str) -> str:
            """Sanitise text so all chars fit within the latin-1 encoding of core fonts."""
            return _ascii(text)

        def header(self):
            """No page header — suppressed intentionally."""
            pass

        def footer(self):
            """Render three-column footer with RULE separator line."""
            self.set_y(-12)
            # RULE line above footer
            self.set_draw_color(*_PDF_RULE)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y() - 1,
                      self.w - self.r_margin, self.get_y() - 1)
            self.set_font("Helvetica", "", 7)
            _pdf_set_muted(self)
            col_w = self.epw / 3
            self.set_x(self.l_margin)
            self.cell(col_w, 4, "FondoMexicoAlfa - Confidential Research", align="L")
            self.cell(col_w, 4, f"Page {self.page_no()} of {{nb}}", align="C")
            self.cell(col_w, 4, "May 2026", align="R")

    pdf = TearsheetPDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(auto=True, margin=14)

    _pdf_page_cover(pdf, step1_rows)
    _pdf_page_model_comparison(pdf, step1_rows)
    _pdf_page_shap(pdf, top10, stability)
    _pdf_page_regime(pdf, regime_df)
    _pdf_page_risk_methodology(pdf, step1_rows)

    pdf.output(str(pdf_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load all source data, build HTML, write it, then render PDF (WeasyPrint -> fpdf2 fallback)."""
    print("[tearsheet] Loading source data ...")

    step1_rows = _parse_step1_table(REPORTS / "step1_xgboost_vs_elasticnet.md")
    top10, stability = _parse_step2_tables(REPORTS / "step2_shap_analysis.md")
    regime_df = _load_regime_perf()

    print(f"  step1 rows: {len(step1_rows)}")
    print(f"  top10 features: {len(top10)}, stability rows: {len(stability)}")
    print(f"  regime_df rows: {len(regime_df)}")

    # Build HTML
    print("[tearsheet] Building HTML ...")
    html_path = REPORTS / "FMIA_Tearsheet.html"
    pdf_path  = REPORTS / "FMIA_Tearsheet.pdf"

    html = build_html(step1_rows, top10, stability, regime_df)
    html_path.write_text(html, encoding="utf-8")
    html_kb = html_path.stat().st_size // 1024
    print(f"[tearsheet] HTML written: {html_path}  ({html_kb} KB)")

    # Verify no local file paths in <img> src attributes
    img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html)
    local_refs = [s for s in img_srcs if not s.startswith("data:")]
    if local_refs:
        print(f"  [warn] {len(local_refs)} non-data-URI image ref(s) found: {local_refs[:3]}")
    else:
        print(f"  all {len(img_srcs)} image(s) embedded as data URIs OK")

    # Render PDF
    print("[tearsheet] Rendering PDF ...")
    success = _render_pdf_weasyprint(html_path, pdf_path)
    if not success:
        _render_pdf_fpdf2(html_path, pdf_path, step1_rows, top10, stability, regime_df)

    if pdf_path.exists():
        pdf_kb = pdf_path.stat().st_size // 1024
        print(f"[tearsheet] PDF written:  {pdf_path}  ({pdf_kb} KB)")
        if pdf_kb > 15_000:
            print(f"  [warn] PDF exceeds 15 MB target ({pdf_kb} KB)")
    else:
        print("[tearsheet] ERROR: PDF not written.")
        sys.exit(1)


if __name__ == "__main__":
    main()
