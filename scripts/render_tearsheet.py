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
  margin: 18mm;
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages) " | FMIA — FondoMéxicoAlfa | Confidential";
    font-size: 7pt;
    color: #8b949e;
    font-family: -apple-system, Segoe UI, Arial, sans-serif;
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, Segoe UI, Arial, sans-serif;
  background: #0d1117;
  color: #e6edf3;
  font-size: 8.5pt;
  line-height: 1.45;
}
.page {
  page-break-after: always;
  min-height: 190mm;
  padding: 4mm 0;
}
.page:last-child { page-break-after: auto; }

/* Typography */
h1 { font-size: 26pt; font-weight: 700; color: #39d353; letter-spacing: -0.5px; }
h2 { font-size: 11pt; font-weight: 600; color: #8b949e; margin-bottom: 6mm; letter-spacing: 1px; text-transform: uppercase; }
h3 { font-size: 9pt; font-weight: 600; color: #e6edf3; margin: 3mm 0 2mm; }
p  { margin-bottom: 2mm; color: #8b949e; }

/* Colours */
.pos { color: #39d353; }
.neg { color: #f85149; }
.amber { color: #f0883e; }
.muted { color: #8b949e; }

/* Stat strip (cover) */
.stat-strip { display: flex; gap: 12mm; margin: 8mm 0; }
.stat-box {
  flex: 1;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6mm 8mm;
  text-align: center;
}
.stat-box .big { font-size: 28pt; font-weight: 700; }
.stat-box .label { font-size: 7.5pt; color: #8b949e; margin-top: 1mm; }

/* Cover */
.cover-header {
  margin-bottom: 4mm;
  border-bottom: 2px solid #39d353;
  padding-bottom: 3mm;
}
.cover-sub { font-size: 11pt; color: #8b949e; margin-top: 2mm; }
.tagline { font-size: 8pt; color: #8b949e; margin-top: 4mm; }
.cover-date { font-size: 8pt; color: #30363d; margin-top: 2mm; }

/* Section header bar */
.section-header {
  background: #161b22;
  border-left: 4px solid #39d353;
  padding: 2mm 4mm;
  margin-bottom: 4mm;
}
.section-header h2 { margin: 0; font-size: 10pt; }

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 7.5pt;
  margin-bottom: 3mm;
}
th {
  background: #161b22;
  color: #8b949e;
  padding: 2mm 3mm;
  text-align: right;
  border-bottom: 1px solid #30363d;
  font-weight: 600;
  white-space: nowrap;
}
th:first-child { text-align: left; }
td {
  padding: 1.5mm 3mm;
  text-align: right;
  border-bottom: 1px solid #21262d;
}
td:first-child { text-align: left; color: #e6edf3; }
tr:nth-child(even) td { background: #0d1117; }
tr:nth-child(odd)  td { background: #161b22; }

/* Figures */
.fig-row { display: flex; gap: 4mm; margin: 2mm 0; }
.fig-row img { flex: 1; max-width: 50%; height: auto; border: 1px solid #30363d; border-radius: 4px; }
img { max-width: 100%; height: auto; }
.fig-full img { width: 100%; border: 1px solid #30363d; border-radius: 4px; }
.missing {
  background: #161b22;
  border: 1px dashed #30363d;
  color: #8b949e;
  padding: 4mm;
  text-align: center;
  font-size: 7.5pt;
  border-radius: 4px;
}

/* Two-column layout */
.two-col { display: flex; gap: 8mm; }
.two-col > * { flex: 1; }

/* Methodology bullets */
.method-list { list-style: none; }
.method-list li {
  padding: 1.5mm 0;
  border-bottom: 1px solid #21262d;
  color: #8b949e;
  font-size: 7.5pt;
}
.method-list li strong { color: #e6edf3; }

/* Disclaimer */
.disclaimer {
  margin-top: 4mm;
  padding: 3mm;
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  font-size: 6.5pt;
  color: #8b949e;
  line-height: 1.4;
}
"""


def _page_cover(stats: dict) -> str:
    """Return the HTML string for Page 1 (cover with headline stat strip)."""
    sharpe = stats.get("sharpe", "—")
    icir   = stats.get("icir",   "—")
    mdd    = stats.get("mdd",    "—")

    # Colour-code sharpe and icir green, mdd red
    sharpe_html = f'<span class="pos">{sharpe}</span>' if sharpe != "—" else sharpe
    icir_html   = f'<span class="pos">{icir}</span>'   if icir   != "—" else icir
    mdd_html    = f'<span class="neg">{mdd}</span>'    if mdd    != "—" else mdd

    return f"""
<div class="page">
  <div class="cover-header">
    <h1>FondoMéxicoAlfa (FMIA)</h1>
    <div class="cover-sub">Systematic Long-Only Equity &amp; FIBRA &nbsp;|&nbsp; Quantitative Research Report</div>
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

    eq_img = _img(FIGURES / "step1_equity_mock.png",
                  "width:100%;border:1px solid #30363d;border-radius:4px;")
    ic_img = _img(FIGURES / "step1_ic_mock.png",
                  "width:100%;border:1px solid #30363d;border-radius:4px;")

    return f"""
<div class="page">
  <div class="section-header"><h2>01 · Model Performance — XGBoost vs ElasticNetCV</h2></div>

  <table>
    <thead>
      <tr><th>Metric</th><th>ElasticNetCV</th><th>XGBoost</th><th>Δ (xgb − elastic)</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="fig-row" style="margin-top:3mm;">
    <div style="flex:1;">{eq_img}</div>
    <div style="flex:1;">{ic_img}</div>
  </div>

  <p style="margin-top:2mm;font-size:6.5pt;">
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

    bee_img = _img(FIGURES / "step2_shap_beeswarm.png",
                   "max-height:90mm;width:auto;border:1px solid #30363d;border-radius:4px;")
    wf_img  = _img(FIGURES / "step2_shap_waterfall.png",
                   "max-height:90mm;width:auto;border:1px solid #30363d;border-radius:4px;")

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
        <thead><tr><th>K</th><th>Pairs</th><th>Mean ρ</th><th>Std ρ</th></tr></thead>
        <tbody>{stab_rows}</tbody>
      </table>
      <p style="font-size:6.5pt;margin-top:1mm;">
        Target for live deployment: ≥ 0.80 on top-5. Current: 0.44 (mock data).
        FIBRA fundamental features (ltv, ffo_yield, cap_rate) dominate importance.
      </p>
    </div>
    <div>
      <h3>Beeswarm — Midpoint Rebalance (2021-10-29)</h3>
      {bee_img}
      <h3 style="margin-top:2mm;">Waterfall — Highest-Predicted Stock</h3>
      {wf_img}
    </div>
  </div>
</div>
"""


def _page_regime(regime_df: pd.DataFrame) -> str:
    """Return the HTML string for Page 4 (macro regime performance table and equity curves figure)."""
    # Regime counts
    if not regime_df.empty:
        count_rows = ""
        for _, row in regime_df.iterrows():
            sr = row.get("stress_regime", "")
            rr = row.get("rate_regime", "")
            n  = int(row.get("n_rebalances", 0))
            ic = float(row.get("ic_mean", float("nan")))
            sh = float(row.get("sharpe", float("nan")))
            top5 = row.get("shap_stability_top5", float("nan"))
            mom  = float("nan")  # not in this CSV, skip
            count_rows += f"""<tr>
              <td>{rr}</td><td>{sr}</td><td>{n}</td>
              <td>{_fmt(ic, '+.3f')}</td>
              <td>{_fmt(sh, '+.2f')}</td>
              <td>{_fmt(top5 if pd.notna(top5) else float('nan'), '.3f', color=False)}</td>
            </tr>"""
    else:
        count_rows = '<tr><td colspan="6" class="missing">Regime performance data not found.</td></tr>'

    eq_img = _img(FIGURES / "step3_regime_equity_curves.png",
                  "width:100%;border:1px solid #30363d;border-radius:4px;")
    ic_img = _img(FIGURES / "step3_ic_by_regime.png",
                  "width:100%;border:1px solid #30363d;border-radius:4px;")

    return f"""
<div class="page">
  <div class="section-header"><h2>03 · Performance by Macro Regime</h2></div>

  <h3>Regime Performance Table (Banxico rate × IPC vol stress)</h3>
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
    <div style="flex:3;">{eq_img}</div>
    <div style="flex:2;">{ic_img}</div>
  </div>
</div>
"""


def _page_risk_methodology(step1_rows: list[dict]) -> str:
    """Return the HTML string for Page 5 (risk metrics, factor exposures, methodology summary)."""
    # Extract XGBoost risk metrics from step1 table
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
        <li><strong>TIGHTENING:</strong> Scale signal ×0.7 (stability 0.41)</li>
        <li><strong>STRESS trigger:</strong> IPC vol &gt; p75 → reduce exposure</li>
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
        <li><strong>Regime conditioning:</strong> Banxico rate (TIGHTENING / EASING / NEUTRAL) × IPC vol stress</li>
        <li><strong>Feature attribution:</strong> Per-rebalance SHAP values via TreeExplainer</li>
        <li><strong>Data sources:</strong> Bloomberg (production) · mock data (research)</li>
        <li><strong>Regulatory framework:</strong> CNBV position limits, LFI structure</li>
      </ul>

      <h3 style="margin-top:3mm;">SHAP Feature Stability</h3>
      <ul class="method-list">
        <li><strong>Overall top-5 stability:</strong> 0.44 mean Spearman (target ≥ 0.80)</li>
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
            .replace("ρ", "r")                          # rho
            .replace("é", "e").replace("á", "a")  # accented vowels
            .encode("latin-1", errors="replace").decode("latin-1"))


def _render_pdf_fpdf2(html_path: Path, pdf_path: Path, step1_rows, top10, stability, regime_df) -> None:
    """Build a compact multi-page PDF using fpdf2 (no system library deps)."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    import io

    class TearsheetPDF(FPDF):
        def normalize_text(self, text: str) -> str:
            """Sanitise text so all chars fit within the latin-1 encoding of core fonts."""
            return _ascii(text)

        def header(self):
            """No page header — suppressed intentionally."""
            pass

        def footer(self):
            """Render the centred page-counter footer on every page."""
            self.set_y(-10)
            self.set_font("Helvetica", "I", 6)
            self.set_text_color(139, 148, 158)
            self.cell(0, 4,
                f"Page {self.page_no()} of {{nb}} | FMIA - FondoMexicoAlfa | Confidential",
                align="C",
            )

    pdf = TearsheetPDF(orientation="L", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=14)

    # Colours
    BG   = (13,  17,  23)
    SURF = (22,  27,  34)
    TXT  = (230, 237, 243)
    MUT  = (139, 148, 158)
    ACC  = (57,  211, 83)
    RED  = (248, 81,  73)

    def set_primary():   """Set text colour to primary (light)."""; pdf.set_text_color(*TXT)
    def set_muted():     """Set text colour to muted grey."""; pdf.set_text_color(*MUT)
    def set_accent():    """Set text colour to accent green."""; pdf.set_text_color(*ACC)
    def set_red():       """Set text colour to warning red."""; pdf.set_text_color(*RED)

    def section_header(title: str):
        """Render a bold section-header cell with a green left-border rule."""
        pdf.set_fill_color(*SURF)
        pdf.set_draw_color(*ACC)
        pdf.set_line_width(0.8)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x(), pdf.get_y() + 6)
        pdf.set_x(pdf.get_x() + 2)
        pdf.set_font("Helvetica", "B", 9)
        set_muted()
        pdf.cell(0, 6, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_line_width(0.1)
        pdf.ln(2)

    def add_img_safe(path: Path, x=None, y=None, w=None, h=None):
        """Insert an image at the given position, silently skipping if the file is absent."""
        if path.exists():
            try:
                if x is not None:
                    pdf.image(str(path), x=x, y=y, w=w, h=h)
                else:
                    pdf.image(str(path), w=w or 240, h=h or 0)
            except Exception:
                pass

    # ---- Page 1: Cover ----
    pdf.add_page()
    pdf.set_fill_color(*BG)
    stats = _extract_cover_stats(step1_rows)

    pdf.set_font("Helvetica", "B", 24)
    set_accent()
    pdf.cell(0, 12, "FondoMéxicoAlfa (FMIA)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    set_muted()
    pdf.cell(0, 6, "Systematic Long-Only Equity & FIBRA  |  Quantitative Research Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Stat strip
    labels = [("XGBoost Sharpe", stats.get("sharpe","—"), ACC),
              ("XGBoost ICIR",   stats.get("icir",  "—"), ACC),
              ("Max Drawdown",   stats.get("mdd",   "—"), RED)]
    col_w = (pdf.epw) / 3

    for i, (label, val, color) in enumerate(labels):
        x = pdf.l_margin + i * col_w
        pdf.set_xy(x, pdf.get_y())
        pdf.set_fill_color(*SURF)
        pdf.rect(x, pdf.get_y(), col_w - 3, 20, style="F")
        pdf.set_xy(x + 2, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(*color)
        pdf.cell(col_w - 5, 10, str(val), align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_xy(x + 2, pdf.get_y() + 12)
        pdf.set_font("Helvetica", "", 7)
        set_muted()
        pdf.cell(col_w - 5, 4, label, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_xy(pdf.l_margin, pdf.get_y() + 26)
    pdf.set_font("Helvetica", "I", 8)
    set_muted()
    pdf.cell(0, 5,
        "Walk-forward validated  ·  Bloomberg data  ·  Mexican equities & FIBRAs (BMV)  ·  Monthly rebalancing  ·  10 bp transaction cost",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 7)
    pdf.cell(0, 5, f"Report date: {date.today().isoformat()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---- Page 2: Model Comparison ----
    pdf.add_page()
    section_header("01 · Model Performance — XGBoost vs ElasticNetCV")

    # Table
    col_widths = [70, 35, 35, 35]
    headers = ["Metric", "ElasticNetCV", "XGBoost", "Δ (xgb − elastic)"]
    pdf.set_font("Helvetica", "B", 7)
    set_muted()
    pdf.set_fill_color(*SURF)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 5, h, border=0, fill=True, align="R" if h != "Metric" else "L")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for i, row in enumerate(step1_rows):
        pdf.set_fill_color(*(SURF if i % 2 else BG))
        set_primary()
        pdf.cell(col_widths[0], 4.5, row["metric"], fill=True)
        set_muted()
        for val, w in [(row["elastic"], col_widths[1]), (row["xgboost"], col_widths[2]), (row["delta"], col_widths[3])]:
            set_muted()
            pdf.cell(w, 4.5, str(val), fill=True, align="R")
        pdf.ln()

    pdf.ln(3)
    # Figures side-by-side
    y_fig = pdf.get_y()
    w_fig = pdf.epw / 2 - 2
    add_img_safe(FIGURES / "step1_equity_mock.png",
                 x=pdf.l_margin, y=y_fig, w=w_fig)
    add_img_safe(FIGURES / "step1_ic_mock.png",
                 x=pdf.l_margin + w_fig + 4, y=y_fig, w=w_fig)

    # ---- Page 3: SHAP ----
    pdf.add_page()
    section_header("02 · Feature Attribution (SHAP) — XGBoost Walk-forward")

    half = pdf.epw / 2 - 2

    # Left: tables
    y0 = pdf.get_y()
    pdf.set_font("Helvetica", "B", 8)
    set_primary()
    pdf.cell(0, 5, "Top-10 Features by Mean |SHAP|", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 7)
    set_muted()
    pdf.set_fill_color(*SURF)
    for hdr, w in [("#", 8), ("Feature", 38), ("Mean |SHAP|", 28), ("Std", 26)]:
        pdf.cell(w, 4.5, hdr, fill=True, align="C")
    pdf.ln()
    for i, r in enumerate(top10[:10]):
        pdf.set_fill_color(*(SURF if i % 2 else BG))
        pdf.set_font("Helvetica", "", 7)
        set_primary()
        pdf.cell(8,  4.5, r["rank"],    fill=True, align="C")
        pdf.cell(38, 4.5, r["feature"], fill=True)
        set_muted()
        pdf.cell(28, 4.5, r["mean_abs"], fill=True, align="R")
        pdf.cell(26, 4.5, r["std_abs"],  fill=True, align="R")
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 8)
    set_primary()
    pdf.cell(0, 5, "Feature Stability (Spearman rank-corr.)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 7)
    set_muted()
    pdf.set_fill_color(*SURF)
    for hdr, w in [("K", 16), ("Pairs", 18), ("Mean ρ", 28), ("Std ρ", 28)]:
        pdf.cell(w, 4.5, hdr, fill=True, align="C")
    pdf.ln()
    for i, r in enumerate(stability):
        pdf.set_fill_color(*(SURF if i % 2 else BG))
        pdf.set_font("Helvetica", "", 7)
        set_primary()
        pdf.cell(16, 4.5, r["K"],     fill=True)
        pdf.cell(18, 4.5, r["pairs"], fill=True, align="C")
        set_muted()
        pdf.cell(28, 4.5, r["mean"],  fill=True, align="R")
        pdf.cell(28, 4.5, r["std"],   fill=True, align="R")
        pdf.ln()

    # Right: figures
    x_right = pdf.l_margin + half + 4
    y_right  = y0
    add_img_safe(FIGURES / "step2_shap_beeswarm.png",
                 x=x_right, y=y_right, w=half, h=50)
    add_img_safe(FIGURES / "step2_shap_waterfall.png",
                 x=x_right, y=y_right + 53, w=half, h=45)

    # ---- Page 4: Regime ----
    pdf.add_page()
    section_header("03 · Performance by Macro Regime")

    if not regime_df.empty:
        pdf.set_font("Helvetica", "B", 7)
        set_muted()
        pdf.set_fill_color(*SURF)
        cols4 = [("Rate regime", 40), ("Stress", 22), ("N", 12),
                 ("IC mean", 24), ("Sharpe", 22), ("Top-5 stab.", 24)]
        for hdr, w in cols4:
            pdf.cell(w, 4.5, hdr, fill=True, align="C")
        pdf.ln()

        for i, (_, row) in enumerate(regime_df.iterrows()):
            pdf.set_fill_color(*(SURF if i % 2 else BG))
            pdf.set_font("Helvetica", "", 7)
            set_primary()
            pdf.cell(40, 4.5, str(row.get("rate_regime","")),   fill=True)
            pdf.cell(22, 4.5, str(row.get("stress_regime","")), fill=True)
            pdf.cell(12, 4.5, str(int(row.get("n_rebalances",0))), fill=True, align="C")
            ic  = row.get("ic_mean", float("nan"))
            sh  = row.get("sharpe",  float("nan"))
            top5= row.get("shap_stability_top5", float("nan"))
            set_muted()
            pdf.cell(24, 4.5, f"{ic:+.3f}" if pd.notna(ic) else "—",  fill=True, align="R")
            pdf.cell(22, 4.5, f"{sh:+.2f}" if pd.notna(sh) else "—",  fill=True, align="R")
            pdf.cell(24, 4.5, f"{top5:.3f}" if pd.notna(top5) else "—", fill=True, align="R")
            pdf.ln()

    pdf.ln(3)
    y_fig4 = pdf.get_y()
    w3 = pdf.epw * 0.58
    w4 = pdf.epw * 0.38
    add_img_safe(FIGURES / "step3_regime_equity_curves.png",
                 x=pdf.l_margin, y=y_fig4, w=w3)
    add_img_safe(FIGURES / "step3_ic_by_regime.png",
                 x=pdf.l_margin + w3 + 4, y=y_fig4, w=w4)

    # ---- Page 5: Risk & Methodology ----
    pdf.add_page()
    section_header("04 · Risk Profile & Methodology")

    lookup = {r["metric"]: r.get("xgboost", "—") for r in step1_rows}
    risk_items = [
        ("Annualized return",     lookup.get("Annualized return", "—")),
        ("Annualized vol",        lookup.get("Annualized vol", "—")),
        ("Sharpe ratio",          lookup.get("Sharpe", "—")),
        ("Sortino ratio",         lookup.get("Sortino", "—")),
        ("Max drawdown",          lookup.get("Max drawdown", "—")),
        ("CVaR 95% (daily)",      lookup.get("CVaR 95% (daily)", "—")),
        ("Avg turnover/rebalance",lookup.get("Turnover (per rebalance)", "—")),
    ]
    method_items = [
        ("Universe",       "Mexican equities + FIBRAs (BMV), ~26 instruments"),
        ("Signal",         "XGBoost cross-sectional return forecast (RandomizedSearchCV + TimeSeriesSplit)"),
        ("Baseline signal","ElasticNetCV (interchangeable via config flag)"),
        ("Optimizer",      "Mean-variance (SLSQP) with market-impact penalty"),
        ("Rebalance",      "Monthly (end-of-month), walk-forward OOS"),
        ("Validation",     "108 rebalances, 2017–2026"),
        ("Costs",          "10 bp per side applied each rebalance"),
        ("Regimes",        "Banxico rate (TIGHTENING/EASING/NEUTRAL) × IPC vol stress"),
        ("Attribution",    "Per-rebalance SHAP via TreeExplainer (shap 0.51)"),
        ("Data",           "Bloomberg (production) · mock data (research)"),
        ("Framework",      "CNBV position limits, LFI structure"),
    ]

    half5 = pdf.epw / 2 - 3
    y5 = pdf.get_y()

    # Left: Risk metrics
    pdf.set_font("Helvetica", "B", 8)
    set_primary()
    pdf.cell(half5, 5, "XGBoost Risk Metrics (Walk-forward OOS)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 7)
    set_muted()
    pdf.set_fill_color(*SURF)
    pdf.cell(half5 * 0.55, 4.5, "Metric", fill=True)
    pdf.cell(half5 * 0.45, 4.5, "Value",  fill=True, align="R")
    pdf.ln()
    for i, (label, val) in enumerate(risk_items):
        pdf.set_fill_color(*(SURF if i % 2 else BG))
        pdf.set_font("Helvetica", "", 7)
        set_primary()
        pdf.cell(half5 * 0.55, 4.5, label, fill=True)
        set_muted()
        pdf.cell(half5 * 0.45, 4.5, str(val), fill=True, align="R")
        pdf.ln()

    # Right: Methodology
    pdf.set_xy(pdf.l_margin + half5 + 6, y5)
    pdf.set_font("Helvetica", "B", 8)
    set_primary()
    pdf.cell(half5, 5, "Methodology", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_xy(pdf.l_margin + half5 + 6, y5 + 7)
    for i, (label, desc) in enumerate(method_items):
        pdf.set_xy(pdf.l_margin + half5 + 6, pdf.get_y())
        pdf.set_fill_color(*(SURF if i % 2 else BG))
        pdf.set_font("Helvetica", "B", 7)
        set_primary()
        pdf.cell(28, 4.5, label + ":", fill=True)
        pdf.set_font("Helvetica", "", 7)
        set_muted()
        pdf.cell(half5 - 28, 4.5, desc[:70], fill=True)
        pdf.ln()

    # Disclaimer
    pdf.set_y(-45)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(*MUT)
    pdf.set_fill_color(*SURF)
    pdf.multi_cell(0, 3.5,
        "DISCLAIMER: For research purposes only. Past performance does not guarantee future results. "
        "All results based on walk-forward out-of-sample simulation. Mock data used where Bloomberg "
        "data unavailable. This document is confidential and intended solely for internal research use.",
        fill=True,
    )

    pdf.output(str(pdf_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load all source data, build HTML, write it, then render PDF (WeasyPrint → fpdf2 fallback)."""
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
        print(f"  all {len(img_srcs)} image(s) embedded as data URIs ✓")

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
