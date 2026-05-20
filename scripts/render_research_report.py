#!/usr/bin/env python
"""Render reports/FMIA_Research_Report.md from FMIA_Research_Report.md.j2.

The template parametrizes the Abstract's headline KPIs. The renderer also
regenerates Table 1 (§5.1) in the body by loading all six available
metrics_{source}_{model}.json files (3 sources × 2 models). Sections that the
template does not own are preserved verbatim from the previous render.

KPI values come from reports/output/metrics_{source}_{model}.json, which is
emitted by every run of the pipeline (see src/pipeline._emit_metrics_json).

Usage:
    python scripts/render_research_report.py [--source bloomberg] [--model elasticnet]

    --source/--model control which JSON drives the Abstract's headline numbers.
    Table 1 always pulls from every JSON it can find.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
REPORTS  = ROOT / "reports"
TEMPLATE = REPORTS / "FMIA_Research_Report.md.j2"
OUTPUT   = REPORTS / "FMIA_Research_Report.md"


def _fmt_pct(value: float, sign: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:{'+' if sign else ''}.2f}%"


def _fmt_ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _load_metrics(source: str, model: str) -> dict:
    path = ROOT / "reports" / "output" / f"metrics_{source}_{model}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/run_all.py --source {source} "
            f"--model {model}` first so the pipeline can emit the metrics JSON."
        )
    return json.loads(path.read_text())


_TABLE1_SOURCES = ("Bloomberg", "Yahoo", "Refinitiv")
_TABLE1_MODELS = (("ElasticNetCV", "elasticnet"), ("LightGBM", "lightgbm"))


def _build_table1() -> str:
    """Regenerate Table 1 from all metrics_*.json files in reports/output.

    Returns a complete Markdown table block. Rows for which the corresponding
    JSON is missing are rendered with em-dashes so the operator can see at a
    glance which (source, model) pairs still need a run.
    """
    header = (
        "| Source | Model | Return | Vol | Sharpe | Sortino | Max DD | CVaR 95% | Turnover |\n"
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    rows = [header]
    for src_label in _TABLE1_SOURCES:
        src_key = src_label.lower()
        for model_label, model_key in _TABLE1_MODELS:
            path = ROOT / "reports" / "output" / f"metrics_{src_key}_{model_key}.json"
            if not path.exists():
                cells = ["—"] * 7
            else:
                m = (json.loads(path.read_text()).get("metrics") or {})
                cells = [
                    _fmt_pct(m.get("annualized_return")),
                    _fmt_pct(m.get("annualized_vol")),
                    _fmt_ratio(m.get("sharpe")),
                    _fmt_ratio(m.get("sortino")),
                    _fmt_pct(m.get("max_drawdown"), sign=True),
                    _fmt_pct(m.get("cvar_95"), sign=True),
                    _fmt_pct(m.get("turnover")),
                ]
            rows.append(f"| {src_label} | {model_label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


_TABLE1_RE = re.compile(
    r"\|\s*Source\s*\|\s*Model\s*\|.*?Turnover\s*\|\n"   # header
    r"\|:?-+:?\|.*?\|\n"                                  # separator row
    r"(?:\|[^\n]*\|\n){1,12}",                            # 1..12 data rows
    re.DOTALL,
)


def _splice_table1(body: str, table_md: str) -> str:
    """Replace the first Table 1 block in `body` with `table_md`.

    The regex matches the unique markdown header `| Source | Model |  ... | Turnover |`
    plus its data rows. If no match is found the original body is returned and
    a warning is printed so the caller knows the table was NOT refreshed.
    """
    new_body, count = _TABLE1_RE.subn(table_md + "\n", body, count=1)
    if count == 0:
        print("[research-report] Could not locate Table 1 header — body left unchanged.")
    return new_body


def _extract_body(previous_md: Path) -> str:
    """Pull the post-Abstract content of the previous render to preserve prose.

    Returns the substring starting at the first top-level section after the
    Abstract (section `## 1. Introduction`). Falls back to empty string when
    the file is missing — the resulting render will then contain only the
    abstract, which is a clear signal to the operator.
    """
    if not previous_md.exists():
        print(f"[research-report] {previous_md.name} not found — output will contain Abstract only.")
        return ""
    text = previous_md.read_text(encoding="utf-8")
    marker = "## 1. Introduction"
    idx = text.find(marker)
    if idx < 0:
        print(f"[research-report] '{marker}' not found in {previous_md.name} — body left empty.")
        return ""
    return text[idx:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="bloomberg")
    p.add_argument("--model", default="elasticnet")
    args = p.parse_args()

    if not TEMPLATE.exists():
        print(f"[research-report] Template not found: {TEMPLATE}", file=sys.stderr)
        return 1

    try:
        payload = _load_metrics(args.source, args.model)
    except FileNotFoundError as exc:
        print(f"[research-report] {exc}", file=sys.stderr)
        return 1

    metrics = payload.get("metrics") or {}
    sharpe   = _fmt_ratio(metrics.get("sharpe"))
    ret_bbg  = _fmt_pct(metrics.get("annualized_return"))
    vol_bbg  = _fmt_pct(metrics.get("annualized_vol"))
    mdd_bbg  = _fmt_pct(metrics.get("max_drawdown"), sign=True)

    template = TEMPLATE.read_text(encoding="utf-8")
    body = _extract_body(OUTPUT)
    body = _splice_table1(body, _build_table1())
    rendered = (
        template
        .replace("{{ sharpe_bbg }}", sharpe)
        .replace("{{ ret_bbg }}", ret_bbg)
        .replace("{{ vol_bbg }}", vol_bbg)
        .replace("{{ mdd_bbg }}", mdd_bbg)
        .replace("{{ body }}", body)
    )
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"[research-report] Wrote {OUTPUT} "
          f"(source={args.source}, model={args.model}, as_of={payload.get('as_of','?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
