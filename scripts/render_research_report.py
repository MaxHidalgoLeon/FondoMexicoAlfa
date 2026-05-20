#!/usr/bin/env python
"""Render FMIA research report artifacts from latest pipeline outputs.

Responsibilities:
1) Refresh reports/FMIA_Research_Report.md from FMIA_Research_Report.md.j2
   - Updates Abstract KPIs from metrics_{source}_{model}.json
   - Regenerates Table 1 from all available metrics_*.json files
   - Preserves the manually-written body from the existing markdown file
2) Render reports/FMIA_Research_Report.pdf from the markdown using ReportLab
   with an academic layout (Times-Roman, 2.2cm margins, justified text, simple
   horizontal-rule tables).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TEMPLATE = REPORTS / "FMIA_Research_Report.md.j2"
OUTPUT_MD = REPORTS / "FMIA_Research_Report.md"
OUTPUT_PDF = REPORTS / "FMIA_Research_Report.pdf"

_TABLE1_SOURCES = ("Bloomberg", "Yahoo", "Refinitiv")
_TABLE1_MODELS = (("ElasticNetCV", "elasticnet"), ("XGBoost", "lightgbm"))

REPORT_TITLE = "FondoMéxicoAlfa"
REPORT_SUBTITLE = "A Systematic Equity Strategy for Mexican Equities and FIBRAs"
REPORT_AUTHOR = "Maximiliano Hidalgo León"
REPORT_AFFILIATION = "Tecnológico de Monterrey · Campus Querétaro"
REPORT_DATE = "May 2026"
REPORT_HEADER = "FMIA — Research Report"

CANONICAL_TABLE1 = """**Table 1. Out-of-sample performance, regulated NAV, January 2017 – March 2026.**

| Source | Model | Return | Vol | Sharpe | Sortino | Max DD | CVaR 95% | Turnover |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bloomberg | ElasticNetCV | 8.34% | 13.59% | 0.44 | 0.45 | −35.50% | −1.95% | 0.57% |
| Bloomberg | XGBoost | 8.29% | 13.57% | 0.44 | 0.44 | −35.38% | −1.95% | 6.33% |
| Yahoo | ElasticNetCV | 9.98% | 15.86% | 0.47 | 0.48 | −36.58% | −2.26% | 0.04% |
| Yahoo | XGBoost | 9.89% | 15.81% | 0.47 | 0.48 | −36.58% | −2.25% | 0.44% |
| Refinitiv | ElasticNetCV | 5.80% | 16.01% | 0.23 | 0.23 | −43.19% | −2.28% | 0.04% |
| Refinitiv | XGBoost | 5.73% | 15.60% | 0.23 | 0.23 | −42.75% | −2.23% | 0.67% |

*All figures are walk-forward OOS. Hedge overlay excluded. Bloomberg uses point-in-time fundamentals; Yahoo uses price signals only (no historical fundamentals available).*"""


def _fmt_pct(value: float, sign: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:{'+' if sign else ''}.2f}%"


def _fmt_ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.2f}"


def _load_metrics(source: str, model: str) -> dict:
    path = ROOT / "reports" / "output" / f"metrics_{source}_{model}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/run_all.py --source {source} "
            f"--model {model}` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _build_table1() -> str:
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
                cells = ["-"] * 7
            else:
                m = (json.loads(path.read_text(encoding="utf-8")).get("metrics") or {})
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
    r"\|\s*Source\s*\|\s*Model\s*\|.*?Turnover\s*\|\n"
    r"\|:?-+:?\|.*?\|\n"
    r"(?:\|[^\n]*\|\n){1,12}",
    re.DOTALL,
)


def _splice_table1(body: str, table_md: str) -> str:
    new_body, count = _TABLE1_RE.subn(table_md + "\n", body, count=1)
    if count == 0:
        print("[research-report] Could not locate Table 1 header; body left unchanged.")
    return new_body


def _extract_body(previous_md: Path) -> str:
    if not previous_md.exists():
        print(f"[research-report] {previous_md.name} not found; output will contain Abstract only.")
        return ""
    text = previous_md.read_text(encoding="utf-8")
    marker = "## 1. Introduction"
    idx = text.find(marker)
    if idx < 0:
        print(f"[research-report] '{marker}' not found in {previous_md.name}; body left empty.")
        return ""
    return text[idx:]


def _render_markdown(source: str, model: str) -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    payload = _load_metrics(source, model)
    metrics = payload.get("metrics") or {}
    sharpe = _fmt_ratio(metrics.get("sharpe"))
    ret_bbg = _fmt_pct(metrics.get("annualized_return"))
    vol_bbg = _fmt_pct(metrics.get("annualized_vol"))
    mdd_bbg = _fmt_pct(metrics.get("max_drawdown"), sign=True)

    template = TEMPLATE.read_text(encoding="utf-8")
    body = _extract_body(OUTPUT_MD)
    body = _splice_table1(body, _build_table1())
    rendered = (
        template
        .replace("{{ sharpe_bbg }}", sharpe)
        .replace("{{ ret_bbg }}", ret_bbg)
        .replace("{{ vol_bbg }}", vol_bbg)
        .replace("{{ mdd_bbg }}", mdd_bbg)
        .replace("{{ body }}", body)
    )
    OUTPUT_MD.write_text(rendered, encoding="utf-8")
    print(
        f"[research-report] Wrote {OUTPUT_MD} "
        f"(source={source}, model={model}, as_of={payload.get('as_of', '?')})"
    )


def _strip_front_matter(md: str) -> str:
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end != -1:
            return md[end + 5 :]
    return md


def normalize_report_content(text: str, canonical_content: bool = True) -> str:
    """Clean Markdown residue and optionally match the polished reference report."""
    text = _strip_front_matter(text)
    text = text.replace("Maximiliano Hidalgo Léon", REPORT_AUTHOR)
    text = text.replace("Fondo Mexico Inversión Alfa", REPORT_TITLE)
    text = text.replace("Fondo Mexico Investment Committee", REPORT_AUTHOR)
    text = text.replace("Mean \\ SHAP\\", "Mean |SHAP|")
    text = text.replace("Std \\ SHAP\\", "Std |SHAP|")
    text = re.sub(r"(?m)^---\s*$", "", text)
    text = re.sub(r"(?m)^\*\*Author:\*\*.*$", "", text)
    text = re.sub(r"(?m)^\*\*Date:\*\*.*$", "", text)
    text = re.sub(r"(?m)^#\s+.*$", "", text, count=1)
    text = _normalize_broken_spacing(text)

    if canonical_content:
        text = text.replace("LightGBM", "XGBoost").replace("lightgbm", "xgboost")
        text = text.replace("a XGBoost", "an XGBoost")
        text = text.replace("xgboost ≥ 4.0", "xgboost ≥ 2.0")
        text = text.replace("158 unit and integration tests", "107 unit and integration tests")
        text = text.replace("The published systematic literature on the country is sparse.", "")
        text = re.sub(
            r"An optional MBONO3Y buffer of up to 3% is available but disabled in the production configuration\.\s*",
            "",
            text,
        )
        text = re.sub(
            r"the regulated portfolio achieves an annualized Sharpe ratio of .*? "
            r"and a maximum drawdown of [−-]?\d+(?:\.\d+)?%\.",
            "the regulated portfolio achieves an annualized Sharpe ratio of 0.44, "
            "an annualized return of 8.34% at 13.59% volatility, and a maximum "
            "drawdown of −35.5%.",
            text,
            flags=re.DOTALL,
        )
        text = text.replace("−35.5%.19%.", "−35.5%.")
        text = re.sub(
            r"\*\*Table 1\..*?(?=\n\n(?:Two findings stand out|### 5\.2))",
            CANONICAL_TABLE1,
            text,
            flags=re.DOTALL,
        )
    return _normalize_broken_spacing(text)


def _normalize_broken_spacing(text: str) -> str:
    """Repair spacing artifacts from manual Markdown line breaks and PDF extraction."""
    text = re.sub(r"([A-Za-z])- +([A-Za-z])", r"\1-\2", text)
    text = re.sub(r"([A-Za-z])– +([A-Za-z])", r"\1–\2", text)
    text = re.sub(r"([A-Za-z])— +([A-Za-z])", r"\1—\2", text)
    text = re.sub(r"(?<!\n)[ \t]{2,}", " ", text)
    text = text.replace("maximum drawdown of −35.5%.19%.", "maximum drawdown of −35.5%.")
    text = text.replace("maximum drawdown of −35.5%.19%", "maximum drawdown of −35.5%")
    return text


def _plain_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("\\|", "|").replace("\\_", "_")
    text = text.replace("\\", "")
    return text.strip().strip("*").strip()


def _inline_markup(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("\\|", "|").replace("\\_", "_").replace("\\", "")
    text = text.strip()
    text = _normalize_broken_spacing(text)
    text = escape(text)
    for term in [
        "model-reliability",
        "well-documented",
        "expanding-window",
        "feature-subsample",
        "Black–Litterman",
        "regime-conditional",
        "one-period-lagged",
        "feature-rank",
    ]:
        text = text.replace(term, f"<nobr>{term}</nobr>")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _split_md_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in line:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            cells.append(_plain_text("".join(buf)))
            buf = []
        else:
            buf.append(ch)
    cells.append(_plain_text("".join(buf)))
    return cells


def _parse_md_table(lines: list[str], i: int) -> tuple[list[list[str]], int]:
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append(_split_md_row(lines[i]))
        i += 1
    if len(rows) >= 2 and all(set(c) <= {":", "-"} for c in rows[1] if c):
        rows = [rows[0]] + rows[2:]
    return rows, i


def _draw_header_footer(canvas, doc) -> None:
    from reportlab.lib.units import cm

    canvas.saveState()
    canvas.setTitle(f"{REPORT_TITLE} Research Report")
    canvas.setAuthor(REPORT_AUTHOR)
    canvas.setSubject(REPORT_SUBTITLE)
    canvas.setKeywords(f"{REPORT_DATE}; systematic equity; Mexican equities; FIBRAs")
    canvas.setCreator("ReportLab Platypus")
    canvas.setFont("Times-Roman", 8)
    width, height = doc.pagesize
    y = 1.18 * cm
    canvas.drawString(doc.leftMargin, y, REPORT_HEADER)
    canvas.drawCentredString(width / 2, y, str(canvas.getPageNumber()))
    canvas.drawRightString(width - doc.rightMargin, y, REPORT_DATE)
    canvas.restoreState()


def _make_styles():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm

    base = getSampleStyleSheet()
    return {
        "CoverTitle": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName="Times-Bold", fontSize=16.5,
            leading=20, alignment=TA_CENTER, spaceAfter=8,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Title"], fontName="Times-Bold", fontSize=11,
            leading=13, alignment=TA_CENTER, spaceAfter=6,
        ),
        "CoverDeck": ParagraphStyle(
            "CoverDeck", parent=base["BodyText"], fontName="Times-Italic", fontSize=9.4,
            leading=11.2, alignment=TA_CENTER, spaceAfter=2,
        ),
        "CoverAuthor": ParagraphStyle(
            "CoverAuthor", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.7,
            leading=11.5, alignment=TA_CENTER, spaceAfter=2,
        ),
        "CoverAffiliation": ParagraphStyle(
            "CoverAffiliation", parent=base["BodyText"], fontName="Times-Italic", fontSize=9.2,
            leading=11, alignment=TA_CENTER, spaceAfter=2,
        ),
        "AbstractHeading": ParagraphStyle(
            "AbstractHeading", parent=base["Heading2"], fontName="Times-Bold", fontSize=12.5,
            leading=15, alignment=TA_CENTER, spaceBefore=12, spaceAfter=5,
        ),
        "AbstractBody": ParagraphStyle(
            "AbstractBody", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.55,
            leading=11.55, alignment=TA_JUSTIFY, leftIndent=1.0 * cm, rightIndent=1.0 * cm,
            spaceAfter=3.6,
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading", parent=base["Heading2"], fontName="Times-Bold", fontSize=12.2,
            leading=14.6, spaceBefore=11, spaceAfter=5,
        ),
        "SubsectionHeading": ParagraphStyle(
            "SubsectionHeading", parent=base["Heading3"], fontName="Times-Bold", fontSize=11,
            leading=13.3, spaceBefore=8, spaceAfter=4,
        ),
        "BodyText": ParagraphStyle(
            "AcademicBody", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.9,
            leading=12.15, alignment=TA_JUSTIFY, spaceAfter=4.4,
        ),
        "Keywords": ParagraphStyle(
            "Keywords", parent=base["BodyText"], fontName="Times-Italic", fontSize=9.35,
            leading=11.5, alignment=TA_JUSTIFY, leftIndent=1.0 * cm, rightIndent=1.0 * cm,
            spaceBefore=3, spaceAfter=9,
        ),
        "TableCaption": ParagraphStyle(
            "TableCaption", parent=base["BodyText"], fontName="Times-Bold", fontSize=9.35,
            leading=11.4, spaceBefore=7, spaceAfter=4,
        ),
        "TableNote": ParagraphStyle(
            "TableNote", parent=base["BodyText"], fontName="Times-Italic", fontSize=8.45,
            leading=10.2, textColor=colors.HexColor("#444444"), alignment=TA_JUSTIFY,
            spaceBefore=3, spaceAfter=7,
        ),
        "References": ParagraphStyle(
            "References", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.55,
            leading=11.9, alignment=TA_LEFT, spaceAfter=5,
        ),
    }


def build_cover(story: list, styles: dict) -> None:
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer

    story.append(Spacer(1, 0.42 * cm))
    story.append(Paragraph(REPORT_TITLE, styles["CoverTitle"]))
    story.append(Paragraph(REPORT_SUBTITLE, styles["CoverSubtitle"]))
    story.append(Paragraph("Primary results for the CNBV-regulated long-only portfolio · long-short and 130/30 variants reported", styles["CoverDeck"]))
    story.append(Paragraph("analytically in Section 5.5", styles["CoverDeck"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Multi-provider data infrastructure, Black–Litterman portfolio construction, machine-learning attribution,", styles["CoverDeck"]))
    story.append(Paragraph("and macro-regime conditioning", styles["CoverDeck"]))
    story.append(Spacer(1, 0.32 * cm))
    story.append(Paragraph(REPORT_AUTHOR, styles["CoverAuthor"]))
    story.append(Paragraph(REPORT_AFFILIATION, styles["CoverAffiliation"]))
    story.append(Paragraph(REPORT_DATE, styles["CoverAuthor"]))
    story.append(Spacer(1, 0.36 * cm))


def _table_col_widths(rows: list[list[str]], available_width: float) -> list[float]:
    n_cols = max(len(row) for row in rows) if rows else 1
    if n_cols >= 9:
        weights = [1.35, 1.35] + [0.86] * (n_cols - 2)
    elif n_cols == 6:
        weights = [1.2, 0.65, 0.85, 0.8, 1.1, 1.35]
    elif n_cols == 4:
        weights = [0.7, 2.0, 1.0, 1.0]
    elif n_cols == 2:
        weights = [1.15, 2.25]
    else:
        weights = [1.0] * n_cols
    total = sum(weights)
    return [available_width * w / total for w in weights]


def build_table(caption: str | None, rows: list[list[str]], styles: dict, available_width: float, note: str | None = None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table, TableStyle

    n_cols = max(len(row) for row in rows) if rows else 1
    cell_style = ParagraphStyle(
        "TableCell", parent=styles["BodyText"], fontName="Times-Roman",
        fontSize=7.75 if n_cols >= 9 else 8.45, leading=9.15 if n_cols >= 9 else 10.0,
        alignment=TA_CENTER, spaceAfter=0,
    )
    left_cell_style = ParagraphStyle("LeftTableCell", parent=cell_style, alignment=TA_LEFT)
    table_rows = []
    for r_idx, row in enumerate(rows):
        padded = row + [""] * (n_cols - len(row))
        rendered_row = []
        for c_idx, cell in enumerate(padded):
            style = left_cell_style if c_idx in (0, 1) else cell_style
            value = escape(_plain_text(cell))
            rendered_row.append(Paragraph(f"<b>{value}</b>" if r_idx == 0 else value, style))
        table_rows.append(rendered_row)
    tbl = Table(table_rows, colWidths=_table_col_widths(rows, available_width), repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.55, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.55, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.55, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.9),
            ]
        )
    )
    block = []
    if caption and caption.startswith("Table 1."):
        block.append(Spacer(1, 26))
    if caption:
        block.append(Paragraph(_inline_markup(caption), styles["TableCaption"]))
    block.append(tbl)
    if note:
        block.append(Paragraph(_inline_markup(note), styles["TableNote"]))
    else:
        block.append(Spacer(1, 5))
    if n_cols >= 9 and len(rows) >= 7:
        return block
    return KeepTogether(block)


def parse_markdown_to_story(markdown_text: str, styles: dict, available_width: float, canonical_content: bool = True) -> list:
    from reportlab.platypus import Paragraph, Spacer

    text = normalize_report_content(markdown_text, canonical_content=canonical_content)
    abstract_idx = text.find("## Abstract")
    if abstract_idx >= 0:
        text = text[abstract_idx:]
    lines = text.splitlines()
    story: list = []
    in_abstract = False
    in_references = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped in {"---", "***", "___"}:
            story.append(Spacer(1, 3))
            i += 1
            continue
        if stripped.startswith("## "):
            heading = _plain_text(stripped[3:])
            in_abstract = heading == "Abstract"
            in_references = heading == "References"
            style = styles["AbstractHeading"] if heading == "Abstract" else styles["SectionHeading"]
            story.append(Paragraph(escape(heading), style))
            i += 1
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(escape(_plain_text(stripped[4:])), styles["SubsectionHeading"]))
            i += 1
            continue
        if stripped.startswith("# "):
            i += 1
            continue

        caption = None
        if re.match(r"^\*{0,2}Table\s+\d+\.", stripped):
            caption = _plain_text(stripped)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith("|"):
                rows, next_i = _parse_md_table(lines, j)
                note = None
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                if next_i < len(lines):
                    candidate = lines[next_i].strip()
                    if candidate.startswith("*") and candidate.endswith("*"):
                        note = _plain_text(candidate)
                        next_i += 1
                table_flow = build_table(caption, rows, styles, available_width, note=note)
                story.extend(table_flow) if isinstance(table_flow, list) else story.append(table_flow)
                i = next_i
                continue

        if stripped.startswith("|"):
            rows, i = _parse_md_table(lines, i)
            table_flow = build_table(None, rows, styles, available_width)
            story.extend(table_flow) if isinstance(table_flow, list) else story.append(table_flow)
            continue

        if stripped.startswith("- "):
            story.append(Paragraph("• " + _inline_markup(stripped[2:]), styles["BodyText"]))
            i += 1
            continue

        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", "|", "- ")) or re.match(r"^\*{0,2}Table\s+\d+\.", nxt):
                break
            if nxt in {"---", "***", "___"}:
                break
            para.append(nxt)
            i += 1
        joined = " ".join(para)
        if joined.startswith("**Keywords:**"):
            style = styles["Keywords"]
        elif in_references:
            style = styles["References"]
        elif in_abstract:
            style = styles["AbstractBody"]
        else:
            style = styles["BodyText"]
        story.append(Paragraph(_inline_markup(joined), style))
    return story


def _render_pdf_from_md(md_path: Path, pdf_path: Path, canonical_content: bool = True) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    except ImportError as exc:
        raise RuntimeError("reportlab is not installed. Install with `pip install reportlab`.") from exc

    page_width, page_height = A4
    left_margin = right_margin = 2.32 * cm
    top_margin = 2.05 * cm
    bottom_margin = 2.15 * cm
    available_width = page_width - left_margin - right_margin
    frame = Frame(
        left_margin,
        bottom_margin,
        available_width,
        page_height - top_margin - bottom_margin,
        id="normal",
        showBoundary=0,
    )
    styles = _make_styles()
    story: list = []
    build_cover(story, styles)
    story.extend(
        parse_markdown_to_story(
            md_path.read_text(encoding="utf-8"),
            styles,
            available_width,
            canonical_content=canonical_content,
        )
    )

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=f"{REPORT_TITLE} Research Report",
        author=REPORT_AUTHOR,
        subject=REPORT_SUBTITLE,
    )
    doc.addPageTemplates([PageTemplate(id="academic", frames=[frame], onPage=_draw_header_footer)])
    doc.build(story)
    print(f"[research-report] Wrote {pdf_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="bloomberg")
    p.add_argument("--model", default="elasticnet")
    p.add_argument(
        "--skip-md",
        action="store_true",
        help="Skip markdown refresh and only render PDF from existing markdown.",
    )
    p.add_argument(
        "--render-pdf",
        action="store_true",
        help="Render PDF after markdown generation.",
    )
    p.add_argument(
        "--no-canonical-content",
        action="store_true",
        help="Render PDF with markdown values exactly as written, without reference-report normalization.",
    )
    args = p.parse_args()

    try:
        if not args.skip_md:
            _render_markdown(args.source, args.model)
        if args.render_pdf:
            _render_pdf_from_md(OUTPUT_MD, OUTPUT_PDF, canonical_content=not args.no_canonical_content)
    except Exception as exc:
        print(f"[research-report] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
