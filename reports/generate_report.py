#!/usr/bin/env python
"""
Self-contained strategy report generator.

Usage:
    python reports/generate_report.py
    python reports/generate_report.py --source yahoo --start 2020-01-01
    python reports/generate_report.py --source mock --hedge
    python reports/generate_report.py --source bloomberg --hedge --out reports/output/my_report.html
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")

import yaml

from src.pipeline import run_pipeline
from reports.charts import build_dashboard_html


def _load_config() -> dict:
    """Load config.yaml from repo root, returning empty dict on failure."""
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "lseg", "refinitiv"]
DEFAULT_MULTI_PROVIDERS = ["yahoo", "refinitiv", "bloomberg"]
DEFAULT_BENCHMARKS = ["IPC", "GBMCRE", "GBMNEAR", "GBMMOD", "GBMALFA"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate Fondo Mexico strategy HTML report.")
    p.add_argument(
        "--source",
        default="mock",
        help=(
            "Data source(s): mock|yahoo|bloomberg|lseg|refinitiv, "
            "comma-separated list, or 'all' for yahoo,refinitiv,bloomberg."
        ),
    )
    p.add_argument("--start", default="2017-01-01")
    p.add_argument("--end", default="2026-03-31")
    p.add_argument("--hedge", action="store_true", help="Include Layer 2 hedge overlay.")
    p.add_argument("--reform", action="store_true", help="Include LFI reform scenario comparison (4 structures).")
    p.add_argument("--optimizer", choices=["mv", "cvar", "robust", "both"], default="mv",
                   help="Portfolio optimizer (default: mv). Use 'both' to compare MV vs min-CVaR.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _normalize_sources(raw_source: str) -> list[str]:
    source_text = str(raw_source).strip().lower()
    if source_text == "all":
        candidates = DEFAULT_MULTI_PROVIDERS.copy()
    else:
        candidates = [s.strip().lower() for s in source_text.split(",") if s.strip()]

    if not candidates:
        raise ValueError("No data source provided.")

    invalid = [s for s in candidates if s not in SUPPORTED_SOURCES]
    if invalid:
        valid = ", ".join(SUPPORTED_SOURCES + ["all"])
        raise ValueError(f"Invalid source(s): {', '.join(invalid)}. Valid values: {valid}")

    return list(dict.fromkeys(candidates))


def _output_path_for_source(base_out: str, source: str, multi_source: bool) -> Path:
    base = Path(base_out)
    return base.with_name(f"{base.stem}_{source}{base.suffix}")


def main():
    args = parse_args()
    sources = _normalize_sources(args.source)
    out_base = args.out or str(Path(__file__).parent / "output" / "strategy_report.html")
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    multi_source = len(sources) > 1

    config = _load_config()
    # Forward config.yaml keys that run_pipeline() reads via resolve_settings()
    pipeline_settings = {
        "etf_sector_anchor": config.get("etf_sector_anchor", {"enabled": True, "source": "bloomberg", "band": 0.15}),
        "bl_views":          config.get("bl_views",          {}),
        "bootstrap_n_reps":  config.get("bootstrap_n_reps",  5000),
        "bootstrap_enabled": config.get("bootstrap_enabled",  True),
    }

    total_sources = len(sources)
    for idx, source in enumerate(sources, start=1):
        out_path = _output_path_for_source(out_base, source, multi_source)

        print(f"[{idx}/{total_sources}] Running pipeline  source={source}  {args.start} to {args.end} ...")
        bench = DEFAULT_BENCHMARKS if source in ("yahoo", "refinitiv", "lseg", "bloomberg") else None
        results = run_pipeline(
            hedge_mode=args.hedge,
            data_source=source,
            start_date=args.start,
            end_date=args.end,
            optimizer=args.optimizer,
            benchmark_tickers=bench,
            hedge_reform=args.reform,
            settings=pipeline_settings,
        )

        print("[2/3] Building dashboard ...")
        html = build_dashboard_html(results, hedge_mode=args.hedge, data_source=source, reform=args.reform)

        print(f"[3/3] Saving -> {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\nDone. Open in browser:\n  file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
