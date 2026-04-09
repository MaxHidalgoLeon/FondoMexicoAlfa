#!/usr/bin/env python
"""
Example: switching data providers in the pipeline.

Usage:
    python scripts/load_data_example.py --source mock
    python scripts/load_data_example.py --source yahoo
    python scripts/load_data_example.py --source bloomberg
    python scripts/load_data_example.py --source refinitiv
"""
import argparse
from src.pipeline import run_pipeline, print_summary

PARSER = argparse.ArgumentParser(description="Run the Fondo Mexico pipeline with a chosen data source.")
PARSER.add_argument("--source", choices=["mock", "yahoo", "bloomberg", "refinitiv"], default="mock")
PARSER.add_argument("--start", default="2018-01-01")
PARSER.add_argument("--end", default="2025-12-31")
PARSER.add_argument("--hedge", action="store_true", help="Enable hedge overlay (Layer 2).")

if __name__ == "__main__":
    args = PARSER.parse_args()
    print(f"Running pipeline with data_source='{args.source}' ({args.start} → {args.end})")
    results = run_pipeline(
        hedge_mode=args.hedge,
        data_source=args.source,
        start_date=args.start,
        end_date=args.end,
    )
    print_summary(results, hedge_mode=args.hedge)
