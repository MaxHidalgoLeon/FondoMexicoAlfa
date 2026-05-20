#!/usr/bin/env python
"""
fetch_public_macro.py

Downloads public macro series using FREDBanxicoMacroProvider (FRED + Banxico SIE + INEGI)
and saves them to data/bloomberg/macro_public.parquet.

This file is used as a fallback when Bloomberg does not return macro columns
(banxico_rate, inflation_yoy, us_ip_yoy, us_fed_rate).

Usage
-----
    python scripts/fetch_public_macro.py
    python scripts/fetch_public_macro.py --start 2010-01-01 --end 2026-03-31
    python scripts/fetch_public_macro.py --output data/bloomberg/macro_public.parquet

Environment
-----------
    BANXICO_TOKEN  — free token from https://www.banxico.org.mx/SieAPIRest/service/v1/token
                     Required for banxico_rate and inflation_yoy.
                     us_fed_rate and us_ip_yoy come from FRED (no token needed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_TARGET_COLS = [
    "banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate",
    "IMAI", "industrial_production_yoy", "exports_yoy",
]


def fetch_all(start: str, end: str) -> pd.DataFrame:
    from src.data_providers import FREDBanxicoMacroProvider

    provider = FREDBanxicoMacroProvider()
    logger.info("Fetching macro from FRED + Banxico SIE (start=%s, end=%s)...", start, end)
    macro = provider.get_macro(start, end)

    if macro.empty:
        logger.error("FREDBanxicoMacroProvider returned empty DataFrame.")
        return pd.DataFrame(columns=["date"] + _TARGET_COLS)

    macro["date"] = pd.to_datetime(macro["date"] if "date" in macro.columns else macro.index)

    # Build a business-day index and forward-fill monthly series to daily
    all_dates = pd.date_range(start=start, end=end, freq="B")
    macro_indexed = macro.set_index("date").sort_index()

    df = pd.DataFrame(index=all_dates)
    df.index.name = "date"

    for col in _TARGET_COLS:
        if col in macro_indexed.columns:
            df[col] = macro_indexed[col].reindex(all_dates, method="ffill")
        else:
            df[col] = float("nan")
            logger.warning("Column '%s' not returned by provider — will be NaN", col)

    df = df.reset_index()

    filled = df[_TARGET_COLS].notna().sum()
    logger.info("Non-NaN rows per column:\n%s", filled.to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public macro data (Banxico SIE + FRED)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "bloomberg" / "macro_public.parquet"),
    )
    args = parser.parse_args()

    df = fetch_all(args.start, args.end)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows × %d cols → %s", len(df), len(df.columns), out_path)


if __name__ == "__main__":
    main()
