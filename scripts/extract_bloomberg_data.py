#!/usr/bin/env python
"""
extract_bloomberg_data.py

Runs on the PC with Bloomberg Terminal (requires xbbg + blpapi).
Reads tickers from config/ticker_map.yaml and saves data as parquet files.

Usage:
    python scripts/extract_bloomberg_data.py
    python scripts/extract_bloomberg_data.py --start 2017-01-01 --end 2026-03-31
    python scripts/extract_bloomberg_data.py --output-dir data/bloomberg

Install on the Bloomberg PC:
    pip install xbbg pandas pyarrow pyyaml
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Bonds are not in ticker_map.yaml — they have their own internal mapping
_BOND_TICKERS = {
    "CETES28":  "GCETAA28 Index",
    "CETES91":  "GCETAA91 Index",
    "MBONO3Y":  "CTMXN3Y Govt",
    "MBONO5Y":  "CTMXN5Y Govt",
    "MBONO10Y": "CTMXN10Y Govt",
}

_MACRO_TICKERS = {
    "MMACTIVI Index": "IMAI",
    "MXIPYOY Index":  "industrial_production_yoy",
    "MXEXPORT Index": "exports_yoy",
    "USDMXN Curncy":  "usd_mxn",
    "MXONBRAN Index": "banxico_rate",
    "MXCPYOY Index":  "inflation_yoy",
    "IP YOY Index":   "us_ip_yoy",
    "FDTR Index":     "us_fed_rate",
}

_RATE_COLS = {"banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate",
              "industrial_production_yoy", "exports_yoy"}

_FIBRA_PREFIXES = {"FUNO", "FIBRA", "TERRA", "FMTY"}


def _load_ticker_map(repo_root: Path, explicit_path: str | None = None) -> dict:
    if explicit_path:
        path = Path(explicit_path)
    else:
        # Search in: 1) next to script, 2) repo_root/config, 3) cwd/config
        candidates = [
            Path(__file__).parent / "ticker_map.yaml",
            repo_root / "config" / "ticker_map.yaml",
            Path.cwd() / "config" / "ticker_map.yaml",
            Path.cwd() / "ticker_map.yaml",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "ticker_map.yaml not found. Use --ticker-map to specify the path.\n"
                f"Paths searched:\n" + "\n".join(f"  {p}" for p in candidates)
            )
    with open(path) as f:
        return yaml.safe_load(f)


def _equity_tickers(ticker_map: dict) -> dict[str, str]:
    """Return {bloomberg_ticker: canonical} for tickers where bloomberg != null."""
    result = {}
    for canonical, providers in ticker_map.items():
        if not isinstance(providers, dict):
            continue
        bbg = providers.get("bloomberg")
        if not bbg:
            continue
        # Append " MM Equity" suffix if no asset class is present
        if not any(x in bbg for x in (" Equity", " Index", " Govt", " Curncy", " Corp")):
            bbg = f"{bbg} MM Equity"
        result[bbg] = canonical
    return result


def _is_fibra(canonical: str) -> bool:
    return any(canonical.startswith(p) for p in _FIBRA_PREFIXES)


def _to_pandas(raw) -> pd.DataFrame:
    """Convert to pandas if xbbg returns Polars (new PyEngine)."""
    if hasattr(raw, "to_pandas"):
        return raw.to_pandas()
    return raw


def _is_empty(raw) -> bool:
    if raw is None:
        return True
    if hasattr(raw, "is_empty"):   # polars
        return raw.is_empty()
    return raw.empty               # pandas


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a DatetimeIndex, regardless of whether PyEngine returns the date as a column or index."""
    if "date" in df.columns:
        df = df.set_index("date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df


def _collapse_multiindex(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [reverse_map.get(str(c), str(c)) for c in df.columns]
    return df


def _to_wide_format(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Convert df to wide format (DatetimeIndex × canonical ticker).
    Handles both MultiIndex columns (classic xbbg) and long format (PyEngine).
    """
    # Case 1: MultiIndex columns → classic xbbg
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
        df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
        df.columns.name = None
        return df

    # Case 2: long format — security name column with repeated dates in index
    sec_col = next(
        (c for c in df.columns if c.lower() in ("security", "ticker", "name")),
        None,
    )
    if sec_col is not None:
        # Find value column: "value" first, then any column that is not security/date/field
        val_col = next(
            (c for c in df.columns if c.lower() == "value"),
            next(
                (c for c in df.columns if c.lower() not in (sec_col.lower(), "date", "field")),
                None,
            ),
        )
        if val_col:
            df = df.pivot(columns=sec_col, values=val_col)
            df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
            df.columns.name = None
        return df

    # Case 3: already wide, no MultiIndex — just rename columns
    df.columns = [mapping.get(str(c).strip(), str(c).strip()) for c in df.columns]
    return df


def _divide_rates_if_pct(df: pd.DataFrame) -> pd.DataFrame:
    for col in _RATE_COLS:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.abs().median() > 1.0:
                df[col] = df[col] / 100.0
    return df


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_prices(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Prices: %d tickers...", len(mapping))
    raw = blp.bdh(list(mapping), "PX_LAST", start, end,
                  adjustmentNormal=True, adjustmentAbnormal=True, adjustmentSplit=True)
    raw = _to_pandas(raw)
    if _is_empty(raw):
        logger.warning("Prices: empty response")
        return pd.DataFrame()
    raw = _normalize_index(raw)
    raw = _to_wide_format(raw, mapping)
    bdays = pd.bdate_range(start, end)
    return raw.reindex(bdays).ffill(limit=5)


def extract_volume(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Volume: %d tickers...", len(mapping))
    try:
        raw = blp.bdh(list(mapping), "PX_VOLUME", start, end)
        raw = _to_pandas(raw)
        if _is_empty(raw):
            return pd.DataFrame()
        raw = _normalize_index(raw)
        raw = _to_wide_format(raw, mapping)
        raw = raw.apply(pd.to_numeric, errors="coerce")
        bdays = pd.bdate_range(start, end)
        return raw.reindex(bdays).fillna(0.0)
    except Exception as e:
        logger.warning("Volume failed: %s", e)
        return pd.DataFrame()


def _bdh_monthly(blp, ticker: str, fields: list, start: str, end: str) -> pd.DataFrame:
    """bdh with monthly periodicity — compatible with new and old PyEngine."""
    try:
        raw = blp.bdh(ticker, fields, start, end, periodicitySelection="MONTHLY")
    except Exception:
        raw = blp.bdh(ticker, fields, start, end, Per="M")
    return _to_pandas(raw)


def extract_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PE_RATIO", "PX_TO_BOOK_RATIO", "RETURN_ON_EQUITY",
              "PROF_MARGIN", "NET_DEBT_TO_EBITDA", "EBITDA_GROWTH", "CAPEX_TO_SALES"]
    rename = {"PE_RATIO": "pe_ratio", "PX_TO_BOOK_RATIO": "pb_ratio",
              "RETURN_ON_EQUITY": "roe", "PROF_MARGIN": "profit_margin",
              "NET_DEBT_TO_EBITDA": "net_debt_to_ebitda", "EBITDA_GROWTH": "ebitda_growth",
              "CAPEX_TO_SALES": "capex_to_sales"}

    equity_map = {bbg: can for bbg, can in mapping.items() if not _is_fibra(can)}
    logger.info("Fundamentals: %d tickers...", len(equity_map))
    records = []
    for bbg, canonical in equity_map.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Fundamentals %s failed: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_fibra_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["CAP_RATE", "FFO_YIELD", "DVD_SH_12M", "LOAN_TO_VALUE", "VACANCY_RATE"]
    rename = {"CAP_RATE": "cap_rate", "FFO_YIELD": "ffo_yield", "DVD_SH_12M": "dividend_yield",
              "LOAN_TO_VALUE": "ltv", "VACANCY_RATE": "vacancy_rate"}

    fibra_map = {bbg: can for bbg, can in mapping.items() if _is_fibra(can)}
    logger.info("FIBRA fundamentals: %d tickers...", len(fibra_map))
    records = []
    for bbg, canonical in fibra_map.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("FIBRA fundamentals %s failed: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_macro(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Macro: %d indicators...", len(_MACRO_TICKERS))
    raw = _bdh_monthly(blp, list(_MACRO_TICKERS), "PX_LAST", start, end)
    if _is_empty(raw):
        logger.warning("Macro: empty response")
        return pd.DataFrame()
    raw = _normalize_index(raw)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=_MACRO_TICKERS)
    raw = raw.reset_index()
    return _divide_rates_if_pct(raw)


def extract_bonds(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PX_LAST", "YLD_YTM_MID", "DUR_MID", "Z_SPRD_MID"]
    rename = {"PX_LAST": "price", "YLD_YTM_MID": "ytm",
              "DUR_MID": "duration", "Z_SPRD_MID": "credit_spread"}
    logger.info("Bonds: %d tickers...", len(_BOND_TICKERS))
    records = []
    for canonical, bbg in _BOND_TICKERS.items():
        try:
            raw = _bdh_monthly(blp, bbg, fields, start, end)
            if _is_empty(raw):
                continue
            raw = _normalize_index(raw)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            raw["asset_class"] = "fixed_income"
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Bond %s failed: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
    return pd.concat(records, ignore_index=True)


def extract_market_caps(mapping: dict[str, str]) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Market caps: %d tickers...", len(mapping))
    try:
        raw = blp.bdp(tickers=list(mapping), flds=["CUR_MKT_CAP"])
        raw = _to_pandas(raw)
        if _is_empty(raw):
            return pd.DataFrame(columns=["ticker", "market_cap_mxn"])
        # PyEngine returns columns in lowercase
        # PyEngine may return security as a column or as the index
        for sec_col in ("security", "Security", "ticker", "Ticker"):
            if sec_col in raw.columns:
                raw = raw.set_index(sec_col)
                break
        # Take the first numeric column (we only requested CUR_MKT_CAP)
        num_cols = raw.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            logger.warning("Market caps: no numeric column found. Columns: %s", list(raw.columns))
            return pd.DataFrame(columns=["ticker", "market_cap_mxn"])
        cap_col = num_cols[0]
        # Normalize whitespace in the index
        raw.index = pd.Index([re.sub(r"\s+", " ", str(s)).strip() for s in raw.index])
        records = []
        for bbg, canonical in mapping.items():
            bbg_key = re.sub(r"\s+", " ", bbg).strip()
            if bbg_key in raw.index:
                val = raw.loc[bbg_key, cap_col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.notna(val):
                    records.append({"ticker": canonical, "market_cap_mxn": float(val)})
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("Market caps failed: %s", e)
        return pd.DataFrame(columns=["ticker", "market_cap_mxn"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract data from Bloomberg Terminal and save as parquet files.")
    parser.add_argument("--output-dir", default="data/bloomberg", help="Output directory (default: data/bloomberg)")
    parser.add_argument("--start", default="2017-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--ticker-map", default=None, help="Path to ticker_map.yaml (optional, auto-detected)")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Bloomberg extraction  %s → %s ===", args.start, args.end)
    logger.info("Output: %s", output_dir.resolve())

    ticker_map = _load_ticker_map(repo_root, explicit_path=args.ticker_map)
    mapping    = _equity_tickers(ticker_map)
    logger.info("Bloomberg tickers found in ticker_map.yaml: %d", len(mapping))

    steps = [
        ("prices.parquet",            lambda: extract_prices(mapping, args.start, args.end)),
        ("volume.parquet",            lambda: extract_volume(mapping, args.start, args.end)),
        ("fundamentals.parquet",      lambda: extract_fundamentals(mapping, args.start, args.end)),
        ("fibra_fundamentals.parquet",lambda: extract_fibra_fundamentals(mapping, args.start, args.end)),
        ("macro.parquet",             lambda: extract_macro(args.start, args.end)),
        ("bonds.parquet",             lambda: extract_bonds(args.start, args.end)),
        ("market_caps.parquet",       lambda: extract_market_caps(mapping)),
    ]

    for filename, fn in steps:
        logger.info("--- %s ---", filename)
        try:
            df = fn()
            if df is None or df.empty:
                logger.warning("%s: no data, file not saved", filename)
                continue
            df.to_parquet(output_dir / filename)
            logger.info("%s saved  (%d rows)", filename, len(df))
        except Exception as e:
            logger.error("%s FAILED: %s", filename, e)

    logger.info("=== Extraction complete. Copy the '%s' folder to your laptop. ===", output_dir)


if __name__ == "__main__":
    main()
