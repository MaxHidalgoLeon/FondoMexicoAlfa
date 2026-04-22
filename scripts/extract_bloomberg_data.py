#!/usr/bin/env python
"""
extract_bloomberg_data.py

Corre en la PC con Bloomberg Terminal (requiere xbbg + blpapi).
Lee los tickers desde config/ticker_map.yaml y guarda los datos en parquet.

Uso:
    python scripts/extract_bloomberg_data.py
    python scripts/extract_bloomberg_data.py --start 2017-01-01 --end 2026-03-31
    python scripts/extract_bloomberg_data.py --output-dir data/bloomberg

Instalar en la PC Bloomberg:
    pip install xbbg pandas pyarrow pyyaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Bonds no están en ticker_map.yaml — tienen su propio mapeo interno
_BOND_TICKERS = {
    "CETES28":  "CETES 28D Govt",
    "CETES91":  "CETES 91D Govt",
    "MBONO3Y":  "MBONO 3Y Govt",
    "MBONO5Y":  "MBONO 5Y Govt",
    "MBONO10Y": "MBONO 10Y Govt",
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


def _load_ticker_map(repo_root: Path) -> dict:
    path = repo_root / "config" / "ticker_map.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _equity_tickers(ticker_map: dict) -> dict[str, str]:
    """Devuelve {bloomberg_ticker: canonical} para tickers con bloomberg != null."""
    result = {}
    for canonical, providers in ticker_map.items():
        if not isinstance(providers, dict):
            continue
        bbg = providers.get("bloomberg")
        if not bbg:
            continue
        # Agregar sufijo " MM Equity" si no tiene asset class
        if not any(x in bbg for x in (" Equity", " Index", " Govt", " Curncy", " Corp")):
            bbg = f"{bbg} MM Equity"
        result[bbg] = canonical
    return result


def _is_fibra(canonical: str) -> bool:
    return any(canonical.startswith(p) for p in _FIBRA_PREFIXES)


def _collapse_multiindex(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [reverse_map.get(str(c), str(c)) for c in df.columns]
    return df


def _divide_rates_if_pct(df: pd.DataFrame) -> pd.DataFrame:
    for col in _RATE_COLS:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0 and s.abs().median() > 1.0:
                df[col] = df[col] / 100.0
    return df


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------

def extract_prices(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Precios: %d tickers...", len(mapping))
    raw = blp.bdh(list(mapping), "PX_LAST", start, end,
                  CshAdjNormal=True, CshAdjAbnormal=True, CapChg=True)
    if raw.empty:
        logger.warning("Precios: respuesta vacía")
        return pd.DataFrame()
    raw = _collapse_multiindex(raw, {v: v for v in mapping.values()})
    raw.index = pd.DatetimeIndex(raw.index)
    bdays = pd.bdate_range(start, end)
    return raw.reindex(bdays).ffill(limit=5)


def extract_volume(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Volumen: %d tickers...", len(mapping))
    try:
        raw = blp.bdh(list(mapping), "PX_VOLUME", start, end)
        if raw.empty:
            return pd.DataFrame()
        raw = _collapse_multiindex(raw, {v: v for v in mapping.values()})
        raw.index = pd.DatetimeIndex(raw.index)
        bdays = pd.bdate_range(start, end)
        return raw.reindex(bdays).fillna(0.0)
    except Exception as e:
        logger.warning("Volumen falló: %s", e)
        return pd.DataFrame()


def extract_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PE_RATIO", "PX_TO_BOOK_RATIO", "RETURN_ON_EQUITY",
              "PROF_MARGIN", "NET_DEBT_TO_EBITDA", "EBITDA_GROWTH", "CAPEX_TO_SALES"]
    rename = {"PE_RATIO": "pe_ratio", "PX_TO_BOOK_RATIO": "pb_ratio",
              "RETURN_ON_EQUITY": "roe", "PROF_MARGIN": "profit_margin",
              "NET_DEBT_TO_EBITDA": "net_debt_to_ebitda", "EBITDA_GROWTH": "ebitda_growth",
              "CAPEX_TO_SALES": "capex_to_sales"}

    equity_map = {bbg: can for bbg, can in mapping.items() if not _is_fibra(can)}
    logger.info("Fundamentales: %d tickers...", len(equity_map))
    records = []
    for bbg, canonical in equity_map.items():
        try:
            raw = blp.bdh(bbg, fields, start, end, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            raw.index.name = "date"
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Fundamentales %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_fibra_fundamentals(mapping: dict[str, str], start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["CAP_RATE", "FFO_YIELD", "DVD_SH_12M", "LOAN_TO_VALUE", "VACANCY_RATE"]
    rename = {"CAP_RATE": "cap_rate", "FFO_YIELD": "ffo_yield", "DVD_SH_12M": "dividend_yield",
              "LOAN_TO_VALUE": "ltv", "VACANCY_RATE": "vacancy_rate"}

    fibra_map = {bbg: can for bbg, can in mapping.items() if _is_fibra(can)}
    logger.info("FIBRA fundamentales: %d tickers...", len(fibra_map))
    records = []
    for bbg, canonical in fibra_map.items():
        try:
            raw = blp.bdh(bbg, fields, start, end, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            raw.index.name = "date"
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("FIBRA fundamentales %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker"] + list(rename.values()))
    return pd.concat(records, ignore_index=True)


def extract_macro(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Macro: %d indicadores...", len(_MACRO_TICKERS))
    raw = blp.bdh(list(_MACRO_TICKERS), "PX_LAST", start, end, Per="M")
    if raw.empty:
        logger.warning("Macro: respuesta vacía")
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=_MACRO_TICKERS)
    raw.index.name = "date"
    raw = raw.reset_index()
    return _divide_rates_if_pct(raw)


def extract_bonds(start: str, end: str) -> pd.DataFrame:
    from xbbg import blp
    fields = ["PX_LAST", "YLD_YTM_MID", "DUR_MID", "Z_SPRD_MID"]
    rename = {"PX_LAST": "price", "YLD_YTM_MID": "ytm",
              "DUR_MID": "duration", "Z_SPRD_MID": "credit_spread"}
    logger.info("Bonos: %d tickers...", len(_BOND_TICKERS))
    records = []
    for canonical, bbg in _BOND_TICKERS.items():
        try:
            raw = blp.bdh(bbg, fields, start, end, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=rename)
            raw["ticker"] = canonical
            raw["asset_class"] = "fixed_income"
            raw.index.name = "date"
            records.append(raw.reset_index())
        except Exception as e:
            logger.warning("Bono %s falló: %s", bbg, e)
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
    return pd.concat(records, ignore_index=True)


def extract_market_caps(mapping: dict[str, str]) -> pd.DataFrame:
    from xbbg import blp
    logger.info("Market caps: %d tickers...", len(mapping))
    try:
        raw = blp.bdp(tickers=list(mapping), flds=["CUR_MKT_CAP"])
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["ticker", "market_cap_mxn"])
        records = []
        for bbg, canonical in mapping.items():
            if bbg in raw.index:
                val = raw.loc[bbg, "cur_mkt_cap"]
                if pd.notna(val):
                    records.append({"ticker": canonical, "market_cap_mxn": float(val)})
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning("Market caps falló: %s", e)
        return pd.DataFrame(columns=["ticker", "market_cap_mxn"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae datos de Bloomberg Terminal y los guarda como parquet.")
    parser.add_argument("--output-dir", default="data/bloomberg", help="Carpeta de salida (default: data/bloomberg)")
    parser.add_argument("--start", default="2017-01-01", help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   default="2026-12-31", help="Fecha fin YYYY-MM-DD")
    args = parser.parse_args()

    repo_root  = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Extracción Bloomberg  %s → %s ===", args.start, args.end)
    logger.info("Salida: %s", output_dir.resolve())

    ticker_map = _load_ticker_map(repo_root)
    mapping    = _equity_tickers(ticker_map)
    logger.info("Tickers Bloomberg encontrados en ticker_map.yaml: %d", len(mapping))

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
                logger.warning("%s: sin datos, archivo no guardado", filename)
                continue
            df.to_parquet(output_dir / filename)
            logger.info("%s guardado  (%d filas)", filename, len(df))
        except Exception as e:
            logger.error("%s FALLÓ: %s", filename, e)

    logger.info("=== Extracción completada. Copia la carpeta '%s' a tu laptop. ===", output_dir)


if __name__ == "__main__":
    main()
