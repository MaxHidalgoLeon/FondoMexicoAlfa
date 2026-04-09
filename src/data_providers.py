"""Multi-source data provider abstraction for the Mexico quant strategy."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDataProvider(ABC):
    """Abstract interface for all data providers."""

    @abstractmethod
    def get_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return adjusted close prices as a business-day DataFrame (dates × tickers)."""

    @abstractmethod
    def get_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format fundamentals: date, ticker, pe_ratio, pb_ratio, roe,
        profit_margin, net_debt_to_ebitda, ebitda_growth, capex_to_sales."""

    @abstractmethod
    def get_macro(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return monthly macro DataFrame: date, IMAI, industrial_production_yoy,
        exports_yoy, usd_mxn, banxico_rate, inflation_yoy, us_ip_yoy, us_fed_rate."""

    @abstractmethod
    def get_fibra_fundamentals(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format FIBRA metrics: date, ticker, cap_rate, ffo_yield,
        dividend_yield, ltv, vacancy_rate."""

    @abstractmethod
    def get_bonds(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return long-format bond data: date, ticker, asset_class, price, ytm,
        duration, credit_spread."""


# ---------------------------------------------------------------------------
# Mock provider — delegates to existing data_loader functions
# ---------------------------------------------------------------------------

class MockDataProvider(BaseDataProvider):
    """Routes all requests to the mock generators in data_loader."""

    def __init__(self) -> None:
        from .data_loader import (  # noqa: F401 — validated at init time
            generate_mock_price_series,
            build_mock_fundamentals,
            build_mock_fibra_fundamentals,
            build_mock_bonds,
            build_mock_macro_series,
            get_investable_universe,
        )

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import generate_mock_price_series
        return generate_mock_price_series(tickers, start_date=start_date, end_date=end_date)

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_fundamentals
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_fundamentals(tickers, dates)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_macro_series
        return build_mock_macro_series(start_date=start_date, end_date=end_date)

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        from .data_loader import build_mock_fibra_fundamentals
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_fibra_fundamentals(tickers, dates)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from .data_loader import build_mock_bonds
        dates = pd.date_range(start_date, end_date, freq="ME")
        return build_mock_bonds(dates)


# ---------------------------------------------------------------------------
# Yahoo Finance provider
# ---------------------------------------------------------------------------

class YahooFinanceProvider(BaseDataProvider):
    """Fetches data via the yfinance library (equities/FIBRAs on BMV)."""

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError("Install yfinance: pip install yfinance")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mx_tickers(tickers: List[str]) -> dict:
        """Map original tickers to BMV Yahoo Finance tickers (append .MX)."""
        return {f"{t}.MX": t for t in tickers}

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        mapping = self._to_mx_tickers(tickers)
        mx_tickers = list(mapping.keys())

        raw = yf.download(
            mx_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]

        # yfinance may return a Series when only one ticker is requested
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=mx_tickers[0])

        # Rename .MX tickers back to originals
        raw.columns = [mapping.get(str(c), str(c)) for c in raw.columns]
        raw.index = pd.DatetimeIndex(raw.index)

        # Reindex to business days and forward-fill illiquid sessions
        bdays = pd.bdate_range(start_date, end_date)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        today = pd.Timestamp.today().normalize()
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(f"{ticker}.MX").info
            except Exception:
                info = {}

            total_debt = info.get("totalDebt") or np.nan
            ebitda = info.get("ebitda") or np.nan
            capex = info.get("capitalExpenditures") or np.nan
            revenue = info.get("totalRevenue") or np.nan

            net_debt_to_ebitda = (
                total_debt / ebitda
                if (not math.isnan(total_debt) and not math.isnan(ebitda) and ebitda != 0)
                else np.nan
            )
            capex_to_sales = (
                abs(capex) / revenue
                if (not math.isnan(capex) and not math.isnan(revenue) and revenue != 0)
                else np.nan
            )

            records.append({
                "date": today,
                "ticker": ticker,
                "pe_ratio": info.get("trailingPE", np.nan),
                "pb_ratio": info.get("priceToBook", np.nan),
                "roe": info.get("returnOnEquity", np.nan),
                "profit_margin": info.get("profitMargins", np.nan),
                "net_debt_to_ebitda": net_debt_to_ebitda,
                "ebitda_growth": info.get("revenueGrowth", np.nan),
                "capex_to_sales": capex_to_sales,
            })
        return pd.DataFrame.from_records(records)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Macro data not available from Yahoo Finance. "
            "Use Bloomberg or Refinitiv, or provide macro manually."
        )

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        import yfinance as yf

        today = pd.Timestamp.today().normalize()
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(f"{ticker}.MX").info
            except Exception:
                info = {}

            records.append({
                "date": today,
                "ticker": ticker,
                "cap_rate": np.nan,
                "ffo_yield": np.nan,
                "dividend_yield": info.get("dividendYield", np.nan),
                "ltv": np.nan,
                "vacancy_rate": np.nan,
            })
        return pd.DataFrame.from_records(records)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Bond data not available from Yahoo Finance.")


# ---------------------------------------------------------------------------
# Bloomberg provider
# ---------------------------------------------------------------------------

# Bond ticker mapping: internal name → Bloomberg ticker
_BBG_BOND_TICKERS: dict = {
    "CETES28": "CETES 28D Govt",
    "CETES91": "CETES 91D Govt",
    "MBONO3Y": "MBONO 3Y Govt",
    "MBONO5Y": "MBONO 5Y Govt",
    "MBONO10Y": "MBONO 10Y Govt",
    "CORP1": "MBCORP1 Corp",
    "CORP2": "MBCORP2 Corp",
}

_BBG_MACRO_TICKERS: dict = {
    "MMACTIVI Index": "IMAI",
    "MXIPYOY Index": "industrial_production_yoy",
    "MXEXPORT Index": "exports_yoy",
    "USDMXN Curncy": "usd_mxn",
    "MXONBRAN Index": "banxico_rate",
    "MXCPYOY Index": "inflation_yoy",
    "IP YOY Index": "us_ip_yoy",
    "FDTR Index": "us_fed_rate",
}

_RATE_COLUMNS = {"banxico_rate", "inflation_yoy", "us_ip_yoy", "us_fed_rate", "industrial_production_yoy", "exports_yoy"}


class BloombergProvider(BaseDataProvider):
    """Fetches data via xbbg (blpapi wrapper)."""

    def __init__(self) -> None:
        try:
            from xbbg import blp  # noqa: F401
        except ImportError:
            raise ImportError("Install xbbg and blpapi: pip install xbbg")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _equity_bbg(tickers: List[str]) -> dict:
        """Map original tickers to Bloomberg equity tickers (append ' MM Equity')."""
        return {f"{t} MM Equity": t for t in tickers}

    @staticmethod
    def _collapse_multiindex(df: pd.DataFrame, reverse_map: dict) -> pd.DataFrame:
        """Flatten a MultiIndex column DataFrame produced by blp.bdh."""
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df.columns = [reverse_map.get(str(c), str(c)) for c in df.columns]
        return df

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    @staticmethod
    def _maybe_divide_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Divide a rate column by 100 when Bloomberg returns it as a percentage."""
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0 and series.abs().median() > 1.0:
                df[col] = df[col] / 100.0
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        bbg_tickers = list(mapping.keys())

        raw = blp.bdh(bbg_tickers, "PX_LAST", start_date, end_date)
        raw = self._collapse_multiindex(raw, {v: v for v in mapping.values()})

        bdays = pd.bdate_range(start_date, end_date)
        raw.index = pd.DatetimeIndex(raw.index)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        fields = [
            "PE_RATIO", "PX_TO_BOOK_RATIO", "RETURN_ON_EQUITY",
            "PROF_MARGIN", "NET_DEBT_TO_EBITDA", "EBITDA_GROWTH", "CAPEX_TO_SALES",
        ]
        field_rename = {
            "PE_RATIO": "pe_ratio",
            "PX_TO_BOOK_RATIO": "pb_ratio",
            "RETURN_ON_EQUITY": "roe",
            "PROF_MARGIN": "profit_margin",
            "NET_DEBT_TO_EBITDA": "net_debt_to_ebitda",
            "EBITDA_GROWTH": "ebitda_growth",
            "CAPEX_TO_SALES": "capex_to_sales",
        }

        records = []
        for bbg_ticker, orig_ticker in mapping.items():
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = orig_ticker
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        bbg_tickers = list(_BBG_MACRO_TICKERS.keys())
        raw = blp.bdh(bbg_tickers, "PX_LAST", start_date, end_date, Per="M")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns=_BBG_MACRO_TICKERS)
        raw.index.name = "date"
        raw = raw.reset_index()

        for col in _RATE_COLUMNS:
            raw = self._maybe_divide_rates(raw, col)

        return raw

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        from xbbg import blp

        mapping = self._equity_bbg(tickers)
        fields = ["CAP_RATE", "FFO_YIELD", "DVD_SH_12M", "LOAN_TO_VALUE", "VACANCY_RATE"]
        field_rename = {
            "CAP_RATE": "cap_rate",
            "FFO_YIELD": "ffo_yield",
            "DVD_SH_12M": "dividend_yield",
            "LOAN_TO_VALUE": "ltv",
            "VACANCY_RATE": "vacancy_rate",
        }

        records = []
        for bbg_ticker, orig_ticker in mapping.items():
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = orig_ticker
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        from xbbg import blp

        fields = ["PX_LAST", "YLD_YTM_MID", "DUR_MID", "Z_SPRD_MID"]
        field_rename = {
            "PX_LAST": "price",
            "YLD_YTM_MID": "ytm",
            "DUR_MID": "duration",
            "Z_SPRD_MID": "credit_spread",
        }

        records = []
        for ticker in tickers:
            bbg_ticker = _BBG_BOND_TICKERS.get(ticker)
            if bbg_ticker is None:
                continue
            raw = blp.bdh(bbg_ticker, fields, start_date, end_date, Per="M")
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(1)
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = ticker
            raw["asset_class"] = "fixed_income"
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
        return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Refinitiv / LSEG provider
# ---------------------------------------------------------------------------

_RD_BOND_RICS: dict = {
    "CETES28": "MX0001Y=",
    "CETES91": "MX0003M=",
    "MBONO3Y": "MX3YT=RR",
    "MBONO5Y": "MX5YT=RR",
    "MBONO10Y": "MX10YT=RR",
}

_RD_MACRO_RICS: dict = {
    "MXIMAI=ECI": "IMAI",
    "MXIP=ECI": "industrial_production_yoy",
    "MXEX=ECI": "exports_yoy",
    "MXN=": "usd_mxn",
    "MXCBRATE=ECI": "banxico_rate",
    "MXCPI=ECI": "inflation_yoy",
    "USIP=ECI": "us_ip_yoy",
    "USFEDFS=ECI": "us_fed_rate",
}


class RefinitivProvider(BaseDataProvider):
    """Fetches data via the LSEG Data Library (refinitiv-data)."""

    _session_opened: bool = False

    def __init__(self) -> None:
        try:
            import refinitiv.data as rd  # noqa: F401
        except ImportError:
            raise ImportError("Install LSEG Data Library: pip install refinitiv-data")

    def _ensure_session(self) -> None:
        import refinitiv.data as rd
        if not RefinitivProvider._session_opened:
            rd.open_session()
            RefinitivProvider._session_opened = True

    @staticmethod
    def _to_rics(tickers: List[str]) -> dict:
        """Map original tickers to Refinitiv RICs (append .MX)."""
        return {f"{t}.MX": t for t in tickers}

    @staticmethod
    def _forward_fill_prices(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
        return df.ffill(limit=limit)

    @staticmethod
    def _maybe_divide_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0 and series.abs().median() > 1.0:
                df[col] = df[col] / 100.0
        return df

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import refinitiv.data as rd

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())

        raw = rd.get_history(universe=rics, fields=["CLOSE"], start=start_date, end=end_date)

        # Pivot to wide format if necessary
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index().pivot(index="Date", columns="Instrument", values="CLOSE")
        elif "Instrument" in raw.columns:
            raw = raw.reset_index().pivot(index="Date", columns="Instrument", values="CLOSE")
        else:
            raw.index.name = "Date"

        raw.index = pd.DatetimeIndex(raw.index)
        raw.columns = [mapping.get(str(c), str(c)) for c in raw.columns]

        bdays = pd.bdate_range(start_date, end_date)
        raw = raw.reindex(bdays).pipe(self._forward_fill_prices)
        return raw

    def get_fundamentals(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import refinitiv.data as rd

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())
        fields = [
            "TR.PE", "TR.PriceToBVPerShare", "TR.ROE", "TR.NetProfitMarginPct",
            "TR.TotalDebtToEBITDA", "TR.RevenueGrowth", "TR.CapexToRevenue",
        ]
        field_rename = {
            "TR.PE": "pe_ratio",
            "TR.PriceToBVPerShare": "pb_ratio",
            "TR.ROE": "roe",
            "TR.NetProfitMarginPct": "profit_margin",
            "TR.TotalDebtToEBITDA": "net_debt_to_ebitda",
            "TR.RevenueGrowth": "ebitda_growth",
            "TR.CapexToRevenue": "capex_to_sales",
        }

        raw = rd.get_history(
            universe=rics,
            fields=fields,
            start=start_date,
            end=end_date,
            interval="quarterly",
        )

        records = []
        for ric, orig_ticker in mapping.items():
            if ric not in raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns:
                continue
            try:
                ticker_df = raw[ric].copy() if isinstance(raw.columns, pd.MultiIndex) else raw
            except (KeyError, TypeError):
                continue

            ticker_df = ticker_df.rename(columns=field_rename)
            # Forward-fill quarterly to monthly cadence
            monthly_idx = pd.date_range(start_date, end_date, freq="ME")
            ticker_df.index = pd.DatetimeIndex(ticker_df.index)
            ticker_df = ticker_df.reindex(ticker_df.index.union(monthly_idx)).ffill().reindex(monthly_idx)
            ticker_df["ticker"] = orig_ticker
            ticker_df.index.name = "date"
            ticker_df = ticker_df.reset_index()
            records.append(ticker_df)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        import refinitiv.data as rd

        self._ensure_session()
        rics = list(_RD_MACRO_RICS.keys())

        raw = rd.get_history(universe=rics, fields=["CLOSE"], start=start_date, end=end_date)

        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index().pivot(index="Date", columns="Instrument", values="CLOSE")
        raw.index = pd.DatetimeIndex(raw.index)
        raw = raw.rename(columns=_RD_MACRO_RICS)

        # MXN= is MXN per 1 USD; take reciprocal to get USD/MXN
        if "usd_mxn" in raw.columns:
            raw["usd_mxn"] = 1.0 / raw["usd_mxn"].replace(0, np.nan)

        for col in _RATE_COLUMNS:
            raw = self._maybe_divide_rates(raw, col)

        raw.index.name = "date"
        raw = raw.reset_index()
        return raw

    def get_fibra_fundamentals(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        import refinitiv.data as rd

        self._ensure_session()
        mapping = self._to_rics(tickers)
        rics = list(mapping.keys())
        fields = [
            "TR.CapRate", "TR.FFOYield", "TR.DividendYield",
            "TR.LoanToValue", "TR.VacancyRatePercent",
        ]
        field_rename = {
            "TR.CapRate": "cap_rate",
            "TR.FFOYield": "ffo_yield",
            "TR.DividendYield": "dividend_yield",
            "TR.LoanToValue": "ltv",
            "TR.VacancyRatePercent": "vacancy_rate",
        }

        raw = rd.get_history(
            universe=rics,
            fields=fields,
            start=start_date,
            end=end_date,
            interval="quarterly",
        )

        records = []
        for ric, orig_ticker in mapping.items():
            try:
                ticker_df = raw[ric].copy() if isinstance(raw.columns, pd.MultiIndex) else raw
            except (KeyError, TypeError):
                continue

            ticker_df = ticker_df.rename(columns=field_rename)
            monthly_idx = pd.date_range(start_date, end_date, freq="ME")
            ticker_df.index = pd.DatetimeIndex(ticker_df.index)
            ticker_df = ticker_df.reindex(ticker_df.index.union(monthly_idx)).ffill().reindex(monthly_idx)
            ticker_df["ticker"] = orig_ticker
            ticker_df.index.name = "date"
            ticker_df = ticker_df.reset_index()
            records.append(ticker_df)

        if not records:
            return pd.DataFrame(columns=["date", "ticker"] + list(field_rename.values()))
        return pd.concat(records, ignore_index=True)

    def get_bonds(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        import refinitiv.data as rd

        self._ensure_session()
        fields = ["YIELD", "MDURATION", "ZSPREAD", "PRICE"]
        field_rename = {
            "YIELD": "ytm",
            "MDURATION": "duration",
            "ZSPREAD": "credit_spread",
            "PRICE": "price",
        }

        records = []
        for ticker in tickers:
            ric = _RD_BOND_RICS.get(ticker)
            if ric is None:
                continue
            raw = rd.get_history(universe=[ric], fields=fields, start=start_date, end=end_date)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw[ric]
            raw = raw.rename(columns=field_rename)
            raw["ticker"] = ticker
            raw["asset_class"] = "fixed_income"
            raw.index.name = "date"
            raw = raw.reset_index()
            records.append(raw)

        if not records:
            return pd.DataFrame(columns=["date", "ticker", "asset_class", "price", "ytm", "duration", "credit_spread"])
        return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_provider(source: str, **kwargs) -> BaseDataProvider:
    """
    Factory function to get the appropriate data provider.

    Args:
        source: One of "mock", "yahoo", "bloomberg", "refinitiv"
        **kwargs: Provider-specific keyword arguments (e.g., bloomberg_host, refinitiv_app_key)

    Returns:
        An instantiated BaseDataProvider subclass.

    Raises:
        ValueError: If source is not recognized.
        ImportError: If required library for the provider is not installed.
    """
    source = source.lower().strip()
    if source == "mock":
        return MockDataProvider()
    elif source in ("yahoo", "yfinance"):
        return YahooFinanceProvider()
    elif source in ("bloomberg", "bbg"):
        return BloombergProvider()
    elif source in ("refinitiv", "lseg", "eikon"):
        return RefinitivProvider()
    else:
        raise ValueError(
            f"Unknown data source: '{source}'. "
            f"Valid options: 'mock', 'yahoo', 'bloomberg', 'refinitiv'"
        )
