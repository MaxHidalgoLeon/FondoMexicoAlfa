"""Macro regime classifiers for FMIA walk-forward analysis.

Two orthogonal regime axes:
  1. Rate regime  — TIGHTENING | EASING | NEUTRAL
     Based on Banxico overnight rate trailing 3-month change.
  2. Stress regime — STRESS | CALM
     Based on IPC 60-day realised volatility vs a percentile threshold.

No-lookahead guarantee: regime at rebalance date `t` is assigned using
only macro/price data strictly before `t` (as of end of month `t-1`).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime label constants
TIGHTENING = "TIGHTENING"
EASING     = "EASING"
NEUTRAL    = "NEUTRAL"
STRESS     = "STRESS"
CALM       = "CALM"

_RATE_LOOKBACK_MONTHS = 3
_IPC_VOL_WINDOW       = 60   # calendar days of daily prices
_STRESS_PERCENTILE    = 75


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_macro_before(macro_df: pd.DataFrame, date) -> pd.DataFrame:
    """Return all macro rows with a ``date`` column value strictly before ``date``."""
    cutoff = pd.Timestamp(date)
    col = "date" if "date" in macro_df.columns else macro_df.index.name
    if col and col in macro_df.columns:
        return macro_df[pd.to_datetime(macro_df[col]) < cutoff]
    return macro_df[macro_df.index < cutoff]


def _banxico_trailing_change(
    macro_df: pd.DataFrame,
    date,
    lookback_months: int = _RATE_LOOKBACK_MONTHS,
) -> float | None:
    """Compute the Banxico overnight rate change over the trailing period.

    Args:
        macro_df: Monthly macro DataFrame with ``date`` and ``banxico_rate``
            columns.
        date: Reference date; only rows strictly before this date are used.
        lookback_months: Number of months to look back (default 3).

    Returns:
        Rate change in percentage points, or ``None`` if insufficient history
        or the ``banxico_rate`` column is absent.
    """
    before = _latest_macro_before(macro_df, date).copy()
    if before.empty:
        return None

    if "banxico_rate" not in before.columns:
        logger.warning("banxico_rate column missing from macro_df — rate regime unavailable.")
        return None

    before = before.sort_values("date")
    rate_series = before.set_index("date")["banxico_rate"].dropna()
    if len(rate_series) < lookback_months + 1:
        return None

    latest = rate_series.iloc[-1]
    lag    = rate_series.iloc[-(lookback_months + 1)]
    return float(latest - lag)


def _ipc_equity_returns(prices_df: pd.DataFrame, equity_tickers: list[str]) -> pd.Series:
    """Compute the equal-weighted daily log return of the equity sub-universe.

    Used as an IPC (Índice de Precios y Cotizaciones) proxy because no IPC
    index ticker is present in the prices panel.

    Args:
        prices_df: Wide daily price DataFrame; ``index`` is date, columns are
            ticker symbols.
        equity_tickers: Subset of ticker columns to include in the
            equal-weighted average. Tickers absent from ``prices_df`` are
            silently dropped.

    Returns:
        Series indexed by date containing daily log returns of the IPC proxy.
        Returns an empty Series if none of ``equity_tickers`` are present.
    """
    cols = [c for c in equity_tickers if c in prices_df.columns]
    if not cols:
        logger.warning("No equity tickers found in prices_df for IPC proxy.")
        return pd.Series(dtype=float)

    px = prices_df[cols].dropna(how="all")
    ew_price = px.mean(axis=1)          # equal-weighted level
    log_ret   = np.log(ew_price / ew_price.shift(1))
    return log_ret.dropna()


def _ipc_vol_60d(ipc_returns: pd.Series, date) -> float | None:
    """Compute the 60-day realised annualised volatility of the IPC proxy.

    Args:
        ipc_returns: Daily log return series of the IPC proxy (from
            ``_ipc_equity_returns``).
        date: Reference date; only returns strictly before this date are used.

    Returns:
        Annualised volatility (``std * sqrt(252)``), or ``None`` if fewer than
        20 daily observations are available in the 60-day window.
    """
    cutoff = pd.Timestamp(date)
    before = ipc_returns[ipc_returns.index < cutoff]
    window = before.iloc[-_IPC_VOL_WINDOW:]
    if len(window) < 20:         # need at least 20 days
        return None
    return float(window.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_rate_regime(macro_df: pd.DataFrame, date) -> str:
    """Classify the Banxico rate regime at a rebalance date.

    Uses only macro data strictly before ``date`` (no lookahead). Falls back
    to ``NEUTRAL`` when insufficient Banxico rate history is available.

    Args:
        macro_df: Monthly macro DataFrame with ``date`` and ``banxico_rate``
            columns.
        date: Rebalance date to classify.

    Returns:
        One of ``TIGHTENING``, ``EASING``, or ``NEUTRAL``.
    """
    delta = _banxico_trailing_change(macro_df, date)
    if delta is None:
        logger.debug("Insufficient Banxico history at %s — defaulting to NEUTRAL.", date)
        return NEUTRAL
    if delta > 0:
        return TIGHTENING
    if delta < 0:
        return EASING
    return NEUTRAL


def assign_stress_regime(
    ipc_returns: pd.Series,
    date,
    vol_threshold: float,
) -> str:
    """Classify the market stress regime at a rebalance date.

    Uses IPC proxy 60-day realised volatility computed from data strictly
    before ``date``. Falls back to ``CALM`` when insufficient price history
    is available.

    Args:
        ipc_returns: Daily log return series of the IPC proxy (from
            ``_ipc_equity_returns``).
        date: Rebalance date to classify.
        vol_threshold: Annualised vol threshold separating STRESS from CALM.
            Derived from the full OOS window via ``compute_stress_threshold``
            — not computed here to avoid per-call overhead.

    Returns:
        ``STRESS`` if 60-day IPC vol exceeds ``vol_threshold``, else ``CALM``.
    """
    vol = _ipc_vol_60d(ipc_returns, date)
    if vol is None:
        return CALM
    return STRESS if vol > vol_threshold else CALM


def compute_stress_threshold(
    rebalance_dates: Sequence,
    ipc_returns: pd.Series,
    percentile: float = _STRESS_PERCENTILE,
) -> float:
    """Compute the IPC vol threshold from the full out-of-sample window.

    Intended to be called once before ``build_regime_table``. The threshold
    is a research-only descriptor of the volatility distribution — it does
    NOT introduce per-period lookahead because the cut point is fixed at the
    start of analysis and not updated as periods roll forward.

    Args:
        rebalance_dates: Ordered sequence of rebalance dates spanning the
            full OOS window.
        ipc_returns: Daily log return series of the IPC proxy.
        percentile: Percentile of the vol distribution to use as the STRESS
            threshold (default 75).

    Returns:
        Annualised vol level at the ``percentile``-th percentile across all
        rebalance dates. Falls back to ``0.20`` if no observations are
        available.
    """
    vols = []
    for d in rebalance_dates:
        v = _ipc_vol_60d(ipc_returns, d)
        if v is not None:
            vols.append(v)
    if not vols:
        logger.warning("No IPC vol observations — using vol_threshold=0.20.")
        return 0.20
    return float(np.percentile(vols, percentile))


def build_regime_table(
    rebalance_dates: Sequence,
    macro_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    equity_tickers: list[str],
    vol_threshold: float | None = None,
) -> pd.DataFrame:
    """Build a per-rebalance regime classification table.

    For each rebalance date assigns a rate regime (``TIGHTENING`` /
    ``EASING`` / ``NEUTRAL``) and a stress regime (``STRESS`` / ``CALM``),
    along with diagnostic columns for the underlying raw signals.

    Args:
        rebalance_dates: Ordered sequence of rebalance dates spanning the
            analysis window.
        macro_df: Monthly macro DataFrame with ``date`` and ``banxico_rate``
            columns.
        prices_df: Wide daily price DataFrame; ``index`` is date, columns are
            ticker symbols.
        equity_tickers: Equity ticker subset used to construct the IPC proxy.
        vol_threshold: Annualised vol threshold for the STRESS regime.
            If ``None``, computed automatically at the 75th percentile of
            the OOS window via ``compute_stress_threshold``.

    Returns:
        DataFrame indexed by date with columns:
        ``rate_regime``, ``stress_regime``, ``rate_change_3m``,
        ``banxico_rate_level``, ``ipc_vol_60d``.
    """
    dates = sorted(pd.Timestamp(d) for d in rebalance_dates)
    ipc_ret = _ipc_equity_returns(prices_df, equity_tickers)

    if vol_threshold is None:
        vol_threshold = compute_stress_threshold(dates, ipc_ret)
        logger.info("Computed stress vol_threshold=%.4f (p75 of OOS window).", vol_threshold)

    rows = []
    for d in dates:
        rate_chg = _banxico_trailing_change(macro_df, d)
        ipc_v    = _ipc_vol_60d(ipc_ret, d)

        # Latest banxico level before d
        before_macro = _latest_macro_before(macro_df, d)
        if not before_macro.empty and "banxico_rate" in before_macro.columns:
            rate_level = float(before_macro.sort_values("date")["banxico_rate"].iloc[-1])
        else:
            rate_level = float("nan")

        rows.append({
            "date":             d,
            "rate_regime":      assign_rate_regime(macro_df, d),
            "stress_regime":    assign_stress_regime(ipc_ret, d, vol_threshold),
            "rate_change_3m":   rate_chg if rate_chg is not None else float("nan"),
            "banxico_rate_level": rate_level,
            "ipc_vol_60d":      ipc_v if ipc_v is not None else float("nan"),
        })

    return pd.DataFrame(rows).set_index("date")
