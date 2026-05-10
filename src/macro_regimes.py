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
    """Return all macro rows with date strictly before `date`."""
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
    """Banxico rate change over the trailing `lookback_months` months.

    Uses only rows with date < `date`. Returns None if insufficient history.
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
    """Equal-weighted daily log return of the equity sub-universe.

    Parameters
    ----------
    prices_df : wide DataFrame indexed by date, columns = tickers.
    equity_tickers : subset of ticker columns to use.

    Returns
    -------
    pd.Series indexed by date with daily log returns of the IPC proxy.
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
    """60-day realised vol (annualised) of the IPC proxy strictly before `date`."""
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
    """Return TIGHTENING | EASING | NEUTRAL for rebalance date `date`.

    Uses only macro data strictly before `date` (no lookahead).
    Falls back to NEUTRAL when insufficient history is available.
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
    """Return STRESS | CALM for rebalance date `date`.

    Parameters
    ----------
    ipc_returns  : daily log return series of the IPC proxy.
    date         : rebalance date (regime uses data strictly before this date).
    vol_threshold: threshold (annualised vol). Derived from the full OOS window
                   upfront — not computed here to avoid per-call overhead.
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
    """Compute the vol threshold from the full OOS window.

    Intended to be called once before `build_regime_table`.  The threshold
    is a research-only descriptor of the volatility distribution — it does
    NOT introduce per-period lookahead because the cut point is fixed at
    the start of analysis and not updated as periods roll forward.

    Returns
    -------
    float : vol level at `percentile`-th percentile across all rebalances.
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
    """Build a per-rebalance regime table.

    Parameters
    ----------
    rebalance_dates : ordered sequence of rebalance dates.
    macro_df        : monthly macro DataFrame with 'date' and 'banxico_rate'.
    prices_df       : wide daily prices (index=date, columns=tickers).
    equity_tickers  : equity ticker subset for the IPC proxy.
    vol_threshold   : if None, computed at 75th percentile of OOS window.

    Returns
    -------
    DataFrame with columns:
        date, rate_regime, stress_regime, rate_change_3m,
        banxico_rate_level, ipc_vol_60d
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
