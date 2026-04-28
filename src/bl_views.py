"""Black-Litterman view construction.

Centralizes the assembly of investor views fed into the BL posterior.
Two sources are combined:

1. Per-ticker model views from the elastic-net forecast (existing behavior).
2. Sector-level macro views derived from the macro panel
   (industrial_production_yoy, exports_yoy, banxico_rate, usd_mxn,
   us_ip_yoy, inflation_yoy), expanded to all tickers in each sector.

Macro views default to a low confidence (~0.20) so they nudge the
elastic-net signal rather than overpowering it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


SectorViews = Dict[str, float]
TickerViews = Dict[str, float]
Confidences = Dict[str, float]


_MACRO_SECTOR_RULES = {
    "industrial_production_yoy": {"Industrial": +1.0, "Logistics": +0.5, "Infrastructure": +0.5},
    "exports_yoy":               {"Industrial": +1.0, "Logistics": +0.7},
    "us_ip_yoy":                 {"Industrial": +0.8, "Logistics": +0.6},
    "banxico_rate_delta":        {"FIBRA": -1.0, "Infrastructure": -0.4},
    "usd_mxn_mom":               {},
    "inflation_yoy":             {"FIBRA": -0.4},
}


def _expanding_zscore(series: pd.Series, min_periods: int = 24) -> float:
    """Expanding-window z-score of the latest observation. Returns 0.0 when undefined."""
    s = series.dropna()
    if len(s) < min_periods:
        return 0.0
    mean = s.iloc[:-1].mean()
    std = s.iloc[:-1].std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float((s.iloc[-1] - mean) / std)


def build_elastic_net_views(forecast_df_opt: pd.DataFrame) -> Tuple[TickerViews, Confidences]:
    """Construye las vistas BL por ticker a partir del modelo ElasticNet.

    Para cada ticker, la vista = retorno esperado promedio del modelo ElasticNet.
    La confianza se escala proporcionalmente a |vista| dentro del rango [0.30, 0.70]:
      - Ticker con el mayor retorno esperado en valor absoluto → confianza 0.70.
      - Ticker con el menor → confianza 0.30.
    Esto hace que las vistas más fuertes del modelo tengan más peso en el BL.
    """
    if forecast_df_opt is None or forecast_df_opt.empty:
        return {}, {}

    views = forecast_df_opt.groupby("ticker")["expected_return"].mean().to_dict()
    abs_views = pd.Series(views).abs()
    if abs_views.empty:
        return {}, {}

    view_range = float(abs_views.max() - abs_views.min())
    if view_range > 1e-9:
        confidences = (
            0.30 + 0.40 * (abs_views - abs_views.min()) / view_range
        ).to_dict()
    else:
        confidences = {t: 0.50 for t in views}
    return views, confidences


def build_macro_views(
    macro: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: dict,
) -> Tuple[TickerViews, Confidences]:
    """Construye vistas BL sectoriales a partir de indicadores macro.

    Flujo:
      1. Calcula z-score expansivo de cada señal macro (IP, exportaciones, Banxico, FX).
      2. Convierte z-scores a magnitudes de vista vía tanh (bounded en ±max_mag ≈ 1.5%).
      3. Expande las vistas sectoriales a cada ticker del universo.
      4. Agrega un ajuste adicional por exposición USD: tickers con >40% de ingresos en USD
         se benefician cuando el MXN se deprecia (FX momentum positivo).

    Desactivado por default (use_macro=False en config). Cuando está activo, las vistas
    macro tienen confianza baja (~0.20) para nudear la señal ElasticNet sin dominarla.

    Sector-level macro views expanded to per-ticker dicts.

    For each macro signal we compute an expanding-window z-score of the latest
    observation, then convert to a sector view via the rule table. The view
    magnitude is bounded by ``bl_macro_view_max_magnitude`` (default 1.5%).

    Tickers with non-trivial USD revenue exposure also get a peso-weakening
    view based on usd_mxn 21d momentum (bullish when MXN depreciates).
    """
    if macro is None or macro.empty or universe is None or universe.empty:
        return {}, {}
    if not bool(cfg.get("bl_views", {}).get("use_macro", False)):
        return {}, {}

    confidence = float(cfg.get("bl_views", {}).get("macro_view_confidence", 0.20))
    max_mag = float(cfg.get("bl_views", {}).get("macro_view_max_magnitude", 0.015))

    sector_views: SectorViews = {}

    def _add_sector_signal(signal_name: str, z: float) -> None:
        rules = _MACRO_SECTOR_RULES.get(signal_name, {})
        magnitude = max_mag * float(np.tanh(z))
        for sector, sign in rules.items():
            sector_views[sector] = sector_views.get(sector, 0.0) + sign * magnitude

    macro_sorted = macro.sort_index() if isinstance(macro.index, pd.DatetimeIndex) else macro
    for col in ("industrial_production_yoy", "exports_yoy", "us_ip_yoy", "inflation_yoy"):
        if col in macro_sorted.columns:
            z = _expanding_zscore(macro_sorted[col])
            _add_sector_signal(col, z)

    # Banxico rate: use 21-day delta (rate hikes hurt FIBRAs)
    if "banxico_rate" in macro_sorted.columns:
        delta = macro_sorted["banxico_rate"].diff(21)
        z_delta = _expanding_zscore(delta)
        _add_sector_signal("banxico_rate_delta", z_delta)

    # USD/MXN: per-ticker effect via usd_exposure rather than sector
    usd_view: TickerViews = {}
    if "usd_mxn" in macro_sorted.columns and "usd_exposure" in universe.columns:
        fx_mom = macro_sorted["usd_mxn"].pct_change(21)
        z_fx = _expanding_zscore(fx_mom)
        fx_magnitude = max_mag * float(np.tanh(z_fx))
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            usd_exp = float(row.get("usd_exposure", 0.0) or 0.0)
            if usd_exp > 0.4:
                # Peso depreciation (positive z_fx) helps USD-revenue names
                usd_view[ticker] = (usd_exp - 0.4) / 0.6 * fx_magnitude

    # Expand sector views to tickers
    sector_map = universe.set_index("ticker")["sector"].to_dict()
    ticker_views: TickerViews = {}
    for ticker, sector in sector_map.items():
        v = sector_views.get(sector, 0.0)
        if abs(v) > 1e-9:
            ticker_views[ticker] = v

    # Combine sector + USD-exposure views (sum, then clip to ±max_mag)
    for ticker, v in usd_view.items():
        ticker_views[ticker] = ticker_views.get(ticker, 0.0) + v

    if not ticker_views:
        return {}, {}

    # Clip combined magnitude
    ticker_views = {t: float(np.clip(v, -max_mag, max_mag)) for t, v in ticker_views.items()}
    confidences = {t: confidence for t in ticker_views}
    return ticker_views, confidences


def combine_views(
    *sources: Tuple[TickerViews, Confidences],
) -> Tuple[TickerViews, Confidences]:
    """Combina múltiples fuentes de vistas BL con ponderación por confianza.

    Para cada ticker presente en al menos una fuente:
      vista_final = Σ(vista_i × conf_i) / Σ(conf_i)   (promedio ponderado)
      conf_final  = max(conf_i)                          (no acumular confianza)

    Confidence-weighted blend of multiple view sources.

    For each ticker present in any source:
        view_final = Σ(view_i * conf_i) / Σ(conf_i)
        conf_final = max(conf_i)   (avoid double-counting confidence)
    """
    all_tickers: set[str] = set()
    for views, _ in sources:
        all_tickers.update(views.keys())

    final_views: TickerViews = {}
    final_conf: Confidences = {}
    for ticker in all_tickers:
        num = 0.0
        den = 0.0
        max_conf = 0.0
        for views, confs in sources:
            if ticker in views:
                v = float(views[ticker])
                c = float(confs.get(ticker, 0.0))
                num += v * c
                den += c
                if c > max_conf:
                    max_conf = c
        if den <= 1e-12:
            continue
        final_views[ticker] = num / den
        final_conf[ticker] = max_conf
    return final_views, final_conf


def views_breakdown(
    sources: Iterable[Tuple[str, TickerViews, Confidences]],
) -> pd.DataFrame:
    """Long-format DataFrame describing each (source, ticker) view for reporting."""
    rows = []
    for source_name, views, confs in sources:
        for ticker, view in views.items():
            rows.append({
                "source": source_name,
                "target": ticker,
                "view_pct": float(view),
                "confidence": float(confs.get(ticker, 0.0)),
            })
    return pd.DataFrame(rows, columns=["source", "target", "view_pct", "confidence"])
