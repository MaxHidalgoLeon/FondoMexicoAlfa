from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional
from scipy.stats import genextreme, kstest
from sklearn.covariance import LedoitWolf

from .settings import resolve_settings

logger = logging.getLogger(__name__)


def _as_macro_frame(macro: Optional[pd.DataFrame]) -> pd.DataFrame:
    if macro is None or len(macro) == 0:
        return pd.DataFrame()
    df = macro.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _expanding_zscore(series: pd.Series, min_periods: int = 3) -> pd.Series:
    clean = pd.Series(series, dtype=float)
    mean = clean.expanding(min_periods=min_periods).mean()
    std = clean.expanding(min_periods=min_periods).std(ddof=0)
    return (clean - mean) / (std.replace(0.0, np.nan) + 1e-9)


def blend_regime_constraints(previous: dict, current: dict, alpha: float) -> dict:
    """Linearly interpolate asset-class bounds between two regimes."""
    weight = float(np.clip(alpha, 0.0, 1.0))
    blended: dict[str, dict[str, float]] = {}
    asset_classes = sorted(set(previous) | set(current))
    for asset_class in asset_classes:
        prev_bounds = previous.get(asset_class, {})
        curr_bounds = current.get(asset_class, {})
        if not prev_bounds:
            blended[asset_class] = dict(curr_bounds)
            continue
        if not curr_bounds:
            blended[asset_class] = dict(prev_bounds)
            continue
        blended[asset_class] = {
            "min": (1.0 - weight) * float(prev_bounds.get("min", 0.0)) + weight * float(curr_bounds.get("min", 0.0)),
            "max": (1.0 - weight) * float(prev_bounds.get("max", 0.0)) + weight * float(curr_bounds.get("max", 0.0)),
        }
    return blended


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Ratio de Sharpe anualizado: (μ_exceso / σ_exceso) × √252.

    Mide el retorno obtenido por unidad de riesgo total.  La tasa libre de riesgo
    se divide entre 252 para convertirla a diaria antes de restar.
    """
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / (excess.std(ddof=0) + 1e-9)


def compute_sortino(returns: pd.Series, required_return: float = 0.0) -> float:
    """Ratio de Sortino anualizado: (μ - MAR) / semidesvación_negativa × √252.

    A diferencia del Sharpe, solo penaliza la volatilidad a la baja (downside deviation),
    ignorando la dispersión de los retornos positivos.  MAR = Minimum Acceptable Return.
    """
    downside = returns[returns < required_return] - required_return  # exceso de pérdida bajo MAR
    # Lower partial moment (standard semideviation)
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 1 else 0.0
    excess_mean = returns.mean() - required_return
    return excess_mean / (downside_std + 1e-9) * np.sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """Máxima caída desde un pico previo (Max Drawdown), como fracción negativa.

    Calcula el valor acumulado (índice de riqueza), identifica el pico histórico
    en cada punto y devuelve la caída porcentual máxima desde cualquier pico.
    Típicamente negativo, p.ej. -0.25 = caída máxima del 25%.
    """
    cumulative = np.exp(returns.fillna(0.0).cumsum())
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def compute_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """VaR histórico al nivel de confianza alpha (p.ej. 95%).

    Devuelve el percentil (1-alpha) de los retornos: el retorno que solo se supera
    hacia abajo el (1-alpha)% de los días.  Típicamente un número negativo.
    """
    return np.percentile(returns.dropna(), 100 * (1 - alpha))


def compute_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """CVaR (Expected Shortfall) al nivel alpha: promedio de los retornos en la cola.

    Es el promedio de todos los retornos menores o iguales al VaR.
    Más conservador que el VaR porque considera la magnitud de las pérdidas extremas,
    no solo la frontera.
    """
    var_level = compute_var(returns, alpha)
    tail = returns[returns <= var_level]
    if len(tail) == 0:
        return float(var_level)
    if len(tail) == 1:
        return float(tail.iloc[0])
    return float(tail.mean())


def stress_test(
    portfolio_returns: pd.Series,
    scenario_shocks: dict[str, float],
    exposures: dict[str, float],
    risk_free_rate: float = 0.02,
    shock_days: int = 5,
    event_spacing_days: int = 126,
) -> pd.DataFrame:
    """Stress test determinístico: aplica shocks de escenario al historial de retornos.

    Para cada escenario (banxico_shock, peso_depreciation, us_slowdown, tmec_disruption):
      - Shock efectivo por día = shock_total × exposure / shock_days
      - Se aplica en ventanas de shock_days días repetidas cada event_spacing_days
        para mantener los escenarios comparables en las métricas agregadas.

    Devuelve un DataFrame con retorno promedio, volatilidad, Sharpe y Max Drawdown
    bajo cada escenario.  Las exposiciones (0-1) escalan el impacto del shock según
    qué tan expuesto está el portafolio a ese factor de riesgo.
    """
    scenario_results = []
    n = max(int(shock_days), 1)
    spacing = max(int(event_spacing_days), n)
    for label, shock in scenario_shocks.items():
        exposure = exposures.get(label, 0.0)
        adjusted = portfolio_returns.copy()
        # Apply repeated finite stress windows over the sample so scenarios remain
        # comparable and visible in aggregate metrics, while avoiding permanent drift.
        shock_per_day = (shock * exposure) / n
        if len(adjusted) > 0:
            for start in range(0, len(adjusted), spacing):
                end = min(start + n, len(adjusted))
                adjusted.iloc[start:end] = adjusted.iloc[start:end] + shock_per_day
        scenario_results.append(
            {
                "scenario": label,
                "mean_return": adjusted.mean(),
                "volatility": adjusted.std(ddof=0),
                "sharpe": compute_sharpe(adjusted, risk_free_rate=risk_free_rate),
                "max_drawdown": max_drawdown(adjusted),
            }
        )
    return pd.DataFrame(scenario_results)


def fit_garch(returns: pd.Series, model: str = "GJR") -> arch_model:
    """Ajusta un modelo GARCH a la serie de retornos.

    Modelo por defecto: GJR-GARCH(1,1) — captura el efecto asimétrico de leverage:
    las caídas del mercado generan más volatilidad que las subidas de igual magnitud.

    Ecuación de varianza:  σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I_{ε<0} + β·σ²_{t-1}
    donde γ (gamma) > 0 captura el efecto asimétrico (el "GJ" de GJR = Glosten-Jagannathan-Runkle).

    Si GJR no converge, cae al GARCH(1,1) estándar (sin asimetría) como fallback.
    """
    import warnings

    if model == "GARCH":
        mod = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
    elif model == "GJR":
        mod = arch_model(returns, vol="Garch", p=1, o=1, q=1, rescale=True)
    elif model == "EGARCH":
        mod = arch_model(returns, vol="EGarch", p=1, q=1, rescale=True)
    else:
        raise ValueError("Unsupported model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mod.fit(disp="off", show_warning=False, options={"maxiter": 500})
    if result.convergence_flag != 0 and model != "GARCH":
        # Fallback to simpler standard GARCH(1,1) which is easier to fit
        logger.debug("GARCH (%s) did not converge — retrying with GARCH(1,1).", model)
        mod2 = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result2 = mod2.fit(disp="off", show_warning=False, options={"maxiter": 500})
        if result2.convergence_flag == 0:
            return result2
    return result


def garch_forecast_vol(fitted_result, horizon: int = 21) -> float:
    """Proyecta la volatilidad anualizada para un horizonte de 'horizon' días hábiles.

    Extrae la varianza condicional pronosticada (h pasos adelante) del modelo GARCH,
    promedia sobre el horizonte y anualiza multiplicando por 252.
    Default horizon=21 ≈ 1 mes de trading.
    """
    forecast = fitted_result.forecast(horizon=horizon)
    horizon_var = forecast.variance.iloc[-1].values
    mean_daily_var = float(np.nanmean(horizon_var))
    vol = np.sqrt(max(mean_daily_var, 0.0) * 252)
    return vol


def rolling_garch_forecast(
    returns: pd.Series,
    horizon: int = 21,
    lookback: int = 504,
    refit_every: int = 21,
) -> pd.Series:
    """Build a rolling annualized GARCH forecast series.

    The model is re-fit every ``refit_every`` observations using a trailing
    ``lookback`` window, then the forecast is forward-filled until next refit.
    """
    r = returns.dropna()
    if len(r) < max(lookback, 100):
        return pd.Series(index=returns.index, dtype=float)

    out = pd.Series(index=r.index, dtype=float)
    step = max(int(refit_every), 1)
    lb = max(int(lookback), 100)

    for i in range(lb, len(r), step):
        train = r.iloc[i - lb:i]
        if len(train) < 100:
            continue
        try:
            # Robustify fit inputs and cap extreme forecasts vs recent realized vol.
            lo, hi = train.quantile([0.01, 0.99])
            train_fit = train.clip(lower=float(lo), upper=float(hi))
            fitted = fit_garch(train_fit, "GJR")
            fcst = garch_forecast_vol(fitted, horizon=horizon)

            recent_realized = float(train.std(ddof=0) * np.sqrt(252))
            if np.isfinite(recent_realized) and recent_realized > 1e-9:
                lower = 0.60 * recent_realized
                upper = 1.60 * recent_realized
                fcst = float(np.clip(fcst, lower, upper))

            out.loc[r.index[i]] = fcst
        except Exception:
            continue

    return out.reindex(returns.index).ffill()


def dynamic_var(returns: pd.Series, alpha: float = 0.95, method: str = "garch") -> pd.Series:
    """VaR dinámico (condicional): usa la volatilidad GARCH como escala de riesgo.

    En lugar de un percentil estático, estima VaR_t = σ_t × z_α donde:
      - σ_t es la volatilidad condicional del modelo GJR-GARCH(1,1).
      - z_α es el percentil (1-α) de los residuos estandarizados históricos.

    Esto permite que el VaR se adapte a regímenes de alta/baja volatilidad,
    siendo más conservador en crisis y menos restrictivo en períodos tranquilos.
    """
    if method == "garch":
        r = returns.dropna()
        if len(r) < 60:
            return r.reindex(returns.index).rolling(252, min_periods=20).quantile(1 - alpha)

        # Fit in percent space for numerical stability, then convert sigma back to decimal.
        fitted = fit_garch(r * 100.0, "GJR")
        sigma = pd.Series(fitted.conditional_volatility, index=r.index) / 100.0

        std_resid = pd.Series(fitted.std_resid, index=r.index).replace([np.inf, -np.inf], np.nan).dropna()
        if len(std_resid) >= 30:
            z_alpha = float(np.percentile(std_resid, (1 - alpha) * 100))
        else:
            z_alpha = float(np.percentile((r / (r.std(ddof=0) + 1e-9)).dropna(), (1 - alpha) * 100))

        # VaR remains in decimal return units (typically a small negative number).
        var = sigma * z_alpha
        var = var.reindex(returns.index).ffill()
    elif method == "empirical":
        var = returns.rolling(252).quantile(1 - alpha)
    else:
        raise ValueError("Unsupported method")
    return var


def monte_carlo_var(
    returns: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    alpha: float = 0.95,
    n_sim: int = 10000,
    horizon: int = 1,
) -> float:
    """Monte Carlo VaR.

    If *asset_returns* (T × N) and *weights* (N,) are supplied, simulates
    correlated multi-asset portfolio returns using a Ledoit-Wolf covariance
    estimated from the individual asset return history.  This captures
    cross-ticker correlation that the univariate fallback misses.

    Falls back to fitting a 1-D normal on the portfolio return series when
    per-asset data is unavailable.
    """
    rng = np.random.default_rng(42)

    if asset_returns is not None and weights is not None and not asset_returns.empty:
        # Align weights to asset_returns columns and normalise to sum = 1
        common = [c for c in asset_returns.columns if c in weights.index]
        ar = asset_returns[common].fillna(0.0)
        w = weights.reindex(common).fillna(0.0)
        w_sum = w.sum()
        if w_sum > 1e-9:
            w = w / w_sum
        else:
            # degenerate — fall through to univariate
            asset_returns = None

    if asset_returns is not None and weights is not None and not asset_returns.empty:
        means = ar.mean().values
        cov = LedoitWolf().fit(ar.values).covariance_
        # Simulate horizon-day asset returns
        sim_asset = rng.multivariate_normal(means * horizon, cov * horizon, size=n_sim)
        sim_portfolio = sim_asset @ w.values
    else:
        # Univariate fallback
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        sim_portfolio = rng.normal(mean * horizon, std * np.sqrt(horizon), size=n_sim)

    return float(np.percentile(sim_portfolio, (1 - alpha) * 100))


def gev_var(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """VaR y CVaR usando la distribución GEV (Valores Extremos Generalizados).

    Ajusta una distribución GEV a la cola izquierda de los retornos (pérdidas).
    La GEV modela mejor los extremos del mercado que la normal, capturando colas
    pesadas (fat tails) características de activos financieros en crisis.

    Incluye prueba de bondad de ajuste K-S (Kolmogorov-Smirnov):
    si p-valor < 0.05, rechaza GEV y usa el cuantil empírico como fallback.

    Devuelve (VaR, CVaR) como números negativos (pérdida = número negativo).
    """
    tail = -returns[returns < 0]
    if len(tail) < 20:
        # Not enough tail observations — fall back to empirical
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    params = genextreme.fit(tail)
    # Goodness-of-fit check: reject GEV if K-S p-value < 0.05
    _ks_stat, ks_pval = kstest(tail, "genextreme", args=params)
    if ks_pval < 0.05:
        logger.debug(
            "GEV fit rejected by K-S test (p=%.4f); falling back to empirical quantile.", ks_pval
        )
        empirical_var = float(np.percentile(returns.dropna(), (1 - alpha) * 100))
        return empirical_var, float(returns[returns <= empirical_var].mean() if any(returns <= empirical_var) else empirical_var)
    # ppf gives positive loss magnitude; negate to match return convention (negative = loss)
    loss_var = float(genextreme.ppf(alpha, *params))
    x_tail = np.linspace(loss_var, genextreme.ppf(0.9999, *params), 1000)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    loss_cvar = float(_trapz(x_tail * genextreme.pdf(x_tail, *params), x_tail) / (1 - alpha))
    return -loss_var, -loss_cvar


def compute_macro_regime_history(
    macro: Optional[pd.DataFrame],
    settings: dict | None = None,
) -> pd.DataFrame:
    """Construye la historia de regímenes macro a lo largo del tiempo.

    Dos métodos (configurado via settings['regime_method']):

    'threshold_discrete':
      - Reglas fijas: stress si IP < 0 o MXN se depreció >5% en 3 meses.
                      tightening si Banxico subió >25 bps en 3 meses.
                      expansion en otro caso.
      - Simple y transparente, pero sin suavización → muchos cambios de régimen.

    'ewma_composite' (default):
      - Construye un score = z(IP) - z(FX) - z(Banxico) donde z = z-score expansivo.
      - Suaviza el score con EWMA (span configurable).
      - Aplica umbrales al score suavizado (expansion_threshold / stress_threshold).
      - Requiere confianza mínima para aceptar un cambio de régimen (evita ruido).
    """
    cfg = resolve_settings(settings)
    df = _as_macro_frame(macro)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "regime",
                "regime_score",
                "regime_score_smoothed",
                "regime_confidence",
                "banxico_delta_3m",
                "fx_mom_3m",
            ]
        )

    out = pd.DataFrame(index=df.index)
    out["banxico_delta_3m"] = pd.Series(df.get("banxico_rate", 0.0), index=df.index, dtype=float).diff(3)
    out["fx_mom_3m"] = pd.Series(df.get("usd_mxn", 0.0), index=df.index, dtype=float).pct_change(3)
    out["ip_yoy"] = pd.Series(df.get("industrial_production_yoy", 0.02), index=df.index, dtype=float)

    if str(cfg["regime_method"]).lower() == "threshold_discrete":
        conditions = [
            (out["ip_yoy"] < 0.0) | (out["fx_mom_3m"] > 0.05),
            out["banxico_delta_3m"] > 0.25,
        ]
        choices = ["stress", "tightening"]
        out["regime"] = np.select(conditions, choices, default="expansion")
        out["regime_score"] = np.nan
        out["regime_score_smoothed"] = np.nan
        out["regime_confidence"] = np.nan
        return out

    ip_z = _expanding_zscore(out["ip_yoy"])
    fx_z = _expanding_zscore(out["fx_mom_3m"])
    banxico_z = _expanding_zscore(out["banxico_delta_3m"])
    out["regime_score"] = ip_z - fx_z - banxico_z
    out["regime_score_smoothed"] = out["regime_score"].ewm(
        span=int(cfg["regime_ewma_span"]),
        min_periods=3,
        adjust=False,
    ).mean()
    out["regime_confidence"] = out["regime_score_smoothed"].abs().clip(0.0, 3.0)

    upper = float(cfg["regime_threshold_expansion"])
    lower = float(cfg["regime_threshold_stress"])
    out["raw_regime"] = np.where(
        out["regime_score_smoothed"] > upper,
        "expansion",
        np.where(out["regime_score_smoothed"] < lower, "stress", "tightening"),
    )
    out["raw_regime"] = out["raw_regime"].astype(object).where(out["regime_score_smoothed"].notna(), "expansion")

    min_conf = float(cfg["regime_min_confidence_for_switch"])
    effective: list[str] = []
    previous = "expansion"
    for idx in out.index:
        raw_regime = str(out.at[idx, "raw_regime"])
        confidence = float(out.at[idx, "regime_confidence"]) if pd.notna(out.at[idx, "regime_confidence"]) else np.nan
        if effective and raw_regime != previous and np.isfinite(confidence) and confidence < min_conf:
            effective.append(previous)
        else:
            effective.append(raw_regime)
            previous = raw_regime
    out["regime"] = pd.Series(effective, index=out.index)
    return out


def detect_macro_regime(macro: Optional[pd.DataFrame], settings: dict | None = None) -> str:
    """Return the latest macro regime state from the configured detector."""
    history = compute_macro_regime_history(macro, settings=settings)
    if history.empty:
        return "expansion"
    return str(history["regime"].iloc[-1])


def regime_asset_class_constraints(regime: str) -> dict:
    """Return asset-class min/max constraints keyed by regime.

    expansion : full risk-on — max equity, minimal bonds
    tightening: reduce equity, add bonds as vol buffer
    stress    : defensive — cut equity, max bonds

    The __asset_class_map__ key must be added by the caller.
    """
    _regimes = {
        "expansion": {
            "equity":       {"min": 0.55, "max": 0.90},
            "fibra":        {"min": 0.05, "max": 0.25},
            "fixed_income": {"min": 0.00, "max": 0.10},
        },
        "tightening": {
            "equity":       {"min": 0.45, "max": 0.75},
            "fibra":        {"min": 0.05, "max": 0.20},
            "fixed_income": {"min": 0.05, "max": 0.20},
        },
        "stress": {
            "equity":       {"min": 0.35, "max": 0.60},
            "fibra":        {"min": 0.03, "max": 0.15},
            "fixed_income": {"min": 0.10, "max": 0.30},
        },
    }
    return dict(_regimes.get(regime, _regimes["expansion"]))


def distributional_stress_test(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
    macro: Optional[pd.DataFrame],
    n_reps: int = 5000,
    window_days: int = 21,
    seed: int = 42,
) -> dict[str, dict]:
    """Stress test distribucional: bootstrapea ventanas históricas de crisis.

    En lugar del shock determinístico de stress_test(), este enfoque:
      1. Identifica ventanas de window_days días donde ocurrió cada tipo de crisis
         (usando datos macro reales: Banxico, FX, IP industrial, S&P500).
      2. Calcula el P&L del portafolio actual en cada ventana histórica real.
      3. Bootstrapea n_reps muestras de esas ventanas para generar la distribución
         del P&L bajo cada escenario (percentiles p5, p25, p50, p75, p95).
      4. Para escenarios con < 3 ventanas históricas, usa fallbacks con fechas de
         crisis conocidas (COVID-19 2020, tightening 2022, tarifas 2025, etc.).

    Devuelve distribuciones de P&L por escenario con CVaR al 95% para cuantificar
    la pérdida esperada en la cola de cada tipo de estrés.
    """
    if asset_returns is None or asset_returns.empty or current_weights is None or current_weights.empty:
        return {}

    returns = asset_returns.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = current_weights.reindex(returns.columns).fillna(0.0)
    if float(weights.abs().sum()) <= 1e-9:
        return {}
    weights = weights / max(float(weights.sum()), 1e-9)

    macro_df = _as_macro_frame(macro)
    if macro_df.empty:
        return {}
    macro_daily = macro_df.reindex(returns.index, method="ffill")
    span = max(int(window_days), 2)

    banxico_delta = pd.Series(macro_daily.get("banxico_rate", 0.0), index=returns.index, dtype=float).diff(span)
    fx_mom = pd.Series(macro_daily.get("usd_mxn", 0.0), index=returns.index, dtype=float).pct_change(span)
    if "sp500" in macro_daily.columns:
        sp500_mom = pd.Series(macro_daily["sp500"], index=returns.index, dtype=float).pct_change(span)
    else:
        sp500_mom = pd.Series(np.nan, index=returns.index, dtype=float)
    us_ip = pd.Series(macro_daily.get("us_ip_yoy", np.nan), index=returns.index, dtype=float)
    if "exports_yoy" in macro_daily.columns:
        exports_yoy = pd.Series(macro_daily["exports_yoy"], index=returns.index, dtype=float)
        tmec_mask = (fx_mom > 0.04) & (us_ip < 1.0) & (exports_yoy < 0.0)
    else:
        tmec_mask = (fx_mom > 0.04) & (us_ip < 1.0)

    scenario_masks = {
        "banxico_shock": banxico_delta.abs() > 0.01,
        "peso_depreciation": fx_mom > 0.05,
        "us_slowdown": (us_ip < 0.0) & (sp500_mom < -0.08),
        "tmec_disruption": tmec_mask,
    }

    fallback_windows = [
        ("2020-03-02", "2020-03-31", "2020-03 COVID shock"),
        ("2022-10-03", "2022-10-31", "2022-10 global tightening"),
        ("2025-02-03", "2025-02-24", "2025-02 tariff shock"),
        ("2018-05-30", "2018-06-29", "2018-06 NAFTA acero/aluminio tariffs"),
        ("2019-05-30", "2019-06-07", "2019-06 amenaza arancel migratorio MX"),
    ]

    def _window_pnl(start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, str] | None:
        window = returns.loc[start:end]
        if len(window) < span // 2:
            return None
        pnl = float(np.exp(window.dot(weights).sum()) - 1.0)
        label = f"{window.index[0].date()} to {window.index[-1].date()}"
        return pnl, label

    results: dict[str, dict] = {}
    min_spacing = max(span // 2, 5)
    for idx, (scenario, mask) in enumerate(scenario_masks.items()):
        windows: list[tuple[float, str]] = []
        active_dates = mask.fillna(False)
        start_idx = -span
        for date in active_dates.index[active_dates]:
            try:
                current_idx = returns.index.get_loc(date)
            except KeyError:
                continue
            if isinstance(current_idx, slice):
                current_idx = current_idx.start
            if current_idx - start_idx < min_spacing:
                continue
            end_idx = min(current_idx + span - 1, len(returns.index) - 1)
            result = _window_pnl(returns.index[current_idx], returns.index[end_idx])
            if result is not None:
                windows.append(result)
                start_idx = current_idx

        if len(windows) < 3:
            for start_raw, end_raw, label in fallback_windows:
                start = pd.Timestamp(start_raw)
                end = pd.Timestamp(end_raw)
                result = _window_pnl(start, end)
                if result is not None:
                    windows.append((result[0], label))

        if not windows:
            continue

        pnl_values = np.asarray([p for p, _ in windows], dtype=float)
        rng = np.random.default_rng(int(seed) + idx)
        sampled = rng.choice(pnl_values, size=int(n_reps), replace=True)
        p5 = float(np.quantile(sampled, 0.05))
        tail = sampled[sampled <= p5]
        worst_idx = int(np.argmin(pnl_values))
        results[scenario] = {
            "n_historical_windows": int(len(windows)),
            "pnl_distribution": {
                "mean": float(np.mean(sampled)),
                "p5": p5,
                "p25": float(np.quantile(sampled, 0.25)),
                "p50": float(np.quantile(sampled, 0.50)),
                "p75": float(np.quantile(sampled, 0.75)),
                "p95": float(np.quantile(sampled, 0.95)),
                "std": float(np.std(sampled, ddof=1)) if len(sampled) > 1 else 0.0,
            },
            "cvar_95_pnl": float(tail.mean()) if len(tail) else p5,
            "worst_historical_window": windows[worst_idx][1],
            "historical_pnls": pnl_values,
        }

    return results
