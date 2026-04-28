from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Parámetros de configuración global del pipeline.
    # Cada sección controla un módulo diferente del sistema.
    # Para sobreescribir, pasar un dict con las claves a modificar a resolve_settings().
    # ------------------------------------------------------------------
    # Covarianza / EWMA
    # ------------------------------------------------------------------
    "covariance_method": "ewma_ledoit_wolf",
    "rolling_cov_window": 63,
    "ewma_lambda_cov": 0.94,
    "ewma_window_cov": 252,
    "ewma_min_periods_cov": 60,
    "realized_vol_method": "ewma",
    "realized_vol_span": 21,
    # ------------------------------------------------------------------
    # Macro regime
    # ------------------------------------------------------------------
    "regime_method": "ewma_composite",
    "regime_ewma_span": 6,
    "regime_threshold_expansion": 0.5,
    "regime_threshold_stress": -0.5,
    "regime_min_confidence_for_switch": 0.3,
    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------
    "adtv_method": "ewma",
    "adtv_window": 252,
    "adtv_ewma_lambda": 0.97,
    "adtv_min_periods": 60,
    # ------------------------------------------------------------------
    # Bootstrap / significance
    # ------------------------------------------------------------------
    "bootstrap_enabled": True,
    "bootstrap_n_reps": 5000,
    "bootstrap_block_size": 20,
    "bootstrap_confidence": 0.95,
    "bootstrap_seed": 42,
    # ------------------------------------------------------------------
    # Fan chart
    # ------------------------------------------------------------------
    "fan_chart_enabled": True,
    "fan_chart_n_paths": 1000,
    "fan_chart_block_size": 20,
    # ------------------------------------------------------------------
    # Signal IC
    # ------------------------------------------------------------------
    "ic_diagnostics_enabled": True,
    "ic_bootstrap_block_size": 6,
    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------
    "stress_distributional_enabled": True,
    "stress_window_days": 21,
    # ------------------------------------------------------------------
    # Black-Litterman views
    # ------------------------------------------------------------------
    "bl_views": {
        "use_macro": True,
        "macro_view_confidence": 0.20,
        "macro_view_max_magnitude": 0.015,
    },
    # ------------------------------------------------------------------
    # ETF → normal anchor (soft sector budget propagated from ETF run)
    # ------------------------------------------------------------------
    "etf_sector_anchor": {
        "enabled": False,
        "band": 0.15,
        "source": "bloomberg",
        "fallback_to_unanchored": True,
    },
    # ------------------------------------------------------------------
    # Reporting / comparisons
    # ------------------------------------------------------------------
    "enable_method_comparison": True,
    # Numerical tolerances
    "covariance_psd_tolerance": 1e-9,
    # ------------------------------------------------------------------
    # Forecast model (ElasticNetCV)
    # ------------------------------------------------------------------
    "elasticnet_cv_folds": 5,
    "elasticnet_l1_ratios": [0.1, 0.5, 0.9],
    "elasticnet_max_iter": 10000,
    "elasticnet_tol": 1e-3,
    "forecast_forward_days": 21,
    "forecast_min_train_rows": 50,
    # ------------------------------------------------------------------
    # Portfolio / optimizer hyperparameters
    # Defaults are kept bit-identical to the values currently hardcoded in
    # backtest.py and portfolio.py, so an unmodified run reproduces
    # pre-migration results exactly.
    # ------------------------------------------------------------------
    "bl_risk_aversion": 2.5,
    "bl_tau": 0.05,
    "mv_risk_aversion": 4.0,
    "mv_turnover_penalty": 0.05,
    "mv_market_impact_eta": 0.1,
    "cvar_risk_aversion": 25.0,
    "cvar_turnover_penalty": 0.01,
    "cvar_alpha": 0.99,
    "robust_risk_aversion": 4.0,
    "robust_turnover_penalty": 0.05,
    "michaud_n_simulations": 100,
    "michaud_t_effective": 252,
    "target_net_exposure_mv": 0.90,
    "target_net_exposure_cvar": 0.75,
    "target_net_exposure_robust": 0.90,
    "fx_hedge_ratio_default": 0.5,
    "garch_refit_every": 5,
    "garch_forecast_horizon": 21,
    "garch_lookback": 252,
}


def resolve_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Devuelve una copia de DEFAULT_SETTINGS con las claves de 'settings' aplicadas encima.

    Permite que cada función del pipeline reciba parámetros personalizados sin
    romper los defaults. Las claves None en 'settings' se ignoran (no sobreescriben).
    """
    merged = deepcopy(DEFAULT_SETTINGS)
    if settings:
        for key, value in settings.items():
            if value is not None:
                merged[key] = value
    return merged
