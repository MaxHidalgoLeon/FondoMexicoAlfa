from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_SETTINGS: dict[str, Any] = {
    # ------------------------------------------------------------------
    # Global pipeline configuration parameters.
    # Each section controls a different module of the system.
    # To override, pass a dict with the keys to modify to resolve_settings().
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
    "regime_method": "ewma_composite",  # only supported method; kept for auditability
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
    # Forecast model (ElasticNetCV is the baseline; LightGBM is an
    # interchangeable alternative selected by `forecast_model`).
    # ------------------------------------------------------------------
    "forecast_model": "elasticnet",         # "elasticnet" | "lightgbm"
    "elasticnet_cv_folds": 5,
    "elasticnet_l1_ratios": [0.1, 0.5, 0.9],
    "elasticnet_max_iter": 10000,
    "elasticnet_tol": 1e-3,
    "elasticnet_random_state": 42,
    "forecast_forward_days": 21,
    "forecast_min_train_rows": 50,
    # LightGBM-specific knobs (ignored when forecast_model != "lightgbm").
    # Kept in DEFAULT_SETTINGS so resolve_settings() always returns them.
    "forecast_lgbm_n_iter": 20,              # RandomizedSearchCV draws
    "forecast_lgbm_cv_splits": 5,            # TimeSeriesSplit folds
    "forecast_lgbm_n_estimators_cap": 2000,  # early-stopping ceiling
    "forecast_lgbm_early_stopping_rounds": 50,
    "forecast_lgbm_scoring": "neg_mean_squared_error",  # or "ic"
    "forecast_lgbm_random_state": 42,
    "forecast_lgbm_n_jobs": 1,
    "forecast_lgbm_search_n_jobs": 1,
    # SHAP attribution (only consumed when forecast_model == "lightgbm").
    "compute_shap": True,           # set False to skip SHAP and save runtime
    "shap_output_path": "data/shap_values.parquet",  # overwritten each run
    # ------------------------------------------------------------------
    # Portfolio / optimizer hyperparameters
    # Defaults are kept bit-identical to the values currently hardcoded in
    # backtest.py and portfolio.py, so an unmodified run reproduces
    # pre-migration results exactly.
    # ------------------------------------------------------------------
    # bl_risk_aversion is the implied market δ used to back out CAPM equilibrium
    # returns in Black-Litterman (π = δ·Σ·w_mkt). It is NOT the investor's
    # utility risk-aversion — that role belongs to mv_risk_aversion below.
    # Default 2.5 ≈ empirical Sharpe-ratio-matching δ for broad equity markets.
    "bl_risk_aversion": 2.5,
    "bl_tau": 0.05,
    # mv_risk_aversion is the investor's utility λ in the MV objective:
    # max  μ'w − λ/2 · w'Σw  subject to constraints.
    # Higher λ → more conservative; independent of bl_risk_aversion above.
    "mv_risk_aversion": 4.0,
    "mv_turnover_penalty": 0.05,
    "mv_market_impact_eta": 0.1,
    "cvar_risk_aversion": 25.0,
    "cvar_turnover_penalty": 0.01,
    "cvar_alpha": 0.99,
    # Number of trailing trading days used as the scenario set for CVaR optimization.
    # 252 (1 year) is adequate for daily backtests; monthly rebalancing benefits from
    # a longer window (504–756 days) to ensure enough observations in the tail at α=0.99.
    "cvar_scenario_window": 504,
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
    # ------------------------------------------------------------------
    # Liquidity sleeve (CETES28 + CETES91 deterministic weights by regime).
    # MBONO3Y is an optional buffer enabled by mbono3y_buffer_enabled.
    # ------------------------------------------------------------------
    "liquidity_sleeve_min_expansion": 0.03,
    "liquidity_sleeve_max_expansion": 0.05,
    "liquidity_sleeve_min_tightening": 0.05,
    "liquidity_sleeve_max_tightening": 0.08,
    "liquidity_sleeve_min_stress": 0.08,
    "liquidity_sleeve_max_stress": 0.15,
    "mbono3y_buffer_enabled": False,
    "mbono3y_buffer_max": 0.03,
    # ------------------------------------------------------------------
    # Risk-free rate fallback used when banxico_rate is unavailable.
    # Always interpreted as a decimal; values >1.0 are normalized as percent.
    # ------------------------------------------------------------------
    "risk_free_rate": 0.04,
    # ------------------------------------------------------------------
    # FX vol regime calibration (used by fx_directional_overlay):
    #   neutral   — annualized USD/MXN vol where the boost is zero (default 15%).
    #   spread    — width of one boost band; vol_zscore = (vol - neutral) / spread.
    #   boost_cap — max ±adjustment to hedge_ratio from the vol regime (default 5%).
    # ------------------------------------------------------------------
    "fx_vol_neutral": 0.15,
    "fx_vol_spread": 0.10,
    "fx_vol_boost_cap": 0.05,
}


# ---------------------------------------------------------------------------
# Universe-specific overlays applied automatically by run_etf_pipeline.
# The ETF universe (EWW, INDS, IGF, ILF) is highly liquid and has only four
# buckets, so two parameters are structurally different from the BMV universe:
#   - mv_market_impact_eta: ETFs trade ~100x the ADV of the BMV equities, so
#     the Almgren-Chriss linear impact term should be much smaller.
#   - mv_risk_aversion: with 4 deeply liquid buckets there is no penalty for
#     larger active bets, so a less risk-averse weighting is appropriate.
#   - regime_ewma_span: a slightly slower regime filter is more stable with
#     a small cross-section.
# All other parameters (bl_*, cvar_*, forecast_forward_days, elasticnet_*) are
# left at the global defaults — past Optuna runs on this 4-instrument universe
# captured fold noise rather than robust signal (cvar_risk_aversion swung 6x
# between providers for the same four ETFs).
# ---------------------------------------------------------------------------
ETF_UNIVERSE_OVERRIDES: dict[str, Any] = {
    "mv_market_impact_eta": 0.03,
    "mv_risk_aversion":     2.0,
    "regime_ewma_span":     9,
}


def resolve_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of DEFAULT_SETTINGS with the keys in 'settings' applied on top.

    Allows each pipeline function to receive custom parameters without breaking
    the defaults. None values in 'settings' are ignored (do not override).

    Unknown keys (not in DEFAULT_SETTINGS) are silently merged but logged as a
    warning — they are almost always typos or stale parameter names (e.g., a
    renamed key) that would otherwise be silently ignored.
    """
    merged = deepcopy(DEFAULT_SETTINGS)
    if settings:
        unknown = [k for k in settings if k not in DEFAULT_SETTINGS and settings[k] is not None]
        if unknown:
            logger.warning(
                "resolve_settings: unknown key(s) %s — these are not in DEFAULT_SETTINGS "
                "and will have no effect. Check for typos or stale parameter names.",
                unknown,
            )
        for key, value in settings.items():
            if value is not None:
                merged[key] = value
    return merged
