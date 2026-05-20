"""Unit tests for src/settings.py — resolve_settings semantics."""
from __future__ import annotations

from src.settings import DEFAULT_SETTINGS, ETF_UNIVERSE_OVERRIDES, resolve_settings


def test_resolve_settings_none_returns_defaults():
    out = resolve_settings(None)
    for key, value in DEFAULT_SETTINGS.items():
        assert out[key] == value


def test_resolve_settings_overrides_one_key():
    out = resolve_settings({"forecast_forward_days": 42})
    assert out["forecast_forward_days"] == 42
    # Other keys remain at defaults.
    assert out["bootstrap_n_reps"] == DEFAULT_SETTINGS["bootstrap_n_reps"]


def test_resolve_settings_ignores_none_values():
    """None values must NOT override a real default — important for argparse merges."""
    out = resolve_settings({"forecast_forward_days": None})
    assert out["forecast_forward_days"] == DEFAULT_SETTINGS["forecast_forward_days"]


def test_resolve_settings_does_not_mutate_defaults():
    """resolve_settings must deep-copy DEFAULT_SETTINGS; mutating the result must not leak."""
    out = resolve_settings({"forecast_forward_days": 7})
    out["bl_views"]["use_macro"] = False
    assert DEFAULT_SETTINGS["bl_views"]["use_macro"] is True


def test_resolve_settings_carries_new_keys():
    """Sleeve, mbono3y, fx-vol, hedge-regulated and rf keys must be in defaults."""
    out = resolve_settings()
    for k in (
        "liquidity_sleeve_min_expansion", "liquidity_sleeve_max_stress",
        "mbono3y_buffer_enabled", "mbono3y_buffer_max",
        "risk_free_rate", "fx_vol_neutral", "fx_vol_spread", "fx_vol_boost_cap",
    ):
        assert k in out, f"Missing default for {k}"


def test_etf_universe_overrides_keys_and_values():
    """Overrides must cover the three structurally-different parameters."""
    assert ETF_UNIVERSE_OVERRIDES["mv_market_impact_eta"] == 0.03
    assert ETF_UNIVERSE_OVERRIDES["mv_risk_aversion"] == 2.0
    assert ETF_UNIVERSE_OVERRIDES["regime_ewma_span"] == 9


def test_etf_overrides_merge_with_caller_winning():
    """When run_etf_pipeline merges {**overrides, **caller}, caller-provided
    values must take precedence; non-overridden keys come from defaults."""
    merged = resolve_settings({**ETF_UNIVERSE_OVERRIDES, "mv_risk_aversion": 7.5})
    assert merged["mv_market_impact_eta"] == 0.03    # from override
    assert merged["mv_risk_aversion"] == 7.5         # caller wins
    assert merged["bl_risk_aversion"] == DEFAULT_SETTINGS["bl_risk_aversion"]  # untouched default
