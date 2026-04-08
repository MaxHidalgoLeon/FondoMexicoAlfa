from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .backtest import run_backtest
from .data_loader import load_mock_data
from .features import build_signal_matrix
from .signals import score_cross_section, forecast_returns
from .portfolio import black_litterman, apply_fx_overlay, optimize_portfolio
from .risk import stress_test, fit_garch, garch_forecast_vol, dynamic_var, monte_carlo_var, gev_var

logger = logging.getLogger(__name__)


def run_pipeline(hedge_mode: bool = False) -> dict[str, object]:
    data = load_mock_data()
    universe = data["universe"]
    prices = data["prices"]
    fundamentals = data["fundamentals"]
    fibra_fundamentals = data["fibra_fundamentals"]
    bonds = data["bonds"]
    macro = data["macro"]

    feature_df = build_signal_matrix(prices, fundamentals, fibra_fundamentals, bonds, macro, universe)
    scored = score_cross_section(feature_df)
    forecast_df = forecast_returns(scored, prices.pct_change().fillna(0.0))
    
    # Black-Litterman
    forecast_tickers = forecast_df["ticker"].unique() if not forecast_df.empty else []
    market_weights = universe.set_index("ticker")["market_cap_mxn"] / (universe["market_cap_mxn"].sum() + 1e-9)
    market_weights = market_weights.reindex(forecast_tickers).fillna(0.0)

    # Warn about any ticker mismatch between forecast and universe
    missing_from_universe = set(forecast_tickers) - set(universe["ticker"])
    if missing_from_universe:
        logger.warning("Tickers in forecast but not in universe (will get 0 market weight): %s", missing_from_universe)

    # Build covariance with Ledoit-Wolf shrinkage
    raw_returns = prices.pct_change().dropna(how="all")
    cov_tickers = [t for t in forecast_tickers if t in raw_returns.columns]
    if cov_tickers:
        lw = LedoitWolf()
        lw.fit(raw_returns[cov_tickers].fillna(0.0))
        cov_matrix = pd.DataFrame(lw.covariance_, index=cov_tickers, columns=cov_tickers)
    else:
        cov_matrix = pd.DataFrame(dtype=float)

    views = forecast_df.groupby("ticker")["expected_return"].mean().to_dict()
    # Data-driven view confidences: scale |expected_return| to [0.30, 0.70]
    abs_views = pd.Series(views).abs()
    view_range = abs_views.max() - abs_views.min()
    if view_range > 1e-9:
        view_confidences = (0.30 + 0.40 * (abs_views - abs_views.min()) / view_range).to_dict()
    else:
        view_confidences = {t: 0.50 for t in views}

    bl_returns = black_litterman(market_weights, cov_matrix, views, view_confidences)
    
    # FX overlay
    usd_exposure = universe.set_index("ticker")["usd_exposure"]
    expected_usdmxn_return = macro["usd_mxn"].pct_change().mean()  # simple proxy
    adjusted_returns = apply_fx_overlay(bl_returns, usd_exposure, macro["usd_mxn"].iloc[-1], expected_usdmxn_return)
    
    # TODO: pass asset_class_constraints to optimize_portfolio once implemented
    backtest_results = run_backtest(prices, forecast_df, universe)

    exposures = {
        "banxico_shock": 0.5,
        "peso_depreciation": 0.6,
        "us_slowdown": 0.4,
    }
    scenario_shocks = {
        "banxico_shock": -0.03,
        "peso_depreciation": -0.05,
        "us_slowdown": -0.04,
    }
    stress = stress_test(backtest_results["returns"], scenario_shocks, exposures)
    
    # Additional risk metrics
    returns = backtest_results["returns"]
    garch_vol = garch_forecast_vol(fit_garch(returns))
    dyn_var = dynamic_var(returns).iloc[-1]
    mc_var = monte_carlo_var(returns)
    gev_v, gev_cv = gev_var(returns)
    
    summary = {
        "universe_size": len(universe),
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "metrics": backtest_results["metrics"],
        "stress": stress,
        "garch_vol_forecast": garch_vol if np.isfinite(garch_vol) else None,
        "dynamic_var": float(dyn_var) if np.isfinite(dyn_var) else None,
        "monte_carlo_var": mc_var,
        "gev_var": gev_v,
        "gev_cvar": gev_cv,
    }

    results = {
        "data": data,
        "feature_df": feature_df,
        "forecast_df": forecast_df,
        "backtest": backtest_results,
        "summary": summary,
    }
    
    # Layer 2 hedge mode
    if hedge_mode:
        from .hedge_overlay import run_hedge_backtest
        hedge_results = run_hedge_backtest(
            prices,
            forecast_df,
            universe,
            macro,
            max_leverage=1.5,
            cvar_limit=0.02,
            transaction_cost=0.0015,
        )
        results["hedge_layer"] = hedge_results

    return results


def print_summary(results: dict[str, object], hedge_mode: bool = False) -> None:
    summary = results["summary"]
    print("=== Strategy Pipeline Summary ===")
    print(f"Universe size: {summary['universe_size']}")
    print(f"Backtest period: {summary['start_date'].date()} to {summary['end_date'].date()}")
    
    if hedge_mode and "hedge_layer" in results:
        # Side-by-side comparison
        print("\n=== Layer 1 vs Layer 2 (Hedge) Metrics ===")
        print(f"{'Metric':<22} {'Layer 1':<15} {'Layer 2':<15}")
        print("-" * 52)
        
        layer1_metrics = summary["metrics"]
        layer2_metrics = results["hedge_layer"]["metrics"]
        
        for key in layer1_metrics.keys():
            l1_val = layer1_metrics.get(key, 0.0)
            l2_val = layer2_metrics.get(key, 0.0)
            l1_str = f"{l1_val:>14.4f}" if isinstance(l1_val, (int, float)) and np.isfinite(l1_val) else f"{'N/A':>14}"
            l2_str = f"{l2_val:>14.4f}" if isinstance(l2_val, (int, float)) and np.isfinite(l2_val) else f"{'N/A':>14}"
            print(f"{key:<22} {l1_str} {l2_str}")
        
        print("\nTail hedge analysis (Layer 2):")
        tail_hedge = results["hedge_layer"]["tail_hedge"]
        print(f"  Unhedged loss at 99%: {tail_hedge['unhedged_loss_at_99']:.4f}")
        print(f"  Hedge payoff: {tail_hedge['hedge_payoff']:.4f}")
        print(f"  Daily cost drag: {tail_hedge['daily_cost_drag']:.6f}")
        print(f"  Net benefit: {tail_hedge['net_benefit']:.4f}")
        print(f"  Recommended: {tail_hedge['recommended']}")
    else:
        print("\nLayer 1 metrics:")
        for key, value in summary["metrics"].items():
            val_str = f"{value:.4f}" if isinstance(value, (int, float)) and np.isfinite(value) else "N/A"
            print(f"{key}: {val_str}")
        print("\nStress test results:")
        print(summary["stress"].to_string(index=False))


if __name__ == "__main__":
    results = run_pipeline()
    print_summary(results)
