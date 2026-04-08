from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def get_investable_universe() -> pd.DataFrame:
    """Create the initial thematic universe for Mexico industrial research."""
    tickers = [
        "GFM0", "FIBRA1", "LOGI2", "INDU3", "UTIL4",
        "TRANS5", "ENER6", "TECH7", "MEXI8", "ENGR9",
        "AUT10", "RWL11", "IND12", "FIBR13", "LOG14",
        "IND15", "FRA16", "STOR17", "CARG18", "ECO19",
        "CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2",
    ]
    names = [
        "Grupo Fabricación", "Fibra Industrial Uno", "Logística Norte", "Industria MX", "Utility Industrial",
        "Transporte Integral", "Energía Manufacturera", "Tecnologías de Planta", "México Industrial", "Ingeniería 4.0",
        "Automotriz Supply", "Rieles Logísticos", "Industrial Desarrollo", "Fibra Park", "Logística Global",
        "Industria Sostenible", "FIBRA Renta", "Storage Infra", "Cargo Rex", "Energía Conexión",
        "Cetes 28d", "Cetes 91d", "Mbono 3yr", "Mbono 5yr", "Mbono 10yr", "Corporate Bond 1", "Corporate Bond 2",
    ]
    sectors = [
        "Industrial", "FIBRA", "Logistics", "Industrial", "Utilities",
        "Logistics", "Energy", "Industrial", "Industrial", "Industrial",
        "Industrial", "Logistics", "Industrial", "FIBRA", "Logistics",
        "Industrial", "FIBRA", "Logistics", "Logistics", "Energy",
        "Government", "Government", "Government", "Government", "Government", "Corporate", "Corporate",
    ]
    asset_classes = [
        "equity", "fibra", "equity", "equity", "equity",
        "equity", "equity", "equity", "equity", "equity",
        "equity", "equity", "equity", "fibra", "equity",
        "equity", "fibra", "equity", "equity", "equity",
        "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income", "fixed_income",
    ]
    flags = [
        True, True, True, True, False,
        True, False, True, True, True,
        True, True, True, True, True,
        True, True, True, True, False,
        True, True, True, True, True, True, True,
    ]
    usd_exposure = [0.2, 0.4, 0.1, 0.35, 0.15, 0.25, 0.55, 0.1, 0.3, 0.2, 0.4, 0.2, 0.3, 0.45, 0.2, 0.25, 0.4, 0.05, 0.15, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    market_caps = np.linspace(10_000, 120_000, 20).tolist() + [0] * 7  # bonds have no market cap
    liquidity = np.linspace(0.25, 1.0, 20).tolist() + [1.0] * 7  # bonds are liquid
    df = pd.DataFrame({
        "ticker": tickers,
        "name": names,
        "sector": sectors,
        "asset_class": asset_classes,
        "investable": flags,
        "usd_exposure": usd_exposure,
        "market_cap_mxn": market_caps,
        "liquidity_score": liquidity,
    })
    return df


def generate_mock_price_series(
    tickers: list[str],
    start_date: str = "2018-01-01",
    end_date: str = "2025-12-31",
    freq: str = "B",
) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq=freq)
    n = len(dates)
    prices = {}
    np.random.seed(42)
    for ticker in tickers:
        drift = np.random.uniform(0.02, 0.12)
        vol = np.random.uniform(0.18, 0.35)
        shocks = np.random.normal(loc=(drift / 252), scale=(vol / np.sqrt(252)), size=n)
        level = 100 * np.exp(np.cumsum(shocks))
        prices[ticker] = level
    price_df = pd.DataFrame(prices, index=dates).clip(lower=1)
    return price_df


def build_mock_fundamentals(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate per-ticker fundamentals with AR(1) persistence — no fresh random draw per month."""
    rng = np.random.default_rng(24)
    records = []
    for ticker in tickers:
        # Initial values per ticker drawn once
        pe = float(rng.uniform(8.0, 24.0))
        pb = float(rng.uniform(1.0, 3.5))
        roe = float(rng.normal(0.16, 0.05))
        margin = float(rng.normal(0.18, 0.07))
        leverage = float(max(0.0, rng.normal(2.5, 1.0)))
        ebitda_growth = float(rng.normal(0.08, 0.08))
        capex = float(max(0.01, rng.normal(0.06, 0.02)))
        for date in dates:
            # Slow AR(1) random walk — economically realistic persistence
            pe = float(np.clip(pe * (1 + rng.normal(0.0, 0.03)), 4.0, 45.0))
            pb = float(np.clip(pb * (1 + rng.normal(0.0, 0.03)), 0.4, 7.0))
            roe = float(np.clip(roe + rng.normal(0.0, 0.008), 0.01, 0.40))
            margin = float(np.clip(margin + rng.normal(0.0, 0.008), 0.01, 0.50))
            leverage = float(np.clip(leverage + rng.normal(0.0, 0.08), 0.0, 8.0))
            ebitda_growth = float(np.clip(ebitda_growth + rng.normal(0.0, 0.015), -0.25, 0.35))
            capex = float(np.clip(capex + rng.normal(0.0, 0.004), 0.01, 0.15))
            records.append({
                "date": date,
                "ticker": ticker,
                "ebitda_growth": ebitda_growth,
                "net_debt_to_ebitda": leverage,
                "roe": roe,
                "profit_margin": margin,
                "capex_to_sales": capex,
                "pe_ratio": pe,
                "pb_ratio": pb,
            })
    return pd.DataFrame.from_records(records)


def build_mock_fibra_fundamentals(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate per-ticker FIBRA fundamentals with AR(1) persistence."""
    rng = np.random.default_rng(25)
    records = []
    for ticker in tickers:
        cap_rate = float(rng.uniform(0.05, 0.12))
        ffo_yield = float(rng.uniform(0.06, 0.15))
        div_yield = float(rng.uniform(0.04, 0.10))
        ltv = float(rng.uniform(0.30, 0.65))
        vacancy = float(rng.uniform(0.02, 0.12))
        for date in dates:
            cap_rate = float(np.clip(cap_rate + rng.normal(0.0, 0.003), 0.03, 0.16))
            ffo_yield = float(np.clip(ffo_yield + rng.normal(0.0, 0.004), 0.03, 0.20))
            div_yield = float(np.clip(div_yield + rng.normal(0.0, 0.003), 0.02, 0.15))
            ltv = float(np.clip(ltv + rng.normal(0.0, 0.01), 0.10, 0.80))
            vacancy = float(np.clip(vacancy + rng.normal(0.0, 0.005), 0.01, 0.25))
            records.append({
                "date": date,
                "ticker": ticker,
                "cap_rate": cap_rate,
                "ffo_yield": ffo_yield,
                "dividend_yield": div_yield,
                "ltv": ltv,
                "vacancy_rate": vacancy,
            })
    return pd.DataFrame.from_records(records)
   


def _bond_price(ytm: float, coupon_rate: float, n_years: float) -> float:
    """Price a bond with annual coupons using the present value formula."""
    if ytm <= 0 or n_years <= 0:
        return 100.0
    coupon = 100.0 * coupon_rate
    discount = (1.0 + ytm) ** (-n_years)
    # Price = PV of coupons + PV of face
    return coupon * (1.0 - discount) / ytm + 100.0 * discount


def build_mock_bonds(dates: pd.DatetimeIndex) -> pd.DataFrame:
    bond_tickers = ["CETES28", "CETES91", "MBONO3Y", "MBONO5Y", "MBONO10Y", "CORP1", "CORP2"]
    records = []
    rng = np.random.default_rng(26)
    # Base YTM per bond type with AR persistence
    base_ytm = {
        "CETES28": 0.055, "CETES91": 0.058,
        "MBONO3Y": 0.075, "MBONO5Y": 0.080, "MBONO10Y": 0.085,
        "CORP1": 0.095, "CORP2": 0.105,
    }
    ytm_state = dict(base_ytm)
    credit_base = {"CETES28": 0.0, "CETES91": 0.0, "MBONO3Y": 0.0, "MBONO5Y": 0.0, "MBONO10Y": 0.0, "CORP1": 0.015, "CORP2": 0.022}
    for date in dates:
        for ticker in bond_tickers:
            # Durations (fixed contract characteristics)
            if "CETES28" in ticker:
                duration = 28 / 365
                maturity = duration
            elif "CETES91" in ticker:
                duration = 91 / 365
                maturity = duration
            elif "MBONO3Y" in ticker:
                duration = 2.7
                maturity = 3.0
            elif "MBONO5Y" in ticker:
                duration = 4.3
                maturity = 5.0
            elif "MBONO10Y" in ticker:
                duration = 7.5
                maturity = 10.0
            elif ticker == "CORP1":
                duration = 3.5
                maturity = 4.0
            else:  # CORP2
                duration = 5.5
                maturity = 7.0

            # AR(1) YTM random walk
            ytm_state[ticker] = float(np.clip(
                ytm_state[ticker] + rng.normal(0.0, 0.002), 0.02, 0.18
            ))
            ytm = ytm_state[ticker]
            credit_spread = float(np.clip(
                credit_base[ticker] + rng.normal(0.0, 0.001), 0.0, 0.06
            ))

            # Coupon rate set at par (at issuance the bond was priced at par)
            coupon_rate = max(ytm - credit_spread - 0.005, 0.01)  # slight below-YTM coupon
            price = _bond_price(ytm, coupon_rate, maturity)

            records.append({
                "date": date,
                "ticker": ticker,
                "asset_class": "fixed_income",
                "price": price,
                "ytm": ytm,
                "duration": duration,
                "credit_spread": credit_spread,
            })
    return pd.DataFrame.from_records(records)


def build_mock_macro_series(start_date: str = "2018-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="ME")
    np.random.seed(9)
    
    # Compute us_fed_rate: 5.25 through 2024-Q2, then -25bps per quarter, floor at 4.0
    us_fed_rate = []
    cutoff_date = pd.Timestamp("2024-06-30")  # End of Q2 2024
    for date in dates:
        if date <= cutoff_date:
            rate = 5.25
        else:
            # Quarters since 2024-Q3
            quarters_elapsed = (date.year - 2024) * 4 + (date.quarter - 3)
            rate = 5.25 - (quarters_elapsed * 0.0025)
            rate = max(rate, 4.0)
        us_fed_rate.append(rate)
    
    macro = pd.DataFrame(
        {
            "date": dates,
            "IMAI": np.clip(100 + np.cumsum(np.random.normal(0.15, 0.9, len(dates))), 80, None),
            "industrial_production_yoy": np.random.normal(0.04, 0.03, len(dates)),
            "exports_yoy": np.random.normal(0.06, 0.05, len(dates)),
            "usd_mxn": np.clip(19.5 + np.cumsum(np.random.normal(0.01, 0.1, len(dates))), 17.0, None),
            "banxico_rate": np.clip(4.0 + np.cumsum(np.random.normal(0.02, 0.1, len(dates))), 4.0, 12.0),
            "inflation_yoy": np.clip(0.03 + np.random.normal(0.0, 0.01, len(dates)), 0.02, 0.09),
            "us_ip_yoy": np.random.normal(0.03, 0.03, len(dates)),
            "us_fed_rate": us_fed_rate,
        }
    )
    return macro


def load_mock_data() -> Dict[str, pd.DataFrame]:
    universe = get_investable_universe()
    tickers = universe.loc[universe["investable"], "ticker"].tolist()
    prices = generate_mock_price_series(tickers)
    fundamentals = build_mock_fundamentals(tickers, pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    fibra_tickers = universe.loc[universe["asset_class"] == "fibra", "ticker"].tolist()
    fibra_fundamentals = build_mock_fibra_fundamentals(fibra_tickers, pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    bonds = build_mock_bonds(pd.date_range(prices.index[0], prices.index[-1], freq="ME"))
    macro = build_mock_macro_series(prices.index[0].strftime("%Y-%m-%d"), prices.index[-1].strftime("%Y-%m-%d"))
    return {
        "universe": universe,
        "prices": prices,
        "fundamentals": fundamentals,
        "fibra_fundamentals": fibra_fundamentals,
        "bonds": bonds,
        "macro": macro,
    }
