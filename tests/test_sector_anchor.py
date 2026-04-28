import json
import os
import unittest

import numpy as np
import pandas as pd

from src.pipeline import (
    _ETF_TO_NORMAL_SECTOR_GROUPS,
    _etf_sector_weights_path,
    _load_etf_sector_targets,
    _persist_etf_sector_weights,
)
from src.portfolio import optimize_portfolio


def _make_universe():
    return pd.DataFrame({
        "ticker":      ["NEMAKA", "GISSAA", "FUNO11", "FIBRAPL14"],
        "sector":      ["Industrial", "Industrial", "FIBRA", "FIBRA"],
        "asset_class": ["equity", "equity", "fibra", "fibra"],
    })


class TestSectorAnchor(unittest.TestCase):
    def setUp(self):
        self.tag = "unit_test_anchor"
        self.path = _etf_sector_weights_path(self.tag)
        if os.path.exists(self.path):
            os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_persist_aggregates_etf_sectors_drops_government(self):
        weights = pd.DataFrame(0.0, index=pd.date_range("2025-01-01", periods=2),
                                columns=["INDUSTRIAL", "FIBRATC14", "CONSUMER", "CETES28"])
        weights.iloc[-1] = [0.55, 0.30, 0.10, 0.05]
        universe_etf = pd.DataFrame({
            "ticker": ["INDUSTRIAL", "FIBRATC14", "CONSUMER", "CETES28"],
            "sector": ["Industrial", "FIBRA", "Consumer", "Government"],
        })
        _persist_etf_sector_weights({"weights": weights}, universe_etf, self.tag)

        with open(self.path) as f:
            payload = json.load(f)
        sw = payload["sector_weights"]
        self.assertNotIn("Government", sw)
        self.assertAlmostEqual(sw["Industrial"], 0.55)
        self.assertAlmostEqual(sw["Consumer"], 0.10)

    def test_load_disabled_returns_none(self):
        cfg = {"etf_sector_anchor": {"enabled": False}}
        sc, payload = _load_etf_sector_targets(cfg, _make_universe())
        self.assertIsNone(sc)
        self.assertIsNone(payload)

    def test_load_missing_file_with_fallback(self):
        cfg = {"etf_sector_anchor": {
            "enabled": True, "source": "does_not_exist", "fallback_to_unanchored": True
        }}
        sc, _ = _load_etf_sector_targets(cfg, _make_universe())
        self.assertIsNone(sc)

    def test_load_aggregates_to_normal_groups(self):
        # Persist a synthetic ETF run
        weights = pd.DataFrame(0.0, index=pd.date_range("2025-01-01", periods=2),
                                columns=["INDUSTRIAL", "FIBRATC14", "CONSUMER"])
        weights.iloc[-1] = [0.55, 0.30, 0.15]
        universe_etf = pd.DataFrame({
            "ticker": ["INDUSTRIAL", "FIBRATC14", "CONSUMER"],
            "sector": ["Industrial", "FIBRA", "Consumer"],
        })
        _persist_etf_sector_weights({"weights": weights}, universe_etf, self.tag)

        cfg = {"etf_sector_anchor": {
            "enabled": True, "source": self.tag, "band": 0.10, "fallback_to_unanchored": True
        }}
        universe_normal = _make_universe()
        sc, payload = _load_etf_sector_targets(cfg, universe_normal)

        self.assertIsNotNone(sc)
        # Industrial bucket aggregates Industrial + Consumer = 0.70
        self.assertAlmostEqual(sc["Industrial"]["min"], 0.60, places=10)
        self.assertAlmostEqual(sc["Industrial"]["max"], 0.80, places=10)
        # FIBRA bucket = 0.30 only
        self.assertAlmostEqual(sc["FIBRA"]["min"], 0.20, places=10)
        self.assertAlmostEqual(sc["FIBRA"]["max"], 0.40, places=10)
        # Sector map covers all normal-mode tickers
        sm = sc["__sector_map__"]
        self.assertEqual(sm["NEMAKA"], "Industrial")
        self.assertEqual(sm["FUNO11"], "FIBRA")

    def test_optimizer_wide_band_matches_unanchored(self):
        # With a very wide band, the constraint should not bind, so the
        # solution should equal the unanchored optimum (within SLSQP tol).
        rng = np.random.default_rng(0)
        tickers = ["NEMAKA", "GISSAA", "FUNO11", "FIBRAPL14"]
        mu = pd.Series([0.08, 0.05, 0.06, 0.07], index=tickers)
        cov = pd.DataFrame(np.diag([0.04, 0.05, 0.03, 0.035]), index=tickers, columns=tickers)

        ac = {
            "__asset_class_map__": {"NEMAKA": "equity", "GISSAA": "equity",
                                       "FUNO11": "fibra", "FIBRAPL14": "fibra"},
            "equity": {"min": 0.40, "max": 0.95},
            "fibra":  {"min": 0.05, "max": 0.60},
        }
        w_free = optimize_portfolio(mu, cov, max_position=0.50, target_net_exposure=1.0,
                                     asset_class_constraints=ac)

        sc = {
            "__sector_map__": {"NEMAKA": "Industrial", "GISSAA": "Industrial",
                                "FUNO11": "FIBRA", "FIBRAPL14": "FIBRA"},
            "Industrial": {"min": 0.0, "max": 1.0},
            "FIBRA":      {"min": 0.0, "max": 1.0},
        }
        w_anchored = optimize_portfolio(mu, cov, max_position=0.50, target_net_exposure=1.0,
                                         asset_class_constraints=ac, sector_constraints=sc)

        # Wide-band solution should match unanchored (sum of |diff| < 1e-3)
        diff = float((w_free - w_anchored).abs().sum())
        self.assertLess(diff, 1e-3, msg=f"wide-band anchor diverged from free: diff={diff}")

    def test_optimizer_narrow_band_enforces_target(self):
        # Narrow band centered on a target forces sector sums into [t-δ, t+δ].
        tickers = ["NEMAKA", "GISSAA", "FUNO11", "FIBRAPL14"]
        mu = pd.Series([0.08, 0.05, 0.06, 0.07], index=tickers)
        cov = pd.DataFrame(np.diag([0.04, 0.05, 0.03, 0.035]), index=tickers, columns=tickers)

        ac = {
            "__asset_class_map__": {"NEMAKA": "equity", "GISSAA": "equity",
                                       "FUNO11": "fibra", "FIBRAPL14": "fibra"},
            "equity": {"min": 0.10, "max": 0.95},
            "fibra":  {"min": 0.05, "max": 0.90},
        }
        sc = {
            "__sector_map__": {"NEMAKA": "Industrial", "GISSAA": "Industrial",
                                "FUNO11": "FIBRA", "FIBRAPL14": "FIBRA"},
            "Industrial": {"min": 0.30, "max": 0.34},
            "FIBRA":      {"min": 0.66, "max": 0.70},
        }
        w = optimize_portfolio(mu, cov, max_position=0.50, target_net_exposure=1.0,
                                asset_class_constraints=ac, sector_constraints=sc)
        ind = w[["NEMAKA", "GISSAA"]].sum()
        fib = w[["FUNO11", "FIBRAPL14"]].sum()
        self.assertGreaterEqual(ind, 0.30 - 5e-3)
        self.assertLessEqual(ind, 0.34 + 5e-3)
        self.assertGreaterEqual(fib, 0.66 - 5e-3)
        self.assertLessEqual(fib, 0.70 + 5e-3)

    def test_etf_to_normal_mapping_exhaustive(self):
        # Sanity: Industrial bucket must include all multi-cap normal-mode equity sectors
        ind_group = _ETF_TO_NORMAL_SECTOR_GROUPS["Industrial"]
        for s in ("Industrial", "Logistics", "Infrastructure", "Materials"):
            self.assertIn(s, ind_group)
        self.assertIn("FIBRA", _ETF_TO_NORMAL_SECTOR_GROUPS["FIBRA"])


if __name__ == "__main__":
    unittest.main()
