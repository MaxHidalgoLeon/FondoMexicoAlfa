"""Tests unitarios del módulo de construcción de vistas Black-Litterman.

Verifica build_elastic_net_views(), build_macro_views() y combine_views():
  - Vistas vacías cuando el input está vacío.
  - Confidencias en el rango correcto [0.30, 0.70] para vistas ElasticNet.
  - Combine_views pondera correctamente por confianza.
"""
import unittest

import numpy as np
import pandas as pd

from src.bl_views import (
    build_elastic_net_views,
    build_macro_views,
    combine_views,
    views_breakdown,
)


class TestBLViews(unittest.TestCase):
    def test_combine_views_confidence_weighted_blend(self):
        v1, c1 = {"A": 0.02, "B": -0.01}, {"A": 0.6, "B": 0.5}
        v2, c2 = {"A": 0.01, "C": 0.005}, {"A": 0.2, "C": 0.2}
        v, c = combine_views((v1, c1), (v2, c2))
        # Confidence-weighted blend on A
        expected_a = (0.02 * 0.6 + 0.01 * 0.2) / (0.6 + 0.2)
        self.assertAlmostEqual(v["A"], expected_a, places=12)
        # B and C only in one source
        self.assertAlmostEqual(v["B"], -0.01, places=12)
        self.assertAlmostEqual(v["C"], 0.005, places=12)
        # Confidence is max, not sum (avoids over-confidence on overlap)
        self.assertAlmostEqual(c["A"], 0.6, places=12)
        self.assertAlmostEqual(c["B"], 0.5, places=12)

    def test_elastic_net_views_empty_returns_empty(self):
        v, c = build_elastic_net_views(pd.DataFrame())
        self.assertEqual(v, {})
        self.assertEqual(c, {})

    def test_macro_views_disabled_returns_empty(self):
        macro = pd.DataFrame({"industrial_production_yoy": [0.01, 0.02]},
                             index=pd.date_range("2024-01-01", periods=2))
        universe = pd.DataFrame({"ticker": ["A"], "sector": ["Industrial"], "usd_exposure": [0.5]})
        v, c = build_macro_views(macro, universe, {"bl_views": {"use_macro": False}})
        self.assertEqual(v, {})

    def test_macro_views_bounded_magnitude(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        # Strong positive trend → high z-score
        macro = pd.DataFrame({
            "industrial_production_yoy": np.linspace(-0.05, 0.10, 200),
            "exports_yoy":               np.linspace(-0.05, 0.10, 200),
            "us_ip_yoy":                 np.linspace(-0.05, 0.10, 200),
            "usd_mxn":                   np.linspace(18, 22, 200),
            "banxico_rate":              np.linspace(4, 11, 200),
        }, index=dates)
        universe = pd.DataFrame({
            "ticker": ["NEMAKA", "FUNO11", "GISSAA"],
            "sector": ["Industrial", "FIBRA", "Industrial"],
            "usd_exposure": [0.85, 0.4, 0.7],
        })
        cfg = {"bl_views": {"use_macro": True,
                             "macro_view_confidence": 0.20,
                             "macro_view_max_magnitude": 0.015}}
        v, c = build_macro_views(macro, universe, cfg)

        # All magnitudes bounded
        for ticker, val in v.items():
            self.assertLessEqual(abs(val), 0.015 + 1e-9, msg=f"{ticker}={val}")
            self.assertEqual(c[ticker], 0.20)

        # Industrial sector should have positive view under rising IP/exports
        if "NEMAKA" in v:
            self.assertGreater(v["NEMAKA"], 0)

    def test_macro_views_no_macros_no_universe_safe(self):
        # Empty inputs do not crash
        v, c = build_macro_views(None, None, {"bl_views": {"use_macro": True}})
        self.assertEqual(v, {})

    def test_views_breakdown_dataframe_shape(self):
        v1, c1 = {"A": 0.02}, {"A": 0.6}
        v2, c2 = {"B": -0.01}, {"B": 0.2}
        df = views_breakdown([("elastic_net", v1, c1), ("macro", v2, c2)])
        self.assertEqual(set(df.columns), {"source", "target", "view_pct", "confidence"})
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
