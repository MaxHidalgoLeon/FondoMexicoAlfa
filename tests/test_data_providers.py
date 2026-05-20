"""Unit tests for src/data_providers.py — provider factory and ticker mapping."""
from __future__ import annotations

import pytest

from src.data_providers import (
    MockDataProvider,
    YahooFinanceProvider,
    _resolve_symbols,
    get_provider,
)


def test_get_provider_mock():
    assert isinstance(get_provider("mock"), MockDataProvider)


def test_get_provider_yahoo_alias():
    assert isinstance(get_provider("yahoo"), YahooFinanceProvider)
    assert isinstance(get_provider("yfinance"), YahooFinanceProvider)


def test_get_provider_normalizes_case_and_whitespace():
    assert isinstance(get_provider("  Mock  "), MockDataProvider)


def test_get_provider_unknown_raises():
    with pytest.raises((ValueError, ImportError)):
        get_provider("does-not-exist")


def test_resolve_symbols_default_no_suffix():
    """No suffix and no map entry ⇒ the canonical ticker is its own provider symbol."""
    mapping = _resolve_symbols(["AMXL"], provider="mock", suffix="")
    assert "AMXL" in mapping.values()


def test_resolve_symbols_yahoo_index_passthrough():
    """Yahoo-style indices starting with ^ must not get the .MX suffix appended."""
    mapping = _resolve_symbols(["^MXX"], provider="yahoo", suffix=".MX")
    # Either no entry (if filtered) or kept as ^MXX without suffix.
    for sym in mapping.keys():
        assert not sym.endswith("^MXX.MX")


def test_mock_provider_get_prices_returns_dataframe():
    provider = MockDataProvider()
    df = provider.get_prices(["AMXL", "WALMEX"], "2021-01-01", "2021-04-01")
    assert df.shape[0] > 30  # at least three months of bdays
    assert "AMXL" in df.columns or "WALMEX" in df.columns
