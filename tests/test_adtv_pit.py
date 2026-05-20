"""PIT (point-in-time) invariant tests for compute_adtv_liquidity_scores.

The core invariant: the ADTV score for date t must depend ONLY on price/volume
data up to and including t — never on future observations.

This guards against the historical bug where compute_adtv_liquidity_scores
was called once with the full panel and the resulting (end-of-panel) Series
was reused at every rebalance date, retroactively letting future liquidity
information influence past optimization decisions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import (
    compute_adtv_liquidity_scores,
    compute_adtv_liquidity_scores_panel,
)


def _synthetic_panel(n_days: int = 800, n_tickers: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n_days)
    cols = [f"T{i}" for i in range(n_tickers)]
    # Prices: random walk
    log_ret = rng.normal(0.0003, 0.012, (n_days, n_tickers))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(log_ret, axis=0)), index=idx, columns=cols)
    # Volume: lognormal with a deliberate regime shift mid-panel so that
    # end-of-panel ADTV differs significantly from mid-panel ADTV.
    base_vol = np.exp(rng.normal(13.0, 0.4, (n_days, n_tickers)))
    base_vol[n_days // 2:, :] *= 5.0  # 5× volume from midpoint onward
    volume = pd.DataFrame(base_vol, index=idx, columns=cols)
    return prices, volume


def test_adtv_score_pit_invariant():
    """ADTV score at date t with full panel == score with panel truncated to t."""
    prices, volume = _synthetic_panel()
    test_dates = prices.index[::100][3:]  # every 100 bdays, skip first to ensure window filled

    failures = []
    for d in test_dates:
        full = compute_adtv_liquidity_scores(prices, volume, window=252, as_of_date=d)
        trunc = compute_adtv_liquidity_scores(
            prices.loc[prices.index <= d],
            volume.loc[volume.index <= d],
            window=252,
        )
        if not np.allclose(full.values, trunc.values, atol=1e-12):
            failures.append(f"  {d.date()}: max abs diff = {np.max(np.abs(full.values - trunc.values)):.2e}")

    assert not failures, (
        "PIT violation: compute_adtv_liquidity_scores(..., as_of_date=t) "
        "differs from the truncated-panel computation:\n" + "\n".join(failures)
    )


def test_adtv_score_no_future_leakage():
    """Score at date t must be unaffected by appending future volume rows."""
    prices, volume = _synthetic_panel(n_days=500)
    mid_date = prices.index[300]

    score_before = compute_adtv_liquidity_scores(prices, volume, window=252, as_of_date=mid_date)

    # Append rows after mid_date with extreme volume
    extra_idx = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=50)
    last_px = prices.iloc[-1].values
    extra_px = pd.DataFrame(
        np.tile(last_px, (len(extra_idx), 1)), index=extra_idx, columns=prices.columns
    ).astype(float)
    extra_vol = pd.DataFrame(1e12, index=extra_idx, columns=volume.columns)
    prices_extended = pd.concat([prices, extra_px])
    volume_extended = pd.concat([volume, extra_vol])

    score_after = compute_adtv_liquidity_scores(prices_extended, volume_extended, window=252, as_of_date=mid_date)
    assert np.allclose(score_before.values, score_after.values, atol=1e-12), (
        "Appending extreme future volume changed the ADTV score at a past date — "
        "as_of_date filter is not correctly truncating the panel."
    )


def test_adtv_panel_each_row_is_pit():
    """Every row in the PIT panel matches a point-wise computation at that date."""
    prices, volume = _synthetic_panel()
    panel_dates = prices.index[100:800:120]
    panel = compute_adtv_liquidity_scores_panel(
        prices, volume, panel_dates, window=252, method="uniform"
    )

    assert isinstance(panel, pd.DataFrame)
    assert panel.shape[0] == len(panel_dates)
    assert set(panel.columns) <= set(prices.columns)

    for d in panel_dates:
        expected = compute_adtv_liquidity_scores(prices, volume, window=252, as_of_date=d)
        np.testing.assert_allclose(
            panel.loc[d].reindex(expected.index).values, expected.values, atol=1e-12,
            err_msg=f"Panel row {d.date()} does not match point-wise PIT computation."
        )


def test_adtv_panel_changes_over_time():
    """Panel must show meaningful variation across dates (sanity check).

    If the panel were broken (e.g., all rows = end-of-panel), this test would
    catch it because the synthetic data has a 5× volume regime shift midway.
    """
    prices, volume = _synthetic_panel(n_days=800)
    early = prices.index[300]
    late = prices.index[700]
    panel = compute_adtv_liquidity_scores_panel(
        prices, volume, pd.DatetimeIndex([early, late]),
        window=252, method="uniform",
    )

    # The two rows should differ because ADTV captures the volume regime shift
    early_scores = panel.loc[early]
    late_scores = panel.loc[late]
    assert not np.allclose(early_scores.values, late_scores.values, atol=1e-6), (
        "PIT panel rows are identical — the panel is not actually time-varying."
    )
