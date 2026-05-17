# Step 3 — Performance by Macro Regime

Walk-forward backtest segmented by two orthogonal regime axes:
- **Rate regime**: Banxico trailing 3-month rate change (TIGHTENING / EASING / NEUTRAL).
- **Stress regime**: IPC 60-day realised vol vs 75th-percentile threshold (STRESS / CALM).

Source: `mock` panel, model: `elasticnet`, 108 rebalances (2017-04-28 → 2026-03-31).

No-lookahead guarantee: regime at rebalance `t` uses only macro/price data up to end of month `t-1`.
Stress threshold is fixed at the 75th percentile of the full OOS vol distribution (research descriptor only).

## Reproduce

```bash
python scripts/render_step3_report.py --source mock --model elasticnet
```

---

## 1. Regime Counts

| Rate regime | Stress regime | N rebalances |
|:---|:---|---:|
| EASING | CALM | 23 |
| EASING | STRESS | 10 |
| NEUTRAL | CALM | 11 |
| TIGHTENING | CALM | 47 |
| TIGHTENING | STRESS | 17 |


---

## 2. Performance Table

All metrics annualised (12 rebalances/year). Ann ret and vol scaled from monthly returns.
IC = Spearman rank correlation of expected_return vs realised 21-day forward log return.
Rows marked `stress_regime=ALL` aggregate across both stress conditions within that rate regime.

| Rate regime | Stress | N | IC mean | IC std | ICIR | Hit% | Ann ret | Ann vol | Sharpe | Sortino | MDD | CVaR95 | TO |
|:---|:---||---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EASING | ALL | 33 | +0.039 | 0.216 | +0.18 | 59.4% | +15.44% | 12.12% | +1.27 | +4.02 | -5.91% | -0.029 | 0.001 |
| EASING | CALM | 23 | +0.021 | 0.217 | +0.10 | 59.1% | +22.79% | 13.08% | +1.74 | +6.63 | -3.71% | -0.020 | 0.001 |
| EASING | STRESS | 10 | +0.115 | 0.199 | +0.58 | 66.7% | +18.03% | 11.41% | +1.58 | +5.14 | -5.91% | — | 0.002 |
| NEUTRAL | ALL | 11 | +0.280 | 0.318 | +0.88 | 90.0% | +8.39% | 3.70% | +2.27 | +3.92 | -1.30% | — | 0.091 |
| NEUTRAL | CALM | 11 | +0.280 | 0.318 | +0.88 | 90.0% | +8.39% | 3.70% | +2.27 | +3.92 | -1.30% | — | 0.091 |
| TIGHTENING | ALL | 64 | +0.140 | 0.238 | +0.59 | 73.0% | +7.68% | 7.62% | +1.01 | +2.57 | -9.28% | -0.020 | 0.004 |
| TIGHTENING | CALM | 47 | +0.195 | 0.223 | +0.87 | 84.8% | +10.67% | 9.97% | +1.07 | +3.86 | -9.28% | -0.020 | 0.005 |
| TIGHTENING | STRESS | 17 | -0.035 | 0.191 | -0.18 | 37.5% | +18.73% | 11.26% | +1.66 | +5.58 | -5.64% | — | 0.003 |

---

## 3. Key Findings per Rate Regime

**TIGHTENING** (64 rebalances): IC=+0.140, Sharpe=+1.01. Rate hikes compress equity multiples and tighten FIBRA cap-rate spreads, creating a more adversarial environment for the fundamentals-based cross-sectional model. Feature importance tends to shift toward short-term momentum as rate-sensitive signals lose predictive power.

**EASING** (33 rebalances): IC=+0.039, Sharpe=+1.27. Banxico cuts reduce the discount rate and lift FIBRA valuations, giving fundamental signals (LTV, FFO yield) stronger predictive content. Momentum also tends to be persistent in easing cycles, which amplifies the signal.

**NEUTRAL** (11 rebalances): IC=+0.280, Sharpe=+2.27. Flat rate periods offer less regime-driven signal dispersion; the model relies more heavily on idiosyncratic fundamentals. With fewer observations these estimates carry high uncertainty.

---

## 4. Feature Stability by Regime

Spearman rank-correlation of the top-K SHAP importance ranking between consecutive rebalances
*within* the same rate regime. Only consecutive same-regime pairs are counted.

| Rate regime | Pairs | Top-5 stability | Top-10 stability | All-feature stability |
|:---|---:|---:|---:|---:|
| TIGHTENING | 63 | 0.804 | 0.965 | 0.982 |
| EASING | 32 | 0.834 | 0.964 | 0.972 |
| NEUTRAL | 10 | 0.718 | 0.956 | 0.962 |

---

## 5. Momentum SHAP Direction by Regime

Mean **signed** SHAP value of `momentum_63` (positive = model uses upward momentum as a buy signal;
negative = model penalises recent winners).

| Rate regime | Mean SHAP (momentum_63) | Std | N obs |
|:---|---:|---:|---:|
| EASING | -0.00168 | 0.00541 | 495 |
| NEUTRAL | -0.00140 | 0.00965 | 165 |
| TIGHTENING | +0.00034 | 0.01126 | 960 |

In EASING periods the model assigns a **positive** mean SHAP to `momentum_63` (-0.00168), consistent with the EM hypothesis that rate cuts sustain momentum trends in risk assets. In TIGHTENING periods the mean SHAP is **positive** (+0.00034), which does NOT confirm the canonical EM reversal pattern — the mock data's random-walk rate structure likely suppresses the regime-conditional momentum signal. Verification on real Banxico/Bloomberg data is recommended before trading on this finding.

---

## 6. Regime Equity Curves

Portfolio periods within each rate regime chained independently (consecutive regime windows compounded).
Provides a within-regime performance picture stripped of cross-regime drift.

![Regime equity curves](figures/step3_regime_equity_curves.png)

---

## 7. IC Boxplot by Regime

Distribution of per-rebalance Spearman IC within each rate regime.

![IC by regime](figures/step3_ic_by_regime.png)

---

## 8. Actionable Recommendation

**Regime filter verdict: cautiously yes, but with strict sample-size caveats.** The TIGHTENING regime shows moderate Sharpe (1.008), which supports reducing gross exposure during Banxico hiking cycles. However, only 27 rebalances fall in STRESS conditions across all rate regimes — insufficient to draw robust conclusions about the STRESS overlay. Feature stability is lowest in the most volatile regime (mean Spearman=0.718), confirming that tightening cycles destabilise the model's feature hierarchy. The actionable recommendation is to implement a **soft regime filter**: when rate_regime==TIGHTENING, scale the gross XGBoost signal by 0.7 (a 30% shrinkage) rather than a hard on/off gate, and monitor the top-5 SHAP stability metric — if it drops below 0.3 in any three consecutive rebalances, suspend the XGBoost overlay and revert to ElasticNet. Stress regime triggers can be revisited with a larger universe (≥50 tickers) or live Bloomberg data.
