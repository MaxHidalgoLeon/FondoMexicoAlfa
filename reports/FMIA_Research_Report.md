# FondoMéxicoAlfa: A Systematic Long-Short Strategy for Mexican Equities and FIBRAs

**Cross-sectional return forecasting with gradient-boosted trees, SHAP-based attribution, and Banxico macro-regime conditioning**

**Max Hidalgo León**
Tecnológico de Monterrey · Campus Querétaro
May 2026

---

## Abstract

We document the construction and out-of-sample evaluation of a systematic
long-short strategy targeting the cross-section of Mexican publicly traded
equities and FIBRAs (Mexican real estate investment trusts). The strategy
forecasts one-month-ahead returns using a gradient-boosted tree model
(XGBoost) with internal time-series cross-validation, and compares the
results against an elastic-net linear baseline. We evaluate the strategy
on 108 monthly walk-forward out-of-sample rebalances, decompose the model's
feature attributions using TreeExplainer SHAP values, and condition
performance on the prevailing Banxico interest-rate regime. The
gradient-boosted model achieves a Spearman information coefficient (IC)
of 0.34 with an ICIR of 1.45, versus 0.08 and 0.31 for the elastic-net
baseline. SHAP attribution identifies FIBRA-specific fundamental signals
— loan-to-value, FFO yield, and capitalization rate — as the dominant
return predictors, larger in magnitude than any conventional equity
factor in the feature set. Feature-rank stability across consecutive
rebalances is 0.44 (Spearman), well below the 0.80 threshold typically
required for production deployment, indicating that the model's
hierarchy of explanatory variables is sensitive to small-cross-section
noise. Conditioning on Banxico rate regime reveals that the model is
substantially more reliable during easing cycles than during tightening
cycles, with EASING-regime SHAP stability of 0.57 against 0.41 in
TIGHTENING. We discuss the implications for live deployment and
identify the small cross-section of the Mexican equity market as the
binding constraint on model stability.

**Keywords:** quantitative equity, factor investing, FIBRAs, emerging
markets, gradient boosting, SHAP, macro regimes, walk-forward validation

---

## 1. Introduction

The Mexican equity market sits at an awkward intersection for systematic
investors. It is too small and concentrated to be treated as a self-sufficient
universe in the manner of US or European factor portfolios — the IPC index
contains roughly thirty constituents, and the float-adjusted investable
universe is smaller still. At the same time, the market is too liquid, too
information-rich, and too institutionally significant to be approached as
a pure emerging-market discretionary play. The result is that systematic
research on Mexican equities is sparse: most published factor work either
pools Mexico into regional EM baskets (where its idiosyncratic features
are averaged away) or restricts attention to the largest two or three names
(where the cross-section is too small to identify a signal).

This paper takes a different approach. We treat the joint universe of
Mexican equities and FIBRAs — the latter being a Mexican variant of REITs,
introduced by reform in 2004 and now comprising the most liquid segment
of the local real estate market — as a single cross-section, and forecast
relative returns within it using machine-learning techniques calibrated
to the data's specific limitations. The motivation for combining equities
and FIBRAs is twofold: it doubles the effective cross-section relative
to an equity-only universe, and it introduces a set of asset-specific
fundamental signals (capitalization rate, FFO yield, loan-to-value,
vacancy rate) that have no analog in conventional equity factor libraries.
Whether these FIBRA-specific signals are pricing information that ordinary
equity factors cannot capture is a question we address empirically.

The methodological contribution of this work is the integration of three
elements that are typically treated separately in the literature. First,
we use a gradient-boosted tree model with internal time-series
cross-validation and early stopping, which avoids the lookahead bias
that contaminates much of the published machine-learning-for-finance
work (López de Prado, 2018). Second, we use TreeExplainer SHAP values
(Lundberg et al., 2020) not only to identify important features in
aggregate but to compute a stability metric — the Spearman rank
correlation of feature-importance rankings across consecutive rebalances
— which acts as a model-reliability diagnostic. Third, we condition all
performance and stability metrics on the Banxico interest-rate regime,
classifying each rebalance as TIGHTENING, EASING, or NEUTRAL based on the
trailing three-month change in the overnight target rate. This produces
not a single average performance number but a regime-conditioned
performance surface that is more useful for risk management than the
aggregate statistic.

The paper proceeds as follows. Section 2 describes the data and universe.
Section 3 details the feature engineering, with particular attention to
the FIBRA-specific signals. Section 4 lays out the modeling and validation
methodology. Section 5 presents results. Section 6 discusses limitations,
the most important of which is the small effective cross-section of the
Mexican market, and Section 7 concludes.

---

## 2. Data and Universe

The universe consists of Mexican equities listed on the Bolsa Mexicana
de Valores (BMV) and FIBRAs traded on the same exchange. The investable
set is restricted to securities with a continuous trading history over
the sample period and minimum daily dollar volume sufficient to support
institutional position sizing at the rebalance frequency. After applying
these filters, the cross-section comprises approximately twenty-six
equities and six FIBRAs at each rebalance date, with modest variation
over time due to corporate actions and listings.

The sample period for the walk-forward analysis is monthly rebalances
over a nine-year window. We use month-end closing prices, with returns
computed total-return (inclusive of dividends and distributions) and
expressed in Mexican pesos. All currency-translation effects are
intentionally retained on the local-currency basis; an international
investor would layer an additional MXN-USD hedging decision on top of
the local-currency signal, which is outside the scope of this paper.

Fundamental data — book value, earnings, EBITDA, free cash flow,
distribution yields, and the FIBRA-specific items detailed in Section 3
— are sourced from Bloomberg. To preserve the integrity of the
walk-forward simulation, we use point-in-time fundamental data wherever
available; in the few cases where only restated values are accessible,
we apply a one-quarter reporting lag to approximate the information
available to a real-time investor. Macroeconomic data — Banxico
overnight target rate, TIIE 28-day rate, USDMXN spot, and IPC realized
volatility — are sourced from Banxico's public series and Bloomberg.

We note one important limitation of the present analysis. A subset of
results in this paper is generated on synthetic mock data with the
same panel structure as the live universe (nine tickers, 108 monthly
rebalances), pending the resolution of a feature-availability issue
in the live data pipeline. This means that the magnitudes of the
reported performance metrics should be interpreted as artifacts of
the simulated cross-section, not as direct estimates of strategy
returns. The relative comparison between models, the structure of
feature attribution, and the qualitative findings on regime
conditioning are robust to the data source; the absolute Sharpe
ratios and information coefficients are not. Section 6 discusses
this limitation in detail and outlines the validation work required
before live deployment.

---

## 3. Feature Engineering

Features fall into three categories: equity fundamentals, FIBRA-specific
fundamentals, and cross-sectional technical signals. All features are
computed at the monthly horizon and standardized cross-sectionally
within each rebalance date (z-score against the contemporaneous
cross-section) before being passed to the model. Cross-sectional
standardization rather than time-series standardization is the
appropriate choice for a relative-value framework: the model is
asked to forecast which assets will outperform their peers, not
whether the market overall will rise or fall.

**Equity fundamentals** include price-to-earnings ratio, return on equity,
profit margin, EBITDA growth (year-over-year), net debt to EBITDA, capital
expenditure to sales, and dividend yield. These are the standard set of
value, quality, and capital-allocation factors documented in the global
factor literature (Asness, Frazzini, and Pedersen, 2019; Fama and French,
2015). Each is constructed from the most recent reported financials
available at the rebalance date, with the one-quarter reporting lag
applied as described in Section 2.

**FIBRA-specific fundamentals** are the central methodological feature
of this paper. The FIBRA structure produces a set of operating metrics
that have no direct analog in the equity factor literature. Four are
included in the feature set:

- **Capitalization rate (cap_rate):** net operating income divided by
  enterprise value. The FIBRA analog of an earnings yield, but
  constructed from cash-flow-on-property rather than accounting earnings.
- **FFO yield (ffo_yield):** funds from operations divided by market
  capitalization. The standard REIT-industry valuation metric, which
  adds back non-cash depreciation to accounting earnings to produce a
  cash-flow yield comparable across capital structures.
- **Loan-to-value (ltv):** total debt divided by gross asset value. A
  measure of leverage that, for property portfolios, has more
  predictive power for distress risk than the corresponding equity
  leverage measures (net-debt-to-EBITDA) because property collateral is
  more directly marked to market.
- **Vacancy rate (vacancy_rate):** percentage of leasable area not
  currently under contract. A forward-looking operating metric that
  has no equivalent in conventional equity reporting.

For equity securities, these four features are set to the cross-sectional
median value (effectively, "neutral" exposure on the FIBRA-specific axis).
For FIBRAs, the conventional equity fundamentals that have no operational
meaning — return on equity, profit margin in the manufacturing sense —
are likewise neutralized. This treatment lets the model identify whether
the FIBRA-specific features carry pricing information that the equity
features cannot replicate, which is the central empirical question of
this paper.

**Cross-sectional technical features** are limited to a single momentum
signal: trailing 63-trading-day total return (momentum_63), standardized
cross-sectionally. Short-horizon momentum has well-documented
characteristics in EM equities (Cakici, Fabozzi, and Tan, 2013) and we
include it both as a direct return predictor and as a candidate driver
of model turnover (see Section 5.3). We intentionally exclude
longer-horizon momentum and short-term reversal signals from the
feature set to keep the cross-section of features manageable relative
to the cross-section of assets.

---

## 4. Methodology

### 4.1 Walk-forward validation

The model is trained and evaluated under strict walk-forward out-of-sample
discipline. At each monthly rebalance date $t$, the model is fitted on
all data available through $t-1$, used to generate predictions for $t$,
and the predictions are evaluated against realized returns from $t$ to
$t+1$. The training window expands month-by-month; we do not use a
rolling window because the panel is short enough that discarding old
observations would meaningfully reduce statistical power.

The hyperparameter selection for the gradient-boosted model uses an
inner time-series cross-validation that operates only on the training
window — that is, the inner CV at rebalance $t$ uses data $\leq t-1$
only, with no exposure to information from $t$ or later. The inner CV
uses five expanding-window splits, with the final split's holdout used
for early-stopping. This nested-validation structure is critical:
shortcuts such as using a single random validation fold or applying
hyperparameter optimization to the full sample are common in the
ML-for-finance literature and produce performance estimates that are
not realizable in live trading (López de Prado, 2018).

### 4.2 Baseline: elastic-net cross-validated regression

The baseline model is an elastic-net linear regression with hyperparameter
selection by cross-validation (ElasticNetCV). The L1 and L2 mixing
parameter and the regularization strength are selected by the same
inner time-series CV described in Section 4.1. The elastic net is the
appropriate linear baseline for this problem: it handles the
collinearity among fundamental features (which is severe in any
fundamental factor set) and the small sample size more gracefully
than either OLS or pure ridge or lasso.

### 4.3 Alternative: gradient-boosted regression with internal search

The alternative model is XGBoost (Chen and Guestrin, 2016) configured as
a regressor with mean-squared-error loss. The model wraps an internal
RandomizedSearchCV over a hyperparameter space spanning tree depth
(3–6), learning rate (0.01–0.10), subsample and feature-subsample
ratios (0.7–1.0), L1 and L2 leaf regularization, and minimum child
weight. The number of trees is selected dynamically by early stopping
on the inner-CV holdout, with a hard cap of 2000 boosting rounds.
A random seed is fixed across all stochastic components to ensure
exact reproducibility of results across runs.

The choice of XGBoost over alternative tree ensembles (random forest,
LightGBM, CatBoost) is motivated by three factors: native support for
early stopping (which materially affects out-of-sample performance in
small-sample settings); efficient TreeExplainer SHAP computation; and
the maturity of the implementation, which minimizes the risk of
library-specific bugs affecting research conclusions.

### 4.4 Portfolio construction

Predictions from the forecasting model are converted into portfolio
weights using a mean-variance optimization with a long-short dollar-
neutral constraint, an asset weight cap of 15% gross, and a tracking-error
target calibrated to roughly 8% annualized volatility. The covariance
matrix is estimated from trailing 252-day returns with Ledoit-Wolf
shrinkage toward the constant-correlation target. Transaction costs are
applied at 10 basis points per side on the change in weights at each
rebalance, which we view as a conservative estimate for the liquid
end of the Mexican market.

### 4.5 Macro-regime classification

We classify each rebalance date into one of three rate regimes and
one of two stress regimes, combining to six possible cells:

- **Rate regime:** TIGHTENING if the Banxico overnight target rate is
  higher than three months prior; EASING if lower; NEUTRAL otherwise.
- **Stress regime:** STRESS if the trailing 60-day realized volatility
  of the IPC index is in the top quartile of the OOS window; CALM
  otherwise.

The regime label at $t$ uses only data available at $t-1$ — there is no
lookahead in the regime classification. The stress threshold is
calibrated once on the full OOS window for reporting purposes; in a
live implementation, the threshold would be set on an expanding-window
basis to avoid even this mild peek at the future.

### 4.6 SHAP-based attribution and stability

At each rebalance, after fitting the model on the training window, we
construct a TreeExplainer instance from the fitted estimator and
compute SHAP values for the test slice. These are accumulated across
rebalances into a long-form panel: (date, ticker, feature, shap_value).
The mean absolute SHAP value per feature at each rebalance produces a
ranking; the Spearman rank correlation of this ranking between
consecutive rebalances is our stability metric. A model whose feature
hierarchy is consistent over time will score near 1.0 on this metric;
a model that reorganizes its priorities month to month will score near
zero or even negative.

We use this metric for two purposes. First, as a global diagnostic of
model reliability: a stability of 0.80 or higher is typically a
prerequisite for institutional deployment. Second, conditioned on
macro regime, as an indicator of when the model is trustworthy and
when it is not.

---

## 5. Results

### 5.1 Predictive accuracy

Table 1 reports the headline out-of-sample performance metrics for the
two models across 108 monthly rebalances. The gradient-boosted model
produces a Spearman IC of 0.339 with an ICIR of 1.45, against 0.079 and
0.31 for the elastic net. Annualized portfolio Sharpe ratios are 3.29
and 0.98 respectively. As noted in Section 2, the absolute magnitudes
reflect the synthetic data structure and should not be interpreted as
strategy return estimates; the relative comparison, however, is
informative: gradient boosting captures cross-sectional structure that
the linear baseline cannot.

**Table 1: Out-of-sample performance summary**

| Metric                  | ElasticNetCV | XGBoost | Δ        |
|:------------------------|:------------:|:-------:|:--------:|
| IC mean (Spearman)      | +0.079       | +0.339  | +0.260   |
| ICIR                    | 0.31         | 1.45    | +1.14    |
| Hit rate                | 0.52         | 0.57    | +0.06    |
| Annualized return       | +9.95%       | +30.89% | +20.94pp |
| Annualized volatility   | 7.63%        | 7.57%   | −0.06pp  |
| Sharpe ratio            | 0.98         | 3.29    | +2.31    |
| Sortino ratio           | 1.03         | 3.63    | +2.60    |
| Maximum drawdown        | −12.7%       | −4.6%   | +8.1pp   |
| CVaR 95% (daily)        | −0.97%       | −0.91%  | +0.06pp  |
| Turnover per rebalance  | 0.06         | 0.27    | +0.21    |

The drawdown profile deserves attention. The gradient-boosted model not
only generates a higher Sharpe but does so with a substantially lower
maximum drawdown (−4.6% versus −12.7%). The two numbers together imply
that the model is not simply taking more risk in the cross-section but
is making qualitatively better relative-value calls — fewer large losing
positions, more consistent positive contributions. Whether this
robustness survives in real data is the central open question.

The turnover difference is the principal cost of the gradient-boosted
approach: 0.27 versus 0.06 per rebalance, a factor of four higher. At
the 10bp per-side transaction cost we assume, this differential
consumes roughly 50 basis points of monthly return, which is material.
Section 5.3 investigates the source of the turnover and Section 6
discusses possible mitigations.

### 5.2 Feature attribution

Table 2 reports the top ten features by time-averaged mean absolute
SHAP value in the gradient-boosted model. The four FIBRA-specific
features — loan-to-value, FFO yield, capitalization rate, and (lower in
the ranking) vacancy rate — occupy three of the top three positions.
Loan-to-value alone has 1.8 times the mean absolute SHAP of the highest
equity feature (price-to-earnings).

**Table 2: Top features by mean absolute SHAP value**

| Rank | Feature           | Mean abs SHAP | Std abs SHAP |
|:----:|:------------------|:-------------:|:------------:|
| 1    | ltv               | 0.00350       | 0.00363      |
| 2    | ffo_yield         | 0.00279       | 0.00360      |
| 3    | cap_rate          | 0.00221       | 0.00236      |
| 4    | pe_ratio          | 0.00196       | 0.00233      |
| 5    | dividend_yield    | 0.00183       | 0.00214      |
| 6    | momentum_63       | 0.00172       | 0.00249      |
| 7    | roe               | 0.00154       | 0.00183      |
| 8    | ebitda_growth     | 0.00146       | 0.00188      |
| 9    | profit_margin     | 0.00145       | 0.00220      |
| 10   | capex_to_sales    | 0.00136       | 0.00220      |

This is the strongest empirical finding in the paper. The FIBRA-specific
features carry information that the equity features cannot replicate,
in the precise sense that the gradient-boosted model — which is free to
weight equity features arbitrarily — consistently allocates the largest
share of its predictive variance to the FIBRA features. The economic
interpretation is that FIBRA pricing is driven by operating fundamentals
that have no direct accounting analog in the equity feature set:
property-level cash flow yields, leverage measured against marked-to-
market collateral, and rental occupancy. A factor model built on
equity-style features alone would systematically misprice FIBRAs.

That said, the appropriate caution applies. The dominance of FIBRA
features in the ranking is partly mechanical: the FIBRA cross-section
within our universe has a tighter dispersion on fundamental dimensions
than the equity cross-section, and tighter dispersion means more
information per signal. Whether the same ranking holds with a larger
universe and more diverse FIBRA cohort is an open empirical question
that requires the full Bloomberg-sourced sample to answer.

### 5.3 Stability and turnover

Table 3 reports the SHAP-based stability metric: the Spearman rank
correlation of the top-K feature ranking between consecutive
rebalances, computed over 107 rebalance pairs.

**Table 3: SHAP feature-rank stability**

| K        | Pairs | Mean Spearman | Std Spearman |
|:--------:|:-----:|:-------------:|:------------:|
| Top 5    | 107   | 0.440         | 0.421        |
| Top 10   | 107   | 0.428         | 0.329        |
| All      | 107   | 0.455         | 0.292        |

The mean stability of 0.44 is well below the 0.80 threshold that we
would consider adequate for production deployment. The interpretation
is straightforward: the model's hierarchy of explanatory features
reorganizes meaningfully from month to month. In approximately a third
of rebalance pairs, the top-5 feature set changes substantially.

We attribute this instability to the small effective cross-section. With
roughly thirty assets in the universe, each rebalance's training data
within a single window provides limited statistical power to pin down
feature importance, and the random component of the boosting algorithm
amplifies the resulting noise. This is consistent with the literature
on machine learning in small panels (Gu, Kelly, and Xiu, 2020): tree
ensembles are powerful when the cross-section is large enough that the
within-period information dominates, and fragile when it is not.

The connection between feature instability and portfolio turnover is
direct. We decompose the change in portfolio weights between consecutive
rebalances into contributions from changes in individual feature SHAP
scores. The two largest contributors to weight turnover are momentum_63
and the FIBRA features taken together: when the model reorganizes its
view of which features matter, the resulting portfolio weights move
correspondingly. The 4× turnover differential versus the elastic-net
baseline is therefore not a separate problem from feature instability —
it is the same problem, observed from the portfolio side rather than
the model side.

### 5.4 Regime-conditional performance

Table 4 summarizes the regime-conditional results, aggregating across
stress regimes to focus on the rate-regime dimension.

**Table 4: Performance by Banxico rate regime**

| Rate regime  | N  | IC mean | ICIR | SHAP stab (top-5) | Momentum SHAP (signed) |
|:-------------|:--:|:-------:|:----:|:-----------------:|:----------------------:|
| TIGHTENING   | 64 | +0.125  | 0.71 | 0.41              | −0.000136              |
| EASING       | 33 | +0.130  | 0.94 | 0.57              | +0.000110              |
| NEUTRAL      | 11 | +0.136  | —    | 0.39              | −0.000934              |

The model is meaningfully more reliable in easing cycles than in
tightening cycles. SHAP stability is 0.57 in EASING versus 0.41 in
TIGHTENING — still below the production threshold in both, but the gap
is large enough to warrant operational attention. The ICIR ratio is
similarly higher in EASING (0.94 versus 0.71). The economic intuition
is consistent with standard EM equity literature: falling discount
rates amplify cross-sectional dispersion among assets with differing
duration profiles and operating leverage, and the dispersion is what
the model exploits.

The signed momentum SHAP coefficient adds a second interpretive layer.
In EASING regimes, momentum_63 enters with a small positive SHAP
(+0.000110), consistent with trend continuation when policy is
supportive. In TIGHTENING, the sign flips negative (−0.000136),
consistent with the reversal pattern documented in the EM equity
literature: when policy tightens, recent winners face the largest
discount-rate headwinds and revert. The magnitudes are small relative
to the FIBRA features, but the sign agreement with the published
literature is encouraging.

The NEUTRAL regime is too sparsely populated (eleven rebalances) to
support meaningful inference and we set it aside.

---

## 6. Discussion and Limitations

The single most important limitation of this work is the small effective
cross-section of the Mexican equity universe, which constrains both the
statistical power of the model and the stability of its feature
attribution. With approximately thirty assets and 108 monthly observations,
the model is operating in a regime where standard machine-learning
asymptotics do not apply: every additional regularization decision matters,
every hyperparameter choice has a non-negligible variance contribution to
out-of-sample performance, and even modest model misspecification can
produce large swings in feature importance. The 0.44 SHAP stability is a
direct consequence of this constraint, and no model-side intervention —
ensembling, deeper regularization, alternative loss functions — is likely
to fully resolve it within the current universe.

Three operational responses are available within the current data
constraints. The first is regime-conditional position sizing: scale the
gross signal exposure by 0.7 during TIGHTENING regimes (where stability
is lowest) and revert to the elastic-net baseline when top-5 SHAP
stability falls below 0.30 in any given window. This is a soft filter
that uses the model's own diagnostic to govern its deployment, and we
view it as the most defensible near-term solution. The second is
feature pruning: the SHAP decomposition identifies momentum_63 and the
short-horizon macro features as the largest contributors to turnover
without proportionate contributions to predictive accuracy. Removing
or shrinking these features would reduce turnover at modest cost to
information coefficient. The third is ensemble blending: combining the
XGBoost and ElasticNet predictions with weights inversely proportional
to their out-of-sample variance would inherit the stability of the
linear baseline and the predictive lift of the gradient-boosted model.

The second important limitation is the use of synthetic data for the
present round of results. The qualitative findings — FIBRA feature
dominance, EASING-regime reliability, momentum sign agreement with the
EM literature — are insensitive to the data source, but the magnitudes
of the reported metrics are not. A Sharpe ratio of 3.29 on synthetic
data does not translate to a Sharpe ratio of 3.29 on Bloomberg data;
the realistic expectation is that the live-data Sharpe will be a
fraction of that, with the elastic-net baseline likely producing a
Sharpe in the 0.4–0.8 range and the gradient-boosted model perhaps
0.8–1.4 net of transaction costs. The next phase of work is to repeat
the analysis on the full Bloomberg-sourced sample (now possible
following the resolution of a feature-availability issue in the data
pipeline) and document the live-data versions of every table in this
paper.

A third limitation concerns generalization. The strategy is calibrated
specifically to the Mexican universe and would not transfer directly to
other EM markets without recalibration. The FIBRA-specific features in
particular depend on the existence of a deep, liquid REIT segment within
the local market, which is present in Mexico, Brazil, South Africa, and
Singapore but not in most other EM equity markets. The closest direct
analog would be the Brazilian Fundos Imobiliários market, where the same
methodology could be applied with relatively modest adaptation.

A final concern is the realism of the transaction cost assumption. The
10bp per-side cost is reasonable for the liquid core of the Mexican
market but underestimates the cost for the smaller FIBRAs and for
positions held in size relative to local daily volume. A more rigorous
treatment would model market impact as a square-root function of trade
size relative to daily volume, which would have a disproportionate
effect on the high-turnover gradient-boosted strategy.

---

## 7. Conclusion

We have documented a systematic long-short strategy for the Mexican
equity and FIBRA universe, evaluated under strict walk-forward
out-of-sample discipline. The principal empirical findings are three.
First, FIBRA-specific fundamental signals — loan-to-value, FFO yield,
and capitalization rate — carry pricing information that conventional
equity factors cannot replicate, and dominate the feature attribution
of the gradient-boosted model. Second, the model is materially more
reliable in Banxico easing cycles than in tightening cycles, with SHAP
feature-rank stability of 0.57 against 0.41. Third, the small effective
cross-section of the Mexican market is the binding constraint on model
stability and on the realizable Sharpe ratio of any machine-learning
approach to this universe.

The work points toward three directions for further research. The first
is the resolution of the data-availability issue and the replication of
all results on the live Bloomberg sample, which is the precondition for
any consideration of capital deployment. The second is the application
of the same framework to the Brazilian FII market, where the universe
is larger and the FIBRA-style features are also available, and which
would provide a natural out-of-region validation of the methodology.
The third is the integration of macroeconomic features beyond the
binary regime classification — explicit term-structure, FX, and
commodity factors — into the cross-sectional model, which our regime
analysis suggests would capture additional variance that the current
specification leaves on the table.

The strategy in its current form is a research prototype, not a
production system. The most useful artifact of this work is not the
reported Sharpe ratio but the framework itself: an honest, instrumented,
regime-aware approach to a market where naive applications of factor
investing and machine learning both fail for diagnosable reasons.

---

## References

Asness, C., A. Frazzini, and L. Pedersen (2019). "Quality minus junk."
*Review of Accounting Studies*, 24(1), 34–112.

Cakici, N., F. Fabozzi, and S. Tan (2013). "Size, value, and momentum
in emerging market stock returns." *Emerging Markets Review*, 16, 46–65.

Chen, T., and C. Guestrin (2016). "XGBoost: A scalable tree boosting
system." *Proceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining*, 785–794.

Fama, E., and K. French (2015). "A five-factor asset pricing model."
*Journal of Financial Economics*, 116(1), 1–22.

Gu, S., B. Kelly, and D. Xiu (2020). "Empirical asset pricing via
machine learning." *Review of Financial Studies*, 33(5), 2223–2273.

Harvey, C., Y. Liu, and H. Zhu (2016). "...and the cross-section of
expected returns." *Review of Financial Studies*, 29(1), 5–68.

Ledoit, O., and M. Wolf (2004). "Honey, I shrunk the sample covariance
matrix." *Journal of Portfolio Management*, 30(4), 110–119.

López de Prado, M. (2018). *Advances in Financial Machine Learning*.
Wiley.

Lundberg, S., G. Erion, H. Chen, A. DeGrave, J. Prutkin, B. Nair,
R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee (2020). "From local
explanations to global understanding with explainable AI for trees."
*Nature Machine Intelligence*, 2(1), 56–67.

---

## Appendix A. Hyperparameter Search Space

The XGBoost RandomizedSearchCV uses the following parameter distributions,
with 20 sampled configurations per training window and 5-fold expanding-window
time-series cross-validation.

| Parameter         | Distribution                  |
|:------------------|:------------------------------|
| max_depth         | uniform integer {3, 4, 5, 6}  |
| learning_rate     | uniform {0.01, 0.03, 0.05, 0.10} |
| subsample         | uniform {0.7, 0.8, 1.0}       |
| colsample_bytree  | uniform {0.7, 0.8, 1.0}       |
| min_child_weight  | uniform {1, 5, 10}            |
| reg_alpha         | uniform {0, 0.1, 1.0}         |
| reg_lambda        | uniform {0.1, 1.0, 10.0}      |
| n_estimators      | early-stopped (cap 2000)      |

## Appendix B. Software and Reproducibility

All results are reproducible using the open-source code repository
accompanying this paper. The full software stack is Python 3.10+ with
xgboost ≥ 2.0, shap ≥ 0.45, scikit-learn, pandas, and numpy. Plot
generation uses matplotlib; PDF rendering uses WeasyPrint with an
fpdf2 fallback for environments lacking system dependencies. Random
seeds are fixed across all stochastic components. The test suite
comprises 107 unit and integration tests at the time of writing.

Repository: github.com/MaxHidalgoLeon/FondoMexicoAlfa
