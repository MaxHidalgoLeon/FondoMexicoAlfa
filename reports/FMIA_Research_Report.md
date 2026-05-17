# FondoMéxicoAlfa

## A Systematic Equity Strategy for Mexican Equities and FIBRAs

*Primary results for the CNBV-regulated long-only portfolio;
long-short and 130/30 variants reported on an analytical basis in Section 5.5*

*Multi-provider data infrastructure, Black–Litterman portfolio construction,
machine-learning attribution, and macro-regime conditioning*

**Maximiliano Hidalgo León**
Tecnológico de Monterrey · Campus Querétaro
May 2026

---

## Abstract

We document the construction and out-of-sample evaluation of a systematic equity
strategy for the joint universe of Mexican publicly traded equities and FIBRAs (the
Mexican variant of REITs), with primary results reported for the CNBV-regulated
long-only portfolio. The strategy is built on a multi-provider data infrastructure
(Bloomberg, Refinitiv, Yahoo Finance), a Black–Litterman portfolio construction layer
that combines machine-learning views with low-confidence macro tilts, mean-variance
and CVaR optimizers operating under CNBV regulatory constraints, a Layer 2 FX hedge
overlay, and a complete suite of overfitting diagnostics including deflated Sharpe
ratios and combinatorially symmetric cross-validation. We extend the pipeline with an
XGBoost cross-sectional return forecaster, TreeExplainer SHAP attribution, and Banxico
macro-regime conditioning. Evaluated on 108 monthly walk-forward out-of-sample
rebalances from 2017 through early 2026 with Bloomberg point-in-time fundamentals,
the regulated portfolio achieves an annualized Sharpe ratio of 0.44, an annualized
return of 8.34% at 13.59% volatility, and a maximum drawdown of -35.5%. ElasticNetCV
and XGBoost produce statistically indistinguishable performance on real data; the
value of the gradient-boosted model lies in its attribution framework rather than
raw return lift. SHAP analysis identifies FIBRA-specific operating metrics —
loan-to-value, FFO yield, capitalization rate — as the dominant return predictors,
with stability of 0.44 (Spearman) across consecutive rebalances. Regime conditioning
reveals materially higher reliability during Banxico easing cycles than during
tightening cycles. We argue that the binding constraint on machine-learning approaches
to this market is the small effective cross-section of the Mexican universe, and we
discuss the implications for live deployment.

**Keywords:** quantitative equity, factor investing, FIBRAs, emerging markets,
Black–Litterman, gradient boosting, SHAP, macro regimes, walk-forward validation,
CNBV regulation.

---

## 1. Introduction

The systematic investment literature has converged on a relatively uniform methodology
for developed equity markets: factor portfolios constructed from large cross-sections,
ranked by composite scores spanning value, quality, momentum, and low-volatility
signals, and reweighted by mean-variance or risk-parity optimization. The
methodology works because the cross-section is large enough — typically several
thousand stocks in the US and several hundred in major European markets — for the
law of large numbers to overwhelm the noise in individual signals. When the same
methodology is applied to emerging markets, results typically degrade in proportion
to the cross-section size: regional pooling helps, but the per-country signal is
fragile.

The Mexican equity market is an extreme case of this problem. The IPC index contains
roughly thirty constituents; the float-adjusted investable universe is somewhat smaller.
At the same time, the local fixed-income market is deep, the regulatory environment
is sophisticated (CNBV operates a position-concentration regime modeled on European
UCITS), and the FIBRA market — Mexico's variant of REITs, introduced by reform in
2004 — has matured into a meaningful component of the listed real-asset universe.
Mexico is therefore too small to be approached with standard developed-market methods
and too institutionally complex to be approached as a pure emerging-market
discretionary play. The published systematic literature on the country is sparse.

This paper documents a complete systematic framework calibrated to these specific
constraints. The contributions are three. First, the framework integrates a set of
FIBRA-specific fundamental features — loan-to-value, FFO yield, capitalization rate,
vacancy rate — into a unified cross-sectional model alongside equity fundamentals
and price-based signals. These features have no direct analog in standard equity
factor libraries, and they are constructed point-in-time from Bloomberg fundamentals
with a 90-day reporting lag. Second, the framework layers a Black–Litterman
posterior over the cross-sectional model output: per-ticker return views from
ElasticNetCV or XGBoost are combined with low-confidence (0.20) macro sector views
derived from industrial production, exports, the Banxico overnight target rate, the
USDMXN exchange rate, and US inflation. The Black–Litterman step is essential in a
small cross-section because it controls the variance of the forecast inputs that
feed the optimizer; without it, the mean-variance solution becomes unstable across
rebalances. Third, the framework includes a complete suite of overfitting and model-
reliability diagnostics: SHAP-based feature stability metrics, Banxico regime
conditioning, deflated Sharpe ratios, and combinatorially symmetric cross-validation
following Bailey and López de Prado (2014, 2016). The diagnostics are not appendices
to the main result; they are the result, in the sense that they tell the operator
when the model is trustworthy and when it is not.

The paper proceeds as follows. Section 2 describes the data infrastructure and
universe. Section 3 details the feature engineering. Section 4 lays out the modeling
methodology. Section 5 presents results. Section 6 discusses what we view as the
most important methodological finding of the work — that XGBoost does not
materially outperform a well-regularized linear baseline in this universe — and the
practical implications for live deployment. Section 7 concludes.

---

## 2. Data Infrastructure and Universe

The data layer integrates four providers with automatic fallback. Bloomberg via
BLPAPI is the production source, providing point-in-time fundamental data with full
historical coverage from 2017 through early 2026. Refinitiv/LSEG is an institutional
alternative with somewhat shorter coverage for several constituents. Yahoo Finance
is available without credentials but provides only a snapshot of current
fundamentals; for the backtest window, only price-based signals (momentum, realized
volatility, liquidity) can be constructed from this source. Macro data — Banxico
overnight target rate, TIIE 28-day rate, USDMXN, US industrial production, US
inflation — is sourced from Banxico SIE and FRED.

The pipeline operates under a `strict_data_mode` flag set to true in the production
configuration. When this flag is enabled, the pipeline aborts on any data failure
rather than substituting synthetic or default values. This is the appropriate
behavior for a research and pre-production system where silent data substitution is
worse than an explicit failure. All fundamental data is lagged by 90 calendar days
to approximate the information set available to a real-time investor; restated
historical values that would not have been available in real time are excluded.

The investable universe consists of Mexican equities listed on the Bolsa Mexicana de
Valores (BMV) and FIBRAs traded on the same exchange. The cross-section comprises
approximately twenty-six equities and six FIBRAs at each rebalance date, with
modest variation due to corporate actions. The backtest period is monthly rebalances
from January 2017 through March 2026 — 108 rebalance dates after warm-up. Returns are
total-return, inclusive of dividends and distributions, expressed in Mexican pesos.

Several CNBV regulatory constraints are imposed at every rebalance. Individual
positions are capped at 10% of NAV, with a consolidated 10% issuer concentration
limit applied across tickers sharing the same issuer ID. A liquidity sleeve of
CETES28 and CETES91 instruments occupies 3% to 15% of NAV depending on the
prevailing macroeconomic regime, expanding in stress periods. An optional MBONO3Y
buffer of up to 3% is available but disabled in the production configuration. These
constraints are fixed structural parameters, not variables in the hyperparameter
search.

---

## 3. Feature Engineering

Three categories of features feed the cross-sectional model. Equity fundamentals
include the price-to-earnings ratio, return on equity, EBITDA growth, profit margin,
net debt to EBITDA, capital expenditure to sales, and dividend yield. These are the
standard set of value, quality, and capital-allocation factors that the global factor
literature has documented (Asness, Frazzini, and Pedersen, 2019; Fama and French,
2015). They are constructed from the most recent reported financials available at
the rebalance date with the 90-day lag applied.

FIBRA-specific fundamentals are the central methodological contribution of the
feature layer. The FIBRA structure produces a set of operating metrics that have
no direct analog in equity reporting and that are routinely used by real-asset
specialists to assess relative value. We include four. The capitalization rate is
net operating income divided by enterprise value — the FIBRA analog of an earnings
yield, but constructed from property-level cash flow rather than accounting earnings.
The FFO yield is funds from operations divided by market capitalization, the
standard REIT-industry valuation metric that adds back non-cash depreciation to
produce a cash-flow yield comparable across capital structures. Loan-to-value is
total debt divided by gross asset value, a measure of leverage that for property
portfolios has substantially more predictive power for distress risk than the
corresponding accounting leverage measures, because property collateral is more
directly marked to market than non-real-estate assets. Vacancy rate is the
percentage of leasable area not currently under contract — a forward-looking
operating metric that has no equivalent in conventional equity reporting.

For equity securities, the four FIBRA-specific features are set to the cross-sectional
median (effectively, a neutral exposure). For FIBRAs, equity-specific features that
lack operating meaning — return on equity in the manufacturing sense, profit margin
on a non-property base — are likewise neutralized. This treatment lets the model
identify, empirically, whether the FIBRA features carry pricing information that
the equity features cannot replicate; this is the central question of the SHAP
analysis in Section 5.2.

Cross-sectional technical features are limited to a 63-trading-day momentum signal
and a 63-trading-day realized volatility signal. Short-horizon momentum has well-
documented characteristics in emerging-market equities (Cakici, Fabozzi, and Tan,
2013). All features are standardized cross-sectionally within each rebalance date
(z-score against the contemporaneous cross-section) before entering the model.
Cross-sectional rather than time-series standardization is the appropriate choice
for a relative-value framework: the model is asked to forecast which assets will
outperform their peers, not whether the overall market will rise or fall.

---

## 4. Methodology

### 4.1 Walk-forward validation

The pipeline is trained and evaluated under strict walk-forward out-of-sample
discipline. At each monthly rebalance date *t*, the cross-sectional model is fitted
on data available through *t*−1, used to generate forecasts for *t*, and the
forecasts are evaluated against realized returns from *t* to *t*+1. The training
window expands monthly; we do not use a rolling window because the panel is short
enough that discarding old observations would meaningfully reduce statistical power.

Hyperparameter selection for the gradient-boosted model uses an inner time-series
cross-validation that operates only on the training window, with five expanding-
window splits and early stopping on the final fold's holdout. The inner CV at
rebalance *t* uses data through *t*−1 only, with no exposure to information from
*t* or later. Nested validation discipline of this kind is essential: shortcuts
such as a single random validation fold or hyperparameter optimization over the
full sample are common in the machine-learning-for-finance literature and produce
performance estimates that are not realizable in live trading (López de Prado, 2018).

### 4.2 ElasticNetCV baseline and XGBoost alternative

The baseline cross-sectional model is an elastic-net linear regression with
hyperparameter selection by cross-validation. The L1/L2 mixing parameter and
regularization strength are selected by the inner time-series CV. The elastic net
is the appropriate linear baseline for this problem: it handles the collinearity
among fundamental features (which is severe in any fundamental factor set) and the
small sample size more gracefully than either ordinary least squares or pure ridge
and lasso variants.

The alternative is XGBoost (Chen and Guestrin, 2016) configured as a regressor with
mean-squared-error loss. The model wraps an internal RandomizedSearchCV over a
hyperparameter space spanning tree depth, learning rate, subsample and feature-
subsample ratios, L1 and L2 leaf regularization, and minimum child weight. The
number of trees is selected dynamically by early stopping on the inner-CV holdout,
with a hard cap of 2000 boosting rounds. Random seeds are fixed across all
stochastic components to ensure exact reproducibility.

### 4.3 Black–Litterman posterior

Forecasts from the cross-sectional model do not feed the portfolio optimizer
directly. They are instead converted into per-ticker views and combined with
macro sector views in a Black–Litterman posterior (Black and Litterman, 1992). The
per-ticker views are assigned confidence weights derived from the model's in-sample
fit quality. Macro sector views — derived from industrial production, exports, the
Banxico target rate, USDMXN momentum, and US inflation — are blended at low
confidence (0.20) specifically chosen so that macro information nudges rather than
dominates the quantitative signal. The posterior mean is what enters the optimizer
as expected returns.

The Black–Litterman step is essential in a small cross-section. Raw cross-sectional
forecasts have substantial variance across rebalances, and feeding them directly
into a mean-variance optimizer produces a portfolio that turns over excessively.
The BL posterior pulls the expected-return vector toward the equilibrium prior
weighted by view confidence, which materially stabilizes the optimizer output
without dampening the underlying signal.

### 4.4 Portfolio optimization

The expected-return vector from the BL posterior enters a portfolio optimizer that
operates under CNBV regulatory constraints. Three solvers are available: a sequential
least-squares mean-variance optimizer with a market-impact penalty, a min-CVaR
optimizer at 95%, and a Michaud robust solver that averages 100 mean-variance
solutions with bootstrap-perturbed expected returns. The covariance matrix is
estimated by EWMA with Ledoit-Wolf shrinkage toward the constant-correlation target
(Ledoit and Wolf, 2004). Transaction costs are applied at 10 basis points per side
on the change in weights at each rebalance.

The production configuration runs the mean-variance and min-CVaR optimizers in
parallel for diagnostic comparison; the regulatory NAV reported in this paper uses
the mean-variance solution.

### 4.5 Macro-regime classification

We classify each rebalance into one of three Banxico rate regimes and one of two
market stress regimes, combining to up to six cells. The rate regime is TIGHTENING
if the Banxico overnight target rate is higher than three months prior, EASING if
lower, NEUTRAL otherwise. The stress regime is STRESS if the 60-day realized
volatility of the IPC index is in the top quartile of the OOS window, CALM
otherwise. Regime labels at time *t* use only data available at *t*−1.

### 4.6 SHAP attribution and feature-rank stability

At each rebalance, after the XGBoost model is fitted, we construct a TreeExplainer
SHAP instance from the fitted estimator and compute SHAP values for the test slice
(Lundberg et al., 2020). These accumulate across rebalances into a panel indexed
by (date, ticker, feature, shap_value). The mean absolute SHAP value per feature
at each rebalance produces a feature ranking; the Spearman rank correlation of
this ranking between consecutive rebalances is our stability metric.

The stability metric serves two purposes. As a global diagnostic, it indicates
whether the model's hierarchy of explanatory variables is consistent over time —
a stability of 0.80 or higher is typically required for institutional deployment
of a tree-based signal. Conditioned on macro regime, it indicates whether the
model is more or less reliable in particular environments, which is operationally
useful for deciding when to defer to the simpler baseline.

### 4.7 Hyperparameter optimization

A separate Bayesian search over a broader set of pipeline parameters — Black–
Litterman risk aversion, mean-variance and CVaR optimizer parameters, EWMA
covariance lambda, and ElasticNet mixing ratios — is run with Optuna's TPE
sampler across 50 trials per data source. The objective is a turnover-penalized
Sharpe ratio evaluated under purged walk-forward cross-validation with a 21-day
gap between training and validation. The hyperopt results are reported alongside
deflated-Sharpe and combinatorially symmetric cross-validation diagnostics following
Bailey and López de Prado (2014, 2016).

---

## 5. Results

### 5.1 Performance: ElasticNetCV and XGBoost on real data

Table 1 reports the headline out-of-sample performance metrics for both models on
each of the three data sources. The figures are for the regulated NAV portfolio
under the mean-variance optimizer with CNBV constraints applied; transaction costs
are 10 basis points per side, hedge overlay excluded.

**Table 1. Out-of-sample performance, regulated NAV, January 2017 – March 2026.**

| Source | Model | Return | Vol | Sharpe | Sortino | Max DD | CVaR 95% | Turnover |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bloomberg | ElasticNetCV | 8.34% | 13.59% | 0.44 | 0.45 | −35.50% | −1.95% | 0.57% |
| Bloomberg | XGBoost      | 8.29% | 13.57% | 0.44 | 0.44 | −35.38% | −1.95% | 6.33% |
| Yahoo     | ElasticNetCV | 9.98% | 15.86% | 0.47 | 0.48 | −36.58% | −2.26% | 0.04% |
| Yahoo     | XGBoost      | 9.89% | 15.81% | 0.47 | 0.48 | −36.58% | −2.25% | 0.44% |
| Refinitiv | ElasticNetCV | 5.80% | 16.01% | 0.23 | 0.23 | −43.19% | −2.28% | 0.04% |
| Refinitiv | XGBoost      | 5.73% | 15.60% | 0.23 | 0.23 | −42.75% | −2.23% | 0.67% |

Two findings stand out. First, within each data source, ElasticNetCV and XGBoost
produce statistically indistinguishable performance. The Sharpe ratios are identical
to two decimal places across all three providers; returns differ by less than five
basis points annualized; drawdowns differ by less than one percentage point. The
95% confidence interval on the Bloomberg Sharpe estimate (paired stationary
bootstrap, 5000 replications) is [−0.25, 1.18], which dwarfs any difference between
the two models. There is no meaningful sense in which XGBoost outperforms the
linear baseline on this data. The only material difference between the two models
is turnover, where XGBoost is approximately ten times higher than ElasticNet — a
direct cost driver that, absent a corresponding return advantage, is a liability
rather than an asset.

Second, performance differs substantially across data providers. Bloomberg achieves
Sharpe 0.44 with point-in-time fundamentals and full historical coverage; Yahoo
Finance achieves marginally higher Sharpe (0.47) but using only price signals
(momentum and liquidity), because historical fundamental data is not available
through that channel; Refinitiv falls to Sharpe 0.23, attributable to coverage gaps
in the local-equity feed that compress cross-sectional dispersion. The Yahoo result
is not directly comparable to Bloomberg because the signal sets differ — the
counterintuitive finding that a smaller signal set produces a higher Sharpe in this
sample reflects the contribution of model-fit variance from the additional
fundamental features in a small cross-section, not a real informational advantage
of price-only signals.

### 5.2 SHAP feature attribution

Despite the absence of a performance lift from XGBoost, the SHAP attribution
framework produces interpretable and economically sensible results. Table 2
reports the top ten features by time-averaged mean absolute SHAP value across the
Bloomberg walk-forward sample.

**Table 2. Top features by time-averaged mean |SHAP| value.**

| Rank | Feature | Mean \|SHAP\| | Std \|SHAP\| |
|:---:|:---|:---:|:---:|
| 1  | ltv             | 0.00350 | 0.00363 |
| 2  | ffo_yield       | 0.00279 | 0.00360 |
| 3  | cap_rate        | 0.00221 | 0.00236 |
| 4  | pe_ratio        | 0.00196 | 0.00233 |
| 5  | dividend_yield  | 0.00183 | 0.00214 |
| 6  | momentum_63     | 0.00172 | 0.00249 |
| 7  | roe             | 0.00154 | 0.00183 |
| 8  | ebitda_growth   | 0.00146 | 0.00188 |
| 9  | profit_margin   | 0.00145 | 0.00220 |
| 10 | capex_to_sales  | 0.00136 | 0.00220 |

Three of the four FIBRA-specific features occupy the top three positions, with
loan-to-value alone showing a mean |SHAP| value 1.8 times larger than the highest
equity feature (price-to-earnings). The economic interpretation is that FIBRA pricing
is driven by operating metrics — property-level cash flow yields, leverage measured
against marked-to-market collateral — that have no accounting analog in the equity
feature set. A factor model built exclusively on equity-style features would
systematically misprice FIBRAs; combining the two universes is informationally
productive precisely because the FIBRA features add genuinely orthogonal signal.

The Spearman rank correlation of the feature ranking between consecutive rebalances
is reported in Table 3. The mean stability across 107 rebalance pairs is 0.440 for
the top five features and 0.428 for the top ten — well below the 0.80 threshold
typically required for production deployment. The interpretation is direct: the
model's hierarchy of explanatory features reorganizes meaningfully from month to
month.

**Table 3. SHAP feature-rank stability across consecutive rebalances.**

| K       | Pairs | Mean Spearman | Std Spearman |
|:---:|:---:|:---:|:---:|
| Top 5   | 107   | 0.440         | 0.421        |
| Top 10  | 107   | 0.428         | 0.329        |
| All     | 107   | 0.455         | 0.292        |

We attribute the instability to the small effective cross-section. With approximately
thirty assets in the universe, each rebalance's training data provides limited
statistical power to pin down feature importance, and the stochastic component of the
boosting algorithm amplifies the resulting noise. This is consistent with the
finding in Section 5.1 — XGBoost matches but does not exceed ElasticNet on raw
performance — and points to the same underlying constraint. Gu, Kelly, and Xiu
(2020) document that tree ensembles deliver meaningful performance lift over linear
baselines only when the cross-section is large enough that within-period information
dominates the noise in feature attribution; the Mexican universe is on the wrong
side of that threshold.

### 5.3 Regime-conditioned performance

Table 4 reports the regime-conditioned metrics, aggregated across stress regimes
to focus on the Banxico rate-regime dimension.

**Table 4. Performance metrics conditional on Banxico rate regime.**

| Rate regime | N  | IC mean | ICIR | SHAP stab (top-5) | Momentum SHAP (signed) |
|:---|:---:|:---:|:---:|:---:|:---:|
| TIGHTENING  | 64 | +0.125 | 0.71 | 0.41 | −0.000136 |
| EASING      | 33 | +0.130 | 0.94 | 0.57 | +0.000110 |
| NEUTRAL     | 11 | +0.136 | —    | 0.39 | −0.000934 |

*Momentum SHAP is the signed (not absolute) mean SHAP contribution of momentum_63 in each regime. Regime metrics computed on walk-forward OOS validation sample; see Table 1 for Bloomberg regulated NAV headline figures.*

The model is meaningfully more reliable in easing cycles than in tightening cycles.
SHAP stability is 0.57 in EASING versus 0.41 in TIGHTENING — neither value reaches
the production threshold, but the gap is large enough to warrant operational attention.
The ICIR is correspondingly higher in EASING (0.94 versus 0.71). The economic
intuition is consistent with standard emerging-market equity literature: falling
discount rates amplify cross-sectional dispersion among assets with differing
duration profiles and operating leverage, and the dispersion is what the
cross-sectional model exploits.

The signed momentum SHAP coefficient adds a second interpretive layer. In EASING
regimes, momentum_63 enters with a small positive SHAP contribution, consistent with
trend continuation when monetary policy is supportive. In TIGHTENING, the sign flips
negative, consistent with the reversal pattern documented in the emerging-market
literature: when policy tightens, recent winners face the largest discount-rate
headwinds and revert. The magnitudes are small relative to the FIBRA-feature
contributions, but the sign agreement with the published literature is supportive.

The NEUTRAL regime is too sparsely populated to support meaningful inference and
we set it aside.

### 5.4 Hyperparameter optimization and overfitting diagnostics

The Bayesian search over the broader pipeline parameter space (50 trials per source,
purged walk-forward cross-validation with 21-day gap) produces best validated Sharpe
estimates of 0.43 on Bloomberg, 0.57 on Yahoo, and 0.26 on Refinitiv. These are
consistent with the production runs reported in Section 5.1.

The deflated Sharpe ratio adjusts the observed Sharpe for skewness, kurtosis, and
the number of configurations tested (Bailey and López de Prado, 2014). Under the
null hypothesis of zero skill, the expected maximum Sharpe across 50 trials is 2.23
on the Bloomberg sample. The observed best validated Sharpe of 0.43 is below this
ceiling, which is the appropriate honest finding: the search does not produce
evidence of statistically distinguishable skill against the multiple-testing-adjusted
null. The PBO statistic via combinatorially symmetric cross-validation is moderate
on Bloomberg and low on Yahoo, indicating that the production parameter choices
are robust within the explored space.

These results are not a failure of the search; they are a calibration of expectations.
A Sharpe of 0.43 on a CNBV-regulated, low-turnover, transaction-cost-net portfolio
in a small emerging-market universe is a defensible outcome. The deflated-Sharpe
adjustment correctly prevents the operator from overstating the result, which is
the central function of the diagnostic.

### 5.5 Layer 2 analytical overlay and reform scenarios

For completeness, the Bloomberg run with the Layer 2 FX hedge overlay activated
achieves an annualized return of 45.7%, an annualized volatility of 23.7%, and a
Sharpe ratio of 1.50 on an analytical basis. These figures are not included in the
regulated NAV because the Layer 2 overlay operates with leverage and currency
positions that fall outside the CNBV-reportable scope; they are reported as a
reference for what the same signal stack could achieve under a less constrained
mandate.

The LFI reform scenario analysis compares the regulated structure against three
alternatives: 130/30 long-short, market-neutral, and 130/30 sector-neutral. The
130/30 structure on Bloomberg achieves a Sharpe of 1.82 versus 1.47 for the
regulated structure on the hedge basis. This is the relevant comparison for
policy-design discussions about Mexican fund structures rather than for
investment-strategy conclusions per se.

---

## 6. Discussion

The most important methodological finding of this work is that the XGBoost
cross-sectional forecaster does not materially outperform the elastic-net baseline
on real Mexican-market data. This is not a failure of the implementation — both
models are constructed with internal walk-forward cross-validation, both are fitted
under identical out-of-sample discipline, both are evaluated on the same rebalance
sequence — and it is not a surprising result given the constraints of the universe.
It is, instead, a useful negative finding that has direct implications for live
deployment and for the broader literature on machine learning in emerging-market
equities.

The mechanism is the small effective cross-section. Tree ensembles draw their
predictive advantage over linear models from the ability to learn non-linear
interactions among features without overfitting, but this advantage requires
sufficient within-period information for the algorithm to identify those
interactions reliably. With roughly thirty assets at each rebalance, the available
information per training window is below the threshold at which the algorithm can
extract reliable non-linear structure; the model defaults toward the same
relationships that the linear baseline already captures, plus added noise from the
boosting randomization. The 0.44 SHAP feature-rank stability is a direct measurement
of this noise. The 10× turnover differential without a corresponding return
advantage is its portfolio-level manifestation. Gu, Kelly, and Xiu (2020) document
this same pattern in cross-country studies: the machine-learning advantage scales
with cross-section size, and small universes do not benefit.

This finding does not mean the XGBoost component is without value. The attribution
framework — SHAP per rebalance, feature-rank stability, regime-conditioned
performance — is genuinely informative independent of whether XGBoost itself
generates excess return. The framework tells the operator which features the model
is weighting, how those weights move over time, and under what macroeconomic
conditions the model is reliable. These are operational diagnostics that the linear
baseline cannot produce, and they have value even when the linear baseline matches
the gradient-boosted model on raw performance.

Three operational responses follow from these findings. The first is regime-
conditional model selection: use the XGBoost forecasts during EASING regimes,
where SHAP stability reaches 0.57, and revert to the ElasticNet baseline during
TIGHTENING and NEUTRAL regimes. The regime classification is itself a one-period-
lagged signal with no lookahead risk. This is a soft filter that uses the model's
own diagnostic to govern its deployment. The second is feature pruning: the SHAP
decomposition identifies momentum_63 as a substantial contributor to turnover
without proportionate contribution to predictive accuracy. Removing or shrinking
short-horizon technical features would reduce turnover at modest cost to the
information coefficient. The third is ensemble blending: combining ElasticNet and
XGBoost predictions at weights inversely proportional to their out-of-sample
forecast variance would inherit the stability of the linear baseline and add
whatever marginal lift the gradient-boosted model can deliver in favorable regimes.

Several limitations deserve explicit acknowledgment. The strategy is calibrated
specifically to the Mexican universe and would not transfer to other emerging
markets without recalibration, particularly because the FIBRA-specific features
depend on the existence of a deep, liquid REIT segment within the local market.
The closest direct analog is the Brazilian Fundos Imobiliários market, where the
same methodology could be applied with relatively modest adaptation. A more
rigorous treatment of transaction costs would model market impact as a square-root
function of trade size relative to daily volume, which would have a disproportionate
effect on the high-turnover XGBoost configuration and would likely strengthen the
case for the linear baseline in any live deployment. Finally, the 95% confidence
interval on the Sharpe estimate is wide ([−0.25, 1.18] on Bloomberg) because the
nine-year backtest contains only one full Banxico tightening cycle and one easing
cycle; precise estimation of regime-conditional performance will require additional
years of data or a multi-country extension that pools across markets with analogous
regime structures.

---

## 7. Conclusion

We have documented a complete systematic framework for the Mexican equity and FIBRA
universe, evaluated under strict walk-forward out-of-sample discipline across three
data providers. The framework integrates multi-provider data infrastructure, a
Black–Litterman portfolio construction layer that blends machine-learning views
with low-confidence macro tilts, multiple optimizers operating under CNBV regulatory
constraints, a Layer 2 FX hedge overlay reported on an analytical basis, an XGBoost
cross-sectional forecaster with TreeExplainer SHAP attribution, Banxico macro-regime
conditioning, and a complete suite of overfitting diagnostics. The regulated
portfolio achieves a Sharpe ratio of 0.44 on Bloomberg point-in-time fundamentals
over the 2017–2026 window.

The principal empirical findings are three. First, FIBRA-specific operating metrics
— loan-to-value, FFO yield, capitalization rate — dominate the SHAP attribution
of the gradient-boosted model and carry pricing information that conventional equity
factors cannot replicate. Second, XGBoost and ElasticNetCV produce statistically
indistinguishable performance on real data within the Mexican cross-section,
attributable to the small effective universe; the value of the gradient-boosted
model lies in its attribution framework rather than in raw return lift. Third,
model reliability is regime-dependent: Banxico easing cycles produce SHAP feature-
rank stability of 0.57 against 0.41 in tightening cycles, with corresponding
differences in the information coefficient.

The work points toward three directions for further research. The first is an
out-of-region replication of the framework on the Brazilian Fundos Imobiliários
market, which has a larger and more diverse REIT cohort and would provide a
natural validation of the FIBRA-feature methodology. The second is a more granular
macroeconomic conditioning that moves beyond the binary regime classification to
explicit term-structure, FX, and commodity factors; the existing regime analysis
suggests there is variance to capture that the current binary specification leaves
on the table. The third, and most operationally important, is the construction of
an ensemble between the ElasticNet and XGBoost paths with regime-conditional
weights, exploiting the asymmetric reliability of the two models across the
macroeconomic cycle. The framework as constructed provides the diagnostic
infrastructure to make such an ensemble robust; the empirical work of fitting it
remains to be done.

The strategy in its current form is a research prototype, not a production system.
The most useful artifact of this work is not the reported Sharpe ratio but the
framework itself: an honest, instrumented, regime-aware approach to a market where
naive applications of factor investing and machine learning both fail for diagnosable
reasons.

---

## References

Asness, C., A. Frazzini, and L. Pedersen (2019). "Quality minus junk."
*Review of Accounting Studies*, 24(1), 34–112.

Bailey, D., and M. López de Prado (2014). "The deflated Sharpe ratio: Correcting
for selection bias, backtest overfitting, and non-normality." *Journal of Portfolio
Management*, 40(5), 94–107.

Bailey, D., J. Borwein, M. López de Prado, and Q. Zhu (2016). "The probability of
backtest overfitting." *Journal of Computational Finance*, 20(4), 39–69.

Black, F., and R. Litterman (1992). "Global portfolio optimization." *Financial
Analysts Journal*, 48(5), 28–43.

Cakici, N., F. Fabozzi, and S. Tan (2013). "Size, value, and momentum in emerging
market stock returns." *Emerging Markets Review*, 16, 46–65.

Chen, T., and C. Guestrin (2016). "XGBoost: A scalable tree boosting system."
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining*, 785–794.

Fama, E., and K. French (2015). "A five-factor asset pricing model." *Journal of
Financial Economics*, 116(1), 1–22.

Gu, S., B. Kelly, and D. Xiu (2020). "Empirical asset pricing via machine learning."
*Review of Financial Studies*, 33(5), 2223–2273.

Ledoit, O., and M. Wolf (2004). "Honey, I shrunk the sample covariance matrix."
*Journal of Portfolio Management*, 30(4), 110–119.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Lundberg, S., G. Erion, H. Chen, A. DeGrave, J. Prutkin, B. Nair, R. Katz, J.
Himmelfarb, N. Bansal, and S.-I. Lee (2020). "From local explanations to global
understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1),
56–67.

---

## Appendix A. Hyperparameter Search Space

The XGBoost internal RandomizedSearchCV samples 20 configurations from the following
distributions for each training window, with a 5-fold expanding-window TimeSeriesSplit
and early stopping (50 rounds) on the inner-CV holdout.

| Parameter         | Distribution                  |
|:---|:---|
| max_depth         | uniform integer {3, 4, 5, 6}  |
| learning_rate     | uniform {0.01, 0.03, 0.05, 0.10} |
| subsample         | uniform {0.7, 0.8, 1.0}       |
| colsample_bytree  | uniform {0.7, 0.8, 1.0}       |
| min_child_weight  | uniform {1, 5, 10}            |
| reg_alpha         | uniform {0, 0.1, 1.0}         |
| reg_lambda        | uniform {0.1, 1.0, 10.0}      |
| n_estimators      | early-stopped (cap 2000)      |

The outer Optuna TPE search operates over a broader pipeline parameter space —
Black–Litterman risk aversion, mean-variance and CVaR optimizer parameters, EWMA
covariance lambda, ElasticNet mixing ratios — for 50 trials per data source under
purged walk-forward cross-validation with a 21-day gap.

## Appendix B. Software and Reproducibility

All results are reproducible from the open-source repository accompanying this paper.
The software stack is Python 3.10 or higher with xgboost ≥ 2.0, shap ≥ 0.45,
scikit-learn, pandas, numpy, Optuna, and CVXPY (CVaR optimization). Matplotlib is
used for figure generation; the PDF tearsheet pipeline uses WeasyPrint with an
fpdf2 fallback for environments lacking the pango and gobject system libraries.
Random seeds are fixed across all stochastic components to ensure exact
reproducibility. The test suite comprises 107 unit and integration tests.

Repository: github.com/MaxHidalgoLeon/FondoMexicoAlfa
