---
title: "Fondo Mexico Inversión Alfa: Machine Learning Asset Selection in a Small-Cross-Section Emerging Market with Regulatory Constraints"
author: "Maximiliano Hidalgo León"
date: "May 2026"
---

# Fondo Mexico Inversión Alfa: Machine Learning Asset Selection in a Small-Cross-Section Emerging Market with Regulatory Constraints

**Author:** Maximiliano Hidalgo León
**Date:** May 2026

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
ratios and combinatorially symmetric cross-validation. We extend the pipeline with a
LightGBM cross-sectional return forecaster, TreeExplainer SHAP attribution, and Banxico
macro-regime conditioning. Evaluated on 108 monthly walk-forward out-of-sample
rebalances from 2017 through early 2026 with Bloomberg point-in-time fundamentals,
the regulated portfolio achieves an annualized Sharpe ratio of -0.18, an annualized
return of 5.25% at 14.56% volatility, and a maximum drawdown of -39.83%. ElasticNetCV
and LightGBM produce statistically indistinguishable performance on real data; the
value of the gradient-boosted model lies in its attribution framework rather than
raw return lift. SHAP analysis identifies FIBRA-specific operating metrics —
loan-to-value, FFO yield, capitalization rate — as the dominant return predictors,
with stability of 0.44 (Spearman) across consecutive rebalances. Regime conditioning
reveals higher signal quality (ICIR) during TIGHTENING cycles and higher attribution
stability (SHAP) during EASING cycles — a finding with direct operational implications
for regime-conditional model governance. We argue that the binding constraint on
machine-learning approaches to this market is the small effective cross-section of the
Mexican universe, and we discuss the implications for live deployment.

**Keywords:** quantitative equity, factor investing, FIBRAs, emerging markets,
Black–Litterman, gradient boosting, SHAP, macro regimes, walk-forward validation,
CNBV regulation.

---

## 1. Introduction

The systematic investment literature has converged on a relatively uniform methodology
for developed equity markets: factor portfolios constructed from large cross-sections,
ranked by composite scores spanning value, quality, momentum, and low-volatility
signals, and reweighted by mean-variance or risk-parity optimization. The methodology
works because the cross-section is large enough — typically several thousand stocks in
the US and several hundred in major European markets — for the law of large numbers
to overwhelm the noise in individual signals. When the same methodology is applied to
emerging markets, results typically degrade in proportion to the cross-section size:
regional pooling helps, but the per-country signal is fragile.

The Mexican equity market is an extreme case of this problem. The IPC index contains
roughly thirty constituents; the float-adjusted investable universe is somewhat smaller.
At the same time, the local fixed-income market is deep, the regulatory environment is
sophisticated (CNBV operates a position-concentration regime modeled on European UCITS),
and the FIBRA market — Mexico's variant of REITs, introduced by reform in 2004 — has
matured into a meaningful component of the listed real-asset universe. Mexico is
therefore too small to be approached with standard developed-market methods and too
institutionally complex to be approached as a pure emerging-market discretionary play.
The published systematic literature on the country is sparse.

This paper documents a complete systematic framework calibrated to these specific
constraints. The contributions are three. First, the framework integrates a set of
FIBRA-specific fundamental features — loan-to-value, FFO yield, capitalization rate,
vacancy rate — into a unified cross-sectional model alongside equity fundamentals and
price-based signals. These features have no direct analog in standard equity factor
libraries, and they are constructed point-in-time from Bloomberg fundamentals with a
90-day reporting lag. Second, the framework layers a Black–Litterman posterior over the
cross-sectional model output: per-ticker return views from ElasticNetCV or LightGBM are
combined with low-confidence (0.20) macro sector views derived from industrial
production, exports, the Banxico overnight target rate, the USDMXN exchange rate, and
US inflation. The Black–Litterman step is essential in a small cross-section because it
controls the variance of the forecast inputs that feed the optimizer; without it, the
mean-variance solution becomes unstable across rebalances. Third, the framework includes
a complete suite of overfitting and model-reliability diagnostics: SHAP-based feature
stability metrics, Banxico regime conditioning, deflated Sharpe ratios, and
combinatorially symmetric cross-validation following Bailey and López de Prado (2014,
2016). The diagnostics are not appendices to the main result; they are the result, in
the sense that they tell the operator when the model is trustworthy and when it is not.

The paper proceeds as follows. Section 2 describes the data infrastructure and universe.
Section 3 details the feature engineering. Section 4 lays out the modeling methodology.
Section 5 presents results. Section 6 discusses what we view as the most important
methodological finding of the work — that LightGBM does not materially outperform a
well-regularized linear baseline in this universe — and the practical implications for
live deployment. Section 7 concludes.

---

## 2. Data Infrastructure and Universe

The data layer integrates four providers with automatic fallback. Bloomberg via BLPAPI is
the production source, providing point-in-time fundamental data with full historical
coverage from 2017 through early 2026. Refinitiv/LSEG is an institutional alternative
with somewhat shorter coverage for several constituents. Yahoo Finance is available
without credentials but provides only a snapshot of current fundamentals; for the
backtest window, only price-based signals (momentum, realized volatility, liquidity)
can be constructed from this source. Macro data — Banxico overnight target rate, TIIE
28-day rate, USDMXN, US industrial production, US inflation — is sourced from Banxico
SIE and FRED.

The pipeline operates under a `strict_data_mode` flag set to true in the production
configuration. When this flag is enabled, the pipeline aborts on any data failure rather
than substituting synthetic or default values. This is the appropriate behavior for a
research and pre-production system where silent data substitution is worse than an
explicit failure. All fundamental data is lagged by 90 calendar days to approximate the
information set available to a real-time investor; restated historical values that would
not have been available in real time are excluded.

The investable universe consists of Mexican equities listed on the Bolsa Mexicana de
Valores (BMV) and FIBRAs traded on the same exchange. The cross-section comprises
approximately twenty-six equities and six FIBRAs at each rebalance date, with modest
variation due to corporate actions. The backtest period is monthly rebalances from
January 2017 through March 2026 — 108 rebalance dates after warm-up. Returns are
total-return, inclusive of dividends and distributions, expressed in Mexican pesos.

Several CNBV regulatory constraints are imposed at every rebalance. Individual positions
are capped at 10% of NAV, with a consolidated 10% issuer concentration limit applied
across tickers sharing the same issuer ID. A liquidity sleeve of CETES28 and CETES91
instruments occupies 3% to 15% of NAV depending on the prevailing macroeconomic regime,
expanding in stress periods (3–5% in expansion, 5–8% in tightening, 8–15% in stress).
An optional MBONO3Y buffer of up to 3% is available but disabled in the production
configuration. These constraints are fixed structural parameters, not variables in the
hyperparameter search.

---

## 3. Feature Engineering

Three categories of features feed the cross-sectional model. Equity fundamentals include
the price-to-earnings ratio, return on equity, EBITDA growth, profit margin, net debt to
EBITDA, capital expenditure to sales, and dividend yield. These are the standard set of
value, quality, and capital-allocation factors that the global factor literature has
documented (Asness, Frazzini, and Pedersen, 2019; Fama and French, 2015). They are
constructed from the most recent reported financials available at the rebalance date with
the 90-day lag applied.

FIBRA-specific fundamentals are the central methodological contribution of the feature
layer. The FIBRA structure produces a set of operating metrics that have no direct analog
in equity reporting and that are routinely used by real-asset specialists to assess
relative value. We include four. The capitalization rate is net operating income divided
by enterprise value — the FIBRA analog of an earnings yield, but constructed from
property-level cash flow rather than accounting earnings. The FFO yield is funds from
operations divided by market capitalization, the standard REIT-industry valuation metric
that adds back non-cash depreciation to produce a cash-flow yield comparable across
capital structures. Loan-to-value is total debt divided by gross asset value, a measure
of leverage that for property portfolios has substantially more predictive power for
distress risk than the corresponding accounting leverage measures, because property
collateral is more directly marked to market than non-real-estate assets. Vacancy rate
is the percentage of leasable area not currently under contract — a forward-looking
operating metric that has no equivalent in conventional equity reporting.

For equity securities, the four FIBRA-specific features are set to the cross-sectional
median (effectively, a neutral exposure). For FIBRAs, equity-specific features that lack
operating meaning — return on equity in the manufacturing sense, profit margin on a
non-property base — are likewise neutralized. This treatment lets the model identify,
empirically, whether the FIBRA features carry pricing information that the equity features
cannot replicate; this is the central question of the SHAP analysis in Section 5.2.

Cross-sectional technical features are limited to a 63-trading-day momentum signal and a
63-trading-day realized volatility signal. Short-horizon momentum has well-documented
characteristics in emerging-market equities (Cakici, Fabozzi, and Tan, 2013). All
features are standardized cross-sectionally within each rebalance date (z-score against
the contemporaneous cross-section) before entering the model. Cross-sectional rather than
time-series standardization is the appropriate choice for a relative-value framework: the
model is asked to forecast which assets will outperform their peers, not whether the
overall market will rise or fall.

---

## 4. Methodology

### 4.1 Walk-forward validation

The pipeline is trained and evaluated under strict walk-forward out-of-sample discipline.
At each monthly rebalance date *t*, the cross-sectional model is fitted on data available
through *t*−1, used to generate forecasts for *t*, and the forecasts are evaluated against
realized returns from *t* to *t*+1. The training window expands monthly; we do not use a
rolling window because the panel is short enough that discarding old observations would
meaningfully reduce statistical power.

Hyperparameter selection for the gradient-boosted model uses an inner time-series
cross-validation that operates only on the training window, with expanding-window splits
and early stopping on the final fold's holdout. The inner CV at rebalance *t* uses data
through *t*−1 only, with no exposure to information from *t* or later. Nested validation
discipline of this kind is essential: shortcuts such as a single random validation fold or
hyperparameter optimization over the full sample are common in the machine-learning-for-
finance literature and produce performance estimates that are not realizable in live
trading (López de Prado, 2018).

### 4.2 ElasticNetCV baseline and LightGBM alternative

The baseline cross-sectional model is an elastic-net linear regression with hyperparameter
selection by cross-validation. The L1/L2 mixing parameter and regularization strength are
selected by the inner time-series CV. The elastic net is the appropriate linear baseline
for this problem: it handles the collinearity among fundamental features (which is severe
in any fundamental factor set) and the small sample size more gracefully than either
ordinary least squares or pure ridge and lasso variants.

The alternative is LightGBM configured as a regressor with mean-squared-error loss. The
inner CV scoring metric is the Spearman information coefficient (`ic`), chosen to align
the model-selection criterion with the rank-IC objective of the cross-sectional signal.
The model wraps an internal RandomizedSearchCV (5 draws, 3 TimeSeriesSplit folds) over a
hyperparameter space spanning tree depth, learning rate, subsample and feature-subsample
ratios, L1 and L2 leaf regularization, and minimum child weight. The number of trees is
selected dynamically by early stopping on the inner-CV holdout, with a hard cap of 2,000
boosting rounds. The `n_jobs = 1` threading configuration is empirically optimal for the
small cross-section: OpenMP thread-synchronization overhead (psynch_cvwait) dominates
actual compute with ~30 assets, making serial execution 5–11× faster than multi-threaded.
Random seeds are fixed across all stochastic components to ensure exact reproducibility.

### 4.3 Black–Litterman posterior

Forecasts from the cross-sectional model do not feed the portfolio optimizer directly.
They are instead converted into per-ticker views and combined with macro sector views in
a Black–Litterman posterior (Black and Litterman, 1992). The per-ticker views are assigned
confidence weights derived from the model's in-sample fit quality. Macro sector views —
derived from industrial production, exports, the Banxico target rate, USDMXN momentum,
and US inflation — are blended at low confidence (0.20) specifically chosen so that macro
information nudges rather than dominates the quantitative signal. The posterior mean is
what enters the optimizer as expected returns.

The Black–Litterman step is essential in a small cross-section. Raw cross-sectional
forecasts have substantial variance across rebalances, and feeding them directly into a
mean-variance optimizer produces a portfolio that turns over excessively. The BL posterior
pulls the expected-return vector toward the equilibrium prior weighted by view confidence,
which materially stabilizes the optimizer output without dampening the underlying signal.

### 4.4 Portfolio optimization

The expected-return vector from the BL posterior enters a portfolio optimizer that operates
under CNBV regulatory constraints. Three solvers are available: a sequential least-squares
mean-variance optimizer with a market-impact penalty, a min-CVaR optimizer at 95% with a
504-day scenario window, and a Michaud robust solver that averages 100 mean-variance
solutions with bootstrap-perturbed expected returns. The covariance matrix is estimated
by EWMA with Ledoit-Wolf shrinkage toward the constant-correlation target (Ledoit and
Wolf, 2004), using decay λ = 0.94. Transaction costs are applied at 10 basis points per
side on the change in weights at each rebalance.

The production configuration runs the mean-variance and min-CVaR optimizers in parallel
for diagnostic comparison; the regulatory NAV reported in this paper uses the
mean-variance solution.

### 4.5 Macro-regime classification

We classify each rebalance into one of three Banxico rate regimes and one of two market
stress regimes, combining to up to six cells. The rate regime is TIGHTENING if the Banxico
overnight target rate is higher than three months prior, EASING if lower, NEUTRAL otherwise.
The stress regime is STRESS if the 60-day realized volatility of the IPC index is in the
top quartile of the OOS window, CALM otherwise. Regime labels at time *t* use only data
available at *t*−1. The regime score is smoothed by an EWMA with span 6 and a confidence
threshold of 0.30 required for a regime switch.

### 4.6 SHAP attribution and feature-rank stability

At each rebalance, after the LightGBM model is fitted, we construct a TreeExplainer SHAP
instance from the fitted estimator and compute SHAP values for the test slice (Lundberg
et al., 2020). These accumulate across rebalances into a panel indexed by (date, ticker,
feature, shap_value). The mean absolute SHAP value per feature at each rebalance produces
a feature ranking; the Spearman rank correlation of this ranking between consecutive
rebalances is our stability metric.

The stability metric serves two purposes. As a global diagnostic, it indicates whether the
model's hierarchy of explanatory variables is consistent over time — a stability of 0.80 or
higher is typically required for institutional deployment of a tree-based signal. Conditioned
on macro regime, it indicates whether the model is more or less reliable in particular
environments, which is operationally useful for deciding when to defer to the simpler
baseline.

### 4.7 Hyperparameter optimization

A separate Bayesian search over a broader set of pipeline parameters — Black–Litterman risk
aversion, mean-variance and CVaR optimizer parameters, EWMA covariance lambda, and
ElasticNet mixing ratios — is run with Optuna's TPE sampler across 25 trials per data
source. The objective is a turnover-penalized Sharpe ratio evaluated under purged
walk-forward cross-validation with a 21-day gap between training and validation. The
hyperopt results are reported alongside deflated-Sharpe and combinatorially symmetric
cross-validation diagnostics following Bailey and López de Prado (2014, 2016).

---

## 5. Results

### 5.1 Performance: ElasticNetCV and LightGBM on real data

Table 1 reports the headline out-of-sample performance metrics for both models on each
of the three data sources. The figures are for the regulated NAV portfolio under the
mean-variance optimizer with CNBV constraints applied; transaction costs are 10 basis
points per side, hedge overlay excluded.

**Table 1. Out-of-sample performance, regulated NAV, January 2017 – March 2026.**

| Source | Model | Return | Vol | Sharpe | Sortino | Max DD | CVaR 95% | Turnover |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bloomberg | ElasticNetCV | 5.25% | 14.56% | -0.18 | -0.18 | -39.83% | -2.11% | 3.48% |
| Bloomberg | LightGBM | 5.28% | 14.56% | -0.17 | -0.18 | -39.46% | -2.11% | 3.49% |
| Yahoo | ElasticNetCV | 3.32% | 15.49% | -0.31 | -0.31 | -43.37% | -2.29% | 1.01% |
| Yahoo | LightGBM | 3.39% | 15.40% | -0.31 | -0.31 | -43.11% | -2.27% | 1.02% |
| Refinitiv | ElasticNetCV | 2.63% | 16.59% | -0.33 | -0.33 | -48.64% | -2.44% | 0.98% |
| Refinitiv | LightGBM | 2.77% | 16.28% | -0.33 | -0.33 | -47.80% | -2.38% | 1.33% |

Two findings stand out. First, within each data source, ElasticNetCV and LightGBM produce
statistically indistinguishable performance. Although the Bloomberg point estimates differ
numerically (ElasticNetCV: −0.18 vs LightGBM: −0.17), the 95% confidence interval on the
Bloomberg ElasticNetCV Sharpe estimate (paired stationary bootstrap, 5,000 replications)
is [−0.849, 0.534] — a band of width 1.38 that dwarfs the numerical gap between the two
models. For Yahoo and Refinitiv, where each source's Sharpe rounds to −0.31 and −0.33
respectively, the two models are indistinguishable to two decimal places. There is no
statistically meaningful sense in which either model outperforms the other; both results
are consistent with zero skill at conventional significance levels on this universe.
On Bloomberg, ElasticNetCV and LightGBM now carry essentially identical monthly turnover
(3.48% vs 3.49%), converging from the divergence seen in earlier pipeline versions.

Second, performance differs substantially across data providers. Bloomberg achieves the
highest Sharpe (−0.18) because it provides point-in-time fundamental data with full
historical coverage; Yahoo Finance and Refinitiv achieve negative Sharpe (−0.31 and −0.33
respectively). For Yahoo Finance, only price-based signals are available because historical
point-in-time fundamental data is not accessible through that channel; the absence of
fundamental features constrains the signal set. For Refinitiv, coverage gaps in the
local-equity feed compress cross-sectional dispersion. The Bloomberg result, while
positive, is estimated with very wide uncertainty: the OOS period contains only one full
Banxico tightening cycle and one easing cycle, providing limited statistical power.

### 5.2 SHAP feature attribution

Despite the absence of a performance lift from LightGBM, the SHAP attribution framework
produces interpretable and economically sensible results. Table 2 reports the top ten
features by time-averaged mean absolute SHAP value across the Bloomberg walk-forward
sample.

**Table 2. Top features by time-averaged mean |SHAP| value (Bloomberg, LightGBM).**

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
loan-to-value alone showing a mean |SHAP| value 1.8 times larger than the highest equity
feature (price-to-earnings). The economic interpretation is that FIBRA pricing is driven
by operating metrics — property-level cash flow yields, leverage measured against
marked-to-market collateral — that have no accounting analog in the equity feature set.
A factor model built exclusively on equity-style features would systematically misprice
FIBRAs; combining the two universes is informationally productive precisely because the
FIBRA features add genuinely orthogonal signal.

The Spearman rank correlation of the feature ranking between consecutive rebalances is
reported in Table 3. The mean stability across 107 rebalance pairs is 0.440 for the top
five features and 0.428 for the top ten — well below the 0.80 threshold typically required
for production deployment. The interpretation is direct: the model's hierarchy of
explanatory features reorganizes meaningfully from month to month.

**Table 3. SHAP feature-rank stability across consecutive rebalances (Bloomberg, LightGBM).**

| K       | Pairs | Mean Spearman | Std Spearman |
|:---:|:---:|:---:|:---:|
| Top 5   | 107   | 0.440         | 0.421        |
| Top 10  | 107   | 0.428         | 0.329        |
| All     | 107   | 0.455         | 0.292        |

We attribute the instability to the small effective cross-section. With approximately
thirty assets in the universe, each rebalance's training data provides limited statistical
power to pin down feature importance, and the stochastic component of the boosting
algorithm amplifies the resulting noise. This is consistent with the finding in Section 5.1
— LightGBM matches but does not exceed ElasticNet on raw performance — and points to the
same underlying constraint. Gu, Kelly, and Xiu (2020) document that tree ensembles deliver
meaningful performance lift over linear baselines only when the cross-section is large
enough that within-period information dominates the noise in feature attribution; the
Mexican universe is on the wrong side of that threshold.

### 5.3 Regime-conditioned performance

Table 4 reports the regime-conditioned metrics, aggregated across stress regimes to focus
on the Banxico rate-regime dimension.

**Table 4. Performance metrics conditional on Banxico rate regime.**

| Rate regime | N  | IC mean | ICIR | SHAP stab (top-5) | SHAP stab (top-10) |
|:---|:---:|:---:|:---:|:---:|:---:|
| TIGHTENING  | 64 | +0.140  | 0.59 | 0.407 | 0.379 |
| EASING      | 33 | +0.039  | 0.18 | 0.566 | 0.492 |
| NEUTRAL     | 11 | +0.280  | 0.88 | 0.386 | 0.404 |

*Regime metrics computed on walk-forward OOS validation sample; see Table 1 for Bloomberg
regulated NAV headline figures. NEUTRAL regime has only 11 observations; treat those
statistics as indicative.*

The regime picture is more nuanced than a simple easing-vs-tightening ranking. On the
signal quality dimension (IC mean and ICIR), TIGHTENING periods are superior to EASING:
the mean information coefficient is 0.140 in TIGHTENING versus 0.039 in EASING, and the
ICIR (IC mean scaled by IC volatility) is 0.59 versus 0.18. This finding may appear
counterintuitive — falling discount rates would be expected to amplify cross-sectional
dispersion — but is consistent with the hypothesis that TIGHTENING cycles create more
differentiated fundamental trajectories across issuers, particularly between property
companies with different loan-to-value profiles facing higher refinancing costs.

On the attribution stability dimension (SHAP), the pattern reverses: EASING cycles produce
feature-rank stability of 0.57 versus 0.41 in TIGHTENING. Feature attribution is therefore
more reliable in easing environments even though the raw signal quality is lower. The
operational implication is that EASING periods offer a better environment for model
interpretation and diagnostic monitoring, while TIGHTENING periods offer better raw signal
predictability (as measured by IC) with less attribution clarity.

Neither TIGHTENING nor EASING reaches the 0.80 SHAP stability threshold required for
institutional deployment. The NEUTRAL regime shows the highest IC mean (0.280) and
ICIR (0.88), but with only 11 rebalance observations this result lacks statistical power
and should not be extrapolated.

### 5.4 Hyperparameter optimization and overfitting diagnostics

The Bayesian search over the broader pipeline parameter space (25 trials per source,
purged walk-forward cross-validation with 21-day gap) produces best validated Sharpe
estimates of 0.15 on Bloomberg, 0.22 on Yahoo, and 0.04 on Refinitiv. These are modest
values consistent with the full OOS performance reported in Section 5.1.

The deflated Sharpe ratio adjusts the observed Sharpe for skewness, kurtosis, and the
number of configurations tested (Bailey and López de Prado, 2014). Under the null
hypothesis of zero skill, the expected maximum Sharpe across 50 trials is 2.23 on the
Bloomberg sample (computed via the DSR formula of Bailey & López de Prado, 2014, applied
to 50 trials with the skewness and kurtosis of the Bloomberg return series). The observed
best validated Sharpe of 0.15 is substantially below this ceiling, which is the appropriate
honest finding: the search does not produce evidence of statistically distinguishable skill
against the multiple-testing-adjusted null. The PBO statistic via combinatorially symmetric
cross-validation is moderate on Bloomberg and low on Yahoo and Refinitiv, indicating that
the production parameter choices are robust within the explored space but do not deliver
strong evidence of out-of-sample generalization.

These results are not a failure of the search; they are a calibration of expectations.
A best-trial Sharpe of 0.15 on a CNBV-regulated, low-turnover, transaction-cost-net
portfolio in a small emerging-market universe is a defensible outcome. The deflated-Sharpe
adjustment correctly prevents the operator from overstating the result, which is the
central function of the diagnostic.

### 5.5 Layer 2 analytical overlay and reform scenarios

The Bloomberg results with the Layer 2 FX hedge overlay activated are reported on an
analytical basis only; they are not included in the CNBV-regulated NAV because the overlay
operates with leverage and currency positions that fall outside the CNBV-reportable scope.
Bloomberg ElasticNetCV with hedge: Sharpe −0.18, annualized return 3.25%, volatility 24.5%.
Bloomberg LightGBM with hedge: Sharpe 0.21, annualized return 13.41%, volatility 22.8%.
Neither result reflects the regulated investment outcome; they are reported as a reference
for the incremental effect of the overlay on the analytical return stream.

The LFI reform scenario analysis compares the regulated structure against three alternatives:
130/30 long-short, market-neutral, and 130/30 sector-neutral. Scenario-level metrics for
each structure are computed in the full pipeline run and are available in
`reports/output/strategy_report_{source}_{model}.html`. The primary regulatory NAV results
in Table 1 are the appropriate basis for any compliance-relevant evaluation.

---

## 6. Discussion

The most important methodological finding of this work is that the LightGBM cross-sectional
forecaster does not materially outperform the elastic-net baseline on real Mexican-market
data. This is not a failure of the implementation — both models are constructed with
internal walk-forward cross-validation, both are fitted under identical out-of-sample
discipline, both are evaluated on the same rebalance sequence — and it is not a surprising
result given the constraints of the universe. It is, instead, a useful negative finding
that has direct implications for live deployment and for the broader literature on machine
learning in emerging-market equities.

The mechanism is the small effective cross-section. Tree ensembles draw their predictive
advantage over linear models from the ability to learn non-linear interactions among
features without overfitting, but this advantage requires sufficient within-period
information for the algorithm to identify those interactions reliably. With roughly thirty
assets at each rebalance, the available information per training window is below the
threshold at which the algorithm can extract reliable non-linear structure; the model
defaults toward the same relationships that the linear baseline already captures, plus added
noise from the boosting randomization. The 0.44 SHAP feature-rank stability is a direct
measurement of this noise. Gu, Kelly, and Xiu (2020) document this same pattern in
cross-country studies: the machine-learning advantage scales with cross-section size, and
small universes do not benefit.

This finding does not mean the LightGBM component is without value. The attribution
framework — SHAP per rebalance, feature-rank stability, regime-conditioned performance —
is genuinely informative independent of whether LightGBM itself generates excess return.
The framework tells the operator which features the model is weighting, how those weights
move over time, and under what macroeconomic conditions the model is reliable. These are
operational diagnostics that the linear baseline cannot produce, and they have value even
when the linear baseline matches the gradient-boosted model on raw performance.

Three operational responses follow from these findings. The first is regime-conditional
model selection: use the LightGBM forecasts and their attribution during EASING regimes,
where SHAP stability reaches 0.57, and rely more heavily on the ElasticNet baseline during
TIGHTENING regimes, where raw signal quality is higher (IC mean 0.14, ICIR 0.59) but
attribution is less stable. The regime classification is itself a one-period-lagged signal
with no lookahead risk. The second is feature pruning: the SHAP decomposition identifies
momentum_63 as a contributor to turnover without proportionate contribution to predictive
accuracy. Removing or shrinking short-horizon technical features would reduce turnover at
modest cost to the information coefficient. The third is ensemble blending: combining
ElasticNet and LightGBM predictions at weights inversely proportional to their out-of-
sample forecast variance would inherit the stability of the linear baseline and add whatever
marginal lift the gradient-boosted model can deliver in favorable regimes.

Several limitations deserve explicit acknowledgment. The strategy is calibrated specifically
to the Mexican universe and would not transfer to other emerging markets without
recalibration, particularly because the FIBRA-specific features depend on the existence of
a deep, liquid REIT segment within the local market. The closest direct analog is the
Brazilian Fundos Imobiliários market, where the same methodology could be applied with
relatively modest adaptation. A more rigorous treatment of transaction costs would model
market impact as a square-root function of trade size relative to daily volume, which
would affect both Bloomberg configurations approximately equally at their current
~3.5% monthly turnover. Finally, the 95% confidence interval on the Sharpe estimate is
very wide ([−0.849, 0.534] on Bloomberg) because the nine-year backtest contains only one
full Banxico tightening cycle and one easing cycle; precise estimation of regime-conditional
performance will require additional years of data or a multi-country extension.

---

## 7. Conclusion

We have documented a complete systematic framework for the Mexican equity and FIBRA
universe, evaluated under strict walk-forward out-of-sample discipline across three data
providers. The framework integrates multi-provider data infrastructure, a Black–Litterman
portfolio construction layer that blends machine-learning views with low-confidence macro
tilts, multiple optimizers operating under CNBV regulatory constraints, a Layer 2 FX hedge
overlay reported on an analytical basis, a LightGBM cross-sectional forecaster with
TreeExplainer SHAP attribution, Banxico macro-regime conditioning, and a complete suite of
overfitting diagnostics. The regulated portfolio achieves an annualized Sharpe ratio of
−0.18 on Bloomberg point-in-time fundamentals over the 2017–2026 window, with a 95%
bootstrap confidence interval of [−0.849, 0.534].

The principal empirical findings are three. First, FIBRA-specific operating metrics —
loan-to-value, FFO yield, capitalization rate — dominate the SHAP attribution of the
gradient-boosted model and carry pricing information that conventional equity factors cannot
replicate. Second, LightGBM and ElasticNetCV produce statistically indistinguishable
performance on real data within the Mexican cross-section, attributable to the small
effective universe; the value of the gradient-boosted model lies in its attribution
framework rather than in raw return lift. Third, model reliability is regime-dependent
in an asymmetric way: TIGHTENING cycles exhibit higher raw signal quality (ICIR 0.59 vs
0.18), while EASING cycles exhibit higher SHAP feature-rank stability (0.57 vs 0.41) —
a distinction with direct operational implications for regime-conditional model governance.

The work points toward three directions for further research. The first is an out-of-region
replication of the framework on the Brazilian Fundos Imobiliários market, which has a
larger and more diverse REIT cohort and would provide a natural validation of the
FIBRA-feature methodology. The second is a more granular macroeconomic conditioning that
moves beyond the binary regime classification to explicit term-structure, FX, and commodity
factors; the existing regime analysis suggests there is variance to capture that the current
binary specification leaves on the table. The third, and most operationally important, is
the construction of an ensemble between the ElasticNet and LightGBM paths with
regime-conditional weights, exploiting the asymmetric reliability of the two models across
the macroeconomic cycle. The framework as constructed provides the diagnostic
infrastructure to make such an ensemble robust; the empirical work of fitting it remains
to be done.

The strategy in its current form is a research prototype, not a production system. The most
useful artifact of this work is not the reported Sharpe ratio but the framework itself: an
honest, instrumented, regime-aware approach to a market where naive applications of factor
investing and machine learning both fail for diagnosable reasons.

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

Chen, T., and C. Guestrin (2016). "LightGBM: A scalable tree boosting system."
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

The LightGBM internal RandomizedSearchCV samples 5 configurations from the following
distributions for each training window, with a 3-fold expanding-window TimeSeriesSplit
and early stopping (50 rounds) on the inner-CV holdout. The inner-CV scoring metric is
the Spearman information coefficient (`ic`).

| Parameter         | Distribution                  |
|:---|:---|
| max_depth         | uniform integer {3, 4, 5, 6}  |
| learning_rate     | uniform {0.01, 0.03, 0.05, 0.10} |
| subsample         | uniform {0.7, 0.8, 1.0}       |
| colsample_bytree  | uniform {0.7, 0.8, 1.0}       |
| min_child_weight  | uniform {1, 5, 10}            |
| reg_alpha         | uniform {0, 0.1, 1.0}         |
| reg_lambda        | uniform {0.1, 1.0, 10.0}      |
| n_estimators      | early-stopped (cap 2,000)     |

The outer Optuna TPE search operates over a broader pipeline parameter space —
Black–Litterman risk aversion (BL τ and risk-aversion), mean-variance and CVaR optimizer
parameters, EWMA covariance lambda (bounded [0.90, 0.99]), ElasticNet L1 ratio grid,
macro-regime EWMA span, forecast horizon, and ADTV lambda — for 25 trials per data
source under purged walk-forward cross-validation with a 21-day gap. The objective is
a turnover-penalized Sharpe ratio with penalty weight 0.5.

## Appendix B. Software and Reproducibility

All results are reproducible from the repository accompanying this paper. The software
stack is Python 3.10 or higher with lightgbm ≥ 4.0, shap ≥ 0.45, scikit-learn, pandas,
numpy, Optuna, arch (GARCH), and scipy. Matplotlib and Plotly are used for figure
generation; the PDF tearsheet pipeline uses WeasyPrint with an fpdf2 fallback for
environments lacking the pango and gobject system libraries. Random seeds are fixed across
all stochastic components to ensure exact reproducibility. The test suite comprises 172
unit and integration tests covering walk-forward data integrity, portfolio constraint
compliance, SHAP schema, macro-regime no-lookahead, overfitting diagnostics, and
bootstrap CI.

Repository: github.com/MaxHidalgoLeon/FondoMexicoAlfa

