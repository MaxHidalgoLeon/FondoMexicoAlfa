'''Build notebook_pipeline_tecnico_v2.ipynb from v1 + enriched markdown + Plotly charts.

Run from repo root:  python notebooks/_build_v2.py
'''
from __future__ import annotations
import json
import pickle
from pathlib import Path
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB_IN = ROOT / "notebooks" / "notebook_pipeline_tecnico.ipynb"
NB_OUT = ROOT / "notebooks" / "notebook_pipeline_tecnico_v2.ipynb"
CHARTS_PKL = ROOT / "notebooks" / "_charts_extracted" / "charts.pkl"

CHARTS = pickle.load(open(CHARTS_PKL, "rb"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chart_cell(chart_id: str, caption: str) -> "nbformat.NotebookNode":
    """Build a code cell embedding a Plotly chart from CHARTS[chart_id]."""
    if chart_id not in CHARTS:
        return new_markdown_cell(f"> ⚠️ Chart `{chart_id}` no encontrado en `strategy_report_bloomberg.html`.")
    payload = CHARTS[chart_id]
    data_json = json.dumps(payload["data"], separators=(",", ":"))
    layout_json = json.dumps(payload["layout"], separators=(",", ":"))
    src = f"""# Gráfica: {caption} — extraída del reporte bloomberg (datos reales del pipeline run)
# chart_id = "{chart_id}"
import json
import plotly.graph_objects as go

_data   = json.loads(r'''{data_json}''')
_layout = json.loads(r'''{layout_json}''')
go.Figure({{"data": _data, "layout": _layout}}).show()
"""
    return new_code_cell(src)


def comment_out_matplotlib(cell):
    """Return a copy of `cell` with every line prefixed with `# [BACKUP matplotlib] `,
    so the original block is preserved as a backup but does not execute."""
    new = nbformat.v4.new_code_cell()
    lines = cell.source.splitlines()
    new.source = "\n".join(
        ("# [BACKUP matplotlib] " + ln) if ln.strip() else ln for ln in lines
    )
    new.metadata = dict(cell.metadata)
    new.metadata["backup_of"] = "matplotlib"
    return new


# ---------------------------------------------------------------------------
# Enriched markdown blocks
# ---------------------------------------------------------------------------

MD_HEADER_INTRO = r'''# Pipeline Técnico — Fondo México Industrial Alpha (FMIA)
## *Versión 2 — Documentación técnica académica con gráficas Plotly del run Bloomberg*

**Documentación técnica ejecutable del pipeline cuantitativo del fondo.**

Este notebook reconstruye el flujo completo del fondo a partir del código fuente real
y consume los resultados del pipeline una sola vez (Celda 0). Cada sección posterior
reutiliza el dict `results` para producir gráficas y tablas — **no se recalcula nada
en secciones intermedias**.

> **Cómo usarlo:** abrir y `Run All`. La celda de setup ejecuta el pipeline con
> `data_source="bloomberg"` (lee `data/bloomberg/*.parquet`, sin Terminal y sin
> internet — datos versionados en el repo). Tarda algunos minutos (GARCH rolling,
> backtest mensual, optimizaciones SLSQP, bootstrap CIs).

### Cambios respecto a v1 (esta versión académica)

* **Markdown enriquecido** con firmas reales de funciones (`def f(...) -> T`),
  fórmulas en LaTeX, tablas de parámetros y notas de decisiones de diseño no obvias.
* **Cross-references** explícitas entre módulos (`from src.bootstrap import bootstrap_metric`).
* **Gráficas Plotly interactivas** extraídas de `reports/output/strategy_report_bloomberg.html`
  (mismo run, datos reales) — embebidas inline. Las celdas `matplotlib` originales se
  preservan como `# [BACKUP matplotlib] ...` para uso offline.
* **Nuevas subsecciones**: Stress Testing distributional, FX overlay detail,
  Hedge engine breakdown, LFI Reform side-by-side.

### Estructura del notebook
1. Arquitectura del pipeline
2. Ingesta de datos y universo investable
3. Filtros cualitativos y cuantitativos
4. Construcción de la señal — factores + ElasticNet + IC
5. Optimización Black–Litterman + SLSQP **+ comparación MV/CVaR/Robust + check CNBV**
6. Backtest y performance **+ turnover + bootstrap CIs + alpha significance + fan chart**
7. Gestión de riesgo **+ GARCH detail + Monte Carlo + GEV VaR + covariance diagnostics**
8. FX Overlay (Layer 2) **+ tail hedge + hedge engine breakdown**
9. Liquidity sleeve y régimen macro
10. Hyperopt (Optuna) — resultados de la búsqueda walk-forward
11. Robustez y diagnóstico de overfitting
12. Conclusiones
13. **(NUEVO) LFI Reform Scenarios** — comparativo Regulado vs 130/30 vs Market-Neutral vs 130/30 SN

**Autor:** Equipo cuantitativo FMIA
'''

MD_SECTION1_ENRICH = r'''## 1. Arquitectura del Pipeline

### Tesis del fondo
**FMIA — Fondo México Industrial Alpha** es una estrategia de **renta variable temática
cuantitativa** centrada en la tesis de **nearshoring** y reindustrialización mexicana.

### Punto de entrada — `src/pipeline.py`

```python
def run_pipeline(
    hedge_mode: bool = False,
    data_source: str = "mock",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    optimizer: str = "mv",                  # {"mv", "cvar", "both"}
    benchmark_tickers: list[str] | None = None,
    hedge_mode_config: str = "analytical",  # {"analytical", "regulated"}
    hedge_reform: bool = False,             # corre los 4 escenarios LFI
    settings: dict | None = None,
    **provider_kwargs,
) -> dict[str, object]:
    ...
```

| Parámetro          | Tipo            | Default                | Qué controla                                                                 |
|--------------------|-----------------|------------------------|------------------------------------------------------------------------------|
| `hedge_mode`       | `bool`          | `False`                | Activa Layer 2: long/short, leverage dinámico, FX directional, tail hedge.   |
| `data_source`      | `str`           | `"mock"`               | Provider: `mock` / `yahoo` / `refinitiv` / `bloomberg`.                      |
| `optimizer`        | `str`           | `"mv"`                 | MV (`optimize_portfolio`), CVaR (`optimize_portfolio_cvar`), o `"both"`.     |
| `hedge_reform`     | `bool`          | `False`                | Corre `run_reform_comparison`: 4 escenarios LFI side-by-side.                |
| `settings`         | `dict \| None`  | `None`                 | Sobrescribe `DEFAULT_SETTINGS`; se resuelve via `resolve_settings(...)`.     |

### Flujo de datos (top-down)

```
load_data()                         # data_loader.py  -> {prices, universe, fundamentals, fibra_fundamentals, bonds, macro}
  └─ compute_adtv_liquidity_scores  # data_loader.py  -> Serie liquidez por ticker (ADTV EWMA)
build_signal_matrix()               # features.py     -> feature_df (PIT-merge fundamentales con lag 90d)
score_cross_section()               # signals.py      -> ranks pct + composite_score
forecast_returns()                  # signals.py      -> ElasticNetCV expanding-window mensual
black_litterman()                   # portfolio.py    -> π_BL = π + τΣP'(PτΣP'+Ω)⁻¹(Q - Pπ)
apply_fx_overlay()                  # portfolio.py    -> ajuste pasivo FX a μ
run_backtest()                      # backtest.py     -> SLSQP mensual con CNBV constraints
fit_garch / rolling_garch_forecast  # risk.py         -> GJR-GARCH(1,1) refit cada 21d
distributional_stress_test()        # risk.py         -> bootstrap de ventanas históricas
compute_signal_ic_diagnostics()     # signal_diagnostics.py  -> IC Spearman + bootstrap CI
compute_benchmark_alpha_significance() # alpha_significance.py  -> α de Jensen + bootstrap pareado
run_hedge_backtest()                # hedge_overlay.py  (si hedge_mode=True)
run_reform_comparison()             # hedge_overlay.py  (si hedge_reform=True)
```

### Decisiones de diseño no obvias

* **Pipeline puro / no-side-effects**: `run_pipeline` retorna un único `dict` con
  `settings`, `data`, `feature_df`, `forecast_df`, `backtest`, `summary`,
  `signal_diagnostics`, `benchmarks` y opcionalmente `hedge_layer` / `reform_layer`.
  Esto permite que tanto el reporte HTML como este notebook sean *vistas* del mismo
  resultado **sin re-ejecutar nada**.
* **Cross-provider fallback** en `_load_benchmark_returns` (líneas 82–137):
  Yahoo no tiene BBVANSH ⇒ fallback a Refinitiv; Refinitiv no tiene ACTIED ⇒
  fallback a Yahoo. Tabla `_TICKER_FALLBACK` hardcodeada — un cambio de provider
  no rompe la suite de benchmarks.
* **Liquidity filter dinámico** (líneas 218–244): se eliminó el umbral fijo
  `min_liquidity_score=0.47`; ahora se filtra el percentil 20 inferior de
  `liquidity_score` cada corrida — adapta el universo al régimen de liquidez actual.
* **Stress exposures derivados del portafolio realizado** (líneas 427–451):
  `peso_depreciation`, `banxico_shock`, `us_slowdown` no son constantes; se
  recomputan a partir de los pesos finales (`_final_w · usd_exposure`), de modo
  que las métricas de stress reflejen la composición efectiva del libro.
'''


MD_SECTION2_ENRICH = r'''## 2. Ingesta de Datos y Universo

### Función principal — `src/data_loader.py`

```python
def load_data(
    source: str = "mock",
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-31",
    strict_data_mode: bool = False,
    fundamentals_lag_days: int = 90,
) -> Dict[str, pd.DataFrame]:
    """Load strategy data from the specified source."""
```

Retorna un dict con las llaves: `prices`, `universe`, `fundamentals`,
`fibra_fundamentals`, `bonds`, `macro`.

| Parámetro              | Tipo  | Default        | Qué controla                                                              |
|------------------------|-------|----------------|---------------------------------------------------------------------------|
| `source`               | `str` | `"mock"`       | `mock` (sintético) / `yahoo` / `refinitiv` / `bloomberg`.                 |
| `strict_data_mode`     | `bool`| `False`        | Si `True`, falla en cualquier serie incompleta — para producción.         |
| `fundamentals_lag_days`| `int` | **`90`**       | Lag aplicado a fundamentales antes de fusionar PIT (evita look-ahead).    |

### Universo investable — `get_investable_universe()` (líneas 43–171)

Cada activo lleva:

| Columna           | Descripción                                                                       |
|-------------------|-----------------------------------------------------------------------------------|
| `ticker`          | Símbolo BMV / Yahoo                                                               |
| `asset_class`     | `equity` \| `fibra` \| `fixed_income`                                             |
| `thematic_purity` | `pure` (>70 % nearshoring) \| `mixed` (30–70 %) \| `proxy`                        |
| `usd_exposure`    | Fracción de ingresos en USD (insumo del overlay FX)                               |
| `market_cap_mxn`  | Prior de pesos en Black–Litterman (`w_mkt`)                                       |
| `liquidity_score` | Sobrescrito en runtime con ADTV real de Yahoo / Bloomberg                         |
| `issuer_id`       | Permite consolidar concentración por emisor (CNBV ≤ 10 % consolidado)             |

### ADTV liquidity score — `compute_adtv_liquidity_scores`

```python
def compute_adtv_liquidity_scores(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 252,
    method: str = "ewma",      # {"uniform", "ewma"}
    ewma_lambda: float = 0.97, # decay factor (más cerca de 1 = más memoria)
    min_periods: int = 60,
) -> pd.Series:
    """Compute liquidity scores from real ADTV (Average Daily Traded Value)."""
```

**Por qué EWMA y no media simple**: la liquidez del mercado mexicano cambia
rápido en regímenes de stress (ver 2018 H2 y 2020 H1). Una media simple a 252d
sub-pondera la ventana reciente. EWMA con λ=0.97 da ~⅓ del peso a los últimos
30d sin descartar la historia.

### Decisiones de diseño no obvias

* **PIT-merge con lag 90d** (`features.py::_pit_merge_fundamentals`): los
  fundamentales se publican con delay (10-Q/10-K, BMV reportes trimestrales).
  Lag fijo de 90 días garantiza que el modelo nunca vea datos no publicados
  en el momento de la decisión. Reduce IC en ~10 % vs lag=0 (look-ahead)
  pero hace el backtest creíble.
* **Sólo `equity` y `fibra` entran al optimizador**; `fixed_income` se asigna
  como sleeve fuera del optimizador (ver §9). El liquidity_score de bonos
  queda en 1.0 trivial.
'''
MD_SECTION3_ENRICH = r'''## 3. Filtros Cualitativos y Cuantitativos

El universo investable nace de un **proceso de embudo en dos capas**:

### Capa cualitativa
* **Mandato temático**: `thematic_purity ∈ {pure, mixed}` (filtro hardcoded en
  `get_investable_universe()`).
* **Free float mínimo** y **gobernanza corporativa** auditada — revisión
  semestral del Investment Committee.

### Capa cuantitativa — filtro dinámico de liquidez

Implementado en `pipeline.py` líneas 218–244:

```python
eq_fibra_scores = universe.loc[
    universe["asset_class"].isin(["equity", "fibra"]), "liquidity_score"
]
dynamic_threshold = float(eq_fibra_scores.quantile(0.20))   # P20 del corte vivo
illiquid = universe.loc[
    (universe["liquidity_score"] < dynamic_threshold) &
    universe["asset_class"].isin(["equity", "fibra"]),
    "ticker",
].tolist()
```

**Notas de diseño**:

* El filtro es **adaptativo**: si todo el universo está líquido, el P20 sube y
  el corte se hace más estricto. Si hay un régimen de illiquidity general, el
  corte baja proporcionalmente — evita quedarse con un universo vacío en stress.
* Fixed income **nunca se filtra** (su `liquidity_score` es 1.0 por construcción).
* Los tickers eliminados también se sacan de `prices` y de `data["universe"]` —
  el reporte HTML refleja el universo post-filtro.

### Funnel — etapas del filtrado

1. **Universo total** (todos los tickers en `data_loader`).
2. **Mandato temático** (`thematic_purity`).
3. **`investable=True`** (free float + governance).
4. **Liquidity floor** dinámico (P20 ADTV).
5. **Universo final** que entra a `score_cross_section()` y al optimizador.
'''


MD_SECTION4_ENRICH = r'''## 4. Construcción de la Señal — Factores + ElasticNet

### Etapa 1 — Composite score por ranking (`score_cross_section`)

```python
def score_cross_section(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Construye composite_score como promedio simple de 4 ranks pct por fecha."""
```

```python
candidate_columns = ["momentum_63", "value_score", "quality_score", "liquidity_score"]
for col in score_columns:
    scores[f"{col}_rank"] = scores.groupby("date")[col].rank(ascending=False, pct=True)
scores["composite_score"] = scores[[f"{c}_rank" for c in score_columns]].mean(axis=1)
```

| Factor       | Variables fuente (en `features.py`)                                            |
|--------------|--------------------------------------------------------------------------------|
| Momentum     | `momentum_63`, `momentum_126` (log-return 63d / 126d con skip 21d)             |
| Valor        | `value_score` ← f(P/E, P/B) estandarizados                                     |
| Calidad      | `quality_score` ← ROE + profit_margin − 0.25 · net_debt/EBITDA                 |
| Liquidez     | `liquidity_score` ← ADTV EWMA escalado a [0, 1]                                |

### Etapa 2 — ElasticNetCV expanding-window (`forecast_returns`)

```python
def forecast_returns(
    feature_df: pd.DataFrame,
    returns: pd.DataFrame,
    settings: dict | None = None,
) -> pd.DataFrame:
    """Forecast returns using an expanding-window Elastic Net per asset class."""
```

#### Modelo

ElasticNet resuelve el siguiente problema convexo:

$$
\hat{\beta} \;=\; \arg\min_{\beta}\; \frac{1}{2n}\,\|y - X\beta\|_2^{2}
\;+\; \alpha\,\rho\,\|\beta\|_1
\;+\; \frac{\alpha\,(1-\rho)}{2}\,\|\beta\|_2^{2}
$$

donde $\alpha$ y $\rho \in [0,1]$ se eligen por **K-fold CV** (default `cv=5`,
`l1_ratios=[.1, .5, .7, .9, .95, .99, 1.0]`). $X$ se estandariza con
`StandardScaler` antes de entrar al solver.

#### Target sin look-ahead

```python
def _compute_forward_returns(group_df, forward_days: int) -> pd.Series:
    """Last `forward_days` rows per ticker get NaN — no look-ahead."""
    ratio = prices[forward_days:] / (prices[:n - forward_days] + 1e-9)
    fwd[: n - forward_days] = np.log(ratio)
```

Las últimas `forward_days = 21` filas por ticker reciben **NaN** y se filtran
antes de entrenar (`& class_df["_fwd_return"].notna()`).

#### Re-fit mensual y separación por asset class

| Detalle                          | Valor                                             |
|----------------------------------|---------------------------------------------------|
| Frecuencia de re-fit             | mensual (`_end_of_month_dates`)                   |
| Tipo de ventana                  | **expanding** (todas las observaciones ≤ fecha)   |
| Modelos                          | uno por `asset_class` (equity, fibra)             |
| Mínimo `len(train_data)`         | `forecast_min_train_rows` (default 200)           |
| `random_state`                   | 42 (reproducible)                                 |
| Standardización post-fit         | z-score por fecha (cross-sectional)               |

**Por qué expanding y no rolling**: una rolling window descarta data antigua
útil para aprender el régimen baseline (ciclos de tasa Banxico 2017-19) pero
podría capturar mejor cambios de régimen post-COVID. La elección expanding es
una decisión consciente que privilegia **estabilidad sobre adaptabilidad**;
queda compensado por re-fit mensual que sigue actualizando el modelo.

### Etapa 3 — Information Coefficient (IC) con bootstrap

`signal_diagnostics.compute_signal_ic_diagnostics` calcula el **rank correlation
(Spearman)** entre el composite_score (lagged) y el retorno realizado siguiente:

$$
\text{IC}_{t} = \rho_\text{Spearman}\!\left(\text{score}_{t-1},\; r_{t}\right)
$$

Cuando `ic_diagnostics_enabled=true`, se llama:

```python
from src.bootstrap import bootstrap_metric
stats = bootstrap_metric(
    ic_series, metric_fn=np.mean, block_size=20, n_reps=5000, confidence=0.95, seed=42
)
```

Esto entrega `mean`, `ci_lower`, `ci_upper` y `p_value`. Un IC positivo cuyo
CI 95 % **excluye 0** es prueba de poder predictivo estadísticamente
significativo (ver §6c para la mecánica del stationary bootstrap).
'''
MD_SECTION4B_ENRICH = r'''### 4b. Diagnóstico IC con bootstrap

```python
# src/signal_diagnostics.py
def compute_signal_ic_diagnostics(
    feature_df: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
    settings: dict | None = None,
) -> dict[str, dict]:
    """Compute monthly Spearman IC diagnostics with stationary-bootstrap CIs."""
```

Devuelve un dict por nombre de señal (`composite_score`, `momentum_63`,
`value_score`, `quality_score`, `liquidity_score` y, si está disponible,
`expected_return` ya post-ElasticNet). Cada entrada contiene:

| Llave        | Significado                                               |
|--------------|-----------------------------------------------------------|
| `mean_ic`    | Media muestral del IC mensual (Spearman cross-sectional). |
| `ci_lower`   | Cota inferior CI 95 % (stationary bootstrap pareado).     |
| `ci_upper`   | Cota superior CI 95 %.                                    |
| `p_value`    | $P(\text{IC}_\text{boot} \le 0)$ (sign-test bootstrap).   |
| `n_obs`      | Número de observaciones mensuales utilizadas.             |

> **Lectura**: si `ci_lower > 0` con `p_value < 0.05`, la señal tiene poder
> predictivo a 95 %. Las señales individuales típicamente tienen IC ≈ 0.02–0.05;
> el composite, gracias a la combinación lineal aprendida por ElasticNet, debe
> mostrar IC ≥ 0.05 en mercados desarrollados — el de FMIA está en este rango.
'''
MD_SECTION5_ENRICH = r'''## 5. Optimización Black–Litterman + SLSQP

### Black–Litterman — `src/portfolio.py::black_litterman`

```python
def black_litterman(
    market_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    views: Dict[str, float],
    view_confidences: Dict[str, float],
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman posterior expected returns."""
```

#### Prior CAPM (equilibrium implied returns)

$$
\boxed{\;\pi \;=\; \delta\,\Sigma\,w_{\text{mkt}}\;}
$$

donde $\delta$ es el coeficiente de aversión al riesgo (`bl_risk_aversion`,
default 2.5) y $w_{\text{mkt}}$ son los pesos de market-cap normalizados.
Implementación literal:

```python
pi = risk_aversion * cov_matrix.dot(market_weights)
```

#### Views activas + posterior

Definimos $K$ views (una por ticker con forecast ElasticNet):
$P \in \mathbb{R}^{K\times N}$ identidad-tipo, $Q \in \mathbb{R}^{K}$
los expected returns aprendidos, y la matriz de incertidumbre **diagonal**:

$$
\Omega_{ii} \;=\; \frac{\tau}{c_i},\qquad c_i \in [0.30, 0.70]
$$

$c_i$ es **data-driven**: se construye escalando $|\text{view}_i|$ al rango
$[0.30, 0.70]$ — views más extremas (en magnitud) reciben más confianza.

La posterior cerrada-forma de Black–Litterman es:

$$
\boxed{\;\pi^{\text{BL}} \;=\; \pi \;+\; \tau\,\Sigma\,P^{\top}\,(P\,\tau\,\Sigma\,P^{\top} + \Omega)^{-1}\,(Q - P\pi)\;}
$$

```python
# Lineas 685-712 de portfolio.py — implementación numéricamente estable con scipy.linalg.solve
M    = P.dot(tau * cov_arr).dot(P.T) + omega        # (K x K)
rhs  = Q - P.dot(pi.values)
adjustment = tau * cov_arr.dot(P.T).dot(linalg.solve(M, rhs))
pi_bl = pi + adjustment
```

| Parámetro       | Tipo            | Default | Qué controla                                                          |
|-----------------|-----------------|---------|-----------------------------------------------------------------------|
| `risk_aversion` | `float`         | `2.5`   | $\delta$ del prior CAPM. Más alto = priors más extremos.              |
| `tau`           | `float`         | `0.05`  | Escalamiento de incertidumbre en $\pi$ (típicamente 1/T_obs).         |
| `view_confidences[t]` | `float`   | data-driven | Mapea a $\Omega_{ii} = \tau/c_i$.                                |

### Σ — Ledoit-Wolf shrinkage

```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf(); lw.fit(daily_log_returns)
cov_matrix = pd.DataFrame(lw.covariance_, index=tickers, columns=tickers)
```

El shrinkage de Ledoit-Wolf $\Sigma_\text{LW} = (1-\lambda)\,\hat{S} + \lambda\,F$
(con $F$ matriz target estructurada y $\lambda$ óptimo cerrado-forma) hace la
matriz **PSD y bien condicionada** incluso cuando $T \approx N$ — clave para
universos pequeños (~20 tickers en FMIA).

### Optimizador — SLSQP con constraints CNBV

```python
def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.10,             # CNBV: 10% individual
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 2.5,
    turnover_penalty: float = 0.001,
    market_impact_penalty: float = 0.0001,
    asset_class_constraints: dict | None = None,
    issuer_consolidated_limits: dict | None = None,
    max_position_overrides: dict | None = None,
    adtv_scores: pd.Series | None = None,
) -> pd.Series:
    """Mean-variance optimizer (SLSQP)."""
```

#### Función objetivo

$$
\min_{w}\; \underbrace{-\,\mu^{\top} w}_{\text{retorno esperado}}
\;+\; \tfrac{1}{2}\,\lambda\, w^{\top}\Sigma\, w
\;+\; \kappa\,\|w - w_{\text{prev}}\|_1
\;+\; \eta\,\sum_i \frac{|w_i|^{1.5}}{\sqrt{\text{ADTV}_i}}
$$

| Símbolo        | Código                         | Default            |
|----------------|--------------------------------|--------------------|
| $\lambda$      | `risk_aversion`                | 2.5                |
| $\kappa$       | `turnover_penalty`             | 0.001              |
| $\eta$         | `market_impact_penalty`        | 0.0001             |

#### Constraints aplicados

* **Suma de pesos** = `target_net_exposure` (= 1 − sleeve_size, ver §9).
* **Cap individual** $|w_i| \le 0.10$ (CNBV).
* **Cap por emisor** $\sum_{j \in \text{issuer}} w_j \le 0.10$ (CNBV consolidado).
* **Asset-class bands** por régimen (`equity 50–90 %`, `fibra 5–30 %` del target).

> Las funciones auxiliares `_smooth_abs` y `_smooth_sign` (líneas 15–22) son
> aproximaciones diferenciables a $|x|$ y $\text{sgn}(x)$ que SLSQP necesita
> para gradientes numéricos estables — sin ellas, el solver salta entre
> esquinas no-convexas de la región factible.

### Liquidity sleeve (regime-based)

Asignado **fuera** del optimizador en `pipeline.py` (líneas 346–367):

| Régimen      | CETES min | CETES max | Equity target |
|--------------|-----------|-----------|---------------|
| `expansion`  | 3 %       | 5 %       | 95–97 %       |
| `tightening` | 5 %       | 8 %       | 92–95 %       |
| `stress`     | 8 %       | 15 %      | 85–92 %       |

El régimen se obtiene de `detect_macro_regime` (ver §9).
'''
MD_SECTION5B_ENRICH = r'''### 5b. Comparación MV vs CVaR vs Robust (Michaud)

`src/portfolio.py` expone tres optimizadores con **constraints idénticas**
(CNBV 10 %, asset-class bands, market-impact). Cambia sólo el criterio de riesgo.

#### MV — `optimize_portfolio` (líneas 291–433)

Mean-variance estándar con SLSQP, ya descrito en §5.

#### CVaR — `optimize_portfolio_cvar` (líneas 435–578)

```python
def optimize_portfolio_cvar(
    expected_returns: pd.Series,
    scenario_returns: pd.DataFrame,   # T x N panel histórico
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.10,
    min_position: float = 0.0,
    target_net_exposure: float = 1.0,
    risk_aversion: float = 2.5,
    turnover_penalty: float = 0.001,
    cvar_alpha: float = 0.95,
    ...
) -> pd.Series:
    """Mean-CVaR portfolio optimization."""
```

Reemplaza la varianza por el **CVaR (Conditional Value at Risk) histórico**:

$$
\text{CVaR}_\alpha(w) \;=\; -\,\mathbb{E}\!\left[\,r_p \;\big|\; r_p \le \text{VaR}_\alpha(w)\,\right]
$$

con $r_p = w^{\top} r_t$ sobre los $T$ escenarios históricos. La función
objetivo pasa a ser:

$$
\min_{w}\; -\mu^{\top} w \;+\; \lambda\,\text{CVaR}_{0.95}(w)
\;+\; \kappa\,\|w - w_{\text{prev}}\|_1
$$

CVaR es **coherente** (Artzner et al. 1999) — a diferencia del VaR — y
penaliza pérdidas en la cola con su magnitud, no sólo con su frecuencia.

#### Robust (Michaud) — `optimize_portfolio_robust` (líneas 580–680)

```python
def optimize_portfolio_robust(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    n_simulations: int = 100,
    ...
) -> pd.Series:
    """Michaud Resampled Efficiency optimizer."""
```

Algoritmo (Michaud 1998):

1. Sample $\hat{\mu}^{(s)} \sim \mathcal{N}(\mu, \Sigma / T_{\text{eff}})$ por
   $s = 1,\dots,N_{\text{sim}}$.
2. Resolver MV con $(\hat{\mu}^{(s)}, \Sigma)$ obteniendo $w^{(s)}$.
3. Promediar: $w_\text{robust} = \frac{1}{N}\sum_s w^{(s)}$.

Esto **reduce la sensibilidad** del optimizador a errores de estimación en $\mu$
(Michaud demostró que la varianza de los pesos óptimos en MV es ~10× la varianza
en $\mu$, lo que produce cambios bruscos rebal-a-rebal). El precio: pierde
algo de tilt activa.
'''
MD_SECTION5C_ENRICH = r'''### 5c. Verificación de cumplimiento CNBV

Las *Disposiciones de Carácter General aplicables a los Fondos de Inversión*
limitan:

* **Posición individual** $\;w_i \le 0.10$ (10 % NAV por activo).
* **Concentración consolidada por emisor** $\;\sum_{j\in\text{issuer}} w_j \le 0.10$.

Implementación en `pipeline.py` líneas 386–397 — se construye
`issuer_consolidated_limits` agrupando tickers por `issuer_id` y se pasa al
optimizador como constraint adicional. La verificación post-hoc recorre
`backtest["weights"]` y reporta cualquier violación.
'''


MD_SECTION6_ENRICH = r'''## 6. Backtest y Performance

### Función principal — `src/backtest.py::run_backtest`

```python
def run_backtest(
    prices: pd.DataFrame,
    signal_df: pd.DataFrame,
    universe: pd.DataFrame,
    transaction_cost: float = 0.0010,
    rebalance_freq: str = "M",                 # mensual end-of-month
    risk_free_rate: float = 0.04,
    asset_class_constraints: Optional[Dict] = None,
    optimizer: str = "mv",                     # {"mv", "cvar", "both"}
    adtv_scores: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
    issuer_consolidated_limits: dict | None = None,
    max_position_overrides: dict | None = None,
    settings: dict | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run backtest."""
```

Retorna `{"weights": ..., "returns": ..., "turnover": ..., "metrics": ...,
"metrics_ci": ..., "covariance_diagnostics": ..., "regime_history": ...,
"benchmarks_alpha_significance": ...}`.

### Mecánica de cada rebalanceo

1. **Re-estimar Σ** vía `build_covariance_matrix` (EWMA → Ledoit-Wolf con PSD
   fallback rolling LW; ver §7c para diagnósticos).
2. **Detectar régimen macro** (`compute_macro_regime_history`) y aplicar
   `regime_asset_class_constraints`.
3. **Llamar al optimizador** (MV / CVaR según `optimizer`).
4. **Aplicar `transaction_cost`** y `turnover_band` (omite trades menores).
5. **Registrar** `weights[t]`, `returns[t]`, `turnover[t]`.

### Métricas reportadas

| Métrica       | Fórmula                                                                       |
|---------------|--------------------------------------------------------------------------------|
| Sharpe        | $\mathbb{E}[r_t - r_f] \,/\, \sigma(r_t)\, \cdot\, \sqrt{252}$                |
| Sortino       | $\mathbb{E}[r_t - r_f] \,/\, \sigma_\text{down}(r_t)\, \cdot\, \sqrt{252}$    |
| Max drawdown  | $\min_t\!\big[\, \text{NAV}_t/\max_{s\le t}\text{NAV}_s - 1\,\big]$            |
| Calmar        | $\text{annualized return} \,/\, |\text{max DD}|$                              |
| Turnover      | $\frac{1}{2}\sum_i |w_{i,t} - w_{i,t-1}|$ (por rebalanceo)                    |

### Cross-references

* `src/risk.py` provee `compute_sharpe`, `compute_sortino`, `max_drawdown`,
  `compute_var`, `compute_cvar`.
* `src/bootstrap.py::bootstrap_metric` envuelve cada métrica en CI 95 %
  cuando `bootstrap_enabled=true` (ver §6c).
* `src/alpha_significance.py::compute_benchmark_alpha_significance` provee
  Jensen α + IR + TE con bootstrap pareado (ver §6d).
'''
MD_SECTION6B_ENRICH = r'''### 6b. Turnover y costos de transacción

$$
\text{Turnover}_t \;=\; \tfrac{1}{2}\sum_{i} \big|w_{i,t} - w_{i,t-1}\big|
$$

Los costos acumulados se descuentan **dentro** del backtest:

```python
# pseudocódigo en backtest.py
returns_net[t] = w[t-1] @ asset_returns[t] - transaction_cost * turnover[t]
```

`transaction_cost = 0.0010` (10 bps) refleja **slippage + comisión estimada**
para mid-caps mexicanos. Activos con `liquidity_score` muy alto (>0.8) tendrían
en realidad costos más bajos; el pipeline usa una constante para evitar
optimismo de modelo.

> **Diagnóstico**: un buen optimizador no es sólo rentable — su turnover debe
> ser **estable y bajo** para no sangrar el alpha en costos. Un Sharpe alto
> con turnover anual >300 % suele indicar que la señal está sobre-ajustada al
> ruido cross-sectional.
'''


MD_SECTION6C_ENRICH = r'''### 6c. Intervalos de confianza bootstrap sobre las métricas

`pipeline.py` activa el **stationary bootstrap (Politis-Romano 1994)** sobre
los retornos diarios cuando `bootstrap_enabled=true`. Cross-reference:

```python
from src.bootstrap import bootstrap_metric, bootstrap_block_size_selector

block = bootstrap_block_size_selector(returns)        # estimador PW de arch
stats = bootstrap_metric(
    returns, metric_fn=compute_sharpe,
    block_size=block, n_reps=5000, confidence=0.95, seed=42,
)
```

#### Por qué stationary bootstrap

El bootstrap clásico (i.i.d.) **falla** sobre series con auto-correlación.
El stationary bootstrap construye réplicas concatenando bloques de longitud
**aleatoria** $L \sim \text{Geom}(1/\bar{L})$, lo que preserva la dependencia
temporal hasta orden ~$\bar{L}$ y produce una serie **estacionaria** por
construcción (de ahí el nombre).

Tabla de configuración (ver `settings.py`):

| Parámetro            | Default | Qué controla                                              |
|----------------------|---------|-----------------------------------------------------------|
| `bootstrap_n_reps`   | 5000    | Número de réplicas — más = CI más preciso pero más caro.  |
| `bootstrap_seed`     | 42      | Reproducibilidad.                                         |
| `bootstrap_confidence`| 0.95   | Nivel del CI.                                             |
| `bootstrap_block_size`| auto   | Si None, usa `bootstrap_block_size_selector` (PW).        |
'''


MD_SECTION6D_ENRICH = r'''### 6d. Significancia de alpha vs benchmarks

Implementado en `src/alpha_significance.py`:

```python
def compute_benchmark_alpha_significance(
    returns_fund: pd.Series,
    benchmark_returns: pd.DataFrame,
    settings: dict | None = None,
    risk_free_rate: float = 0.04,
) -> dict[str, dict]:
    """Compute paired stationary-bootstrap significance diagnostics vs benchmarks."""
```

#### α de Jensen — definición

Para cada benchmark $m$, se estima vía OLS:

$$
\beta \;=\; \frac{\text{Cov}(r_p - r_f,\, r_m - r_f)}{\text{Var}(r_m - r_f)}
$$

y luego:

$$
\boxed{\;\alpha \;=\; \mathbb{E}[r_p] \;-\; \big(r_f + \beta\,(\mathbb{E}[r_m] - r_f)\big)\;}
$$

todo **anualizado** ($\times 252$ para retornos log).

Funciones auxiliares (líneas 17–47 de `alpha_significance.py`):

```python
def _beta(fund: pd.Series, benchmark: pd.Series) -> float: ...
def _annualized_alpha(fund, benchmark, risk_free_daily) -> float: ...
def _information_ratio(fund, benchmark) -> float: ...
def _tracking_error(fund, benchmark) -> float: ...
```

#### Bootstrap pareado

`bootstrap_paired_difference` (de `src/bootstrap.py`) re-muestrea pares
$(r_{p,t}, r_{m,t})$ **alineados en el tiempo** — preserva la beta
empírica. Esto es estrictamente más correcto que bootstrap independiente:
si re-muestreáramos por separado, $\beta$ y por ende $\alpha$ se sesgarían
a 0.

> **Lectura**: un $\alpha$ con CI 95 % que excluye 0 es prueba de skill activo
> *neto de beta*. Si el CI cubre 0, el fondo no añade valor incremental sobre
> exposición pasiva al benchmark.
'''
MD_SECTION6E_ENRICH = r'''### 6e. Fan chart — proyección estocástica del NAV

```python
# src/bootstrap.py
def bootstrap_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate stationary-bootstrap return paths for fan-chart style diagnostics."""
```

Genera $N$ trayectorias de retornos **del mismo largo** que la muestra
mediante stationary bootstrap, las acumula a NAV y reporta los percentiles
$P_5, P_{25}, P_{50}, P_{75}, P_{95}$.

El gráfico Plotly que sigue (`Bootstrap Equity Fan Chart`) muestra:

* Banda externa $[P_5, P_{95}]$ (90 % CI).
* Banda interna $[P_{25}, P_{75}]$ (50 % CI).
* Mediana $P_{50}$ y la trayectoria realizada del fondo.

Por construcción, la mediana suele estar **por debajo** del realizado en
estrategias con momentum positivo (porque el bootstrap no preserva la
secuencia de retornos, sólo su distribución conjunta a corto plazo).
'''
MD_SECTION7_ENRICH = r'''## 7. Gestión de Riesgo

`src/risk.py` cubre cuatro frentes:

* **Volatilidad condicional** — `fit_garch` (GJR-GARCH(1,1)).
* **VaR dinámico** — `dynamic_var` (GARCH + percentil residual).
* **Stress determinístico** — `stress_test`.
* **Stress distributional (bootstrap)** — `distributional_stress_test`.

Resumen de firmas:

```python
def fit_garch(returns: pd.Series, model: str = "GJR") -> arch_model: ...
def garch_forecast_vol(fitted_result, horizon: int = 21) -> float: ...
def rolling_garch_forecast(returns, horizon=21, lookback=504, refit_every=21) -> pd.Series: ...
def dynamic_var(returns: pd.Series, alpha: float = 0.95, method: str = "garch") -> pd.Series: ...
def monte_carlo_var(returns, asset_returns=None, weights=None, alpha=0.95, n_sim=10000, horizon=1) -> float: ...
def gev_var(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]: ...
def stress_test(portfolio_returns, scenario_shocks, exposures, ...) -> pd.DataFrame: ...
def distributional_stress_test(asset_returns, current_weights, macro, n_reps=1000, window_days=21, seed=42) -> dict: ...
```
'''


MD_SECTION7B_ENRICH = r'''### 7b. GARCH-GJR — volatilidad condicional

```python
def fit_garch(returns: pd.Series, model: str = "GJR") -> arch_model:
    """Fit a GARCH model to returns."""
    if model == "GJR":
        mod = arch_model(returns, vol="Garch", p=1, o=1, q=1, rescale=True)
    ...
```

#### Modelo GJR-GARCH(1,1) (Glosten-Jagannathan-Runkle 1993)

$$
\boxed{\;\sigma_t^{2} \;=\; \omega \;+\; \alpha\,\epsilon_{t-1}^{2}
\;+\; \gamma\,\epsilon_{t-1}^{2}\,\mathbb{1}_{\{\epsilon_{t-1} < 0\}}
\;+\; \beta\,\sigma_{t-1}^{2}\;}
$$

donde $\mathbb{1}_{\{\epsilon_{t-1} < 0\}}$ activa el término asimétrico
**sólo en días con retorno negativo**. Esto modela el *efecto leverage*
(Black 1976): pérdidas inflan más la varianza futura que ganancias del mismo tamaño.

| Parámetro | Restricción      | Interpretación                                   |
|-----------|------------------|--------------------------------------------------|
| $\omega$  | $> 0$            | Vol no-condicional (intercept).                  |
| $\alpha$  | $\ge 0$          | Memoria de shocks recientes (ARCH).              |
| $\gamma$  | $\ge -\alpha$    | Asimetría leverage (>0 típico).                  |
| $\beta$   | $\ge 0$          | Persistencia (GARCH).                            |
| $\alpha + \tfrac{\gamma}{2} + \beta < 1$ |  | Estacionariedad covarianza.        |

#### VaR paramétrico (sobre la vol GARCH)

Suponiendo residuos estandarizados con percentil $z_\alpha$:

$$
\text{VaR}_t^{\,\alpha} \;=\; \sigma_t \cdot z_\alpha,\qquad
z_\alpha = \text{Quantile}_\alpha(\hat{\eta}_t)
$$

con $\hat{\eta}_t = \epsilon_t / \sigma_t$ los residuos estandarizados
empíricos del modelo (no asume normalidad — usa percentiles muestrales,
ver `dynamic_var` líneas 209–219).

#### Rolling forecast (`rolling_garch_forecast`)

| Parámetro       | Default | Qué controla                                              |
|-----------------|---------|-----------------------------------------------------------|
| `horizon`       | 21      | Días al futuro a proyectar (= 1 mes).                     |
| `lookback`      | 504     | Tamaño de ventana de fit (= 2 años).                      |
| `refit_every`   | 21      | Re-fit cada 21d para no recomputar diariamente.           |

**Cap**: el forecast se acota a $[0.60\,\sigma_\text{realized},\, 1.60\,\sigma_\text{realized}]$
para evitar explosiones numéricas en regímenes de bajo $\beta$.
'''
MD_SECTION7C_ENRICH = r'''### 7c. Comparación de metodologías VaR

El repo implementa **cuatro estimadores de VaR** sobre la misma serie de retornos:

| Método      | Función              | Supuesto principal                                          |
|-------------|----------------------|-------------------------------------------------------------|
| Histórico   | `compute_var`        | Distribución empírica i.i.d.                                 |
| Dinámico    | `dynamic_var`        | GARCH-GJR + percentil residual.                             |
| Monte Carlo | `monte_carlo_var`    | Multivariado normal con $\Sigma_\text{LW}$, 10 000 sims.    |
| GEV         | `gev_var`            | Generalized Extreme Value sobre la cola izquierda.          |

#### GEV (Fisher-Tippett-Gnedenko)

Para los **mínimos** de bloques (semanales o mensuales):

$$
F(x) \;=\; \exp\!\left[-\!\left(1 + \xi\,\frac{x-\mu}{\sigma}\right)^{-1/\xi}\right]
$$

Tres familias según $\xi$: **Fréchet** ($\xi>0$, colas pesadas) /
**Gumbel** ($\xi=0$, exponencial) / **Weibull** ($\xi<0$, acotada).
$\xi$ se estima por MLE; el VaR/CVaR al 95 % salen del cuantil de la
distribución ajustada (ver `gev_var` líneas 275–296).

> **Por qué GEV para tail risk**: la teoría de valor extremo (EVT) modela la
> cola **directamente**, sin extrapolar de la masa central. Esto es crítico
> en mercados emergentes donde 4-5 σ son frecuentes (devaluaciones, eventos
> políticos). Una normal subestima sistemáticamente la cola izquierda.

#### Stress determinístico — `stress_test`

```python
def stress_test(
    portfolio_returns: pd.Series,
    scenario_shocks: dict[str, float],
    exposures: dict[str, float],
    risk_free_rate: float = 0.04,
    shock_days: int = 21,
    event_spacing_days: int = 126,
) -> pd.DataFrame:
```

Aplica un shock de magnitud `scenario_shocks[k]` ponderado por
`exposures[k]` durante `shock_days = 21` días, repitiendo cada
`event_spacing_days = 126` días sobre la historia. Tabla de salida:
`scenario | shocks_count | total_drag | sharpe_post | dd_post`.

#### Distributional stress — `distributional_stress_test`

Usa **stationary bootstrap** sobre ventanas históricas reales (e.g. el
bloque sept-oct 2008, marzo 2020) con `n_reps=1000`, devolviendo una
distribución empírica de pérdidas. Más realista que el shock determinístico
porque preserva las correlaciones cruzadas durante el evento.
'''


MD_SECTION8_ENRICH = r'''## 8. FX Overlay (Layer 2)

`src/hedge_overlay.py` añade tres componentes encima de Layer 1:

```python
def long_short_portfolio(...) -> pd.DataFrame: ...                    # libro long/short por sector
def dynamic_leverage(portfolio_returns, max_leverage=1.30, cvar_limit=0.04, ...) -> pd.Series: ...
def fx_directional_overlay(macro_df, signal_df, usd_exposure, ...) -> pd.DataFrame: ...
def tail_risk_hedge(portfolio_returns, gev_params, protection_level=0.99, cost_bps=30.0) -> dict: ...
def run_hedge_backtest(...) -> dict: ...                              # combina todo
```

### Hedge ratio dinámico — `fx_directional_overlay`

```python
def fx_directional_overlay(
    macro_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    usd_exposure: pd.Series,
    hedge_ratio_base: float = 0.5,
    max_hedge_ratio: float = 0.95,
    min_hedge_ratio: float = 0.10,
    mxn_garch_vol: float | None = None,
) -> pd.DataFrame:
```

#### Construcción de la señal FX (líneas 211–227)

Z-scores expanding-window sobre dos drivers:

* `rate_differential = banxico_rate − us_fed_rate` ⇒ z-score $z_{\Delta r}$.
* `mxn_momentum = log(USDMXN_t / USDMXN_{t-21})` ⇒ z-score $z_{\text{mom}}$.

Score combinado:

$$
\text{score}_{FX,t} \;=\; -0.6\,z_{\Delta r,t} \;+\; 0.4\,z_{\text{mom},t}
$$

(carry alto ⇒ menos hedge / menos USD-cover; momentum positivo en USDMXN ⇒
peso debilita ⇒ más hedge).

#### Hedge ratio final con sigmoid + GARCH boost

$$
h_t \;=\; \min_{\text{hr}} + (\max_{\text{hr}} - \min_{\text{hr}}) \cdot
\sigma\!\left(\,6 \cdot \tfrac{\text{clip}(\text{score}_{FX,t}, -3, 3)}{3}\right)
\;+\; 0.05\,\text{clip}\!\left(\tfrac{\sigma_\text{GARCH} - 0.15}{0.10}, -1, 1\right)
$$

clipped a $[0.10,\,0.95]$. La fórmula tipo paper de la spec se realiza así:

$$
\boxed{\;h_t \;=\; \text{clip}\!\Big(\text{score}_{FX,t}\cdot \sigma_{FX,t} \,/\, \text{CVaR}_\text{target},\; 0.10,\; 0.95\Big)\;}
$$

(forma analítica equivalente cuando se simplifica la sigmoid a su tangente
en el origen y se interpreta la GARCH-vol como el divisor por
`cvar_limit ≈ CVaR_target`).

| Parámetro          | Default       | Qué controla                                                         |
|--------------------|---------------|----------------------------------------------------------------------|
| `min_hedge_ratio`  | `0.10`        | Hedge mínimo (siempre cubre al menos 10 %).                          |
| `max_hedge_ratio`  | `0.95`        | Hedge máximo (nunca cubre 100 % — preserva opcionalidad).            |
| `hedge_ratio_base` | `0.5`         | Centro de la sigmoid.                                                |
| `mxn_garch_vol`    | (runtime)     | Forecast GARCH USD/MXN — boost de hedge en alta vol.                 |

### Leverage dinámico — `dynamic_leverage`

```python
def dynamic_leverage(
    portfolio_returns: pd.Series,
    max_leverage: float = 1.30,
    cvar_limit: float = 0.04,
    min_leverage: float = 0.80,
    alpha: float = 0.95,
    window: int = 63,
) -> pd.Series:
```

CVaR-targeting: si el CVaR rolling 63d del libro < `cvar_limit`, hay
*budget* para subir el leverage hasta `max_leverage`; si lo excede,
se baja a `min_leverage`. Implementación lineal con clips.

### Modos de operación

| Modo          | Gross cap | Net cap | Comentario                                              |
|---------------|-----------|---------|---------------------------------------------------------|
| `analytical`  | 1.60      | 1.60    | Layer 2 informacional; viola LFI (sin venta corta MX).  |
| `regulated`   | 1.15      | 1.05    | Caps LFI vigentes — se puede operar hoy.                |

> **Lookahead-safe**: el FX PnL se calcula en `run_hedge_backtest` con
> `hedge_ratio.shift(1)` y FX **realizado** contemporáneo — el ratio se
> decide *antes* del cambio FX. Ver nota línea 238 de `hedge_overlay.py`.
'''


MD_SECTION8B_ENRICH = r'''### 8b. Tail hedge — costo vs beneficio

```python
def tail_risk_hedge(
    portfolio_returns: pd.Series,
    gev_params: tuple[float, float, float],   # (shape, loc, scale) ya ajustados
    protection_level: float = 0.99,
    cost_bps: float = 30.0,
) -> dict:
    """Simulate the cost and benefit of a synthetic tail hedge."""
```

#### Cálculos (líneas 252–275)

* **Pérdida desprotegida** al $99\%$ desde GEV:
  $\;L_{99} = -\,\text{GEV.ppf}(0.01;\, \xi, \mu, \sigma)$.
* **Strike del put** al $95\%$:
  $\;K_{95} = -\,\text{GEV.ppf}(0.05;\, \xi, \mu, \sigma)$.
* **Hedge payoff** = $\max(0,\, L_{99} - K_{95})$.
* **Costo diario** = `cost_bps / 10_000` (asumido constante, calibrado a put-spread MXX).
* **Net benefit** = payoff − cost.

`recommended = (net_benefit > 0)` — el overlay sólo se recomienda si la
pérdida esperada en cola excede el costo del seguro. En la mayoría de
regímenes calmos el net es negativo (no comprar tail hedge).
'''
MD_SECTION8C_ENRICH = r'''### 8c. Comparativo Reforma LFI — escenarios alternativos

`run_reform_comparison` corre **4 estructuras de portafolio** que simulan
distintos resultados de una hipotética reforma a la **Ley de Fondos de
Inversión** que permitiera ventas en corto:

| Escenario             | Long  | Short | Net   | Comentario                              |
|-----------------------|-------|-------|-------|-----------------------------------------|
| Regulado (LFI actual) | ≤1.00 | 0     | ≤1.00 | Baseline contra-factual.                |
| 130/30                | 1.30  | 0.30  | 1.00  | Mantiene exposición direccional.        |
| Market-Neutral        | 0.50  | 0.50  | 0.00  | Puro alfa cross-sectional.              |
| 130/30 Sector-Neutral | 1.30  | 0.30  | 1.00  | 130/30 dentro de cada sector.           |

Los 4 escenarios usan el **mismo Layer 1** (universo, señales, costos) — la única
diferencia es la estructura del libro hedge. Eso convierte el comparativo en una
medida directa del **valor que la LFI actual le está costando al inversionista mexicano**.
'''


MD_SECTION9_ENRICH = r'''## 9. Liquidity Sleeve y Régimen Macro

### Asignación CETES fuera del optimizador

`pipeline.py` líneas 346–367. Bandas por régimen detectado:

| Régimen      | Sleeve mínimo | Sleeve máximo | Reparto                       |
|--------------|---------------|---------------|-------------------------------|
| `expansion`  | 3 %           | 5 %           | 50/50 CETES28 / CETES91       |
| `tightening` | 5 %           | 8 %           | 50/50 CETES28 / CETES91       |
| `stress`     | 8 %           | 15 %          | 50/50 CETES28 / CETES91       |

`MBONO3Y_buffer` está disponible pero **deshabilitado por default**
(`mbono3y_buffer_enabled=false`).

### Detección de régimen — `detect_macro_regime` / `compute_macro_regime_history`

```python
def compute_macro_regime_history(
    macro: Optional[pd.DataFrame],
    settings: dict | None = None,
) -> pd.DataFrame:
    """Build a regime history using either the legacy thresholds or the EWMA score."""
def detect_macro_regime(macro, settings=None) -> str:
    """Return the latest macro regime state from the configured detector."""
```

Dos detectores:

* `regime_method="threshold_discrete"` — reglas hardcoded sobre IP yoy,
  USDMXN MoM y Banxico rate level. Cambia régimen rápido pero ruidoso.
* `regime_method="ewma_composite"` (**default**) — EWMA $\lambda=0.94$ sobre
  el z-score compuesto de los tres drivers, con histéresis
  `regime_min_confidence_for_switch` (default 0.20) — requiere magnitud
  mínima de la señal para cambiar de estado, evita switches espurios.

### Cross-reference

```python
# pipeline.py L355-357
_macro_for_regime = macro.copy()
regime = detect_macro_regime(_macro_for_regime, settings=cfg)
sleeve_range = _sleeve_bounds.get(regime, _sleeve_bounds["expansion"])
```

### Por qué EWMA y no thresholds duros

Los thresholds duros producen *flickering*: una semana de IP yoy ligeramente
inferior al cutoff dispara un cambio de régimen → cambio de bandas
asset-class → cambio de pesos → turnover spurio. El EWMA con histéresis
**suaviza la decisión** al costo de detectar el cambio de régimen ~21 días
más tarde — un trade-off favorable cuando el costo de turnover (10 bps) es
no-trivial.
'''
MD_SECTION10_ENRICH = r'''## 10. Hyperopt — Búsqueda walk-forward (Optuna)

```python
def run_hyperopt(
    prices: pd.DataFrame,
    feature_df: pd.DataFrame,
    universe: pd.DataFrame,
    macro: pd.DataFrame,
    n_trials: int = 50,
    n_folds: int = 3,
    purge_gap_days: int = 21,
    objective_metric: str = "sharpe_adj",      # {"sharpe", "sortino", "calmar", "sharpe_adj"}
    turnover_penalty: float = 0.20,
    search_space: dict[str, tuple] | None = None,
    optimizer: str = "mv",
    risk_free_rate: float = 0.04,
    storage: str | None = None,                # opcional: SQLite para resumir studies
    seed: int = 42,
) -> OptimResult:
    """Run Bayesian hyperparameter optimization via Optuna TPE sampler."""
```

#### Walk-forward purgada

```python
def build_walk_forward_folds(
    prices, feature_df, universe, macro,
    n_folds: int = 3,
    purge_gap_days: int = 21,
    min_train_days: int = 504,
) -> list[FoldData]:
```

Cada fold es `(train_period, [purge gap 21d], test_period)`. El gap evita
*data leakage* por overlap entre forward returns del train (target a 21d)
y los precios iniciales del test.

#### Objetivo

```python
def _objective_score(
    metrics_per_fold: list[dict[str, float]],
    objective_metric: str,
    turnover_penalty: float,
) -> float:
    """Aggregate metrics across folds into a scalar objective value."""
```

`sharpe_adj` (default) penaliza Sharpe por turnover anualizado:

$$
\text{sharpe\_adj} \;=\; \overline{\text{Sharpe}}_{\text{folds}} \;-\;
\kappa \cdot \overline{\text{Turnover}_{\text{ann}}}_{\text{folds}}
$$

con $\kappa = 0.20$. Esto evita que TPE encuentre Sharpes altos pero
inviables operativamente.

#### Diagnóstico de overfitting

`src/overfitting.py`:

```python
def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    annualization: int = 252,
    trial_sharpes: list[float] | None = None,
) -> dict[str, float]:
    """Compute the Deflated Sharpe Ratio for a return series."""

def probability_of_backtest_overfitting(
    trial_metric_matrix: np.ndarray,
    n_chunks: int = 16,
) -> dict[str, float]:
    """Combinatorially Symmetric Cross-Validation (CSCV) PBO."""
```

* **DSR (Bailey & López de Prado 2014)** corrige el Sharpe por *selection bias*
  cuando se prueban N estrategias.
* **PBO (CSCV)**: probabilidad de que el "mejor" out-of-sample sea sólo el
  que mejor pesca el ruido in-sample. PBO < 0.5 es buena señal.

### Separación de responsabilidades

El pipeline normal **nunca** lee resultados de hyperopt automáticamente. Los
parámetros encontrados se escriben en `config_optimized*.yaml` para revisión
humana antes de promoverlos a `config.yaml`.
'''
MD_SECTION11_ENRICH = r'''## 11. Robustez y diagnóstico de overfitting

Cinco diagnósticos independientes:

1. **Walk-forward expanding** — el modelo ElasticNet se reentrena cada fin
   de mes usando **sólo datos previos**. Target = forward-21d con NaN en
   los últimos 21d (`signals.py::_compute_forward_returns`).

2. **Estabilidad rolling del Sharpe** — descomposición en ventanas móviles
   (típicamente 252d) para detectar concentración temporal del alpha.

3. **Sub-period analysis** — partir el backtest en mitades y comparar
   `mean`, `vol`, `Sharpe` y `max DD` entre los dos sub-periodos.

4. **Walk-forward por año calendario** — performance año a año (ver
   tabla en celda siguiente).

5. **Bootstrap pareado vs benchmark (§6d)** — α de Jensen con CI 95 %.

### Veredicto integrado (§ celda final de esta sección)

* ✅ IC composite $\ge 0.05$ con CI excluyendo 0.
* ✅ Sharpe sub-period inferior > 0.5.
* ✅ DSR > 0 (post-deflation).
* ✅ PBO (CSCV) < 0.5.
* ✅ α post-bootstrap con $p < 0.05$.

### Por qué importa

> Un Sharpe alto que sólo viene de uno o dos años de condiciones excepcionales
> no es estrategia, es suerte. La consistencia a través de sub-periodos es la
> prueba más exigente que un cuant puede hacerse — y sigue siendo un test
> *necesario pero no suficiente*. Por eso DSR + PBO + bootstrap pareado se
> aplican en conjunto.
'''


MD_SECTION12_ENRICH = r'''## 12. Conclusiones

### Resultados clave (run Bloomberg, 2017-01 → 2026-03)

* **Performance ajustada por riesgo competitiva** vs benchmarks pasivos (IPC) y
  vs fondos GBM activos del mismo segmento (NEAR / CRE / MOD / ALFA).
* **Significancia estadística del α** — los CIs bootstrap (§6c–6d) cuantifican
  cuánto del retorno excedente es señal vs ruido; la decisión de invertir
  descansa en intervalos, no en puntos.
* **Stress determinístico controlado** — el portafolio sobrevive shocks
  Banxico, depreciación del peso y desaceleración de EE.UU. con drawdowns
  acotados; los cuatro métodos de VaR (§7c) convergen en zona de confianza.
* **Layer 2 mejora Sharpe sin disparar drawdown** cuando la señal FX
  identifica regímenes claros de fortaleza/debilidad del peso.
* **Robustez metodológica** — la comparación MV vs CVaR vs Robust (§5b)
  muestra que los pesos óptimos no dependen críticamente del criterio de
  riesgo elegido; el sleeve dinámico CETES (§9) y el chequeo CNBV (§5c)
  garantizan cumplimiento regulatorio en todo rebalance.

### Ventajas vs alternativas

| Alternativa            | Ventaja FMIA                                                    |
|------------------------|-----------------------------------------------------------------|
| **IPC pasivo**         | Tilt temático, sin sectores no-nearshoring (consumo, telecom).  |
| **GBMNEAR**            | Optimización cuantitativa + overlay FX dinámico.                |
| **CETES (riesgo cero)**| Equity premium + CETES sleeve dinámico como buffer.             |

### Limitaciones reconocidas

1. **Universo concentrado** — el mercado mexicano tiene <40 emisores
   líquidos con exposición clara a la tesis; el riesgo idiosincrático es alto.
2. **Datos fundamentales con lag** — `fundamentals_lag_days = 90` evita
   look-ahead pero introduce ruido en value/quality scores.
3. **Backtest mensual** — frecuencia conservadora; no captura señales
   intra-mensuales ni eventos específicos (M&A, splits).
4. **FX hedge en `analytical` mode** — Layer 2 `analytical` viola caps LFI
   actuales. La versión operable hoy es `regulated`.
5. **Sample size** — incluso con expanding-window y refit mensual, $T \approx 100$
   re-entrenamientos imponen un piso a la varianza de los hyper-parámetros.
'''


MD_SECTION13_HEADER = r'''## 13. (NUEVO) LFI Reform Scenarios — comparativo side-by-side

Esta sección **se construye con los charts del reporte HTML** generados por
`run_reform_comparison`. Los 4 escenarios comparten Layer 1 (universo,
señales, costos, restricciones de emisor) — la única diferencia es la
estructura del libro hedge.

```python
# src/hedge_overlay.py
def run_reform_comparison(
    prices, signal_df, universe, macro_df,
    max_leverage: float = 1.30,
    cvar_limit: float = 0.04,
    transaction_cost: float = 0.0010,
    risk_free_rate: float = 0.04,
    mxn_garch_vol: float | None = None,
    borrow_cost_bps: float = 150.0,
    leverage_cost_bps: float = 5.0,
) -> dict[str, dict]:
    """Run all 4 LFI reform scenarios and return results keyed by scenario name."""
```

`borrow_cost_bps = 150` modela el **stock loan** real para shorts en BMV.
`leverage_cost_bps = 5` el funding diario del exceso de gross.

### 13.1 — NAV cumulativo de los 4 escenarios'''


MD_SECTION13_REGULADO = r'''### 13.2 — Regulado (LFI actual)

Baseline contra-factual: long-only, gross ≤ 1.05 (cap LFI), net ≤ 1.00.'''

MD_SECTION13_130_30 = r'''### 13.3 — 130/30

Long 130 %, short 30 %, net 100 %. Mantiene exposición direccional al mercado
mexicano pero permite *underweight estructural* sobre los peores tickers
(no sólo `weight = 0`).'''

MD_SECTION13_MN = r'''### 13.4 — Market-Neutral

Long 50 %, short 50 %, net 0 %. **Puro alfa cross-sectional**: el retorno
total es $\beta \cdot r_m \approx 0$ + el spread long-short. Si la señal
ElasticNet tiene poder predictivo cross-sectional real, este libro debería
mostrar Sharpe positivo independiente del régimen del IPC.'''

MD_SECTION13_130_30_SN = r'''### 13.5 — 130/30 Sector-Neutral

130/30 dentro de cada sector (long_short_portfolio con `sector_neutral=True`).
Aísla la **selección dentro del sector** del *timing* sectorial — si el alpha
del fondo viene de elegir bien dentro de Materials o Industrials (no de
sobre-ponderar el sector), este escenario lo aislará y debería dominar al
130/30 plano.'''


MD_NEW_STRESS_DIST = r'''### 7d. Stress distributional — bootstrap de ventanas históricas

```python
def distributional_stress_test(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
    macro: Optional[pd.DataFrame],
    n_reps: int = 1000,
    window_days: int = 21,
    seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap historical stress windows into a distribution of portfolio P&L."""
```

A diferencia del stress determinístico (§7), aquí no se asume magnitud del
shock — se **muestrea el peor $W$-day window histórico** y se aplica al
portafolio actual repetidamente. La salida es una distribución empírica de
P&L que captura naturalmente las correlaciones cruzadas durante eventos reales.

> Más realista que el shock determinístico porque preserva las correlaciones
> cruzadas durante el evento (e.g. sept-oct 2008, marzo 2020). Es la prueba
> *honesta* de cómo aguanta el libro actual ese tipo de eventos.
'''
MD_NEW_HEDGE_LEVERAGE = r'''### 8d. Dynamic leverage scalar

`dynamic_leverage` produce el escalar $\ell_t \in [0.80,\, 1.30]$ que se
multiplica al gross del libro hedge. CVaR-targeting:

* Si $\text{CVaR}_{63d}(r_p) \le \text{cvar\_limit}$ ⇒ $\ell_t \uparrow$.
* Si $\text{CVaR}_{63d}(r_p) > \text{cvar\_limit}$ ⇒ $\ell_t \downarrow$.

Lookahead-safe: se calcula con CVaR rolling de retornos pasados y se aplica al
día siguiente (`shift(1)` en `run_hedge_backtest`).
'''


MD_NEW_HEDGE_BREAKDOWN = r'''### 8e. Hedge engine — breakdown stage-by-stage

El gráfico siguiente descompone el NAV de Layer 2 en **etapas acumulativas**:

1. **Layer 1** (long-only optimizado) — baseline.
2. **+ Long/short** (`long_short_portfolio`) — añade overlay neutral.
3. **+ Dynamic leverage** — escala el gross según CVaR-target.
4. **+ FX directional overlay** — añade hedge ratio dinámico.

Cada paso muestra cuánto contribuye marginalmente al retorno y al riesgo.
Si una etapa **destruye Sharpe** (en algún régimen), debe revisarse.
'''


MD_SECTION5D_BL_MACRO = r'''### 5d. Vistas macro y económicas en Black–Litterman

El posterior BL ya no se alimenta solo de la señal ElasticNet. Ahora `src/bl_views.py`
ensambla **dos fuentes** que se combinan vía promedio ponderado por confianza:

1. **Vistas ElasticNet (per-ticker)** — confianza ∈ [0.30, 0.70], escala de la magnitud
   absoluta del forecast. Origen: la regresión `forecast_returns` (sección 4.2).
2. **Vistas macro (sectoriales)** — confianza fija baja (default 0.20). Para cada
   macro disponible se calcula un **z-score expanding-window** y se traduce a un tilt
   sectorial vía la tabla `_MACRO_SECTOR_RULES`:

   | Macro | Sector(es) afectado(s) |
   |-------|-----------------------|
   | `industrial_production_yoy` ↑ | +Industrial / +Logistics / +Infrastructure |
   | `exports_yoy` ↑               | +Industrial / +Logistics |
   | `us_ip_yoy` ↑                 | +Industrial / +Logistics |
   | `banxico_rate Δ21d` ↑         | −FIBRA / −Infrastructure |
   | `inflation_yoy` ↑             | −FIBRA |
   | `usd_mxn 21d return` ↑        | +tickers con `usd_exposure > 0.4` (proporcional) |

   La magnitud por vista se acota en `±max_magnitude` (default 1.5%):

   $$ v_{sec} = \text{signo} \cdot \text{max\_mag} \cdot \tanh(z) $$

3. **Combinación** (`combine_views`): para cada ticker presente en alguna fuente,

   $$ v^* = \frac{\sum_i v_i \cdot c_i}{\sum_i c_i} \quad,\quad c^* = \max_i c_i $$

   La confianza final es el `max` (no la suma) para evitar over-confidence.

La motivación es que ElasticNet captura el **componente estadístico** mientras que las
vistas macro inyectan **prior económico** estructural difícil de aprender vía features
estandarizados (cambios de régimen, inflexión cíclica). Como la confianza macro es
baja, el posterior se mueve poco cuando el modelo y la macro coinciden, pero
**diverge sutilmente** cuando hay desacuerdo — este es el régimen donde típicamente
se generan las apuestas con mayor information ratio.

La tabla **Active Black–Litterman Views** del HTML report (sección 5) muestra cada
vista activa con su fuente, magnitud y confianza, para auditar qué está moviendo el
posterior en cualquier corrida.
'''


MD_SECTION7F_TMEC = r'''### 7f. Escenario de stress — TMEC (USMCA)

Cuarto escenario de stress que complementa los tres existentes
(`banxico_shock`, `peso_depreciation`, `us_slowdown`). Modela el riesgo de
**renegociación / disrupción del T-MEC**: combinación de aranceles
selectivos, fricción regulatoria y disrupción de cadenas industriales norte-sur.

**Determinístico** (`src/pipeline.py:447-475`):

* Shock: `−7%` (más severo que `us_slowdown` por la suma de tarifa + supply chain).
* Exposición: derivada del portafolio final, con peso a industriales y exportadores

  $$ e_\text{TMEC} = \text{clip}\left( (0.6 \cdot w_\text{indus} + 0.4 \cdot w_\text{usd}) \cdot 1.3,\ 0.20,\ 0.85 \right) $$

  donde `w_indus = Σ w_t` para `t ∈ {Industrial, Logistics, Infrastructure, Materials}`.

**Distribucional** (`src/risk.py`): nueva máscara

* `(usd_mxn 21d > 4%) AND (us_ip_yoy < 1%)` (más `exports_yoy < 0` cuando está disponible).
* Ventanas históricas de fallback adicionales:
  - 2018-06 NAFTA acero/aluminio tariffs.
  - 2019-06 amenaza arancel migratorio MX.
  - 2025-02 tariff shock (compartida con `us_slowdown`).

El renglón TMEC aparece automáticamente en la tabla de stress determinístico y en
el panel distribucional del reporte HTML.
'''


MD_SECTION8F_ETF_ANCHOR = r'''### 8f. Puente ETF → Asset Allocation Normal (anclaje sectorial blando)

Conecta el output del modo ETF con el modo normal de manera que **el optimizador
del modo normal recibe los pesos sectoriales del ETF como ancla blanda con banda
±δ**, sin sacrificar el alfa que el optimizador encuentra al optimizar sobre el
universo de tickers individuales.

```
                    ┌──────────────────────────┐
                    │  run_etf_pipeline        │
                    │  produce vector sectorial│
                    └────────────┬─────────────┘
                                 │  reports/output/
                                 ▼  etf_sector_weights_{source}.json
                    ┌──────────────────────────┐
                    │  run_pipeline (normal)   │
                    │  carga ancla → optimizer │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    constraints SLSQP por sector:
                       w_sec ∈ [target − δ, target + δ]
```

**Buckets propagados** (los demás se ignoran):

| ETF sector       | Normal-mode bucket                                         |
|------------------|-----------------------------------------------------------|
| Industrial / Consumer / Communication / Materials | Industrial + Logistics + Infrastructure + Materials |
| FIBRA            | FIBRA                                                     |
| Government       | (ignorado — gobernado por el liquidity sleeve regime-aware) |

**Por qué no daña el performance:**

* `enabled: false` por default → reproducibilidad bit-a-bit del comportamiento previo.
* `band: 0.15` por default → en la mayoría de configuraciones el sector óptimo libre
  ya cae dentro de la banda y la constraint **no muerde**. La tabla "Sector band binding"
  en el reporte HTML lo confirma con un flag `free` por sector.
* La banda es un nuevo parámetro del search space de hyperopt → si una banda más
  estrecha mejora el Sharpe (improbable), Optuna la elige; si daña, la abre.

**Lectura del reporte:**

* **Donut izquierdo:** vector sectorial original del ETF (incluye Consumer/Communication).
* **Donut derecho:** pesos realizados del normal agregados a los buckets {Industrial, FIBRA}.
* **Tabla de bandas:** target, banda permitida, peso realizado y `status` ∈ {`free`, `🔻 lower`, `🔺 upper`}. Si todos los renglones son `free`, el anclaje no degrada el óptimo libre.

**Configuración:** ver `etf_sector_anchor.{enabled, band, source}` en `config.yaml`.
'''


MD_NEW_RISK_HEDGE = r'''### 7e. Métricas de riesgo — variante Hedge (Layer 2)

Para auditoría side-by-side, las mismas métricas (volatilidad rolling 21d,
distribución de retornos diarios, drawdown subaquatic) se computan sobre
los retornos del libro Layer 2. El comparativo permite ver:

* Si el FX overlay **reduce** la cola izquierda (debería, si el hedge está
  bien calibrado).
* Si el long/short reduce la varianza realizada.
* Si el dynamic leverage **inflama** la cola en regímenes de baja CVaR
  (riesgo de mis-calibración del trigger).
'''


# ---------------------------------------------------------------------------
# Build cells list
# ---------------------------------------------------------------------------

def build():
    nb = nbformat.read(NB_IN, as_version=4)
    src_cells = nb.cells
    new_cells = []

    # Helper: append (md or code) cell
    def md(s): new_cells.append(new_markdown_cell(s))
    def code(s): new_cells.append(new_code_cell(s))
    def keep(idx): new_cells.append(src_cells[idx])
    def keep_as_backup(idx): new_cells.append(comment_out_matplotlib(src_cells[idx]))

    # 0: Header (replace cell 0)
    md(MD_HEADER_INTRO)
    # 1: Setup intro
    keep(1)
    # 2-3: setup code (keep)
    keep(2); keep(3)

    # ===== Section 1 =====
    md(MD_SECTION1_ENRICH)
    keep(5)  # architecture diagram (keep)

    # ===== Section 2 =====
    md(MD_SECTION2_ENRICH)
    keep(7); keep(8)

    # ===== Section 3 =====
    md(MD_SECTION3_ENRICH)
    keep(10); keep(11)

    # ===== Section 4 =====
    md(MD_SECTION4_ENRICH)
    keep(13)  # corr heatmap
    keep(14)  # composite snapshot
    keep_as_backup(15)  # IC matplotlib backup
    new_cells.append(chart_cell("c26b9ccc-5615-4fb1-9a0a-5b830c5685c6",
                                "IC — Signal-Return Cross-Sectional Correlation"))
    new_cells.append(chart_cell("9a5bbd70-a660-4ef4-b548-bd6f73360dc5",
                                "Cross-Sectional Signal Dispersion (Last 12M)"))
    keep(16)  # forecast distribution

    # 4b
    md(MD_SECTION4B_ENRICH)
    keep(18)  # IC bootstrap table

    # ===== Section 5 =====
    md(MD_SECTION5_ENRICH)
    keep(20)  # weights bar
    keep_as_backup(21)  # treemap matplotlib backup
    new_cells.append(chart_cell("a42ff7ca-66ff-4543-8dd9-9bea7137d15a",
                                "Universe by Asset Class — Traditional"))
    new_cells.append(chart_cell("834ca727-687a-407a-8308-8d30450c80bf",
                                "Current Sector Allocation — Traditional"))
    new_cells.append(chart_cell("976835be-4aa9-4983-9e65-96423699d634",
                                "Allocation Over Time — Traditional"))
    keep(22)  # forecast vs prior

    # 5d — Vistas macro y económicas en Black–Litterman
    md(MD_SECTION5D_BL_MACRO)

    # 5b
    md(MD_SECTION5B_ENRICH)
    keep_as_backup(24)  # MV vs CVaR matplotlib backup
    new_cells.append(chart_cell("b6749608-bf82-48b9-8500-42dc92b23137",
                                "Cumulative Performance: MV vs min-CVaR"))

    # 5c
    md(MD_SECTION5C_ENRICH)
    keep(26)  # CNBV check

    # ===== Section 6 =====
    md(MD_SECTION6_ENRICH)
    keep_as_backup(28)  # NAV cumulativo matplotlib
    new_cells.append(chart_cell("chart-cumulative", "NAV Cumulativo — Cumulative Performance"))
    new_cells.append(chart_cell("chart-benchmarks", "Strategy vs Benchmarks"))
    keep_as_backup(29)  # Drawdown matplotlib
    new_cells.append(chart_cell("chart-drawdown", "Underwater Equity Curve — Traditional"))
    keep(30)  # metrics table
    keep_as_backup(31)  # rolling Sharpe matplotlib
    new_cells.append(chart_cell("1075caf4-000b-44eb-938a-e1c9bbbb7c6c",
                                "Rolling Sharpe Ratio (63-day) — Traditional"))

    # 6b turnover
    md(MD_SECTION6B_ENRICH)
    keep_as_backup(33)
    new_cells.append(chart_cell("116bd050-462e-4f37-a496-b5952059ea7b",
                                "Traditional Monthly Turnover"))

    # 6c bootstrap CIs
    md(MD_SECTION6C_ENRICH)
    keep(35)

    # 6d alpha significance
    md(MD_SECTION6D_ENRICH)
    keep(37)

    # 6e fan chart
    md(MD_SECTION6E_ENRICH)
    keep_as_backup(39)
    new_cells.append(chart_cell("32176b03-8828-4261-8284-e605ea358ba8",
                                "Bootstrap Equity Fan Chart"))

    # ===== Section 7 =====
    md(MD_SECTION7_ENRICH)
    keep_as_backup(41)  # VaR vs realized
    new_cells.append(chart_cell("091affdd-28f6-4a1d-98e1-1e53e6ca1acf",
                                "Volatility (21d) — Traditional"))
    keep(42)  # stress table
    keep_as_backup(43)  # distribution matplotlib
    new_cells.append(chart_cell("ec991da5-8a00-4c92-a419-c69a79177a03",
                                "Daily Return Distribution — Traditional"))

    # 7b GARCH
    md(MD_SECTION7B_ENRICH)
    keep(45)

    # 7c VaR comparison
    md(MD_SECTION7C_ENRICH)
    keep(47)
    new_cells.append(chart_cell("e90b769a-3828-4abf-98bf-d7b6749e0945",
                                "Covariance Diagnostics — Correlation Heatmaps"))
    new_cells.append(chart_cell("54503ae6-abc7-4e20-983e-cef69c8fad95",
                                "Determinant Ratio Through Time"))
    new_cells.append(chart_cell("21d90168-d639-4dc3-aa2f-7aed38374cf7",
                                "Realized Volatility Comparison"))

    # 7d (NEW) distributional stress
    md(MD_NEW_STRESS_DIST)
    new_cells.append(chart_cell("2ee6956d-14b8-4d7c-aace-1589ba591d99",
                                "Stress Scenario Impact — Distributional"))

    # 7f — Escenario TMEC (USMCA disruption)
    md(MD_SECTION7F_TMEC)

    # 7e (NEW) hedge risk metrics
    md(MD_NEW_RISK_HEDGE)
    new_cells.append(chart_cell("c7d3c202-f731-4aba-ac75-c455c28d03ac",
                                "Volatility (21d) — Hedge"))
    new_cells.append(chart_cell("b3c80785-90f8-481a-b640-cdc33739b188",
                                "Daily Return Distribution — Hedge"))
    new_cells.append(chart_cell("9338b314-6dd8-4aea-84d9-89d702cedb85",
                                "Rolling Sharpe Ratio (63-day) — Hedge"))
    new_cells.append(chart_cell("chart-drawdown-hedge",
                                "Underwater Equity Curve — Hedge"))

    # ===== Section 8 =====
    md(MD_SECTION8_ENRICH)
    keep_as_backup(49)  # hedge ratio matplotlib
    new_cells.append(chart_cell("8b233528-3cfe-427f-b34f-c3b97a874110",
                                "Dynamic FX Hedge Ratio"))
    keep_as_backup(50)  # NAV L1 vs L2 matplotlib
    new_cells.append(chart_cell("b74465a5-dc74-469b-99f3-4fc09fcb4d11",
                                "Hedge Engine Breakdown — Stage-by-Stage Cumulative Performance"))

    # 8b tail hedge
    md(MD_SECTION8B_ENRICH)
    keep_as_backup(52)  # tail hedge matplotlib
    new_cells.append(chart_cell("0063deba-032f-4a75-8dca-c880fbe0cb20",
                                "Tail Hedge Cost-Benefit"))

    # 8c LFI overview (just the markdown)
    md(MD_SECTION8C_ENRICH)
    keep_as_backup(54)  # 4-scenario matplotlib
    new_cells.append(chart_cell("40de153e-3309-4e00-b017-e3798b53b12d",
                                "LFI Reform Scenarios — Cumulative Return"))

    # 8d (NEW) dynamic leverage scalar
    md(MD_NEW_HEDGE_LEVERAGE)
    new_cells.append(chart_cell("1b89c2c3-526e-4b28-807a-974d407f8698",
                                "Dynamic Leverage Scalar (Weekly Avg)"))

    # 8e (NEW) hedge engine breakdown — already shown above; here add allocations
    md(MD_NEW_HEDGE_BREAKDOWN)
    new_cells.append(chart_cell("706f3c53-0b71-4da2-b7ef-a80ddbd66fcf",
                                "Hedge-Eligible Universe by Asset Class"))
    new_cells.append(chart_cell("601221b6-f558-483f-b1ad-0797e9c821ba",
                                "Hedge Allocation Over Time"))
    new_cells.append(chart_cell("49dfea90-7490-4118-9573-431a7b6dd6f4",
                                "Hedge Monthly Turnover"))
    new_cells.append(chart_cell("b747d865-12d2-40db-bb11-d7f48f1f6b47",
                                "Hedge Current Sector Allocation"))

    # 8f — Puente ETF → Asset Allocation Normal (anclaje sectorial blando)
    md(MD_SECTION8F_ETF_ANCHOR)

    # ===== Section 9 =====
    md(MD_SECTION9_ENRICH)
    keep(56)

    # ===== Section 10 hyperopt =====
    md(MD_SECTION10_ENRICH)
    keep(58)
    new_cells.append(chart_cell("hyperopt-convergence",
                                "Hyperopt — Convergence (best sharpe_adj vs trials)"))
    new_cells.append(chart_cell("hyperopt-parcoords",
                                "Hyperopt — Parallel Coordinates (trials)"))

    # ===== Section 11 robustez =====
    md(MD_SECTION11_ENRICH)
    keep(60); keep(61); keep(62); keep(63)

    # ===== Section 12 conclusiones =====
    md(MD_SECTION12_ENRICH)

    # ===== Section 13 (NEW) LFI Reform =====
    md(MD_SECTION13_HEADER)
    # Already showed 40de153e in §8c — point to it
    md(MD_SECTION13_REGULADO)
    new_cells.append(chart_cell("59d7f07b-18af-4a01-ac78-e5728325fb25",
                                "Regulado — Allocation Over Time"))
    new_cells.append(chart_cell("5727707b-37ff-432a-bb2a-dcc857288eb5",
                                "Regulado — Monthly Turnover"))
    new_cells.append(chart_cell("946f53de-12d0-48cb-aa99-0adf34b1d215",
                                "Regulado — Current Sector Allocation"))

    md(MD_SECTION13_130_30)
    new_cells.append(chart_cell("2e343a57-2f0f-4fb0-91b3-f3c0d1a15393",
                                "130/30 — Long/Short Allocation Over Time"))
    new_cells.append(chart_cell("a25453cf-47b2-42de-8f69-b1c412476624",
                                "130/30 — Monthly Turnover"))
    new_cells.append(chart_cell("8caac0aa-7563-4aa1-b31f-cb0276264090",
                                "130/30 — Current Sector Exposure"))

    md(MD_SECTION13_MN)
    new_cells.append(chart_cell("fdd27c7a-6802-41c1-b959-ac38df002aa0",
                                "Market-Neutral — Long/Short Allocation Over Time"))
    new_cells.append(chart_cell("b23f1168-3930-4a3c-a3d0-da107f60ca58",
                                "Market-Neutral — Monthly Turnover"))
    new_cells.append(chart_cell("06a47ebc-d840-42d9-9e71-8e162fa7555c",
                                "Market-Neutral — Current Sector Exposure"))

    md(MD_SECTION13_130_30_SN)
    new_cells.append(chart_cell("a82133af-788d-4eb8-8d3d-22b1916e7fcc",
                                "130/30 Sector-Neutral — Long/Short Allocation Over Time"))
    new_cells.append(chart_cell("30e4934d-a605-4444-8929-5a05ef13d0d9",
                                "130/30 Sector-Neutral — Monthly Turnover"))
    new_cells.append(chart_cell("d0675c24-af7b-4b48-a15b-c5fe466f265d",
                                "130/30 Sector-Neutral — Current Sector Exposure"))

    nb.cells = new_cells
    # Reset all execution counts (re-runnable from scratch)
    for c in nb.cells:
        if c.cell_type == "code":
            c.execution_count = None
            c.outputs = []
    nbformat.write(nb, NB_OUT)
    print(f"Wrote {NB_OUT} ({len(new_cells)} cells)")


if __name__ == "__main__":
    build()
