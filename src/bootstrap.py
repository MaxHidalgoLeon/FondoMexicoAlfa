from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length


def _as_clean_series(values: pd.Series | np.ndarray | list[float]) -> pd.Series:
    series = pd.Series(values, dtype=float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _sign_p_value(point: float, distribution: np.ndarray) -> float:
    dist = np.asarray(distribution, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or abs(point) < 1e-12:
        return 1.0
    if point > 0:
        return float(np.mean(dist <= 0.0))
    return float(np.mean(dist >= 0.0))


def bootstrap_block_size_selector(returns: pd.Series) -> int:
    """Estima el tamaño óptimo de bloque para el bootstrap estacionario.

    Usa el selector de Patton-Politis-White (PPW) implementado en la librería arch.
    El tamaño de bloque controla cuánta autocorrelación serial se preserva en las
    muestras bootstrap: bloques más grandes = más dependencia temporal preservada.
    Resultado acotado en [5, 60] días como guardarraíles de sentido común.
    """
    clean = _as_clean_series(returns)
    if len(clean) < 10:
        return 20
    try:
        block_df = optimal_block_length(clean.values)
        block = float(block_df["b_sb"].iloc[0])
        if not np.isfinite(block):
            return 20
        return int(np.clip(round(block), 5, 60))
    except Exception:
        return 20


def bootstrap_metric(
    returns: pd.Series,
    metric_fn: Callable,
    block_size: int = 20,
    n_reps: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Calcula una métrica escalar con intervalos de confianza bootstrap estacionario.

    El bootstrap estacionario (Politis-Romano 1994) remuestrea bloques de longitud
    aleatoria (geométrica con media=block_size) para preservar la autocorrelación
    de los retornos financieros, a diferencia del bootstrap i.i.d. clásico.

    Devuelve:
      point        → valor observado de la métrica.
      ci_low/high  → intervalo de confianza percentílico al nivel 'confidence'.
      se           → error estándar bootstrap de la distribución.
      distribution → array de n_reps valores bootstrap de la métrica.
    """
    clean = _as_clean_series(returns)
    point = float(metric_fn(clean)) if len(clean) else np.nan
    if len(clean) < 5:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "se": 0.0,
            "distribution": np.asarray([point], dtype=float),
        }

    block = max(int(block_size), 2)
    bs = StationaryBootstrap(block, clean.values, seed=seed)

    def _wrapped(sample: np.ndarray) -> np.ndarray:
        value = float(metric_fn(pd.Series(np.asarray(sample, dtype=float))))
        return np.asarray([value], dtype=float)

    distribution = bs.apply(_wrapped, reps=int(n_reps)).reshape(-1)
    distribution = distribution[np.isfinite(distribution)]
    alpha = (1.0 - float(confidence)) / 2.0
    if distribution.size == 0:
        ci_low = ci_high = point
        se = 0.0
        distribution = np.asarray([point], dtype=float)
    else:
        ci_low = float(np.quantile(distribution, alpha))
        ci_high = float(np.quantile(distribution, 1.0 - alpha))
        se = float(np.std(distribution, ddof=1)) if distribution.size > 1 else 0.0
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "distribution": distribution,
    }


def bootstrap_paired_difference(
    returns_fund: pd.Series,
    returns_benchmark: pd.Series,
    metric_fn: Callable,
    block_size: int = 20,
    n_reps: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap por pares de la diferencia fondo-benchmark preservando alineación temporal.

    En cada réplica remuestrea los mismos índices para el fondo y el benchmark,
    manteniendo el emparejamiento correcto de fechas.  Esto es crucial para el alpha
    de Jensen, el IR y el tracking error, donde el timing de cada retorno importa.

    Devuelve lo mismo que bootstrap_metric() más 'p_value' (fracción de réplicas
    donde la métrica toma el signo opuesto — test de significancia unilateral).
    """
    aligned = pd.concat(
        [pd.Series(returns_fund, dtype=float), pd.Series(returns_benchmark, dtype=float)],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {
            "point": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "se": np.nan,
            "distribution": np.asarray([], dtype=float),
            "p_value": np.nan,
        }

    fund = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    point = float(metric_fn(fund, bench))
    if len(aligned) < 5:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "se": 0.0,
            "distribution": np.asarray([point], dtype=float),
            "p_value": 1.0,
        }

    block = max(int(block_size), 2)
    bs = StationaryBootstrap(block, fund.values, bench.values, seed=seed)

    def _wrapped(sample_fund: np.ndarray, sample_bench: np.ndarray) -> np.ndarray:
        value = float(
            metric_fn(
                pd.Series(np.asarray(sample_fund, dtype=float)),
                pd.Series(np.asarray(sample_bench, dtype=float)),
            )
        )
        return np.asarray([value], dtype=float)

    distribution = bs.apply(_wrapped, reps=int(n_reps)).reshape(-1)
    distribution = distribution[np.isfinite(distribution)]
    alpha = (1.0 - float(confidence)) / 2.0
    if distribution.size == 0:
        ci_low = ci_high = point
        se = 0.0
        distribution = np.asarray([point], dtype=float)
    else:
        ci_low = float(np.quantile(distribution, alpha))
        ci_high = float(np.quantile(distribution, 1.0 - alpha))
        se = float(np.std(distribution, ddof=1)) if distribution.size > 1 else 0.0
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "distribution": distribution,
        "p_value": _sign_p_value(point, distribution),
    }


def bootstrap_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate stationary-bootstrap return paths for fan-chart style diagnostics."""
    clean = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return np.empty((0, 0))

    block = max(int(block_size), 2)
    bs = StationaryBootstrap(block, clean.values, seed=seed)
    samples: list[np.ndarray] = []
    for data, _ in bs.bootstrap(int(n_paths)):
        samples.append(np.asarray(data[0], dtype=float))
    return np.vstack(samples) if samples else np.empty((0, len(clean)))


__all__ = [
    "bootstrap_block_size_selector",
    "bootstrap_metric",
    "bootstrap_paired_difference",
    "bootstrap_paths",
]
