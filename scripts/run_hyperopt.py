#!/usr/bin/env python
"""
CLI: python scripts/run_hyperopt.py [--n-trials N] [--n-folds K]
                                    [--optimizer mv|cvar] [--output PATH]

Runs Bayesian hyperparameter optimization (Optuna TPE) and writes:
  - reports/hyperopt_data/hyperopt_results_{source}.json — best params,
    trial history (with per-fold OOS sharpes for CSCV-PBO), validation metrics
  - config_optimized_{source}.yaml — ready-to-use config with best params

The hyperopt diagnostics (convergence, parallel coords, importance, top trials,
DSR, PBO) are rendered inside the main strategy report (run_all.py). No standalone
HTML report is emitted.

The optimizer searches over DEFAULT_SEARCH_SPACE (see src/hyperopt.py).
Regulatory keys (max_position, issuer_concentration_limit, fx_overlay_notional_cap,
liquidity_sleeve_*) are excluded — they are CNBV/prospectus-fixed.

Inputs are controlled by config.yaml under the `hyperopt_*` keys, CLI flags
override config entries.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


def _load_yaml_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        loaded = yaml.safe_load(f) or {}
    return loaded


def _dump_yaml(data: dict[str, Any], path: Path) -> None:
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _merge_params_into_config(base_cfg: dict[str, Any], best_params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_cfg)
    for key, value in best_params.items():
        merged[key] = value
    return merged


SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "refinitiv"]


def _normalize_sources(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        candidates = [str(s).strip().lower() for s in raw if str(s).strip()]
    else:
        raw = str(raw).strip().lower()
        candidates = ["yahoo", "refinitiv", "bloomberg"] if raw == "all" else [s.strip() for s in raw.split(",") if s.strip()]
    invalid = [s for s in candidates if s not in SUPPORTED_SOURCES]
    if invalid:
        raise ValueError(f"Fuente(s) inválida(s): {', '.join(invalid)}. Válidas: {', '.join(SUPPORTED_SOURCES)}, all")
    return list(dict.fromkeys(candidates))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FMIA hyperparameter optimization (Optuna).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Fuente(s) de datos separadas por coma, o 'all' para yahoo,refinitiv.\n"
            "Ejemplos: --source yahoo   --source yahoo,refinitiv   --source all\n"
            "Cada fuente genera su propio config_optimized_{source}.yaml."
        ),
    )
    parser.add_argument("--start", dest="start_date", default=None)
    parser.add_argument("--end", dest="end_date", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--purge-gap-days", type=int, default=None)
    parser.add_argument("--objective", choices=["sharpe_adj", "sortino", "calmar"], default=None)
    parser.add_argument("--turnover-penalty", type=float, default=None)
    parser.add_argument("--optimizer", choices=["mv", "cvar", "robust"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _run_single_source(
    source: str,
    cfg: dict,
    start_date: str,
    end_date: str,
    n_trials: int,
    n_folds: int,
    purge_gap_days: int,
    objective_metric: str,
    turnover_penalty: float,
    optimizer: str,
    seed: int,
    search_keys: list | None,
    logger: logging.Logger,
) -> bool:
    """Run hyperopt for one source. Returns True on success."""
    output_path = ROOT / f"reports/hyperopt_data/hyperopt_results_{source}.json"
    config_out  = ROOT / f"config_optimized_{source}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "=== [%s] trials=%d folds=%d purge=%d objective=%s optimizer=%s ===",
        source, n_trials, n_folds, purge_gap_days, objective_metric, optimizer,
    )

    from src.data_loader import load_data
    from src.features import build_signal_matrix
    from src.hyperopt import run_hyperopt

    try:
        data = load_data(source=source, start_date=start_date, end_date=end_date)
    except Exception as exc:
        logger.error("[%s] Fallo al cargar datos: %s", source, exc)
        return False

    prices = data["prices"]
    universe = data["universe"]
    feature_df = build_signal_matrix(
        prices,
        data["fundamentals"],
        data["fibra_fundamentals"],
        data["bonds"],
        data["macro"],
        universe,
    )

    result = run_hyperopt(
        prices=prices,
        feature_df=feature_df,
        universe=universe,
        macro=data["macro"],
        n_trials=n_trials,
        n_folds=n_folds,
        purge_gap_days=purge_gap_days,
        objective_metric=objective_metric,
        turnover_penalty=turnover_penalty,
        seed=seed,
        settings=cfg,
        optimizer=optimizer,
        search_keys=search_keys,
    )

    result_payload = {
        "source": source,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "validation_metrics": result.validation_metrics,
        "n_trials_completed": result.n_trials_completed,
        "optimization_time_seconds": result.optimization_time_seconds,
        "objective_metric": result.objective_metric,
        "turnover_penalty": result.turnover_penalty,
        "search_space": {
            k: {"kind": v[0], "low": v[1], "high": v[2], "log": v[3]} if v[0] != "categorical"
            else {"kind": "categorical", "choices": v[1]}
            for k, v in result.search_space.items()
        },
        "trial_history": result.trial_history.to_dict(orient="records") if not result.trial_history.empty else [],
    }
    with open(output_path, "w") as f:
        json.dump(result_payload, f, indent=2, default=str)
    logger.info("[%s] Wrote hyperopt JSON: %s", source, output_path)

    if result.best_params:
        merged = _merge_params_into_config(cfg, result.best_params)
        try:
            _dump_yaml(merged, config_out)
            logger.info("[%s] Wrote optimized config: %s", source, config_out)
        except ImportError:
            logger.warning("PyYAML not available — skipping %s.", config_out)

    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_hyperopt")

    args = _build_parser().parse_args()
    cfg = _load_yaml_config()

    raw_source = args.source or cfg.get("source") or "mock"
    sources = _normalize_sources(raw_source)

    start_date      = args.start_date     or cfg.get("start_date", "2017-01-01")
    end_date        = args.end_date       or cfg.get("end_date", "2026-03-31")
    n_trials        = args.n_trials       or int(cfg.get("hyperopt_n_trials", 50))
    n_folds         = args.n_folds        or int(cfg.get("hyperopt_n_folds", 3))
    purge_gap_days  = args.purge_gap_days or int(cfg.get("hyperopt_purge_gap_days", 21))
    objective_metric = args.objective     or str(cfg.get("hyperopt_objective", "sharpe_adj"))
    turnover_penalty = args.turnover_penalty or float(cfg.get("hyperopt_turnover_penalty", 0.5))
    optimizer       = args.optimizer      or str(cfg.get("hyperopt_optimizer", "mv"))
    seed            = args.seed           or int(cfg.get("hyperopt_seed", 42))
    search_keys     = cfg.get("hyperopt_search_keys")

    print(f"\nFondo Mexico — Hyperparameter Optimization")
    print(f"  Fuente(s)  : {', '.join(sources)}")
    print(f"  Periodo    : {start_date} → {end_date}")
    print(f"  Trials     : {n_trials}  |  Folds: {n_folds}  |  Purge gap: {purge_gap_days}d")
    print(f"  Objetivo   : {objective_metric}  |  Optimizador: {optimizer}\n")

    successful, failed = [], []
    for source in sources:
        ok = _run_single_source(
            source=source, cfg=cfg,
            start_date=start_date, end_date=end_date,
            n_trials=n_trials, n_folds=n_folds, purge_gap_days=purge_gap_days,
            objective_metric=objective_metric, turnover_penalty=turnover_penalty,
            optimizer=optimizer, seed=seed, search_keys=search_keys,
            logger=logger,
        )
        (successful if ok else failed).append(source)

    print("\n" + "=" * 60)
    if successful:
        print(f"[OK] Completado: {', '.join(successful)}")
        for s in successful:
            print(f"     config_optimized_{s}.yaml  →  listo para run_all.py")
    if failed:
        print(f"[ERROR] Fallaron: {', '.join(failed)}")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
