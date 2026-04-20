#!/usr/bin/env python
"""
CLI: python scripts/run_hyperopt.py [--n-trials N] [--n-folds K]
                                    [--optimizer mv|cvar] [--output PATH]

Runs Bayesian hyperparameter optimization (Optuna TPE) and writes:
  - reports/output/hyperopt_results.json  — best params + trial history
  - reports/output/hyperopt_report.html   — visual diagnostics
  - config_optimized.yaml                 — ready-to-use config with best params

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FMIA hyperparameter optimization (Optuna).")
    parser.add_argument("--source", choices=["mock", "yahoo", "bloomberg", "refinitiv"], default=None)
    parser.add_argument("--start", dest="start_date", default=None)
    parser.add_argument("--end", dest="end_date", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--purge-gap-days", type=int, default=None)
    parser.add_argument("--objective", choices=["sharpe_adj", "sortino", "calmar"], default=None)
    parser.add_argument("--turnover-penalty", type=float, default=None)
    parser.add_argument("--optimizer", choices=["mv", "cvar", "robust"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="JSON output path.")
    parser.add_argument("--config-out", default=None, help="Optimized config.yaml output path.")
    parser.add_argument("--report-out", default=None, help="HTML report output path.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("run_hyperopt")

    args = _build_parser().parse_args()
    cfg = _load_yaml_config()

    source = args.source or (cfg.get("source") if isinstance(cfg.get("source"), str) else "mock")
    if isinstance(source, list) and source:
        source = source[0]
    start_date = args.start_date or cfg.get("start_date", "2017-01-01")
    end_date = args.end_date or cfg.get("end_date", "2026-03-31")
    n_trials = args.n_trials or int(cfg.get("hyperopt_n_trials", 50))
    n_folds = args.n_folds or int(cfg.get("hyperopt_n_folds", 3))
    purge_gap_days = args.purge_gap_days or int(cfg.get("hyperopt_purge_gap_days", 21))
    objective_metric = args.objective or str(cfg.get("hyperopt_objective", "sharpe_adj"))
    turnover_penalty = args.turnover_penalty or float(cfg.get("hyperopt_turnover_penalty", 0.5))
    optimizer = args.optimizer or str(cfg.get("hyperopt_optimizer", "mv"))
    seed = args.seed or int(cfg.get("hyperopt_seed", 42))
    output_path = Path(args.output or cfg.get("hyperopt_output", "reports/output/hyperopt_results.json"))
    config_out = Path(args.config_out or cfg.get("hyperopt_config_out", "config_optimized.yaml"))
    report_out = Path(args.report_out or cfg.get("hyperopt_report_out", "reports/output/hyperopt_report.html"))
    search_keys = cfg.get("hyperopt_search_keys")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Hyperopt inputs: source=%s trials=%d folds=%d purge=%d objective=%s optimizer=%s",
        source, n_trials, n_folds, purge_gap_days, objective_metric, optimizer,
    )

    # Lazy imports so the CLI can --help without the full stack installed.
    from src.data_loader import load_data
    from src.features import build_signal_matrix
    from src.hyperopt import run_hyperopt

    data = load_data(source=source, start_date=start_date, end_date=end_date)
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
    logger.info("Wrote hyperopt JSON: %s", output_path)

    if result.best_params:
        merged = _merge_params_into_config(cfg, result.best_params)
        try:
            _dump_yaml(merged, config_out)
            logger.info("Wrote optimized config: %s", config_out)
        except ImportError:
            logger.warning("PyYAML not available — skipping config_optimized.yaml.")

    try:
        from reports.charts import generate_hyperopt_report

        generate_hyperopt_report(result, output_path=report_out)
        logger.info("Wrote hyperopt HTML report: %s", report_out)
    except Exception as exc:
        logger.warning("HTML report generation skipped: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
