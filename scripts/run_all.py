#!/usr/bin/env python
"""
run_all.py — Master script for the Fondo Mexico project.

Runs in order:
  1. Tests (pytest)
  2. Full pipeline
  3. HTML report generation

Configuration: edit config.yaml in the project root.
Command-line arguments override config.yaml.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --source yahoo
    python scripts/run_all.py --source refinitiv --start 2020-01-01 --hedge
    python scripts/run_all.py --skip-tests
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_mpl_cache = ROOT / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

DEFAULT_BENCHMARKS = ["IPC", "GBMCRE", "GBMNEAR", "GBMMOD", "GBMALFA"]
SUPPORTED_SOURCES = ["mock", "yahoo", "bloomberg", "refinitiv"]
DEFAULT_MULTI_PROVIDERS = ["yahoo", "refinitiv", "bloomberg"]

# Keys that run_hyperopt.py is allowed to tune — must mirror DEFAULT_SEARCH_SPACE
# (excluding REGULATORY_FIXED_KEYS). Only these are imported from config_optimized.yaml
# so that operational settings (source, dates, hedge, etc.) are never overridden.
HYPEROPT_TUNABLE_KEYS = frozenset({
    "bl_risk_aversion",
    "bl_tau",
    "mv_risk_aversion",
    "mv_turnover_penalty",
    "mv_market_impact_eta",
    "cvar_risk_aversion",
    "ewma_lambda_cov",
    "elasticnet_l1_ratios",
    "regime_ewma_span",
    "forecast_forward_days",
    "adtv_ewma_lambda",
})

# ---------------------------------------------------------------------------
# Load .env (credentials) if it exists
# ---------------------------------------------------------------------------
def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    config_path = ROOT / "config.yaml"
    defaults = {
        "source": "mock",
        "start_date": "2017-01-01",
        "end_date": "2026-03-31",
        "hedge": False,
        "reform": False,
        "benchmark_tickers": [],
        "report_output": "reports/output/strategy_report.html",
        "abort_on_test_failure": True,
        "optimizer": "mv",
    }
    if not config_path.exists():
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update({k: v for k, v in file_cfg.items() if v is not None})
    except ImportError:
        print("[WARNING] pyyaml not installed — using default values. Install with: pip install pyyaml")
    return defaults


def _maybe_run_hyperopt(source: str, config: dict) -> None:
    """Auto-run hyperopt for `source` when hyperopt_enabled is True and no
    optimized config exists yet. Failures are warned, not fatal.

    Mock data is excluded — tuning on synthetic data is meaningless and would
    just block the pipeline for hours during smoke runs.
    """
    if source == "mock":
        return
    if not bool(config.get("hyperopt_enabled", False)):
        return
    optimized_path = ROOT / f"config_optimized_{source}.yaml"
    if optimized_path.exists():
        return
    print(f"[Hyperopt:{source}] hyperopt_enabled=true and no {optimized_path.name} found — running hyperopt.")
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts" / "run_hyperopt.py"), "--source", source]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARNING] hyperopt run failed for {source}: {exc}. Continuing with current config.")


def _apply_optimized_params(config: dict, source: str) -> dict:
    """Overlay source-specific optimized hyperparams from config_optimized_{source}.yaml if it exists."""
    optimized_path = ROOT / f"config_optimized_{source}.yaml"
    if not optimized_path.exists():
        return config
    try:
        import yaml
        with open(optimized_path) as f:
            opt_cfg = yaml.safe_load(f) or {}
        applied = {k: v for k, v in opt_cfg.items() if k in HYPEROPT_TUNABLE_KEYS and v is not None}
        if applied:
            config = {**config, **applied}
            print(f"[Hyperopt:{source}] Applying {len(applied)} optimized parameters "
                  f"from config_optimized_{source}.yaml: {', '.join(sorted(applied))}")
    except Exception as exc:
        print(f"[WARNING] Could not read config_optimized_{source}.yaml: {exc}")
    return config

# ---------------------------------------------------------------------------
# CLI arguments (override config.yaml)
# ---------------------------------------------------------------------------
def _parse_args(config: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fondo Mexico — full pipeline")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Data source(s): mock|yahoo|bloomberg|refinitiv, "
            "comma-separated list, or 'all' for yahoo,refinitiv,bloomberg "
            f"(config.yaml: {config['source']})"
        ),
    )
    p.add_argument("--start", default=None, help=f"Start date YYYY-MM-DD (config.yaml: {config['start_date']})")
    p.add_argument("--end",   default=None, help=f"End date   YYYY-MM-DD (config.yaml: {config['end_date']})")
    p.add_argument("--hedge", action="store_true", default=None, help="Enable hedge overlay (Layer 2)")
    p.add_argument("--reform", action="store_true", default=None, help="Include LFI reform scenario comparison (4 structures)")
    p.add_argument("--out",   default=None, help="Output HTML report path")
    p.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated benchmark tickers (e.g. ^MXX,GBMINTBO.MX)",
    )
    p.add_argument("--skip-tests", action="store_true", help="Skip the test stage")
    p.add_argument(
        "--optimizer",
        choices=["mv", "cvar", "robust", "both"],
        default=None,
        help=f"Portfolio optimizer (config.yaml: {config.get('optimizer', 'mv')})",
    )
    p.add_argument(
        "--model",
        choices=["elasticnet", "lightgbm"],
        default=None,
        help=(
            "Cross-sectional return predictor "
            f"(config.yaml: {config.get('forecast_model', 'elasticnet')})"
        ),
    )
    return p.parse_args()


def _normalize_sources(raw_source: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_source, (list, tuple)):
        candidates = [str(s).strip().lower() for s in raw_source if str(s).strip()]
    else:
        source_text = str(raw_source).strip().lower()
        if source_text == "all":
            candidates = DEFAULT_MULTI_PROVIDERS.copy()
        else:
            candidates = [s.strip().lower() for s in source_text.split(",") if s.strip()]

    if not candidates:
        raise ValueError("No data source specified.")

    invalid = [s for s in candidates if s not in SUPPORTED_SOURCES]
    if invalid:
        valid = ", ".join(SUPPORTED_SOURCES + ["all"])
        raise ValueError(f"Invalid source(s): {', '.join(invalid)}. Valid options: {valid}")

    # Preserve order and remove duplicates
    return list(dict.fromkeys(candidates))


def _output_path_for_source(base_out: str, source: str, forecast_model: str) -> str:
    out_path = Path(base_out)
    return str(out_path.with_name(f"{out_path.stem}_{source}_{forecast_model}{out_path.suffix}"))

# ---------------------------------------------------------------------------
# Step 1 — Tests
# ---------------------------------------------------------------------------
def run_tests(abort_on_failure: bool) -> bool:
    print("\n" + "=" * 60)
    print("STEP 1/3 — Running tests")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        if abort_on_failure:
            print("\n[ERROR] Tests failed. Aborting pipeline.")
            print("        To skip tests use: --skip-tests")
            return False
        else:
            print("\n[WARNING] Tests failed but continuing (abort_on_test_failure=false in config.yaml).")
    return True

# ---------------------------------------------------------------------------
# Steps 2 + 3 — Pipeline and report
# ---------------------------------------------------------------------------
def run_report(
    source: str,
    start: str,
    end: str,
    hedge: bool,
    out_path: str,
    optimizer: str = "mv",
    benchmark_tickers: list[str] | None = None,
    settings: dict | None = None,
    reform: bool = False,
) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 2/3 — Running pipeline  [{source}]  {start} → {end}")
    print("=" * 60)

    # Inject Refinitiv App Key from the environment if present
    provider_kwargs = {}
    if source == "refinitiv":
        app_key = os.environ.get("REFINITIV_APP_KEY", "")
        if app_key and app_key != "paste_your_app_key_here":
            provider_kwargs["app_key"] = app_key
        else:
            print("[WARNING] REFINITIV_APP_KEY not configured in .env")
    elif source == "bloomberg":
        data_dir = os.environ.get("BLOOMBERG_DATA_DIR", "data/bloomberg")
        provider_kwargs["data_dir"] = data_dir

    from src.pipeline import run_pipeline
    results = run_pipeline(
        hedge_mode=hedge,
        data_source=source,
        start_date=start,
        end_date=end,
        optimizer=optimizer,
        benchmark_tickers=benchmark_tickers,
        hedge_reform=reform,
        settings=settings,
        **provider_kwargs,
    )

    print("\n" + "=" * 60)
    print("STEP 3/3 — Generating HTML report")
    print("=" * 60)

    from reports.charts import build_dashboard_html
    html = build_dashboard_html(results, hedge_mode=hedge, data_source=source, reform=reform)

    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n[OK] Report saved to: {out}")

# ---------------------------------------------------------------------------
# Parallel provider execution
# ---------------------------------------------------------------------------
def _run_sources_parallel(
    sources: list[str],
    *,
    start: str,
    end: str,
    hedge: bool,
    reform: bool,
    out: str,
    optimizer: str,
    forecast_model: str,
    benchmarks: list[str] | None,
) -> tuple[list[str], list[tuple[str, str]], dict[str, float]]:
    """Launch one subprocess per source in parallel, each with FMIA_SOURCE_LABEL set.

    Returns (successful_sources, failed_sources, timings_seconds).
    Tests must have already run in the parent — every subprocess gets --skip-tests.
    """
    import time

    base_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_all.py"),
        "--skip-tests",
        "--start", start, "--end", end,
        "--optimizer", optimizer, "--model", forecast_model,
    ]
    if hedge:
        base_cmd.append("--hedge")
    if reform:
        base_cmd.append("--reform")
    if benchmarks:
        base_cmd += ["--benchmarks", ",".join(benchmarks)]
    if out:
        base_cmd += ["--out", out]

    launched: list[tuple[str, subprocess.Popen, float]] = []
    for source in sources:
        env = {**os.environ, "FMIA_SOURCE_LABEL": source}
        cmd = base_cmd + ["--source", source]
        print(f"[Parallel] Launching {source}…")
        p = subprocess.Popen(cmd, env=env)
        launched.append((source, p, time.time()))

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    timings: dict[str, float] = {}
    for source, p, t0 in launched:
        p.wait()
        timings[source] = time.time() - t0
        if p.returncode == 0:
            successful.append(source)
        else:
            failed.append((source, f"exit code {p.returncode}"))

    return successful, failed, timings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _load_env()
    config  = _load_config()
    args    = _parse_args(config)

    # Merge: CLI > config.yaml > defaults
    source_value = args.source or config["source"]
    start   = args.start      or config["start_date"]
    end     = args.end        or config["end_date"]
    hedge   = args.hedge      if args.hedge else config["hedge"]
    reform  = args.reform     if args.reform else config.get("reform", False)
    out     = args.out        or config["report_output"]
    abort     = config["abort_on_test_failure"]
    optimizer = args.optimizer or config.get("optimizer", "mv")
    forecast_model = args.model or config.get("forecast_model", "elasticnet")
    pipeline_settings = dict(config)
    pipeline_settings["forecast_model"] = forecast_model
    sources = _normalize_sources(source_value)
    cli_bench = [x.strip() for x in args.benchmarks.split(",") if x.strip()] if args.benchmarks else None
    cfg_bench = config.get("benchmark_tickers")
    if isinstance(cfg_bench, str):
        cfg_bench = [x.strip() for x in cfg_bench.split(",") if x.strip()]

    def _benchmarks_for_source(current_source: str) -> list[str]:
        if cli_bench is not None:
            return cli_bench
        if cfg_bench is not None:
            return cfg_bench
        return DEFAULT_BENCHMARKS if current_source in ("yahoo", "refinitiv", "bloomberg") else []

    print(f"\nFondo Mexico — Full Pipeline")
    print(f"  Source(s)  : {', '.join(sources)}")
    print(f"  Period     : {start} → {end}")
    print(f"  Hedge      : {hedge}")
    print(f"  Reform LFI : {reform}")
    print(f"  Optimizer  : {optimizer}")
    print(f"  Forecast   : {forecast_model}")
    print(f"  Report     : {out}")

    parallel_providers = bool(config.get("parallel_providers", False))
    # Single source or explicit --skip-tests child process → never spawn sub-children.
    if len(sources) == 1 or args.skip_tests:
        parallel_providers = False

    if not args.skip_tests:
        ok = run_tests(abort_on_failure=abort)
        if not ok:
            sys.exit(1)
    else:
        print("\n[WARNING] Tests skipped via --skip-tests")

    successful_sources: list[str] = []
    failed_sources: list[tuple[str, str]] = []
    timings: dict[str, float] = {}

    if parallel_providers:
        print(f"\n[Parallel] Running {len(sources)} providers simultaneously…")
        benchmarks_for_parallel = _benchmarks_for_source(sources[0]) if sources else None
        successful_sources, failed_sources, timings = _run_sources_parallel(
            sources,
            start=start, end=end, hedge=hedge, reform=reform,
            out=out, optimizer=optimizer, forecast_model=forecast_model,
            benchmarks=benchmarks_for_parallel,
        )
    else:
        import time as _time
        for source in sources:
            benchmark_tickers = _benchmarks_for_source(source)
            out_for_source = _output_path_for_source(out, source, forecast_model)
            _maybe_run_hyperopt(source, pipeline_settings)
            source_settings = _apply_optimized_params(dict(pipeline_settings), source)
            os.environ["FMIA_SOURCE_LABEL"] = source
            print(
                f"\n[RUN] source={source} | benchmarks={', '.join(benchmark_tickers) if benchmark_tickers else 'N/A'} "
                f"| out={out_for_source}"
            )
            t0 = _time.time()
            try:
                run_report(
                    source,
                    start,
                    end,
                    hedge,
                    out_for_source,
                    optimizer,
                    benchmark_tickers,
                    settings=source_settings,
                    reform=reform,
                )
                successful_sources.append(source)
            except Exception as exc:
                failed_sources.append((source, str(exc)))
                print(f"\n[ERROR] {source}: {exc}")
                print("        Continuing with next provider.")
            finally:
                timings[source] = _time.time() - t0

    # --- Summary ---------------------------------------------------------
    if failed_sources:
        print("\n" + "=" * 60)
        print("PROVIDER ERROR SUMMARY")
        print("=" * 60)
        for source, err in failed_sources:
            print(f"- {source}: {err}")

    if not successful_sources:
        print("\n[ERROR] No provider completed successfully.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for source in sources:
        secs = timings.get(source, 0)
        m = int(secs) // 60
        label = "✓" if source in successful_sources else "✗"
        print(f"  {label} {source:<12} {m}m")
    if parallel_providers and timings:
        wall = max(timings.values())
        print(f"  Total wall time: {int(wall) // 60}m  (parallel)")
    else:
        total = sum(timings.values())
        print(f"  Total:           {int(total) // 60}m")

    print("\n[DONE] Pipeline completed successfully.\n")

if __name__ == "__main__":
    main()
