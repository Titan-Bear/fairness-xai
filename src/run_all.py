"""Run the whole audit pipeline end to end.

    python -m src.run_all            # full run, results in results/
    python -m src.run_all --quick    # tiny samples, results in results/quick/

Steps, for both datasets and both settings:
  load -> preprocess -> train -> evaluate -> fairness -> explain (SHAP, LIME)
  -> compare explanations -> statistical tests -> trade-off
  -> mitigation (RUN_MITIGATION) -> stability (RUN_STABILITY)

--quick writes to a separate folder, so a test run never overwrites the full
results that the Streamlit app displays. Explanations are cached, so a second
run only recomputes what changed.
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from src import config
from src.compare import build_tradeoff, run_comparison, save_comparison_tables, save_tradeoff
from src.evaluate import evaluate_models, save_performance_figures, save_performance_table
from src.explain import QUICK_SETTINGS, ExplainSettings, run_explanations
from src.fairness import run_fairness, save_fairness_tables
from src.mitigation import run_mitigation
from src.models import load_model, model_path, train_all
from src.preprocess import prepare_dataset
from src.stability import QUICK_STABILITY, StabilitySettings, run_stability_all
from src.stats import run_stat_tests, save_stat_tests

logger = logging.getLogger("src.run_all")


@contextmanager
def timed(step: str) -> Iterator[None]:
    """Log the start and duration of a pipeline step."""
    logger.info("=== %s ...", step)
    start = time.perf_counter()
    yield
    logger.info("=== %s done in %.1fs", step, time.perf_counter() - start)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Fairness-aware XAI audit pipeline.")
    parser.add_argument("--quick", action="store_true",
                        help="tiny explanation samples and bootstrap; results go to results/quick/")
    parser.add_argument("--datasets", nargs="+", choices=config.DATASETS, default=list(config.DATASETS),
                        help="which datasets to run (default: all)")
    parser.add_argument("--skip-training", action="store_true",
                        help="load saved models from models/ instead of retraining (if they exist)")
    parser.add_argument("--force-explain", action="store_true",
                        help="ignore cached explanations and recompute them")
    parser.add_argument("--no-mitigation", action="store_true",
                        help="skip the bias-mitigation step even if enabled in config")
    parser.add_argument("--no-stability", action="store_true",
                        help="skip the explanation-stability step even if enabled in config")
    parser.add_argument("--no-figures", action="store_true", help="write tables only")
    return parser.parse_args(argv)


def get_models(dataset: str, split, metadata, skip_training: bool) -> dict:
    """Train all models, or load them if requested and available."""
    if skip_training and all(model_path(dataset, s, m).exists()
                             for s in config.SETTINGS for m in config.MODEL_NAMES):
        logger.info("Loading saved models for %s", dataset)
        return {(s, m): load_model(dataset, s, m) for s in config.SETTINGS for m in config.MODEL_NAMES}
    return train_all(split, metadata)


def run_mitigation_step(prepared: dict, explain_settings: ExplainSettings, n_boot: int, figures: bool) -> None:
    """Bias mitigation: Logistic Regression before vs after ExponentiatedGradient."""
    with timed("Mitigation"):
        run_mitigation(prepared, explain_settings, n_boot, save_figures=figures)


def run_stability_step(prepared: dict, quick: bool, n_boot: int, figures: bool) -> None:
    """Explanation stability under input noise (teammate's module)."""
    with timed("Stability"):
        run_stability_all(prepared, QUICK_STABILITY if quick else StabilitySettings(), n_boot, save_figures=figures)


def main(argv: list[str] | None = None) -> None:
    """Run every step and write all tables and figures."""
    args = parse_args(argv)
    config.setup_logging()
    if args.quick:
        config.set_results_dir(config.PROJECT_ROOT / "results" / "quick")
        explain_settings, n_boot = QUICK_SETTINGS, config.QUICK_N_BOOTSTRAP
    else:
        explain_settings, n_boot = ExplainSettings(), config.N_BOOTSTRAP
    config.ensure_dirs()
    figures = not args.no_figures
    logger.info("Results folder: %s | explanation sample: %s | bootstrap: %d",
                config.RESULTS_DIR, explain_settings.tag, n_boot)

    total_start = time.perf_counter()
    performance, fairness, fairness_groups = [], [], []
    comparison_parts: dict[str, list[pd.DataFrame]] = {}
    prepared = {}

    for dataset in args.datasets:
        with timed(f"{dataset}: load + preprocess"):
            loaded, split = prepare_dataset(dataset)
            metadata = loaded.metadata
            prepared[dataset] = (loaded, split)
        with timed(f"{dataset}: train"):
            fitted = get_models(dataset, split, metadata, args.skip_training)
        with timed(f"{dataset}: evaluate"):
            performance.append(evaluate_models(fitted, split, dataset))
            if figures:
                save_performance_figures(fitted, split, dataset)
        with timed(f"{dataset}: fairness"):
            summary, by_group = run_fairness(dataset, fitted, split, metadata, save_figures=figures)
            fairness.append(summary)
            fairness_groups.append(by_group)
        with timed(f"{dataset}: explain (SHAP + LIME)"):
            instances, tables = run_explanations(dataset, fitted, split, metadata,
                                                 explain_settings, force=args.force_explain)
        with timed(f"{dataset}: compare explanations"):
            results = run_comparison(split, metadata, instances, tables, n_boot, save_figures=figures)
            for key, table in results.items():
                comparison_parts.setdefault(key, []).append(table)

    with timed("Save tables, statistical tests, trade-off"):
        performance_table = pd.concat(performance, ignore_index=True)
        fairness_table = pd.concat(fairness, ignore_index=True)
        save_performance_table(performance_table)
        save_fairness_tables(fairness_table, pd.concat(fairness_groups, ignore_index=True))
        comparison = {k: pd.concat(v, ignore_index=True) for k, v in comparison_parts.items()}
        save_comparison_tables(comparison)
        save_stat_tests(run_stat_tests(comparison["per_instance"]))
        save_tradeoff(build_tradeoff(performance_table, fairness_table, comparison["comparison"]))

    if config.RUN_MITIGATION and not args.no_mitigation:
        run_mitigation_step(prepared, explain_settings, n_boot, figures)
    if config.RUN_STABILITY and not args.no_stability:
        run_stability_step(prepared, args.quick, n_boot, figures)

    logger.info("All done in %.1f min. Tables: %s", (time.perf_counter() - total_start) / 60, config.TABLES_DIR)


if __name__ == "__main__":
    main()
