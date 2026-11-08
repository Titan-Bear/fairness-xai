"""Central configuration for the Fairness-Aware XAI Audit project.

Every tunable value (paths, seed, sample sizes, model parameters, flags) lives
here so that:
  * results are reproducible (one seed, used everywhere);
  * the notebooks, the pipeline runner and the Streamlit app all agree on
    where files are stored;
  * an examiner can see every experimental choice in one place.

Paths are built with pathlib relative to the project root, so the project
works unchanged on Windows, Linux and Google Colab.
"""

from __future__ import annotations

import logging
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# This file is src/config.py, so the project root is one level above src/.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
# Cached explanation outputs (SHAP/LIME are slow, so the app reads these).
CACHE_DIR: Path = RESULTS_DIR / "cache"

ALL_DIRS: tuple[Path, ...] = (
    DATA_RAW,
    DATA_PROCESSED,
    MODELS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    CACHE_DIR,
)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# One seed for splits, models, sampling, LIME and bootstrap.
SEED: int = 42

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
DATASETS: tuple[str, ...] = ("adult", "compas")

# ProPublica's cleaned two-year recidivism file (raw GitHub URL).
COMPAS_URL: str = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)
COMPAS_FILE: Path = DATA_RAW / "compas-scores-two-years.csv"

# fetch_adult downloads from OpenML and caches here, so later runs work offline.
ADULT_OPENML_CACHE: Path = DATA_RAW / "openml_cache"
# Offline fallback: the original UCI files, placed manually in data/raw/.
ADULT_UCI_TRAIN: Path = DATA_RAW / "adult.data"
ADULT_UCI_TEST: Path = DATA_RAW / "adult.test"

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
TEST_SIZE: float = 0.30

# True  -> sensitive attributes are model inputs.
# False -> "fairness through unawareness": sensitive columns are dropped from
#          the features, so we can test whether proxies still carry the bias.
INCLUDE_SENSITIVE: bool = True

# run_all runs both settings; the key is used in file names and tables.
SETTINGS: dict[str, bool] = {
    "with_sensitive": True,
    "without_sensitive": False,
}

# --------------------------------------------------------------------------
# Models (used from Phase 2)
# --------------------------------------------------------------------------
MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")

# Readable names for tables, charts and plain-English sentences.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression_mitigated": "Logistic Regression (mitigated)",
}
DATASET_DISPLAY_NAMES: dict[str, str] = {"adult": "Adult Income", "compas": "COMPAS"}

# Probability above which a prediction counts as the positive class.
DECISION_THRESHOLD: float = 0.5

# Colours for the two groups in every group-comparison chart (blue/orange
# pass colour-blind-safety checks). Privileged is always blue.
GROUP_COLORS: dict[str, str] = {"privileged": "#2a78d6", "unprivileged": "#eb6834"}

LOGREG_PARAMS: dict = {"max_iter": 1000, "random_state": SEED}
# min_samples_leaf=5 is added to the spec's parameters. With the default (1)
# every tree grows until each leaf holds one training row and memorises noise:
# on COMPAS, train accuracy 0.84 vs test 0.62. With 5, test accuracy rises to
# about 0.67 on COMPAS and 0.85 on Adult. Set it back to 1 to reproduce the
# unregularised forest.
RF_PARAMS: dict = {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1, "random_state": SEED}
XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
}

# --------------------------------------------------------------------------
# Explanations and comparison (used from Phase 4)
# --------------------------------------------------------------------------
SHAP_N_PER_GROUP: int = 200
LIME_N_PER_GROUP: int = 100
LIME_NUM_SAMPLES: int = 1000

# Training rows used as the background ("average person") for LinearExplainer.
SHAP_BACKGROUND_N: int = 500

XAI_METHODS: tuple[str, ...] = ("shap", "lime")

N_PROXIES: int = 3
# Override the automatic mutual-information proxy choice, e.g.
# {("adult", "sex"): ["relationship", "marital-status", "occupation"]}.
PROXY_OVERRIDES: dict[tuple[str, str], list[str]] = {}

TOP_K_JACCARD: int = 5
TOP_K_PLOT: int = 10
N_BOOTSTRAP: int = 1000

# Significance level for the Mann-Whitney tests (after Holm correction).
ALPHA: float = 0.05

# Tiny sample sizes for `python -m src.run_all --quick`.
QUICK_SHAP_N_PER_GROUP: int = 20
QUICK_LIME_N_PER_GROUP: int = 10
QUICK_LIME_NUM_SAMPLES: int = 200
QUICK_N_BOOTSTRAP: int = 100

# --------------------------------------------------------------------------
# Bias mitigation
# --------------------------------------------------------------------------
RUN_MITIGATION: bool = True
# ExponentiatedGradient settings: eps = allowed constraint violation,
# max_iter = number of reduction rounds (each fits one Logistic Regression).
MITIGATION_EPS: float = 0.01
MITIGATION_MAX_ITER: int = 50

# --------------------------------------------------------------------------
# Stability module (teammate's method, integrated)
# --------------------------------------------------------------------------
RUN_STABILITY: bool = True
# 0.0 is always added as a baseline: explaining the SAME input again measures
# the explainer's own randomness (LIME samples randomly; Tree/Linear SHAP is
# deterministic, so its baseline is exactly 1).
STABILITY_NOISE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10)
STABILITY_N_INSTANCES: int = 100
# Repeat the experiment with three seeds derived from SEED (42, 43, 44), so
# the results do not depend on one lucky sample.
STABILITY_SEEDS: tuple[int, ...] = (SEED, SEED + 1, SEED + 2)
STABILITY_LIME_SAMPLES: int = LIME_NUM_SAMPLES

QUICK_STABILITY_N_INSTANCES: int = 10
QUICK_STABILITY_SEEDS: tuple[int, ...] = (SEED,)


def set_results_dir(results_dir: Path) -> None:
    """Point every results path at another folder (used by run_all --quick).

    Modules read config.TABLES_DIR etc. when they save, not at import time,
    so after this call all tables, figures and caches go to the new folder.
    The quick test run therefore never overwrites the full results.
    """
    global RESULTS_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR, ALL_DIRS
    RESULTS_DIR = Path(results_dir)
    TABLES_DIR = RESULTS_DIR / "tables"
    FIGURES_DIR = RESULTS_DIR / "figures"
    CACHE_DIR = RESULTS_DIR / "cache"
    ALL_DIRS = (DATA_RAW, DATA_PROCESSED, MODELS_DIR, TABLES_DIR, FIGURES_DIR, CACHE_DIR)


def table_path(filename: str) -> Path:
    """Path of a results table, resolved at call time (see set_results_dir)."""
    return TABLES_DIR / filename


def ensure_dirs() -> None:
    """Create every output folder if it does not exist yet.

    Called by loaders and the pipeline so a fresh clone needs no manual setup.
    """
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure one consistent log format for scripts and notebooks.

    We use logging instead of print in src/ so messages carry a timestamp and
    module name, and can be silenced or redirected without editing code.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
