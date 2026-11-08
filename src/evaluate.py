"""Test-set performance: metrics table, ROC curves and confusion matrices.

Why several metrics instead of accuracy alone?
  * Accuracy hides class imbalance: on Adult, always predicting "<=50K"
    already gives about 75% accuracy.
  * Precision = of those predicted positive, how many really are.
  * Recall (true positive rate) = of the real positives, how many we found.
    Recall per group is also what "equal opportunity" compares in Phase 3.
  * F1 = harmonic mean of precision and recall.
  * ROC-AUC uses the predicted probabilities, so it measures ranking quality
    independent of the 0.5 threshold.

Run `python -m src.evaluate` to train all models and write the outputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.models import train_all
from src.preprocess import SplitData, prepare_dataset

logger = logging.getLogger(__name__)

PERFORMANCE_FILE = "performance.csv"


def new_figure(figsize: tuple[float, float]) -> Figure:
    """Create a figure without pyplot.

    pyplot ties figures to a GUI backend (Tk on Windows), which crashes when a
    script creates many figures and exits. A bare Figure needs no GUI, is
    freed like any Python object, and still displays in Jupyter
    (display(fig)) and Streamlit (st.pyplot(fig)).
    """
    return Figure(figsize=figsize)


def predict_positive_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (column 1 of predict_proba)."""
    return pipeline.predict_proba(X)[:, 1]


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = config.DECISION_THRESHOLD,
    y_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """All performance metrics from true labels and positive-class probabilities.

    By default the class predictions are y_proba >= threshold. A model whose
    predictions are made differently (the randomised mitigated model) can
    pass its own y_pred; ROC-AUC always uses y_proba.
    The confusion matrix is stored as four numbers (tn, fp, fn, tp) so it fits
    in one CSV row and the app can redraw it.
    """
    y_true = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int) if y_pred is None else np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0: if a model never predicts positive, precision is 0
        # rather than a warning and NaN.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_models(
    fitted: dict[tuple[str, str], Pipeline], split: SplitData, dataset: str
) -> pd.DataFrame:
    """One metrics row per (setting, model) on the test set."""
    rows = []
    for (setting, model_name), pipeline in fitted.items():
        proba = predict_positive_proba(pipeline, split.X_test)
        rows.append({"dataset": dataset, "setting": setting, "model": model_name,
                     **compute_metrics(split.y_test, proba)})
    return pd.DataFrame(rows)


def plot_roc_curves(
    fitted: dict[tuple[str, str], Pipeline], split: SplitData, dataset: str, setting: str
) -> Figure:
    """ROC curves of all models for one dataset and setting, on one chart.

    The dashed diagonal is a random classifier (AUC = 0.5); curves closer to
    the top-left corner are better.
    """
    fig = new_figure((6, 5))
    ax = fig.subplots()
    for (s, model_name), pipeline in fitted.items():
        if s != setting:
            continue
        proba = predict_positive_proba(pipeline, split.X_test)
        fpr, tpr, _ = roc_curve(split.y_test, proba)
        auc = roc_auc_score(split.y_test, proba)
        ax.plot(fpr, tpr, label=f"{config.MODEL_DISPLAY_NAMES[model_name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random (AUC 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curves - {config.DATASET_DISPLAY_NAMES[dataset]} ({setting.replace('_', ' ')})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    fitted: dict[tuple[str, str], Pipeline], split: SplitData, dataset: str, setting: str
) -> Figure:
    """Side-by-side confusion matrices of all models for one setting."""
    items = [(m, p) for (s, m), p in fitted.items() if s == setting]
    fig = new_figure((4.5 * len(items), 4))
    axes = fig.subplots(1, len(items))
    for ax, (model_name, pipeline) in zip(np.atleast_1d(axes), items):
        y_pred = (predict_positive_proba(pipeline, split.X_test) >= config.DECISION_THRESHOLD).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            split.y_test, y_pred, labels=[0, 1], ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(config.MODEL_DISPLAY_NAMES[model_name])
    fig.suptitle(f"Confusion matrices - {config.DATASET_DISPLAY_NAMES[dataset]} ({setting.replace('_', ' ')})")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, filename: str) -> None:
    """Save a figure to results/figures/ (PNG, 120 dpi)."""
    config.ensure_dirs()
    fig.savefig(config.FIGURES_DIR / filename, dpi=120, bbox_inches="tight")


def save_performance_figures(
    fitted: dict[tuple[str, str], Pipeline], split: SplitData, dataset: str
) -> None:
    """Write ROC and confusion-matrix figures for every setting of one dataset."""
    for setting in sorted({s for s, _ in fitted}):
        save_figure(plot_roc_curves(fitted, split, dataset, setting), f"roc_{dataset}_{setting}.png")
        save_figure(plot_confusion_matrices(fitted, split, dataset, setting),
                    f"confusion_{dataset}_{setting}.png")


def save_performance_table(table: pd.DataFrame) -> None:
    """Write results/tables/performance.csv."""
    config.ensure_dirs()
    table.to_csv(config.table_path(PERFORMANCE_FILE), index=False)
    logger.info("Wrote %s (%d rows).", PERFORMANCE_FILE, len(table))


def train_and_evaluate(dataset: str) -> tuple[dict[tuple[str, str], Pipeline], SplitData, pd.DataFrame]:
    """Prepare one dataset, train all models for both settings and evaluate them."""
    loaded, split = prepare_dataset(dataset)
    fitted = train_all(split, loaded.metadata)
    table = evaluate_models(fitted, split, dataset)
    save_performance_figures(fitted, split, dataset)
    return fitted, split, table


def main() -> None:
    """Phase 2 check: train and evaluate everything, write performance.csv."""
    config.setup_logging()
    tables = [train_and_evaluate(name)[2] for name in config.DATASETS]
    table = pd.concat(tables, ignore_index=True)
    save_performance_table(table)
    columns = ["dataset", "setting", "model", "accuracy", "f1", "roc_auc"]
    logger.info("Performance summary:\n%s", table[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
