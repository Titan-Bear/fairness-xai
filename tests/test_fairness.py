"""Tests for src/fairness.py: metric values, valid ranges and sentences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset
from src.fairness import evaluate_fairness, fairness_metrics, interpret_demographic_parity
from src.models import train_model
from src.preprocess import split_data


# --------------------------------------------------------------------------
# Hand-checked toy example
# --------------------------------------------------------------------------
# Group A: 4 people, 2 truly positive. Predictions: 3 positive (both true positives + 1 false positive).
# Group B: 4 people, 2 truly positive. Predictions: 1 positive (one true positive).
Y_TRUE = np.array([1, 1, 0, 0, 1, 1, 0, 0])
Y_PRED = np.array([1, 1, 1, 0, 1, 0, 0, 0])
GROUPS = pd.Series(["A"] * 4 + ["B"] * 4)


def test_toy_values_match_hand_calculation() -> None:
    """Selection rates 0.75 vs 0.25; TPR 1.0 vs 0.5; FPR 0.5 vs 0.0."""
    m = fairness_metrics(Y_TRUE, Y_PRED, GROUPS)
    assert m["dp_difference"] == pytest.approx(0.5)            # 0.75 - 0.25
    assert m["dp_ratio"] == pytest.approx(0.25 / 0.75)
    assert m["eo_difference"] == pytest.approx(0.5)            # 1.0 - 0.5
    assert m["eodds_difference"] == pytest.approx(0.5)         # max(0.5, 0.5)


def test_perfect_parity_gives_zero_and_ratio_one() -> None:
    """Identical behaviour in both groups -> no gap."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    groups = pd.Series(["A", "A", "B", "B"])
    m = fairness_metrics(y_true, y_pred, groups)
    assert m["dp_difference"] == pytest.approx(0.0)
    assert m["dp_ratio"] == pytest.approx(1.0)
    assert m["eo_difference"] == pytest.approx(0.0)
    assert m["eodds_difference"] == pytest.approx(0.0)


def test_sentence_reports_the_numbers() -> None:
    """The plain-English sentence contains the group phrases and the right numbers."""
    row = {
        "dataset": "adult", "model": "xgboost", "attribute": "sex",
        "privileged": "Male", "unprivileged": "Female",
        "selection_rate_privileged": 0.31, "selection_rate_unprivileged": 0.11,
        "dp_difference": 0.20,
    }
    assert interpret_demographic_parity(row) == (
        "XGBoost predicts >50K for 31% of men but 11% of women (demographic parity difference 0.20)."
    )


# --------------------------------------------------------------------------
# Real model: every metric falls in its valid range
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def compas_results():
    """Fairness of a Logistic Regression trained on COMPAS (fast)."""
    loaded = load_dataset("compas")
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    pipeline = train_model("logistic_regression", split, loaded.metadata, include_sensitive=True)
    rows, by_group = evaluate_fairness(pipeline, split, loaded.metadata, "with_sensitive", "logistic_regression")
    return loaded, split, rows, by_group


def test_metrics_in_valid_ranges(compas_results) -> None:
    """Differences and ratio in [0, 1]; ratio = min/max of selection rates."""
    _, _, rows, _ = compas_results
    assert len(rows) == 2  # race and sex
    for row in rows:
        for key in ("dp_difference", "dp_ratio", "eo_difference", "eodds_difference"):
            assert 0.0 <= row[key] <= 1.0, key
        rates = (row["selection_rate_privileged"], row["selection_rate_unprivileged"])
        assert row["dp_ratio"] == pytest.approx(min(rates) / max(rates))
        assert row["dp_difference"] == pytest.approx(abs(rates[0] - rates[1]))
        assert row["eo_difference"] == pytest.approx(abs(row["tpr_privileged"] - row["tpr_unprivileged"]))
        # Equalized odds is the larger of the TPR and FPR gaps.
        assert row["eodds_difference"] >= row["eo_difference"] - 1e-12


def test_by_group_rates_and_counts(compas_results) -> None:
    """Per-group rates lie in [0, 1] and group counts add up to the test size."""
    loaded, split, _, by_group = compas_results
    for column in ("accuracy", "selection_rate", "true_positive_rate", "false_positive_rate", "base_rate"):
        assert by_group[column].between(0, 1).all(), column
    for attribute in loaded.metadata.sensitive_attributes:
        assert by_group[by_group["attribute"] == attribute]["count"].sum() == len(split.X_test)


def test_privileged_groups_follow_config(compas_results) -> None:
    """Rows name the privileged group configured in the dataset metadata."""
    loaded, _, rows, _ = compas_results
    for row in rows:
        assert row["privileged"] == loaded.metadata.privileged[row["attribute"]]
        assert row["privileged"] != row["unprivileged"]
        assert isinstance(row["interpretation"], str) and row["interpretation"].endswith(").")
