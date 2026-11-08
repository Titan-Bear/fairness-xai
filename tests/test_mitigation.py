"""Tests for src/mitigation.py (ExponentiatedGradient with demographic parity)."""

from __future__ import annotations

import numpy as np
import pytest
from fairlearn.metrics import demographic_parity_difference

from src import config
from src.data import load_dataset
from src.mitigation import MODEL_STEP, PREPROCESSOR_STEP, fit_mitigated_pipeline
from src.models import train_model
from src.preprocess import split_data


@pytest.fixture(scope="module")
def compas_mitigation():
    """COMPAS Logistic Regression before and after mitigation on race."""
    loaded = load_dataset("compas")
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    before = train_model("logistic_regression", split, loaded.metadata, include_sensitive=True)
    after = fit_mitigated_pipeline(before, split, "race")
    return split, before, after


def test_mitigated_predict_proba_is_valid(compas_mitigation) -> None:
    """Expected probabilities: shape (n, 2), within [0, 1], rows sum to 1; predictions are 0/1."""
    split, _, after = compas_mitigation
    proba = after.predict_proba(split.X_test)
    assert proba.shape == (len(split.X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(after.predict(split.X_test))) <= {0, 1}


def test_preprocessor_is_shared(compas_mitigation) -> None:
    """Before and after use the same fitted preprocessor, so only the classifier differs."""
    _, before, after = compas_mitigation
    assert after.named_steps[PREPROCESSOR_STEP] is before.named_steps[PREPROCESSOR_STEP]


def test_mitigation_reduces_demographic_parity_gap(compas_mitigation) -> None:
    """The randomised mitigated predictions have a smaller race DP gap than the original model."""
    split, before, after = compas_mitigation
    groups = split.s_test["race"]
    y_before = (before.predict_proba(split.X_test)[:, 1] >= config.DECISION_THRESHOLD).astype(int)
    X_t = after.named_steps[PREPROCESSOR_STEP].transform(split.X_test)
    y_after = after.named_steps[MODEL_STEP].mitigator.predict(X_t, random_state=config.SEED)
    dp_before = demographic_parity_difference(split.y_test, y_before, sensitive_features=groups)
    dp_after = demographic_parity_difference(split.y_test, y_after, sensitive_features=groups)
    assert dp_after < dp_before
