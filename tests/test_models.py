"""Tests for src/models.py and src/evaluate.py.

Models are trained on a 3,000-row sample of Adult and on the full COMPAS
training split so the tests run in seconds; files are saved to a temporary
folder so the real models/ folder is not overwritten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, models
from src.data import load_dataset
from src.evaluate import compute_metrics
from src.preprocess import split_data

SAMPLE_ROWS = 3000


@pytest.fixture(scope="module", params=list(config.DATASETS))
def prepared(request: pytest.FixtureRequest) -> tuple:
    """A (possibly sub-sampled) split for each dataset."""
    loaded = load_dataset(request.param)
    X, y, s = loaded.X, loaded.y, loaded.sensitive
    if len(X) > SAMPLE_ROWS:
        idx = X.sample(SAMPLE_ROWS, random_state=config.SEED).index
        X, y, s = X.loc[idx], y.loc[idx], s.loc[idx]
    return loaded.metadata, split_data(X, y, s)


@pytest.mark.parametrize("model_name", config.MODEL_NAMES)
@pytest.mark.parametrize("include_sensitive", [True, False])
def test_predict_proba_shape_and_range(prepared, model_name: str, include_sensitive: bool) -> None:
    """predict_proba returns (n, 2) probabilities in [0, 1] that sum to 1."""
    metadata, split = prepared
    pipeline = models.train_model(model_name, split, metadata, include_sensitive)
    proba = pipeline.predict_proba(split.X_test)
    assert proba.shape == (len(split.X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_without_sensitive_ignores_sensitive_columns(prepared) -> None:
    """In the unaware setting, changing sex/race must not change any prediction."""
    metadata, split = prepared
    pipeline = models.train_model("logistic_regression", split, metadata, include_sensitive=False)
    flipped = split.X_test.copy()
    for attr in metadata.sensitive_attributes:
        values = sorted(flipped[attr].unique())
        flipped[attr] = flipped[attr].map({values[0]: values[1], values[1]: values[0]})
    np.testing.assert_allclose(pipeline.predict_proba(split.X_test), pipeline.predict_proba(flipped))


def test_save_and_load_roundtrip(prepared, tmp_path, monkeypatch) -> None:
    """A saved and reloaded Pipeline gives identical predictions."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    metadata, split = prepared
    pipeline = models.train_model("xgboost", split, metadata, include_sensitive=True)
    models.save_model(pipeline, metadata.name, "with_sensitive", "xgboost")
    loaded = models.load_model(metadata.name, "with_sensitive", "xgboost")
    np.testing.assert_array_equal(pipeline.predict_proba(split.X_test), loaded.predict_proba(split.X_test))


def test_unknown_model_name_raises() -> None:
    """A typo in a model name fails loudly instead of silently."""
    with pytest.raises(ValueError):
        models.make_classifier("svm")


def test_compute_metrics_on_known_example() -> None:
    """Hand-checked example: 4 rows, one false positive, one false negative."""
    y_true = pd.Series([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1])  # predictions at 0.5: 1, 1, 0, 0
    m = compute_metrics(y_true, y_proba)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    # Positives score 0.9 and 0.2, negatives 0.8 and 0.1 -> 3 of 4 pairs ranked correctly.
    assert m["roc_auc"] == pytest.approx(0.75)
