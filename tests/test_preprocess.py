"""Tests for src/preprocess.py: split, preprocessing and one-hot mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_dataset
from src.preprocess import (
    aggregate_by_original,
    build_preprocessor,
    fit_transform_frame,
    map_transformed_to_original,
    select_feature_columns,
    split_data,
)


# --------------------------------------------------------------------------
# Toy-data tests (fast, no download): check the logic exactly.
# --------------------------------------------------------------------------
@pytest.fixture()
def toy() -> pd.DataFrame:
    """Small frame whose column names contain underscores, like COMPAS."""
    return pd.DataFrame(
        {
            "juv_fel_count": [0, 1, 2, 0],
            "age": [25, 40, 33, 51],
            "c_charge_degree": ["F", "M", "F", "M"],
            "sex": ["Male", "Female", "Female", "Male"],
        }
    )


def test_mapping_is_correct_with_underscores(toy: pd.DataFrame) -> None:
    """Every one-hot column maps to the right original feature."""
    pre = build_preprocessor(["juv_fel_count", "age"], ["c_charge_degree", "sex"])
    pre.fit(toy)
    mapping = map_transformed_to_original(pre)
    assert mapping == {
        "juv_fel_count": "juv_fel_count",
        "age": "age",
        "c_charge_degree_F": "c_charge_degree",
        "c_charge_degree_M": "c_charge_degree",
        "sex_Female": "sex",
        "sex_Male": "sex",
    }


def test_aggregate_sums_absolute_values(toy: pd.DataFrame) -> None:
    """Opposite-sign attributions on one feature add up instead of cancelling."""
    pre = build_preprocessor(["juv_fel_count", "age"], ["c_charge_degree", "sex"])
    pre.fit(toy)
    names = list(pre.get_feature_names_out())
    mapping = map_transformed_to_original(pre)
    # One instance; order matches `names`.
    attributions = np.array([[0.1, -0.2, 0.05, -0.05, 0.3, -0.3]])
    agg = aggregate_by_original(attributions, names, mapping)
    assert list(agg.columns) == ["juv_fel_count", "age", "c_charge_degree", "sex"]
    np.testing.assert_allclose(agg.iloc[0].to_numpy(), [0.1, 0.2, 0.1, 0.6])


def test_unknown_category_is_ignored(toy: pd.DataFrame) -> None:
    """A category unseen in training encodes as zeros, not an error."""
    pre = build_preprocessor(["juv_fel_count", "age"], ["c_charge_degree", "sex"])
    pre.fit(toy)
    new = toy.iloc[[0]].assign(c_charge_degree="X")
    row = pd.DataFrame(pre.transform(new), columns=pre.get_feature_names_out())
    assert row[["c_charge_degree_F", "c_charge_degree_M"]].sum(axis=1).iloc[0] == 0


# --------------------------------------------------------------------------
# Real-data tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module", params=list(config.DATASETS))
def dataset(request: pytest.FixtureRequest):
    """Load each dataset once and split it."""
    loaded = load_dataset(request.param)
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    return loaded, split


def test_split_sizes_and_stratification(dataset) -> None:
    """70/30 split; class balance is preserved; sensitive stays aligned."""
    loaded, split = dataset
    n = len(loaded.X)
    assert len(split.X_test) == pytest.approx(config.TEST_SIZE * n, abs=1)
    assert len(split.X_train) + len(split.X_test) == n
    assert split.y_train.mean() == pytest.approx(split.y_test.mean(), abs=0.01)
    assert split.X_test.index.equals(split.s_test.index)
    assert split.X_test.index.equals(split.y_test.index)
    assert split.X_train.index.intersection(split.X_test.index).empty


def test_split_is_reproducible(dataset) -> None:
    """Same seed -> identical test rows (both settings rely on this)."""
    loaded, split = dataset
    again = split_data(loaded.X, loaded.y, loaded.sensitive)
    assert split.X_test.index.equals(again.X_test.index)


@pytest.mark.parametrize("include_sensitive", [True, False])
def test_no_missing_after_preprocessing(dataset, include_sensitive: bool) -> None:
    """Transformed train/test have no NaNs, and sensitive columns are dropped when asked."""
    loaded, split = dataset
    numeric, categorical = select_feature_columns(loaded.metadata, include_sensitive)
    pre = build_preprocessor(numeric, categorical)
    X_train, X_test = fit_transform_frame(pre, split.X_train, split.X_test)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert X_train.shape[1] == X_test.shape[1]

    mapping = map_transformed_to_original(pre)
    assert set(mapping) == set(X_train.columns)
    originals = set(mapping.values())
    for attr in loaded.metadata.sensitive_attributes:
        assert (attr in originals) == include_sensitive
