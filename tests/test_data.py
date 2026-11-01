"""Tests for src/data.py: both datasets load correctly and without leakage.

The first run downloads Adult (OpenML) and COMPAS (GitHub); later runs use
the cached copies in data/raw/.
"""

from __future__ import annotations

import pytest

from src.data import COMPAS_LEAKAGE_COLUMNS, LoadedDataset, load_adult, load_compas


@pytest.fixture(scope="module")
def adult() -> LoadedDataset:
    """Load Adult once for all tests in this module."""
    return load_adult()


@pytest.fixture(scope="module")
def compas() -> LoadedDataset:
    """Load COMPAS once for all tests in this module."""
    return load_compas()


def test_adult_expected_columns(adult: LoadedDataset) -> None:
    """Adult has the documented features; fnlwgt and education are dropped."""
    expected = {
        "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
        "workclass", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country",
    }
    assert set(adult.X.columns) == expected
    assert "fnlwgt" not in adult.X.columns
    assert "education" not in adult.X.columns


def test_adult_clean_and_binary(adult: LoadedDataset) -> None:
    """No missing values, binary target, binarised race, two sexes."""
    assert len(adult.X) > 40_000  # about 45k rows remain after dropping '?'
    assert adult.X.isna().sum().sum() == 0
    assert set(adult.y.unique()) == {0, 1}
    assert set(adult.sensitive["race"].unique()) == {"White", "Non-White"}
    assert set(adult.sensitive["sex"].unique()) == {"Male", "Female"}
    # Roughly a quarter of Adult earns >50K.
    assert 0.20 < adult.y.mean() < 0.30


def test_compas_expected_columns_and_no_leakage(compas: LoadedDataset) -> None:
    """COMPAS has exactly the chosen features and none of the COMPAS scores."""
    expected = {
        "age", "sex", "race", "priors_count", "c_charge_degree",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
    }
    assert set(compas.X.columns) == expected
    for column in COMPAS_LEAKAGE_COLUMNS:
        assert column not in compas.X.columns


def test_compas_filters(compas: LoadedDataset) -> None:
    """ProPublica filters + two races give the well-known 5,278 rows."""
    assert len(compas.X) == 5278
    assert set(compas.sensitive["race"].unique()) == {"African-American", "Caucasian"}
    assert set(compas.X["c_charge_degree"].unique()) <= {"F", "M"}
    assert set(compas.y.unique()) == {0, 1}
    assert compas.X.isna().sum().sum() == 0


@pytest.mark.parametrize("fixture_name", ["adult", "compas"])
def test_alignment_and_metadata(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """X, y and sensitive share an index; privileged groups actually exist."""
    data: LoadedDataset = request.getfixturevalue(fixture_name)
    assert data.X.index.equals(data.y.index)
    assert data.X.index.equals(data.sensitive.index)
    meta = data.metadata
    assert list(data.sensitive.columns) == meta.sensitive_attributes
    assert set(meta.numeric + meta.categorical) == set(data.X.columns)
    for attr, privileged in meta.privileged.items():
        assert privileged in set(data.sensitive[attr])
        # Every sensitive attribute is binary after cleaning.
        assert data.sensitive[attr].nunique() == 2
