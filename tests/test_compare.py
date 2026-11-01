"""Tests for src/explain.py, src/compare.py and src/stats.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.compare import (
    bootstrap_spearman_ci,
    jaccard_top_k,
    pareto_optimal,
    rank_agreement,
    reliance_share,
)
from src.data import load_dataset
from src.explain import ExplainSettings, explain_model, positive_class_shap, sample_instances
from src.models import train_model
from src.preprocess import split_data
from src.stats import holm_adjust, mann_whitney, rank_biserial


# --------------------------------------------------------------------------
# Rank agreement
# --------------------------------------------------------------------------
def test_identical_rankings_give_rho_one() -> None:
    """Same importance order in both groups -> Spearman = Kendall = Jaccard = 1."""
    a = pd.Series([0.5, 0.3, 0.1, 0.05, 0.02, 0.01], index=list("abcdef"))
    b = a * 2  # different scale, same ranking
    result = rank_agreement(a, b)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["jaccard_top5"] == pytest.approx(1.0)


def test_reversed_rankings_give_rho_minus_one() -> None:
    """Opposite order -> Spearman = -1."""
    a = pd.Series([5.0, 4, 3, 2, 1], index=list("abcde"))
    b = pd.Series([1.0, 2, 3, 4, 5], index=list("abcde"))
    assert rank_agreement(a, b)["spearman_rho"] == pytest.approx(-1.0)


def test_jaccard_toy_example() -> None:
    """Top-3 sets {a,b,c} and {a,b,d}: 2 shared out of 4 distinct -> 0.5."""
    a = pd.Series({"a": 9, "b": 8, "c": 7, "d": 1, "e": 0})
    b = pd.Series({"a": 9, "b": 8, "d": 7, "c": 1, "e": 0})
    assert jaccard_top_k(a, b, k=3) == pytest.approx(0.5)
    assert jaccard_top_k(a, a, k=3) == pytest.approx(1.0)


def test_bootstrap_ci_contains_rho_and_is_ordered() -> None:
    """Two groups drawn from the same distribution: CI is ordered and high."""
    rng = np.random.default_rng(0)
    scale = np.array([5.0, 4, 3, 2, 1])
    a = rng.exponential(scale, size=(80, 5))
    b = rng.exponential(scale, size=(80, 5))
    low, high = bootstrap_spearman_ci(a, b, n_boot=200)
    assert -1 <= low <= high <= 1
    assert high == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Reliance and Pareto
# --------------------------------------------------------------------------
def test_reliance_share() -> None:
    """Share = feature attribution / row total; missing features count as 0."""
    attributions = pd.DataFrame({"sex": [1.0, 0.0], "age": [3.0, 0.0]})
    np.testing.assert_allclose(reliance_share(attributions, ["sex"]), [0.25, 0.0])
    np.testing.assert_allclose(reliance_share(attributions, ["race"]), [0.0, 0.0])


def test_pareto_toy_example() -> None:
    """Row C is worse than A on everything; A and B trade off -> A, B optimal."""
    frame = pd.DataFrame({"acc": [0.9, 0.8, 0.7], "gap": [0.3, 0.1, 0.4]}, index=list("ABC"))
    result = pareto_optimal(frame, {"acc": "max", "gap": "min"})
    assert result.to_dict() == {"A": True, "B": True, "C": False}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_mann_whitney_returns_valid_p() -> None:
    """p in [0, 1]; clearly shifted samples give a small p and r near 1."""
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 1.0, 100)
    y = rng.normal(0.0, 1.0, 100)
    result = mann_whitney(x, y)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["p_value"] < 0.01
    assert 0 < result["rank_biserial"] <= 1
    same = mann_whitney(np.ones(5), np.ones(5))
    assert same["p_value"] == 1.0 and same["rank_biserial"] == 0.0
    assert np.isnan(mann_whitney([1.0], [2.0, 3.0])["p_value"])


def test_rank_biserial_extremes() -> None:
    """U = n1*n2 (x always larger) -> r = 1; U = 0 -> r = -1; U = half -> 0."""
    assert rank_biserial(12, 3, 4) == pytest.approx(1.0)
    assert rank_biserial(0, 3, 4) == pytest.approx(-1.0)
    assert rank_biserial(6, 3, 4) == pytest.approx(0.0)


def test_holm_adjustment() -> None:
    """Hand-computed Holm values; NaNs are preserved and not counted."""
    p = np.array([0.01, 0.04, 0.03, np.nan])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max 0.06
    np.testing.assert_allclose(holm_adjust(p), [0.03, 0.06, 0.06, np.nan])
    assert np.all(holm_adjust(np.array([0.9, 0.8])) <= 1.0)


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def test_positive_class_shap_shapes() -> None:
    """List, 3-D and 2-D shap outputs all become (n, f) positive-class values."""
    two_d = np.arange(6.0).reshape(3, 2)
    three_d = np.stack([-two_d, two_d], axis=2)
    np.testing.assert_array_equal(positive_class_shap(two_d), two_d)
    np.testing.assert_array_equal(positive_class_shap(three_d), two_d)
    np.testing.assert_array_equal(positive_class_shap([-two_d, two_d]), two_d)


@pytest.fixture(scope="module")
def compas_small():
    """COMPAS split, a fitted model and a tiny explanation sample."""
    loaded = load_dataset("compas")
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    pipeline = train_model("logistic_regression", split, loaded.metadata, include_sensitive=True)
    settings = ExplainSettings(shap_n_per_group=6, lime_n_per_group=3, lime_num_samples=200)
    return loaded, split, pipeline, settings


def test_sampling_lime_subset_of_shap(compas_small) -> None:
    """Per group: n SHAP instances, the LIME ones are a subset; seeded -> reproducible."""
    loaded, split, _, settings = compas_small
    instances = sample_instances(split, loaded.metadata, settings)
    for attr in loaded.metadata.sensitive_attributes:
        counts = instances[instances[f"in_shap_{attr}"]].groupby(f"group_{attr}").size()
        assert (counts == settings.shap_n_per_group).all()
        assert not (instances[f"in_lime_{attr}"] & ~instances[f"in_shap_{attr}"]).any()
    again = sample_instances(split, loaded.metadata, settings)
    pd.testing.assert_frame_equal(instances, again)


def test_explain_model_outputs(compas_small, tmp_path, monkeypatch) -> None:
    """Attributions are non-negative per original feature; fidelity <= 1; cache reused."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    loaded, split, pipeline, settings = compas_small
    instances = sample_instances(split, loaded.metadata, settings)
    table = explain_model(pipeline, split, instances, "compas", "with_sensitive",
                          "logistic_regression", settings)
    features = set(loaded.metadata.feature_names)
    assert features <= set(table.columns)
    assert (table[list(features)] >= 0).all().all()
    lime = table[table["method"] == "lime"]
    assert lime["fidelity"].le(1.0).all()
    assert table[table["method"] == "shap"]["fidelity"].isna().all()
    cached = explain_model(pipeline, split, instances, "compas", "with_sensitive",
                           "logistic_regression", settings)
    pd.testing.assert_frame_equal(table.reset_index(drop=True), cached, check_dtype=False)
