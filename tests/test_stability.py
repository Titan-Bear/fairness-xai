"""Tests for src/stability.py (teammate's module, integrated)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_dataset
from src.models import train_model
from src.preprocess import split_data
from src.stability import (
    StabilitySettings,
    compare_rankings,
    make_perturbations,
    noise_levels_with_baseline,
    run_stability,
    stability_by_group,
    summarize_stability,
    wilcoxon_shap_vs_lime,
)


def test_baseline_level_always_added() -> None:
    """0.0 is added and levels are sorted without duplicates."""
    assert noise_levels_with_baseline((0.10, 0.01, 0.01)) == [0.0, 0.01, 0.10]


def test_compare_rankings() -> None:
    """Identical vectors -> rho = top3 = top5 = 1; a constant vector -> NaN (skipped)."""
    a = np.array([5.0, 4, 3, 2, 1, 0.5])
    assert compare_rankings(a, a * 3) == (pytest.approx(1.0), 1.0, 1.0)
    assert all(np.isnan(v) for v in compare_rankings(a, np.zeros(6)))


@pytest.fixture(scope="module")
def compas():
    """COMPAS data, split and a fitted Logistic Regression."""
    loaded = load_dataset("compas")
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    pipeline = train_model("logistic_regression", split, loaded.metadata, include_sensitive=True)
    return loaded, split, pipeline


def test_perturbations(compas) -> None:
    """0% is unchanged, only numeric columns move, values stay in range, same direction per level."""
    loaded, split, _ = compas
    numeric = loaded.metadata.numeric
    X = split.X_test.head(20)
    perts = make_perturbations(X, numeric, split.X_train, [0.0, 0.01, 0.10], np.random.default_rng(0))
    pd.testing.assert_frame_equal(perts[0.0][numeric].astype(float), X[numeric].astype(float))
    for column in loaded.metadata.categorical:
        assert (perts[0.10][column] == X[column]).all()
    low, high = split.X_train[numeric].min(), split.X_train[numeric].max()
    assert ((perts[0.10][numeric] >= low) & (perts[0.10][numeric] <= high)).all().all()
    # Shared draw z: where nothing was clipped, the 10% shift is 10x the 1% shift.
    d1 = perts[0.01]["age"] - X["age"]
    d10 = perts[0.10]["age"] - X["age"]
    unclipped = (perts[0.10]["age"] > low["age"]) & (perts[0.10]["age"] < high["age"])
    np.testing.assert_allclose(d10[unclipped], 10 * d1[unclipped], rtol=1e-6)


def test_run_stability_output(compas) -> None:
    """One row per instance x level; SHAP baseline is exactly 1; values in valid ranges."""
    loaded, split, pipeline = compas
    settings = StabilitySettings(n_instances=6, seeds=(config.SEED,), lime_samples=100)
    out = run_stability("compas", "logistic_regression", "shap", (0.01, 0.05), settings,
                        split=split, metadata=loaded.metadata, pipeline=pipeline)
    assert len(out) == 6 * 3  # levels 0, 0.01, 0.05
    assert {"rho", "top3", "top5", "pred_flip", "dprob", "group_race", "group_sex"} <= set(out.columns)
    assert out.loc[out["noise"] == 0.0, "rho"].dropna().eq(1.0).all()
    assert out["rho"].dropna().between(-1, 1).all()
    assert out["pred_flip"].isin([0, 1]).all()
    assert (out["dprob"] >= 0).all()


def _toy_raw() -> pd.DataFrame:
    """Synthetic raw table: SHAP more stable than LIME, group A slightly more stable."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        group = "A" if i % 2 else "B"
        for method, base in (("shap", 0.95), ("lime", 0.80)):
            rows.append({"dataset": "toy", "model": "m", "method": method, "noise": 0.05, "seed": 1,
                         "instance_id": i, "pair_key": f"1-{i}", "rho": base + rng.normal(0, 0.02),
                         "top3": 1.0, "top5": 0.8, "pred_flip": 0, "dprob": 0.01, "group_attr": group})
    return pd.DataFrame(rows)


def test_summary_and_tests_on_toy_data() -> None:
    """Summary CI contains the mean; Wilcoxon detects SHAP > LIME; group test returns valid p."""
    raw = _toy_raw()
    summary = summarize_stability(raw, n_boot=200)
    for row in summary.itertuples():
        assert row.rho_ci_low <= row.rho <= row.rho_ci_high
    tests = wilcoxon_shap_vs_lime(raw)
    rho_tests = tests[tests["metric"] == "rho"]  # top5 is identical (0.8) for both by construction
    assert (rho_tests["shap_mean"] > rho_tests["lime_mean"]).all()
    assert rho_tests["reject_h0"].all()
    assert not tests[tests["metric"] == "top5"]["reject_h0"].any()
    groups = stability_by_group(raw, {("toy", "attr"): "A"})
    assert groups["p_value"].between(0, 1).all()
    assert set(groups["n_privileged"]) == {20}
