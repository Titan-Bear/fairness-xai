"""Bias mitigation: does enforcing demographic parity change the explanations?

We retrain Logistic Regression with fairlearn's ExponentiatedGradient under a
DemographicParity constraint on each dataset's PRIMARY sensitive attribute
(Adult: sex, COMPAS: race), then recompute for the mitigated model:
  * performance (accuracy, F1, ROC-AUC),
  * fairness for every sensitive attribute (the unconstrained one too, to
    see side effects),
  * LIME explanations and all comparison metrics (reliance, rank agreement,
    fidelity) on the SAME instances as the original model.

How ExponentiatedGradient works (for the viva): it solves a sequence of
cost-sensitive classification problems, re-weighting training rows so that
the selection rates of the groups move together. The result is a
*randomised* mixture of several Logistic Regressions, each with a weight:
for every person one member is drawn at random (by weight) to decide.
Its fairness guarantee holds for these random predictions, so accuracy,
F1 and fairness of the mitigated model are measured on them (seeded).
ROC-AUC and LIME use the mixture's expected probability (see below).

Why only LIME? The mitigated model is a weighted mixture, not one tree or
linear model, so TreeExplainer/LinearExplainer do not apply. LIME is
model-agnostic: it only needs a predict_proba function.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.compare import compare_one, select_proxies
from src.data import DatasetMetadata
from src.evaluate import compute_metrics, new_figure, save_figure
from src.explain import ExplainSettings, explain_model, sample_instances
from src.fairness import evaluate_fairness
from src.models import MODEL_STEP, PREPROCESSOR_STEP, load_model
from src.preprocess import SplitData
from src.stats import run_stat_tests

logger = logging.getLogger(__name__)

MITIGATION_FILE = "mitigation.csv"
MITIGATION_TESTS_FILE = "mitigation_tests.csv"
BASE_MODEL = "logistic_regression"
MITIGATED_MODEL = "logistic_regression_mitigated"


class MitigatedClassifier:
    """Adapter giving a fitted ExponentiatedGradient a predict_proba method.

    ExponentiatedGradient.predict() is RANDOMISED (it picks one of its
    Logistic Regressions at random for every row). For explanations and
    ROC-AUC we need deterministic probabilities, i.e. the expected
    prediction of the mixture. fairlearn computes exactly this in
    `_pmf_predict` (weighted average of the members' predictions). It is a
    private method (leading underscore), so a future fairlearn version could
    rename it; we pin fairlearn==0.12.0 in requirements.txt for that reason.
    """

    def __init__(self, mitigator: ExponentiatedGradient) -> None:
        self.mitigator = mitigator
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, sensitive_features: np.ndarray) -> "MitigatedClassifier":
        """Fit the wrapped mitigator (lets the adapter act as a final Pipeline step)."""
        self.mitigator.fit(X, y, sensitive_features=sensitive_features)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Expected class probabilities of the randomised mixture, shape (n, 2)."""
        return np.asarray(self.mitigator._pmf_predict(X))  # private fairlearn API, see class docstring

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Randomised predictions of the mixture (seeded, so reproducible)."""
        return self.mitigator.predict(X, random_state=config.SEED)


def fit_mitigated_pipeline(before: Pipeline, split: SplitData, attribute: str) -> Pipeline:
    """ExponentiatedGradient(LogisticRegression, DemographicParity) on `attribute`.

    The already-fitted preprocessor of the original model is reused, so the
    original and the mitigated model see identical inputs and only the
    classifier differs. The sensitive attribute is needed at TRAINING time
    (to measure the constraint) even in the "without sensitive" setting -
    it is never a prediction input.
    """
    preprocessor = before.named_steps[PREPROCESSOR_STEP]
    mitigator = ExponentiatedGradient(
        LogisticRegression(**config.LOGREG_PARAMS),
        constraints=DemographicParity(),
        eps=config.MITIGATION_EPS,
        max_iter=config.MITIGATION_MAX_ITER,
    )
    classifier = MitigatedClassifier(mitigator).fit(
        preprocessor.transform(split.X_train), split.y_train.to_numpy(),
        sensitive_features=split.s_train[attribute].to_numpy())
    logger.info("Mitigator fitted: %d Logistic Regressions in the mixture.",
                int((mitigator.weights_ > 0).sum()))
    return Pipeline([(PREPROCESSOR_STEP, preprocessor), (MODEL_STEP, classifier)])


def _stage_rows(
    pipeline: Pipeline,
    split: SplitData,
    metadata: DatasetMetadata,
    setting: str,
    stage: str,
    model_name: str,
    instances: pd.DataFrame,
    explain_settings: ExplainSettings,
    proxies: dict[str, list[str]],
    n_boot: int,
    constrained: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Performance, fairness and LIME comparison of one model (before or after)."""
    proba = pipeline.predict_proba(split.X_test)[:, 1]
    y_pred, thresholded_dp = None, {}
    if stage == "after":
        # The mitigated model is RANDOMISED: fairlearn's demographic-parity
        # guarantee holds for its random predictions, so accuracy, F1 and all
        # fairness metrics use them (seeded -> reproducible). Thresholding the
        # expected probability at 0.5 would instead always pick the member
        # with the largest weight - a different model - so we only report
        # that version's DP difference for transparency.
        X_t = pipeline.named_steps[PREPROCESSOR_STEP].transform(split.X_test)
        y_pred = pipeline.named_steps[MODEL_STEP].mitigator.predict(X_t, random_state=config.SEED)
        y_thresholded = (proba >= config.DECISION_THRESHOLD).astype(int)
        for attribute in metadata.sensitive_attributes:
            thresholded_dp[attribute] = demographic_parity_difference(
                split.y_test, y_thresholded, sensitive_features=split.s_test[attribute])
    performance = compute_metrics(split.y_test, proba, y_pred=y_pred)
    fairness_rows, _ = evaluate_fairness(pipeline, split, metadata, setting, model_name, y_pred=y_pred)

    if stage == "before":
        # Same call as the main pipeline -> reuses the cached SHAP+LIME table.
        table = explain_model(pipeline, split, instances, metadata.name, setting, model_name, explain_settings)
    else:
        table = explain_model(pipeline, split, instances, metadata.name, setting, model_name,
                              explain_settings, methods=("lime",))

    rows, per_instances = [], []
    for fair in fairness_rows:
        attribute = fair["attribute"]
        comp, per_instance, _ = compare_one(table, instances, metadata, setting, model_name, "lime",
                                            attribute, proxies[attribute], n_boot)
        per_instances.append(per_instance.assign(model=f"{BASE_MODEL} ({stage})"))
        rows.append({
            "dataset": metadata.name, "setting": setting, "attribute": attribute,
            "constrained_attribute": constrained, "stage": stage,
            "accuracy": performance["accuracy"], "f1": performance["f1"], "roc_auc": performance["roc_auc"],
            "selection_rate_privileged": fair["selection_rate_privileged"],
            "selection_rate_unprivileged": fair["selection_rate_unprivileged"],
            "dp_difference": fair["dp_difference"], "dp_ratio": fair["dp_ratio"],
            "eo_difference": fair["eo_difference"], "eodds_difference": fair["eodds_difference"],
            "dp_difference_thresholded": thresholded_dp.get(attribute, np.nan),
            "sensitive_reliance_privileged": comp["sensitive_reliance_privileged"],
            "sensitive_reliance_unprivileged": comp["sensitive_reliance_unprivileged"],
            "proxy_reliance_privileged": comp["proxy_reliance_privileged"],
            "proxy_reliance_unprivileged": comp["proxy_reliance_unprivileged"],
            "spearman_rho": comp["spearman_rho"], "spearman_ci_low": comp["spearman_ci_low"],
            "spearman_ci_high": comp["spearman_ci_high"], "jaccard_top5": comp["jaccard_top5"],
            "fidelity_privileged": comp["fidelity_privileged"],
            "fidelity_unprivileged": comp["fidelity_unprivileged"], "fidelity_gap": comp["fidelity_gap"],
            "interpretation": fair["interpretation"],
        })
    return rows, per_instances


def mitigate_dataset(
    split: SplitData,
    metadata: DatasetMetadata,
    explain_settings: ExplainSettings,
    n_boot: int = config.N_BOOTSTRAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Before vs after mitigation for both settings of one dataset."""
    constrained = metadata.primary_sensitive
    instances = sample_instances(split, metadata, explain_settings)
    proxies = {a: select_proxies(split, metadata, a)[0] for a in metadata.sensitive_attributes}
    rows, per_instances = [], []
    for setting in config.SETTINGS:
        before = load_model(metadata.name, setting, BASE_MODEL)
        logger.info("Mitigating %s | %s | constraint: demographic parity on %s", metadata.name, setting, constrained)
        after = fit_mitigated_pipeline(before, split, constrained)
        for stage, pipeline, name in (("before", before, BASE_MODEL), ("after", after, MITIGATED_MODEL)):
            stage_rows, stage_instances = _stage_rows(pipeline, split, metadata, setting, stage, name, instances,
                                                      explain_settings, proxies, n_boot, constrained)
            rows.extend(stage_rows)
            per_instances.extend(stage_instances)
    return pd.DataFrame(rows), pd.concat(per_instances, ignore_index=True)


def plot_mitigation(table: pd.DataFrame, dataset: str, setting: str) -> Figure:
    """Before vs after bars for the constrained attribute."""
    part = table[(table["dataset"] == dataset) & (table["setting"] == setting)
                 & (table["attribute"] == table["constrained_attribute"])].set_index("stage")
    metrics = {"accuracy": "Accuracy", "dp_difference": "DP difference", "eo_difference": "EO difference",
               "spearman_rho": "Spearman rho\n(LIME, between groups)", "fidelity_gap": "LIME fidelity gap"}
    x = np.arange(len(metrics))
    fig = new_figure((10, 4))
    ax = fig.subplots()
    for offset, stage, color in ((-0.2, "before", "#86b6ef"), (0.2, "after", "#2a78d6")):
        values = [part.loc[stage, m] for m in metrics]
        bars = ax.bar(x + offset, values, 0.38, color=color, label=stage)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x, list(metrics.values()), fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    constrained = part["constrained_attribute"].iloc[0]
    ax.set_title(f"{config.DATASET_DISPLAY_NAMES[dataset]} - Logistic Regression before vs after "
                 f"demographic-parity mitigation on {constrained} ({setting.replace('_', ' ')})", fontsize=10)
    fig.tight_layout()
    return fig


def run_mitigation(prepared: dict, explain_settings: ExplainSettings, n_boot: int,
                   save_figures: bool = True) -> pd.DataFrame:
    """Mitigation for every dataset; writes mitigation.csv and mitigation_tests.csv.

    `prepared` maps dataset name -> (LoadedDataset, SplitData), as built by run_all.
    """
    tables, per_instances = [], []
    for dataset, (loaded, split) in prepared.items():
        table, per_instance = mitigate_dataset(split, loaded.metadata, explain_settings, n_boot)
        tables.append(table)
        per_instances.append(per_instance)
    table = pd.concat(tables, ignore_index=True)
    tests = run_stat_tests(pd.concat(per_instances, ignore_index=True))
    config.ensure_dirs()
    table.to_csv(config.table_path(MITIGATION_FILE), index=False)
    tests.to_csv(config.table_path(MITIGATION_TESTS_FILE), index=False)
    logger.info("Wrote %s (%d rows) and %s (%d rows).", MITIGATION_FILE, len(table), MITIGATION_TESTS_FILE, len(tests))
    if save_figures:
        for dataset in prepared:
            for setting in config.SETTINGS:
                save_figure(plot_mitigation(table, dataset, setting), f"mitigation_{dataset}_{setting}.png")
    return table
