"""Group fairness metrics (RQ1): how unfair are the models?

For every dataset x setting x model x sensitive attribute we compute, on the
test set, per-group metrics with fairlearn's MetricFrame and four summary
numbers:

* Demographic parity difference = |selection rate A - selection rate B|.
  Selection rate = share of the group predicted positive. 0 means both groups
  get the positive prediction equally often.
* Demographic parity ratio = smaller selection rate / larger one. 1 is
  perfect; below 0.8 fails the US "four-fifths rule" of thumb.
* Equal opportunity difference = |TPR A - TPR B|. Among people who REALLY are
  positive, is the model equally good at finding them in both groups? Unlike
  demographic parity, this does not punish a model for different base rates.
* Equalized odds difference = max(|TPR gap|, |FPR gap|). Also requires equal
  false positive rates - in COMPAS a false positive means a defendant who did
  not re-offend was labelled as likely to (ProPublica's main finding).

All differences lie in [0, 1]; 0 means no gap.

Run `python -m src.fairness` (after `python -m src.evaluate`) to write
results/tables/fairness.csv, fairness_by_group.csv and the figures.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src import config
from src.data import DatasetMetadata
from src.evaluate import new_figure, predict_positive_proba, save_figure
from src.models import load_model
from src.preprocess import SplitData, prepare_dataset

logger = logging.getLogger(__name__)

FAIRNESS_FILE = "fairness.csv"
FAIRNESS_BY_GROUP_FILE = "fairness_by_group.csv"

# Metrics computed per group by MetricFrame.
GROUP_METRICS = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "true_positive_rate": true_positive_rate,
    "false_positive_rate": false_positive_rate,
    "count": count,
}

# Readable plural phrases for sentences ("31% of men but 11% of women").
GROUP_PHRASES: dict[tuple[str, str, str], str] = {
    ("adult", "sex", "Male"): "men",
    ("adult", "sex", "Female"): "women",
    ("adult", "race", "White"): "White people",
    ("adult", "race", "Non-White"): "Non-White people",
    ("compas", "race", "African-American"): "African-American defendants",
    ("compas", "race", "Caucasian"): "Caucasian defendants",
    ("compas", "sex", "Male"): "male defendants",
    ("compas", "sex", "Female"): "female defendants",
}

# What "actually positive" means, for the equal-opportunity sentence.
ACTUAL_POSITIVE_PHRASES: dict[str, str] = {
    "adult": "actually earn >50K",
    "compas": "actually re-offended within two years",
}


def group_phrase(dataset: str, attribute: str, group: str) -> str:
    """Plural phrase for a group, falling back to 'the <group> group'."""
    return GROUP_PHRASES.get((dataset, attribute, group), f"the {group} group")


def other_group(groups: list[str], privileged: str) -> str:
    """The unprivileged group of a binary attribute."""
    rest = [g for g in groups if g != privileged]
    if len(rest) != 1:
        raise ValueError(f"Expected exactly two groups, got {groups}")
    return rest[0]


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
def group_metric_frame(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray, groups: pd.Series
) -> MetricFrame:
    """fairlearn MetricFrame of accuracy, selection rate, TPR, FPR and count per group."""
    return MetricFrame(
        metrics=GROUP_METRICS,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sensitive_features=np.asarray(groups),
    )


def fairness_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray, groups: pd.Series
) -> dict[str, float]:
    """The four summary fairness numbers for one binary sensitive attribute.

    Equal opportunity difference is the between-group difference of the TPR
    column of the MetricFrame - exactly its definition.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    frame = group_metric_frame(y_true, y_pred, groups)
    return {
        "dp_difference": demographic_parity_difference(y_true, y_pred, sensitive_features=groups),
        "dp_ratio": demographic_parity_ratio(y_true, y_pred, sensitive_features=groups),
        "eo_difference": float(frame.difference(method="between_groups")["true_positive_rate"]),
        "eodds_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=groups),
    }


def evaluate_fairness(
    pipeline: Pipeline,
    split: SplitData,
    metadata: DatasetMetadata,
    setting: str,
    model_name: str,
    y_pred: np.ndarray | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Fairness of one fitted model, for every sensitive attribute.

    Predictions are predict_proba >= 0.5 unless `y_pred` is given (the
    randomised mitigated model passes its own predictions).

    Returns (summary rows, per-group table). Summary rows hold privileged and
    unprivileged values side by side so a sentence or a chart can be built
    from one row. We also store each group's BASE RATE (share of real
    positives in the test data): comparing it with the selection rate shows
    whether the model narrows, copies or amplifies the gap in the data.
    """
    if y_pred is None:
        y_pred = (predict_positive_proba(pipeline, split.X_test) >= config.DECISION_THRESHOLD).astype(int)
    y_pred = np.asarray(y_pred)
    y_true = split.y_test.to_numpy()
    summaries, by_group_tables = [], []

    for attribute in metadata.sensitive_attributes:
        groups = split.s_test[attribute]
        frame = group_metric_frame(y_true, y_pred, groups)
        by_group = frame.by_group.reset_index().rename(columns={"sensitive_feature_0": "group"})
        by_group["base_rate"] = [y_true[(groups == g).to_numpy()].mean() for g in by_group["group"]]
        by_group.insert(0, "attribute", attribute)
        by_group.insert(0, "model", model_name)
        by_group.insert(0, "setting", setting)
        by_group.insert(0, "dataset", metadata.name)
        by_group_tables.append(by_group)

        priv = metadata.privileged[attribute]
        unpriv = other_group(list(by_group["group"]), priv)
        p = by_group.set_index("group").loc[priv]
        u = by_group.set_index("group").loc[unpriv]
        summary = {
            "dataset": metadata.name, "setting": setting, "model": model_name,
            "attribute": attribute, "privileged": priv, "unprivileged": unpriv,
            "n_privileged": int(p["count"]), "n_unprivileged": int(u["count"]),
            "base_rate_privileged": p["base_rate"], "base_rate_unprivileged": u["base_rate"],
            "base_rate_difference": abs(p["base_rate"] - u["base_rate"]),
            "selection_rate_privileged": p["selection_rate"],
            "selection_rate_unprivileged": u["selection_rate"],
            "tpr_privileged": p["true_positive_rate"], "tpr_unprivileged": u["true_positive_rate"],
            "fpr_privileged": p["false_positive_rate"], "fpr_unprivileged": u["false_positive_rate"],
            "accuracy_privileged": p["accuracy"], "accuracy_unprivileged": u["accuracy"],
            **fairness_metrics(y_true, y_pred, groups),
        }
        summary["interpretation"] = interpret_demographic_parity(summary)
        summary["interpretation_eo"] = interpret_equal_opportunity(summary)
        summaries.append(summary)

    return summaries, pd.concat(by_group_tables, ignore_index=True)


# --------------------------------------------------------------------------
# Plain-English interpretation
# --------------------------------------------------------------------------
def _higher_first(row: dict, prefix: str) -> tuple[str, float, str, float]:
    """Return (group, value, other group, other value) with the larger value first."""
    pairs = [
        (row["privileged"], row[f"{prefix}_privileged"]),
        (row["unprivileged"], row[f"{prefix}_unprivileged"]),
    ]
    (g1, v1), (g2, v2) = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    return g1, v1, g2, v2


def interpret_demographic_parity(row: dict) -> str:
    """One sentence on demographic parity, built only from the row's numbers.

    Example: "XGBoost predicts >50K for 31% of men but 11% of women
    (demographic parity difference 0.20)."
    """
    g1, v1, g2, v2 = _higher_first(row, "selection_rate")
    joiner = "but" if row["dp_difference"] >= 0.005 else "and"
    return (
        f"{config.MODEL_DISPLAY_NAMES[row['model']]} predicts "
        f"{_positive_label(row['dataset'])} for {v1:.0%} of "
        f"{group_phrase(row['dataset'], row['attribute'], g1)} {joiner} {v2:.0%} of "
        f"{group_phrase(row['dataset'], row['attribute'], g2)} "
        f"(demographic parity difference {row['dp_difference']:.2f})."
    )


def interpret_equal_opportunity(row: dict) -> str:
    """One sentence on equal opportunity (TPR gap)."""
    g1, v1, g2, v2 = _higher_first(row, "tpr")
    joiner = "but only" if row["eo_difference"] >= 0.005 else "and"
    return (
        f"Among people who {ACTUAL_POSITIVE_PHRASES.get(row['dataset'], 'are actually positive')}, "
        f"it correctly identifies {v1:.0%} of {group_phrase(row['dataset'], row['attribute'], g1)} "
        f"{joiner} {v2:.0%} of {group_phrase(row['dataset'], row['attribute'], g2)} "
        f"(equal opportunity difference {row['eo_difference']:.2f})."
    )


def _positive_label(dataset: str) -> str:
    """Plain-English positive prediction for a dataset."""
    return {"adult": ">50K", "compas": "re-offending within two years"}.get(dataset, "the positive class")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
PLOT_METRICS = {
    "selection_rate": "Selection rate",
    "true_positive_rate": "True positive rate",
    "false_positive_rate": "False positive rate",
    "accuracy": "Accuracy",
}


def plot_group_metrics(
    by_group: pd.DataFrame, metadata: DatasetMetadata, setting: str, attribute: str
) -> Figure:
    """Per-group bars for every model: one panel per metric.

    Privileged group in blue, unprivileged in orange, values printed on the
    bars so the chart does not rely on colour alone.
    """
    # Filter on dataset too: "race" exists in both datasets with different groups.
    part = by_group[
        (by_group["dataset"] == metadata.name)
        & (by_group["setting"] == setting)
        & (by_group["attribute"] == attribute)
    ]
    priv = metadata.privileged[attribute]
    unpriv = other_group(sorted(part["group"].unique()), priv)
    models = [m for m in config.MODEL_NAMES if m in set(part["model"])]
    x = np.arange(len(models))
    width = 0.38

    fig = new_figure((4.2 * len(PLOT_METRICS), 4.2))
    axes = fig.subplots(1, len(PLOT_METRICS), sharey=True)
    for ax, (metric, title) in zip(axes, PLOT_METRICS.items()):
        for offset, group, role in ((-width / 2, priv, "privileged"), (width / 2, unpriv, "unprivileged")):
            values = [part[(part["model"] == m) & (part["group"] == group)][metric].iloc[0] for m in models]
            bars = ax.bar(x + offset, values, width * 0.95, color=config.GROUP_COLORS[role],
                          label=f"{group} ({role})")
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, [config.MODEL_DISPLAY_NAMES[m].replace(" ", "\n") for m in models], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(
        f"{config.DATASET_DISPLAY_NAMES[metadata.name]} - metrics by {attribute} "
        f"({setting.replace('_', ' ')})",
        x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_fairness_summary(summary: pd.DataFrame, dataset: str, attribute: str) -> Figure:
    """DP and EO differences per model, with vs without the sensitive feature."""
    part = summary[(summary["dataset"] == dataset) & (summary["attribute"] == attribute)]
    models = [m for m in config.MODEL_NAMES if m in set(part["model"])]
    x = np.arange(len(models))
    width = 0.38
    fig = new_figure((10, 4))
    axes = fig.subplots(1, 2, sharey=True)
    settings = list(config.SETTINGS)
    shades = {settings[0]: "#2a78d6", settings[-1]: "#86b6ef"}  # dark/light blue = with/without
    for ax, (metric, title) in zip(axes, (("dp_difference", "Demographic parity difference"),
                                          ("eo_difference", "Equal opportunity difference"))):
        for i, setting in enumerate(settings):
            values = [part[(part["model"] == m) & (part["setting"] == setting)][metric].iloc[0] for m in models]
            bars = ax.bar(x + (i - 0.5) * width, values, width * 0.95, color=shades[setting],
                          label=setting.replace("_", " "))
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, [config.MODEL_DISPLAY_NAMES[m] for m in models], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("difference (0 = no gap)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{config.DATASET_DISPLAY_NAMES[dataset]} - fairness gaps by {attribute}", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_fairness(
    dataset: str,
    fitted: dict[tuple[str, str], Pipeline] | None = None,
    split: SplitData | None = None,
    metadata: DatasetMetadata | None = None,
    save_figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fairness for every setting x model of one dataset.

    If `fitted` is not given, models are loaded from models/ (trained in
    Phase 2), so this step can run on its own.
    """
    if split is None or metadata is None:
        loaded, split = prepare_dataset(dataset, save=False)
        metadata = loaded.metadata
    if fitted is None:
        fitted = {(s, m): load_model(dataset, s, m) for s in config.SETTINGS for m in config.MODEL_NAMES}

    summaries, by_groups = [], []
    for (setting, model_name), pipeline in fitted.items():
        rows, by_group = evaluate_fairness(pipeline, split, metadata, setting, model_name)
        summaries.extend(rows)
        by_groups.append(by_group)
    summary = pd.DataFrame(summaries)
    by_group = pd.concat(by_groups, ignore_index=True)

    if save_figures:
        for attribute in metadata.sensitive_attributes:
            for setting in sorted(summary["setting"].unique()):
                save_figure(plot_group_metrics(by_group, metadata, setting, attribute),
                            f"fairness_groups_{dataset}_{setting}_{attribute}.png")
            if summary["setting"].nunique() > 1:
                save_figure(plot_fairness_summary(summary, dataset, attribute),
                            f"fairness_gaps_{dataset}_{attribute}.png")
    return summary, by_group


def save_fairness_tables(summary: pd.DataFrame, by_group: pd.DataFrame) -> None:
    """Write fairness.csv (one row per comparison) and fairness_by_group.csv."""
    config.ensure_dirs()
    summary.to_csv(config.table_path(FAIRNESS_FILE), index=False)
    by_group.to_csv(config.table_path(FAIRNESS_BY_GROUP_FILE), index=False)
    logger.info("Wrote %s (%d rows) and %s (%d rows).", FAIRNESS_FILE, len(summary),
                FAIRNESS_BY_GROUP_FILE, len(by_group))


def main() -> None:
    """Phase 3 check: compute fairness for everything and print the sentences."""
    config.setup_logging()
    results = [run_fairness(name) for name in config.DATASETS]
    summary = pd.concat([r[0] for r in results], ignore_index=True)
    by_group = pd.concat([r[1] for r in results], ignore_index=True)
    save_fairness_tables(summary, by_group)

    columns = ["dataset", "setting", "model", "attribute", "dp_difference", "dp_ratio",
               "eo_difference", "eodds_difference"]
    logger.info("Fairness summary:\n%s", summary[columns].round(3).to_string(index=False))
    for row in summary[summary["setting"] == "with_sensitive"].to_dict("records"):
        logger.info("%s", row["interpretation"])


if __name__ == "__main__":
    main()
