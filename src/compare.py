"""Group-wise comparison of explanations (RQ2) and the trade-off table (RQ4).

For every dataset x setting x model x XAI method x sensitive attribute we ask:
does the model "reason" differently for the two groups, and are the
explanations equally trustworthy for both?

1. Sensitive-feature reliance: per instance, the share of total absolute
   attribution assigned to the sensitive feature itself.
     reliance = |attr(sex)| / sum over all features |attr|
   A share (not the raw value) makes models and methods comparable, because
   SHAP and LIME use different units.
2. Proxy reliance: the same share for the proxy features - the 3 features
   with the highest mutual information with the sensitive attribute on the
   training data. In the "without sensitive" setting this shows whether the
   model has shifted its reliance onto proxies.
3. Global importance per group: mean absolute attribution of each feature
   within each group, ranked.
4. Rank agreement: Spearman rho and Kendall tau between the two groups'
   importance vectors (1 = same ranking), Jaccard overlap of the top-5
   features, and a bootstrap 95% CI for rho.
5. Fidelity gap (LIME only): mean local-surrogate R-squared per group and
   their difference. A gap means explanations are less faithful for one group.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from src import config
from src.data import DatasetMetadata
from src.evaluate import new_figure, save_figure
from src.explain import feature_columns
from src.fairness import other_group
from src.preprocess import SplitData

logger = logging.getLogger(__name__)

COMPARISON_FILE = "explanation_comparison.csv"
PER_INSTANCE_FILE = "per_instance_reliance.csv"
IMPORTANCE_FILE = "global_importance.csv"
PROXIES_FILE = "proxies.csv"
TRADEOFF_FILE = "tradeoff.csv"


# --------------------------------------------------------------------------
# Proxies
# --------------------------------------------------------------------------
def proxy_scores(
    split: SplitData, metadata: DatasetMetadata, attribute: str, seed: int = config.SEED
) -> pd.DataFrame:
    """Mutual information between every non-sensitive feature and `attribute`.

    Mutual information measures how much knowing a feature reduces
    uncertainty about the group - it catches non-linear links that a
    correlation would miss. Computed on the TRAINING set only. All sensitive
    attributes are excluded from the candidates, so the proxies are the same
    in both settings and are "ordinary" features.
    Categorical features are ordinal-encoded and flagged as discrete so
    sklearn uses the right estimator for them.
    """
    candidates = [f for f in metadata.feature_names if f not in metadata.sensitive_attributes]
    X = split.X_train[candidates].copy()
    categorical = [c for c in candidates if c in metadata.categorical]
    if categorical:
        X[categorical] = OrdinalEncoder().fit_transform(X[categorical])
    discrete = np.array([c in categorical for c in candidates])
    mi = mutual_info_classif(X.to_numpy(dtype=float), split.s_train[attribute],
                             discrete_features=discrete, random_state=seed)
    scores = pd.DataFrame({"feature": candidates, "mutual_information": mi})
    return scores.sort_values("mutual_information", ascending=False).reset_index(drop=True)


def select_proxies(
    split: SplitData, metadata: DatasetMetadata, attribute: str, n: int = config.N_PROXIES
) -> tuple[list[str], pd.DataFrame]:
    """Top-n features by mutual information, unless overridden in config."""
    scores = proxy_scores(split, metadata, attribute)
    override = config.PROXY_OVERRIDES.get((metadata.name, attribute))
    proxies = list(override) if override else list(scores["feature"].head(n))
    scores["selected"] = scores["feature"].isin(proxies)
    scores["source"] = "config override" if override else "mutual information"
    scores.insert(0, "attribute", attribute)
    scores.insert(0, "dataset", metadata.name)
    return proxies, scores


# --------------------------------------------------------------------------
# Per-instance reliance
# --------------------------------------------------------------------------
def reliance_share(attributions: pd.DataFrame, features: list[str]) -> pd.Series:
    """Share of each row's total absolute attribution that goes to `features`.

    Features missing from the table (e.g. the sensitive feature in the
    'without sensitive' setting) contribute nothing. Rows whose attributions
    are all zero get share 0 instead of a division by zero.
    """
    present = [f for f in features if f in attributions.columns]
    total = attributions.sum(axis=1)
    part = attributions[present].sum(axis=1) if present else pd.Series(0.0, index=attributions.index)
    return (part / total.where(total > 0)).fillna(0.0)


def per_instance_reliance(
    table: pd.DataFrame,
    instances: pd.DataFrame,
    metadata: DatasetMetadata,
    attribute: str,
    method: str,
    proxies: list[str],
) -> pd.DataFrame:
    """Per-instance sensitive reliance, proxy reliance and fidelity for one comparison.

    Sensitive reliance is NaN (not applicable) when the sensitive feature is
    not a model input - it would be 0 by construction, and testing a column
    of zeros would be meaningless.
    """
    rows = table[table["method"] == method].set_index("instance_id")
    sample = instances.index[instances[f"in_{method}_{attribute}"]]
    rows = rows.loc[rows.index.intersection(sample)]
    attributions = rows[feature_columns(table)]
    groups = instances.loc[rows.index, f"group_{attribute}"]
    out = pd.DataFrame({
        "instance_id": rows.index,
        "group": groups.to_numpy(),
        "is_privileged": (groups == metadata.privileged[attribute]).to_numpy(),
        "sensitive_reliance": (reliance_share(attributions, [attribute]).to_numpy()
                               if attribute in attributions.columns else np.nan),
        "proxy_reliance": reliance_share(attributions, proxies).to_numpy(),
        "fidelity": rows["fidelity"].to_numpy(),
    })
    return out


# --------------------------------------------------------------------------
# Global importance and rank agreement
# --------------------------------------------------------------------------
def group_importance(attributions: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Mean absolute attribution per feature (rows) for each group (columns)."""
    return attributions.groupby(groups.to_numpy()).mean().T


def jaccard_top_k(a: pd.Series, b: pd.Series, k: int = config.TOP_K_JACCARD) -> float:
    """|top-k(a) intersect top-k(b)| / |top-k(a) union top-k(b)|. 1 = same top-k set."""
    top_a = set(a.sort_values(ascending=False).index[:k])
    top_b = set(b.sort_values(ascending=False).index[:k])
    return len(top_a & top_b) / len(top_a | top_b)


def rank_agreement(a: pd.Series, b: pd.Series, k: int = config.TOP_K_JACCARD) -> dict[str, float]:
    """Spearman rho, Kendall tau and top-k Jaccard between two importance vectors.

    Spearman rho is the correlation of the two RANKINGS (1 = identical order,
    0 = unrelated, -1 = reversed). Kendall tau counts pairs of features that
    are ordered the same way in both rankings; it is more robust to small
    swaps. Both are computed over all features, Jaccard only over the top k.
    """
    b = b.loc[a.index]
    return {
        "spearman_rho": float(spearmanr(a, b).statistic),
        "kendall_tau": float(kendalltau(a, b).statistic),
        "jaccard_top5": jaccard_top_k(a, b, k),
    }


def bootstrap_spearman_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.SEED, level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho between two groups' mean importances.

    a, b: per-instance attribution matrices (instances x features) of the two
    groups. Each resample draws instances WITH replacement within each group,
    recomputes both groups' mean importances and their Spearman rho. The
    middle 95% of the rho values is the CI. A wide interval means the
    ranking comparison is uncertain (e.g. small groups).
    """
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        mean_a = a[rng.integers(0, len(a), len(a))].mean(axis=0)
        mean_b = b[rng.integers(0, len(b), len(b))].mean(axis=0)
        rhos[i] = spearmanr(mean_a, mean_b).statistic
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(rhos, [tail, 100 - tail])
    return float(low), float(high)


# --------------------------------------------------------------------------
# One comparison
# --------------------------------------------------------------------------
def compare_one(
    table: pd.DataFrame,
    instances: pd.DataFrame,
    metadata: DatasetMetadata,
    setting: str,
    model_name: str,
    method: str,
    attribute: str,
    proxies: list[str],
    n_boot: int = config.N_BOOTSTRAP,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """All comparison metrics for one (setting, model, method, attribute).

    Returns (summary row, per-instance table, global-importance long table).
    Gaps are privileged minus unprivileged, so the sign says which group the
    model relies on the feature more for.
    """
    per_instance = per_instance_reliance(table, instances, metadata, attribute, method, proxies)
    rows = table[table["method"] == method].set_index("instance_id").loc[per_instance["instance_id"]]
    attributions = rows[feature_columns(table)]
    groups = pd.Series(per_instance["group"].to_numpy(), index=attributions.index)

    priv = metadata.privileged[attribute]
    unpriv = other_group(sorted(groups.unique()), priv)
    importance = group_importance(attributions, groups)
    agreement = rank_agreement(importance[priv], importance[unpriv])
    ci_low, ci_high = bootstrap_spearman_ci(
        attributions[groups == priv].to_numpy(), attributions[groups == unpriv].to_numpy(), n_boot
    )

    p_mask = per_instance["is_privileged"]
    means = per_instance.groupby("is_privileged")[["sensitive_reliance", "proxy_reliance", "fidelity"]].mean()
    row = {
        "dataset": metadata.name, "setting": setting, "model": model_name, "method": method,
        "attribute": attribute, "privileged": priv, "unprivileged": unpriv,
        "n_privileged": int(p_mask.sum()), "n_unprivileged": int((~p_mask).sum()),
        "sensitive_reliance_privileged": means.loc[True, "sensitive_reliance"],
        "sensitive_reliance_unprivileged": means.loc[False, "sensitive_reliance"],
        "proxies": ";".join(proxies),
        "proxy_reliance_privileged": means.loc[True, "proxy_reliance"],
        "proxy_reliance_unprivileged": means.loc[False, "proxy_reliance"],
        **agreement,
        "spearman_ci_low": ci_low, "spearman_ci_high": ci_high,
        "top5_privileged": ";".join(importance[priv].sort_values(ascending=False).index[:config.TOP_K_JACCARD]),
        "top5_unprivileged": ";".join(importance[unpriv].sort_values(ascending=False).index[:config.TOP_K_JACCARD]),
        "fidelity_privileged": means.loc[True, "fidelity"] if method == "lime" else np.nan,
        "fidelity_unprivileged": means.loc[False, "fidelity"] if method == "lime" else np.nan,
    }
    row["sensitive_reliance_gap"] = row["sensitive_reliance_privileged"] - row["sensitive_reliance_unprivileged"]
    row["proxy_reliance_gap"] = row["proxy_reliance_privileged"] - row["proxy_reliance_unprivileged"]
    row["fidelity_gap"] = row["fidelity_privileged"] - row["fidelity_unprivileged"]

    keys = {"dataset": metadata.name, "setting": setting, "model": model_name,
            "method": method, "attribute": attribute}
    per_instance = per_instance.assign(**keys)
    imp_long = importance.rename_axis("feature").reset_index().melt(
        id_vars="feature", var_name="group", value_name="mean_abs_attribution")
    imp_long["rank"] = imp_long.groupby("group")["mean_abs_attribution"].rank(ascending=False, method="min")
    imp_long = imp_long.assign(**keys)
    return row, per_instance, imp_long


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _feature_label(feature: str, attribute: str, proxies: list[str]) -> str:
    """Mark the sensitive feature and proxies in chart labels (not by colour alone)."""
    if feature == attribute:
        return f"{feature} (sensitive)"
    if feature in proxies:
        return f"{feature} (proxy)"
    return feature


def _clean_axes(ax) -> None:
    """Recessive grid and no top/right spines - shared chart style."""
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_top_features(
    imp_long: pd.DataFrame, row: dict, proxies: list[str], k: int = config.TOP_K_PLOT
) -> Figure:
    """Each group's own top-k features side by side (shared x-axis)."""
    fig = new_figure((11, 0.42 * k + 1.6))
    axes = fig.subplots(1, 2, sharex=True)
    for ax, group, role in ((axes[0], row["privileged"], "privileged"),
                            (axes[1], row["unprivileged"], "unprivileged")):
        part = imp_long[imp_long["group"] == group].nlargest(k, "mean_abs_attribution")[::-1]
        ax.barh([_feature_label(f, row["attribute"], proxies) for f in part["feature"]],
                part["mean_abs_attribution"], color=config.GROUP_COLORS[role], height=0.7)
        ax.set_title(f"{group} ({role})", fontsize=11)
        ax.set_xlabel("mean |attribution|")
        _clean_axes(ax)
    fig.suptitle(
        f"{config.DATASET_DISPLAY_NAMES[row['dataset']]} - {config.MODEL_DISPLAY_NAMES[row['model']]} - "
        f"{row['method'].upper()} - top {k} features by {row['attribute']} ({row['setting'].replace('_', ' ')})\n"
        f"Spearman rho {row['spearman_rho']:.2f} [95% CI {row['spearman_ci_low']:.2f}, "
        f"{row['spearman_ci_high']:.2f}], top-5 Jaccard {row['jaccard_top5']:.2f}",
        x=0.01, ha="left", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _group_boxplot(ax, per_instance: pd.DataFrame, column: str, row: dict, title: str) -> None:
    """Box plot of one per-instance measure for the two groups."""
    order = [(row["privileged"], "privileged"), (row["unprivileged"], "unprivileged")]
    data = [per_instance.loc[per_instance["group"] == g, column].dropna().to_numpy() for g, _ in order]
    box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True)
    for patch, (_, role) in zip(box["boxes"], order):
        patch.set_facecolor(config.GROUP_COLORS[role])
        patch.set_alpha(0.85)
    for median in box["medians"]:
        median.set_color("black")
    ax.set_xticks([1, 2], [f"{g}\n(mean {d.mean():.3f})" if len(d) else g for (g, _), d in zip(order, data)])
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_reliance_boxes(per_instance: pd.DataFrame, row: dict) -> Figure:
    """Sensitive and proxy reliance per group (box = middle 50%, line = median)."""
    has_sensitive = per_instance["sensitive_reliance"].notna().any()
    fig = new_figure((9 if has_sensitive else 5, 4))
    axes = np.atleast_1d(fig.subplots(1, 2 if has_sensitive else 1))
    if has_sensitive:
        _group_boxplot(axes[0], per_instance, "sensitive_reliance", row,
                       f"Reliance on '{row['attribute']}' (share of attribution)")
    _group_boxplot(axes[-1], per_instance, "proxy_reliance", row,
                   f"Reliance on proxies ({row['proxies'].replace(';', ', ')})")
    fig.suptitle(f"{config.DATASET_DISPLAY_NAMES[row['dataset']]} - {config.MODEL_DISPLAY_NAMES[row['model']]} - "
                 f"{row['method'].upper()} ({row['setting'].replace('_', ' ')})", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_fidelity_boxes(per_instance: pd.DataFrame, row: dict) -> Figure:
    """LIME fidelity (local R-squared) per group."""
    fig = new_figure((5, 4))
    ax = fig.subplots()
    _group_boxplot(ax, per_instance, "fidelity", row, "LIME fidelity (local R-squared)")
    fig.suptitle(f"{config.DATASET_DISPLAY_NAMES[row['dataset']]} - {config.MODEL_DISPLAY_NAMES[row['model']]} "
                 f"({row['setting'].replace('_', ' ')}) by {row['attribute']}", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _figure_name(kind: str, row: dict) -> str:
    """Consistent figure file name for one comparison."""
    return f"{kind}_{row['dataset']}_{row['setting']}_{row['model']}_{row['method']}_{row['attribute']}.png"


# --------------------------------------------------------------------------
# Runner for one dataset
# --------------------------------------------------------------------------
def run_comparison(
    split: SplitData,
    metadata: DatasetMetadata,
    instances: pd.DataFrame,
    tables: dict[tuple[str, str], pd.DataFrame],
    n_boot: int = config.N_BOOTSTRAP,
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """All comparisons for one dataset. Returns a dict of result tables."""
    proxy_tables, rows, per_instances, importances = [], [], [], []
    proxies_by_attr = {}
    for attribute in metadata.sensitive_attributes:
        proxies, scores = select_proxies(split, metadata, attribute)
        proxies_by_attr[attribute] = proxies
        proxy_tables.append(scores)
        logger.info("%s | proxies for %s: %s", metadata.name, attribute, proxies)

    for (setting, model_name), table in tables.items():
        for method in config.XAI_METHODS:
            for attribute in metadata.sensitive_attributes:
                proxies = proxies_by_attr[attribute]
                row, per_instance, imp_long = compare_one(
                    table, instances, metadata, setting, model_name, method, attribute, proxies, n_boot)
                rows.append(row)
                per_instances.append(per_instance)
                importances.append(imp_long)
                if save_figures:
                    save_figure(plot_top_features(imp_long, row, proxies), _figure_name("topfeatures", row))
                    save_figure(plot_reliance_boxes(per_instance, row), _figure_name("reliance", row))
                    if method == "lime":
                        save_figure(plot_fidelity_boxes(per_instance, row), _figure_name("fidelity", row))
    return {
        "comparison": pd.DataFrame(rows),
        "per_instance": pd.concat(per_instances, ignore_index=True),
        "importance": pd.concat(importances, ignore_index=True),
        "proxies": pd.concat(proxy_tables, ignore_index=True),
    }


def save_comparison_tables(results: dict[str, pd.DataFrame]) -> None:
    """Write the comparison, per-instance, importance and proxy tables."""
    config.ensure_dirs()
    files = {"comparison": COMPARISON_FILE, "per_instance": PER_INSTANCE_FILE,
             "importance": IMPORTANCE_FILE, "proxies": PROXIES_FILE}
    for key, filename in files.items():
        results[key].to_csv(config.table_path(filename), index=False)
        logger.info("Wrote %s (%d rows).", filename, len(results[key]))


# --------------------------------------------------------------------------
# Plain-English interpretation
# --------------------------------------------------------------------------
def interpret_comparison(row: dict, tests: pd.DataFrame | None = None) -> str:
    """One paragraph summarising a comparison row (and its tests, if given).

    Every number comes from the row; nothing is hard-coded.
    """
    model = config.MODEL_DISPLAY_NAMES[row["model"]]
    method = row["method"].upper()
    priv, unpriv = row["privileged"], row["unprivileged"]
    parts = []
    if pd.notna(row["sensitive_reliance_privileged"]):
        parts.append(
            f"For {model}, {method} assigns on average {row['sensitive_reliance_privileged']:.1%} of the total "
            f"attribution to '{row['attribute']}' for {priv} instances and "
            f"{row['sensitive_reliance_unprivileged']:.1%} for {unpriv} instances."
        )
    else:
        parts.append(f"For {model} the sensitive feature '{row['attribute']}' is not a model input in this setting.")
    parts.append(
        f"The proxy features ({row['proxies'].replace(';', ', ')}) receive {row['proxy_reliance_privileged']:.1%} "
        f"({priv}) vs {row['proxy_reliance_unprivileged']:.1%} ({unpriv})."
    )
    rho, low, high = row["spearman_rho"], row["spearman_ci_low"], row["spearman_ci_high"]
    strength = "very similar" if rho >= 0.9 else "similar" if rho >= 0.7 else "noticeably different"
    parts.append(
        f"The two groups' feature rankings are {strength} (Spearman rho {rho:.2f}, 95% CI {low:.2f} to {high:.2f}; "
        f"{row['jaccard_top5']:.0%} overlap in the top 5 features)."
    )
    if row["method"] == "lime" and pd.notna(row["fidelity_gap"]):
        parts.append(
            f"LIME's local fidelity is {row['fidelity_privileged']:.2f} for {priv} and "
            f"{row['fidelity_unprivileged']:.2f} for {unpriv} (gap {row['fidelity_gap']:+.2f})."
        )
    if tests is not None and len(tests):
        sig = tests[tests["significant_holm"]]
        if len(sig):
            names = ", ".join(f"{m.replace('_', ' ')} (adjusted p = {p:.3g})"
                              for m, p in zip(sig["measure"], sig["p_holm"]))
            parts.append(f"After Holm correction the group difference is statistically significant for: {names}.")
        else:
            parts.append("After Holm correction none of the tested group differences is statistically significant.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Trade-off summary (RQ4)
# --------------------------------------------------------------------------
# Direction of "better" for each trade-off column.
TRADEOFF_DIRECTIONS: dict[str, str] = {
    "accuracy": "max",
    "f1": "max",
    "dp_difference": "min",
    "eo_difference": "min",
    "spearman_rho": "max",
    "fidelity_gap_abs": "min",
}


def pareto_optimal(frame: pd.DataFrame, directions: dict[str, str] = TRADEOFF_DIRECTIONS) -> pd.Series:
    """True for rows that no other row dominates.

    Row B dominates row A if B is at least as good as A on every column and
    strictly better on at least one (the standard Pareto definition). Columns
    where either row is NaN (fidelity gap for SHAP rows) are skipped for that
    pair. No weighted score is used: Pareto-optimal rows are the "best
    compromises", and choosing among them is a value judgement.
    """
    signs = np.array([1.0 if directions[c] == "max" else -1.0 for c in directions])
    values = frame[list(directions)].to_numpy(dtype=float) * signs  # now "bigger is better"
    optimal = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        for j in range(len(frame)):
            if i == j:
                continue
            both = ~np.isnan(values[i]) & ~np.isnan(values[j])
            if not both.any():
                continue
            vi, vj = values[i][both], values[j][both]
            if np.all(vj >= vi) and np.any(vj > vi):
                optimal[i] = False
                break
    return pd.Series(optimal, index=frame.index)


def build_tradeoff(
    performance: pd.DataFrame, fairness: pd.DataFrame, comparison: pd.DataFrame
) -> pd.DataFrame:
    """Trade-off table: rows = model x method, per dataset, setting and attribute.

    The spec asks for one table per dataset and setting; because the
    fairness columns depend on the attribute, we build one per attribute as
    well (the primary attribute first) and mark Pareto rows within each.
    """
    keys = ["dataset", "setting", "model"]
    table = comparison[keys + ["method", "attribute", "spearman_rho", "fidelity_gap"]].merge(
        performance[keys + ["accuracy", "f1"]], on=keys, how="left"
    ).merge(
        fairness[keys + ["attribute", "dp_difference", "eo_difference"]], on=keys + ["attribute"], how="left"
    )
    table["fidelity_gap_abs"] = table["fidelity_gap"].abs()
    table = table[keys + ["method", "attribute"] + list(TRADEOFF_DIRECTIONS)]
    # SHAP has no surrogate model, so its fidelity gap is not *measured* and
    # stays NaN in the table. For the Pareto check we treat it as 0: Tree and
    # Linear SHAP are exact (attributions sum to the model output for every
    # instance - SHAP's "local accuracy"), so fidelity is perfect in both
    # groups. Leaving it NaN would ignore this advantage and let a LIME row
    # dominate a SHAP row on the other columns alone.
    pareto_input = table.copy()
    pareto_input.loc[pareto_input["method"] == "shap", "fidelity_gap_abs"] = 0.0
    table["is_pareto"] = False
    for _, index in table.groupby(["dataset", "setting", "attribute"]).groups.items():
        table.loc[index, "is_pareto"] = pareto_optimal(pareto_input.loc[index]).to_numpy()
    return table.reset_index(drop=True)


def save_tradeoff(table: pd.DataFrame) -> None:
    """Write results/tables/tradeoff.csv."""
    config.ensure_dirs()
    table.to_csv(config.table_path(TRADEOFF_FILE), index=False)
    logger.info("Wrote %s (%d rows, %d Pareto-optimal).", TRADEOFF_FILE, len(table), int(table["is_pareto"].sum()))
