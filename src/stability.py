"""Stability module: how much do explanations change under small input noise?

Method designed by the teammate (Stability module, originally a Colab
notebook) and integrated here into the shared pipeline: same data, same
saved models, same SHAP/LIME configuration and same one-hot aggregation as
the Fairness module, so both modules analyse exactly the same models.

Idea
----
A trustworthy explanation should not change much when a person's numeric
features change very slightly (e.g. age 40 -> 40.2). For each sampled test
instance we:
  1. add Gaussian noise to the NUMERIC features only, with standard deviation
     = level x (training std of that feature), clipped to the training range.
     The same random draw z is reused for every level (x + level * std * z),
     so 1%, 5% and 10% differ only in size, not in direction;
  2. explain the original and the perturbed instance;
  3. compare the two explanations (aggregated per original feature):
       rho  - Spearman rank correlation of the attributions (1 = same ranking)
       top3 / top5 - share of the top-3 / top-5 features kept
     and the model itself: pred_flip (did the 0.5-threshold decision change?)
     and dprob (absolute change of the predicted probability).
A noise level of 0 is always included as a baseline: explaining the SAME
input again measures the explainer's own randomness (LIME samples randomly;
Tree/Linear SHAP is deterministic, so its baseline is exactly 1).

Hypotheses (statistical tests)
------------------------------
  S-H1 (SHAP vs LIME): H0 - stability is the same for SHAP and LIME on the
       same instances. Wilcoxon signed-rank test (paired), Holm-corrected.
  S-H2 (across models): H0 - stability is the same for LR, RF and XGBoost.
       Friedman test (paired by instance) with Kendall's W as effect size,
       plus pairwise Wilcoxon post-hoc tests, Holm-corrected.
  S-H3 (across groups, link to the Fairness module): H0 - stability is the
       same for the privileged and unprivileged group. Mann-Whitney U,
       Holm-corrected.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.pipeline import Pipeline

from src import config
from src.compare import pareto_optimal
from src.data import DatasetMetadata
from src.evaluate import new_figure, save_figure
from src.explain import lime_values, make_lime_explainer, shap_values
from src.models import PREPROCESSOR_STEP, load_model
from src.preprocess import SplitData, aggregate_by_original, map_transformed_to_original, prepare_dataset
from src.stats import holm_adjust, mann_whitney

logger = logging.getLogger(__name__)

RAW_FILE = "stability_raw.csv"
SUMMARY_FILE = "stability_summary.csv"
TEST_METHODS_FILE = "stability_test_shap_vs_lime.csv"
TEST_MODELS_FILE = "stability_test_models.csv"
POSTHOC_FILE = "stability_posthoc.csv"
BY_GROUP_FILE = "stability_by_group.csv"
TRADEOFF_FILE = "stability_tradeoff.csv"

# Models are analysed in the "with sensitive features" setting (the teammate's
# original setup used all features).
STABILITY_SETTING = "with_sensitive"


@dataclass(frozen=True)
class StabilitySettings:
    """Sample sizes for one stability run (full or --quick)."""

    n_instances: int = config.STABILITY_N_INSTANCES
    seeds: tuple[int, ...] = config.STABILITY_SEEDS
    lime_samples: int = config.STABILITY_LIME_SAMPLES


QUICK_STABILITY = StabilitySettings(
    n_instances=config.QUICK_STABILITY_N_INSTANCES,
    seeds=config.QUICK_STABILITY_SEEDS,
    lime_samples=config.QUICK_LIME_NUM_SAMPLES,
)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def noise_levels_with_baseline(levels: tuple[float, ...]) -> list[float]:
    """Sorted noise levels with the 0.0 baseline always included."""
    return sorted({0.0, *(float(level) for level in levels)})


def make_perturbations(
    X_sample: pd.DataFrame,
    numeric: list[str],
    X_train: pd.DataFrame,
    levels: list[float],
    rng: np.random.Generator,
) -> dict[float, pd.DataFrame]:
    """Perturbed copies of X_sample, one per noise level.

    Noise is added to numeric features only (categories cannot be nudged a
    little). One standard-normal draw z per cell is shared by all levels, so
    a larger level moves the same instance further in the SAME direction.
    Values are clipped to the training min/max so no impossible values
    (e.g. negative age) appear.
    """
    std, low, high = X_train[numeric].std(), X_train[numeric].min(), X_train[numeric].max()
    z = rng.standard_normal((len(X_sample), len(numeric)))
    perturbed = {}
    for level in levels:
        X_p = X_sample.copy()
        for k, column in enumerate(numeric):
            values = X_sample[column].astype(float).to_numpy() + level * std[column] * z[:, k]
            X_p[column] = np.clip(values, low[column], high[column])
        perturbed[level] = X_p
    return perturbed


def compare_rankings(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Spearman rho and top-3 / top-5 overlap between two attribution vectors.

    If either vector is constant (e.g. all zeros) the ranking is undefined,
    so all three are NaN and the instance is counted as skipped.
    """
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan, np.nan
    rho = float(spearmanr(a, b).statistic)

    def top(v: np.ndarray, k: int) -> set[int]:
        return set(np.argsort(-v, kind="stable")[:k])

    return rho, len(top(a, 3) & top(b, 3)) / 3, len(top(a, 5) & top(b, 5)) / 5


def _explain(
    method: str, pipeline: Pipeline, X_train: pd.DataFrame, X: pd.DataFrame,
    lime_explainer, lime_samples: int, seed: int,
) -> np.ndarray:
    """Raw attributions (n, n_transformed) with the shared explainers."""
    if method == "shap":
        return shap_values(pipeline, X_train, X)
    weights, _, _ = lime_values(pipeline, X_train, X, lime_samples, seed, explainer=lime_explainer)
    return weights


# --------------------------------------------------------------------------
# Public interface (called by the app and by run_all)
# --------------------------------------------------------------------------
def run_stability(
    dataset_name: str,
    model_name: str,
    xai_method: str,
    perturbation_levels: tuple[float, ...] = config.STABILITY_NOISE_LEVELS,
    settings: StabilitySettings = StabilitySettings(),
    *,
    split: SplitData | None = None,
    metadata: DatasetMetadata | None = None,
    pipeline: Pipeline | None = None,
) -> pd.DataFrame:
    """Explanation stability of one dataset x model x XAI method.

    Inputs
      dataset_name        "adult" or "compas"
      model_name          "logistic_regression", "random_forest" or "xgboost"
      xai_method          "shap" or "lime"
      perturbation_levels noise sizes as a fraction of each feature's
                          training std; 0.0 is always added as a baseline
      settings            instances per seed, seeds, LIME samples
      split, metadata, pipeline - optional, already loaded objects (run_all
                          passes them to avoid reloading); otherwise the
                          shared split and the saved model are loaded.

    Output: one row per (seed, instance, noise level) with columns
      dataset, model, method, noise, seed, instance_id, pair_key, rho, top3,
      top5, pred_flip, dprob, and group_<attribute> for every sensitive
      attribute (so stability can be compared between demographic groups).
    """
    if split is None or metadata is None:
        loaded, split = prepare_dataset(dataset_name, save=False)
        metadata = loaded.metadata
    if pipeline is None:
        pipeline = load_model(dataset_name, STABILITY_SETTING, model_name)

    levels = noise_levels_with_baseline(perturbation_levels)
    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    names = list(preprocessor.get_feature_names_out())
    mapping = map_transformed_to_original(preprocessor)
    numeric = [c for c in metadata.numeric if c in split.X_train.columns]

    records = []
    for seed in settings.seeds:
        rng = np.random.default_rng(seed)
        n = min(settings.n_instances, len(split.X_test))
        ids = split.X_test.index[rng.choice(len(split.X_test), size=n, replace=False)]
        X_sample = split.X_test.loc[ids]
        perturbed = make_perturbations(X_sample, numeric, split.X_train, levels, rng)
        # One LIME explainer per seed and model: its random generator keeps
        # running, so every explain call is an independent LIME run.
        lime_explainer = make_lime_explainer(pipeline, split.X_train, seed) if xai_method == "lime" else None

        logger.info("Stability %s | %s | %s | seed %d | %d instances x %d levels",
                    dataset_name, model_name, xai_method, seed, n, len(levels))
        original = aggregate_by_original(
            _explain(xai_method, pipeline, split.X_train, X_sample, lime_explainer, settings.lime_samples, seed),
            names, mapping).to_numpy()
        prob_original = pipeline.predict_proba(X_sample)[:, 1]

        for level in levels:
            if level == 0.0 and xai_method == "shap":
                # Tree/Linear SHAP is deterministic: re-explaining the same
                # input returns identical values, so reuse them.
                again = original
            else:
                again = aggregate_by_original(
                    _explain(xai_method, pipeline, split.X_train, perturbed[level], lime_explainer,
                             settings.lime_samples, seed),
                    names, mapping).to_numpy()
            prob_perturbed = pipeline.predict_proba(perturbed[level])[:, 1]
            for i, instance_id in enumerate(ids):
                rho, top3, top5 = compare_rankings(original[i], again[i])
                record = {
                    "dataset": dataset_name, "model": model_name, "method": xai_method,
                    "noise": level, "seed": seed, "instance_id": instance_id,
                    "pair_key": f"{seed}-{instance_id}",
                    "rho": rho, "top3": top3, "top5": top5,
                    "pred_flip": int((prob_original[i] >= config.DECISION_THRESHOLD)
                                     != (prob_perturbed[i] >= config.DECISION_THRESHOLD)),
                    "dprob": abs(prob_original[i] - prob_perturbed[i]),
                }
                for attribute in metadata.sensitive_attributes:
                    record[f"group_{attribute}"] = split.s_test.at[instance_id, attribute]
                records.append(record)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Summaries and statistical tests
# --------------------------------------------------------------------------
def bootstrap_mean_ci(values: pd.Series, n_boot: int = config.N_BOOTSTRAP,
                      seed: int = config.SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean (NaNs dropped)."""
    x = values.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_stability(raw: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP) -> pd.DataFrame:
    """Mean stability per dataset x model x method x noise level, with a CI for rho."""
    rows = []
    for (dataset, model, method, noise), part in raw.groupby(["dataset", "model", "method", "noise"]):
        low, high = bootstrap_mean_ci(part["rho"], n_boot)
        rows.append({
            "dataset": dataset, "model": model, "method": method, "noise": noise,
            "rho": part["rho"].mean(), "rho_ci_low": low, "rho_ci_high": high,
            "top3": part["top3"].mean(), "top5": part["top5"].mean(),
            "pred_flip_rate": part["pred_flip"].mean(), "mean_abs_dprob": part["dprob"].mean(),
            "n_valid": int(part["rho"].notna().sum()), "n_skipped": int(part["rho"].isna().sum()),
        })
    return pd.DataFrame(rows)


def _safe_wilcoxon(a: pd.Series, b: pd.Series) -> float:
    """Wilcoxon signed-rank p-value; 1.0 if all paired differences are zero."""
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def wilcoxon_shap_vs_lime(raw: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """S-H1: Wilcoxon signed-rank, SHAP vs LIME on the same instances (noise > 0)."""
    rows = []
    noisy = raw[raw["noise"] > 0]
    for metric in ("rho", "top5"):
        for (dataset, model, noise), part in noisy.groupby(["dataset", "model", "noise"]):
            wide = part.pivot_table(index="pair_key", columns="method", values=metric).dropna()
            if len(wide) < 10 or not {"shap", "lime"} <= set(wide.columns):
                continue
            rows.append({
                "dataset": dataset, "model": model, "noise": noise, "metric": metric, "n": len(wide),
                "shap_mean": wide["shap"].mean(), "lime_mean": wide["lime"].mean(),
                "mean_diff": (wide["shap"] - wide["lime"]).mean(),
                "p_value": _safe_wilcoxon(wide["shap"], wide["lime"]),
            })
    return _add_holm(pd.DataFrame(rows), alpha)


def friedman_across_models(raw: pd.DataFrame, alpha: float = config.ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S-H2: Friedman test across the three models, plus pairwise Wilcoxon post-hoc.

    Kendall's W = chi2 / (n (k - 1)) is the effect size: 0 = the models rank
    instances' stability randomly, 1 = perfectly consistent ordering.
    """
    friedman_rows, posthoc_rows = [], []
    noisy = raw[raw["noise"] > 0]
    for metric in ("rho", "top5"):
        for (dataset, method, noise), part in noisy.groupby(["dataset", "method", "noise"]):
            wide = part.pivot_table(index="pair_key", columns="model", values=metric).dropna()
            if len(wide) < 10 or wide.shape[1] < 3:
                continue
            chi2, p = friedmanchisquare(*[wide[c] for c in wide.columns])
            friedman_rows.append({
                "dataset": dataset, "method": method, "noise": noise, "metric": metric, "n": len(wide),
                "friedman_chi2": chi2, "p_value": p, "kendalls_w": chi2 / (len(wide) * (wide.shape[1] - 1)),
            })
            for a, b in itertools.combinations(wide.columns, 2):
                posthoc_rows.append({
                    "dataset": dataset, "method": method, "noise": noise, "metric": metric,
                    "pair": f"{a} vs {b}", "mean_diff": (wide[a] - wide[b]).mean(),
                    "p_value": _safe_wilcoxon(wide[a], wide[b]),
                })
    return _add_holm(pd.DataFrame(friedman_rows), alpha), _add_holm(pd.DataFrame(posthoc_rows), alpha)


def stability_by_group(raw: pd.DataFrame, privileged: dict[tuple[str, str], str],
                       alpha: float = config.ALPHA) -> pd.DataFrame:
    """S-H3: is stability different for the privileged and unprivileged group?

    Each instance-seed pair contributes ONE value: its mean rho over the
    noise levels > 0, so repeated levels of the same person are not counted
    as independent observations. Mann-Whitney U, Holm-corrected.
    `privileged` maps (dataset, attribute) -> privileged group label.
    """
    rows = []
    noisy = raw[raw["noise"] > 0]
    for (dataset, attribute), priv in privileged.items():
        column = f"group_{attribute}"
        part_ds = noisy[noisy["dataset"] == dataset]
        if part_ds.empty or column not in part_ds:
            continue
        for (model, method), part in part_ds.groupby(["model", "method"]):
            per_pair = part.groupby("pair_key").agg(rho=("rho", "mean"), group=(column, "first"))
            priv_values = per_pair.loc[per_pair["group"] == priv, "rho"]
            unpriv_values = per_pair.loc[per_pair["group"] != priv, "rho"]
            test = mann_whitney(priv_values, unpriv_values)
            rows.append({
                "dataset": dataset, "model": model, "method": method, "attribute": attribute,
                "privileged": priv,
                "unprivileged": per_pair.loc[per_pair["group"] != priv, "group"].iloc[0]
                if (per_pair["group"] != priv).any() else "",
                "n_privileged": test["n1"], "n_unprivileged": test["n2"],
                "rho_privileged": priv_values.mean(), "rho_unprivileged": unpriv_values.mean(),
                "rho_gap": priv_values.mean() - unpriv_values.mean(),
                "u_statistic": test["u_statistic"], "p_value": test["p_value"],
                "rank_biserial": test["rank_biserial"],
            })
    return _add_holm(pd.DataFrame(rows), alpha)


def _add_holm(table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Holm-adjusted p-values and the reject-H0 decision for one family of tests."""
    if table.empty:
        return table
    table["p_holm"] = holm_adjust(table["p_value"])
    table["reject_h0"] = table["p_holm"] < alpha
    return table


def stability_tradeoff(summary: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs stability per dataset, with Pareto-optimal rows marked.

    Replaces the original notebook's "balance score" (a harmonic mean of AUC
    and stability): a weighted score hides the trade-off, so - as in the
    Fairness module - we only mark rows that no other row beats on every
    column.
    """
    noisy = summary[summary["noise"] > 0]
    stab = noisy.groupby(["dataset", "model", "method"]).agg(
        stability_rho=("rho", "mean"), stability_top5=("top5", "mean"),
        pred_flip_rate=("pred_flip_rate", "mean")).reset_index()
    perf = performance[performance["setting"] == STABILITY_SETTING][["dataset", "model", "accuracy", "roc_auc"]]
    table = stab.merge(perf, on=["dataset", "model"], how="left")
    directions = {"roc_auc": "max", "stability_rho": "max", "stability_top5": "max", "pred_flip_rate": "min"}
    table["is_pareto"] = False
    for _, index in table.groupby("dataset").groups.items():
        table.loc[index, "is_pareto"] = pareto_optimal(table.loc[index], directions).to_numpy()
    return table


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}
METHOD_STYLES = {"shap": "-", "lime": "--"}


def plot_stability_vs_noise(summary: pd.DataFrame, dataset: str) -> Figure:
    """Spearman rho (with 95% CI) and top-5 overlap against the noise level."""
    part = summary[summary["dataset"] == dataset]
    fig = new_figure((12, 4.5))
    axes = fig.subplots(1, 2)
    for (model, method), group in part.groupby(["model", "method"]):
        group = group.sort_values("noise")
        x = group["noise"] * 100
        label = f"{config.MODEL_DISPLAY_NAMES[model]} + {method.upper()}"
        style = dict(color=MODEL_COLORS[model], linestyle=METHOD_STYLES[method], marker="o", linewidth=2)
        axes[0].errorbar(x, group["rho"], yerr=[group["rho"] - group["rho_ci_low"],
                                                group["rho_ci_high"] - group["rho"]],
                         capsize=3, label=label, **style)
        axes[1].plot(x, group["top5"], label=label, **style)
    for ax, title in zip(axes, ("Spearman rho (original vs perturbed)", "Top-5 feature overlap")):
        ax.set_title(title)
        ax.set_xlabel("Input noise (% of feature std)")
        ax.set_xticks(sorted(part["noise"].unique() * 100))
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean over instances")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{config.DATASET_DISPLAY_NAMES[dataset]} - explanation stability vs input noise "
                 "(solid = SHAP, dashed = LIME)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------
# Runner used by run_all
# --------------------------------------------------------------------------
def run_stability_all(
    prepared: dict,
    settings: StabilitySettings = StabilitySettings(),
    n_boot: int = config.N_BOOTSTRAP,
    save_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """Stability for every dataset x model x method; writes all stability tables.

    `prepared` maps dataset name -> (LoadedDataset, SplitData), as built by run_all.
    """
    raws, privileged = [], {}
    for dataset, (loaded, split) in prepared.items():
        for attribute in loaded.metadata.sensitive_attributes:
            privileged[(dataset, attribute)] = loaded.metadata.privileged[attribute]
        for model_name in config.MODEL_NAMES:
            pipeline = load_model(dataset, STABILITY_SETTING, model_name)
            for method in config.XAI_METHODS:
                raws.append(run_stability(dataset, model_name, method, config.STABILITY_NOISE_LEVELS, settings,
                                          split=split, metadata=loaded.metadata, pipeline=pipeline))
    raw = pd.concat(raws, ignore_index=True)
    summary = summarize_stability(raw, n_boot)
    friedman, posthoc = friedman_across_models(raw)
    performance_path = config.table_path("performance.csv")
    tables = {
        RAW_FILE: raw,
        SUMMARY_FILE: summary,
        TEST_METHODS_FILE: wilcoxon_shap_vs_lime(raw),
        TEST_MODELS_FILE: friedman,
        POSTHOC_FILE: posthoc,
        BY_GROUP_FILE: stability_by_group(raw, privileged),
    }
    if performance_path.exists():
        tables[TRADEOFF_FILE] = stability_tradeoff(summary, pd.read_csv(performance_path))
    config.ensure_dirs()
    for filename, table in tables.items():
        table.to_csv(config.table_path(filename), index=False)
        logger.info("Wrote %s (%d rows).", filename, len(table))
    if save_figures:
        for dataset in prepared:
            save_figure(plot_stability_vs_noise(summary, dataset), f"stability_vs_noise_{dataset}.png")
    return tables
