"""SHAP and LIME explanations for sampled test instances of each group.

Pipeline of this module
-----------------------
1. sample_instances: for every sensitive attribute and each of its two
   groups, draw up to SHAP_N_PER_GROUP test rows (seeded). The first
   LIME_N_PER_GROUP of those rows are the LIME sample, so LIME instances are
   always a subset of SHAP instances ("same instances for SHAP and LIME").
   The sample depends only on the test split and the seed, so every model
   and both settings explain exactly the same people.
2. shap_values / lime_values: attributions for every transformed column,
   for the positive class.
3. aggregate_by_original (from preprocess): absolute attributions summed
   over each original feature's one-hot columns.
4. Results are cached in results/cache/ so the app never recomputes them.

Units: SHAP values are in log-odds for Logistic Regression and XGBoost and in
probability for Random Forest; LIME weights are slopes of a local linear
model. The comparison stage only uses *shares* of total attribution and
*rankings*, which are unit-free, so this does not bias the comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.data import DatasetMetadata
from src.models import MODEL_STEP, PREPROCESSOR_STEP
from src.preprocess import SplitData, aggregate_by_original, map_transformed_to_original

logger = logging.getLogger(__name__)

# Columns of the cache files that are not features.
META_COLUMNS = ["instance_id", "method", "model_proba", "fidelity", "local_pred"]


@dataclass(frozen=True)
class ExplainSettings:
    """Sample sizes for one explanation run (full or --quick)."""

    shap_n_per_group: int = config.SHAP_N_PER_GROUP
    lime_n_per_group: int = config.LIME_N_PER_GROUP
    lime_num_samples: int = config.LIME_NUM_SAMPLES
    seed: int = config.SEED

    def __post_init__(self) -> None:
        if self.lime_n_per_group > self.shap_n_per_group:
            raise ValueError("LIME instances must be a subset of SHAP instances (lime_n <= shap_n).")

    @property
    def tag(self) -> str:
        """Short id used in cache file names, so full and quick runs never mix."""
        return f"s{self.shap_n_per_group}_l{self.lime_n_per_group}_n{self.lime_num_samples}"


QUICK_SETTINGS = ExplainSettings(
    shap_n_per_group=config.QUICK_SHAP_N_PER_GROUP,
    lime_n_per_group=config.QUICK_LIME_N_PER_GROUP,
    lime_num_samples=config.QUICK_LIME_NUM_SAMPLES,
)


# --------------------------------------------------------------------------
# 1. Sampling
# --------------------------------------------------------------------------
def sample_instances(
    split: SplitData, metadata: DatasetMetadata, settings: ExplainSettings
) -> pd.DataFrame:
    """Choose which test rows to explain.

    Returns one row per sampled test instance (index = original row id) with,
    for every sensitive attribute A:
      group_A    - the instance's group label for A
      in_shap_A  - True if sampled for the SHAP comparison of A
      in_lime_A  - True if sampled for the LIME comparison of A

    Sampling per group (instead of a random sample of the whole test set)
    guarantees enough instances from small groups such as Non-White or
    female defendants. If a group is smaller than n, all its rows are used.
    """
    shap_ids: dict[str, set] = {}
    lime_ids: dict[str, set] = {}
    for attribute in metadata.sensitive_attributes:
        rng = np.random.default_rng(settings.seed)
        shap_ids[attribute], lime_ids[attribute] = set(), set()
        for _, index in sorted(split.s_test.groupby(attribute).groups.items()):
            shuffled = rng.permutation(np.asarray(index))
            shap_ids[attribute].update(shuffled[: settings.shap_n_per_group])
            # The first lime_n of the same shuffled order -> LIME subset of SHAP.
            lime_ids[attribute].update(shuffled[: settings.lime_n_per_group])

    all_ids = sorted(set().union(*shap_ids.values()))
    instances = pd.DataFrame(index=pd.Index(all_ids, name="instance_id"))
    for attribute in metadata.sensitive_attributes:
        instances[f"group_{attribute}"] = split.s_test.loc[all_ids, attribute].to_numpy()
        instances[f"in_shap_{attribute}"] = instances.index.isin(list(shap_ids[attribute]))
        instances[f"in_lime_{attribute}"] = instances.index.isin(list(lime_ids[attribute]))
    return instances


def lime_instance_ids(instances: pd.DataFrame) -> list:
    """Rows that need a LIME explanation (in the LIME sample of any attribute)."""
    mask = instances.filter(like="in_lime_").any(axis=1)
    return list(instances.index[mask])


# --------------------------------------------------------------------------
# 2a. SHAP
# --------------------------------------------------------------------------
def positive_class_shap(values: np.ndarray | list) -> np.ndarray:
    """Return SHAP values for the positive class as an (n, n_features) array.

    Why this is needed: shap returns different shapes per model type.
      * XGBoost and LinearExplainer: one (n, f) array - a binary model has a
        single output (log-odds of class 1), so there is nothing to select.
      * scikit-learn Random Forest: it predicts BOTH class probabilities, so
        shap returns one set of values per class - a list [class0, class1] in
        older shap versions, a 3-D array (n, f, 2) in newer ones. Class 0
        values are exactly minus the class 1 values, so we keep class 1.
    """
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def shap_values(pipeline: Pipeline, X_train: pd.DataFrame, X_explain: pd.DataFrame) -> np.ndarray:
    """SHAP values (positive class) of the transformed columns for X_explain.

    * Tree models: TreeExplainer computes exact Shapley values from the tree
      structure (fast, no sampling).
    * Logistic Regression: LinearExplainer. For a linear model the Shapley
      value of a feature is coef * (x - mean(x)), where the mean is taken
      over a background sample of training rows.
    """
    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    model = pipeline.named_steps[MODEL_STEP]
    X_t = preprocessor.transform(X_explain)
    if isinstance(model, LogisticRegression):
        rng = np.random.default_rng(config.SEED)
        X_train_t = preprocessor.transform(X_train)
        size = min(config.SHAP_BACKGROUND_N, len(X_train_t))
        background = X_train_t[rng.choice(len(X_train_t), size=size, replace=False)]
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)
    return positive_class_shap(explainer.shap_values(X_t))


# --------------------------------------------------------------------------
# 2b. LIME
# --------------------------------------------------------------------------
def make_lime_explainer(pipeline: Pipeline, X_train: pd.DataFrame, seed: int) -> LimeTabularExplainer:
    """LimeTabularExplainer on the transformed training data.

    One-hot columns are declared categorical, so LIME perturbs them by
    sampling 0/1 with the training frequencies instead of treating them as
    continuous numbers. (Known LIME limitation: it perturbs each one-hot
    column independently, so a perturbed row can have two categories "on".)
    Numeric columns are discretised into quartiles (LIME's default), which
    makes the local surrogate more stable.
    """
    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    names = list(preprocessor.get_feature_names_out())
    mapping = map_transformed_to_original(preprocessor)
    # A one-hot column's name differs from its original feature (sex_Male vs sex).
    categorical_idx = [i for i, name in enumerate(names) if mapping[name] != name]
    return LimeTabularExplainer(
        training_data=preprocessor.transform(X_train),
        mode="classification",
        feature_names=names,
        class_names=["0", "1"],
        categorical_features=categorical_idx,
        discretize_continuous=True,
        # We keep all features, so LIME's feature-selection step is skipped.
        feature_selection="none",
        random_state=seed,
    )


def lime_values(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    num_samples: int,
    seed: int,
    explainer: LimeTabularExplainer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LIME weights (n, n_transformed), fidelity (n,) and local predictions (n,).

    For each instance LIME samples `num_samples` perturbed rows around it,
    asks the model for their probabilities (predict_fn = the classifier's
    predict_proba on transformed rows), and fits a weighted linear model.

    exp.score is the R-squared of that linear surrogate on the perturbed
    samples (weighted by proximity). We call it FIDELITY: how well the simple
    explanation imitates the model near this person. Low fidelity means the
    explanation should be trusted less. Note it is measured on LIME's own
    synthetic neighbourhood, not on real data.

    An existing `explainer` can be passed in: its random generator then
    continues from where it stopped instead of restarting at `seed`. The
    stability module uses this, so explaining the same row twice gives two
    genuinely independent LIME runs (LIME's own randomness).
    """
    if explainer is None:
        explainer = make_lime_explainer(pipeline, X_train, seed)
    model = pipeline.named_steps[MODEL_STEP]
    X_t = pipeline.named_steps[PREPROCESSOR_STEP].transform(X_explain)
    n_features = X_t.shape[1]
    weights = np.zeros_like(X_t, dtype=float)
    fidelity = np.zeros(len(X_t))
    local_pred = np.zeros(len(X_t))
    for i, row in enumerate(X_t):
        exp = explainer.explain_instance(
            row, model.predict_proba, num_features=n_features,
            num_samples=num_samples, labels=(1,),
        )
        for feature_idx, weight in exp.as_map()[1]:
            weights[i, feature_idx] = weight
        fidelity[i] = exp.score
        local_pred[i] = np.ravel(exp.local_pred)[0]
    return weights, fidelity, local_pred


# --------------------------------------------------------------------------
# 3. One model: explain, aggregate, cache
# --------------------------------------------------------------------------
def cache_path(dataset: str, setting: str, model_name: str, settings: ExplainSettings) -> Path:
    """Cache file for one (dataset, setting, model) explanation run."""
    return config.CACHE_DIR / f"explanations_{dataset}_{setting}_{model_name}_{settings.tag}.csv"


def _load_valid_cache(path: Path, proba: pd.Series) -> pd.DataFrame | None:
    """Return the cached table if it exists AND matches the current model.

    We store the model's predicted probability for every explained row. If
    the model was retrained differently (e.g. new parameters), the stored
    probabilities no longer match and the cache is ignored. This prevents
    silently showing explanations of an old model.
    """
    if not path.exists():
        return None
    cached = pd.read_csv(path)
    cached_proba = cached.drop_duplicates("instance_id").set_index("instance_id")["model_proba"]
    if set(cached_proba.index) != set(proba.index):
        return None  # different instances were sampled
    if not np.allclose(cached_proba.loc[proba.index].to_numpy(), proba.to_numpy(), atol=1e-6):
        logger.info("Cache %s is stale (model changed) - recomputing.", path.name)
        return None
    return cached


def explain_model(
    pipeline: Pipeline,
    split: SplitData,
    instances: pd.DataFrame,
    dataset: str,
    setting: str,
    model_name: str,
    settings: ExplainSettings,
    force: bool = False,
    methods: tuple[str, ...] = config.XAI_METHODS,
) -> pd.DataFrame:
    """SHAP and/or LIME for one fitted model; returns the per-instance table.

    `methods` defaults to both. The mitigation step passes ("lime",) because
    the mitigated model is not a tree or linear model, so only the
    model-agnostic LIME can explain it.

    Output: one row per (instance, method) with columns
      instance_id, method, model_proba, fidelity (LIME only), local_pred,
      then one column per ORIGINAL feature holding the absolute attribution
      summed over its one-hot columns.
    """
    shap_ids = list(instances.index)
    lime_ids = lime_instance_ids(instances)
    explained_ids = shap_ids if "shap" in methods else lime_ids
    proba = pd.Series(pipeline.predict_proba(split.X_test.loc[explained_ids])[:, 1], index=explained_ids)

    path = cache_path(dataset, setting, model_name, settings)
    if not force:
        cached = _load_valid_cache(path, proba)
        if cached is not None:
            logger.info("Using cached explanations %s", path.name)
            return cached

    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    names = list(preprocessor.get_feature_names_out())
    mapping = map_transformed_to_original(preprocessor)

    frames = []
    if "shap" in methods:
        logger.info("SHAP  %s | %s | %s: %d instances", dataset, setting, model_name, len(shap_ids))
        shap_raw = shap_values(pipeline, split.X_train, split.X_test.loc[shap_ids])
        shap_frame = aggregate_by_original(shap_raw, names, mapping)
        shap_frame.insert(0, "local_pred", np.nan)
        shap_frame.insert(0, "fidelity", np.nan)
        shap_frame.insert(0, "model_proba", proba.loc[shap_ids].to_numpy())
        shap_frame.insert(0, "method", "shap")
        shap_frame.insert(0, "instance_id", shap_ids)
        frames.append(shap_frame)

    if "lime" in methods:
        frames.append(_lime_frame(pipeline, split, lime_ids, proba, names, mapping, settings,
                                  f"{dataset} | {setting} | {model_name}"))

    table = pd.concat(frames, ignore_index=True)
    config.ensure_dirs()
    table.to_csv(path, index=False)
    return table


def _lime_frame(
    pipeline: Pipeline,
    split: SplitData,
    lime_ids: list,
    proba: pd.Series,
    names: list[str],
    mapping: dict[str, str],
    settings: ExplainSettings,
    label: str,
) -> pd.DataFrame:
    """LIME rows of the explanation table (aggregated per original feature)."""
    logger.info("LIME  %s: %d instances x %d samples",
                label, len(lime_ids), settings.lime_num_samples)
    lime_raw, fidelity, local_pred = lime_values(
        pipeline, split.X_train, split.X_test.loc[lime_ids], settings.lime_num_samples, settings.seed
    )
    lime_frame = aggregate_by_original(lime_raw, names, mapping)
    lime_frame.insert(0, "local_pred", local_pred)
    lime_frame.insert(0, "fidelity", fidelity)
    lime_frame.insert(0, "model_proba", proba.loc[lime_ids].to_numpy())
    lime_frame.insert(0, "method", "lime")
    lime_frame.insert(0, "instance_id", lime_ids)
    return lime_frame


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Original-feature columns of an explanation table."""
    return [c for c in table.columns if c not in META_COLUMNS]


def tidy_explanations(table: pd.DataFrame, instances: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Long ("tidy") view for one sensitive attribute.

    Columns: instance_id, group, method, original_feature, attribution - only
    for the instances sampled for this attribute and method.
    """
    long = table.melt(
        id_vars=["instance_id", "method"], value_vars=feature_columns(table),
        var_name="original_feature", value_name="attribution",
    )
    in_sample = pd.concat([
        instances[f"in_shap_{attribute}"].rename("shap"),
        instances[f"in_lime_{attribute}"].rename("lime"),
    ], axis=1)
    keep = [bool(in_sample.at[i, m]) for i, m in zip(long["instance_id"], long["method"])]
    long = long[keep].copy()
    long.insert(1, "group", instances.loc[long["instance_id"], f"group_{attribute}"].to_numpy())
    return long.reset_index(drop=True)


def instances_path(dataset: str, settings: ExplainSettings) -> Path:
    """Cache file for the sampled instances of one dataset."""
    return config.CACHE_DIR / f"instances_{dataset}_{settings.tag}.csv"


def run_explanations(
    dataset: str,
    fitted: dict[tuple[str, str], Pipeline],
    split: SplitData,
    metadata: DatasetMetadata,
    settings: ExplainSettings = ExplainSettings(),
    force: bool = False,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Explain every (setting, model) of one dataset on the same instances."""
    instances = sample_instances(split, metadata, settings)
    config.ensure_dirs()
    instances.to_csv(instances_path(dataset, settings))
    tables = {
        key: explain_model(pipeline, split, instances, dataset, key[0], key[1], settings, force)
        for key, pipeline in fitted.items()
    }
    return instances, tables
