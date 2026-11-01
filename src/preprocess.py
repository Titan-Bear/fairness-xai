"""Train/test splitting, feature preprocessing and feature-name mapping.

Three ideas to remember for the viva:

1. The split is stratified on the target, so train and test keep the same
   share of positive cases. The sensitive attributes are split with the same
   row indices, so every test row still knows its demographic group.

2. Preprocessing (scaling + one-hot encoding) lives in a ColumnTransformer
   that is later placed inside each model's Pipeline. It is fitted on the
   training data only, so no information from the test set leaks in.

3. One-hot encoding turns one original feature (e.g. `sex`) into several
   columns (`sex_Female`, `sex_Male`). Explanations are produced per column,
   but people think per original feature, so we map every transformed column
   back to its original feature and sum absolute attributions over them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import DatasetMetadata, LoadedDataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Train/test split with the sensitive attributes kept row-aligned."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    s_train: pd.DataFrame
    s_test: pd.DataFrame


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> SplitData:
    """Stratified train/test split that keeps `sensitive` aligned with X.

    We split the row *indices* once and use them for X, y and sensitive, so
    the three can never get out of step. Because the split does not depend on
    which columns X has, both settings (with/without sensitive features) use
    exactly the same train and test rows, which makes them comparable.
    """
    train_idx, test_idx = train_test_split(
        X.index.to_numpy(),
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    return SplitData(
        X_train=X.loc[train_idx],
        X_test=X.loc[test_idx],
        y_train=y.loc[train_idx],
        y_test=y.loc[test_idx],
        s_train=sensitive.loc[train_idx],
        s_test=sensitive.loc[test_idx],
    )


# --------------------------------------------------------------------------
# Feature selection per setting
# --------------------------------------------------------------------------
def select_feature_columns(
    metadata: DatasetMetadata, include_sensitive: bool = config.INCLUDE_SENSITIVE
) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns for one setting.

    With include_sensitive=False we drop the sensitive attributes
    ("fairness through unawareness"). The model can no longer use sex or race
    directly, but correlated features (proxies) may still carry the bias -
    which is exactly what the comparison stage tests.
    """
    drop = set() if include_sensitive else set(metadata.sensitive_attributes)
    numeric = [c for c in metadata.numeric if c not in drop]
    categorical = [c for c in metadata.categorical if c not in drop]
    return numeric, categorical


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer.

    * StandardScaler on numeric columns: Logistic Regression and LIME's
      perturbations both work better when features share a scale. Trees do
      not need it, but one shared preprocessor keeps the models comparable.
    * OneHotEncoder(handle_unknown="ignore"): a category seen only in the test
      set becomes all zeros instead of crashing. Dense output because SHAP and
      LIME expect dense arrays.
    * verbose_feature_names_out=False gives readable names ("age",
      "sex_Male") instead of "num__age", "cat__sex_Male".
    * remainder="drop": columns not listed (e.g. a dropped sensitive
      attribute) never reach the model.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names of the transformed columns, in output order (fitted preprocessor)."""
    return list(preprocessor.get_feature_names_out())


def map_transformed_to_original(preprocessor: ColumnTransformer) -> dict[str, str]:
    """Map every transformed column name to its original feature.

    Example: {"age": "age", "sex_Female": "sex", "sex_Male": "sex"}.

    Why not just split the name at "_"? Because original names can contain
    underscores themselves (COMPAS has `c_charge_degree` and `juv_fel_count`),
    so "c_charge_degree_F" would be split wrongly. Instead we read the fitted
    encoder: each categorical column produces exactly len(categories_) output
    columns, in order. This is exact for any column names.
    """
    original_order: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            original_order.extend(columns)
        elif name == "cat":
            for column, categories in zip(columns, transformer.categories_):
                original_order.extend([column] * len(categories))
        # "remainder" is dropped, so it contributes no output columns.

    transformed = get_transformed_feature_names(preprocessor)
    if len(transformed) != len(original_order):
        raise RuntimeError("Transformed/original column counts differ; mapping would be wrong.")
    return dict(zip(transformed, original_order))


def aggregate_by_original(
    attributions: np.ndarray | pd.DataFrame,
    transformed_names: list[str],
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Sum ABSOLUTE attributions over the columns of each original feature.

    Input: array (n_instances, n_transformed_columns) of SHAP or LIME values.
    Output: DataFrame (n_instances, n_original_features).

    We take absolute values *before* summing because the sign says direction
    (towards or away from the positive class), while we want magnitude
    ("how much did the model rely on sex?"). Without abs, sex_Male = +0.3 and
    sex_Female = -0.3 would cancel to 0 and hide a strong reliance.
    """
    values = np.abs(np.asarray(attributions, dtype=float))
    frame = pd.DataFrame(values, columns=transformed_names)
    groups = [mapping[name] for name in transformed_names]
    # Preserve first-appearance order of original features for readability.
    order = list(dict.fromkeys(groups))
    return frame.T.groupby(groups, sort=False).sum().T[order]


def fit_transform_frame(
    preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on train, transform both, and return labelled DataFrames.

    Used for inspection, tests and the explainers; the models themselves fit
    the preprocessor inside their Pipeline.
    """
    train_arr = preprocessor.fit_transform(X_train)
    test_arr = preprocessor.transform(X_test)
    names = get_transformed_feature_names(preprocessor)
    return (
        pd.DataFrame(train_arr, columns=names, index=X_train.index),
        pd.DataFrame(test_arr, columns=names, index=X_test.index),
    )


# --------------------------------------------------------------------------
# Saving / loading processed splits
# --------------------------------------------------------------------------
SENSITIVE_PREFIX = "sensitive__"


def save_splits(split: SplitData, metadata: DatasetMetadata) -> None:
    """Save train and test splits to data/processed/<dataset>_{train,test}.csv.

    Each file holds the untransformed features, the target and the sensitive
    labels (prefixed "sensitive__"). We save one copy per dataset, not per
    setting, because both settings use the same rows. We save untransformed
    data because the transformation is part of each fitted model Pipeline.
    """
    config.ensure_dirs()
    for part, X, y, s in (
        ("train", split.X_train, split.y_train, split.s_train),
        ("test", split.X_test, split.y_test, split.s_test),
    ):
        frame = X.copy()
        frame[metadata.target_name] = y
        for column in s.columns:
            frame[SENSITIVE_PREFIX + column] = s[column]
        path = config.DATA_PROCESSED / f"{metadata.name}_{part}.csv"
        frame.to_csv(path, index_label="row_id")
        logger.info("Saved %s (%d rows).", path.name, len(frame))


def load_splits(metadata: DatasetMetadata) -> SplitData:
    """Read the splits written by save_splits (keeps the original row ids)."""
    parts = {}
    for part in ("train", "test"):
        path = config.DATA_PROCESSED / f"{metadata.name}_{part}.csv"
        frame = pd.read_csv(path, index_col="row_id")
        sens_cols = [c for c in frame.columns if c.startswith(SENSITIVE_PREFIX)]
        s = frame[sens_cols].rename(columns=lambda c: c[len(SENSITIVE_PREFIX):])
        y = frame[metadata.target_name]
        # .copy() makes X its own frame, so converting columns below does not
        # write into a slice of `frame` (pandas' SettingWithCopyWarning).
        X = frame[metadata.feature_names].copy()
        for col in metadata.categorical:
            X[col] = X[col].astype(str)
        parts[part] = (X, y, s)
    return SplitData(
        X_train=parts["train"][0], X_test=parts["test"][0],
        y_train=parts["train"][1], y_test=parts["test"][1],
        s_train=parts["train"][2], s_test=parts["test"][2],
    )


def prepare_dataset(name: str, save: bool = True) -> tuple[LoadedDataset, SplitData]:
    """Load a dataset, split it and (optionally) save the splits."""
    loaded = load_dataset(name)
    split = split_data(loaded.X, loaded.y, loaded.sensitive)
    if save:
        save_splits(split, loaded.metadata)
    return loaded, split


def main() -> None:
    """Phase 1 check: load both datasets, split, preprocess and report shapes."""
    config.setup_logging()
    for name in config.DATASETS:
        loaded, split = prepare_dataset(name)
        for setting, include in config.SETTINGS.items():
            numeric, categorical = select_feature_columns(loaded.metadata, include)
            pre = build_preprocessor(numeric, categorical)
            Xtr, Xte = fit_transform_frame(pre, split.X_train, split.X_test)
            mapping = map_transformed_to_original(pre)
            logger.info(
                "%s | %s | train %s test %s | %d transformed cols from %d features | NaNs: %d",
                name, setting, Xtr.shape, Xte.shape, Xtr.shape[1],
                len(set(mapping.values())), int(Xtr.isna().sum().sum() + Xte.isna().sum().sum()),
            )
        for attr in loaded.metadata.sensitive_attributes:
            logger.info("%s | test groups by %s: %s", name, attr,
                        split.s_test[attr].value_counts().to_dict())


if __name__ == "__main__":
    main()
