"""Model construction, training, saving and loading.

Every model is a scikit-learn Pipeline:

    raw DataFrame -> ColumnTransformer (scale + one-hot) -> classifier

Why a Pipeline?
  * The preprocessor is fitted on the training data only, inside fit(), so
    no test-set statistics (means, categories) leak into training.
  * One saved object holds both steps, so the app and the explainers can call
    predict_proba on raw rows without re-implementing preprocessing.
  * Explainers that need transformed data can still reach the two steps as
    pipeline.named_steps["preprocessor"] and pipeline.named_steps["model"].
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data import DatasetMetadata
from src.preprocess import SplitData, build_preprocessor, select_feature_columns

logger = logging.getLogger(__name__)

PREPROCESSOR_STEP = "preprocessor"
MODEL_STEP = "model"


def make_classifier(model_name: str) -> ClassifierMixin:
    """Create an unfitted classifier with the parameters from config.py.

    The three models span the interpretability spectrum:
      * Logistic Regression - linear, a classic "glass box" baseline;
      * Random Forest       - bagged trees, non-linear;
      * XGBoost             - boosted trees, usually the most accurate on
                              tabular data and the most "black box".
    """
    if model_name == "logistic_regression":
        return LogisticRegression(**config.LOGREG_PARAMS)
    if model_name == "random_forest":
        return RandomForestClassifier(**config.RF_PARAMS)
    if model_name == "xgboost":
        return XGBClassifier(**config.XGB_PARAMS)
    raise ValueError(f"Unknown model {model_name!r}; choose from {config.MODEL_NAMES}")


def build_pipeline(
    model_name: str, metadata: DatasetMetadata, include_sensitive: bool
) -> Pipeline:
    """Build an unfitted preprocessor + classifier Pipeline for one setting.

    include_sensitive decides which columns the preprocessor keeps; the
    ColumnTransformer drops every other column, so the same raw X can be
    passed in for both settings.
    """
    numeric, categorical = select_feature_columns(metadata, include_sensitive)
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_preprocessor(numeric, categorical)),
            (MODEL_STEP, make_classifier(model_name)),
        ]
    )


def train_model(
    model_name: str, split: SplitData, metadata: DatasetMetadata, include_sensitive: bool
) -> Pipeline:
    """Fit one Pipeline on the training split and return it."""
    pipeline = build_pipeline(model_name, metadata, include_sensitive)
    pipeline.fit(split.X_train, split.y_train)
    return pipeline


def model_path(dataset: str, setting: str, model_name: str) -> Path:
    """File name for a saved model, e.g. models/adult__with_sensitive__xgboost.joblib."""
    return config.MODELS_DIR / f"{dataset}__{setting}__{model_name}.joblib"


def save_model(pipeline: Pipeline, dataset: str, setting: str, model_name: str) -> Path:
    """Save a fitted Pipeline with joblib (efficient for numpy-heavy objects)."""
    config.ensure_dirs()
    path = model_path(dataset, setting, model_name)
    joblib.dump(pipeline, path)
    return path


def load_model(dataset: str, setting: str, model_name: str) -> Pipeline:
    """Load a Pipeline saved by save_model."""
    path = model_path(dataset, setting, model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - run training first (python -m src.evaluate).")
    return joblib.load(path)


def train_all(
    split: SplitData,
    metadata: DatasetMetadata,
    settings: dict[str, bool] | None = None,
    model_names: tuple[str, ...] = config.MODEL_NAMES,
    save: bool = True,
) -> dict[tuple[str, str], Pipeline]:
    """Train every model for every setting of one dataset.

    Returns a dict keyed by (setting, model_name).
    """
    settings = settings if settings is not None else config.SETTINGS
    fitted: dict[tuple[str, str], Pipeline] = {}
    for setting, include in settings.items():
        for model_name in model_names:
            logger.info("Training %s | %s | %s", metadata.name, setting, model_name)
            pipeline = train_model(model_name, split, metadata, include)
            if save:
                save_model(pipeline, metadata.name, setting, model_name)
            fitted[(setting, model_name)] = pipeline
    return fitted
