"""Dataset loaders for Adult Income and COMPAS.

Each loader returns a LoadedDataset with four parts:
  X          - feature DataFrame (what the model may see)
  y          - binary target Series (1 = positive outcome)
  sensitive  - DataFrame of sensitive attributes as readable labels
               (e.g. "Male"/"Female"), same index as X
  metadata   - DatasetMetadata: column lists and privileged groups

Keeping `sensitive` separate from X matters: in the "without sensitive"
setting we drop those columns from X, but we still need them to group the
test rows when we measure fairness.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from typing import NamedTuple

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Columns of the original UCI Adult files (they have no header row).
ADULT_UCI_COLUMNS: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]

# COMPAS columns that are the COMPAS tool's own risk output. Using them as
# features would be target leakage: we would be predicting recidivism from a
# score that was itself built to predict recidivism, and we would inherit the
# bias we are trying to audit. They must never appear in X.
COMPAS_LEAKAGE_COLUMNS: tuple[str, ...] = ("decile_score", "score_text", "v_decile_score")


@dataclass
class DatasetMetadata:
    """Descriptive information about a dataset, used by every later module."""

    name: str
    target_name: str
    positive_label: str  # plain-English meaning of y == 1, for sentences
    sensitive_attributes: list[str]  # primary first
    categorical: list[str]
    numeric: list[str]
    privileged: dict[str, str] = field(default_factory=dict)

    @property
    def primary_sensitive(self) -> str:
        """The main sensitive attribute (first in the list)."""
        return self.sensitive_attributes[0]

    @property
    def feature_names(self) -> list[str]:
        """All original feature names (numeric then categorical)."""
        return self.numeric + self.categorical


class LoadedDataset(NamedTuple):
    """A NamedTuple so callers can write `X, y, sensitive, meta = load_adult()`."""

    X: pd.DataFrame
    y: pd.Series
    sensitive: pd.DataFrame
    metadata: DatasetMetadata


# --------------------------------------------------------------------------
# Adult Income
# --------------------------------------------------------------------------
def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical/object columns to stripped strings and '?' to NaN.

    OpenML returns pandas 'category' columns while the UCI files give plain
    strings with leading spaces; normalising both to str keeps later code
    (one-hot encoding, group labels) identical whichever source was used.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            # astype(object) keeps real missing values as NaN (astype(str)
            # would turn them into the text "nan" and hide them from dropna).
            values = df[col].astype(object)
            values = values.map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = values.replace("?", pd.NA)
    return df


def _fetch_adult_raw() -> pd.DataFrame:
    """Return the raw Adult table (features + 'class'), online or offline.

    First try fairlearn's fetch_adult (OpenML, cached in data/raw/openml_cache
    so it also works offline after the first run). If that fails, fall back
    to the original UCI files adult.data / adult.test in data/raw/.
    """
    try:
        from fairlearn.datasets import fetch_adult

        bunch = fetch_adult(as_frame=True, data_home=str(config.ADULT_OPENML_CACHE))
        frame = bunch.data.copy()
        frame["class"] = bunch.target.astype(str)
        logger.info("Loaded Adult from OpenML via fairlearn (%d rows).", len(frame))
        return frame
    except Exception as exc:  # network errors, OpenML outages, etc.
        logger.warning("fetch_adult failed (%s); trying UCI files in data/raw/.", exc)

    if not (config.ADULT_UCI_TRAIN.exists() and config.ADULT_UCI_TEST.exists()):
        raise FileNotFoundError(
            "Adult could not be downloaded and the UCI fallback files are missing. "
            f"Place adult.data and adult.test in {config.DATA_RAW}."
        )
    read_kwargs = dict(header=None, names=ADULT_UCI_COLUMNS, skipinitialspace=True)
    train = pd.read_csv(config.ADULT_UCI_TRAIN, **read_kwargs)
    # adult.test starts with a comment line ("|1x3 Cross validator").
    test = pd.read_csv(config.ADULT_UCI_TEST, skiprows=1, **read_kwargs)
    frame = pd.concat([train, test], ignore_index=True)
    logger.info("Loaded Adult from UCI files (%d rows).", len(frame))
    return frame


def load_adult() -> LoadedDataset:
    """Load and clean the Adult Income dataset.

    Cleaning choices:
      * target: income >50K -> 1, else 0;
      * drop fnlwgt (a census sampling weight, not a property of the person);
      * drop education (education-num encodes the same levels as a number);
      * '?' is missing; rows with any missing value are dropped;
      * race is binarised to White / Non-White so every fairness comparison
        is between exactly two groups. We binarise the feature column too, so
        the sensitive attribute and the model input are the same variable.
    """
    config.ensure_dirs()
    df = _clean_strings(_fetch_adult_raw())
    df = df.drop(columns=["fnlwgt", "education"])
    df = df.dropna().reset_index(drop=True)

    # The UCI test file writes labels as ">50K." with a trailing dot.
    labels = df.pop("class").astype(str).str.strip().str.rstrip(".")
    y = (labels == ">50K").astype(int).rename("income_gt_50k")

    df["race"] = df["race"].where(df["race"] == "White", "Non-White")

    numeric = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical = [
        "workclass", "marital-status", "occupation", "relationship",
        "race", "sex", "native-country",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col])
    X = df[numeric + categorical].copy()

    sensitive = X[["sex", "race"]].copy()
    metadata = DatasetMetadata(
        name="adult",
        target_name="income_gt_50k",
        positive_label=">50K",
        sensitive_attributes=["sex", "race"],
        categorical=categorical,
        numeric=numeric,
        privileged={"sex": "Male", "race": "White"},
    )
    logger.info("Adult ready: %d rows, %d features, positive rate %.3f.",
                len(X), X.shape[1], y.mean())
    return LoadedDataset(X, y, sensitive, metadata)


# --------------------------------------------------------------------------
# COMPAS
# --------------------------------------------------------------------------
def download_compas(force: bool = False) -> None:
    """Download ProPublica's compas-scores-two-years.csv into data/raw/ once."""
    config.ensure_dirs()
    if config.COMPAS_FILE.exists() and not force:
        return
    logger.info("Downloading COMPAS from %s", config.COMPAS_URL)
    request = urllib.request.Request(config.COMPAS_URL, headers={"User-Agent": "fairness-xai"})
    with urllib.request.urlopen(request, timeout=60) as response:
        config.COMPAS_FILE.write_bytes(response.read())
    logger.info("Saved COMPAS to %s", config.COMPAS_FILE)


def apply_propublica_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ProPublica's standard data-quality filters.

    * days_b_screening_arrest in [-30, 30]: if the COMPAS screening was far
      from the arrest date, we may be looking at the wrong offence.
    * is_recid != -1: -1 means no COMPAS case was found.
    * c_charge_degree != "O": ordinary traffic offences do not lead to jail.
    * score_text != "N/A": the score is missing.
    Rows with a missing days_b_screening_arrest are also dropped, because
    `between` returns False for NaN (the same as ProPublica's R analysis).
    """
    mask = (
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    )
    return df.loc[mask].copy()


def load_compas() -> LoadedDataset:
    """Load and clean COMPAS (two-year recidivism).

    Only African-American and Caucasian defendants are kept so the race
    comparison is between exactly two groups (ProPublica's main comparison).
    """
    download_compas()
    raw = pd.read_csv(config.COMPAS_FILE)
    df = apply_propublica_filters(raw)
    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)

    numeric = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    categorical = ["sex", "race", "c_charge_degree"]
    # NOTE: decile_score, score_text and v_decile_score are deliberately NOT
    # features - they are the COMPAS tool's own output (target leakage).
    X = df[numeric + categorical].copy()
    for col in categorical:
        X[col] = X[col].astype(str)
    y = df["two_year_recid"].astype(int).rename("two_year_recid")

    sensitive = X[["race", "sex"]].copy()
    metadata = DatasetMetadata(
        name="compas",
        target_name="two_year_recid",
        positive_label="re-offending within two years",
        sensitive_attributes=["race", "sex"],
        categorical=categorical,
        numeric=numeric,
        # "Privileged" = the group that receives the favourable outcome more
        # often. In COMPAS the favourable outcome is being predicted NOT to
        # re-offend, which happens more for Caucasian and for female defendants.
        privileged={"race": "Caucasian", "sex": "Female"},
    )
    logger.info("COMPAS ready: %d rows, %d features, positive rate %.3f.",
                len(X), X.shape[1], y.mean())
    return LoadedDataset(X, y, sensitive, metadata)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
LOADERS = {"adult": load_adult, "compas": load_compas}


def load_dataset(name: str) -> LoadedDataset:
    """Load a dataset by name ('adult' or 'compas')."""
    try:
        return LOADERS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(LOADERS)}") from exc
