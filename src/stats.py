"""Statistical tests (RQ3): are the group differences in explanations real?

Hypotheses (for each model x method x attribute x setting and each measure):
  H0: the distribution of the measure (sensitive reliance, proxy reliance or
      LIME fidelity) is the same for the privileged and unprivileged group.
  H1: the distributions differ (two-sided).

Why Mann-Whitney U?
  * The measures are shares and R-squared values - bounded, skewed and often
    with many values near 0, so a t-test's normality assumption is doubtful.
  * Mann-Whitney compares ranks, needs no normality, and works with unequal
    group sizes.

Effect size: rank-biserial correlation r = 2U / (n1 * n2) - 1, in [-1, 1].
  r > 0 means the privileged group tends to have larger values, 0 means no
  tendency. A tiny p-value with |r| near 0 is a statistically detectable but
  practically small difference, so we always report both.

Multiple testing: one run performs many tests, so some p < 0.05 would occur
by chance. The Holm correction controls the family-wise error rate across ALL
tests in the run; we report raw and adjusted p-values.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src import config

logger = logging.getLogger(__name__)

STAT_TESTS_FILE = "stat_tests.csv"
MEASURES: tuple[str, ...] = ("sensitive_reliance", "proxy_reliance", "fidelity")
KEYS: list[str] = ["dataset", "setting", "model", "method", "attribute"]


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from the U statistic of the first sample.

    U / (n1 * n2) is the probability that a random value from sample 1 beats
    a random value from sample 2 (ties count half); rescaling it from [0, 1]
    to [-1, 1] gives r.
    """
    return 2.0 * u / (n1 * n2) - 1.0


def mann_whitney(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> dict[str, float]:
    """Two-sided Mann-Whitney U test between x and y, NaNs dropped.

    Returns U (for x), p_value, rank_biserial, n1 and n2. If either sample has
    fewer than 2 values the test is undefined and NaNs are returned. If all
    values are identical there is no difference at all: p = 1, r = 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"u_statistic": np.nan, "p_value": np.nan, "rank_biserial": np.nan, "n1": n1, "n2": n2}
    if np.ptp(np.concatenate([x, y])) == 0:
        return {"u_statistic": n1 * n2 / 2, "p_value": 1.0, "rank_biserial": 0.0, "n1": n1, "n2": n2}
    result = mannwhitneyu(x, y, alternative="two-sided")
    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial": rank_biserial(result.statistic, n1, n2),
        "n1": n1,
        "n2": n2,
    }


def holm_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are kept and not counted).

    Sort the m p-values ascending; the i-th smallest (i = 1..m) is multiplied
    by (m - i + 1). A running maximum keeps the adjusted values in the same
    order as the raw ones, and values are capped at 1. Rejecting where the
    adjusted p < alpha is the Holm procedure.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    if m == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def run_stat_tests(per_instance: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """Mann-Whitney tests for every comparison and measure, Holm-corrected together.

    Skipped (not applicable): sensitive reliance when the sensitive feature is
    not a model input, and fidelity for SHAP (SHAP has no surrogate model).
    """
    rows = []
    for key, part in per_instance.groupby(KEYS, sort=False):
        for measure in MEASURES:
            values = part[measure]
            if values.isna().all():
                continue
            priv = values[part["is_privileged"]]
            unpriv = values[~part["is_privileged"]]
            rows.append({
                **dict(zip(KEYS, key)),
                "measure": measure,
                "privileged": part.loc[part["is_privileged"], "group"].iloc[0],
                "unprivileged": part.loc[~part["is_privileged"], "group"].iloc[0],
                "mean_privileged": priv.mean(),
                "mean_unprivileged": unpriv.mean(),
                "median_privileged": priv.median(),
                "median_unprivileged": unpriv.median(),
                **mann_whitney(priv, unpriv),
            })
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm_adjust(tests["p_value"])
    tests["significant_raw"] = tests["p_value"] < alpha
    tests["significant_holm"] = tests["p_holm"] < alpha
    logger.info("%d tests: %d significant before and %d after Holm correction (alpha %.2f).",
                len(tests), int(tests["significant_raw"].sum()), int(tests["significant_holm"].sum()), alpha)
    return tests


def save_stat_tests(tests: pd.DataFrame) -> None:
    """Write results/tables/stat_tests.csv."""
    config.ensure_dirs()
    tests.to_csv(config.table_path(STAT_TESTS_FILE), index=False)
    logger.info("Wrote %s (%d rows).", STAT_TESTS_FILE, len(tests))
