"""Home page of the Fairness-Aware XAI Audit Dashboard.

Run from the project root with:  streamlit run app/Home.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
from utils import explain, setup_page

from src import config

setup_page("Home")

st.title("⚖️ Fairness-Aware XAI Audit Dashboard")
st.markdown(
    """
Machine-learning models that make decisions about people can be **black boxes** and can be **biased** against
groups such as women or Black defendants. Explanation methods (**SHAP**, **LIME**) open the black box, and
**fairness metrics** measure the bias - but they are usually used separately.

This dashboard combines them and asks one further question:
**are the explanations themselves equally reliable and similar across demographic groups?**
"""
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Research questions")
    st.markdown(
        """
1. **RQ1** - How unfair are Logistic Regression, Random Forest and XGBoost on Adult Income and COMPAS,
   by demographic parity and equal opportunity?
2. **RQ2** - Do SHAP and LIME explanations differ between demographic groups
   (feature reliance, feature ranking, explanation fidelity)?
3. **RQ3** - Are those differences statistically significant?
4. **RQ4** - Which model + XAI combination gives the best balance of accuracy, fairness and
   explanation consistency across groups?
"""
    )
with col_b:
    st.subheader("Datasets and setup")
    st.markdown(
        """
| | Adult Income | COMPAS |
|---|---|---|
| Task | income > 50K | re-offending within 2 years |
| Primary sensitive attribute | sex (privileged: Male) | race (privileged: Caucasian) |
| Secondary attribute | race (White vs Non-White) | sex (privileged: Female) |

Every model is trained in two **settings**: *with* the sensitive features as inputs, and *without* them
("fairness through unawareness"), to test whether other features act as **proxies**.
"""
    )

st.subheader("How to navigate")
st.markdown(
    """
| Page | What you will find |
|---|---|
| **Data** | dataset size, class balance, group sizes and base rates |
| **Models** | accuracy, F1, ROC-AUC, ROC curves and confusion matrices |
| **Fairness** | the core of the audit - fairness metrics, explanations per group, comparison and statistical tests, trade-off, mitigation |
| **Stability** | explanation stability under small input noise - SHAP vs LIME, across models and groups (teammate's module) |

Use the **sidebar** to switch pages. On the Fairness page, the sidebar selectors choose the dataset, model,
sensitive attribute, explanation method and setting.
"""
)

st.subheader("Results status")
FILES = {
    "performance.csv": "Model performance (Models page)",
    "fairness.csv": "Fairness metrics (RQ1)",
    "fairness_by_group.csv": "Per-group rates (RQ1)",
    "explanation_comparison.csv": "Explanation comparison (RQ2)",
    "per_instance_reliance.csv": "Per-instance reliance and fidelity (RQ2)",
    "global_importance.csv": "Global importance per group (RQ2)",
    "proxies.csv": "Proxy features by mutual information",
    "stat_tests.csv": "Mann-Whitney tests with Holm correction (RQ3)",
    "tradeoff.csv": "Trade-off and Pareto table (RQ4)",
    "mitigation.csv": "Bias mitigation before vs after (optional)",
    "stability_summary.csv": "Explanation stability under input noise (Stability module)",
    "stability_by_group.csv": "Stability per demographic group (Stability module)",
}
rows = []
for filename, purpose in FILES.items():
    path = config.table_path(filename)
    exists = path.exists()
    rows.append({
        "table": filename,
        "purpose": purpose,
        "status": "✅ available" if exists else "❌ missing",
        "rows": len(pd.read_csv(path)) if exists else None,
        "last updated": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exists else "",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
explain(
    "All numbers in this dashboard are read from results/tables/, produced by `python -m src.run_all`. "
    "Re-run that command after changing src/config.py; the app picks up the new files after a page refresh "
    "(use the ⋮ menu → Clear cache if a page still shows old numbers)."
)
