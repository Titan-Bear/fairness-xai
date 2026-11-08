"""Data page: dataset size, class balance and group distribution by sensitive attribute."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (PRIV_COLOR, UNPRIV_COLOR, style_figure, dataset_label, download_csv, explain, load_data,
                   setup_page)

from src import config

setup_page("Data")
st.title("📊 Data")
st.markdown("Explore the two datasets before modelling, with a focus on the demographic groups.")

dataset = st.sidebar.selectbox("Dataset", config.DATASETS, format_func=dataset_label)
data = load_data(dataset)
meta = data.metadata

# ---- Headline numbers -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (after cleaning)", f"{len(data.X):,}")
c2.metric("Features", data.X.shape[1])
c3.metric("Positive rate", f"{data.y.mean():.1%}")
c4.metric("Sensitive attributes", ", ".join(meta.sensitive_attributes))
explain(f"Positive class (y = 1) means: **{meta.positive_label}**. "
        + ("Cleaning: fnlwgt and education dropped, rows with '?' removed, race binarised to White / Non-White."
           if dataset == "adult" else
           "Cleaning: ProPublica's filters, African-American and Caucasian defendants only; "
           "COMPAS's own scores (decile_score, score_text, v_decile_score) excluded to avoid target leakage."))

# ---- Class balance ------------------------------------------------------------
st.subheader("Class balance")
counts = data.y.value_counts().sort_index()
fig = go.Figure(go.Bar(x=["0 (negative)", "1 (positive)"], y=counts.to_numpy(), marker_color="#7f7f7f",
                       text=[f"{c:,} ({c / counts.sum():.1%})" for c in counts], textposition="outside",
                       hovertemplate="%{x}: %{y:,}<extra></extra>"))
fig.update_layout(title="Number of rows per class", yaxis_title="rows")
st.plotly_chart(style_figure(fig, height=330), use_container_width=True)
explain("An imbalanced target means accuracy alone is misleading - the Models page also reports F1 and ROC-AUC.")

# ---- Groups ----------------------------------------------------------------------
st.subheader("Groups by sensitive attribute")
rows = []
columns = st.columns(len(meta.sensitive_attributes))
for col, attribute in zip(columns, meta.sensitive_attributes):
    priv = meta.privileged[attribute]
    groups = [priv] + [g for g in sorted(data.sensitive[attribute].unique()) if g != priv]
    sizes, base_rates = [], []
    for group in groups:
        mask = data.sensitive[attribute] == group
        sizes.append(int(mask.sum()))
        base_rates.append(float(data.y[mask].mean()))
        rows.append({"attribute": attribute, "group": group, "privileged": group == priv,
                     "count": int(mask.sum()), "share": mask.mean(), "base_rate": data.y[mask].mean()})
    colors = [PRIV_COLOR, UNPRIV_COLOR]
    with col:
        fig = go.Figure(go.Bar(x=groups, y=sizes, marker_color=colors,
                               text=[f"{s:,}" for s in sizes], textposition="outside",
                               hovertemplate="%{x}: %{y:,} rows<extra></extra>"))
        fig.update_layout(title=f"Group sizes - {attribute}", yaxis_title="rows")
        st.plotly_chart(style_figure(fig, height=320), use_container_width=True)
        fig = go.Figure(go.Bar(x=groups, y=base_rates, marker_color=colors,
                               text=[f"{r:.1%}" for r in base_rates], textposition="outside",
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.update_layout(title=f"Base rate (share with y = 1) - {attribute}", yaxis_title="share",
                          yaxis_range=[0, max(base_rates) * 1.25])
        st.plotly_chart(style_figure(fig, height=320), use_container_width=True)

summary = pd.DataFrame(rows)
st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
download_csv(summary, f"group_summary_{dataset}.csv")
explain(
    "Blue = privileged group, orange = unprivileged group. The base rate is the share of real positive outcomes "
    "in the data. If base rates differ, even a perfectly accurate model has a demographic-parity gap - "
    "that is why the Fairness page also reports equal opportunity."
)

# ---- Features ------------------------------------------------------------------------
st.subheader("Features")
features = pd.DataFrame({
    "feature": meta.numeric + meta.categorical,
    "type": ["numeric"] * len(meta.numeric) + ["categorical"] * len(meta.categorical),
    "sensitive": [f in meta.sensitive_attributes for f in meta.numeric + meta.categorical],
    "distinct values": [data.X[f].nunique() for f in meta.numeric + meta.categorical],
})
st.dataframe(features, use_container_width=True, hide_index=True)
explain("Numeric features are standardised and categorical features one-hot encoded inside each model pipeline.")

with st.expander("Preview the first 50 rows"):
    preview = data.X.head(50).assign(**{meta.target_name: data.y.head(50)})
    st.dataframe(preview, use_container_width=True)
    download_csv(preview, f"preview_{dataset}.csv", "Download preview")
