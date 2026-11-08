"""Models page: test-set performance, ROC curves and confusion matrices."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from utils import (SETTING_LABELS, confusion_heatmap, dataset_label, explain, model_label,
                   models_available, require_table, roc_points, setup_page, style_figure,
                   table_with_download)

from src import config

setup_page("Models")
st.title("🤖 Models")
st.markdown(
    "Three models - **Logistic Regression**, **Random Forest** and **XGBoost** - each in a scikit-learn "
    "Pipeline with the same preprocessing, evaluated on the same 30% stratified test set."
)

dataset = st.sidebar.selectbox("Dataset", config.DATASETS, format_func=dataset_label)
setting = st.sidebar.radio("Setting", list(config.SETTINGS), format_func=SETTING_LABELS.get)

performance = require_table("performance.csv")
part = performance[performance["dataset"] == dataset].copy()
part["model"] = part["model"].map(model_label)

# ---- Performance table ------------------------------------------------------------
st.subheader(f"Performance - {dataset_label(dataset)}")
best = part[part["setting"] == setting].sort_values("accuracy", ascending=False).iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Best accuracy", f"{best['accuracy']:.3f}", help=f"{best['model']}, {SETTING_LABELS[setting]}")
c2.metric("Its F1", f"{best['f1']:.3f}")
c3.metric("Its ROC-AUC", f"{best['roc_auc']:.3f}")
table_with_download(part[["setting", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]],
                    f"performance_{dataset}.csv")
explain(
    "Accuracy = share of correct predictions. Precision = of those predicted positive, how many are. "
    "Recall = of the real positives, how many were found. F1 = balance of precision and recall. "
    "ROC-AUC = how well the predicted probabilities rank positives above negatives (0.5 = random). "
    "Compare the two settings: a small difference means the sensitive features add little predictive information."
)

# ---- ROC curves ---------------------------------------------------------------------
st.subheader(f"ROC curves - {SETTING_LABELS[setting]}")
if models_available(dataset):
    points = roc_points(dataset, setting)
    fig = go.Figure()
    for model, curve in points.groupby("model", sort=False):
        fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(width=2),
                        name=f"{model} (AUC {curve['auc'].iloc[0]:.3f})",
                        hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra>%{fullData.name}</extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey", width=1),
                    name="Random (AUC 0.5)", hoverinfo="skip")
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
    st.plotly_chart(style_figure(fig, height=450), use_container_width=True)
    explain("Each curve shows the trade-off between catching positives (up) and false alarms (right) as the "
            "decision threshold moves. Curves closer to the top-left corner are better.")
else:
    st.info("Saved models not found - run `python -m src.run_all` to draw ROC curves.")

# ---- Confusion matrices ---------------------------------------------------------------
st.subheader(f"Confusion matrices - {SETTING_LABELS[setting]}")
cols = st.columns(len(config.MODEL_NAMES))
for col, (_, row) in zip(cols, part[part["setting"] == setting].iterrows()):
    with col:
        st.plotly_chart(confusion_heatmap(int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"]),
                                          row["model"]), use_container_width=True)
explain("Rows = actual class, columns = predicted class (threshold 0.5). The diagonal holds correct predictions; "
        "bottom-left = missed positives (false negatives), top-right = false alarms (false positives).")
