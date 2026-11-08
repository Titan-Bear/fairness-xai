"""Stability page - placeholder for the teammate's module.

The page is already wired: it calls src.stability.run_stability(...) and
shows the returned DataFrame. Until the teammate implements that function it
raises NotImplementedError, and this page shows an explanatory message.
"""

from __future__ import annotations

import importlib

import streamlit as st
from utils import METHOD_LABELS, dataset_label, explain, model_label, setup_page, table_with_download

from src import config

setup_page("Stability")
st.title("🧪 Explanation stability")
st.info(
    "This module is maintained by the **teammate** (Stability module). It will measure how much SHAP and LIME "
    "explanations change when the input is slightly perturbed - a stable explanation should not change much for "
    "a nearly identical person."
)

dataset = st.sidebar.selectbox("Dataset", config.DATASETS, format_func=dataset_label)
model = st.sidebar.selectbox("Model", config.MODEL_NAMES, format_func=model_label)
method = st.sidebar.radio("XAI method", config.XAI_METHODS, format_func=METHOD_LABELS.get, horizontal=True)
levels = st.sidebar.multiselect("Perturbation levels (noise as a fraction of each feature's std)",
                                [0.01, 0.05, 0.10, 0.20], default=[0.01, 0.05, 0.10])

st.markdown(
    f"Selected: **{dataset_label(dataset)}** · **{model_label(model)}** · **{METHOD_LABELS[method]}** · "
    f"perturbation levels {', '.join(f'{p:.0%}' for p in sorted(levels)) or '(none)'}"
)

if st.button("Run stability analysis", disabled=not levels):
    try:
        stability = importlib.import_module("src.stability")
        with st.spinner("Running stability analysis..."):
            result = stability.run_stability(dataset, model, method, perturbation_levels=tuple(sorted(levels)))
    except ModuleNotFoundError:
        st.warning("`src/stability.py` does not exist yet.")
    except NotImplementedError as exc:
        st.warning(f"Not implemented yet: {exc}. The page will show the results as soon as "
                   "`run_stability` is implemented.")
    else:
        table_with_download(result, f"stability_{dataset}_{model}_{method}.csv")
        explain("Output of src.stability.run_stability for the selected dataset, model and method.")
