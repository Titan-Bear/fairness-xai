"""Stability page (teammate's module): how much do explanations change under small input noise?

Precomputed results come from `python -m src.run_all` (results/tables/stability_*.csv).
The "Run live" tab calls src.stability.run_stability on a small sample.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (METHOD_LABELS, dataset_label, download_csv, explain, load_data, load_split, load_table, model_label,
                   models_available, setup_page, show_missing_as_na, style_figure, table_with_download)

from src import config
from src.stability import MODEL_COLORS, STABILITY_SETTING, StabilitySettings, run_stability

setup_page("Stability")
st.title("🧪 Explanation stability")
st.markdown(
    "*Module of the teammate (Stability), integrated into the shared pipeline.* "
    "A trustworthy explanation should barely change when a person's numeric features change very slightly. "
    "Noise of 1%, 5% and 10% of each feature's standard deviation is added to numeric features; the original and "
    "perturbed explanations are compared with **Spearman rho** (1 = same feature ranking) and **top-5 overlap**. "
    "**0%** re-explains the same input: it measures the explainer's own randomness."
)

dataset = st.sidebar.selectbox("Dataset", config.DATASETS, format_func=dataset_label)
summary = load_table("stability_summary.csv")


def select(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    """Rows of `frame` matching every column=value condition."""
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    return frame[mask]


tab_noise, tab_tests, tab_groups, tab_trade, tab_live = st.tabs(
    ["Stability vs noise", "Statistical tests", "By demographic group", "Accuracy vs stability", "Run live"])

if summary is None:
    for tab in (tab_noise, tab_tests, tab_groups, tab_trade):
        with tab:
            st.info("No stability results yet. Run `python -m src.run_all` (with `RUN_STABILITY = True`).")
else:
    part = select(summary, dataset=dataset)
    # ---- Stability vs noise -------------------------------------------------------
    with tab_noise:
        st.subheader(f"{dataset_label(dataset)}: stability against input noise")
        for metric, title in (("rho", "Spearman rho (original vs perturbed)"), ("top5", "Top-5 feature overlap")):
            fig = go.Figure()
            for (model, method), group in part.groupby(["model", "method"]):
                group = group.sort_values("noise")
                error = (dict(type="data", symmetric=False, array=group["rho_ci_high"] - group["rho"],
                              arrayminus=group["rho"] - group["rho_ci_low"]) if metric == "rho" else None)
                fig.add_scatter(x=group["noise"] * 100, y=group[metric], error_y=error, mode="lines+markers",
                                name=f"{model_label(model)} + {METHOD_LABELS[method]}",
                                line=dict(color=MODEL_COLORS[model], dash="solid" if method == "shap" else "dash",
                                          width=2),
                                hovertemplate="noise %{x}%: %{y:.3f}<extra>%{fullData.name}</extra>")
            fig.update_layout(title=title, xaxis_title="Input noise (% of feature std)", yaxis_title=metric)
            fig.update_xaxes(tickvals=sorted(part["noise"].unique() * 100))
            st.plotly_chart(style_figure(fig, height=420), use_container_width=True)
        explain("Solid = SHAP, dashed = LIME; colour = model. Error bars on rho are bootstrap 95% CIs of the mean. "
                "Tree models split exactly at observed integer values, so even 1% noise can cross a split - "
                "the direction of the noise matters more than its size.")
        shown = part.copy()
        shown["model"] = shown["model"].map(model_label)
        shown["method"] = shown["method"].map(METHOD_LABELS)
        table_with_download(shown, f"stability_summary_{dataset}.csv")

    # ---- Statistical tests ------------------------------------------------------------
    with tab_tests:
        st.subheader("Hypothesis tests (Holm-corrected within each family)")
        tests_a = load_table("stability_test_shap_vs_lime.csv")
        st.markdown("**S-H1: SHAP vs LIME** - H0: both explainers are equally stable on the same instances "
                    "(Wilcoxon signed-rank, paired).")
        if tests_a is not None and not tests_a.empty:
            table_with_download(select(tests_a, dataset=dataset), f"stability_shap_vs_lime_{dataset}.csv", round_to=4)
        tests_b = load_table("stability_test_models.csv")
        st.markdown("**S-H2: across models** - H0: LR, RF and XGBoost are equally stable "
                    "(Friedman test; Kendall's W = effect size from 0 to 1).")
        if tests_b is not None and not tests_b.empty:
            table_with_download(select(tests_b, dataset=dataset), f"stability_models_{dataset}.csv", round_to=4)
        posthoc = load_table("stability_posthoc.csv")
        if posthoc is not None and not posthoc.empty:
            with st.expander("Pairwise post-hoc tests (Wilcoxon)"):
                table_with_download(select(posthoc, dataset=dataset), f"stability_posthoc_{dataset}.csv", round_to=4)
        explain(f"reject_h0 = True when the Holm-adjusted p < {config.ALPHA}. Tests use noise levels above 0%.")

    # ---- By group (link to the Fairness module) ------------------------------------------
    with tab_groups:
        st.subheader("Are explanations equally stable for every demographic group?")
        by_group = load_table("stability_by_group.csv")
        if by_group is None or by_group.empty:
            st.info("No group results available.")
        else:
            groups_part = select(by_group, dataset=dataset)
            attribute = st.radio("Sensitive attribute", list(dict.fromkeys(groups_part["attribute"])),
                                 horizontal=True)
            rows = select(groups_part, attribute=attribute)
            labels = [f"{model_label(m)} + {METHOD_LABELS[e]}" for m, e in zip(rows["model"], rows["method"])]
            fig = go.Figure()
            for column, role in (("rho_privileged", "privileged"), ("rho_unprivileged", "unprivileged")):
                group_name = rows["privileged"].iloc[0] if role == "privileged" else rows["unprivileged"].iloc[0]
                fig.add_bar(x=labels, y=rows[column], name=f"{group_name} ({role})",
                            marker_color=config.GROUP_COLORS[role],
                            text=[f"{v:.3f}" for v in rows[column]], textposition="outside")
            fig.update_layout(barmode="group", yaxis_title="mean Spearman rho (noise > 0)", yaxis_range=[0, 1.1])
            st.plotly_chart(style_figure(fig, height=400), use_container_width=True)
            table_with_download(rows, f"stability_by_group_{dataset}_{attribute}.csv", round_to=4)
            explain("S-H3: H0 - stability is the same for both groups (Mann-Whitney U, one value per instance and "
                    "seed, Holm-corrected). Sample sizes per group follow the random sample, so small groups have "
                    "little power.")

    # ---- Accuracy vs stability ---------------------------------------------------------------
    with tab_trade:
        st.subheader("Accuracy vs stability (Pareto)")
        trade = load_table("stability_tradeoff.csv")
        if trade is None:
            st.info("No trade-off table available.")
        else:
            rows = select(trade, dataset=dataset).copy()
            rows.insert(0, "Pareto-optimal", rows.pop("is_pareto").map({True: "✓ yes", False: ""}))
            rows["model"] = rows["model"].map(model_label)
            rows["method"] = rows["method"].map(METHOD_LABELS)
            shown = show_missing_as_na(rows.drop(columns=["dataset"]).round(4))
            st.dataframe(shown.style.apply(
                lambda r: ["background-color: #2a78d6; color: white" if r["Pareto-optimal"] else "" for _ in r],
                axis=1), use_container_width=True, hide_index=True)
            download_csv(rows, f"stability_tradeoff_{dataset}.csv")
            explain("Higher ROC-AUC, stability rho and top-5 overlap are better; a lower prediction-flip rate is "
                    "better. Pareto-optimal rows are not beaten on every column by another row. (The original "
                    "notebook's weighted 'balance score' was replaced by this Pareto view.)")

# ---- Run live ------------------------------------------------------------------------------------
with tab_live:
    st.subheader("Run the stability analysis live (small sample)")
    model = st.selectbox("Model", config.MODEL_NAMES, format_func=model_label)
    method = st.radio("XAI method", config.XAI_METHODS, format_func=METHOD_LABELS.get, horizontal=True)
    levels = st.multiselect("Noise levels (fraction of each feature's std)", [0.01, 0.05, 0.10, 0.20],
                            default=list(config.STABILITY_NOISE_LEVELS))
    n = st.slider("Instances", 5, 30, 10, step=5)
    if not models_available(dataset):
        st.info("Saved models are needed - run `python -m src.run_all` first.")
    elif st.button("Run stability analysis", disabled=not levels):
        with st.spinner("Explaining original and perturbed instances..."):
            data, split = load_data(dataset), load_split(dataset)
            live = run_stability(dataset, model, method, tuple(sorted(levels)),
                                 StabilitySettings(n_instances=n, seeds=(config.SEED,), lime_samples=300),
                                 split=split, metadata=data.metadata)
        result = live.groupby("noise")[["rho", "top3", "top5", "pred_flip", "dprob"]].mean().reset_index()
        result.insert(0, "noise level", (result.pop("noise") * 100).map(lambda v: f"{v:g}%"))
        table_with_download(result, f"stability_live_{dataset}_{model}_{method}.csv")
        explain(f"src.stability.run_stability on {n} test instances (seed {config.SEED}, model trained "
                f"{STABILITY_SETTING.replace('_', ' ')} features, 300 LIME samples). Small samples are noisy.")
