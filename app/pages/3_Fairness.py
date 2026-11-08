"""Fairness page (core module): fairness metrics, explanations per group,
explanation comparison with statistical tests, trade-off and mitigation.

Every number comes from results/tables/ (written by `python -m src.run_all`),
except the optional live recomputation in the Comparison tab.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import (METHOD_LABELS, SETTING_LABELS, dataset_label, download_csv, explain, group_bars, group_box,
                   live_comparison, load_table, model_label, models_available, require_table, setup_page,
                   show_missing_as_na,
                   style_figure, table_with_download, top_features_bar)

from src import config
from src.compare import interpret_comparison

setup_page("Fairness")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
fairness = require_table("fairness.csv")
by_group = require_table("fairness_by_group.csv")
comparison = require_table("explanation_comparison.csv")
per_instance = require_table("per_instance_reliance.csv")
importance = require_table("global_importance.csv")
tests = require_table("stat_tests.csv")
tradeoff = require_table("tradeoff.csv")


def select(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    """Rows of `frame` matching every column=value condition."""
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column] == value
    return frame[mask]


# --------------------------------------------------------------------------
# Sidebar selectors
# --------------------------------------------------------------------------
st.sidebar.header("Audit selection")
dataset = st.sidebar.selectbox("Dataset", config.DATASETS, format_func=dataset_label)
model = st.sidebar.selectbox("Model", config.MODEL_NAMES, format_func=model_label)
attributes = list(dict.fromkeys(select(fairness, dataset=dataset)["attribute"]))  # primary first
attribute = st.sidebar.selectbox("Sensitive attribute", attributes,
                                 help="The first option is the dataset's primary sensitive attribute.")
method = st.sidebar.radio("XAI method", config.XAI_METHODS, format_func=METHOD_LABELS.get, horizontal=True)
setting = st.sidebar.radio("Setting", list(config.SETTINGS), format_func=SETTING_LABELS.get)
st.sidebar.caption("Blue = privileged group, orange = unprivileged group, on every chart.")

keys = dict(dataset=dataset, setting=setting, model=model, attribute=attribute)
fair_row = select(fairness, **keys).iloc[0]
comp_row = select(comparison, **keys, method=method).iloc[0].to_dict()
priv, unpriv = fair_row["privileged"], fair_row["unprivileged"]
proxies = comp_row["proxies"].split(";")

st.title("⚖️ Fairness audit")
st.markdown(
    f"**{dataset_label(dataset)}** · **{model_label(model)}** · sensitive attribute **{attribute}** "
    f"(privileged: {priv}) · **{METHOD_LABELS[method]}** · {SETTING_LABELS[setting]}"
)

tab_metrics, tab_expl, tab_comp, tab_trade, tab_mitig = st.tabs(
    ["Fairness metrics", "Explanations", "Comparison", "Trade-off", "Mitigation"])

# --------------------------------------------------------------------------
# Tab 1: fairness metrics (RQ1)
# --------------------------------------------------------------------------
with tab_metrics:
    st.subheader("How unfair are the predictions? (RQ1)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demographic parity difference", f"{fair_row['dp_difference']:.3f}",
              help="|selection rate A - selection rate B|. 0 = both groups predicted positive equally often.")
    c2.metric("Demographic parity ratio", f"{fair_row['dp_ratio']:.3f}",
              help="Smaller selection rate / larger. 1 = equal; below 0.8 fails the four-fifths rule.")
    c3.metric("Equal opportunity difference", f"{fair_row['eo_difference']:.3f}",
              help="|TPR A - TPR B|: among people who really are positive, the gap in how often they are found.")
    c4.metric("Equalized odds difference", f"{fair_row['eodds_difference']:.3f}",
              help="The larger of the TPR gap and the FPR gap.")
    st.info(f"{fair_row['interpretation']}  \n{fair_row['interpretation_eo']}")
    if fair_row["dp_ratio"] < 0.8:
        st.warning(f"DP ratio {fair_row['dp_ratio']:.2f} is below 0.8: this model fails the four-fifths rule "
                   f"for {attribute}.")

    groups = select(by_group, **keys).set_index("group")
    metrics = {"Base rate (data)": "base_rate", "Selection rate": "selection_rate",
               "True positive rate": "true_positive_rate", "False positive rate": "false_positive_rate",
               "Accuracy": "accuracy"}
    st.plotly_chart(group_bars(list(metrics), [groups.loc[priv, m] for m in metrics.values()],
                               [groups.loc[unpriv, m] for m in metrics.values()], priv, unpriv,
                               f"{model_label(model)}: metrics by {attribute}"), use_container_width=True)
    gap_data = fair_row["base_rate_difference"]
    effect = ("amplifies" if fair_row["dp_difference"] > gap_data + 0.01
              else "narrows" if fair_row["dp_difference"] < gap_data - 0.01 else "copies")
    explain(
        f"Base rate = share of real positives in the test data; selection rate = share predicted positive. "
        f"The real gap between the groups is {gap_data:.3f}, the model's gap is {fair_row['dp_difference']:.3f}, "
        f"so the model **{effect}** the disparity already in the data."
    )

    st.markdown("**All models for this attribute and setting**")
    models_part = select(fairness, dataset=dataset, setting=setting, attribute=attribute)
    fig = go.Figure()
    for column, name, color in (("dp_difference", "Demographic parity difference", "#2a78d6"),
                                ("eo_difference", "Equal opportunity difference", "#86b6ef")):
        fig.add_bar(x=models_part["model"].map(model_label), y=models_part[column], name=name, marker_color=color,
                    text=[f"{v:.2f}" for v in models_part[column]], textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="difference (0 = no gap)")
    st.plotly_chart(style_figure(fig, height=340), use_container_width=True)
    explain("Lower is fairer. Compare with the other setting in the sidebar to see whether removing the "
            "sensitive features helps.")
    table_with_download(
        select(fairness, dataset=dataset, attribute=attribute).drop(columns=["interpretation", "interpretation_eo"]),
        f"fairness_{dataset}_{attribute}.csv")

# --------------------------------------------------------------------------
# Tab 2: explanations per group
# --------------------------------------------------------------------------
with tab_expl:
    st.subheader(f"What drives the predictions for each group? ({METHOD_LABELS[method]})")
    imp = select(importance, **keys, method=method)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(top_features_bar(imp, priv, "privileged", attribute, proxies, config.TOP_K_PLOT),
                        use_container_width=True)
    with right:
        st.plotly_chart(top_features_bar(imp, unpriv, "unprivileged", attribute, proxies, config.TOP_K_PLOT),
                        use_container_width=True)
    explain(
        f"Mean absolute {METHOD_LABELS[method]} attribution per original feature within each group "
        f"({comp_row['n_privileged']} {priv} and {comp_row['n_unprivileged']} {unpriv} test instances). "
        f"One-hot columns are summed back to their feature. '(proxy)' marks the features most associated with "
        f"{attribute}: {', '.join(proxies)}. Spearman rho between the two rankings: {comp_row['spearman_rho']:.2f}."
    )
    wide = imp.pivot_table(index="feature", columns="group", values=["mean_abs_attribution", "rank"])
    ranks = pd.DataFrame({
        "feature": wide.index,
        f"importance ({priv})": wide[("mean_abs_attribution", priv)].to_numpy(),
        f"rank ({priv})": wide[("rank", priv)].astype(int).to_numpy(),
        f"importance ({unpriv})": wide[("mean_abs_attribution", unpriv)].to_numpy(),
        f"rank ({unpriv})": wide[("rank", unpriv)].astype(int).to_numpy(),
    }).sort_values(f"rank ({priv})")
    ranks["rank change"] = ranks[f"rank ({unpriv})"] - ranks[f"rank ({priv})"]
    table_with_download(ranks, f"importance_{dataset}_{setting}_{model}_{method}_{attribute}.csv", round_to=4)
    explain("Rank change > 0: the feature matters less (lower rank) for the unprivileged group.")

# --------------------------------------------------------------------------
# Tab 3: comparison + statistical tests (RQ2, RQ3)
# --------------------------------------------------------------------------
with tab_comp:
    st.subheader("Are the explanations equally similar and reliable across groups? (RQ2, RQ3)")
    has_sensitive = pd.notna(comp_row["sensitive_reliance_privileged"])
    c1, c2, c3, c4 = st.columns(4)
    if has_sensitive:
        c1.metric(f"Reliance on {attribute} - {priv}", f"{comp_row['sensitive_reliance_privileged']:.1%}")
        c2.metric(f"Reliance on {attribute} - {unpriv}", f"{comp_row['sensitive_reliance_unprivileged']:.1%}",
                  delta=f"{-comp_row['sensitive_reliance_gap']:+.1%} vs {priv}", delta_color="off")
    else:
        c1.metric(f"Reliance on {attribute}", "n/a")
        c2.caption(f"'{attribute}' is not a model input in this setting, so its reliance is not applicable.")
    c3.metric(f"Proxy reliance - {priv}", f"{comp_row['proxy_reliance_privileged']:.1%}")
    c4.metric(f"Proxy reliance - {unpriv}", f"{comp_row['proxy_reliance_unprivileged']:.1%}",
              delta=f"{-comp_row['proxy_reliance_gap']:+.1%} vs {priv}", delta_color="off")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Spearman rho between groups", f"{comp_row['spearman_rho']:.2f}",
              help=f"95% bootstrap CI {comp_row['spearman_ci_low']:.2f} to {comp_row['spearman_ci_high']:.2f}")
    d1.caption(f"95% CI [{comp_row['spearman_ci_low']:.2f}, {comp_row['spearman_ci_high']:.2f}]")
    d2.metric("Kendall tau", f"{comp_row['kendall_tau']:.2f}")
    d3.metric("Top-5 Jaccard overlap", f"{comp_row['jaccard_top5']:.2f}")
    if method == "lime":
        d4.metric("LIME fidelity gap", f"{comp_row['fidelity_gap']:+.3f}",
                  help=f"Mean local R-squared: {priv} {comp_row['fidelity_privileged']:.3f}, "
                       f"{unpriv} {comp_row['fidelity_unprivileged']:.3f}")
    else:
        d4.metric("Fidelity gap", "n/a", help="SHAP (Tree/Linear) is exact, so it has no surrogate fidelity.")

    run_tests = select(tests, **keys, method=method)
    st.success(interpret_comparison(comp_row, run_tests))

    inst = select(per_instance, **keys, method=method)
    charts = []
    if has_sensitive:
        charts.append(("sensitive_reliance", f"Reliance on '{attribute}'", "share of attribution"))
    charts.append(("proxy_reliance", f"Reliance on proxies ({', '.join(proxies)})", "share of attribution"))
    if method == "lime":
        charts.append(("fidelity", "LIME fidelity (local R-squared)", "R-squared"))
    for col, (column, title, y_title) in zip(st.columns(len(charts)), charts):
        with col:
            st.plotly_chart(group_box(inst, column, priv, unpriv, title, y_title), use_container_width=True)
    explain("Each box shows the middle 50% of instances, the line is the median and the dashed line the mean. "
            "Reliance = share of an instance's total absolute attribution given to the feature(s).")

    st.markdown("**Statistical tests (Mann-Whitney U, Holm-corrected across all tests of the run)**")
    st.markdown("H0: the measure has the same distribution in both groups. H1: the distributions differ.")
    show_tests = run_tests[["measure", "mean_privileged", "mean_unprivileged", "u_statistic", "p_value", "p_holm",
                            "rank_biserial", "significant_holm"]]
    table_with_download(show_tests, f"tests_{dataset}_{setting}_{model}_{method}_{attribute}.csv", round_to=4)
    explain(f"Significant if the Holm-adjusted p < {config.ALPHA}. Rank-biserial r is the effect size "
            f"(-1 to 1; > 0 means {priv} tends to have larger values). Fidelity is tested for LIME only; sensitive "
            f"reliance is not tested when the sensitive feature is not an input.")

    with st.expander("Recompute live on a small sample"):
        if not models_available(dataset):
            st.info("Saved models are needed - run `python -m src.run_all` first.")
        else:
            n = st.slider("Instances per group", 10, 50, 20, step=5,
                          help="SHAP uses this many per group, LIME half of it with 300 samples each.")
            if st.button("Recompute"):
                with st.spinner("Explaining and comparing on a small sample..."):
                    live_row, live_inst = live_comparison(dataset, setting, model, method, attribute, n)
                st.write(interpret_comparison(live_row))
                st.plotly_chart(group_box(live_inst, "proxy_reliance", priv, unpriv,
                                          "Proxy reliance (live sample)", "share of attribution"),
                                use_container_width=True)
                explain("Small samples are noisier than the precomputed results above; use them to check that the "
                        "pipeline works, not to draw conclusions.")

    st.markdown("**All comparisons for this dataset**")
    table_with_download(select(comparison, dataset=dataset), f"explanation_comparison_{dataset}.csv")

# --------------------------------------------------------------------------
# Tab 4: trade-off (RQ4)
# --------------------------------------------------------------------------
with tab_trade:
    st.subheader("Which model + XAI method balances accuracy, fairness and consistency? (RQ4)")
    part = select(tradeoff, dataset=dataset, setting=setting, attribute=attribute).copy()
    part.insert(0, "Pareto-optimal", part.pop("is_pareto").map({True: "✓ yes", False: ""}))
    part["model"] = part["model"].map(model_label)
    part["method"] = part["method"].map(METHOD_LABELS)
    shown = show_missing_as_na(part.drop(columns=["dataset", "setting", "attribute"]).round(3))
    styled = shown.style.apply(
        lambda r: ["background-color: #2a78d6; color: white" if r["Pareto-optimal"] else "" for _ in r], axis=1
    ).format(precision=3)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    download_csv(part, f"tradeoff_{dataset}_{setting}_{attribute}.csv")
    explain(
        "Better = higher accuracy, F1 and Spearman rho; lower DP difference, EO difference and |fidelity gap|. "
        "A row is Pareto-optimal (highlighted) if no other row is at least as good on every column and better on "
        "one. SHAP is exact, so its fidelity gap is shown as n/a and counted as 0. No weighted score is used: "
        "choosing among the Pareto rows depends on which criterion matters most."
    )
    fig = go.Figure()
    for method_name, symbol in (("SHAP", "circle"), ("LIME", "diamond")):
        sub = part[part["method"] == method_name]
        fig.add_scatter(x=sub["dp_difference"], y=sub["accuracy"], mode="markers+text", name=method_name,
                        text=sub["model"], textposition="top center", marker=dict(size=12, symbol=symbol),
                        customdata=sub[["spearman_rho"]],
                        hovertemplate="%{text}<br>DP diff %{x:.3f}<br>accuracy %{y:.3f}"
                                      "<br>rho %{customdata[0]:.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title="Demographic parity difference (lower = fairer)",
                      yaxis_title="Accuracy (higher = better)")
    st.plotly_chart(style_figure(fig, height=400), use_container_width=True)
    explain("The accuracy-fairness view of the table. SHAP and LIME rows of the same model share the same point "
            "on these two axes; they differ in explanation consistency (rho) and fidelity.")

# --------------------------------------------------------------------------
# Tab 5: mitigation
# --------------------------------------------------------------------------
with tab_mitig:
    st.subheader("Bias mitigation: before vs after")
    mitigation = load_table("mitigation.csv")
    if mitigation is None:
        st.info("No mitigation results yet. They are produced by `python -m src.run_all` when "
                "`RUN_MITIGATION = True` in src/config.py.")
    else:
        part = select(mitigation, dataset=dataset, setting=setting, attribute=attribute).set_index("stage")
        constrained = part["constrained_attribute"].iloc[0]
        st.markdown(
            f"**Logistic Regression** retrained with fairlearn's **ExponentiatedGradient** under a "
            f"**demographic-parity** constraint on **{constrained}**. Showing the effect on **{attribute}**"
            + (" (the constrained attribute)." if attribute == constrained else
               f" - not constrained, so this shows side effects.")
        )
        before, after = part.loc["before"], part.loc["after"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DP difference", f"{after['dp_difference']:.3f}",
                  delta=f"{after['dp_difference'] - before['dp_difference']:+.3f}", delta_color="inverse")
        c2.metric("EO difference", f"{after['eo_difference']:.3f}",
                  delta=f"{after['eo_difference'] - before['eo_difference']:+.3f}", delta_color="inverse")
        c3.metric("Accuracy", f"{after['accuracy']:.3f}",
                  delta=f"{after['accuracy'] - before['accuracy']:+.3f}")
        c4.metric("LIME rho between groups", f"{after['spearman_rho']:.2f}",
                  delta=f"{after['spearman_rho'] - before['spearman_rho']:+.2f}")
        explain("Values after mitigation; the small number shows the change from before "
                "(green = improvement).")
        metrics = {"Accuracy": "accuracy", "F1": "f1", "DP difference": "dp_difference",
                   "EO difference": "eo_difference", "LIME rho (groups)": "spearman_rho",
                   "LIME |fidelity gap|": "fidelity_gap"}
        fig = go.Figure()
        for stage, color in (("before", "#86b6ef"), ("after", "#2a78d6")):
            values = [abs(part.loc[stage, m]) if m == "fidelity_gap" else part.loc[stage, m]
                      for m in metrics.values()]
            fig.add_bar(x=list(metrics), y=values, name=stage, marker_color=color,
                        text=[f"{v:.3f}" for v in values], textposition="outside")
        fig.update_layout(barmode="group", yaxis_range=[0, 1.1])
        st.plotly_chart(style_figure(fig, height=380), use_container_width=True)
        explain(
            "The mitigated model is a randomised mixture of Logistic Regressions: accuracy, F1 and fairness use its "
            "(seeded) random predictions, for which fairlearn's guarantee holds. ROC-AUC and LIME use its expected "
            "probability. Demographic parity often trades off against equal opportunity - compare the two bars."
        )
        st.markdown("**Before vs after (all columns)**")
        table_with_download(select(mitigation, dataset=dataset, setting=setting).drop(columns=["interpretation"]),
                            f"mitigation_{dataset}_{setting}.csv")
        mit_tests = load_table("mitigation_tests.csv")
        if mit_tests is not None:
            st.markdown("**Group differences in LIME explanations, before and after (Mann-Whitney, Holm)**")
            table_with_download(
                select(mit_tests, dataset=dataset, setting=setting, attribute=attribute)[
                    ["model", "measure", "mean_privileged", "mean_unprivileged", "p_value", "p_holm",
                     "rank_biserial", "significant_holm"]],
                f"mitigation_tests_{dataset}_{setting}_{attribute}.csv", round_to=4)
