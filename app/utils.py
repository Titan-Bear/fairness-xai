"""Shared helpers for the Streamlit pages: paths, cached loading, downloads, charts.

Caching strategy (why the app is fast):
  * st.cache_data     - for DataFrames (results tables, datasets, ROC points).
                        Streamlit stores a copy and returns it on reruns.
  * st.cache_resource - for fitted models: one shared object, never copied.
All heavy work (training, SHAP, LIME, bootstrap) is done by
`python -m src.run_all`; the app only reads results/tables/. A small live
recomputation is offered on the Fairness page, with a spinner.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts app/ on sys.path; the project root is needed for `import src`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src import config  # noqa: E402
from src.data import LoadedDataset, load_dataset  # noqa: E402
from src.models import load_model  # noqa: E402
from src.preprocess import SplitData, split_data  # noqa: E402

PRIV_COLOR = config.GROUP_COLORS["privileged"]
UNPRIV_COLOR = config.GROUP_COLORS["unprivileged"]

SETTING_LABELS = {
    "with_sensitive": "With sensitive features",
    "without_sensitive": "Without sensitive features (unaware)",
}
METHOD_LABELS = {"shap": "SHAP", "lime": "LIME"}


# --------------------------------------------------------------------------
# Page setup and labels
# --------------------------------------------------------------------------
def setup_page(title: str, icon: str = "⚖️") -> None:
    """Common page configuration; must be the first Streamlit call on a page."""
    st.set_page_config(page_title=f"{title} | Fairness-Aware XAI Audit", page_icon=icon, layout="wide")


def dataset_label(name: str) -> str:
    """Readable dataset name."""
    return config.DATASET_DISPLAY_NAMES.get(name, name)


def model_label(name: str) -> str:
    """Readable model name."""
    return config.MODEL_DISPLAY_NAMES.get(name, name)


def explain(text: str) -> None:
    """Short explanation under a chart or table, in a consistent style."""
    st.caption(text)


# --------------------------------------------------------------------------
# Cached loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_table(filename: str) -> pd.DataFrame | None:
    """Read results/tables/<filename>, or None if it has not been produced yet."""
    path = config.table_path(filename)
    return pd.read_csv(path) if path.exists() else None


def require_table(filename: str) -> pd.DataFrame:
    """Like load_table, but stop the page with instructions if the file is missing."""
    table = load_table(filename)
    if table is None:
        st.warning(
            f"`results/tables/{filename}` was not found. Run the pipeline first:\n\n"
            "```\npython -m src.run_all\n```"
        )
        st.stop()
    return table


@st.cache_data(show_spinner="Loading dataset...")
def load_data(dataset: str) -> LoadedDataset:
    """Load and clean a dataset (uses the copy cached in data/raw/)."""
    return load_dataset(dataset)


@st.cache_data(show_spinner=False)
def load_split(dataset: str) -> SplitData:
    """The same stratified, seeded train/test split as the pipeline."""
    data = load_data(dataset)
    return split_data(data.X, data.y, data.sensitive)


@st.cache_resource(show_spinner="Loading model...")
def get_model(dataset: str, setting: str, model_name: str):
    """A fitted Pipeline from models/ (shared, not copied, across reruns)."""
    return load_model(dataset, setting, model_name)


@st.cache_data(show_spinner="Computing ROC curves...")
def roc_points(dataset: str, setting: str) -> pd.DataFrame:
    """ROC curve points of all models for one dataset and setting."""
    split = load_split(dataset)
    frames = []
    for model_name in config.MODEL_NAMES:
        proba = get_model(dataset, setting, model_name).predict_proba(split.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(split.y_test, proba)
        auc = roc_auc_score(split.y_test, proba)
        frames.append(pd.DataFrame({"model": model_label(model_name), "fpr": fpr, "tpr": tpr, "auc": auc}))
    return pd.concat(frames, ignore_index=True)


def models_available(dataset: str) -> bool:
    """True if every model of a dataset has been trained and saved."""
    from src.models import model_path

    return all(model_path(dataset, s, m).exists() for s in config.SETTINGS for m in config.MODEL_NAMES)


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
def download_csv(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    """CSV download button; the file name doubles as the widget key."""
    st.download_button(
        label=f"⬇ {label}",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


def show_missing_as_na(frame: pd.DataFrame) -> pd.DataFrame:
    """Display copy where missing values read "n/a" instead of Streamlit's "None".

    Only columns that contain a missing value become text, so other numeric
    columns still sort as numbers. (Missing = not applicable here, e.g. SHAP
    has no fidelity, or the sensitive feature is not an input.)
    """
    shown = frame.copy()
    for column in shown.columns[shown.isna().any()]:
        shown[column] = shown[column].map(lambda v: "n/a" if pd.isna(v) else v).astype(str)
    return shown


def table_with_download(frame: pd.DataFrame, filename: str, round_to: int = 3, **kwargs) -> None:
    """Show a table (rounded, n/a for missing) and its CSV download button underneath."""
    st.dataframe(show_missing_as_na(frame.round(round_to)), use_container_width=True, hide_index=True, **kwargs)
    download_csv(frame, filename)


# --------------------------------------------------------------------------
# Charts (plotly, same colours as the PNG figures: privileged blue, unprivileged orange)
# --------------------------------------------------------------------------
def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Consistent chart layout: legend on top, compact margins, recessive grid."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.25,
        bargroupgap=0.06,
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def group_bars(
    categories: list[str],
    priv_values: list[float],
    unpriv_values: list[float],
    priv_name: str,
    unpriv_name: str,
    title: str,
    y_title: str = "rate",
    y_range: tuple[float, float] | None = (0, 1.05),
) -> go.Figure:
    """Grouped bars comparing the privileged and unprivileged group per category."""
    fig = go.Figure()
    for name, values, color in ((f"{priv_name} (privileged)", priv_values, PRIV_COLOR),
                                (f"{unpriv_name} (unprivileged)", unpriv_values, UNPRIV_COLOR)):
        fig.add_bar(x=categories, y=values, name=name, marker_color=color,
                    text=[f"{v:.2f}" for v in values], textposition="outside",
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
    fig.update_layout(title=title, barmode="group", yaxis_title=y_title)
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return style_figure(fig)


def top_features_bar(
    importance: pd.DataFrame, group: str, role: str, attribute: str, proxies: list[str], k: int
) -> go.Figure:
    """One group's top-k features (horizontal bars, largest at the top)."""
    part = importance[importance["group"] == group].nlargest(k, "mean_abs_attribution")[::-1]

    def label(feature: str) -> str:
        if feature == attribute:
            return f"{feature} (sensitive)"
        return f"{feature} (proxy)" if feature in proxies else feature

    color = PRIV_COLOR if role == "privileged" else UNPRIV_COLOR
    fig = go.Figure(go.Bar(
        x=part["mean_abs_attribution"], y=[label(f) for f in part["feature"]], orientation="h",
        marker_color=color, text=[f"{v:.3g}" for v in part["mean_abs_attribution"]], textposition="outside",
        hovertemplate="%{y}<br>mean |attribution| %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"{group} ({role})", xaxis_title="mean |attribution|", showlegend=False)
    return style_figure(fig, height=60 + 32 * len(part))


def group_box(
    per_instance: pd.DataFrame, column: str, priv: str, unpriv: str, title: str, y_title: str
) -> go.Figure:
    """Box plot of a per-instance measure for both groups (all points on hover)."""
    fig = go.Figure()
    for group, role, color in ((priv, "privileged", PRIV_COLOR), (unpriv, "unprivileged", UNPRIV_COLOR)):
        values = per_instance.loc[per_instance["group"] == group, column].dropna()
        fig.add_box(y=values, name=f"{group} ({role})", marker_color=color, boxmean=True,
                    hovertemplate="%{y:.3f}<extra></extra>")
    fig.update_layout(title=title, yaxis_title=y_title, showlegend=False)
    return style_figure(fig)


def confusion_heatmap(tn: int, fp: int, fn: int, tp: int, title: str) -> go.Figure:
    """2x2 confusion matrix with counts written in every cell."""
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted 0", "Predicted 1"], y=["Actual 0", "Actual 1"],
        colorscale="Blues", showscale=False, text=z, texttemplate="%{text}",
        hovertemplate="%{y}, %{x}: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return style_figure(fig, height=300)


# --------------------------------------------------------------------------
# Live recomputation (small samples only)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def live_comparison(
    dataset: str, setting: str, model_name: str, method: str, attribute: str, n_per_group: int
) -> tuple[dict, pd.DataFrame]:
    """Re-run explain + compare for one selection on a SMALL sample.

    Uses exactly the same functions as the pipeline, with fewer instances,
    fewer LIME samples (300) and fewer bootstrap resamples (200), so it
    finishes in seconds. Results are noisier than the precomputed ones and
    are cached so the same request is not recomputed.
    """
    from src.compare import compare_one, select_proxies
    from src.explain import ExplainSettings, explain_model, sample_instances

    data = load_data(dataset)
    split = load_split(dataset)
    settings = ExplainSettings(shap_n_per_group=n_per_group, lime_n_per_group=max(5, n_per_group // 2),
                               lime_num_samples=300)
    instances = sample_instances(split, data.metadata, settings)
    table = explain_model(get_model(dataset, setting, model_name), split, instances,
                          dataset, setting, model_name, settings)
    proxies, _ = select_proxies(split, data.metadata, attribute)
    row, per_instance, _ = compare_one(table, instances, data.metadata, setting, model_name,
                                       method, attribute, proxies, n_boot=200)
    return row, per_instance
