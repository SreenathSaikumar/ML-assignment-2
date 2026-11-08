import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import features_and_target, load_dataset  # noqa: E402
from src.metrics import confusion_and_report, evaluate_classifier, metrics_table  # noqa: E402
from src.paths import FEATURE_COLS, MODEL_DIR, MODEL_FILES, TARGET_COL, TEST_CSV  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("scitos.app")

st.set_page_config(
    page_title="SCITOS · Soft Console",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Neumorphism (soft UI): shared surface color, dual light/dark shadows,
# large radii, soft coral→pink / blue→violet gradient accents.
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #E6EBF5;
  --text: #3D4A5C;
  --muted: #7A8699;
  --shadow-dark: #C8D0E0;
  --shadow-light: #FFFFFF;
  --accent-a: #FF8A7A;
  --accent-b: #FF6BB5;
  --accent-c: #7B8CFF;
  --accent-d: #9B7BFF;
  --neu-out: 9px 9px 18px var(--shadow-dark), -9px -9px 18px var(--shadow-light);
  --neu-out-sm: 5px 5px 12px var(--shadow-dark), -5px -5px 12px var(--shadow-light);
  --neu-in: inset 6px 6px 12px var(--shadow-dark), inset -6px -6px 12px var(--shadow-light);
}

html, body, [class*="css"] {
  font-family: "Nunito", ui-sans-serif, system-ui, sans-serif;
  color: var(--text);
}

.stApp {
  background: var(--bg);
  color: var(--text);
}

.stApp::before { display: none; }

[data-testid="stSidebar"] {
  background: var(--bg) !important;
  border-right: none !important;
}

[data-testid="stSidebar"] > div:first-child {
  background: var(--bg);
}

.block-container {
  padding-top: 1.5rem;
  max-width: 1180px;
}

.hero {
  padding: 1.5rem 1.65rem 1.4rem;
  margin-bottom: 1.4rem;
  border-radius: 28px;
  background: var(--bg);
  box-shadow: var(--neu-out);
  border: none;
}

.hero-kicker {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  font-weight: 600;
  margin-bottom: 0.4rem;
}

.hero h1 {
  margin: 0;
  font-size: clamp(1.6rem, 2.5vw, 2.15rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text);
}

.hero p {
  margin: 0.55rem 0 0;
  color: var(--muted);
  max-width: 62ch;
  line-height: 1.55;
  font-weight: 500;
}

.status-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.7rem;
  margin-top: 1.15rem;
}

.chip {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  background: var(--bg);
  box-shadow: var(--neu-out-sm);
  color: var(--text);
  font-weight: 600;
  border: none;
}

.chip .dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 0.45rem;
  background: linear-gradient(135deg, var(--accent-c), var(--accent-d));
  box-shadow: 0 0 0 3px rgba(123, 140, 255, 0.15);
}

.panel {
  padding: 1.15rem 1.25rem 1.1rem;
  margin: 0.2rem 0 1.15rem;
  border-radius: 24px;
  background: var(--bg);
  box-shadow: var(--neu-out);
  border: none;
}

.panel-label {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.35rem;
  font-weight: 600;
}

.panel-title {
  font-size: 1.12rem;
  font-weight: 750;
  color: var(--text);
  margin: 0;
}

div[data-testid="stMetric"] {
  background: var(--bg);
  border: none;
  border-radius: 22px;
  padding: 0.95rem 1rem;
  box-shadow: var(--neu-out-sm);
}

div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.66rem !important;
  font-weight: 600 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: "Nunito", ui-sans-serif, sans-serif;
  color: var(--text) !important;
  font-weight: 800;
}

.stSelectbox label, .stFileUploader label {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-size: 0.7rem !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
}

div[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {
  background: var(--bg) !important;
  border: none !important;
  border-radius: 18px !important;
  box-shadow: var(--neu-in) !important;
  color: var(--text) !important;
}

.stButton > button {
  background: var(--bg) !important;
  color: var(--text) !important;
  border: none !important;
  border-radius: 999px !important;
  box-shadow: var(--neu-out-sm) !important;
  font-weight: 700 !important;
}

.stButton > button:hover {
  box-shadow: var(--neu-in) !important;
  color: var(--accent-c) !important;
}

.sidebar-note {
  margin-top: 1.1rem;
  padding: 1rem 1.05rem;
  border-radius: 20px;
  background: var(--bg);
  box-shadow: var(--neu-in);
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.5;
  font-weight: 500;
}

.sidebar-note code {
  background: linear-gradient(90deg, var(--accent-c), var(--accent-d));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  font-weight: 700;
}

[data-testid="stExpander"] {
  border: none !important;
  border-radius: 22px !important;
  background: var(--bg) !important;
  box-shadow: var(--neu-out-sm);
}

[data-testid="stExpander"] details {
  border: none !important;
}

pre, .stCodeBlock, [data-testid="stCode"] {
  background: var(--bg) !important;
  border-radius: 18px !important;
  box-shadow: var(--neu-in) !important;
  border: none !important;
}

[data-testid="stDataFrame"],
[data-testid="stDataFrameRecycled"] {
  border-radius: 18px;
  box-shadow: var(--neu-out-sm);
  overflow: hidden;
}

.grad-bar {
  height: 6px;
  width: 72px;
  border-radius: 999px;
  margin: 0.55rem 0 0.15rem;
  background: linear-gradient(90deg, var(--accent-a), var(--accent-b), var(--accent-c), var(--accent-d));
  box-shadow: 2px 2px 6px var(--shadow-dark), -2px -2px 6px var(--shadow-light);
}

h2, h3, h4 {
  color: var(--text) !important;
  font-weight: 750 !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


@st.cache_resource
def load_artifacts() -> tuple[LabelEncoder, dict[str, Pipeline]]:
    encoder: LabelEncoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models: dict[str, Pipeline] = {
        name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()
    }
    return encoder, models


def validate_upload(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {missing}")
    return df[required].copy()


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">SCITOS · Soft Console</div>
          <h1>Wall-Following Robot Navigation</h1>
          <div class="grad-bar"></div>
          <p>
            Evaluate saved ultrasound classifiers on test CSV data without retraining.
          </p>
          <div class="status-row">
            <span class="chip"><span class="dot"></span>Models ready</span>
            <span class="chip">24 sensor channels</span>
            <span class="chip">4 navigation classes</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confusion_figure(
    cm: NDArray[np.int64],
    class_names: list[str] | NDArray[np.str_],
) -> Figure:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor="#E6EBF5")
    ax.set_facecolor("#E6EBF5")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#7B8CFF", as_cmap=True),
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=4,
        linecolor="#E6EBF5",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", color="#7A8699", fontweight="bold")
    ax.set_ylabel("Actual", color="#7A8699", fontweight="bold")
    ax.tick_params(colors="#3D4A5C", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def run_evaluation(
    *,
    model_name: str,
    models: dict[str, Pipeline],
    encoder: LabelEncoder,
    X: pd.DataFrame,
    y_enc: NDArray[np.int64],
    source_label: str,
) -> tuple[dict[str, float], NDArray[np.int64], str, dict[str, dict[str, float]]]:
    logger.info(
        "Starting evaluation | model=%s | rows=%s | source=%s",
        model_name,
        len(X),
        source_label,
    )
    model = models[model_name]
    scores = evaluate_classifier(model, X, y_enc)
    cm, report = confusion_and_report(model, X, y_enc, encoder.classes_)
    logger.info(
        "Selected model done | Acc=%.4f AUC=%.4f F1=%.4f MCC=%.4f",
        scores["Accuracy"],
        scores["AUC"],
        scores["F1"],
        scores["MCC"],
    )

    all_scores: dict[str, dict[str, float]] = {}
    for name, pipe in models.items():
        all_scores[name] = evaluate_classifier(pipe, X, y_enc)
        logger.info(
            "Compared %s | Acc=%.4f F1=%.4f MCC=%.4f",
            name,
            all_scores[name]["Accuracy"],
            all_scores[name]["F1"],
            all_scores[name]["MCC"],
        )

    logger.info("Evaluation finished for %s", model_name)
    return scores, cm, report, all_scores


def load_uploaded_data(uploaded) -> tuple[pd.DataFrame, str]:
    upload_key = f"{uploaded.name}:{uploaded.size}"
    is_new_upload = st.session_state.get("upload_key") != upload_key
    raw = pd.read_csv(uploaded)
    data = validate_upload(raw)
    source_label = uploaded.name
    if is_new_upload:
        st.session_state.pop("eval_result", None)
        st.session_state["upload_key"] = upload_key
        logger.info(
            "Test file uploaded | name=%s | size=%s bytes",
            uploaded.name,
            uploaded.size,
        )
        logger.info(
            "Upload validated | rows=%s | cols=%s | classes=%s",
            len(data),
            len(FEATURE_COLS),
            sorted(data[TARGET_COL].unique().tolist()),
        )
    return data, source_label


def load_bundled_test_data() -> tuple[pd.DataFrame, str]:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            "Bundled test CSV missing. Run training to create data/test_data.csv."
        )
    data = load_dataset(TEST_CSV)
    source_label = str(TEST_CSV.relative_to(ROOT))
    if st.session_state.get("upload_key") != source_label:
        st.session_state.pop("eval_result", None)
        st.session_state["upload_key"] = source_label
        logger.info("Bundled test set selected | path=%s | rows=%s", source_label, len(data))
    return data, source_label


def render_dataset_panel(data: pd.DataFrame, source_label: str, y: pd.Series) -> None:
    st.markdown(
        '<div class="panel"><div class="panel-label">Dataset</div>'
        '<div class="panel-title">Active evaluation set</div>'
        '<div class="grad-bar"></div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Features", len(FEATURE_COLS))
    c3.metric("Source", source_label)
    with st.expander("Preview rows & class mix"):
        st.dataframe(data.head(20), width="stretch")
        st.dataframe(y.value_counts().rename("count").to_frame(), width="stretch")


def render_metrics_panel(
    *,
    active_model: str,
    scores: dict[str, float],
    cm: NDArray[np.int64],
    report: str,
    all_scores: dict[str, dict[str, float]],
    class_names: list[str] | NDArray[np.str_],
) -> None:
    st.markdown(
        f'<div class="panel"><div class="panel-label">Model readout</div>'
        f'<div class="panel-title">{active_model}</div>'
        f'<div class="grad-bar"></div></div>',
        unsafe_allow_html=True,
    )
    metric_cols = st.columns(6)
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for col, key in zip(metric_cols, metric_names, strict=True):
        col.metric(key, f"{scores[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="panel"><div class="panel-label">Diagnostics</div>'
            '<div class="panel-title">Confusion matrix</div></div>',
            unsafe_allow_html=True,
        )
        fig = confusion_figure(cm, class_names)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-label">Diagnostics</div>'
            '<div class="panel-title">Classification report</div></div>',
            unsafe_allow_html=True,
        )
        st.code(report)

    st.markdown(
        '<div class="panel"><div class="panel-label">Comparative board</div>'
        '<div class="panel-title">All models on this test set</div>'
        '<div class="grad-bar"></div></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(metrics_table(all_scores), width="stretch")


def main() -> None:
    inject_theme()
    render_hero()

    if not (MODEL_DIR / "label_encoder.joblib").exists():
        st.error(
            "No saved models found in `model/`. "
            "Run `uv run python -m src.train` or the training notebook first."
        )
        st.stop()

    encoder, models = load_artifacts()

    with st.sidebar:
        st.markdown("### Control deck")
        data_mode = st.radio(
            "Data source",
            options=["Upload CSV", "Bundled test set"],
            index=0,
            help="Metrics stay hidden until you evaluate an upload or explicitly use the bundled test set.",
        )
        uploaded = None
        if data_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload test CSV", type=["csv"])
        else:
            st.caption(
                "Uses `data/test_data.csv` from training. Nothing is shown until you evaluate."
            )

        model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
        run_clicked = st.button("Start evaluation", type="primary", width="stretch")
        st.markdown(
            """
            <div class="sidebar-note">
              Expected schema<br/>
              Features: <code>US1</code> … <code>US24</code><br/>
              Target: <code>class</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.get("data_mode") != data_mode:
        st.session_state["data_mode"] = data_mode
        st.session_state.pop("eval_result", None)
        logger.info("Data source switched to %s", data_mode)

    data: pd.DataFrame | None = None
    source_label = ""
    try:
        if data_mode == "Upload CSV":
            if uploaded is None:
                st.info(
                    "Upload a test CSV, choose a model, then press **Start evaluation**. "
                    "Or switch to **Bundled test set** to evaluate the held-out training split."
                )
                st.stop()
            data, source_label = load_uploaded_data(uploaded)
        else:
            data, source_label = load_bundled_test_data()
    except Exception as exc:  # noqa: BLE001 — show friendly UI errors
        logger.exception("Failed to load evaluation data")
        st.error(f"Could not load data: {exc}")
        st.stop()

    X, y = features_and_target(data)
    try:
        y_enc = encoder.transform(y)
    except ValueError as exc:
        st.error(f"Target labels do not match training classes: {exc}")
        st.stop()

    render_dataset_panel(data, source_label, y)

    if run_clicked:
        with st.spinner(f"Evaluating `{model_name}` on {len(data)} rows…"):
            scores, cm, report, all_scores = run_evaluation(
                model_name=model_name,
                models=models,
                encoder=encoder,
                X=X,
                y_enc=y_enc,
                source_label=source_label,
            )
        st.session_state["eval_result"] = {
            "model_name": model_name,
            "source_label": source_label,
            "data_mode": data_mode,
            "scores": scores,
            "cm": cm,
            "report": report,
            "all_scores": all_scores,
        }

    result = st.session_state.get("eval_result")
    if result is None:
        if data_mode == "Upload CSV":
            st.info("CSV ready. Press **Start evaluation** to compute metrics.")
        else:
            st.info(
                "Bundled test set selected. Press **Start evaluation** to show held-out test metrics."
            )
        st.stop()

    if result.get("data_mode") != data_mode:
        st.info("Data source changed. Press **Start evaluation** again to refresh metrics.")
        st.stop()

    render_metrics_panel(
        active_model=result["model_name"],
        scores=result["scores"],
        cm=result["cm"],
        report=result["report"],
        all_scores=result["all_scores"],
        class_names=encoder.classes_,
    )


if __name__ == "__main__":
    main()
