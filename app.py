# app.py

import os
import io
import tempfile
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
# --- Optional PDF support ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# --- Domain-specific imports ---
import ip_parse as ip
import multisig_fit as msf

PLOTLY_DARK_STYLE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", size=14),
    xaxis=dict(color="#E2E8F0", gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(color="#E2E8F0", gridcolor="rgba(255,255,255,0.08)"),
    hoverlabel=dict(bgcolor="#1E293B", font_color="white", bordercolor="#38BDF8"),
)

def fmt4(x):
    """Format numeric to 4 d.p. (safe)."""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "—"

# --------------------------------------------------------
# Streamlit page config
# --------------------------------------------------------
st.set_page_config(
    page_title="MultiSig AUC Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------
# Utility: load an SE .ip1 / .ip2 file from path or upload
# --------------------------------------------------------
def load_se_file(source: str, uploaded_file=None):
    """
    Returns (meta, df_raw, label)
    - source: "upload" or a filename like "Sim1b.ip1"
    """
    if source == "upload":
        if uploaded_file is None:
            return None, None, None

        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            meta, df_raw = ip.parse_ip_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        label = uploaded_file.name
        return meta, df_raw, label

    # Example dataset from local directory
    meta, df_raw = ip.parse_ip_file(source)
    label = source
    return meta, df_raw, label


# --------------------------------------------------------
# Utility: apply trimming and downsampling
# --------------------------------------------------------
def trim_and_downsample(df_raw, r_min, r_max, downsample_n):
    if df_raw is None:
        return None
    df = df_raw.copy()
    if ("radius_cm" in df.columns) and (r_min is not None) and (r_max is not None):
        df = df[(df["radius_cm"] >= r_min) & (df["radius_cm"] <= r_max)]
    if downsample_n and downsample_n > 1:
        df = df.iloc[::downsample_n, :].reset_index(drop=True)
    return df


# --------------------------------------------------------
# Session state initialisation
# --------------------------------------------------------
if "fit_results" not in st.session_state:
    st.session_state["fit_results"] = None
    st.session_state["fit_meta"] = None
    st.session_state["fit_df"] = None
    st.session_state["fit_label"] = None

if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
    st.session_state["meta"] = None
    st.session_state["dataset_label"] = None
if "fit_config" not in st.session_state:
    st.session_state["fit_config"] = {}
    


# --------------------------------------------------------
# Sidebar: navigation + analysis settings
# --------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚗️ MultiSig AUC\nWorkbench")

    nav_labels = {
        "Overview": "🏠 Overview",
        "Analysis": "📈 Analysis",
        "Datasets": "📂 Datasets",
        "Reports": "📄 Reports",
        "Settings": "⚙️ Settings",
        "Help": "❓ Help",
    }

    section = st.radio(
        "Navigation",
        list(nav_labels.keys()),
        index=0,
        label_visibility="collapsed",
        format_func=lambda x: nav_labels[x],
    )

    st.markdown("---")
    st.markdown("**Analysis settings**")

    # --- Data source ---
    st.markdown("##### Data source")
    input_mode = st.radio("Choose input", ["Upload SE file", "Example dataset"], index=0)

    meta = None
    df_raw = None
    dataset_label = None

    uploaded_file = None
    example_name = None

    if input_mode == "Upload SE file":
        uploaded_file = st.file_uploader(
            "Upload .ip1 / .ip2 file",
            type=["ip1", "IP1", "ip2", "IP2", "txt"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            meta, df_raw, dataset_label = load_se_file("upload", uploaded_file)
    else:
        example_name = st.selectbox(
            "Example dataset",
            ["Sim1b.ip1", "IgG1.IP1", "pullulan.ip2"],
        )
        if st.button("Load example dataset"):
            meta, df_raw, dataset_label = load_se_file(example_name)

    # If we loaded something new, store it in session_state so all pages see it
    if df_raw is not None:
        st.session_state["df_raw"] = df_raw
        st.session_state["meta"] = meta
        st.session_state["dataset_label"] = dataset_label

    # Work with whatever is in state from here on
    df_raw = st.session_state["df_raw"]
    meta = st.session_state["meta"]
    dataset_label = st.session_state["dataset_label"]

    # Defaults (will be overwritten once data exists)
    r_ref = 7.00
    sigma_ref = 1.50
    lam_smooth = 0.02
    num_runs = 20
    downsample_n = 1
    pub_style = True
    vbar = None
    rho = None

    if df_raw is not None:
        st.markdown("---")
        st.markdown("##### Fitting parameters")

        r_ref = st.number_input(
            "Reference radius r_ref (cm)",
            value=7.00,
            step=0.05,
            format="%.4f",
            help=(
                "Radius (in cm) used as the reference point r_ref. "
                "J_ref is taken near this radius when normalising the fringe data."
            ),
        )
        sigma_ref = st.number_input(
            "Sigma reference σ_ref (1/cm², grid scale)",
            value=1.50,
            step=0.10,
            format="%.4f",
            help=(
                "σ_ref sets the centre/scale of the log-spaced σ grid (units 1/cm²). "
                "It controls where the algorithm searches for peaks, but does not "
                "change the raw experimental data."
            ),
        )

        lam_smooth = st.number_input(
            "Smoothing λ",
            value=0.02,
            min_value=0.0,
            step=0.01,
            format="%.4f",
            help=(
                "Non-negative smoothing parameter. Larger λ penalises sharp jumps "
                "between adjacent σ-bins, giving smoother distributions."
            ),
        )

        num_runs = st.number_input(
            "Repeat fits (num_runs)",
            value=20,
            min_value=1,
            step=1,
            help=(
                "Number of jittered MultiSig fits to run. Results are averaged and "
                "precision (SD, %RSD) is estimated across these runs."
            ),
        )
        # Optimiser choice
        optimiser_label_to_code = {
            "TRF (bounded, non-negative coefficients)": "trf",
            "LM (unconstrained, Levenberg–Marquardt style)": "lm",
        }
        optimiser_choice = st.selectbox(
            "Optimiser",
            list(optimiser_label_to_code.keys()),
            index=0,
            help=(
                "TRF uses bounds and respects non-negativity of coefficients. "
                "LM ignores bounds and behaves more like classic Levenberg–Marquardt fits."
            ),
        )
        solver_code = optimiser_label_to_code[optimiser_choice]

        # v-bar and rho with dataset-specific defaults
        st.markdown("##### Solution properties")
        default_vbar = 0.73
        default_rho = 1.0
        if dataset_label is not None:
            name_lower = dataset_label.lower()
            if "igg1" in name_lower:
                default_vbar, default_rho = 0.731, 1.00452
            elif "pullulan" in name_lower:
                default_vbar, default_rho = 0.602, 1.0

        vbar = st.number_input(
            "Partial specific volume v̄ (mL/g)",
            value=float(default_vbar),
            format="%.4f",
            help=(
                "Partial specific volume of the solute in mL/g (cm³/g). "
                "Used in σ → molecular-weight conversion."
            ),
        )
        rho = st.number_input(
            "Density ρ (g/mL)",
            value=float(default_rho),
            format="%.6f",
            help=(
                "Solution density in g/mL (g/cm³). Combined with v̄ to compute "
                "molecular weights from σ."
            ),
        )
        

        # --- Trimming controls ---
        st.markdown("##### Trimming and downsampling")

        r_min_raw = float(df_raw["radius_cm"].min())
        r_max_raw = float(df_raw["radius_cm"].max())

        trim_range = st.slider(
            "Radius range (cm)",
            min_value=round(r_min_raw, 3),
            max_value=round(r_max_raw, 3),
            value=(round(r_min_raw, 3), round(r_max_raw, 3)),
            step=0.001,
        )

        downsample_n = st.number_input(
            "Downsample: keep every n-th point",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )

        st.markdown("##### Plot options")
        pub_style = st.checkbox(
            "Publication-style plots (white axes, grid off)",
            value=True,
        )

        run_fit = st.button("Run MultiSig fit", type="primary")
    else:
        run_fit = False
        trim_range = (None, None)
        
# --------------------------------------------------------
# PDF generation
# --------------------------------------------------------

def build_pdf_report(
    buffer,
    summary_df,
    mw_diag,
    species_df,
    fit_config,
    meta,
    dataset_label,
    fig_overlay=None,
    fig_coeff=None,
    fig_resid=None,
):
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    # --- Title ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "MultiSig AUC Analysis Report")
    y -= 40

    # --- Dataset info ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Dataset: {dataset_label}")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Sample name: {meta.get('sample_name', '—')}")
    y -= 15
    c.drawString(40, y, f"Rotor speed: {meta.get('rpm','—')} rpm")
    y -= 15
    c.drawString(40, y, f"Temperature: {meta.get('temperature_c','—')} °C")
    y -= 30

    # --- Fit configuration ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Fit configuration:")
    y -= 20
    c.setFont("Helvetica", 10)
    for key, val in fit_config.items():
        c.drawString(50, y, f"- {key}: {val}")
        y -= 14
        if y < 100:
            c.showPage()
            y = height - 50

    # --- Summary table ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Molecular-weight summary:")
    y -= 20

    text_obj = c.beginText(40, y)
    text_obj.setFont("Helvetica", 9)
    text_obj.textLines(summary_df.to_string(index=False))
    c.drawText(text_obj)
    y -= (14 * (len(summary_df) + 3))

    # --- Precision table ---
    if mw_diag is not None:
        if y < 200:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Precision across repeated fits:")
        y -= 20

        text_obj = c.beginText(40, y)
        text_obj.setFont("Helvetica", 9)
        text_obj.textLines(mw_diag.to_string(index=False))
        c.drawText(text_obj)
        y -= (14 * (len(mw_diag) + 3))

    # --- Species table ---
    if species_df is not None:
        if y < 200:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Dominant species:")
        y -= 20

        text_obj = c.beginText(40, y)
        text_obj.setFont("Helvetica", 9)
        text_obj.textLines(species_df.to_string(index=False))
        c.drawText(text_obj)
        y -= (14 * (len(species_df) + 3))

    # --- Insert plots as images ---
    def insert_fig(fig, y):
        """Convert matplotlib fig to PNG and insert into ReportLab PDF."""
        if fig is None:
            return y
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=250, bbox_inches="tight")
        buf.seek(0)
        img = ImageReader(buf)
        img_width = width - 80
        img_height = img_width * 0.55  # maintain aspect ratio nicely
        if y < img_height + 40:
            c.showPage()
            y = height - 50
        c.drawImage(img, 40, y - img_height, width=img_width, height=img_height)
        return y - img_height - 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Figures")
    y -= 30

    y = insert_fig(fig_overlay, y)
    y = insert_fig(fig_coeff, y)
    y = insert_fig(fig_resid, y)

    c.showPage()
    c.save()
# --------------------------------------------------------
# Global DARK theme CSS + animations + plot hover cards
# --------------------------------------------------------
dark_css = """
<style>
:root {
    --color-primary: #2D5BFF;
    --color-secondary: #020617;
    --color-accent: #6C8BE8;
    --color-background: #020617;
    --font-family-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
}
html, body, .stApp {
    background-color: var(--color-secondary) !important;
    color: #e5e7eb !important;
    font-family: var(--font-family-sans);
}

/* Main container */
.block-container {
    padding-top: 1.1rem;
    background: radial-gradient(circle at top left, #1f2937 0, #020617 45%, #020617 100%);
}

/* Titles and captions */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption, .stText {
    color: #e5e7eb !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2D5BFF, #6C8BE8);
    color: #f9fafb;
    border: none;
    border-radius: 999px;
    padding: 0.45rem 1.2rem;
    font-weight: 500;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.8);
    filter: brightness(1.05);
}

/* Metrics */
.stMetric {
    background: radial-gradient(circle at top left, rgba(45,91,255,0.35), rgba(15,23,42,0.95));
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* Dataset cards with hover animation */
.dataset-card {
    background: radial-gradient(circle at top left, #2D5BFF33, #020617);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    border: 1px solid rgba(148, 163, 184, 0.4);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    cursor: pointer;
}
.dataset-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 16px 40px rgba(15,23,42,0.9);
    border-color: rgba(129, 140, 248, 0.9);
}

/* Pill tabs */
.pill-tab {
    background: linear-gradient(90deg, #1f2937, #020617);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: #e5e7eb;
    display: inline-block;
}

/* Plot cards: hover scale */
.plot-card {
    background: radial-gradient(circle at top left, rgba(37,99,235,0.35), rgba(15,23,42,0.98));
    border-radius: 18px;
    padding: 0.75rem 0.9rem 0.3rem 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.4);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.plot-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 18px 48px rgba(15,23,42,0.95);
    border-color: rgba(129, 140, 248, 0.95);
}

/* Inputs/dropdowns/textareas on dark background */
.stTextInput > div > input,
.stNumberInput input,
.stSelectbox > div > div,
.stTextArea textarea {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 0.75rem !important;
    border: 1px solid rgba(148,163,184,0.6) !important;
}
.stTextInput > label,
.stNumberInput > label,
.stSelectbox > label,
.stSlider > label {
    color: #e5e7eb !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #020617;
    border-radius: 999px;
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
    border: 1px solid rgba(148,163,184,0.6);
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: rgba(129,140,248,0.9);
}

/* Shimmer skeleton for loading placeholder */
@keyframes shimmer {
    0%   { background-position: -450px 0; }
    100% { background-position: 450px 0; }
}
.shimmer-card {
    border-radius: 16px;
    height: 140px;
    background: linear-gradient(
        90deg,
        #020617 0px,
        #111827 40px,
        #020617 80px
    );
    background-size: 450px 140px;
    animation: shimmer 1.3s infinite linear;
    border: 1px solid rgba(31,41,55,0.9);
    margin-bottom: 0.5rem;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)


# --------------------------------------------------------
# Apply trimming + downsampling if we have data
# --------------------------------------------------------
df_trim = None
if (df_raw is not None) and (trim_range[0] is not None):
    df_trim = trim_and_downsample(df_raw, trim_range[0], trim_range[1], downsample_n)


# --------------------------------------------------------
# If user clicked "Run MultiSig fit", perform the analysis
# --------------------------------------------------------
if run_fit and (df_trim is not None) and (len(df_trim) > 0) and (meta is not None):
    r_data = df_trim["radius_cm"].to_numpy(dtype=float)
    J_data = df_trim["fringe"].to_numpy(dtype=float)

    rpm = meta.get("rpm", None) if isinstance(meta, dict) else None
    temp_c = meta.get("temperature_c", 20.0) if isinstance(meta, dict) else 20.0

    results = msf.repeat_fit(
        r_data,
        J_data,
        r_ref=r_ref,
        sigma_ref=sigma_ref,
        Jref_mode="nearest",
        num_runs=int(num_runs),
        lam_smooth=float(lam_smooth),
        nonneg=True,
        rpm=rpm,
        temp_c=temp_c,
        v_ml_per_g=vbar,
        rho_g_per_ml=rho,
        solver=solver_code,
    )

    # Save full context of the fit so reporting is reproducible
    fit_config = {
        "r_ref": float(r_ref),
        "sigma_ref": float(sigma_ref),
        "lam_smooth": float(lam_smooth),
        "num_runs": int(num_runs),
        "vbar_ml_per_g": float(vbar),
        "rho_g_per_ml": float(rho),
        "rpm": rpm,
        "temperature_c": temp_c,
        "radius_min_cm": float(df_trim["radius_cm"].min()),
        "radius_max_cm": float(df_trim["radius_cm"].max()),
        "n_points": int(len(df_trim)),
        "Jref_mode": "nearest",
        "pub_style": bool(pub_style),
        "solver": solver_code,  # new
    }

    st.session_state["fit_results"] = results
    st.session_state["fit_meta"] = meta
    st.session_state["fit_df"] = df_trim
    st.session_state["fit_label"] = dataset_label
    st.session_state["fit_config"] = fit_config

# --------------------------------------------------------
# Unlock results and metadata from session_state
# --------------------------------------------------------
results = st.session_state.get("fit_results")
meta_state = st.session_state.get("fit_meta")
df_state = st.session_state.get("fit_df")
label_state = st.session_state.get("fit_label")
fit_config = st.session_state.get("fit_config", {})


# --------------------------------------------------------
# COMMON: top header bar
# --------------------------------------------------------
col_title, col_search, col_user = st.columns([3, 2, 1])

with col_title:
    st.markdown("### 🧪 AUC Data Analysis Workbench")
    if label_state:
        st.caption(f"Active dataset: **{label_state}**")

with col_search:
    st.text_input(
        "Search datasets…",
        placeholder="Search datasets…",
        label_visibility="collapsed",
    )

with col_user:
    if meta_state and isinstance(meta_state, dict):
        rpm_txt = meta_state.get("rpm", "—")
        temp_txt = meta_state.get("temperature_c", "—")
    else:
        rpm_txt, temp_txt = "—", "—"
    st.metric("Rotor speed (rpm)", f"{rpm_txt}")
    st.metric("Temperature (°C)", f"{temp_txt}")

st.markdown("---")


# Helper to compute Mw/Mn/Mz display text
def extract_mass_metrics(results_dict):
    Mw_txt = "—"
    Mn_txt = "—"
    Mz_txt = "—"
    if not (isinstance(results_dict, dict) and "summary" in results_dict):
        return Mw_txt, Mn_txt, Mz_txt

    try:
        summary_df = results_dict["summary"].set_index("metric")
        if "Mw_g_per_mol" in summary_df.index:
            Mw_txt = f"{summary_df.loc['Mw_g_per_mol', 'mean'] / 1000.0:,.2f} kDa"
        if "Mn_g_per_mol" in summary_df.index:
            Mn_txt = f"{summary_df.loc['Mn_g_per_mol', 'mean'] / 1000.0:,.2f} kDa"
        if "Mz_g_per_mol" in summary_df.index:
            Mz_txt = f"{summary_df.loc['Mz_g_per_mol', 'mean'] / 1000.0:,.2f} kDa"
    except Exception:
        pass
    return Mw_txt, Mn_txt, Mz_txt

def build_mw_diagnostics(results_dict):
    """
    Build a table with Mn/Mw/Mz in kDa, SD, %RSD, and 95% CI.
    Returns a dataframe or None.
    """
    if not (isinstance(results_dict, dict) and "summary" in results_dict):
        return None

    summary = results_dict["summary"].copy()
    if "metric" not in summary.columns:
        return None

    summary = summary.set_index("metric")
    rows = []
    labels = [
        ("Mn_g_per_mol", "Mn"),
        ("Mw_g_per_mol", "Mw"),
        ("Mz_g_per_mol", "Mz"),
    ]
    for key, pretty in labels:
        if key not in summary.index:
            continue
        mean = float(summary.loc[key, "mean"])
        sd = float(summary.loc[key, "sd"]) if "sd" in summary.columns else float("nan")
        if mean <= 0:
            rsd = float("nan")
        else:
            rsd = 100.0 * sd / mean

        mean_kda = mean / 1000.0
        sd_kda = sd / 1000.0
        ci_low = (mean - 1.96 * sd) / 1000.0
        ci_high = (mean + 1.96 * sd) / 1000.0

        rows.append(
            {
                "Metric": pretty,
                "Mean (kDa)": mean_kda,
                "SD (kDa)": sd_kda,
                "%RSD": rsd,
                "95% CI low (kDa)": ci_low,
                "95% CI high (kDa)": ci_high,
            }
        )

    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df.round(
        {
            "Mean (kDa)": 2,
            "SD (kDa)": 3,
            "%RSD": 2,
            "95% CI low (kDa)": 2,
            "95% CI high (kDa)": 2,
        }
    )


def build_species_table(results_dict, max_rows=5):
    """
    Build a small table describing the dominant mass bins (species).
    Returns dataframe or None.
    """
    if not isinstance(results_dict, dict):
        return None
    coef = results_dict.get("coef_table")
    if coef is None or "c_mean" not in coef.columns:
        return None

    df = coef.copy()
    total = df["c_mean"].sum()
    if total <= 0:
        return None

    df["fraction"] = df["c_mean"] / total
    if "M_bin_gmol" in df.columns:
        df["M_bin_kDa"] = df["M_bin_gmol"] / 1000.0
    else:
        df["M_bin_kDa"] = float("nan")

    df = df.sort_values("fraction", ascending=False)
    df = df.head(max_rows)

    out = df[["M_bin_kDa", "fraction"]].copy()
    out = out.rename(
        columns={
            "M_bin_kDa": "Approx. M (kDa)",
            "fraction": "Fraction of total",
        }
    )
    out["Fraction of total"] = (100.0 * out["Fraction of total"]).round(1)
    return out

def build_numerical_validation_table(results_dict, multisig_reference: dict):
    """
    multisig_reference example:
      {
        "Mn_kDa": 99.0000,
        "Mw_kDa": 99.0000,
        "Mz_kDa": 99.0000,
        "E": 0.1234,
      }
    """
    if results_dict is None or "summary" not in results_dict:
        return None

    summary = results_dict["summary"].set_index("metric")

    def get_kda(metric_key):
        if metric_key not in summary.index:
            return np.nan
        return float(summary.loc[metric_key, "mean"]) / 1000.0

    our = {
        "Mn_kDa": get_kda("Mn_g_per_mol"),
        "Mw_kDa": get_kda("Mw_g_per_mol"),
        "Mz_kDa": get_kda("Mz_g_per_mol"),
        "E": float(summary.loc["baseline_E", "mean"]) if "baseline_E" in summary.index else np.nan,
    }

    rows = []
    for k, our_val in our.items():
        ref_val = float(multisig_reference.get(k, np.nan))
        abs_err = our_val - ref_val
        rel_err = (abs_err / ref_val * 100.0) if (ref_val not in (0.0, np.nan) and np.isfinite(ref_val)) else np.nan
        rows.append({
            "Metric": k,
            "This GUI": our_val,
            "MULTISIG": ref_val,
            "Abs. diff": abs_err,
            "% diff": rel_err,
        })

    df = pd.DataFrame(rows)

    # enforce 4 d.p. display (density etc handled separately elsewhere)
    for col in ["This GUI", "MULTISIG", "Abs. diff", "% diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ========================================================
# SECTION: OVERVIEW
# ========================================================
if section == "Overview":
    Mw_txt, Mn_txt, Mz_txt = extract_mass_metrics(results)
    fit_quality_txt = "—"
    if isinstance(results, dict):
        fit_quality_txt = f"λ={results.get('lam_smooth', 0.0):.3f}"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Number-average Mw (Mn)", Mn_txt)
    with kpi2:
        st.metric("Weight-average Mw (Mw)", Mw_txt)
    with kpi3:
        st.metric("Z-average Mw (Mz)", Mz_txt)
    with kpi4:
        st.metric("Fit settings", f"λ={fmt4(results.get('lam_smooth', 0.0))}" if isinstance(results, dict) else "—")

    st.markdown("### 📊 Fit Distribution")
    st.caption("MultiSig SE analysis overview")

    # Shimmer placeholders if no results yet but a dataset is loaded
    if (results is None) and (df_state is not None):
        cols_shim = st.columns(3)
        for c in cols_shim:
            with c:
                st.markdown('<div class="shimmer-card"></div>', unsafe_allow_html=True)
        st.info("Upload/choose a dataset and click **Run MultiSig fit** in the sidebar to populate this view.")
        st.stop()

    # Main content: overlay + molecular weights + σ-distribution + residuals
    row1_left, row1_right = st.columns([2, 1])

    with row1_left:
        st.markdown("#### Overlay: data vs MultiSig fit ↪")
    
        if results is None or df_state is None:
            st.info("Upload data and click **Run MultiSig fit** to see overlay plots.")
        else:
            if pub_style:
                plt.style.use("default")
            else:
                plt.style.use("dark_background")
    
            r = df_state["radius_cm"].to_numpy(dtype=float)
            J = df_state["fringe"].to_numpy(dtype=float)
    
            # --- static matplotlib overlay in the hover-card ---
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            fig_fit, ax_fit = msf.plot_fit(
                r,
                J,
                results["sigma_bins"],
                results["c_mean"],
                results["E_mean"],
                results["J_ref"],
                results["r_ref"],
            )
            ax_fit.set_title("Experimental data vs MultiSig fit")
            st.pyplot(fig_fit, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
            # PNG download for static overlay
            buf = io.BytesIO()
            fig_fit.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                "Download overlay plot (PNG)",
                data=buf.getvalue(),
                file_name="multisig_overlay.png",
                mime="image/png",
            )
    
            # Save overlay for PDF export
            buf_overlay = io.BytesIO()
            fig_fit.savefig(buf_overlay, format="png", dpi=300, bbox_inches="tight")
            buf_overlay.seek(0)
            st.session_state["pdf_overlay"] = buf_overlay
    
            # --- interactive Plotly overlay with bright colours ---
            J_fit = msf.predict_J(
                r,
                results["c_mean"],
                results["E_mean"],
                results["sigma_bins"],
                results["J_ref"],
                results["r_ref"],
            )
    
            with st.expander("Show interactive overlay (hover)"):
                df_overlay = pd.DataFrame(
                    {"radius_cm": r, "J_obs": J, "J_fit": J_fit}
                )
    
                fig_int_overlay = go.Figure()
    
                # Data points
                fig_int_overlay.add_scatter(
                    x=df_overlay["radius_cm"],
                    y=df_overlay["J_obs"],
                    mode="markers",
                    name="Data",
                    marker=dict(color="#38BDF8", size=6, opacity=0.9),
                )
    
                # Fit curve
                fig_int_overlay.add_scatter(
                    x=df_overlay["radius_cm"],
                    y=df_overlay["J_fit"],
                    mode="lines",
                    name="Fit",
                    line=dict(color="#F97316", width=3),
                )
    
                fig_int_overlay.update_layout(
                    **PLOTLY_DARK_STYLE,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title="Radius (cm)",
                    yaxis_title="Fringe",
                )
    
                st.plotly_chart(fig_int_overlay, use_container_width=True)

            # Save overlay for PDF export
            buf_overlay = io.BytesIO()
            fig_fit.savefig(buf_overlay, format="png", dpi=300, bbox_inches="tight")
            buf_overlay.seek(0)
            st.session_state["pdf_overlay"] = buf_overlay

    with row1_right:
        st.markdown("#### Molecular weights")

        if results is None or "summary" not in results:
            st.info("No fit summary available yet.")
        else:
            summary_df = results["summary"].copy()
            try:
                mw_row = summary_df[summary_df["metric"] == "Mw_g_per_mol"]["mean"].iloc[0]
                mn_row = summary_df[summary_df["metric"] == "Mn_g_per_mol"]["mean"].iloc[0]
                mz_row = summary_df[summary_df["metric"] == "Mz_g_per_mol"]["mean"].iloc[0]
                st.metric("Mw", f"{mw_row/1000.0:,.1f} kDa")
                st.metric("Mn", f"{mn_row/1000.0:,.1f} kDa")
                st.metric("Mz", f"{mz_row/1000.0:,.1f} kDa")

                st.caption("Download mass summary as CSV")
                csv_buf = summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download summary (.csv)",
                    data=csv_buf,
                    file_name="multisig_summary.csv",
                    mime="text/csv",
                )
            except Exception:
                st.write(summary_df)

    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.markdown("#### σ-distribution (coefficients)")

        if results is None or "coef_table" not in results:
            st.info("Run a fit to view coefficient distribution.")
        else:
            coef_table = results["coef_table"]

            x_axis = "mass" if results.get("mass_bins_gmol") is not None else "sigma"

            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            fig_coeffs, ax_coeffs = msf.plot_coeffs(coef_table, x_axis=x_axis)
            ax_coeffs.set_title("Coefficient mean ± SD")
            st.pyplot(fig_coeffs, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Save coefficient plot for PNG download
            buf_c = io.BytesIO()
            fig_coeffs.savefig(buf_c, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                "Download coefficient plot (PNG)",
                data=buf_c.getvalue(),
                file_name="multisig_coeffs.png",
                mime="image/png",
            )
                    # --- Optional interactive σ / mass plot ---
        

            with st.expander("Show interactive σ / mass distribution (hover)"):
            
                df_i = coef_table.copy()
                df_i["bin_index"] = range(1, len(df_i)+1)
            
                fig_int_coeff = px.bar(
                    df_i,
                    x="M_bin_gmol" if x_axis == "mass" else "sigma_bin",
                    y="c_mean",
                    labels={
                        "M_bin_gmol": "M bin (g/mol)",
                        "sigma_bin": "σ bin (1/cm²)",
                        "c_mean": "Coefficient mean",
                    },
                    color_discrete_sequence=["#38BDF8"],  # bright cyan
                )
            
                fig_int_coeff.update_traces(marker_line_width=0)
            
                fig_int_coeff.update_layout(
                    **PLOTLY_DARK_STYLE,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
            
                st.plotly_chart(fig_int_coeff, use_container_width=True)

            # Save coefficient plot for PDF export
            buf_coeff = io.BytesIO()
            fig_coeffs.savefig(buf_coeff, format="png", dpi=300, bbox_inches="tight")
            buf_coeff.seek(0)
            st.session_state["pdf_coeffs"] = buf_coeff

            # Coefficient table CSV
            coef_csv = coef_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download coefficients (.csv)",
                data=coef_csv,
                file_name="multisig_coefficients.csv",
                mime="text/csv",
            )

    with row2_right:
        st.markdown("#### Residual analysis")
    
        if results is None or df_state is None:
            st.info("Run a fit to view residuals.")
        else:
            r = df_state["radius_cm"].to_numpy(dtype=float)
            J = df_state["fringe"].to_numpy(dtype=float)
            J_hat = msf.predict_J(
                r,
                results["c_mean"],
                results["E_mean"],
                results["sigma_bins"],
                results["J_ref"],
                results["r_ref"],
            )
            resid = J - J_hat
    
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            fig_res, ax_res = plt.subplots()
            ax_res.scatter(r, resid, s=4, alpha=0.6)
            ax_res.axhline(0.0, color="red", linewidth=1)
            ax_res.set_xlabel("Radius (cm)")
            ax_res.set_ylabel("Residual (J_obs - J_fit)")
            ax_res.set_title("Residuals vs radius")
            st.pyplot(fig_res, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
            # --- Interactive residuals plot (Plotly) ---
            with st.expander("Show interactive residuals (hover)"):
            
                df_res = pd.DataFrame({"radius_cm": r, "residual": resid})
            
                fig_int_res = px.scatter(
                    df_res,
                    x="radius_cm",
                    y="residual",
                    labels={
                        "radius_cm": "Radius (cm)",
                        "residual": "J_obs - J_fit",
                    },
                    color_discrete_sequence=["#38BDF8"],  # bright cyan points
                )
            
                # horizontal zero-line
                fig_int_res.add_hline(
                    y=0.0,
                    line_color="#F87171",  # red
                    line_width=2,
                    opacity=0.8,
                )
            
                fig_int_res.update_traces(marker=dict(size=6, opacity=0.9))
            
                fig_int_res.update_layout(
                    **PLOTLY_DARK_STYLE,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
            
                st.plotly_chart(fig_int_res, use_container_width=True)
    
                # Save residual plot for PDF export
                buf_resid = io.BytesIO()
                fig_res.savefig(buf_resid, format="png", dpi=300, bbox_inches="tight")
                buf_resid.seek(0)
                st.session_state["pdf_residuals"] = buf_resid
    
                # Residuals CSV
                resid_df = pd.DataFrame({"radius_cm": r, "residual": resid})
                resid_csv = resid_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download residuals (.csv)",
                    data=resid_csv,
                    file_name="multisig_residuals.csv",
                    mime="text/csv",
                )
            # Save plots to session_state for PDF export

# ========================================================
# SECTION: ANALYSIS (detailed)
# ========================================================
elif section == "Analysis":
    st.markdown("### 📈 Detailed Analysis")
    if label_state:
        st.caption(f"Active dataset: **{label_state}**")

    if df_state is None or results is None:
        st.info("Run a MultiSig fit on the Overview page (sidebar → Run MultiSig fit) to populate this view.")
    else:
        tabs = st.tabs(
            [
                "Raw vs trimmed profile",
                "Model details",
                "Sigma / mass tables",
            ]
        )

        # Tab 1
        with tabs[0]:
            st.markdown("#### Raw vs trimmed SE profile")

            if pub_style:
                plt.style.use("default")
            else:
                plt.style.use("dark_background")

            r_trim = df_state["radius_cm"].to_numpy(float)
            J_trim = df_state["fringe"].to_numpy(float)

            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            fig_raw, ax_raw = plt.subplots()
            ax_raw.scatter(r_trim, J_trim, s=8, alpha=0.8, label="Trimmed data")
            ax_raw.set_xlabel("Radius (cm)")
            ax_raw.set_ylabel("Fringe")
            ax_raw.set_title("Trimmed SE profile used for MultiSig fit")
            ax_raw.legend()
            st.pyplot(fig_raw, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 2
        with tabs[1]:
            st.markdown("#### Model curves and J(r) reconstruction")

            r = df_state["radius_cm"].to_numpy(float)
            J = df_state["fringe"].to_numpy(float)
            J_pred = msf.predict_J(
                r,
                results["c_mean"],
                results["E_mean"],
                results["sigma_bins"],
                results["J_ref"],
                results["r_ref"],
            )

            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            fig_model, ax_model = plt.subplots()
            ax_model.scatter(r, J, s=6, alpha=0.7, label="Data")
            ax_model.plot(r, J_pred, "-", lw=2, label="MultiSig prediction")
            ax_model.set_xlabel("Radius (cm)")
            ax_model.set_ylabel("Fringe")
            ax_model.legend()
            st.pyplot(fig_model, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Tab 3
        with tabs[2]:
            st.markdown("#### Sigma and mass bin tables")

            coef_table = results.get("coef_table")
            if coef_table is None:
                st.info("No coefficient table available.")
            else:
                display_cols = [
                    c
                    for c in ["bin", "sigma_bin", "M_bin_gmol", "c_mean", "c_sd"]
                    if c in coef_table.columns
                ]
                st.dataframe(
                    coef_table[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Mass bins are derived using the σ→M mapping from your Methods chapter.")

# ========================================================
# SECTION: DATASETS
# ========================================================
elif section == "Datasets":
    st.markdown("### 📂 Datasets")
    st.caption("Manage uploaded datasets, inspect metadata and preview raw SE files.")

    filter_text = st.text_input(
        "Filter example datasets by name",
        placeholder="Type to filter Sim1b, IgG1, pullulan…",
        label_visibility="collapsed",
    )

    example_files = ["Sim1b.ip1", "IgG1.IP1", "pullulan.ip2"]
    filtered = [f for f in example_files if (not filter_text) or (filter_text.lower() in f.lower())]

    cols = st.columns(3)
    for col, name in zip(cols, filtered):
        with col:
            st.markdown(
                f"""
                <div class="dataset-card">
                    <div style="font-weight:600; margin-bottom:0.15rem;">📁 {name}</div>
                    <div style="font-size:0.78rem; color:#9ca3af;">
                        Use the sidebar “Example dataset” selector to load this file into the analysis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown("#### Active dataset preview ↪")
    if label_state is None or df_raw is None or meta is None:
        st.info("No active dataset loaded. Use the sidebar to upload or select a dataset.")
    else:
        st.caption(f"Currently loaded: **{label_state}**")

        meta_col, data_col = st.columns([1.2, 1.8])
        with meta_col:
            st.markdown('<span class="pill-tab">Metadata</span>', unsafe_allow_html=True)
            st.json(meta)

        with data_col:
            st.markdown('<span class="pill-tab">Data preview</span>', unsafe_allow_html=True)
            st.dataframe(df_raw.head(12), use_container_width=True)

        st.markdown("")
        st.markdown('<span class="pill-tab">Basic stats</span>', unsafe_allow_html=True)
        stats = df_raw.describe().T
        st.dataframe(stats, use_container_width=True)

# ========================================================
# SECTION: REPORTS
# ========================================================
elif section == "Reports":
    st.markdown("### 📄 Reports")
    st.caption("Quick summaries, validation, and export options for your current MultiSig fit.")

    if results is None or df_state is None or meta_state is None:
        st.info("No completed fit found. Run a fit on the Overview page first.")
    else:
        Mw_txt, Mn_txt, Mz_txt = extract_mass_metrics(results)
        mw_diag = build_mw_diagnostics(results)
        species_df = build_species_table(results)

        # Keep a single "reference" dict in session so Validation + Export can reuse it
        if "multisig_reference" not in st.session_state:
            st.session_state["multisig_reference"] = {"Mn_kDa": 0.0, "Mw_kDa": 0.0, "Mz_kDa": 0.0, "E": 0.0}
        if "validation_tol_pct" not in st.session_state:
            st.session_state["validation_tol_pct"] = 1.0000

        tabs = st.tabs(
            [
                "Summary",
                "Species & distribution",
                "Metadata & methods",
                "Validation",
                "Export",
            ]
        )

        # -------------------------
        # TAB 1: Summary
        # -------------------------
        with tabs[0]:
            st.markdown("#### Molecular-weight summary")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Number-average Mn", Mn_txt)
            with c2:
                st.metric("Weight-average Mw", Mw_txt)
            with c3:
                st.metric("Z-average Mz", Mz_txt)

            if mw_diag is not None:
                st.markdown("##### Precision across repeated fits")
                st.dataframe(mw_diag, use_container_width=True, hide_index=True)
                st.caption(
                    "95% confidence intervals assume approximate normality of the mass estimates "
                    "across repeated jittered fits."
                )
            else:
                st.info("No detailed precision table available for this fit.")

        # -------------------------
        # TAB 2: Species
        # -------------------------
        with tabs[1]:
            st.markdown("#### Dominant species / mass bins")

            if species_df is None:
                st.info("Coefficient table was not available – cannot build species summary.")
            else:
                st.dataframe(species_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Fractions are based on the fitted MultiSig coefficients. For a clean single-species "
                    "sample, you typically expect one dominant bin (≈90%+ of the signal)."
                )

        # -------------------------
        # TAB 3: Metadata & methods
        # -------------------------
        with tabs[2]:
            st.markdown("#### Dataset metadata")

            rpm = meta_state.get("rpm", "—")
            temp = meta_state.get("temperature_c", "—")
            rotor_pos = meta_state.get("rotor_position", "—")
            sample_name = meta_state.get("sample_name", "—")

            col_meta, col_fit = st.columns(2)

            with col_meta:
                st.markdown("**Sample & experiment**")
                st.write(f"- **Sample name:** `{sample_name}`")
                st.write(f"- **Dataset file:** `{label_state}`")
                st.write(f"- **Rotor position:** {rotor_pos}")
                st.write(f"- **Rotor speed:** {rpm} rpm")
                st.write(f"- **Temperature:** {temp} °C")

            with col_fit:
                st.markdown("**Fit configuration**")
                st.write(f"- **r_ref:** {fit_config.get('r_ref', '—')} cm")
                st.write(f"- **σ_ref (grid scale):** {fit_config.get('sigma_ref', '—')} 1/cm²")
                st.write(f"- **Smoothing λ:** {fit_config.get('lam_smooth', '—')}")
                st.write(f"- **Repeat fits:** {fit_config.get('num_runs', '—')}")
                st.write(f"- **v̄:** {fit_config.get('vbar_ml_per_g', '—')} mL/g")
                st.write(f"- **ρ:** {fit_config.get('rho_g_per_ml', '—')} g/mL")
                st.write(
                    f"- **Radius range (trimmed):** "
                    f"{fit_config.get('radius_min_cm', '—')} – "
                    f"{fit_config.get('radius_max_cm', '—')} cm"
                )
                st.write(f"- **Points after trimming:** {fit_config.get('n_points', '—')}")
                st.write(f"- **J(r) reference mode:** {fit_config.get('Jref_mode', 'nearest')}")
                st.write(f"- **Optimiser:** {fit_config.get('solver', 'trf')}")
                st.write(f"- **Jitter (%):** {fit_config.get('jitter_pct', '—')}")
                st.write(f"- **Bounds enabled:** {fit_config.get('use_bounds', '—')}")


        # -------------------------
        # TAB 4: Validation
        # -------------------------
        with tabs[3]:
            st.markdown("#### Numerical comparison with MULTISIG")
            st.caption("Enter legacy MULTISIG values (in Sigma units) from ProFit output (mean of 10 repeats).")
            
            # --- SAFETY LOCK: Hardcoded RPMs for Validation Datasets ---
            # This ensures physics conversions remain accurate even if metadata parsing fails
            VALIDATION_RPM = {
                "Sim1b.ip1": 17000,
                "IgG1.IP1": 13000,
                "pullulan.ip2": 5000,
            }

            # --- STEP 2: Centralise Sigma -> M conversion (using CGS units for AUC) ---
            def sigma_to_mw_kda(sigma, vbar, rho, rpm, temp_c):
                """
                Canonical conversion function: σ -> M (kDa)
                M = (σ * R * T) / ((1 - vbar*rho) * omega^2)
                """
                if rpm <= 0:
                    raise ValueError("RPM must be > 0 for sigma-to-mass conversion.")
                
                # Physical constants
                # R must be in erg/(mol K) to match CGS units typical for AUC Sigma values (~1-10)
                R_erg = 8.314462618e7 
                
                temp_k = temp_c + 273.15
                omega = 2 * np.pi * (rpm / 60.0) # radians per second
                
                buoyancy = (1.0 - vbar * rho)
                if abs(buoyancy) < 1e-9:
                    return 0.0
                
                # M (g/mol) = (sigma * R * T) / ((1 - vbar*rho) * omega^2)
                M_gmol = (sigma * R_erg * temp_k) / (buoyancy * omega**2)
                
                return M_gmol / 1000.0  # Convert to kDa

            
                # --- STEP 2A: Resolve RPM explicitly for validation ---
            dataset_name = label_state  # e.g. "IgG1.IP1"
            
            if dataset_name in VALIDATION_RPM:
                rpm = VALIDATION_RPM[dataset_name]
            else:
                st.error(
                    f"No validation RPM defined for dataset '{dataset_name}'. "
                    "Cannot perform sigma-to-mass conversion."
                )
                st.stop()
            
            # Other physical parameters (these ARE allowed from metadata/config)
            temp_c = float(meta_state.get("temperature_c", 20.0))
            vbar = float(fit_config.get("vbar_ml_per_g", 0.73))
            rho = float(fit_config.get("rho_g_per_ml", 1.0))

            # --- STEP 1: Change legacy input fields (UI) ---
            # Using session state keys ensures inputs persist when switching tabs
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("**Number Average (N)**")
                leg_sig_n = st.number_input("Legacy σ_N", value=0.0, format="%.6f", key="leg_sig_n")
                leg_sig_n_sd = st.number_input("Legacy σ_N SD (±)", value=0.0, format="%.6f", key="leg_sig_n_sd")
                
            with c2:
                st.markdown("**Weight Average (W)**")
                leg_sig_w = st.number_input("Legacy σ_W", value=0.0, format="%.6f", key="leg_sig_w")
                leg_sig_w_sd = st.number_input("Legacy σ_W SD (±)", value=0.0, format="%.6f", key="leg_sig_w_sd")
                
            with c3:
                st.markdown("**Z-Average (Z) & Baseline**")
                leg_sig_z = st.number_input("Legacy σ_Z", value=0.0, format="%.6f", key="leg_sig_z")
                leg_sig_z_sd = st.number_input("Legacy σ_Z SD (±)", value=0.0, format="%.6f", key="leg_sig_z_sd")
                leg_E = st.number_input("Legacy Baseline E (fringes)", value=0.0, format="%.6f", key="leg_E")
                
            st.markdown("---")
            
            # Tolerance input
            tol = st.number_input(
                "Pass tolerance (% diff)",
                value=float(st.session_state.get("validation_tol_pct", 1.0)),
                format="%.4f",
                key="tol_pct_val_step3"
            )

            # --- STEP 3 & 4: Convert legacy σ inside validation block ---
            # Convert Means
            legacy_Mn_kDa = sigma_to_mw_kda(leg_sig_n, vbar, rho, rpm, temp_c)
            legacy_Mw_kDa = sigma_to_mw_kda(leg_sig_w, vbar, rho, rpm, temp_c)
            legacy_Mz_kDa = sigma_to_mw_kda(leg_sig_z, vbar, rho, rpm, temp_c)
            
            # Convert SDs (Linear scaling: M = k*sigma, so SD_M = k*SD_sigma)
            # We reuse the function passing the SD as the 'sigma' value to get the scale factor
            legacy_Mn_sd_kDa = sigma_to_mw_kda(leg_sig_n_sd, vbar, rho, rpm, temp_c)
            legacy_Mw_sd_kDa = sigma_to_mw_kda(leg_sig_w_sd, vbar, rho, rpm, temp_c)
            legacy_Mz_sd_kDa = sigma_to_mw_kda(leg_sig_z_sd, vbar, rho, rpm, temp_c)

            # Store in session state for the table builder
            st.session_state["multisig_reference"] = {
                "Mn_kDa": legacy_Mn_kDa,
                "Mw_kDa": legacy_Mw_kDa,
                "Mz_kDa": legacy_Mz_kDa,
                "E": leg_E
            }
            st.session_state["validation_tol_pct"] = tol

            # Display the converted legacy values for the user (Feedback)
            if rpm > 0:
                st.info(
                    f"**Legacy Values Converted to kDa (using app parameters):** \n"
                    f"• **Mn:** {legacy_Mn_kDa:.4f} ± {legacy_Mn_sd_kDa:.4f} kDa  \n"
                    f"• **Mw:** {legacy_Mw_kDa:.4f} ± {legacy_Mw_sd_kDa:.4f} kDa  \n"
                    f"• **Mz:** {legacy_Mz_kDa:.4f} ± {legacy_Mz_sd_kDa:.4f} kDa"
                )
            else:
                st.warning("⚠️ Rotor speed is 0 or missing. Cannot convert Sigma to kDa.")

            # Build Comparison Table
            val_df = build_numerical_validation_table(results, st.session_state["multisig_reference"])

            if val_df is None:
                st.info("Run a fit first to generate the comparison table.")
            else:
                # Format table for display
                show = val_df.copy()
                for col in ["This GUI", "MULTISIG", "Abs. diff", "% diff"]:
                    show[col] = show[col].map(lambda v: f"{v:.4f}" if (isinstance(v, (int, float, np.floating)) and np.isfinite(v)) else "—")
                
                st.dataframe(show, use_container_width=True, hide_index=True)

                # Validation status
                pct = pd.to_numeric(val_df["% diff"], errors="coerce").abs()
                n_tot = int(pct.notna().sum())
                n_ok = int((pct <= tol).sum()) if n_tot > 0 else 0
                
                if n_ok == n_tot and n_tot > 0:
                    st.success(f"✅ VALIDATION PASSED: All {n_ok}/{n_tot} metrics within ±{tol:.2f}%")
                elif n_tot > 0:
                    st.warning(f"⚠️ VALIDATION INCOMPLETE: {n_ok}/{n_tot} metrics within ±{tol:.2f}%")

            st.divider()
            
            # --- Chapter 4 Export Section (Auto-filled) ---
            st.subheader("Chapter 4: Export validation table")
            
            if val_df is not None:
                # Reuse the val_df we just built, but format specifically for the CSV export
                # We can reconstruct the exact format you wanted for Chapter 4
                
                dataset_name = label_state
                
                # We already have the 'python' values in 'val_df' column "This GUI"
                # We already have 'legacy' values in 'val_df' column "MULTISIG"
                
                # Let's rebuild the clean dataframe for CSV export to match your thesis format
                # Python Metrics
                if results is None or "summary" not in results:
                    st.info("Run a fit to enable Chapter 4 export.")
                    st.stop()
                summary_df = results["summary"].set_index("metric")
                Mn_kDa_py = summary_df.loc["Mn_g_per_mol", "mean"] / 1000.0
                Mw_kDa_py = summary_df.loc["Mw_g_per_mol", "mean"] / 1000.0
                Mz_kDa_py = summary_df.loc["Mz_g_per_mol", "mean"] / 1000.0
                E_py = summary_df.loc["baseline_E", "mean"]

                final_df = pd.DataFrame([
                    {
                        "Dataset": dataset_name, 
                        "Metric": "Mn (kDa)", 
                        "Python MULTISIG": Mn_kDa_py, 
                        "Legacy MULTISIG": legacy_Mn_kDa,
                        "Abs. diff": Mn_kDa_py - legacy_Mn_kDa,
                        "% diff": ((Mn_kDa_py - legacy_Mn_kDa) / legacy_Mn_kDa) * 100
                                  if legacy_Mn_kDa != 0 else np.nan
                    },
                    {
                        "Dataset": dataset_name, 
                        "Metric": "Mw (kDa)", 
                        "Python MULTISIG": Mw_kDa_py, 
                        "Legacy MULTISIG": legacy_Mw_kDa,
                        "Abs. diff": Mw_kDa_py - legacy_Mw_kDa,
                        "% diff": ((Mw_kDa_py - legacy_Mw_kDa) / legacy_Mw_kDa) * 100
                                  if legacy_Mw_kDa != 0 else np.nan
                    },
                    {
                        "Dataset": dataset_name, 
                        "Metric": "Mz (kDa)", 
                        "Python MULTISIG": Mz_kDa_py, 
                        "Legacy MULTISIG": legacy_Mz_kDa,
                        "Abs. diff": Mz_kDa_py - legacy_Mz_kDa,
                        "% diff": ((Mz_kDa_py - legacy_Mz_kDa) / legacy_Mz_kDa) * 100
                                  if legacy_Mz_kDa != 0 else np.nan
                    },
                    {
                        "Dataset": dataset_name, 
                        "Metric": "E", 
                        "Python MULTISIG": E_py, 
                        "Legacy MULTISIG": leg_E,
                        "Abs. diff": E_py - leg_E,
                        "% diff": ((E_py - leg_E) / leg_E) * 100
                                  if leg_E != 0 else np.nan
                    },
                ])

                st.dataframe(
                    final_df.style.format({
                        "Python MULTISIG": "{:.4f}",
                        "Legacy MULTISIG": "{:.4f}",
                        "Abs. diff": "{:.4f}",
                        "% diff": "{:.4f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        

                st.caption(
                        "Legacy MULTISIG values were supplied in σ-space and converted to molecular "
                        "weight using the sedimentation-equilibrium relationship prior to comparison."
                )
                st.download_button(
                    "Download Chapter 4 validation table (CSV)",
                    data=final_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{dataset_name}_chapter4_validation.csv",
                    mime="text/csv",
                )
            else:
                st.info("Run fit to enable export.")
                
        # -------------------------
        # TAB 5: Export
        # -------------------------
        with tabs[4]:
            st.markdown("#### Export quick report")

            summary_df = results["summary"]
            rpm = meta_state.get("rpm", "—")
            temp = meta_state.get("temperature_c", "—")
            sample_name = meta_state.get("sample_name", "—")

            # Validation DF for export (optional)
            multisig_reference = st.session_state.get("multisig_reference", {"Mn_kDa": 0.0, "Mw_kDa": 0.0, "Mz_kDa": 0.0, "E": 0.0})
            tol = float(st.session_state.get("validation_tol_pct", 1.0))
            val_df = build_numerical_validation_table(results, multisig_reference)

            # Build markdown tables (defensive: handle None)
            mw_table_md = mw_diag.to_markdown(index=False) if mw_diag is not None else ""
            species_table_md = species_df.to_markdown(index=False) if species_df is not None else ""
            val_table_md = val_df.to_markdown(index=False) if val_df is not None else ""

            # Fit config lines (include solver fields)
            fit_cfg_lines = []
            for key in [
                "r_ref",
                "sigma_ref",
                "lam_smooth",
                "num_runs",
                "vbar_ml_per_g",
                "rho_g_per_ml",
                "radius_min_cm",
                "radius_max_cm",
                "n_points",
                "Jref_mode",
                "solver",
                "jitter_pct",
                "use_bounds",
            ]:
                if key in fit_config:
                    fit_cfg_lines.append(f"- **{key}:** {fit_config[key]}")

            # ---------- MARKDOWN EXPORT ----------
            md_lines = [
                "# MultiSig SE quick report",
                "",
                f"**Sample name:** `{sample_name}`",
                f"**Dataset file:** `{label_state}`",
                f"**Rotor speed:** {rpm} rpm",
                f"**Temperature:** {temp} °C",
                "",
                "## Molecular-weight summary (g/mol)",
                "",
                summary_df.to_markdown(index=False),
            ]

            if mw_table_md:
                md_lines.extend(["", "## Precision across repeated fits (kDa)", "", mw_table_md])

            if species_table_md:
                md_lines.extend(["", "## Dominant species (mass bins)", "", species_table_md])

            if val_table_md:
                md_lines.extend(["", "## Validation vs MULTISIG", "", val_table_md])
                md_lines.extend(["", f"**Pass tolerance:** ±{tol:.4f}%"])

            if fit_cfg_lines:
                md_lines.extend(["", "## Fit configuration (for reproducibility)", "", *fit_cfg_lines])

            md_lines.extend(
                [
                    "",
                    "## Notes",
                    "- Auto-generated summary from the Streamlit prototype.",
                    "- For publication, check trimming ranges, λ, and v̄/ρ against your Methods chapter.",
                ]
            )

            report_md = "\n".join(md_lines).encode("utf-8")
            st.download_button(
                "Download quick report (.md)",
                data=report_md,
                file_name="multisig_quick_report.md",
                mime="text/markdown",
            )

            # ---------- TXT EXPORT ----------
            txt_lines = [
                f"Sample name: {sample_name}",
                f"Dataset file: {label_state}",
                f"Rotor speed: {rpm} rpm",
                f"Temperature: {temp} °C",
                "",
                "Molecular-weight summary (g/mol)",
                summary_df.to_string(index=False),
            ]
            if val_df is not None:
                txt_lines.extend(
                    [
                        "",
                        "Validation vs MULTISIG",
                        val_df.to_string(index=False),
                        f"Pass tolerance: ±{tol:.4f}%",
                    ]
                )
            report_txt = "\n".join(txt_lines).encode("utf-8")
            st.download_button(
                "Download summary (.txt)",
                data=report_txt,
                file_name="multisig_summary.txt",
                mime="text/plain",
            )

            # ---------- PDF EXPORT WITH FIGURES + VALIDATION ----------
            if REPORTLAB_AVAILABLE:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet

                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                styleH = styles["Heading2"]
                styleN = styles["BodyText"]

                story = []
                story.append(Paragraph("MultiSig SE quick report", styles["Heading1"]))
                story.append(Spacer(1, 12))

                # Metadata
                story.append(Paragraph(f"Sample name: {sample_name}", styleN))
                story.append(Paragraph(f"Dataset file: {label_state}", styleN))
                story.append(Paragraph(f"Rotor speed: {rpm} rpm", styleN))
                story.append(Paragraph(f"Temperature: {temp} °C", styleN))
                story.append(Spacer(1, 12))

                # MW summary
                story.append(Paragraph("Molecular-weight summary (g/mol)", styleH))
                story.append(Spacer(1, 6))
                story.append(
                    Paragraph(
                        summary_df.to_string(index=False).replace(" ", "&nbsp;").replace("\n", "<br/>"),
                        styleN,
                    )
                )
                story.append(Spacer(1, 12))

                # Precision table
                if mw_diag is not None:
                    story.append(Paragraph("Precision across repeated fits (kDa)", styleH))
                    story.append(Spacer(1, 6))
                    story.append(
                        Paragraph(
                            mw_diag.to_string(index=False).replace(" ", "&nbsp;").replace("\n", "<br/>"),
                            styleN,
                        )
                    )
                    story.append(Spacer(1, 12))

                # Species table
                if species_df is not None:
                    story.append(Paragraph("Dominant species (mass bins)", styleH))
                    story.append(Spacer(1, 6))
                    story.append(
                        Paragraph(
                            species_df.to_string(index=False).replace(" ", "&nbsp;").replace("\n", "<br/>"),
                            styleN,
                        )
                    )
                    story.append(Spacer(1, 12))

                # Validation table (NEW)
                if val_df is not None:
                    story.append(Paragraph("Validation vs MULTISIG", styleH))
                    story.append(Spacer(1, 6))
                    story.append(
                        Paragraph(
                            val_df.to_string(index=False).replace(" ", "&nbsp;").replace("\n", "<br/>"),
                            styleN,
                        )
                    )
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(f"Pass tolerance: ±{tol:.4f}%", styleN))
                    story.append(Spacer(1, 12))

                # Fit config
                if fit_cfg_lines:
                    story.append(Paragraph("Fit configuration", styleH))
                    story.append(Spacer(1, 6))
                    for line in fit_cfg_lines:
                        story.append(Paragraph(line, styleN))
                    story.append(Spacer(1, 12))

                # Figures
                story.append(Paragraph("Figures", styleH))
                story.append(Spacer(1, 12))

                if "pdf_overlay" in st.session_state:
                    story.append(Paragraph("Figure 1: Data vs MultiSig fit", styleN))
                    story.append(Spacer(1, 6))
                    story.append(Image(st.session_state["pdf_overlay"], width=480, height=320))
                    story.append(Spacer(1, 18))

                if "pdf_coeffs" in st.session_state:
                    story.append(Paragraph("Figure 2: Coefficient distribution", styleN))
                    story.append(Spacer(1, 6))
                    story.append(Image(st.session_state["pdf_coeffs"], width=480, height=320))
                    story.append(Spacer(1, 18))

                if "pdf_residuals" in st.session_state:
                    story.append(Paragraph("Figure 3: Residuals vs radius", styleN))
                    story.append(Spacer(1, 6))
                    story.append(Image(st.session_state["pdf_residuals"], width=480, height=320))
                    story.append(Spacer(1, 18))

                doc.build(story)
                pdf_data = pdf_buffer.getvalue()

                st.download_button(
                    "Download quick report (.pdf)",
                    data=pdf_data,
                    file_name="multisig_quick_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.info("Install reportlab to enable PDF export: `pip install reportlab`.")
# ========================================================
# SECTION: SETTINGS
# ========================================================
elif section == "Settings":
    st.markdown("### ⚙️ Settings")
    st.caption("Prototype settings for default ranges and plotting preferences.")

    st.markdown("#### Default analysis preferences")
    with st.form("settings_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            default_lambda = st.number_input(
                "Preferred smoothing λ",
                value=0.02,
                step=0.01,
            )
            default_downsample = st.number_input(
                "Default downsampling factor",
                value=1,
                min_value=1,
                max_value=10,
                step=1,
            )
        with col_b:
            default_vbar = st.number_input(
                "Default v̄ (mL/g)",
                value=0.73,
            )
            default_rho = st.number_input(
                "Default ρ (g/mL)",
                value=1.0,
            )

        pub_default = st.checkbox(
            "Use publication-style plots by default",
            value=True,
        )

        submitted = st.form_submit_button("Save preferences")
    if submitted:
        st.success(
            "Preferences captured for this session. Persistent storage is not implemented in this prototype yet."
        )

    st.markdown("---")
    st.markdown(
        "In a full implementation, these settings would be stored per-user and applied to the sidebar "
        "defaults each time the app is opened."
    )

# ========================================================
# SECTION: HELP
# ========================================================
elif section == "Help":
    st.markdown("### ❓ Help and about")
    st.markdown(
        dedent(
            """
            This Streamlit app is a **prototype MultiSig-style AUC workbench**.

            **What you can do right now**
            - Upload or select example SE profiles (`.ip1` / `.ip2`).
            - Trim the radius range and optionally downsample.
            - Run a MultiSig-style fit with repeated jittered starts.
            - Inspect:
              - Overlay plots of data vs fit
              - σ-distribution / mass-bin coefficients
              - Residuals vs radius
              - Basic metadata and data-statistics per dataset
              - A quick, exportable markdown report

            **Planned extensions**
            - Multiple-solute / multi-species reporting views
            - Per-user saved settings and analysis presets
            - Richer report export (Word/PDF) with embedded figures
            - Advanced trimming diagnostics and automatic suggestions

            If something looks suspicious in a fit, check:
            - Trimming range (radius slider in the sidebar)
            - Smoothing λ
            - v̄ and ρ values for your particular system
            - σ_ref mainly controls the scale of the σ-bin grid. It has units of 1/cm²,
  but in practice behaves like a tuning parameter for how coarse or fine the
  MultiSig basis is.
            """
        )
    )