"""
Tourism Experience Analytics — Streamlit Application
app.py

Pages:
  🏠  Home & Overview
  📊  EDA & Visualizations
  🤖  Regression (Predict Rating)
  🗂️  Classification (Predict Visit Mode)
  🎯  Recommendation Engine
  📈  Model Leaderboard
  💡  Business Insights

Run:
    streamlit run app.py
"""

import sys
import os
import warnings
import pickle
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ARTIFACTS_PATH = "models/tea_artifacts.pkl"

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLES — Advanced Dark Luxury UI
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Root Variables ────────────────────────────────── */
:root {
    --bg-primary:    #080c14;
    --bg-secondary:  #0d1320;
    --bg-card:       #111827;
    --bg-glass:      rgba(255,255,255,0.04);
    --accent-cyan:   #00d4ff;
    --accent-violet: #7c3aed;
    --accent-pink:   #f472b6;
    --accent-amber:  #f59e0b;
    --accent-green:  #10b981;
    --text-primary:  #f1f5f9;
    --text-muted:    #64748b;
    --text-dim:      #334155;
    --border:        rgba(255,255,255,0.06);
    --border-glow:   rgba(0,212,255,0.25);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
    --shadow-glow:   0 0 40px rgba(0,212,255,0.08);
    --shadow-card:   0 4px 32px rgba(0,0,0,0.4);
}

/* ── Base Reset ─────────────────────────────────────── */
* { font-family: 'Outfit', sans-serif !important; box-sizing: border-box; }

/* ── App Background ─────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(124,58,237,0.07) 0%, transparent 60%) !important;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    padding: 10px 14px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--bg-glass) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] span:first-child { display: none !important; }

/* ── Main Content Padding ───────────────────────────── */
.main .block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* ── Page Title ─────────────────────────────────────── */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.25rem !important;
}
h2 { font-weight: 700 !important; color: var(--text-primary) !important; }
h3 { font-weight: 600 !important; color: var(--text-primary) !important; }
p, li, span, label { color: #94a3b8 !important; }

/* ── Metric Cards ───────────────────────────────────── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(0,212,255,0.2); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
    opacity: 0.7;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 1.4rem;
    opacity: 0.4;
}

/* ── Section Headers ────────────────────────────────── */
.section-header {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: var(--accent-cyan) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    margin: 28px 0 14px 0 !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Divider ────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Streamlit Metrics ──────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

/* ── Plotly Charts ──────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 8px !important;
}

/* ── DataFrames / Tables ────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}
.stDataFrame { background: var(--bg-card) !important; }

/* ── Tabs ───────────────────────────────────────────── */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border) !important; }
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 10px 18px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover { color: var(--text-primary) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* ── Buttons ────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.25) !important;
}

/* ── Sliders ────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
}

/* ── Select / Input ─────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
}

/* ── Expander ───────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    padding: 14px 18px !important;
}

/* ── Info / Warning / Success boxes ────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
}

/* ── Caption / Small text ───────────────────────────── */
[data-testid="stCaptionContainer"] p { font-size: 0.78rem !important; color: var(--text-muted) !important; }

/* ── Sidebar logo block ─────────────────────────────── */
.sidebar-brand {
    padding: 28px 20px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.sidebar-brand-icon { font-size: 2.2rem; margin-bottom: 8px; }
.sidebar-brand-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
    line-height: 1.3;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted) !important;
    margin-top: 2px;
    font-weight: 400;
}

/* ── Tag badge ──────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-cyan  { background: rgba(0,212,255,0.12);  color: var(--accent-cyan) !important;   border: 1px solid rgba(0,212,255,0.25); }
.badge-violet{ background: rgba(124,58,237,0.12); color: #a78bfa !important; border: 1px solid rgba(124,58,237,0.25); }
.badge-green { background: rgba(16,185,129,0.12); color: var(--accent-green) !important;  border: 1px solid rgba(16,185,129,0.25); }
.badge-amber { background: rgba(245,158,11,0.12); color: var(--accent-amber) !important;  border: 1px solid rgba(245,158,11,0.25); }

/* ── Insight cards ──────────────────────────────────── */
.insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin-bottom: 12px;
    border-left: 3px solid var(--accent-cyan);
    transition: all 0.2s ease;
}
.insight-card:hover { border-left-color: var(--accent-violet); transform: translateX(4px); }
.insight-title { font-weight: 700; font-size: 0.95rem; color: var(--text-primary) !important; margin-bottom: 6px; }
.insight-text  { font-size: 0.875rem; color: #94a3b8 !important; line-height: 1.6; }

/* ── Prediction result ──────────────────────────────── */
.predict-result {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}
.predict-value { font-size: 3rem; font-weight: 800; color: var(--accent-cyan) !important; letter-spacing: -0.04em; }
.predict-label { font-size: 0.8rem; color: var(--text-muted) !important; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* ── Page title area ────────────────────────────────── */
.page-hero {
    padding: 8px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.page-hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}
.page-hero-sub { font-size: 0.9rem; color: var(--text-muted) !important; max-width: 600px; line-height: 1.6; }

/* ── Code block ─────────────────────────────────────── */
code, pre {
    font-family: 'Space Mono', monospace !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--accent-cyan) !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)

def kpi(col, value: str, label: str, icon: str):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#94a3b8", size=12),
    title_font=dict(family="Outfit", color="#f1f5f9", size=14, weight=600),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", tickfont=dict(color="#64748b")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(t=48, b=12, l=12, r=12),
    coloraxis_colorbar=dict(tickfont=dict(color="#64748b"), title_font=dict(color="#94a3b8")),
)

PALETTE = ["#00d4ff","#7c3aed","#f472b6","#f59e0b","#10b981","#3b82f6","#ef4444","#8b5cf6"]

def apply_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS — auto-retrain on Cloud if pkl missing
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Training models on first run — this takes ~2 mins …")
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        subprocess.run(
            [sys.executable, "train.py", "--data", "./data"],
            check=True
        )
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


arts = load_artifacts()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🌍</div>
        <div class="sidebar-brand-name">Tourism Experience<br>Analytics</div>
        <div class="sidebar-brand-sub">ML · EDA · Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Home & Overview",
            "📊  EDA & Visualizations",
            "🤖  Regression (Rating)",
            "🗂️  Classification (Visit Mode)",
            "🎯  Recommendation Engine",
            "📈  Model Leaderboard",
            "💡  Business Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 0 4px;">
        <div style="font-size:0.7rem; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Stack</div>
        <span class="badge badge-cyan">Scikit-learn</span>&nbsp;
        <span class="badge badge-violet">XGBoost</span>&nbsp;
        <span class="badge badge-green">LightGBM</span>
        <br><br>
        <span class="badge badge-amber">Streamlit</span>&nbsp;
        <span class="badge badge-cyan">Plotly</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#1e293b; text-align:center;">v1.0 · Tourism Analytics</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# GUARD — artifacts not yet built
# ─────────────────────────────────────────────────────────────────
if arts is None:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size:4rem; margin-bottom:16px;">⚠️</div>
        <div style="font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-bottom:8px;">Artifacts Not Found</div>
        <div style="color:#64748b; margin-bottom:24px;">Run the training script to generate the model artifacts.</div>
    </div>
    """, unsafe_allow_html=True)
    st.code("python train.py --data ./data", language="bash")
    st.stop()


# ── Unpack artifacts ──────────────────────────────────────────────
df                   = arts["df"]
item                 = arts["item"]
attr_type            = arts["attr_type"]
le_mode              = arts["le_mode"]
le_dict              = arts["le_dict"]
REG_FEATURES         = arts["REG_FEATURES"]
CLF_FEATURES         = arts["CLF_FEATURES"]
reg_results_df       = arts["reg_results_df"]
clf_results_df       = arts["clf_results_df"]
best_reg_model       = arts["best_reg_model"]
best_clf_model       = arts["best_clf_model"]
scaler_r             = arts["scaler_r"]
scaler_c             = arts["scaler_c"]
predicted_ratings_df = arts["predicted_ratings_df"]
user_item_matrix     = arts["user_item_matrix"]
cos_sim_df           = arts["cos_sim_df"]
rmse_cf              = arts["rmse_cf"]
mae_cf               = arts["mae_cf"]
SCALED_REGS          = arts.get("SCALED_REGS", {"Linear Regression", "Ridge Regression", "Lasso Regression"})
SCALED_CLFS          = arts.get("SCALED_CLFS", {"Logistic Regression", "K-Nearest Neighbors"})


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">🌍 Tourism Experience Analytics</div>
        <div class="page-hero-sub">
            Classification · Prediction · Recommendation System —
            Explore user travel patterns, predict visit modes, and get personalised attraction suggestions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(df):,}",                    "Total Transactions",  "💳")
    kpi(c2, f"{df['UserId'].nunique():,}",      "Unique Users",        "👤")
    kpi(c3, f"{df['AttractionId'].nunique():,}","Unique Attractions",  "📍")
    kpi(c4, f"{df['Rating'].mean():.2f} ★",    "Avg Rating",          "⭐")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        section("Visit Mode Distribution")
        vm = df["VisitMode"].value_counts().reset_index()
        vm.columns = ["VisitMode", "Count"]
        fig = px.pie(vm, names="VisitMode", values="Count",
                     color_discrete_sequence=PALETTE, hole=0.52)
        fig.update_traces(textinfo="label+percent", textfont_size=11,
                          marker=dict(line=dict(color="#080c14", width=2)))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Rating Distribution")
        fig2 = px.histogram(df, x="Rating", nbins=10,
                            color_discrete_sequence=["#00d4ff"])
        fig2.update_traces(marker_line_color="#080c14", marker_line_width=1.5)
        apply_theme(fig2)
        fig2.update_layout(bargap=0.12, xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Objectives ───────────────────────────────────────────────
    section("Project Objectives")
    o1, o2, o3 = st.columns(3)
    for col, icon, title, desc, color in [
        (o1, "🔢", "Regression",       "Predict the rating a user will give an attraction based on demographics & visit details.", "#00d4ff"),
        (o2, "🏷️", "Classification",  "Predict the visit mode (Family, Business, Couples…) from user profile & history.",         "#7c3aed"),
        (o3, "✨", "Recommendation",    "Suggest personalised attractions via Collaborative (SVD) and Content-Based filtering.",    "#10b981"),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid {color}; padding: 20px;">
            <div style="font-size:1.6rem; margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700; font-size:0.95rem; color:#f1f5f9; margin-bottom:6px;">{title}</div>
            <div style="font-size:0.82rem; color:#64748b; line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Dataset quick summary ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Dataset Quick Summary")
    summary = pd.DataFrame({
        "Table": ["Transaction", "User", "City", "Item", "Mode", "Type", "Country", "Region", "Continent"],
        "Description": [
            "User visits to attractions with ratings & visit mode",
            "User demographics: continent, region, country, city",
            "City names and country mapping",
            "Attraction names, types, locations, and addresses",
            "Visit mode labels (Business, Family, Couples…)",
            "Attraction type categories (Beach, Museum, Park…)",
            "Country names and region mapping",
            "Region names and continent mapping",
            "Continent names",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Visualizations":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">📊 Exploratory Data Analysis</div>
        <div class="page-hero-sub">Understand distributions, correlations, and relationships across users, attractions, and visit behaviour.</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["👤  Users", "📍  Attractions", "📅  Temporal", "🌡️  Correlations"])

    with tab1:
        if "Continent" in df.columns:
            section("Users by Continent")
            cont = df.drop_duplicates("UserId")["Continent"].value_counts().reset_index()
            cont.columns = ["Continent", "Users"]
            fig = px.bar(cont, x="Continent", y="Users", color="Continent",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(marker_line_width=0)
            apply_theme(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "Country" in df.columns:
            section("Top 15 User Countries")
            top_c = df.drop_duplicates("UserId")["Country"].value_counts().head(15).reset_index()
            top_c.columns = ["Country", "Users"]
            fig2 = px.bar(top_c, x="Users", y="Country", orientation="h",
                          color="Users", color_continuous_scale=[[0,"#1e293b"],[1,"#00d4ff"]])
            fig2.update_traces(marker_line_width=0)
            apply_theme(fig2)
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

        section("Rating Distribution by Visit Mode")
        fig3 = px.box(df, x="VisitMode", y="Rating", color="VisitMode",
                      color_discrete_sequence=PALETTE)
        apply_theme(fig3)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        if "Attraction" in df.columns:
            section("Top 10 Most Visited Attractions")
            top_a = df["Attraction"].value_counts().head(10).reset_index()
            top_a.columns = ["Attraction", "Visits"]
            fig = px.bar(top_a, x="Visits", y="Attraction", orientation="h",
                         color="Visits", color_continuous_scale=[[0,"#1e293b"],[1,"#7c3aed"]])
            fig.update_traces(marker_line_width=0)
            apply_theme(fig)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        if "AttractionType" in df.columns:
            section("Attraction Types")
            tc = df["AttractionType"].value_counts().head(15).reset_index()
            tc.columns = ["AttractionType", "Count"]
            fig2 = px.bar(tc, x="AttractionType", y="Count",
                          color="Count", color_continuous_scale=[[0,"#1e293b"],[1,"#00d4ff"]])
            fig2.update_traces(marker_line_width=0)
            apply_theme(fig2)
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

            section("Average Rating per Attraction Type")
            avg_type = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False).head(15).reset_index()
            avg_type.columns = ["AttractionType", "AvgRating"]
            fig3 = px.bar(avg_type, x="AvgRating", y="AttractionType", orientation="h",
                          color="AvgRating", color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]])
            fig3.update_traces(marker_line_width=0)
            apply_theme(fig3)
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        section("Monthly Visit Trend")
        monthly = df.groupby("VisitMonth").size().reset_index(name="Visits")
        monthly["Month"] = monthly["VisitMonth"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
        })
        fig = px.area(monthly, x="Month", y="Visits",
                      labels={"Visits": "Number of Visits"},
                      color_discrete_sequence=["#00d4ff"])
        fig.update_traces(line_width=2.5, fillcolor="rgba(0,212,255,0.07)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        if "Season" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                section("Visits by Season")
                season_counts = df["Season"].value_counts().reset_index()
                season_counts.columns = ["Season", "Visits"]
                fig2 = px.pie(season_counts, names="Season", values="Visits",
                              color_discrete_sequence=PALETTE, hole=0.5)
                fig2.update_traces(marker=dict(line=dict(color="#080c14", width=2)))
                apply_theme(fig2)
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                section("Visit Mode × Season Heatmap")
                hm = df.groupby(["VisitMode", "Season"]).size().unstack(fill_value=0)
                fig3 = px.imshow(hm, text_auto=True,
                                 color_continuous_scale=[[0,"#0d1320"],[0.5,"#7c3aed"],[1,"#00d4ff"]],
                                 aspect="auto")
                apply_theme(fig3)
                st.plotly_chart(fig3, use_container_width=True)

        if "VisitYear" in df.columns:
            section("Yearly Visit Trend")
            yearly = df.groupby("VisitYear").size().reset_index(name="Visits")
            fig4 = px.bar(yearly, x="VisitYear", y="Visits",
                          color="Visits", color_continuous_scale=[[0,"#1e293b"],[1,"#3b82f6"]])
            fig4.update_traces(marker_line_width=0)
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        section("Correlation Matrix (Numeric Features)")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f",
                        color_continuous_scale=[[0,"#ef4444"],[0.5,"#0d1320"],[1,"#00d4ff"]],
                        aspect="auto", zmin=-1, zmax=1)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        if "Continent" in df.columns:
            section("Average Rating by Continent")
            cont_rating = df.groupby("Continent")["Rating"].mean().reset_index()
            cont_rating.columns = ["Continent", "AvgRating"]
            fig2 = px.bar(cont_rating, x="Continent", y="AvgRating",
                          color="AvgRating", color_continuous_scale=[[0,"#1e293b"],[1,"#10b981"]])
            fig2.update_traces(marker_line_width=0)
            apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — REGRESSION
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  Regression (Rating)":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">🔢 Regression — Predict Rating</div>
        <div class="page-hero-sub">Predict the rating (1–5) a user would give an attraction based on visit details, user demographics, and attraction characteristics.</div>
    </div>
    """, unsafe_allow_html=True)

    best_r = reg_results_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_r["Model"])
    c2.metric("R²",         f"{best_r['R²']:.4f}")
    c3.metric("RMSE",       f"{best_r['RMSE']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    section("🔮 Live Rating Prediction")
    st.caption("Adjust the sliders and click Predict to see the model's output.")

    col1, col2 = st.columns(2)
    with col1:
        visit_year  = st.slider("Visit Year",  2010, 2025, 2022)
        visit_month = st.slider("Visit Month", 1, 12, 6)
        user_avg    = st.slider("User Avg Rating",       1.0, 5.0, 3.5, 0.1)
        attr_avg    = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0, 0.1)
    with col2:
        attr_visits = st.number_input("Attraction Visit Count", 1, 10000, 100)
        cont_val    = st.selectbox("Continent (encoded)",  list(range(8)))
        region_val  = st.selectbox("Region (encoded)",     list(range(25)))
        season_val  = st.selectbox("Season (encoded)",     list(range(4)),
                                   format_func=lambda x: ["Autumn","Spring","Summer","Winter"][x])

    if st.button("🚀 Predict Rating", type="primary"):
        try:
            feat_map = {
                "Continent_enc": cont_val,  "Region_enc": region_val,
                "Country_enc": 0,           "AttractionType_enc": 0,
                "Season_enc": season_val,   "VisitYear": visit_year,
                "VisitMonth": visit_month,  "UserAvgRating": user_avg,
                "AttrAvgRating": attr_avg,  "AttrVisitCount": attr_visits,
                "AttractionTypeId": 0,      "ContinentId": cont_val,
                "RegionId": region_val,     "CountryId": 0,
            }
            row = np.array([feat_map.get(f, 0) for f in REG_FEATURES], dtype=float).reshape(1, -1)
            if best_r["Model"] in SCALED_REGS:
                row = scaler_r.transform(row)
            pred = np.clip(best_reg_model.predict(row)[0], 1, 5)
            stars = "⭐" * round(pred)
            st.markdown(f"""
            <div class="predict-result">
                <div class="predict-value">{pred:.2f}</div>
                <div style="font-size:1.6rem; margin: 6px 0;">{stars}</div>
                <div class="predict-label">Predicted Rating</div>
                <div style="font-size:0.78rem; color:#334155; margin-top:12px;">
                    Model: {best_r["Model"]} &nbsp;·&nbsp; R² = {best_r["R²"]:.4f} &nbsp;·&nbsp; RMSE = {best_r["RMSE"]:.4f}
                </div>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Model Comparison — All Regression Models")
    fig = px.bar(reg_results_df, x="Model", y=["RMSE", "MAE", "R²"], barmode="group",
                 color_discrete_map={"RMSE": "#ef4444", "MAE": "#f59e0b", "R²": "#10b981"})
    fig.update_traces(marker_line_width=0)
    apply_theme(fig)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(reg_results_df, x="RMSE", y="R²", text="Model", color="Model",
                      color_discrete_sequence=PALETTE,
                      title="RMSE vs R² — top-right corner = best")
    fig2.update_traces(textposition="top center", marker_size=14,
                       marker_line_color="#080c14", marker_line_width=1.5)
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    section("Detailed Results Table")
    st.dataframe(
        reg_results_df.style
            .highlight_min(subset=["RMSE", "MAE"], color="#0f2a1a")
            .highlight_max(subset=["R²"],           color="#0f2a1a")
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
elif page == "🗂️  Classification (Visit Mode)":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">🏷️ Classification — Visit Mode</div>
        <div class="page-hero-sub">Predict whether a user will visit as Business, Family, Couples, Friends, or Solo — enabling targeted marketing campaigns.</div>
    </div>
    """, unsafe_allow_html=True)

    best_c = clf_results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_c["Model"])
    c2.metric("Accuracy",   f"{best_c['Accuracy']:.4f}")
    c3.metric("F1 Score",   f"{best_c['F1']:.4f}")
    c4.metric("Precision",  f"{best_c['Precision']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    section("🔮 Live Visit Mode Prediction")

    col1, col2 = st.columns(2)
    with col1:
        c_year  = st.slider("Visit Year",  2010, 2025, 2022, key="cv_yr")
        c_month = st.slider("Visit Month", 1, 12, 6,         key="cv_mo")
        c_ua    = st.slider("User Avg Rating",       1.0, 5.0, 3.5, 0.1, key="cv_ua")
        c_aa    = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0, 0.1, key="cv_aa")
    with col2:
        c_vc    = st.number_input("Attraction Visit Count", 1, 10000, 100, key="cv_vc")
        c_co    = st.selectbox("Continent (encoded)",  list(range(8)),  key="cv_co")
        c_re    = st.selectbox("Region (encoded)",     list(range(25)), key="cv_re")
        c_se    = st.selectbox("Season (encoded)",     list(range(4)),  key="cv_se",
                               format_func=lambda x: ["Autumn","Spring","Summer","Winter"][x])

    if st.button("🚀 Predict Visit Mode", type="primary"):
        try:
            feat_map = {
                "Continent_enc": c_co,  "Region_enc": c_re,
                "Country_enc": 0,       "AttractionType_enc": 0,
                "Season_enc": c_se,     "VisitYear": c_year,
                "VisitMonth": c_month,  "UserAvgRating": c_ua,
                "AttrAvgRating": c_aa,  "AttrVisitCount": c_vc,
                "AttractionTypeId": 0,  "ContinentId": c_co,
                "RegionId": c_re,       "CountryId": 0,
            }
            row = np.array([feat_map.get(f, 0) for f in CLF_FEATURES], dtype=float).reshape(1, -1)
            if best_c["Model"] in SCALED_CLFS:
                row = scaler_c.transform(row)
            pred_label = best_clf_model.predict(row)[0]
            mode_name  = le_mode.classes_[pred_label] if pred_label < len(le_mode.classes_) else str(pred_label)
            icons = {"Business": "💼", "Family": "👨‍👩‍👧", "Couples": "💑", "Friends": "👫", "Solo": "🧳"}
            icon = icons.get(mode_name, "🗺️")
            st.markdown(f"""
            <div class="predict-result">
                <div style="font-size:3rem;">{icon}</div>
                <div class="predict-value" style="font-size:2rem; margin-top:8px;">{mode_name}</div>
                <div class="predict-label">Predicted Visit Mode</div>
            </div>""", unsafe_allow_html=True)

            if hasattr(best_clf_model, "predict_proba"):
                probs   = best_clf_model.predict_proba(row)[0]
                prob_df = pd.DataFrame({"Visit Mode": le_mode.classes_, "Probability": probs})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                fig = px.bar(prob_df, x="Visit Mode", y="Probability",
                             color="Probability", color_continuous_scale=[[0,"#1e293b"],[1,"#7c3aed"]])
                fig.update_traces(marker_line_width=0)
                apply_theme(fig)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Model Comparison — All Classifiers")
    fig = px.bar(clf_results_df, x="Model",
                 y=["Accuracy", "F1", "Precision", "Recall"], barmode="group",
                 color_discrete_map={"Accuracy":"#00d4ff","F1":"#7c3aed","Precision":"#10b981","Recall":"#f59e0b"})
    fig.update_traces(marker_line_width=0)
    apply_theme(fig)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    section("Detailed Results Table")
    st.dataframe(
        clf_results_df.style
            .highlight_max(subset=["Accuracy", "F1", "Precision", "Recall"], color="#0f2a1a")
            .format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif page == "🎯  Recommendation Engine":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">✨ Recommendation Engine</div>
        <div class="page-hero-sub">Personalised attraction suggestions via Collaborative Filtering (SVD) or Content-Based Filtering (cosine similarity).</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("CF RMSE (SVD)", f"{rmse_cf:.4f}")
    c2.metric("CF MAE  (SVD)", f"{mae_cf:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    tab_cf, tab_cb = st.tabs(["🤝  Collaborative Filtering", "📖  Content-Based Filtering"])

    with tab_cf:
        section("Top-N Personalised Recommendations (SVD)")
        st.caption("Recommend attractions a user hasn't visited yet, ranked by predicted rating.")

        all_users = list(predicted_ratings_df.index)
        user_id   = st.selectbox("Select User ID", all_users[:500])
        n_cf      = st.slider("Number of recommendations", 5, 20, 10)

        if st.button("🎯 Get Recommendations", key="cf_btn"):
            user_row = predicted_ratings_df.loc[user_id]
            visited  = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
            recs     = user_row.drop(index=visited, errors="ignore")
            top_n    = recs.sort_values(ascending=False).head(n_cf).reset_index()
            top_n.columns = ["AttractionId", "PredictedRating"]

            if "Attraction" in item.columns:
                top_n = top_n.merge(
                    item[["AttractionId", "Attraction", "AttractionTypeId"]],
                    on="AttractionId", how="left"
                ).merge(attr_type, on="AttractionTypeId", how="left")
                display_cols = ["Attraction", "AttractionType", "PredictedRating"]
            else:
                display_cols = ["AttractionId", "PredictedRating"]

            display_df = top_n[display_cols].rename(columns={"PredictedRating": "Predicted ★"})
            st.dataframe(display_df, use_container_width=True)

            if "Attraction" in top_n.columns:
                fig = px.bar(top_n, x="PredictedRating", y="Attraction", orientation="h",
                             color="PredictedRating",
                             color_continuous_scale=[[0,"#1e293b"],[1,"#00d4ff"]],
                             title=f"Top-{n_cf} Recommendations for User {user_id}")
                fig.update_traces(marker_line_width=0)
                apply_theme(fig)
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

    with tab_cb:
        section("Similar Attractions (Content-Based)")
        st.caption("Attractions similar to a selected one based on type and location features.")

        if cos_sim_df.empty:
            st.warning("Content-based similarity matrix is empty.")
        else:
            all_attr = list(cos_sim_df.index)
            sel_attr = st.selectbox("Select Attraction ID", all_attr[:500])
            n_cb     = st.slider("Number of similar attractions", 5, 20, 10, key="cb_n")

            if "Attraction" in item.columns:
                name_row = item[item["AttractionId"] == sel_attr]
                if not name_row.empty:
                    st.markdown(f'<span class="badge badge-cyan">Selected: {name_row.iloc[0]["Attraction"]}</span>', unsafe_allow_html=True)

            if st.button("🔍 Find Similar Attractions", key="cb_btn"):
                sims   = cos_sim_df[sel_attr].drop(index=sel_attr, errors="ignore")
                top_cb = sims.sort_values(ascending=False).head(n_cb).reset_index()
                top_cb.columns = ["AttractionId", "Similarity"]

                if "Attraction" in item.columns:
                    top_cb = top_cb.merge(
                        item[["AttractionId", "Attraction", "AttractionTypeId"]],
                        on="AttractionId", how="left"
                    ).merge(attr_type, on="AttractionTypeId", how="left")
                    display_cols = ["Attraction", "AttractionType", "Similarity"]
                else:
                    display_cols = ["AttractionId", "Similarity"]

                st.dataframe(top_cb[display_cols], use_container_width=True)

                if "Attraction" in top_cb.columns:
                    fig = px.bar(top_cb, x="Similarity", y="Attraction", orientation="h",
                                 color="Similarity",
                                 color_continuous_scale=[[0,"#1e293b"],[1,"#10b981"]],
                                 title=f"Top-{n_cb} Similar to {sel_attr}")
                    fig.update_traces(marker_line_width=0)
                    apply_theme(fig)
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════
elif page == "📈  Model Leaderboard":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">📈 Model Leaderboard</div>
        <div class="page-hero-sub">Compare all trained models side-by-side across regression, classification, and recommendation tasks.</div>
    </div>
    """, unsafe_allow_html=True)

    section("🔢 Regression Models")
    st.dataframe(
        reg_results_df.style
            .highlight_min(subset=["RMSE", "MAE"], color="#0f2a1a")
            .highlight_max(subset=["R²"],           color="#0f2a1a")
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )
    fig1 = px.scatter(reg_results_df, x="RMSE", y="R²", text="Model", color="Model",
                      color_discrete_sequence=PALETTE,
                      title="Regression: RMSE vs R² — top-right = best")
    fig1.update_traces(textposition="top center", marker_size=14,
                       marker_line_color="#080c14", marker_line_width=1.5)
    apply_theme(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🏷️ Classification Models")
    st.dataframe(
        clf_results_df.style
            .highlight_max(subset=["Accuracy", "F1", "Precision", "Recall"], color="#0f2a1a")
            .format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}),
        use_container_width=True,
    )

    fig2 = go.Figure()
    metrics_clf = ["Accuracy", "F1", "Precision", "Recall"]
    for i, (_, row) in enumerate(clf_results_df.iterrows()):
        fig2.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_clf],
            theta=metrics_clf,
            fill="toself",
            name=row["Model"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            fillcolor=PALETTE[i % len(PALETTE)].replace("#", "rgba(") + ",0.07)" if "#" in PALETTE[i % len(PALETTE)] else PALETTE[i % len(PALETTE)],
        ))
    fig2.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#334155")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#64748b")),
        ),
        title=dict(text="Classifier Radar Chart — All Metrics", font=dict(color="#f1f5f9", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        showlegend=True,
        margin=dict(t=48, b=12, l=12, r=12),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🎯 Recommendation — SVD Collaborative Filtering")
    cf_metrics = pd.DataFrame({"Metric": ["RMSE", "MAE"], "Value": [rmse_cf, mae_cf]})
    st.dataframe(cf_metrics.style.format({"Value": "{:.4f}"}), use_container_width=True)
    fig3 = px.bar(cf_metrics, x="Metric", y="Value", color="Metric",
                  color_discrete_map={"RMSE": "#ef4444", "MAE": "#f59e0b"})
    fig3.update_traces(marker_line_width=0)
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 7 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "💡  Business Insights":
    st.markdown("""
    <div class="page-hero">
        <div class="page-hero-title">💡 Business Insights</div>
        <div class="page-hero-sub">Actionable findings for tourism agencies, travel platforms, and destination management organisations.</div>
    </div>
    """, unsafe_allow_html=True)

    section("🔑 Key Insights from Analysis")
    insights = [
        ("🔑", "Rating Drivers",          "badge-cyan",
         "Attraction type, user continent, and season are the strongest predictors of user ratings. Platforms should tailor content and placement based on these factors."),
        ("📅", "Seasonal Patterns",        "badge-violet",
         "Families visit predominantly in Summer and Spring. Business travellers show far less seasonal variation, enabling targeted off-peak promotions."),
        ("📈", "Long-Tail Attraction Demand","badge-green",
         "A small set of attractions receives a disproportionately large share of visits. Operators should invest in surfacing hidden-gem attractions to distribute demand."),
        ("🤝", "Personalisation Uplift",   "badge-amber",
         "SVD-based collaborative filtering achieves low RMSE even with sparse user-item data, suggesting meaningful personalisation gains from deploying CF in production."),
        ("🆕", "Cold-Start Solution",      "badge-cyan",
         "Content-based cosine similarity on attraction type & city handles new users with no history, ensuring every visitor receives relevant suggestions from day one."),
        ("📊", "Customer Segmentation",    "badge-violet",
         "Visit mode classification enables precise customer segmentation — family travellers, business visitors, and couples each require tailored communication."),
    ]
    for icon, title, badge, text in insights:
        with st.expander(f"{icon}  {title}", expanded=False):
            st.markdown(f'<span class="badge {badge}">{title}</span><br><br>', unsafe_allow_html=True)
            st.write(text)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🚀 Actionable Recommendations")
    recs = [
        ("💼", "Targeted Promotions",       "#00d4ff",
         "Use the classification model to predict visit mode, then push family or couple packages proactively."),
        ("⚠️", "Dynamic Rating Alerts",     "#f59e0b",
         "Flag attractions predicted to receive low ratings so operators can intervene early."),
        ("🏠", "Personalised Homepage",      "#7c3aed",
         "Surface CF recommendations at login; fall back to CB recommendations for new/anonymous users."),
        ("🌦️","Seasonal Capacity Planning", "#10b981",
         "Use Summer/Spring demand forecasts to help attraction organisers staff up efficiently."),
        ("📡", "Region-Based Marketing",     "#f472b6",
         "Certain continents show strong affinity for specific attraction types — leverage for geo-targeted ads."),
        ("🔄", "Feedback Loop",              "#3b82f6",
         "Collect user clicks on recommendations to retrain models periodically and improve accuracy over time."),
    ]
    col1, col2 = st.columns(2)
    for i, (icon, title, color, desc) in enumerate(recs):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f"""
        <div class="insight-card" style="border-left-color: {color};">
            <div class="insight-title">{icon} {title}</div>
            <div class="insight-text">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("📋 Business Use Cases")
    col1, col2 = st.columns(2)
    use_cases = [
        ("🎯", "Personalized Recommendations", "#00d4ff",
         "Suggest attractions based on past visits, preferences, and demographic data."),
        ("📊", "Tourism Analytics",             "#7c3aed",
         "Insights into popular attractions & regions so businesses can adjust offerings."),
        ("🧩", "Customer Segmentation",          "#10b981",
         "Classify users by travel behaviour, enabling targeted promotions."),
        ("💎", "Customer Retention",             "#f59e0b",
         "Personalised recommendations boost loyalty — users return when they discover great places."),
    ]
    for i, (icon, title, color, desc) in enumerate(use_cases):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f"""
        <div class="kpi-card" style="border-left: 3px solid {color}; padding: 18px;">
            <div style="font-size:1.4rem; margin-bottom:8px;">{icon}</div>
            <div style="font-weight:700; font-size:0.9rem; color:#f1f5f9; margin-bottom:6px;">{title}</div>
            <div style="font-size:0.82rem; color:#64748b; line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("📦 Next Steps & Future Work")
    steps = [
        ("🧠", "Experiment with Neural Collaborative Filtering (NCF) for deeper recommendation accuracy."),
        ("💬", "Incorporate review text sentiment (NLP) as an additional rating predictor."),
        ("🔬", "Set up A/B testing on recommendation strategies to measure real-world uplift."),
        ("🔄", "Build a feedback loop — collect user clicks on recommendations to retrain models."),
        ("🔐", "Deploy with user authentication so personalisation is truly per-user across sessions."),
        ("🗺️", "Add map-based visualisations showing attraction density by region."),
    ]
    for icon, step in steps:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:12px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:1.1rem; flex-shrink:0;">{icon}</span>
            <span style="font-size:0.875rem; color:#94a3b8; line-height:1.6;">{step}</span>
        </div>""", unsafe_allow_html=True)