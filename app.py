"""
Tourism Experience Analytics — Streamlit Application
app.py

Guidance-aligned pages:
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
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"]          { background: #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; padding: 4px 0; }

/* ── Metric cards ─────────────────────────────────── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #4f46e5;
    margin-bottom: 12px;
}
.metric-card h2 { color: #4f46e5; font-size: 2rem; margin: 0; }
.metric-card p  { color: #64748b; margin: 4px 0 0 0; font-size: 0.85rem; }

/* ── Section headers ──────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e293b;
    margin: 18px 0 6px 0;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 4px;
}

/* ── Sidebar logo area ──────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: 16px 0 24px 0;
    font-size: 2rem;
}
.sidebar-title {
    font-weight: 700;
    font-size: 1rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and data …")
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        return None
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


arts = load_artifacts()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌍</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Tourism Experience Analytics</div>', unsafe_allow_html=True)

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

    st.markdown("---")
    st.caption("Tourism Experience Analytics v1.0")
    st.caption("Domain: Tourism · Skills: ML, EDA, Streamlit")


# ─────────────────────────────────────────────────────────────────
# GUARD — artifacts not yet built
# ─────────────────────────────────────────────────────────────────
if arts is None:
    st.warning("⚠️ Model artifacts not found. Please run the training script first.")
    st.code("python train.py --data ./data", language="bash")
    st.info("Place your Excel dataset files inside the **data/** folder, then run the command above.")
    st.stop()


# Unpack artifacts
df                  = arts["df"]
item                = arts["item"]
attr_type           = arts["attr_type"]
le_mode             = arts["le_mode"]
le_dict             = arts["le_dict"]
REG_FEATURES        = arts["REG_FEATURES"]
CLF_FEATURES        = arts["CLF_FEATURES"]
reg_results_df      = arts["reg_results_df"]
clf_results_df      = arts["clf_results_df"]
best_reg_model      = arts["best_reg_model"]
best_clf_model      = arts["best_clf_model"]
scaler_r            = arts["scaler_r"]
scaler_c            = arts["scaler_c"]
predicted_ratings_df= arts["predicted_ratings_df"]
user_item_matrix    = arts["user_item_matrix"]
cos_sim_df          = arts["cos_sim_df"]
rmse_cf             = arts["rmse_cf"]
mae_cf              = arts["mae_cf"]
SCALED_REGS         = arts.get("SCALED_REGS", {"Linear Regression", "Ridge Regression", "Lasso Regression"})
SCALED_CLFS         = arts.get("SCALED_CLFS", {"Logistic Regression", "K-Nearest Neighbors"})


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":
    st.title("🌍 Tourism Experience Analytics")
    st.markdown(
        "**Classification · Prediction · Recommendation System** — "
        "Explore user travel patterns, predict visit modes, and get personalised attraction suggestions."
    )
    st.markdown("---")

    # ── KPI cards ────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(df):,}",                    "Total Transactions"),
        (c2, f"{df['UserId'].nunique():,}",      "Unique Users"),
        (c3, f"{df['AttractionId'].nunique():,}","Unique Attractions"),
        (c4, f"{df['Rating'].mean():.2f} ★",    "Avg Rating"),
    ]:
        col.markdown(
            f'<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-header">Visit Mode Distribution</p>', unsafe_allow_html=True)
        vm = df["VisitMode"].value_counts().reset_index()
        vm.columns = ["VisitMode", "Count"]
        fig = px.pie(vm, names="VisitMode", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Set3, hole=0.42)
        fig.update_traces(textinfo="label+percent", textfont_size=12)
        fig.update_layout(margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Rating Distribution</p>', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="Rating", nbins=10,
                            color_discrete_sequence=["#4f46e5"])
        fig2.update_layout(bargap=0.1, margin=dict(t=20, b=20, l=0, r=0),
                           xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Objectives ───────────────────────────────────────────────
    st.markdown('<p class="section-header">Project Objectives</p>', unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    for col, icon, title, desc in [
        (o1, "🔢", "Regression",
         "Predict the rating a user will give an attraction based on demographics & visit details."),
        (o2, "🏷️", "Classification",
         "Predict the visit mode (Family, Business, Couples…) from user profile & history."),
        (o3, "✨", "Recommendation",
         "Suggest personalised attractions via Collaborative (SVD) and Content-Based filtering."),
    ]:
        col.info(f"**{icon} {title}**\n\n{desc}")

    # ── Dataset quick summary ─────────────────────────────────────
    st.markdown('<p class="section-header">Dataset Quick Summary</p>', unsafe_allow_html=True)
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
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "Understand distributions, correlations, and relationships across "
        "users, attractions, and visit behaviour."
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["👤 Users", "📍 Attractions", "📅 Temporal", "🌡️ Correlations"])

    with tab1:
        if "Continent" in df.columns:
            st.markdown('<p class="section-header">Users by Continent</p>', unsafe_allow_html=True)
            cont = df.drop_duplicates("UserId")["Continent"].value_counts().reset_index()
            cont.columns = ["Continent", "Users"]
            fig = px.bar(cont, x="Continent", y="Users", color="Continent",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title="Unique Users per Continent")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if "Country" in df.columns:
            st.markdown('<p class="section-header">Top 15 User Countries</p>', unsafe_allow_html=True)
            top_c = df.drop_duplicates("UserId")["Country"].value_counts().head(15).reset_index()
            top_c.columns = ["Country", "Users"]
            fig2 = px.bar(top_c, x="Users", y="Country", orientation="h",
                          color="Users", color_continuous_scale="Purples",
                          title="Top 15 Countries by User Count")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-header">Rating Distribution by Visit Mode</p>', unsafe_allow_html=True)
        fig3 = px.box(df, x="VisitMode", y="Rating", color="VisitMode",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title="Rating Spread per Visit Mode")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        if "Attraction" in df.columns:
            st.markdown('<p class="section-header">Top 10 Most Visited Attractions</p>', unsafe_allow_html=True)
            top_a = df["Attraction"].value_counts().head(10).reset_index()
            top_a.columns = ["Attraction", "Visits"]
            fig = px.bar(top_a, x="Visits", y="Attraction", orientation="h",
                         color="Visits", color_continuous_scale="Viridis",
                         title="Top 10 Attractions by Visit Count")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        if "AttractionType" in df.columns:
            st.markdown('<p class="section-header">Attraction Types</p>', unsafe_allow_html=True)
            tc = df["AttractionType"].value_counts().head(15).reset_index()
            tc.columns = ["AttractionType", "Count"]
            fig2 = px.bar(tc, x="AttractionType", y="Count",
                          color="Count", color_continuous_scale="Teal",
                          title="Top Attraction Types by Visit Count")
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

            # Average rating per attraction type
            st.markdown('<p class="section-header">Average Rating per Attraction Type</p>', unsafe_allow_html=True)
            avg_type = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False).head(15).reset_index()
            avg_type.columns = ["AttractionType", "AvgRating"]
            fig3 = px.bar(avg_type, x="AvgRating", y="AttractionType", orientation="h",
                          color="AvgRating", color_continuous_scale="RdYlGn",
                          title="Average Rating per Attraction Type")
            fig3.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown('<p class="section-header">Monthly Visit Trend</p>', unsafe_allow_html=True)
        monthly = df.groupby("VisitMonth").size().reset_index(name="Visits")
        monthly["Month"] = monthly["VisitMonth"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
        })
        fig = px.line(monthly, x="Month", y="Visits", markers=True,
                      title="Monthly Visit Trend",
                      labels={"Visits": "Number of Visits"})
        fig.update_traces(line_color="#4f46e5", marker_size=8)
        st.plotly_chart(fig, use_container_width=True)

        if "Season" in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-header">Visits by Season</p>', unsafe_allow_html=True)
                season_counts = df["Season"].value_counts().reset_index()
                season_counts.columns = ["Season", "Visits"]
                fig2 = px.pie(season_counts, names="Season", values="Visits",
                              color_discrete_sequence=["#4f46e5","#0ea5e9","#10b981","#f59e0b"],
                              hole=0.4, title="Visit Share by Season")
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                st.markdown('<p class="section-header">Visit Mode × Season Heatmap</p>', unsafe_allow_html=True)
                hm = df.groupby(["VisitMode", "Season"]).size().unstack(fill_value=0)
                fig3 = px.imshow(hm, text_auto=True, color_continuous_scale="YlOrRd",
                                 aspect="auto", title="Visit Count: VisitMode × Season")
                st.plotly_chart(fig3, use_container_width=True)

        if "VisitYear" in df.columns:
            st.markdown('<p class="section-header">Yearly Visit Trend</p>', unsafe_allow_html=True)
            yearly = df.groupby("VisitYear").size().reset_index(name="Visits")
            fig4 = px.bar(yearly, x="VisitYear", y="Visits",
                          color="Visits", color_continuous_scale="Blues",
                          title="Yearly Visit Volume")
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.markdown('<p class="section-header">Correlation Matrix (Numeric Features)</p>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        aspect="auto", zmin=-1, zmax=1,
                        title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-header">Average Rating by Continent</p>', unsafe_allow_html=True)
        if "Continent" in df.columns:
            cont_rating = df.groupby("Continent")["Rating"].mean().reset_index()
            cont_rating.columns = ["Continent", "AvgRating"]
            fig2 = px.bar(cont_rating, x="Continent", y="AvgRating",
                          color="AvgRating", color_continuous_scale="Teal",
                          title="Average Rating per Continent")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — REGRESSION
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  Regression (Rating)":
    st.title("🔢 Regression — Predict Attraction Rating")
    st.markdown(
        "The regression model predicts the rating (1–5) a user would give an attraction "
        "based on visit details, user demographics, and attraction characteristics."
    )
    st.markdown("---")

    # ── Best model metrics ────────────────────────────────────────
    best_r = reg_results_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", best_r["Model"])
    c2.metric("R²",  f"{best_r['R²']:.4f}")
    c3.metric("RMSE", f"{best_r['RMSE']:.4f}")

    st.markdown("---")
    st.markdown('<p class="section-header">🔮 Live Rating Prediction</p>', unsafe_allow_html=True)
    st.caption("Adjust the features below and click Predict Rating to see the model's output.")

    col1, col2 = st.columns(2)
    with col1:
        visit_year   = st.slider("Visit Year",  2010, 2025, 2022)
        visit_month  = st.slider("Visit Month", 1, 12, 6)
        user_avg     = st.slider("User Avg Rating",       1.0, 5.0, 3.5, 0.1)
        attr_avg     = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0, 0.1)
    with col2:
        attr_visits  = st.number_input("Attraction Visit Count", 1, 10000, 100)
        cont_val     = st.selectbox("Continent (encoded)",  list(range(8)))
        region_val   = st.selectbox("Region (encoded)",     list(range(25)))
        season_val   = st.selectbox("Season (encoded)",     list(range(4)),
                                    format_func=lambda x: ["Autumn","Spring","Summer","Winter"][x])

    if st.button("🚀 Predict Rating", type="primary"):
        try:
            feat_map = {
                "Continent_enc": cont_val,   "Region_enc": region_val,
                "Country_enc": 0,            "AttractionType_enc": 0,
                "Season_enc": season_val,    "VisitYear": visit_year,
                "VisitMonth": visit_month,   "UserAvgRating": user_avg,
                "AttrAvgRating": attr_avg,   "AttrVisitCount": attr_visits,
                "AttractionTypeId": 0,       "ContinentId": cont_val,
                "RegionId": region_val,      "CountryId": 0,
            }
            row = np.array([feat_map.get(f, 0) for f in REG_FEATURES], dtype=float).reshape(1, -1)
            if best_r["Model"] in SCALED_REGS:
                row = scaler_r.transform(row)
            pred = np.clip(best_reg_model.predict(row)[0], 1, 5)
            st.success(f"### Predicted Rating: **{pred:.2f} ★**")
            stars = "⭐" * round(pred)
            st.markdown(f"<h2 style='color:#f59e0b'>{stars}</h2>", unsafe_allow_html=True)
            st.info(f"Model used: **{best_r['Model']}**  |  R² = {best_r['R²']:.4f}  |  RMSE = {best_r['RMSE']:.4f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.markdown('<p class="section-header">Model Comparison — All Regression Models</p>', unsafe_allow_html=True)
    fig = px.bar(reg_results_df, x="Model", y=["RMSE", "MAE", "R²"], barmode="group",
                 color_discrete_map={"RMSE": "#ef4444", "MAE": "#f97316", "R²": "#22c55e"},
                 title="Regression Model Performance Comparison")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    # RMSE vs R² scatter
    fig2 = px.scatter(reg_results_df, x="RMSE", y="R²", text="Model",
                      color="Model", title="RMSE vs R² (top-right corner = best)")
    fig2.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-header">Detailed Results Table</p>', unsafe_allow_html=True)
    st.dataframe(
        reg_results_df.style
            .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
            .highlight_max(subset=["R²"], color="#d1fae5")
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
elif page == "🗂️  Classification (Visit Mode)":
    st.title("🏷️ Classification — Predict Visit Mode")
    st.markdown(
        "Predict whether a user will visit as **Business, Family, Couples, Friends**, etc. "
        "This enables travel platforms to tailor marketing campaigns and resources."
    )
    st.markdown("---")

    best_c = clf_results_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_c["Model"])
    c2.metric("Accuracy",   f"{best_c['Accuracy']:.4f}")
    c3.metric("F1 Score",   f"{best_c['F1']:.4f}")
    c4.metric("Precision",  f"{best_c['Precision']:.4f}")

    st.markdown("---")
    st.markdown('<p class="section-header">🔮 Live Visit Mode Prediction</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        c_year   = st.slider("Visit Year",  2010, 2025, 2022, key="cv_yr")
        c_month  = st.slider("Visit Month", 1, 12, 6,         key="cv_mo")
        c_ua     = st.slider("User Avg Rating",       1.0, 5.0, 3.5, 0.1, key="cv_ua")
        c_aa     = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0, 0.1, key="cv_aa")
    with col2:
        c_vc     = st.number_input("Attraction Visit Count", 1, 10000, 100, key="cv_vc")
        c_co     = st.selectbox("Continent (encoded)",  list(range(8)),  key="cv_co")
        c_re     = st.selectbox("Region (encoded)",     list(range(25)), key="cv_re")
        c_se     = st.selectbox("Season (encoded)",     list(range(4)),  key="cv_se",
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
            icons = {"Business": "💼", "Family": "👨‍👩‍👧", "Couples": "💑",
                     "Friends": "👫", "Solo": "🧳"}
            icon = icons.get(mode_name, "🗺️")
            st.success(f"### Predicted Visit Mode: **{icon} {mode_name}**")

            if hasattr(best_clf_model, "predict_proba"):
                probs    = best_clf_model.predict_proba(row)[0]
                prob_df  = pd.DataFrame({"Visit Mode": le_mode.classes_, "Probability": probs})
                prob_df  = prob_df.sort_values("Probability", ascending=False)
                fig = px.bar(prob_df, x="Visit Mode", y="Probability",
                             color="Probability", color_continuous_scale="Purples",
                             title="Prediction Probability per Visit Mode")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.markdown('<p class="section-header">Model Comparison — All Classifiers</p>', unsafe_allow_html=True)
    fig = px.bar(clf_results_df, x="Model",
                 y=["Accuracy", "F1", "Precision", "Recall"],
                 barmode="group",
                 color_discrete_map={"Accuracy": "#4f46e5", "F1": "#0ea5e9",
                                     "Precision": "#10b981", "Recall": "#f59e0b"},
                 title="Classifier Performance Comparison")
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Detailed Results Table</p>', unsafe_allow_html=True)
    st.dataframe(
        clf_results_df.style
            .highlight_max(subset=["Accuracy", "F1", "Precision", "Recall"], color="#d1fae5")
            .format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════
elif page == "🎯  Recommendation Engine":
    st.title("✨ Recommendation Engine")
    st.markdown(
        "Get personalised attraction suggestions via **Collaborative Filtering (SVD)** "
        "or **Content-Based Filtering** (cosine similarity on attraction features)."
    )
    st.markdown("---")

    c1, c2 = st.columns(2)
    c1.metric("CF RMSE (SVD)", f"{rmse_cf:.4f}")
    c2.metric("CF MAE  (SVD)", f"{mae_cf:.4f}")

    st.markdown("---")
    tab_cf, tab_cb = st.tabs(["🤝 Collaborative Filtering", "📖 Content-Based Filtering"])

    with tab_cf:
        st.markdown('<p class="section-header">Top-N Personalised Recommendations (SVD)</p>', unsafe_allow_html=True)
        st.caption("Recommend attractions a specific user hasn't visited yet, ranked by predicted rating.")

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
            st.dataframe(
                display_df.style.background_gradient(subset=["Predicted ★"], cmap="Purples"),
                use_container_width=True)

            if "Attraction" in top_n.columns:
                fig = px.bar(top_n, x="PredictedRating", y="Attraction",
                             orientation="h", color="PredictedRating",
                             color_continuous_scale="Viridis",
                             title=f"Top-{n_cf} Recommendations for User {user_id}")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

    with tab_cb:
        st.markdown('<p class="section-header">Similar Attractions (Content-Based)</p>', unsafe_allow_html=True)
        st.caption("Find attractions similar to a selected one based on type and location features.")

        if cos_sim_df.empty:
            st.warning("Content-based similarity matrix is empty — check Item data features.")
        else:
            all_attr = list(cos_sim_df.index)
            sel_attr = st.selectbox("Select Attraction ID", all_attr[:500])
            n_cb     = st.slider("Number of similar attractions", 5, 20, 10, key="cb_n")

            if "Attraction" in item.columns:
                name_row = item[item["AttractionId"] == sel_attr]
                if not name_row.empty:
                    st.info(f"**Selected:** {name_row.iloc[0]['Attraction']}")

            if st.button("🔍 Find Similar Attractions", key="cb_btn"):
                sims  = cos_sim_df[sel_attr].drop(index=sel_attr, errors="ignore")
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

                st.dataframe(
                    top_cb[display_cols].style.background_gradient(subset=["Similarity"], cmap="Greens"),
                    use_container_width=True)

                if "Attraction" in top_cb.columns:
                    fig = px.bar(top_cb, x="Similarity", y="Attraction",
                                 orientation="h", color="Similarity",
                                 color_continuous_scale="Teal",
                                 title=f"Top-{n_cb} Attractions Similar to {sel_attr}")
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════
elif page == "📈  Model Leaderboard":
    st.title("📈 Model Leaderboard")
    st.markdown("Compare all trained models side-by-side across regression, classification, and recommendation tasks.")
    st.markdown("---")

    # ── Regression ───────────────────────────────────────────────
    st.markdown('<p class="section-header">🔢 Regression Models</p>', unsafe_allow_html=True)
    st.dataframe(
        reg_results_df.style
            .highlight_min(subset=["RMSE", "MAE"], color="#d1fae5")
            .highlight_max(subset=["R²"],           color="#d1fae5")
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )
    fig1 = px.scatter(reg_results_df, x="RMSE", y="R²", text="Model",
                      color="Model", size_max=20,
                      title="Regression: RMSE vs R² (top-right = best)")
    fig1.update_traces(textposition="top center", marker_size=12)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ── Classification ───────────────────────────────────────────
    st.markdown('<p class="section-header">🏷️ Classification Models</p>', unsafe_allow_html=True)
    st.dataframe(
        clf_results_df.style
            .highlight_max(subset=["Accuracy", "F1", "Precision", "Recall"], color="#d1fae5")
            .format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}),
        use_container_width=True,
    )

    # Radar chart
    fig2 = go.Figure()
    metrics_clf = ["Accuracy", "F1", "Precision", "Recall"]
    for _, row in clf_results_df.iterrows():
        fig2.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics_clf],
            theta=metrics_clf,
            fill="toself",
            name=row["Model"],
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Classifier Radar Chart — All Metrics",
        showlegend=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Recommendation ───────────────────────────────────────────
    st.markdown('<p class="section-header">🎯 Recommendation — SVD Collaborative Filtering</p>', unsafe_allow_html=True)
    cf_metrics = pd.DataFrame({"Metric": ["RMSE", "MAE"], "Value": [rmse_cf, mae_cf]})
    st.dataframe(cf_metrics.style.format({"Value": "{:.4f}"}), use_container_width=True)

    fig3 = px.bar(cf_metrics, x="Metric", y="Value",
                  color="Metric", color_discrete_map={"RMSE": "#ef4444", "MAE": "#f97316"},
                  title="Collaborative Filtering (SVD) Evaluation Metrics")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 7 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "💡  Business Insights":
    st.title("💡 Business Insights & Recommendations")
    st.markdown(
        "Actionable findings for tourism agencies, travel platforms, "
        "and destination management organisations."
    )
    st.markdown("---")

    # ── Key Insights ─────────────────────────────────────────────
    st.markdown('<p class="section-header">🔑 Key Insights from Analysis</p>', unsafe_allow_html=True)
    insights = [
        ("🔑 Rating Drivers",
         "Attraction type, user continent, and season are the strongest predictors of user ratings. "
         "Platforms should tailor content and placement based on these factors."),
        ("📅 Seasonal Patterns",
         "Families visit predominantly in Summer and Spring. "
         "Business travellers show far less seasonal variation, enabling targeted off-peak promotions."),
        ("📈 Long-Tail Attraction Demand",
         "A small set of attractions receives a disproportionately large share of visits. "
         "Operators should invest in surfacing hidden-gem attractions to distribute demand."),
        ("🤝 Personalisation Uplift",
         "SVD-based collaborative filtering achieves low RMSE even with sparse user-item data, "
         "suggesting meaningful personalisation gains from deploying CF in production."),
        ("🆕 Cold-Start Solution",
         "Content-based cosine similarity on attraction type & city handles new users with no history, "
         "ensuring every visitor receives relevant suggestions from their very first session."),
        ("📊 Customer Segmentation",
         "Visit mode classification enables precise customer segmentation — "
         "family travellers, business visitors, and couples each require tailored communication."),
    ]
    for title, text in insights:
        with st.expander(title, expanded=True):
            st.write(text)

    st.markdown("---")

    # ── Actionable Recommendations ───────────────────────────────
    st.markdown('<p class="section-header">🚀 Actionable Recommendations for Stakeholders</p>', unsafe_allow_html=True)
    recs = [
        ("💼 Targeted Promotions",
         "Use the classification model to predict visit mode, then push family or couple packages proactively."),
        ("⚠️ Dynamic Rating Alerts",
         "Flag attractions predicted to receive low ratings so operators can intervene early."),
        ("🏠 Personalised Homepage",
         "Surface CF recommendations at login; fall back to CB recommendations for new/anonymous users."),
        ("🌦️ Seasonal Capacity Planning",
         "Use Summer/Spring demand forecasts to help attraction organisers staff up efficiently."),
        ("📡 Region-Based Marketing",
         "Certain continents show strong affinity for specific attraction types — leverage this for geo-targeted ads."),
        ("🔄 Feedback Loop",
         "Collect user clicks on recommendations to retrain models periodically and improve accuracy over time."),
    ]
    for icon_title, desc in recs:
        st.markdown(
            f"""<div style="background:white;border-radius:10px;padding:14px 18px;
                margin-bottom:10px;box-shadow:0 1px 6px rgba(0,0,0,0.07);
                border-left:3px solid #4f46e5;">
                <b style="color:#4f46e5">{icon_title}</b><br>
                <span style="color:#374151;font-size:0.92rem">{desc}</span>
            </div>""",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Business Use Cases (from project doc) ────────────────────
    st.markdown('<p class="section-header">📋 Business Use Cases (Project Guidance)</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    use_cases = [
        ("🎯 Personalized Recommendations",
         "Suggest attractions based on past visits, preferences, and demographic data."),
        ("📊 Tourism Analytics",
         "Provide insights into popular attractions & regions so businesses can adjust offerings."),
        ("🧩 Customer Segmentation",
         "Classify users by travel behaviour, enabling targeted promotions."),
        ("💎 Customer Retention",
         "Personalised recommendations boost loyalty — users return when they discover great places."),
    ]
    for i, (title, desc) in enumerate(use_cases):
        col = col1 if i % 2 == 0 else col2
        col.info(f"**{title}**\n\n{desc}")

    st.markdown("---")

    # ── Next Steps ───────────────────────────────────────────────
    st.markdown('<p class="section-header">📦 Next Steps & Future Work</p>', unsafe_allow_html=True)
    for step in [
        "Experiment with **Neural Collaborative Filtering (NCF)** for deeper recommendation accuracy.",
        "Incorporate **review text sentiment (NLP)** as an additional rating predictor.",
        "Set up **A/B testing** on recommendation strategies to measure real-world uplift.",
        "Build a **feedback loop** — collect user clicks on recommendations to retrain models.",
        "Deploy with **user authentication** so personalisation is truly per-user across sessions.",
        "Add **map-based visualisations** showing attraction density by region.",
    ]:
        st.markdown(f"- {step}")
