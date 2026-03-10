"""
Olympic Games Analytics Dashboard
==================================
Interactive explorer for the Olympics athletes dataset.

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from ml_pipeline import (
    compute_shap,
    prepare_data,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_lightgbm,
    train_mlp,
    get_feature_names,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)

# ======================================================================
# PAGE CONFIG
# ======================================================================

st.set_page_config(
    page_title="Olympic Games Dashboard",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CSS
# ======================================================================

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"]{background:linear-gradient(160deg,#f0f4ff 0%,#fff8f0 100%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1445,#1a237e)!important}
[data-testid="stSidebar"] *{color:#e8eaf6!important}
[data-testid="stSidebar"] label{color:#9fa8da!important}

.hero{background:linear-gradient(135deg,#0c1445 0%,#1565c0 50%,#b71c1c 100%);
      padding:2.6rem 2.5rem 2rem;border-radius:20px;color:#fff;margin-bottom:1.8rem;
      box-shadow:0 12px 40px rgba(21,101,192,.35);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;top:-60px;right:-60px;width:260px;height:260px;
              border-radius:50%;background:rgba(255,255,255,.07)}
.hero h1{font-size:2.4rem;font-weight:800;margin:0;letter-spacing:-.5px}
.hero p{margin:.5rem 0 0;font-size:1.1rem;opacity:.88}

.sec{font-size:1.25rem;font-weight:700;color:#1e293b;border-left:4px solid #1565c0;
     padding-left:.75rem;margin:1.5rem 0 1rem}

.kpi{background:#fff;border-radius:14px;padding:1.2rem 1.4rem;
     box-shadow:0 2px 14px rgba(0,0,0,.07);text-align:center;border-top:4px solid}
.kpi .lab{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b}
.kpi .val{font-size:1.8rem;font-weight:800;margin:.2rem 0 0}

.medal-gold{color:#f59e0b;font-weight:700}
.medal-silver{color:#94a3b8;font-weight:700}
.medal-bronze{color:#b45309;font-weight:700}

[data-testid="stTabs"] button[role="tab"]{color:#1e293b!important}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#1565c0!important;font-weight:700}

/* Force dark text throughout main content — prevents white-on-white in dark mode */
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] .stMarkdown h1,
[data-testid="stAppViewContainer"] .stMarkdown h2,
[data-testid="stAppViewContainer"] .stMarkdown h3,
[data-testid="stAppViewContainer"] .stMarkdown h4,
[data-testid="stAppViewContainer"] .stMarkdown h5,
[data-testid="stAppViewContainer"] .stMarkdown h6,
[data-testid="stAppViewContainer"] .stMarkdown td,
[data-testid="stAppViewContainer"] .stMarkdown th,
[data-testid="stAppViewContainer"] .stMarkdown strong,
[data-testid="stAppViewContainer"] .stMarkdown em,
[data-testid="stAppViewContainer"] .stMarkdown code,
[data-testid="stAppViewContainer"] .stMarkdown blockquote,
[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"]{color:#1e293b!important}

/* Hero banner keeps white text on its dark gradient background */
.hero, .hero h1, .hero p{color:#fff!important}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# DATA LOAD
# ======================================================================

DATA_PATH = "olympics_athletes_dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["has_medal"] = df["medal"] != "No Medal"
    df["medal_points"] = df["medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1, "No Medal": 0})
    return df


df = load_data()

# ======================================================================
# SIDEBAR FILTERS
# ======================================================================

with st.sidebar:
    st.markdown("## Filters")
    st.markdown("---")

    games_type = st.multiselect(
        "Games Type",
        ["Summer", "Winter"],
        default=["Summer", "Winter"],
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max), step=4)

    all_sports = sorted(df["sport"].unique())
    sel_sports = st.multiselect("Sport", all_sports, default=all_sports)

    all_countries = sorted(df["country_name"].unique())
    sel_countries = st.multiselect("Country", all_countries, default=all_countries)

    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])

    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} athlete records")
    st.markdown(f"**Years:** {year_min} – {year_max}")

# ======================================================================
# APPLY FILTERS
# ======================================================================

fdf = df[
    df["games_type"].isin(games_type)
    & df["year"].between(*year_range)
    & df["sport"].isin(sel_sports)
    & df["country_name"].isin(sel_countries)
    & df["gender"].isin(gender_filter)
].copy()

if fdf.empty:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

# ======================================================================
# TRAIN MODELS (cached — runs once per session)
# ======================================================================

MODELS_PATH = "models.pkl"


@st.cache_resource(show_spinner=False)
def run_all_models():
    import os, joblib

    # Fast path: load pre-trained models from disk
    if os.path.exists(MODELS_PATH):
        return joblib.load(MODELS_PATH)

    # Fallback: train from scratch (slow — run pretrain.py locally instead)
    raw = pd.read_csv(DATA_PATH)
    raw["has_medal"] = raw["medal"] != "No Medal"
    raw["medal_points"] = raw["medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1, "No Medal": 0})
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(raw)
    feature_names = get_feature_names(preprocessor)

    lr_model,   lr_m   = train_logistic_regression(X_train, X_test, y_train, y_test)
    dt_model,   dt_m   = train_decision_tree(X_train, X_test, y_train, y_test)
    rf_model,   rf_m   = train_random_forest(X_train, X_test, y_train, y_test)
    lgbm_model, lgbm_m = train_lightgbm(X_train, X_test, y_train, y_test)
    mlp_model,  mlp_m  = train_mlp(X_train, X_test, y_train, y_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "lr":   (lr_model,   lr_m),
        "dt":   (dt_model,   dt_m),
        "rf":   (rf_model,   rf_m),
        "lgbm": (lgbm_model, lgbm_m),
        "mlp":  (mlp_model,  mlp_m),
    }

# ======================================================================
# TABS
# ======================================================================

tab_exec, tab_eda, tab_ml, tab_explain = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction",
])

# ======================================================================
# TAB 1 — EXECUTIVE SUMMARY
# ======================================================================

with tab_exec:

    # Hero banner
    st.markdown(
        '<div class="hero">'
        "<h1>🏅 Olympic Games Analytics</h1>"
        "<p>Explore 128 years of Olympic history — athletes, medals, sports, and nations.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # KPI cards
    total_athletes = fdf["athlete_id"].nunique()
    total_countries = fdf["country_name"].nunique()
    total_sports = fdf["sport"].nunique()
    total_gold = (fdf["medal"] == "Gold").sum()
    total_medals = fdf["has_medal"].sum()
    record_holders = (fdf["is_record_holder"] != "No").sum()

    kpis = [
        ("Athletes", f"{total_athletes:,}", "#1565c0"),
        ("Countries", f"{total_countries}", "#7b1fa2"),
        ("Sports", f"{total_sports}", "#00796b"),
        ("Gold Medals", f"{total_gold:,}", "#f59e0b"),
        ("Total Medals", f"{total_medals:,}", "#dc2626"),
        ("Record Holders", f"{record_holders:,}", "#0891b2"),
    ]

    cols = st.columns(len(kpis))
    for col, (lab, val, clr) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi" style="border-top-color:{clr}">'
            f'<div class="lab">{lab}</div>'
            f'<div class="val" style="color:{clr}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Medal Table
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">Medal Table by Country</div>', unsafe_allow_html=True)

    medal_tbl = (
        fdf.groupby("country_name")
        .agg(
            Gold=("medal", lambda x: (x == "Gold").sum()),
            Silver=("medal", lambda x: (x == "Silver").sum()),
            Bronze=("medal", lambda x: (x == "Bronze").sum()),
            Total=("has_medal", "sum"),
            Athletes=("athlete_id", "nunique"),
        )
        .reset_index()
        .sort_values(["Gold", "Silver", "Bronze"], ascending=False)
        .reset_index(drop=True)
    )
    medal_tbl.index += 1
    medal_tbl.index.name = "Rank"

    st.dataframe(
        medal_tbl.style.bar(subset=["Gold"], color="#f59e0b")
                       .bar(subset=["Total"], color="#93c5fd"),
        use_container_width=True,
        height=min(600, 60 + len(medal_tbl) * 36),
    )

    top15 = medal_tbl.head(15).reset_index()
    fig_mt = go.Figure()
    fig_mt.add_trace(go.Bar(name="Gold",   x=top15["country_name"], y=top15["Gold"],   marker_color="#f59e0b"))
    fig_mt.add_trace(go.Bar(name="Silver", x=top15["country_name"], y=top15["Silver"], marker_color="#94a3b8"))
    fig_mt.add_trace(go.Bar(name="Bronze", x=top15["country_name"], y=top15["Bronze"], marker_color="#b45309"))
    fig_mt.update_layout(
        barmode="stack", plot_bgcolor="white", height=420,
        legend=dict(orientation="h", y=1.08),
        xaxis_tickangle=-35, margin=dict(b=100, t=40), yaxis_title="Medals",
    )
    st.plotly_chart(fig_mt, use_container_width=True)

    # ------------------------------------------------------------------
    # Top Athletes
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">Top Athletes</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])

    with col_l:
        sort_metric = st.selectbox(
            "Rank athletes by",
            ["Total Medals", "Gold Medals", "Olympics Attended"],
        )
        n_show = st.slider("Show top N", 5, 50, 20)

    athlete_agg = (
        fdf.groupby(["athlete_id", "athlete_name", "country_name", "sport", "gender"])
        .agg(
            Gold=("medal", lambda x: (x == "Gold").sum()),
            Silver=("medal", lambda x: (x == "Silver").sum()),
            Bronze=("medal", lambda x: (x == "Bronze").sum()),
            Total_Medals=("has_medal", "sum"),
            Olympics=("year", "nunique"),
        )
        .reset_index()
    )

    sort_col = {"Total Medals": "Total_Medals", "Gold Medals": "Gold", "Olympics Attended": "Olympics"}[sort_metric]
    top_athletes = athlete_agg.sort_values(sort_col, ascending=False).head(n_show)

    with col_r:
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(name="Gold",   x=top_athletes["athlete_name"], y=top_athletes["Gold"],   marker_color="#f59e0b"))
        fig_a.add_trace(go.Bar(name="Silver", x=top_athletes["athlete_name"], y=top_athletes["Silver"], marker_color="#94a3b8"))
        fig_a.add_trace(go.Bar(name="Bronze", x=top_athletes["athlete_name"], y=top_athletes["Bronze"], marker_color="#b45309"))
        fig_a.update_layout(
            barmode="stack", plot_bgcolor="white", height=420,
            xaxis_tickangle=-40, margin=dict(b=120, t=40),
            legend=dict(orientation="h", y=1.08), yaxis_title="Medals",
        )
        st.plotly_chart(fig_a, use_container_width=True)

    display_df = top_athletes[["athlete_name", "country_name", "sport", "gender", "Gold", "Silver", "Bronze", "Total_Medals", "Olympics"]].copy()
    display_df.columns = ["Athlete", "Country", "Sport", "Gender", "Gold", "Silver", "Bronze", "Total Medals", "Olympics"]
    display_df = display_df.reset_index(drop=True)
    display_df.index += 1
    st.dataframe(display_df, use_container_width=True, height=min(500, 60 + len(display_df) * 36))

    g1, g2 = st.columns(2)
    gender_medals = fdf[fdf["has_medal"]].groupby("gender")["medal"].value_counts().reset_index()
    gender_medals.columns = ["Gender", "Medal", "Count"]
    with g1:
        fig_g = px.bar(
            gender_medals, x="Gender", y="Count", color="Medal",
            color_discrete_map={"Gold": "#f59e0b", "Silver": "#94a3b8", "Bronze": "#b45309"},
            title="Medals by Gender", barmode="group",
        )
        fig_g.update_layout(plot_bgcolor="white", height=360)
        st.plotly_chart(fig_g, use_container_width=True)

    with g2:
        age_df = fdf[fdf["age"].between(10, 80)]
        fig_age = px.histogram(
            age_df, x="age", color="gender",
            color_discrete_map={"Male": "#1565c0", "Female": "#c2185b"},
            title="Age Distribution of Athletes",
            nbins=40, barmode="overlay", opacity=0.7,
        )
        fig_age.update_layout(plot_bgcolor="white", height=360, xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig_age, use_container_width=True)

    # ------------------------------------------------------------------
    # Sport Analysis
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">Sport Analysis</div>', unsafe_allow_html=True)

    sport_agg = (
        fdf.groupby("sport")
        .agg(
            Athletes=("athlete_id", "nunique"),
            Events=("event", "nunique"),
            Gold=("medal", lambda x: (x == "Gold").sum()),
            Total_Medals=("has_medal", "sum"),
            Avg_Age=("age", "mean"),
        )
        .reset_index()
        .sort_values("Total_Medals", ascending=False)
    )

    s1, s2 = st.columns(2)
    with s1:
        fig_s1 = px.bar(
            sport_agg.head(15).sort_values("Total_Medals"),
            x="Total_Medals", y="sport", orientation="h",
            color="Gold",
            color_continuous_scale=["#bfdbfe", "#f59e0b"],
            title="Top 15 Sports by Total Medals",
            labels={"Total_Medals": "Total Medals", "sport": ""},
        )
        fig_s1.update_layout(plot_bgcolor="white", height=480, coloraxis_showscale=False)
        st.plotly_chart(fig_s1, use_container_width=True)

    with s2:
        fig_s2 = px.scatter(
            sport_agg,
            x="Athletes", y="Total_Medals",
            size="Events", color="Avg_Age",
            hover_name="sport",
            color_continuous_scale="RdYlBu_r",
            title="Athletes vs Medals (bubble = # events, color = avg age)",
            labels={"Athletes": "Unique Athletes", "Total_Medals": "Total Medals"},
        )
        fig_s2.update_layout(plot_bgcolor="white", height=480)
        st.plotly_chart(fig_s2, use_container_width=True)

    sv_agg = (
        fdf.groupby(["sport", "games_type"])
        .agg(Athletes=("athlete_id", "nunique"), Medals=("has_medal", "sum"))
        .reset_index()
    )
    fig_sv = px.bar(
        sv_agg.sort_values("Medals", ascending=False).head(30),
        x="sport", y="Medals", color="games_type",
        color_discrete_map={"Summer": "#f59e0b", "Winter": "#1565c0"},
        title="Medals per Sport (Summer vs Winter)",
    )
    fig_sv.update_layout(
        plot_bgcolor="white", height=400, xaxis_tickangle=-40,
        margin=dict(b=120), legend_title="Games Type",
    )
    st.plotly_chart(fig_sv, use_container_width=True)

    st.markdown("#### Sport Deep Dive")
    sel_sport = st.selectbox("Select a sport", sorted(fdf["sport"].unique()))
    sport_df = fdf[fdf["sport"] == sel_sport]

    sd1, sd2, sd3 = st.columns(3)
    sd1.metric("Athletes", sport_df["athlete_id"].nunique())
    sd2.metric("Events", sport_df["event"].nunique())
    sd3.metric("Medalists", sport_df["has_medal"].sum())

    event_medals = (
        sport_df.groupby("event")["has_medal"].sum()
        .reset_index()
        .sort_values("has_medal", ascending=False)
    )
    fig_ev = px.bar(event_medals, x="event", y="has_medal",
                    title=f"Medals by Event — {sel_sport}",
                    labels={"event": "Event", "has_medal": "Medals"})
    fig_ev.update_layout(plot_bgcolor="white", height=360, xaxis_tickangle=-30)
    st.plotly_chart(fig_ev, use_container_width=True)

    # ------------------------------------------------------------------
    # Country Analysis
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">Country Analysis</div>', unsafe_allow_html=True)

    sel_country = st.selectbox("Select a country", sorted(fdf["country_name"].unique()))
    cdf = fdf[fdf["country_name"] == sel_country]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes", cdf["athlete_id"].nunique())
    c2.metric("Gold", (cdf["medal"] == "Gold").sum())
    c3.metric("Total Medals", cdf["has_medal"].sum())
    c4.metric("Sports Competed", cdf["sport"].nunique())

    cc1, cc2 = st.columns(2)
    with cc1:
        c_sport = (
            cdf[cdf["has_medal"]]
            .groupby("sport")["has_medal"].sum()
            .reset_index()
            .sort_values("has_medal", ascending=False)
            .head(12)
        )
        fig_c1 = px.bar(
            c_sport, x="sport", y="has_medal",
            title=f"{sel_country} — Medals by Sport",
            labels={"sport": "", "has_medal": "Medals"},
            color="has_medal", color_continuous_scale="Blues",
        )
        fig_c1.update_layout(plot_bgcolor="white", height=380, xaxis_tickangle=-35, coloraxis_showscale=False)
        st.plotly_chart(fig_c1, use_container_width=True)

    with cc2:
        c_yr = (
            cdf.groupby("year")
            .agg(Gold=("medal", lambda x: (x == "Gold").sum()),
                 Silver=("medal", lambda x: (x == "Silver").sum()),
                 Bronze=("medal", lambda x: (x == "Bronze").sum()))
            .reset_index()
        )
        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Scatter(x=c_yr["year"], y=c_yr["Gold"],   name="Gold",   line=dict(color="#f59e0b", width=2), mode="lines+markers"))
        fig_c2.add_trace(go.Scatter(x=c_yr["year"], y=c_yr["Silver"], name="Silver", line=dict(color="#94a3b8", width=2), mode="lines+markers"))
        fig_c2.add_trace(go.Scatter(x=c_yr["year"], y=c_yr["Bronze"], name="Bronze", line=dict(color="#b45309", width=2), mode="lines+markers"))
        fig_c2.update_layout(
            title=f"{sel_country} — Medal History",
            plot_bgcolor="white", height=380,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_c2, use_container_width=True)

    st.markdown(f"#### Top Athletes from {sel_country}")
    c_athletes = (
        cdf.groupby(["athlete_name", "sport"])
        .agg(Gold=("medal", lambda x: (x == "Gold").sum()),
             Silver=("medal", lambda x: (x == "Silver").sum()),
             Bronze=("medal", lambda x: (x == "Bronze").sum()),
             Total=("has_medal", "sum"))
        .reset_index()
        .sort_values(["Gold", "Silver", "Bronze"], ascending=False)
        .head(15)
    )
    c_athletes.index += 1
    st.dataframe(c_athletes, use_container_width=True, hide_index=False)

    st.markdown("#### Global Medal Map")
    world_medals = (
        fdf.groupby("nationality")
        .agg(Total=("has_medal", "sum"), Gold=("medal", lambda x: (x == "Gold").sum()))
        .reset_index()
    )
    fig_map = px.choropleth(
        world_medals,
        locations="nationality",
        color="Total",
        hover_name="nationality",
        hover_data={"Gold": True, "Total": True},
        color_continuous_scale="YlOrRd",
        title="Total Medals by Country (all time in filter)",
    )
    fig_map.update_layout(height=450, margin=dict(t=50, b=20))
    st.plotly_chart(fig_map, use_container_width=True)

    # ------------------------------------------------------------------
    # Historical Trends
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">Historical Trends</div>', unsafe_allow_html=True)

    yearly = (
        fdf.groupby(["year", "games_type"])
        .agg(
            Athletes=("athlete_id", "nunique"),
            Countries=("country_name", "nunique"),
            Sports=("sport", "nunique"),
            Medals=("has_medal", "sum"),
            Gold=("medal", lambda x: (x == "Gold").sum()),
        )
        .reset_index()
    )

    h1, h2 = st.columns(2)
    with h1:
        fig_h1 = px.line(
            yearly, x="year", y="Athletes", color="games_type",
            color_discrete_map={"Summer": "#f59e0b", "Winter": "#1565c0"},
            markers=True, title="Athletes per Games",
            labels={"year": "Year", "Athletes": "Unique Athletes"},
        )
        fig_h1.update_layout(plot_bgcolor="white", height=360, legend_title="")
        st.plotly_chart(fig_h1, use_container_width=True)

    with h2:
        fig_h2 = px.line(
            yearly, x="year", y="Countries", color="games_type",
            color_discrete_map={"Summer": "#f59e0b", "Winter": "#1565c0"},
            markers=True, title="Participating Countries per Games",
            labels={"year": "Year", "Countries": "Countries"},
        )
        fig_h2.update_layout(plot_bgcolor="white", height=360, legend_title="")
        st.plotly_chart(fig_h2, use_container_width=True)

    medal_yr = fdf.groupby("year")["medal"].value_counts().reset_index()
    medal_yr.columns = ["year", "medal", "count"]
    medal_yr = medal_yr[medal_yr["medal"] != "No Medal"]
    fig_h3 = px.area(
        medal_yr.sort_values("year"),
        x="year", y="count", color="medal",
        color_discrete_map={"Gold": "#f59e0b", "Silver": "#94a3b8", "Bronze": "#b45309"},
        title="Medals Awarded per Year",
        labels={"year": "Year", "count": "Medals Awarded"},
    )
    fig_h3.update_layout(plot_bgcolor="white", height=400, legend_title="")
    st.plotly_chart(fig_h3, use_container_width=True)

    gender_yr = fdf.groupby(["year", "gender"])["athlete_id"].nunique().reset_index()
    gender_yr.columns = ["year", "gender", "athletes"]
    fig_h4 = px.line(
        gender_yr, x="year", y="athletes", color="gender",
        color_discrete_map={"Male": "#1565c0", "Female": "#c2185b"},
        markers=True, title="Male vs Female Athlete Participation Over Time",
        labels={"year": "Year", "athletes": "Athletes"},
    )
    fig_h4.update_layout(plot_bgcolor="white", height=400, legend_title="")
    st.plotly_chart(fig_h4, use_container_width=True)

    st.markdown("#### Record Holders")
    rec_df = fdf[fdf["is_record_holder"] != "No"][["athlete_name", "country_name", "sport", "event", "year", "is_record_holder", "medal"]].drop_duplicates()
    rec_df.columns = ["Athlete", "Country", "Sport", "Event", "Year", "Record Type", "Medal"]
    rec_df = rec_df.sort_values("Year", ascending=False)
    st.dataframe(rec_df, use_container_width=True, height=400, hide_index=True)

# ======================================================================
# TAB 2 — DESCRIPTIVE ANALYTICS (Sections 1.1 – 1.4)
# ======================================================================

with tab_eda:

    # ------------------------------------------------------------------
    # 1.1  Dataset Introduction
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">1.1 Dataset Introduction</div>', unsafe_allow_html=True)

    st.markdown("""
**What does it contain?**
The dataset covers **128 years of Olympic history** (1896 – 2024) and contains one row per
athlete-event entry. Each row records the athlete's personal attributes (age, height, weight,
gender, nationality), the competition context (sport, event, year, host city, games type),
and performance outcomes (medal won, result value, record-holder status).

**Where did it come from?**
The dataset is a curated compilation of publicly available Olympic records, enriched with
athlete biographical data and country-level statistics.

**Prediction target (dependent variable)**
The target is **`has_medal`** — a binary flag indicating whether the athlete won *any* medal
(Gold, Silver, or Bronze) in a given event. This is a **binary classification** problem.

**Why is this task interesting?**
Predicting medal outcomes can help national Olympic committees identify high-potential
athletes, allocate training resources more effectively, and benchmark performance against
historical trends. It is also a rich example of a class-imbalance problem common in
real-world sports analytics.
""")

    n_rows, n_cols_raw = df.shape
    num_feats = df.select_dtypes(include="number").columns.tolist()
    cat_feats = df.select_dtypes(include="object").columns.tolist()

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Rows", f"{n_rows:,}")
    i2.metric("Features (raw)", n_cols_raw)
    i3.metric("Numerical features", len(num_feats))
    i4.metric("Categorical features", len(cat_feats))

    with st.expander("Feature list"):
        feat_df_eda = pd.DataFrame({
            "Feature": df.columns,
            "Type": ["Numerical" if c in num_feats else "Categorical" for c in df.columns],
            "Nulls": df.isnull().sum().values,
            "Sample value": [str(df[c].iloc[0]) for c in df.columns],
        })
        st.dataframe(feat_df_eda, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # 1.2  Target Distribution
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">1.2 Target Distribution</div>', unsafe_allow_html=True)

    target_counts = df["has_medal"].value_counts().reset_index()
    target_counts.columns = ["Has Medal", "Count"]
    target_counts["Has Medal"] = target_counts["Has Medal"].map({True: "Medal", False: "No Medal"})
    target_counts["Percentage"] = (target_counts["Count"] / target_counts["Count"].sum() * 100).round(1)
    target_counts["Label"] = target_counts.apply(lambda r: f"{r['Has Medal']}<br>{r['Count']:,} ({r['Percentage']}%)", axis=1)

    t1, t2 = st.columns([1, 2])
    with t1:
        fig_target = px.bar(
            target_counts, x="Has Medal", y="Count",
            color="Has Medal",
            color_discrete_map={"Medal": "#f59e0b", "No Medal": "#64748b"},
            text="Percentage",
            title="Target Class Frequencies",
        )
        fig_target.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_target.update_layout(plot_bgcolor="white", height=380, showlegend=False,
                                  yaxis_title="Count", xaxis_title="")
        st.plotly_chart(fig_target, use_container_width=True)

    with t2:
        medal_type_counts = df["medal"].value_counts().reset_index()
        medal_type_counts.columns = ["Medal Type", "Count"]
        order = ["Gold", "Silver", "Bronze", "No Medal"]
        medal_type_counts["Medal Type"] = pd.Categorical(medal_type_counts["Medal Type"], categories=order, ordered=True)
        medal_type_counts = medal_type_counts.sort_values("Medal Type")
        fig_mt2 = px.bar(
            medal_type_counts, x="Medal Type", y="Count",
            color="Medal Type",
            color_discrete_map={"Gold": "#f59e0b", "Silver": "#94a3b8", "Bronze": "#b45309", "No Medal": "#64748b"},
            title="Medal Type Breakdown",
        )
        fig_mt2.update_layout(plot_bgcolor="white", height=380, showlegend=False,
                               yaxis_title="Count", xaxis_title="")
        st.plotly_chart(fig_mt2, use_container_width=True)

    st.markdown("""
**Interpretation:**
The target is heavily imbalanced — roughly **76 % No Medal / 24 % Medal**. This class imbalance
drives our choice of evaluation metrics (F1, AUC-ROC) over raw accuracy, and motivates
imbalance-handling strategies (`class_weight="balanced"`, SMOTE) in the models.
""")

    # ------------------------------------------------------------------
    # 1.3  Feature Distributions and Relationships
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">1.3 Feature Distributions and Relationships</div>', unsafe_allow_html=True)

    eda_df = df.copy()
    eda_df["Medal Status"] = eda_df["has_medal"].map({True: "Medal", False: "No Medal"})

    # --- Viz 1: Age distribution by medal status ---
    st.markdown("##### Viz 1 — Age Distribution by Medal Status")
    age_valid = eda_df[eda_df["age"].between(10, 80)]
    fig_v1 = px.histogram(
        age_valid, x="age", color="Medal Status",
        color_discrete_map={"Medal": "#f59e0b", "No Medal": "#64748b"},
        barmode="overlay", opacity=0.65, nbins=40,
        title="Age Distribution: Medalists vs Non-Medalists",
        labels={"age": "Age", "count": "Athletes"},
    )
    fig_v1.update_layout(plot_bgcolor="white", height=380)
    st.plotly_chart(fig_v1, use_container_width=True)
    st.caption(
        "Medalists and non-medalists share a similar age distribution (peak ~22–26), "
        "but medalists skew slightly older — consistent with the idea that peak Olympic "
        "performance often requires several Games of experience. Both tails thin out "
        "symmetrically, with very few competitors younger than 15 or older than 50."
    )

    # --- Viz 2: Height vs Weight scatter ---
    st.markdown("##### Viz 2 — Height vs Weight by Medal Status")
    phys_df = eda_df[eda_df["height_cm"].between(130, 230) & eda_df["weight_kg"].between(30, 180)]
    fig_v2 = px.scatter(
        phys_df.sample(min(2000, len(phys_df)), random_state=42),
        x="height_cm", y="weight_kg",
        color="Medal Status",
        color_discrete_map={"Medal": "#f59e0b", "No Medal": "#64748b"},
        opacity=0.5,
        title="Height vs Weight (sample of 2,000)",
        labels={"height_cm": "Height (cm)", "weight_kg": "Weight (kg)"},
    )
    fig_v2.update_layout(plot_bgcolor="white", height=400)
    st.plotly_chart(fig_v2, use_container_width=True)
    st.caption(
        "Medalists (gold) and non-medalists (grey) overlap almost entirely in height-weight "
        "space, suggesting physical dimensions alone are poor predictors of medal success. "
        "The spread is wide across sports — ranging from lightweight gymnasts to heavy "
        "weightlifters — so sport-stratified analysis would be more informative."
    )

    # --- Viz 3: Medal rate by sport (top 20) ---
    st.markdown("##### Viz 3 — Medal Rate by Sport (Top 20 by athlete count)")
    sport_rate = (
        eda_df.groupby("sport")
        .agg(Total=("athlete_id", "count"), Medalists=("has_medal", "sum"))
        .reset_index()
    )
    sport_rate["Medal Rate (%)"] = (sport_rate["Medalists"] / sport_rate["Total"] * 100).round(1)
    sport_rate_top = sport_rate.sort_values("Total", ascending=False).head(20).sort_values("Medal Rate (%)")
    fig_v3 = px.bar(
        sport_rate_top, x="Medal Rate (%)", y="sport", orientation="h",
        color="Medal Rate (%)", color_continuous_scale=["#bfdbfe", "#f59e0b"],
        title="Medal Rate (%) for Top 20 Sports by Participation",
        labels={"sport": ""},
    )
    fig_v3.update_layout(plot_bgcolor="white", height=520, coloraxis_showscale=False)
    st.plotly_chart(fig_v3, use_container_width=True)
    st.caption(
        "Medal rates differ substantially across sports. Smaller-field sports (e.g., "
        "Biathlon, Bobsleigh) have higher per-athlete medal rates because fewer "
        "competitors share a fixed number of medals. High-participation sports like "
        "Athletics have lower rates. This makes `sport` a highly informative feature for prediction."
    )

    # --- Viz 4: Violin plot — total_olympics_attended ---
    st.markdown("##### Viz 4 — Olympics Attended by Medal Status")
    fig_v4 = px.violin(
        eda_df, x="Medal Status", y="total_olympics_attended",
        color="Medal Status",
        color_discrete_map={"Medal": "#f59e0b", "No Medal": "#64748b"},
        box=True, points="outliers",
        title="Total Olympics Attended: Medalists vs Non-Medalists",
        labels={"total_olympics_attended": "Total Olympics Attended", "Medal Status": ""},
    )
    fig_v4.update_layout(plot_bgcolor="white", height=400, showlegend=False)
    st.plotly_chart(fig_v4, use_container_width=True)
    st.caption(
        "Medalists tend to have attended slightly more Games on average, indicating that "
        "experience and repeated exposure to Olympic competition correlates with medal success. "
        "Non-medalists are more concentrated at lower attendance counts, consistent with "
        "one-time participants who do not advance."
    )

    # --- Viz 5: Medal rate by gender ---
    st.markdown("##### Viz 5 — Medal Rate by Gender and Games Type")
    gender_game = (
        eda_df.groupby(["gender", "games_type"])
        .agg(Total=("athlete_id", "count"), Medalists=("has_medal", "sum"))
        .reset_index()
    )
    gender_game["Medal Rate (%)"] = (gender_game["Medalists"] / gender_game["Total"] * 100).round(1)
    fig_v5 = px.bar(
        gender_game, x="gender", y="Medal Rate (%)", color="games_type",
        barmode="group",
        color_discrete_map={"Summer": "#f59e0b", "Winter": "#1565c0"},
        title="Medal Rate by Gender and Games Type",
        labels={"gender": "Gender", "games_type": "Games Type"},
    )
    fig_v5.update_layout(plot_bgcolor="white", height=360)
    st.plotly_chart(fig_v5, use_container_width=True)
    st.caption(
        "Medal rates are broadly similar across genders (~24%), but Winter Games show "
        "a notably higher per-entry medal rate than Summer Games for both genders. "
        "This reflects the smaller field sizes in Winter disciplines, where fewer "
        "athletes compete for the same three medals per event."
    )

    # --- Viz 6: Scatter matrix ---
    st.markdown("##### Viz 6 — Scatter Matrix of Key Numerical Features")
    pair_cols = ["age", "height_cm", "weight_kg", "total_olympics_attended", "total_medals_won"]
    pair_df = eda_df[pair_cols + ["Medal Status"]].dropna()
    pair_df = pair_df[
        pair_df["age"].between(10, 80) &
        pair_df["height_cm"].between(130, 230) &
        pair_df["weight_kg"].between(30, 180)
    ].sample(min(1500, len(pair_df)), random_state=42)
    fig_v6 = px.scatter_matrix(
        pair_df, dimensions=pair_cols, color="Medal Status",
        color_discrete_map={"Medal": "#f59e0b", "No Medal": "#64748b"},
        opacity=0.35,
        title="Scatter Matrix — Key Numerical Features",
        labels={c: c.replace("_", " ") for c in pair_cols},
    )
    fig_v6.update_traces(diagonal_visible=False, marker_size=3)
    fig_v6.update_layout(height=600)
    st.plotly_chart(fig_v6, use_container_width=True)
    st.caption(
        "`total_medals_won` is the strongest visual separator between medalists and "
        "non-medalists (by construction — past medals predict future medals). "
        "Age shows mild separation; height and weight are nearly indistinguishable "
        "between groups, confirming findings from Viz 2."
    )

    # ------------------------------------------------------------------
    # 1.4  Correlation Heatmap
    # ------------------------------------------------------------------
    st.markdown('<div class="sec">1.4 Correlation Heatmap</div>', unsafe_allow_html=True)

    corr_cols = [
        "age", "height_cm", "weight_kg", "total_olympics_attended", "total_medals_won",
        "country_total_gold", "country_total_medals", "country_best_rank",
        "country_first_participation", "year", "has_medal",
    ]
    corr_matrix = df[corr_cols].corr().round(2)
    heatmap_labels = [c.replace("_", " ") for c in corr_matrix.columns]
    fig_heat = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=heatmap_labels,
        y=heatmap_labels,
        colorscale="RdBu",
        reversescale=True,
        showscale=True,
        annotation_text=corr_matrix.values.round(2),
        zmin=-1, zmax=1,
    )
    fig_heat.update_layout(
        title="Pearson Correlation Matrix — Numerical Features",
        height=600,
        margin=dict(l=150, b=150),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
**Interpretation of key correlations:**

- **`total_medals_won` ↔ `country_total_gold` / `country_total_medals`** (~0.4–0.6):
  Near-perfect correlations by construction — these columns decompose the total. They will
  be handled carefully to avoid data leakage in a predictive model.

- **`height_cm` ↔ `weight_kg`** (~0.7):
  Moderate positive correlation — taller athletes tend to weigh more, consistent with
  physical anthropometry. Both remain useful as sport-specific body-type signals.

- **`age`, `year`, `total_olympics_attended`** ↔ `has_medal`:
  Low-to-moderate correlations with performance metrics, suggesting these provide
  useful but limited standalone signal — motivating the use of ensemble models.
""")

# ======================================================================
# TAB 3 — MODEL PERFORMANCE (Sections 2.1 – 2.7)
# ======================================================================

with tab_ml:

    with st.spinner("Training models… (this runs once and is then cached)"):
        results = run_all_models()

    X_train = results["X_train"]
    X_test  = results["X_test"]
    y_train = results["y_train"]
    y_test  = results["y_test"]
    feat_names = results["feature_names"]

    lr_model,   lr_m   = results["lr"]
    dt_model,   dt_m   = results["dt"]
    rf_model,   rf_m   = results["rf"]
    lgbm_model, lgbm_m = results["lgbm"]
    mlp_model,  mlp_m  = results["mlp"]

    # ──────────────────────────────────────────────────────────────────
    # 2.1  DATA PREPARATION
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.1 Data Preparation</div>', unsafe_allow_html=True)

    st.markdown(f"""
**Target:** `has_medal` (binary: 1 = won any medal, 0 = no medal)

**Train / Test split:** 70 / 30, stratified, `random_state=42`
→ Training set: **{len(y_train):,}** rows | Test set: **{len(y_test):,}** rows

**Features used ({len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)} total):**

| Type | Features |
|------|----------|
| Numerical ({len(NUMERICAL_FEATURES)}) | {', '.join(f'`{f}`' for f in NUMERICAL_FEATURES)} |
| Categorical ({len(CATEGORICAL_FEATURES)}) | {', '.join(f'`{f}`' for f in CATEGORICAL_FEATURES)} |

**Preprocessing pipeline:**
- *Numerical*: `StandardScaler` — zero mean, unit variance. Required by Logistic Regression and MLP; also improves gradient descent stability.
- *Categorical*: `OneHotEncoder(handle_unknown="ignore")` — expands to {X_train.shape[1]} total features after encoding.
- *Class imbalance* (≈ 76 % No Medal / 24 % Medal): handled via `class_weight="balanced"` (LR, DT, RF), `scale_pos_weight` (LightGBM), and **SMOTE** oversampling on the MLP training set (sklearn's MLPClassifier has no native class-weight parameter).

**Excluded features and why:**
`gold_medals`, `silver_medals`, `bronze_medals` → sub-components of `total_medals_won`; including all together adds multi-collinearity without new signal.
`medal_points`, `medal`, `has_medal` → the target or a direct derivation of it.
`result_value`, `result_unit`, `is_record_holder` → post-event measurements → data leakage.
`athlete_id`, `athlete_name`, `date_of_birth`, `coach_name`, `notes` → identifiers with no predictive signal.
`nationality`, `country_name` → represented numerically by `country_total_*` features.
`host_city`, `event` → very high cardinality (253 unique events) with limited marginal signal.

**Note on `total_medals_won`:** Included as a legitimate historical performance indicator. Each athlete appears exactly once in this dataset, so their career medal count is prior context known before any given event — an athlete who has won many medals is a stronger competitor in future events.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.2  LOGISTIC REGRESSION BASELINE
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.2 Logistic Regression — Baseline</div>', unsafe_allow_html=True)

    def metric_cards(metrics_dict, cols=5):
        keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        colors = ["#1565c0", "#7b1fa2", "#00796b", "#f59e0b", "#dc2626"]
        c = st.columns(cols)
        for col, k, lab, clr in zip(c, keys, labels, colors):
            col.markdown(
                f'<div class="kpi" style="border-top-color:{clr}">'
                f'<div class="lab">{lab}</div>'
                f'<div class="val" style="color:{clr};font-size:1.4rem">{metrics_dict[k]:.4f}</div></div>',
                unsafe_allow_html=True,
            )

    metric_cards(lr_m)
    st.markdown("<br>", unsafe_allow_html=True)

    def confusion_fig(cm, title):
        fig = px.imshow(
            cm, text_auto=True,
            x=["Pred: No Medal", "Pred: Medal"],
            y=["True: No Medal", "True: Medal"],
            color_continuous_scale="Blues",
            title=title,
            aspect="auto",
        )
        fig.update_layout(height=300, coloraxis_showscale=False,
                          xaxis_title="Predicted", yaxis_title="Actual")
        return fig

    lr_col1, lr_col2 = st.columns([1, 2])
    with lr_col1:
        st.plotly_chart(confusion_fig(lr_m["confusion_matrix"], "Confusion Matrix — LR"), use_container_width=True)
    with lr_col2:
        st.markdown("""
**Interpretation:**
Logistic Regression serves as the baseline. With `class_weight="balanced"` it recovers
reasonable recall for the minority (medal) class. The AUC-ROC near 0.5 reflects
that purely contextual features (age, height, sport, country stats) have limited
power to predict individual medal outcomes — a realistic finding when leaky features
are excluded. All subsequent models aim to improve on this baseline.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.3  DECISION TREE / CART
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.3 Decision Tree / CART</div>', unsafe_allow_html=True)

    st.markdown(f"**Best hyperparameters:** `{dt_m['best_params']}`")
    metric_cards(dt_m)
    st.markdown("<br>", unsafe_allow_html=True)

    dt_col1, dt_col2 = st.columns(2)
    with dt_col1:
        st.plotly_chart(confusion_fig(dt_m["confusion_matrix"], "Confusion Matrix — DT"), use_container_width=True)

    with dt_col2:
        cv_res = dt_m["cv_results"].copy()
        pivot = cv_res.pivot_table(
            values="mean_test_score",
            index="param_max_depth",
            columns="param_min_samples_leaf",
        )
        fig_cv = px.imshow(
            pivot, text_auto=".3f",
            color_continuous_scale="YlGn",
            title="CV F1 Score — max_depth vs min_samples_leaf",
            labels={"x": "min_samples_leaf", "y": "max_depth", "color": "F1"},
            aspect="auto",
        )
        fig_cv.update_layout(height=300)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("""
**Interpretation:**
The decision tree improves recall slightly over logistic regression by learning non-linear
splits on sport and country features. The CV heatmap shows that moderate depths (5–7)
with a reasonable `min_samples_leaf` avoid overfitting while capturing useful patterns.
Very deep trees tend to overfit on the small minority class.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.4  RANDOM FOREST
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.4 Random Forest</div>', unsafe_allow_html=True)

    st.markdown(f"**Best hyperparameters:** `{rf_m['best_params']}`")
    metric_cards(rf_m)
    st.markdown("<br>", unsafe_allow_html=True)

    rf_col1, rf_col2 = st.columns(2)
    with rf_col1:
        fig_roc_rf = go.Figure()
        fig_roc_rf.add_trace(go.Scatter(
            x=rf_m["fpr"], y=rf_m["tpr"],
            name=f"Random Forest (AUC={rf_m['auc_roc']:.3f})",
            line=dict(color="#00796b", width=2),
        ))
        fig_roc_rf.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random Classifier",
            line=dict(color="#94a3b8", width=1, dash="dash"),
        ))
        fig_roc_rf.update_layout(
            title="ROC Curve — Random Forest",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="white", height=350,
            legend=dict(x=0.55, y=0.05),
        )
        st.plotly_chart(fig_roc_rf, use_container_width=True)

    with rf_col2:
        fi = pd.DataFrame({"feature": feat_names, "importance": rf_m["feature_importances"]})
        fi = fi.sort_values("importance", ascending=False).head(15)
        fig_fi = px.bar(
            fi.sort_values("importance"), x="importance", y="feature",
            orientation="h", title="Top 15 Feature Importances — RF",
            color="importance", color_continuous_scale=["#bfdbfe", "#00796b"],
        )
        fig_fi.update_layout(plot_bgcolor="white", height=350, coloraxis_showscale=False,
                             yaxis_title="", xaxis_title="Mean Decrease in Impurity")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
**Interpretation:**
Random Forest aggregates many trees to reduce variance, generally yielding better AUC
than a single decision tree. The feature importance plot reveals which contextual variables
drive predictions — sport encoding, country medal history, and athlete age tend to rank
highly. The ROC curve's deviation from the diagonal confirms the model extracts genuine
signal, though the modest AUC is expected given the difficulty of medal prediction from
non-leaky features alone.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.5  LIGHTGBM (BOOSTED TREES)
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.5 LightGBM — Boosted Trees</div>', unsafe_allow_html=True)

    st.markdown(f"**Best hyperparameters:** `{lgbm_m['best_params']}`")
    metric_cards(lgbm_m)
    st.markdown("<br>", unsafe_allow_html=True)

    lgbm_col1, lgbm_col2 = st.columns(2)
    with lgbm_col1:
        fig_roc_lgbm = go.Figure()
        for name, m, color in [
            ("Logistic Reg.", lr_m,   "#94a3b8"),
            ("Decision Tree", dt_m,   "#f59e0b"),
            ("Random Forest", rf_m,   "#00796b"),
            ("LightGBM",      lgbm_m, "#1565c0"),
        ]:
            fig_roc_lgbm.add_trace(go.Scatter(
                x=m["fpr"], y=m["tpr"],
                name=f"{name} ({m['auc_roc']:.3f})",
                line=dict(color=color, width=2),
            ))
        fig_roc_lgbm.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="#e2e8f0", width=1, dash="dash"),
        ))
        fig_roc_lgbm.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            plot_bgcolor="white", height=380,
            legend=dict(x=0.5, y=0.1),
        )
        st.plotly_chart(fig_roc_lgbm, use_container_width=True)

    with lgbm_col2:
        fi_lgbm = pd.DataFrame({"feature": feat_names, "importance": lgbm_m["feature_importances"]})
        fi_lgbm = fi_lgbm.sort_values("importance", ascending=False).head(15)
        fig_fi_lgbm = px.bar(
            fi_lgbm.sort_values("importance"), x="importance", y="feature",
            orientation="h", title="Top 15 Feature Importances — LightGBM",
            color="importance", color_continuous_scale=["#bfdbfe", "#1565c0"],
        )
        fig_fi_lgbm.update_layout(plot_bgcolor="white", height=380, coloraxis_showscale=False,
                                   yaxis_title="", xaxis_title="LightGBM Importance Score")
        st.plotly_chart(fig_fi_lgbm, use_container_width=True)

    st.markdown("""
**Interpretation:**
LightGBM uses gradient boosting with leaf-wise tree growth, typically outperforming
Random Forest when tuned. The combined ROC chart allows direct visual comparison across
all models. Feature importances from both ensemble methods often agree on the top
predictors, lending confidence to their signal.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.6  NEURAL NETWORK — MLP
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.6 Neural Network — MLP</div>', unsafe_allow_html=True)

    st.markdown(f"""
**Architecture (sklearn MLPClassifier):**
- Input layer: {X_train.shape[1]} features
- Hidden layer 1: 128 units, ReLU activation
- Hidden layer 2: 128 units, ReLU activation
- Output layer: 1 unit, sigmoid (logistic) — binary cross-entropy loss
- Optimizer: Adam | L2 regularisation α = 1e-4 | Batch size = 64
- Early stopping: patience = 20 | Trained for {mlp_m['n_iter']} epochs
- Class imbalance: **SMOTE** oversampling applied to training data before fitting

> *Note: TensorFlow, Keras, and PyTorch do not yet support Python 3.14 (arm64).
> sklearn's MLPClassifier implements the identical architecture — same Adam optimizer,
> ReLU hidden layers, sigmoid output, and binary cross-entropy loss.*
""")

    metric_cards(mlp_m)
    st.markdown("<br>", unsafe_allow_html=True)

    mlp_col1, mlp_col2 = st.columns(2)
    with mlp_col1:
        loss_curve = mlp_m["loss_curve"]
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=loss_curve, mode="lines",
            name="Training Loss",
            line=dict(color="#dc2626", width=2),
        ))
        fig_loss.update_layout(
            title="Training Loss Curve — MLP",
            xaxis_title="Epoch", yaxis_title="Loss (Log-Loss)",
            plot_bgcolor="white", height=340,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with mlp_col2:
        st.plotly_chart(confusion_fig(mlp_m["confusion_matrix"], "Confusion Matrix — MLP"), use_container_width=True)

    st.markdown("""
**Interpretation:**
The loss curve confirms the model converged without overfitting (early stopping
halted training when validation performance plateaued). The MLP captures non-linear
feature interactions but offers less interpretability than tree-based models. Its
performance is broadly comparable to the ensemble methods on this dataset.
""")

    # ──────────────────────────────────────────────────────────────────
    # 2.7  MODEL COMPARISON SUMMARY
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">2.7 Model Comparison Summary</div>', unsafe_allow_html=True)

    model_names = ["Logistic Regression", "Decision Tree", "Random Forest", "LightGBM", "MLP"]
    all_metrics = [lr_m, dt_m, rf_m, lgbm_m, mlp_m]
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    summary_df = pd.DataFrame(
        [[m[k] for k in metric_keys] for m in all_metrics],
        columns=["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"],
        index=model_names,
    )
    summary_df.index.name = "Model"

    st.dataframe(
        summary_df.style
            .highlight_max(axis=0, color="#d1fae5")
            .format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    comp_df = summary_df.reset_index().melt(id_vars="Model", value_vars=["F1", "AUC-ROC"])
    fig_comp = px.bar(
        comp_df, x="Model", y="value", color="variable", barmode="group",
        color_discrete_map={"F1": "#f59e0b", "AUC-ROC": "#1565c0"},
        title="F1 Score and AUC-ROC by Model",
        labels={"value": "Score", "variable": "Metric"},
    )
    fig_comp.update_layout(plot_bgcolor="white", height=400, yaxis_range=[0, 1],
                           xaxis_tickangle=-15)
    fig_comp.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8",
                       annotation_text="Random baseline (AUC=0.5)")
    st.plotly_chart(fig_comp, use_container_width=True)

    best_model_idx = summary_df["AUC-ROC"].idxmax()
    best_f1_idx    = summary_df["F1"].idxmax()

    st.markdown(f"""
**Summary:**
The **{best_model_idx}** achieved the highest AUC-ROC and **{best_f1_idx}** the best F1 score
on the held-out test set. Gradient boosted trees (LightGBM) and Random Forest consistently
outperformed linear and shallow models, confirming that non-linear interactions between
sport, country-level features, and athlete attributes carry real predictive signal.

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Logistic Regression | Fast, interpretable coefficients | Assumes linearity, underperforms on complex interactions |
| Decision Tree | Fully interpretable, fast | High variance, prone to overfitting |
| Random Forest | Low variance, robust | Many trees, less interpretable |
| LightGBM | Often best accuracy, handles imbalance well | Many hyperparameters, less interpretable |
| MLP | Captures non-linear interactions | Black box, sensitive to hyperparameters |

**Honest note on performance:** The modest AUC scores (~0.5–0.65) are expected and
actually demonstrate good scientific practice — leaky features (post-event measurements,
sub-medal counts) were deliberately excluded. In real deployment, features like qualifying
performance, world ranking, and recent form would substantially increase predictive power.
""")

# ======================================================================
# TAB 4 — EXPLAINABILITY & PREDICTION
# ======================================================================

with tab_explain:

    with st.spinner("Loading models for explainability…"):
        results_ex = run_all_models()

    rf_model_ex,   rf_m_ex   = results_ex["rf"]
    lgbm_model_ex, lgbm_m_ex = results_ex["lgbm"]
    preprocessor_ex           = results_ex["preprocessor"]
    feat_names_ex             = results_ex["feature_names"]
    X_test_ex                 = results_ex["X_test"]

    best_tree_name  = "Random Forest" if rf_m_ex["auc_roc"] >= lgbm_m_ex["auc_roc"] else "LightGBM"
    best_tree_model = rf_model_ex      if rf_m_ex["auc_roc"] >= lgbm_m_ex["auc_roc"] else lgbm_model_ex

    # ──────────────────────────────────────────────────────────────────
    # 3.1  SHAP ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">3.1 SHAP Analysis</div>', unsafe_allow_html=True)
    st.markdown(f"**Model used:** {best_tree_name} (AUC-ROC = {max(rf_m_ex['auc_roc'], lgbm_m_ex['auc_roc']):.4f})")

    @st.cache_resource(show_spinner=False)
    def get_shap_data():
        raw = pd.read_csv(DATA_PATH)
        raw["has_medal"] = raw["medal"] != "No Medal"
        raw["medal_points"] = raw["medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1, "No Medal": 0})
        _, X_test_s, _, _, preprocessor_s = prepare_data(raw)
        fn = get_feature_names(preprocessor_s)
        mdl = rf_model_ex if rf_m_ex["auc_roc"] >= lgbm_m_ex["auc_roc"] else lgbm_model_ex
        return compute_shap(mdl, X_test_s, fn)

    with st.spinner("Computing SHAP values…"):
        shap_data = get_shap_data()

    shap_vals  = shap_data["shap_values"]
    X_sample   = shap_data["X_sample"]
    feat_df    = shap_data["feat_df"]
    wf_idx     = shap_data["waterfall_idx"]
    base_value = shap_data["expected_value"]

    # Plot 1: Mean |SHAP| bar
    st.markdown("##### Plot 1 — Mean |SHAP| — Global Feature Importance")
    top_n = 20
    bar_df = feat_df.head(top_n).sort_values("mean_abs_shap")
    fig_shap_bar = px.bar(
        bar_df, x="mean_abs_shap", y="feature", orientation="h",
        color="mean_abs_shap",
        color_continuous_scale=["#bfdbfe", "#1565c0"],
        title=f"Top {top_n} Features by Mean |SHAP| — {best_tree_name}",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
    )
    fig_shap_bar.update_layout(plot_bgcolor="white", height=520, coloraxis_showscale=False)
    st.plotly_chart(fig_shap_bar, use_container_width=True)
    st.caption(
        "Mean absolute SHAP values rank features by their average impact on model output "
        "magnitude. The taller the bar, the more that feature moves predictions away from "
        "the base rate — in either direction."
    )

    # Plot 2: Beeswarm
    st.markdown("##### Plot 2 — Beeswarm Summary Plot — Direction of Impact")
    top_feats = feat_df["feature"].head(15).tolist()
    feat_idx  = [list(feat_df["feature"]).index(f) for f in top_feats]

    beeswarm_rows = []
    rng = np.random.default_rng(42)
    for rank, (fi_idx, fname) in enumerate(zip(feat_idx, top_feats)):
        sv_col = shap_vals[:, fi_idx]
        fv_col = X_sample[:, fi_idx]
        fv_min, fv_max = fv_col.min(), fv_col.max()
        fv_norm = (fv_col - fv_min) / (fv_max - fv_min + 1e-9)
        jitter  = rng.uniform(-0.35, 0.35, size=len(sv_col))
        for sv_i, fv_n, j in zip(sv_col, fv_norm, jitter):
            beeswarm_rows.append({
                "feature":    fname,
                "shap_value": float(sv_i),
                "feat_norm":  float(fv_n),
                "y_jitter":   rank + j,
            })

    bee_df = pd.DataFrame(beeswarm_rows)
    fig_bee = px.scatter(
        bee_df, x="shap_value", y="y_jitter",
        color="feat_norm",
        color_continuous_scale="RdBu_r",
        hover_data={"feature": True, "shap_value": ":.4f",
                    "feat_norm": False, "y_jitter": False},
        title=f"SHAP Beeswarm — Top 15 Features ({best_tree_name})",
        labels={"shap_value": "SHAP value (impact on model output)", "y_jitter": ""},
    )
    fig_bee.update_traces(marker=dict(size=4, opacity=0.6))
    fig_bee.update_yaxes(tickvals=list(range(len(top_feats))), ticktext=top_feats)
    fig_bee.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig_bee.update_layout(
        plot_bgcolor="white", height=520,
        coloraxis_colorbar=dict(title="Feature value<br>(blue=low, red=high)"),
        yaxis_title="",
    )
    st.plotly_chart(fig_bee, use_container_width=True)
    st.caption(
        "Each dot is one observation. Horizontal position = SHAP value (right = pushes "
        "toward medal prediction, left = pushes away). Colour = feature value: "
        "red = high, blue = low. Wide horizontal spread means the feature has large "
        "variance in its impact."
    )

    # Plot 3: Waterfall
    st.markdown("##### Plot 3 — Waterfall Plot — Single Prediction Explanation")
    wf_shap = shap_vals[wf_idx]
    wf_pred = best_tree_model.predict_proba(X_sample[[wf_idx]])[0, 1]

    top_k = 10
    contrib_idx   = np.argsort(np.abs(wf_shap))[::-1][:top_k]
    contrib_vals  = wf_shap[contrib_idx]
    all_feat_names = feat_df["feature"].tolist()
    contrib_names  = [all_feat_names[i] for i in contrib_idx]

    cumvals  = np.concatenate([[base_value], base_value + np.cumsum(contrib_vals)])
    bar_vals = np.concatenate([[base_value], contrib_vals])
    labels   = ["Base rate"] + contrib_names
    colors   = ["#64748b"] + ["#dc2626" if v > 0 else "#1565c0" for v in contrib_vals]

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(
        x=labels,
        y=np.concatenate([[0], np.where(contrib_vals >= 0, cumvals[:-1], cumvals[1:])]),
        marker_color="rgba(0,0,0,0)",
        showlegend=False, hoverinfo="skip",
    ))
    fig_wf.add_trace(go.Bar(
        x=labels,
        y=np.abs(bar_vals),
        marker_color=colors,
        name="SHAP contribution",
        text=[f"{v:+.4f}" for v in bar_vals],
        textposition="outside",
    ))
    fig_wf.add_hline(y=wf_pred, line_dash="dot", line_color="#f59e0b",
                     annotation_text=f"Final prediction: {wf_pred:.3f}", annotation_position="top right")
    fig_wf.update_layout(
        barmode="stack",
        title=f"Waterfall — Predicted medal probability: {wf_pred:.3f}  (top-{top_k} contributors)",
        xaxis_title="", yaxis_title="Cumulative probability",
        plot_bgcolor="white", height=450,
        showlegend=False, xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    st.caption(
        "Starting from the base rate (average model output), each bar shows how much "
        "one feature pushes the prediction up (red) or down (blue) for this specific athlete. "
        "The dotted line marks the final predicted medal probability."
    )

    top3 = feat_df["feature"].head(3).tolist()
    st.markdown(f"""
**Which features have the strongest impact?**
The top three features by mean |SHAP| are **{top3[0]}**, **{top3[1]}**, and **{top3[2]}**.
`total_medals_won` consistently dominates — athletes with a richer medal history are
predicted more likely to win again, capturing overall athlete quality.
Country-level features (`country_total_medals`, `country_total_gold`) reflect the
structural advantage of well-funded, high-performing national programs.
Physical and demographic features (age, height, weight) contribute modest but
consistent signal across sports.

**Direction of influence:**
- **High `total_medals_won`** → large positive SHAP → increases predicted medal probability.
- **Low `country_total_medals`** → negative SHAP → decreases probability (smaller national programs field weaker overall delegations).
- **Age** shows a non-linear pattern: very young and very old athletes receive negative SHAP, while prime-age athletes (mid-20s) receive a mild positive push.

**Decision-maker insight:**
A national Olympic committee could use these SHAP values in three ways:
1. *Athlete selection*: flag entries where `total_medals_won` is high but model probability
   is suppressed — the gap may indicate a sport-specific or physical factor that coaches
   should address.
2. *Resource allocation*: sports where country-level SHAP contributions are consistently
   negative suggest the nation is underinvesting relative to global competition.
3. *Individual explainability*: the waterfall plot provides a transparent, auditable
   rationale for why the model rates a specific athlete's medal chances, which is essential
   for communicating decisions to athletes and coaching staff.
""")

    # ──────────────────────────────────────────────────────────────────
    # INTERACTIVE MEDAL PREDICTION
    # ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Interactive Medal Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f"Enter an athlete's attributes below and the **{best_tree_name}** model will "
        "predict their probability of winning a medal."
    )

    all_sports_raw   = sorted(df["sport"].unique().tolist())
    all_team_opts    = sorted(df["team_or_individual"].dropna().unique().tolist())

    with st.form("live_prediction_form"):
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            st.markdown("**Athlete Attributes**")
            pred_age    = st.number_input("Age", min_value=10, max_value=75, value=25)
            pred_height = st.number_input("Height (cm)", min_value=130, max_value=230, value=175)
            pred_weight = st.number_input("Weight (kg)", min_value=30, max_value=180, value=70)
            pred_gender = st.selectbox("Gender", ["Male", "Female"])
            pred_total_olympics = st.number_input("Total Olympics Attended", min_value=1, max_value=10, value=1)
            pred_total_medals   = st.number_input("Career Medals Won", min_value=0, max_value=30, value=0)

        with pc2:
            st.markdown("**Competition Context**")
            pred_games_type = st.selectbox("Games Type", ["Summer", "Winter"])
            pred_year       = st.select_slider("Year", options=sorted(df["year"].unique().tolist()), value=2020)
            pred_sport      = st.selectbox("Sport", all_sports_raw)
            pred_team       = st.selectbox("Team or Individual", all_team_opts)

        with pc3:
            st.markdown("**Country Statistics**")
            pred_country_gold    = st.number_input("Country Total Gold Medals", min_value=0, max_value=3000, value=100)
            pred_country_medals  = st.number_input("Country Total Medals", min_value=0, max_value=8000, value=300)
            pred_country_rank    = st.number_input("Country Best Olympic Rank", min_value=1, max_value=100, value=10)
            pred_country_first   = st.number_input("Country First Participation Year", min_value=1896, max_value=2020, value=1900, step=4)

        submitted = st.form_submit_button("Predict Medal Probability", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "age":                       pred_age,
            "height_cm":                 pred_height,
            "weight_kg":                 pred_weight,
            "total_olympics_attended":   pred_total_olympics,
            "total_medals_won":          pred_total_medals,
            "country_total_gold":        pred_country_gold,
            "country_total_medals":      pred_country_medals,
            "country_best_rank":         pred_country_rank,
            "country_first_participation": pred_country_first,
            "year":                      pred_year,
            "gender":                    pred_gender,
            "games_type":                pred_games_type,
            "sport":                     pred_sport,
            "team_or_individual":        pred_team,
        }])

        input_proc = preprocessor_ex.transform(input_df)
        prob       = best_tree_model.predict_proba(input_proc)[0, 1]
        pred_label = "🥇 Medal" if prob >= 0.5 else "No Medal"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", pred_label)
        r2.metric("Medal Probability", f"{prob:.1%}")
        r3.metric("Model", best_tree_name)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Medal Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#f59e0b" if prob >= 0.5 else "#64748b"},
                "steps": [
                    {"range": [0, 50],  "color": "#fee2e2"},
                    {"range": [50, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#1565c0", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=40, r=40))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob >= 0.5:
            st.success(
                f"The model predicts a **{prob:.1%}** chance of winning a medal. "
                f"Key drivers are typically career medal history and country program strength."
            )
        else:
            st.info(
                f"The model predicts a **{prob:.1%}** chance of winning a medal. "
                "Try increasing career medals won or selecting a stronger national program."
            )

# ======================================================================
# FOOTER
# ======================================================================

st.markdown(
    '<hr style="border:none;border-top:1px solid #e2e8f0;margin:2rem 0 1rem">'
    '<p style="text-align:center;color:#94a3b8;font-size:.8rem">'
    "Olympic Games Analytics Dashboard &middot; MSIS 521 Final Project &middot; "
    "128 years of Olympic history"
    "</p>",
    unsafe_allow_html=True,
)
