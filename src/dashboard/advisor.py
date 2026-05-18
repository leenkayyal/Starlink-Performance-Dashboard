import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Starlink Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# FILE PATHS
# ============================================================
PERIOD1_CLEAN_FILE = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
PERIOD2_CLEAN_FILE = "Cleaned/experiment_B/Starlink_2_cleaned.csv"

THESIS_START = pd.Timestamp("2026-03-07 01:00:00")
THESIS_END = pd.Timestamp("2026-05-12 07:45:00")

AI_MODEL_FILE = "Models/starlink_latency_forecast_model.pkl"
AI_FEATURES_FILE = "Models/starlink_latency_features.pkl"

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@600;700;800&display=swap');

:root {
    --ink: #183B4A;
    --ink-2: #265868;
    --cream: #FFF8EE;
    --paper: #FFFFFF;
    --line: #E8DED0;
    --muted: #6B7B83;
    --orange: #FF7330;
    --yellow: #FFC845;
    --sage: #A7C983;
    --teal: #5EA38F;
    --success: #3D8B65;
    --warning: #D99A20;
    --danger: #D95B4F;
}

html, body, .stApp {
    background:
        radial-gradient(circle at 9% 7%, rgba(255,115,48,0.14) 0, transparent 25%),
        radial-gradient(circle at 88% 5%, rgba(94,163,143,0.20) 0, transparent 28%),
        radial-gradient(circle at 75% 92%, rgba(255,200,69,0.16) 0, transparent 28%),
        linear-gradient(135deg, #FFF9EF 0%, #F6F9ED 43%, #EEF8F4 100%) fixed !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
}
[class*="css"], .main, .block-container {
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}
.block-container {
    padding-top: 2.2rem !important;
    padding-left: 2.4rem !important;
    padding-right: 2.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}

h1, h2, h3 {
    color: var(--ink) !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    letter-spacing: -0.03em !important;
}
p, li, span, label { color: var(--ink); }
hr { border-color: rgba(24,59,74,0.12) !important; }

/* top app bar */
header[data-testid="stHeader"] {
    background: rgba(255,248,238,0.86) !important;
    backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(24,59,74,0.10) !important;
}
header[data-testid="stHeader"] * { color: var(--ink) !important; }
[data-testid="stDecoration"] {
    background: linear-gradient(90deg, #183B4A, #265868, #3F7F78, #5EA38F) !important;
    height: 5px !important;
}

/* login and general panels */
div[data-testid="stVerticalBlock"] > div:has(.login-hero) {
    max-width: 520px !important;
    margin: 7vh auto 0 auto !important;
    background: rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    border-radius: 30px !important;
    padding: 2.1rem 2.2rem 2.4rem !important;
    box-shadow: 0 26px 70px rgba(24,59,74,0.18) !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="stVerticalBlock"] > div:has(.login-hero)::before,
.card::before, .advisor-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 7px;
    background: linear-gradient(90deg, #183B4A, #265868, #3F7F78, #5EA38F);
}
.login-hero {
    background: linear-gradient(135deg, rgba(24,59,74,0.98), rgba(38,88,104,0.96));
    color: #FFFFFF !important;
    border-radius: 24px;
    padding: 1.45rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 18px 40px rgba(24,59,74,0.22);
}
.login-hero h1, .login-hero h2, .login-hero p { color: #FFFFFF !important; margin: 0 !important; }
.login-chip {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
    background: rgba(255,200,69,0.18);
    color: #FFE7A2 !important;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    border-right: 0 !important;
    box-shadow: 8px 0 34px rgba(24,59,74,0.20) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.90) !important;
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* inputs */
.stTextInput input,
[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.96) !important;
    border: 1.5px solid rgba(24,59,74,0.14) !important;
    border-radius: 16px !important;
    color: var(--ink) !important;
    box-shadow: 0 6px 18px rgba(24,59,74,0.05) !important;
}
.stTextInput input:focus,
[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 4px rgba(255,115,48,0.13) !important;
}
[data-baseweb="popover"] { color: var(--ink) !important; }
[data-baseweb="menu"] { background: #FFFFFF !important; }
[data-baseweb="menu"] li { color: var(--ink) !important; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 16px !important;
    font-weight: 900 !important;
    padding: 0.75rem 1.4rem !important;
    box-shadow: 0 14px 28px rgba(24,59,74,0.24) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 18px 38px rgba(24,59,74,0.32) !important;
}

/* KPI switch / toggles - clearer and fully visible */
[data-testid="stToggle"] label,
[data-testid="stToggle"] p,
[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
}
[data-testid="stToggle"] [role="switch"] {
    min-width: 48px !important;
    height: 26px !important;
    border: 2px solid rgba(24,59,74,0.15) !important;
    box-shadow: inset 0 2px 6px rgba(24,59,74,0.14), 0 4px 10px rgba(24,59,74,0.10) !important;
}
[data-testid="stToggle"] [role="switch"][aria-checked="false"] {
    background: #D9E4DF !important;
}
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    border-color: rgba(94,163,143,0.55) !important;
}
[data-testid="stToggle"] [role="switch"] > div {
    background: #FFFFFF !important;
    box-shadow: 0 3px 10px rgba(24,59,74,0.24) !important;
}

/* sliders */
[data-testid="stSlider"] [role="slider"] {
    background: var(--orange) !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 0 0 5px rgba(255,115,48,0.16) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--orange) !important; }

/* cards and metrics */
.card, .advisor-card, [data-testid="metric-container"], [data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.90) !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    border-radius: 24px !important;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10) !important;
    backdrop-filter: blur(12px) !important;
    position: relative !important;
    overflow: hidden !important;
}
.card {
    padding: 1.35rem 1.45rem !important;
    margin-bottom: 0.8rem !important;
}
.card-title, .step-label, .q-label {
    display: inline-block;
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    box-shadow: 0 10px 22px rgba(24,59,74,0.22) !important;
    border-radius: 999px !important;
    padding: 0.32rem 0.78rem !important;
    font-size: 0.68rem !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    font-family: 'IBM Plex Mono', monospace !important;
    margin-bottom: 0.75rem !important;
    border-left: none !important;
}
.advisor-card {
    padding: 1.35rem 1.55rem !important;
    margin-bottom: 1rem !important;
}
.small-note { color: var(--muted) !important; font-size: 0.84rem; line-height: 1.65; }

[data-testid="metric-container"] {
    padding: 1.05rem !important;
    border-top: 6px solid var(--teal) !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-size: 0.66rem !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] div,
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
    font-family: 'Space Grotesk', 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--success) !important;
    font-weight: 800 !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.82) !important;
    border-radius: 999px !important;
    padding: 6px !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    gap: 5px !important;
    box-shadow: 0 10px 30px rgba(24,59,74,0.10) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--ink-2) !important;
    font-weight: 900 !important;
    border-radius: 999px !important;
    padding: 0.62rem 1.2rem !important;
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(94,163,143,0.14) !important; color: var(--ink-2) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--ink), var(--teal)) !important;
    color: #FFFFFF !important;
    box-shadow: 0 10px 22px rgba(24,59,74,0.24) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.35) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255,255,255,0.40) !important;
    padding: 1rem 0.5rem !important;
}

/* dataframe */
[data-testid="stDataFrameResizable"], .dvn-scroller, [data-testid="stDataFrame"] canvas { background: #FFFFFF !important; }
.stDataFrame thead th {
    background: #FFF2DB !important;
    color: var(--ink) !important;
    font-weight: 900 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}
.stDataFrame tbody td { color: var(--ink) !important; background: #FFFFFF !important; }

/* multiselect tags */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(24,59,74,0.10), rgba(94,163,143,0.16)) !important;
    border: 1px solid rgba(94,163,143,0.35) !important;
    color: var(--ink-2) !important;
    border-radius: 999px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: var(--ink-2) !important; font-weight: 800 !important; }

/* alerts / status */
.stAlert {
    border-radius: 18px !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    box-shadow: 0 10px 25px rgba(24,59,74,0.08) !important;
}
.stSpinner > div { border-color: var(--orange) transparent transparent transparent !important; border-width: 3px !important; }
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] svg { color: var(--orange) !important; fill: var(--orange) !important; stroke: var(--orange) !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--orange), var(--yellow), var(--teal)) !important; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #183B4A, #265868, #5EA38F) !important; }
.stDownloadButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 16px !important;
    font-weight: 900 !important;
    padding: 0.75rem 1.4rem !important;
    box-shadow: 0 14px 28px rgba(24,59,74,0.22) !important;
}
.pretty-table-wrap {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(24,59,74,0.10);
    border-radius: 24px;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10);
    padding: 0.35rem 0.55rem;
    overflow-x: auto;
    overflow-y: auto;
    margin: 0.35rem 0 1rem 0;
}
.pretty-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.pretty-table thead th {
    background: #F1F8F5;
    color: var(--ink);
    font-weight: 900;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.82rem 0.78rem;
    border-bottom: 1px solid rgba(24,59,74,0.10);
    text-align: left;
    white-space: nowrap;
}
.pretty-table tbody td {
    color: var(--ink);
    padding: 0.78rem 0.78rem;
    border-bottom: 1px solid rgba(24,59,74,0.08);
    background: rgba(255,255,255,0.94);
    vertical-align: top;
}
.pretty-table tbody tr:nth-child(even) td { background: rgba(94,163,143,0.05); }
.pretty-table tbody tr:hover td { background: rgba(255,200,69,0.10); }

/* ── Advisor colorful dashboard polish ───────────────────── */
.advisor-hero {
    background:
        radial-gradient(circle at 86% 18%, rgba(255,200,69,0.22) 0, transparent 20%),
        radial-gradient(circle at 8% 80%, rgba(94,163,143,0.18) 0, transparent 26%),
        linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,248,238,0.94) 45%, rgba(238,248,244,0.96));
    border-radius: 24px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(24,59,74,0.10);
    border-top: 7px solid #265868;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10);
    position: relative;
    overflow: hidden;
}
.advisor-hero::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 7px;
    background: linear-gradient(90deg,#183B4A,#265868,#5EA38F,#A7C983,#FFC845,#FF7330);
}
.advisor-hero h1 {
    margin: 0 !important;
    font-size: 2.25rem !important;
    line-height: 1.05 !important;
    font-weight: 900 !important;
}
.advisor-hero .hero-sub {
    color: #6B7B83 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    margin-top: 0.55rem;
}
.hero-chip {
    display: inline-block;
    background: linear-gradient(135deg, rgba(255,115,48,0.14), rgba(255,200,69,0.18));
    color: #D85E24 !important;
    border: 1px solid rgba(255,115,48,0.28);
    border-radius: 999px;
    padding: 0.35rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 900;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.step-card {
    border-radius: 18px;
    padding: 0.8rem 1rem;
    text-align: center;
    font-size: 0.82rem;
    font-weight: 900;
    font-family: 'IBM Plex Mono', monospace;
    box-shadow: 0 12px 26px rgba(24,59,74,0.10);
}
.step-active {
    background: linear-gradient(135deg,#FF7330 0%,#FFC845 52%,#5EA38F 120%);
    color: #FFFFFF !important;
    border: 0;
}
.step-done {
    background: linear-gradient(135deg,#EAF7F2,#FFFFFF);
    color: #3D8B65 !important;
    border: 1px solid rgba(94,163,143,0.32);
}
.step-wait {
    background: rgba(255,255,255,0.86);
    color: #183B4A !important;
    border: 1px solid rgba(24,59,74,0.10);
}
.condition-shell {
    background:
        radial-gradient(circle at 0% 0%, rgba(255,115,48,0.10), transparent 25%),
        radial-gradient(circle at 100% 100%, rgba(94,163,143,0.12), transparent 28%),
        rgba(255,255,255,0.48);
    border: 1px solid rgba(24,59,74,0.08);
    border-radius: 28px;
    padding: 1.25rem;
    box-shadow: 0 18px 45px rgba(24,59,74,0.08);
}
.q-label {
    background: linear-gradient(135deg,#183B4A 0%,#265868 55%,#5EA38F 120%) !important;
    box-shadow: 0 10px 22px rgba(24,59,74,0.18) !important;
}
.q-label.q-orange { background: linear-gradient(135deg,#FF7330,#FFC845) !important; }
.q-label.q-sage { background: linear-gradient(135deg,#5EA38F,#A7C983) !important; }
.q-label.q-yellow { background: linear-gradient(135deg,#D99A20,#FFC845) !important; }
.card-title, .step-label {
    background: linear-gradient(135deg,#173C4A 0%,#265868 58%,#5EA38F 125%) !important;
    box-shadow: 0 10px 22px rgba(24,59,74,0.18) !important;
}
.stButton > button {
    background: linear-gradient(135deg,#FF7330 0%,#FFC845 48%,#5EA38F 120%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 16px 34px rgba(255,115,48,0.22) !important;
}
.stButton > button:hover {
    box-shadow: 0 18px 42px rgba(94,163,143,0.28) !important;
}
.metric-pop {
    transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.metric-pop:hover {
    transform: translateY(-2px);
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(94,163,143,0.18), rgba(167,201,131,0.20)) !important;
    border: 1px solid rgba(94,163,143,0.42) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #265868 !important;
}
.pretty-table-wrap {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(24,59,74,0.10);
    border-radius: 24px;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10);
    padding: 0.35rem 0.55rem;
    overflow-x: auto;
    margin: 0.35rem 0 1rem 0;
}
.pretty-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.pretty-table thead th {
    background: #F1F8F5;
    color: #183B4A;
    font-weight: 900;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.82rem 0.78rem;
    border-bottom: 1px solid rgba(24,59,74,0.10);
    text-align: left;
    white-space: nowrap;
}
.pretty-table tbody td {
    color: #183B4A;
    padding: 0.78rem 0.78rem;
    border-bottom: 1px solid rgba(24,59,74,0.08);
    background: rgba(255,255,255,0.94);
}
.pretty-table tbody tr:nth-child(even) td {
    background: rgba(94,163,143,0.05);
}


/* ── Darker premium advisor redesign ─────────────────────── */
html, body, .stApp {
    background:
        radial-gradient(circle at 18% 10%, rgba(255,115,48,0.10) 0, transparent 24%),
        radial-gradient(circle at 88% 12%, rgba(94,163,143,0.14) 0, transparent 26%),
        linear-gradient(135deg, #F9F2E8 0%, #EEF4EF 48%, #E7F2EF 100%) fixed !important;
}

/* Header: calmer and darker, less playful */
.advisor-hero {
    background:
        linear-gradient(135deg, #173C4A 0%, #265868 56%, #3F7F78 100%) !important;
    border-radius: 24px !important;
    padding: 1.75rem 2rem !important;
    margin-bottom: 1.25rem !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-top: 0 !important;
    box-shadow: 0 22px 55px rgba(24,59,74,0.24) !important;
}
.advisor-hero::before {
    height: 6px !important;
    background: linear-gradient(90deg, #5EA38F, #A7C983, #FFC845, #FF7330) !important;
}
.advisor-hero h1 span,
.advisor-hero h1 {
    color: #FFFFFF !important;
}
.advisor-hero .hero-sub {
    color: rgba(255,255,255,0.78) !important;
}
.hero-chip {
    background: rgba(255,255,255,0.12) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.24) !important;
}

/* Main section shell: less empty, more dashboard-card */
.condition-shell {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.82), rgba(238,248,244,0.70)) !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    border-radius: 28px !important;
    padding: 1.35rem 1.45rem 1.55rem !important;
    box-shadow: 0 22px 55px rgba(24,59,74,0.12) !important;
}

/* Progress steps: darker selected, cleaner inactive */
.step-card {
    border-radius: 16px !important;
    padding: 0.88rem 1rem !important;
    box-shadow: 0 12px 28px rgba(24,59,74,0.10) !important;
}
.step-active {
    background: linear-gradient(135deg, #173C4A 0%, #265868 62%, #5EA38F 120%) !important;
    color: #FFFFFF !important;
}
.step-done {
    background: linear-gradient(135deg, #3D8B65, #5EA38F) !important;
    color: #FFFFFF !important;
    border: 0 !important;
}
.step-wait {
    background: rgba(255,255,255,0.90) !important;
    color: #183B4A !important;
    border: 1px solid rgba(24,59,74,0.12) !important;
}

/* Labels: unified darker look, with tiny colored left accent */
.q-label, .card-title, .step-label {
    background: linear-gradient(135deg, #173C4A 0%, #265868 60%, #3F7F78 100%) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 0.42rem 0.82rem !important;
    box-shadow: 0 10px 22px rgba(24,59,74,0.18) !important;
}
.q-label.q-orange,
.q-label.q-yellow,
.q-label.q-sage {
    background: linear-gradient(135deg, #173C4A 0%, #265868 60%, #3F7F78 100%) !important;
    position: relative !important;
}
.q-label.q-orange::before,
.q-label.q-yellow::before,
.q-label.q-sage::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 0.55rem;
    border-radius: 50%;
    background: #FF7330;
    box-shadow: 0 0 12px rgba(255,115,48,0.55);
}
.q-label.q-yellow::before { background: #FFC845; box-shadow: 0 0 12px rgba(255,200,69,0.55); }
.q-label.q-sage::before { background: #A7C983; box-shadow: 0 0 12px rgba(167,201,131,0.55); }

/* Inputs: more defined and less flat */
.stTextInput input,
[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.98) !important;
    border: 1.5px solid rgba(24,59,74,0.16) !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 24px rgba(24,59,74,0.08) !important;
}
.stTextInput input:focus,
[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: #5EA38F !important;
    box-shadow: 0 0 0 4px rgba(94,163,143,0.16) !important;
}

/* Buttons: dark gradient, not candy colored */
.stButton > button {
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    box-shadow: 0 16px 34px rgba(24,59,74,0.24) !important;
}
.stButton > button:hover {
    box-shadow: 0 18px 44px rgba(24,59,74,0.32) !important;
}

/* Multi-select tags: clean dark teal chips */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(24,59,74,0.10), rgba(94,163,143,0.16)) !important;
    border: 1px solid rgba(38,88,104,0.28) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #183B4A !important;
    font-weight: 900 !important;
}

/* Metric cards: more serious dashboard cards */
.metric-pop,
.advisor-card,
.card,
[data-testid="metric-container"] {
    box-shadow: 0 18px 45px rgba(24,59,74,0.11) !important;
}
.metric-pop:hover {
    transform: translateY(-2px);
}

/* Tables: match dashboard, not plain Streamlit */
.pretty-table-wrap {
    background: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    border-radius: 22px !important;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10) !important;
}
.pretty-table thead th {
    background: linear-gradient(135deg, #EEF8F4, #F7FBF8) !important;
}


/* Fix Step 3 number-input steppers and action buttons */
.stButton > button,
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
}

[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.98) !important;
    color: #265868 !important;
    border: 1.5px solid rgba(24,59,74,0.16) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 24px rgba(24,59,74,0.08) !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #EEF8F4 !important;
    color: #173C4A !important;
    border-color: rgba(94,163,143,0.45) !important;
}

[data-testid="stNumberInput"] button svg,
[data-testid="stNumberInput"] button span,
[data-testid="stNumberInput"] button p {
    color: #265868 !important;
    fill: #265868 !important;
}

[data-testid="stNumberInput"] input {
    color: #183B4A !important;
}




/* Softer question label typography */
.q-label,
.q-label.q-orange,
.q-label.q-yellow,
.q-label.q-sage {
    background: transparent !important;
    box-shadow: none !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin-top: 0.45rem !important;
    margin-bottom: 0.6rem !important;
    color: #3D8B65 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    line-height: 1.35 !important;
    display: block !important;
}

.q-label::before,
.q-label.q-orange::before,
.q-label.q-yellow::before,
.q-label.q-sage::before {
    content: none !important;
    display: none !important;
}


/* Lighter placeholder look for unselected select boxes */
div[data-baseweb="select"] input::placeholder {
    color: #A7B1B7 !important;
    opacity: 1 !important;
}


/* Softer borders for Step 3 number inputs */
[data-testid="stNumberInput"] * {
    outline: none !important;
}

[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

[data-testid="stNumberInput"] input {
    border: 1px solid rgba(24,59,74,0.10) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 18px rgba(24,59,74,0.05) !important;
}

[data-testid="stNumberInput"] button {
    border: 1px solid rgba(24,59,74,0.10) !important;
    box-shadow: 0 8px 18px rgba(24,59,74,0.05) !important;
}

[data-testid="stNumberInput"] button + button {
    margin-left: 4px !important;
}


/* Multi-select placeholder */
[data-testid="stMultiSelect"] input::placeholder {
    color: #A7B1B7 !important;
    opacity: 1 !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div {
    min-height: 48px !important;
}


/* More visible placeholders */
div[data-baseweb="select"] input::placeholder,
[data-testid="stMultiSelect"] input::placeholder,
[data-testid="stMultiSelect"] div[role="combobox"] input::placeholder {
    color: #8B959C !important;
    opacity: 1 !important;
}

/* Slightly more spacing under question titles before the input boxes */
.q-label,
.q-label.q-orange,
.q-label.q-yellow,
.q-label.q-sage {
    margin-bottom: 0.9rem !important;
}

/* Push the select / multiselect boxes a little lower */
.stSelectbox,
[data-testid="stMultiSelect"],
[data-testid="stNumberInput"] {
    margin-top: 0.18rem !important;
}


/* Stronger placeholder visibility for select and multiselect */
[data-testid="stSelectbox"] div[data-baseweb="select"] [id*="placeholder"],
[data-testid="stSelectbox"] div[data-baseweb="select"] [class*="placeholder"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] [id*="placeholder"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] [class*="placeholder"],
[data-testid="stSelectbox"] div[data-baseweb="select"] input::placeholder,
[data-testid="stMultiSelect"] div[data-baseweb="select"] input::placeholder {
    color: #8A949B !important;
    opacity: 1 !important;
}

/* Keep chosen values dark after selection */
[data-testid="stSelectbox"] div[data-baseweb="select"] [class*="singleValue"],
[data-testid="stMultiSelect"] div[data-baseweb="select"] [class*="singleValue"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #183B4A !important;
}


/* Gray manual placeholder overlays for empty select boxes */
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"] {
    position: relative !important;
}


/* Keep the multiselect field the same visual height as the other fields */
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div {
    min-height: 40px !important;
    height: 40px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    display: flex !important;
    align-items: center !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
LOCATION_ADJ = {
    "City (Muscat or major urban area)":   {"lat": 0,  "dl": 1.00},
    "Suburb (outside city centre)":        {"lat": 3,  "dl": 0.95},
    "Rural area":                          {"lat": 6,  "dl": 0.90},
    "Desert / very remote area":           {"lat": 10, "dl": 0.82},
}
SKY_ADJ = {
    "Completely clear, no obstructions":   {"lat": 0,  "dl": 1.00},
    "Mostly clear, minor trees or walls":  {"lat": 3,  "dl": 0.95},
    "Partially blocked, buildings nearby": {"lat": 8,  "dl": 0.88},
    "Heavily obstructed":                  {"lat": 18, "dl": 0.70},
}
TIME_ADJ = {
    "Morning (6 AM – 12 PM)":              {"lat": 0,  "dl": 1.00},
    "Afternoon (12 PM – 6 PM)":            {"lat": 2,  "dl": 0.97},
    "Evening (6 PM – 10 PM)":              {"lat": 8,  "dl": 0.88},
    "Night (10 PM – 6 AM)":                {"lat": -2, "dl": 1.05},
}
WEATHER_ADJ = {
    "Clear / sunny":                       {"lat": 0,  "dl": 1.00},
    "Partly cloudy":                       {"lat": 2,  "dl": 0.97},
    "Overcast / cloudy":                   {"lat": 4,  "dl": 0.93},
    "Light rain or drizzle":               {"lat": 7,  "dl": 0.88},
    "Heavy rain or thunderstorm":          {"lat": 15, "dl": 0.75},
    "Sandstorm / dust":                    {"lat": 10, "dl": 0.82},
}
TEMP_ADJ = {
    "Below 20°C":                          {"lat": 0,  "dl": 1.00},
    "20 – 30°C":                           {"lat": 0,  "dl": 1.00},
    "30 – 40°C":                           {"lat": 2,  "dl": 0.97},
    "Above 40°C":                          {"lat": 5,  "dl": 0.93},
}
USE_CASES = {
    "Video calls / online meetings":       {"icon":"📹","lat_max":50, "jitter_max":10,"dl_min":10,"ul_min":5,  "desc":"Needs low latency and stable jitter. Upload matters as much as download."},
    "Live gaming":                         {"icon":"🎮","lat_max":40, "jitter_max":8, "dl_min":15,"ul_min":5,  "desc":"Very sensitive to latency spikes. The most demanding use case."},
    "Streaming (Netflix, YouTube, etc.)":  {"icon":"📺","lat_max":100,"jitter_max":20,"dl_min":25,"ul_min":1,  "desc":"Mostly needs consistent download speed. Latency matters less here."},
    "General browsing / social media":     {"icon":"🌐","lat_max":100,"jitter_max":25,"dl_min":5, "ul_min":1,  "desc":"Low requirements. Works in most conditions."},
    "File uploads / cloud backup":         {"icon":"☁️","lat_max":150,"jitter_max":30,"dl_min":5, "ul_min":10, "desc":"Upload speed is the main factor here."},
    "Remote work / VPN / company systems": {"icon":"💼","lat_max":60, "jitter_max":12,"dl_min":10,"ul_min":5,  "desc":"Needs reliable latency and a stable connection."},
}

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=300)
def load_clean_data():
    dfs = []
    for path in [PERIOD1_CLEAN_FILE, PERIOD2_CLEAN_FILE]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(), False
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps", "upload_mbps", "weather_code", "temperature_c", "humidity_percent", "wind_speed_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[
        (df["ping_avg_rtt_ms"] > 0) & (df["ping_avg_rtt_ms"] < 500) &
        (df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100) &
        (df["download_mbps"] > 0) & (df["download_mbps"] < 1000) &
        (df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)
    ].copy()
    return df, True

@st.cache_data(ttl=300)
def load_base():
    df, ok = load_clean_data()
    if ok and len(df) > 0:
        return {
            "latency": float(df["ping_avg_rtt_ms"].median()),
            "jitter": float(df["ping_jitter_ms"].median()),
            "download": float(df["download_mbps"].median()),
            "upload": float(df["upload_mbps"].median()),
            "loaded": True,
        }
    return {"latency": 38.0, "jitter": 6.5, "download": 145.0, "upload": 22.0, "loaded": False}

@st.cache_resource
def load_saved_latency_model():
    try:
        model = joblib.load(AI_MODEL_FILE)
        features = joblib.load(AI_FEATURES_FILE)
        return model, features, True, "Loaded saved latency model"
    except Exception as e:
        return None, None, False, str(e)

# ============================================================
# FEATURE ENGINEERING + TRAINING FOR STRONGER ADVISOR
# ============================================================
def weather_to_code(weather):
    return {
        "Clear / sunny": 0,
        "Partly cloudy": 2,
        "Overcast / cloudy": 3,
        "Light rain or drizzle": 51,
        "Heavy rain or thunderstorm": 95,
        "Sandstorm / dust": 3,
    }.get(weather, 0)

def weather_to_humidity(weather):
    return {
        "Clear / sunny":                 15,
        "Partly cloudy":                 25,
        "Overcast / cloudy":             45,
        "Light rain or drizzle":         70,
        "Heavy rain or thunderstorm":    90,
        "Sandstorm / dust":              20,
    }.get(weather, 20)

def weather_to_wind(weather):
    return {
        "Clear / sunny":                 2,
        "Partly cloudy":                 3,
        "Overcast / cloudy":             4,
        "Light rain or drizzle":         5,
        "Heavy rain or thunderstorm":    10,
        "Sandstorm / dust":              12,
    }.get(weather, 3)

def time_to_hour(time_of_day):
    return {
        "Morning (6 AM – 12 PM)": 9,
        "Afternoon (12 PM – 6 PM)": 15,
        "Evening (6 PM – 10 PM)": 20,
        "Night (10 PM – 6 AM)": 23,
    }.get(time_of_day, 12)

def temp_to_value(temp):
    return {
        "Below 20°C": 18,
        "20 – 30°C": 26,
        "30 – 40°C": 35,
        "Above 40°C": 43,
    }.get(temp, 35)

def make_supervised(df, target_col):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    base_cols = ["ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps", "upload_mbps"]
    for col in base_cols:
        for lag in range(1, 5):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    if "weather_code" not in df.columns:
        df["weather_code"] = 0
    if "temperature_c" not in df.columns:
        df["temperature_c"] = 35
    if "humidity_percent" not in df.columns:
        df["humidity_percent"] = 20
    if "wind_speed_mps" not in df.columns:
        df["wind_speed_mps"] = 3

    df["target"] = df[target_col].shift(-1)
    features = [
        "hour", "day_of_week", "is_weekend", "weather_code",
        "temperature_c", "humidity_percent", "wind_speed_mps",
        "ping_avg_rtt_ms_lag_1", "ping_avg_rtt_ms_lag_2", "ping_avg_rtt_ms_lag_3", "ping_avg_rtt_ms_lag_4",
        "ping_jitter_ms_lag_1", "ping_jitter_ms_lag_2", "ping_jitter_ms_lag_3", "ping_jitter_ms_lag_4",
        "download_mbps_lag_1", "download_mbps_lag_2", "download_mbps_lag_3", "download_mbps_lag_4",
        "upload_mbps_lag_1", "upload_mbps_lag_2", "upload_mbps_lag_3", "upload_mbps_lag_4",
    ]
    available = [c for c in features if c in df.columns]
    out = df[available + ["target"]].dropna().reset_index(drop=True)
    return out, available

@st.cache_data(ttl=300)
def train_advisor_models():
    df, ok = load_clean_data()
    if not ok or len(df) < 50:
        return {}, {}, {}, False

    targets = {
        "latency": "ping_avg_rtt_ms",
        "jitter": "ping_jitter_ms",
        "download": "download_mbps",
        "upload": "upload_mbps",
    }
    models, features_map, metrics = {}, {}, {}

    for label, target_col in targets.items():
        sup, features = make_supervised(df, target_col)
        if len(sup) < 30:
            continue
        split = int(len(sup) * 0.8)
        X_train, X_test = sup[features].iloc[:split], sup[features].iloc[split:]
        y_train, y_test = sup["target"].iloc[:split], sup["target"].iloc[split:]

        model = RandomForestRegressor(n_estimators=160, max_depth=8, min_samples_leaf=3, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        residuals = y_test.values - pred

        models[label] = model
        features_map[label] = features
        metrics[label] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "residual_std": float(np.std(residuals)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    return models, features_map, metrics, True

# ============================================================
# PREDICTION LOGIC
# ============================================================
def apply_conditions(base, location, sky, time_of_day, weather, temp):
    adjs = [LOCATION_ADJ[location], SKY_ADJ[sky], TIME_ADJ[time_of_day], WEATHER_ADJ[weather], TEMP_ADJ[temp]]
    lat_delta = sum(a["lat"] for a in adjs)
    dl_factor = 1.0
    for a in adjs:
        dl_factor *= a["dl"]
    return {
        "latency": max(10.0, base["latency"] + lat_delta),
        "jitter": max(2.0, base["jitter"] + lat_delta * 0.25),
        "download": max(2.0, base["download"] * dl_factor),
        "upload": max(1.0, base["upload"] * dl_factor),
    }

def build_feature_row(perf, time_of_day, weather, temp, features, live_measurements=None):
    hour = time_to_hour(time_of_day)
    now = pd.Timestamp.now()
    row = {
        "hour": hour,
        "day_of_week": now.dayofweek,
        "is_weekend": int(now.dayofweek in [5, 6]),
        "weather_code": weather_to_code(weather),
        "temperature_c": temp_to_value(temp),
        "humidity_percent": weather_to_humidity(weather),
        "wind_speed_mps": weather_to_wind(weather),
    }

    if live_measurements and len(live_measurements) > 0:
        hist = live_measurements[-4:]
        hist = list(reversed(hist))
        for i in range(1, 5):
            src = hist[i - 1] if i <= len(hist) else hist[-1]
            row[f"ping_avg_rtt_ms_lag_{i}"] = src.get("latency", perf["latency"])
            row[f"ping_jitter_ms_lag_{i}"] = src.get("jitter", perf["jitter"])
            row[f"download_mbps_lag_{i}"] = src.get("download", perf["download"])
            row[f"upload_mbps_lag_{i}"] = src.get("upload", perf["upload"])
    else:
        for i in range(1, 5):
            row[f"ping_avg_rtt_ms_lag_{i}"] = perf["latency"]
            row[f"ping_jitter_ms_lag_{i}"] = perf["jitter"]
            row[f"download_mbps_lag_{i}"] = perf["download"]
            row[f"upload_mbps_lag_{i}"] = perf["upload"]

    X = pd.DataFrame([row])
    for f in features:
        if f not in X.columns:
            X[f] = 0
    return X[features]

def predict_all_kpis(perf, time_of_day, weather, temp, models, features_map, metrics, live_measurements=None):
    predictions = {}
    intervals = {}

    # --- Latency: use the validated saved combined-dataset Linear Regression model ---
    # This is the model evaluated in the thesis (MAE 4.337 ms, RMSE 6.466 ms on 812 test rows).
    # It takes priority over the runtime-trained advisor model for latency specifically.
    latency_predicted = False
    if saved_latency_loaded and saved_latency_model is not None and saved_latency_features:
        try:
            X_lat = build_feature_row(perf, time_of_day, weather, temp, saved_latency_features, live_measurements)
            pred_lat = float(saved_latency_model.predict(X_lat)[0])
            pred_lat = max(0.0, pred_lat)
            # Residual std from thesis combined-dataset evaluation
            SAVED_MODEL_RESIDUAL_STD = 3.30
            predictions["latency"] = pred_lat
            intervals["latency"] = 1.96 * SAVED_MODEL_RESIDUAL_STD
            latency_predicted = True
        except Exception:
            pass  # fall through to runtime model below

    for label in ["latency", "jitter", "download", "upload"]:
        if label == "latency" and latency_predicted:
            continue
        if label in models:
            X = build_feature_row(perf, time_of_day, weather, temp, features_map[label], live_measurements)
            pred = float(models[label].predict(X)[0])
            pred = max(0.0, pred) if label in ["latency", "jitter"] else max(1.0, pred)
            std = metrics.get(label, {}).get("residual_std", 0)
            predictions[label] = pred
            intervals[label] = 1.96 * std
        else:
            predictions[label] = perf[label]
            intervals[label] = 0

    return predictions, intervals

def confidence_level(intervals, live_count=0, weather=None, location=None):
    latency_uncertainty = intervals.get("latency", 999)

    condition_penalty = 0

    if weather in ("Heavy rain or thunderstorm", "Sandstorm / dust"):
        condition_penalty += 1

    if location in ("Desert / very remote area",):
        condition_penalty += 1

    if live_count >= 4 and latency_uncertainty <= 8 and condition_penalty == 0:
        return "High", "#3D8B65"

    if live_count >= 2 and latency_uncertainty <= 10 and condition_penalty <= 1:
        return "Medium", "#D99A20"

    if latency_uncertainty <= 8 and condition_penalty == 0:
        return "Medium", "#D99A20"

    return "Low", "#D95B4F"

def traffic_light(uc_name, lat, jitter, dl, ul):
    uc = USE_CASES[uc_name]
    score = 0
    if lat > uc["lat_max"]: score += 2
    if jitter > uc["jitter_max"]: score += 1
    if dl < uc["dl_min"]: score += 2
    if ul < uc["ul_min"]: score += 2
    return "green" if score == 0 else "amber" if score <= 2 else "red"

def top_feature_importance(model, features, limit=6):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        pairs = sorted(zip(features, vals), key=lambda x: x[1], reverse=True)[:limit]
        return pairs
    if hasattr(model, "coef_"):
        vals = np.abs(model.coef_)
        pairs = sorted(zip(features, vals), key=lambda x: x[1], reverse=True)[:limit]
        return pairs
    return []

# ============================================================
# UI HELPERS
# ============================================================
def metric_card(label, value, unit, color, interval=None):
    uncertainty = f"<div style='color:#6B7B83;font-size:0.74rem;margin-top:0.28rem;font-weight:700;'>± {interval:.1f} {unit}</div>" if interval is not None and interval > 0 else ""
    return f"""
    <div style="background:linear-gradient(145deg, rgba(255,255,255,0.95), {color}18);border:1px solid {color}55;border-top:7px solid {color};border-radius:24px;padding:1.0rem 1.08rem;text-align:center;box-shadow:0 16px 34px {color}20;position:relative;overflow:hidden;">
        <div style="position:relative;color:#6B7B83;font-size:0.66rem;font-weight:900;text-transform:uppercase;letter-spacing:1.4px;font-family:'IBM Plex Mono',monospace;">{label}</div>
        <div style="position:relative;color:#183B4A;font-size:1.85rem;font-weight:900;line-height:1.1;margin-top:0.25rem;font-family:'Space Grotesk','IBM Plex Mono',monospace;">
            {value}<span style="font-size:0.82rem;color:#6B7B83;font-weight:700;"> {unit}</span>
        </div>
        {uncertainty}
    </div>"""

def rating_card(color, uc_name, icon, desc, change_note=""):
    bg = {"green":"linear-gradient(135deg,#EEF8F4,#FFFFFF)", "amber":"linear-gradient(135deg,#FFF6D8,#FFFFFF)", "red":"linear-gradient(135deg,#FFF0EE,#FFFFFF)"}[color]
    border = {"green":"#3D8B65", "amber":"#D99A20", "red":"#D95B4F"}[color]
    label = {"green":"Suitable", "amber":"Limited", "red":"Not Recommended"}[color]
    change_html = f'<div style="color:#6B7B83;font-size:0.76rem;margin-top:0.25rem;">{change_note}</div>' if change_note else ""
    return (
        f'<div style="background:{bg};border:1px solid {border}44;border-left:7px solid {border};border-radius:22px;padding:1rem 1.15rem;display:flex;align-items:center;gap:1rem;margin-bottom:0.65rem;box-shadow:0 12px 26px {border}18;">'
        f'<div style="font-size:1.75rem;background:{border}18;border:1px solid {border}22;border-radius:16px;width:46px;height:46px;display:flex;align-items:center;justify-content:center;">{icon}</div>'
        f'<div style="flex:1;"><div style="font-weight:900;color:#183B4A;">{uc_name}</div><div style="color:#6B7B83;font-size:0.82rem;margin-top:0.15rem;line-height:1.45;">{desc}</div>{change_html}</div>'
        f'<div style="background:{border}18;border:1px solid {border}55;border-radius:999px;padding:0.35rem 0.85rem;color:{border};font-weight:900;font-size:0.80rem;white-space:nowrap;">{label}</div>'
        f'</div>'
    )

def trend_chart(values, label, color):
    series = pd.Series(values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(series) + 1)), y=series.values, mode="lines+markers", name="Measured", line=dict(color=color, width=2), marker=dict(size=6)))
    fig.update_layout(
        title=dict(text=label, font=dict(color="#183B4A", size=13)),
        height=210,
        plot_bgcolor="rgba(255,255,255,0.97)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#253545", family="Inter"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.08)", color="#475569", title="Test #"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", color="#475569"),
        margin=dict(l=10, r=10, t=38, b=10),
    )
    return fig


def render_pretty_table(df, max_height=None):
    if df is None or len(df) == 0:
        st.info("No data available.")
        return
    html = df.copy().to_html(index=False, classes="pretty-table", border=0)
    height_style = f"max-height:{max_height}px;" if max_height else ""
    st.markdown(
        f'<div class="pretty-table-wrap" style="{height_style}">{html}</div>',
        unsafe_allow_html=True,
    )


def render_pretty_table(df, max_height=None):
    if df is None or len(df) == 0:
        st.info("No data available.")
        return
    html = df.copy().to_html(index=False, classes="pretty-table", border=0)
    height_style = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""
    st.markdown(
        f'<div class="pretty-table-wrap" style="{height_style}">{html}</div>',
        unsafe_allow_html=True,
    )

def overlay_placeholder(text, top="-3.4rem"):
    st.markdown(
        f"""
        <div style="position:relative;margin-top:{top};margin-bottom:1.18rem;padding-left:0.95rem;
                    color:#98A2A9;font-size:0.95rem;pointer-events:none;z-index:2;
                    font-family:'Inter',sans-serif;line-height:1.25;min-height:1.2rem;
                    display:flex;align-items:center;white-space:nowrap;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SESSION STATE
# ============================================================
for key, val in {"step": 1, "advisory": None, "measurements": []}.items():
    if key not in st.session_state:
        st.session_state[key] = val

base = load_base()
models, features_map, model_metrics, advisor_models_loaded = train_advisor_models()
saved_latency_model, saved_latency_features, saved_latency_loaded, saved_msg = load_saved_latency_model()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="background:#FFFFFF;border-radius:16px;padding:1.55rem 2.8rem;margin-top:0.85rem;margin-bottom:1.2rem;border:1px solid #DDE7E2;border-top:4px solid #265868;box-shadow:0 2px 12px rgba(24,59,74,0.08);position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#183B4A,#265868,#3F7F78,#5EA38F);pointer-events:none;"></div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:800;line-height:1.1;letter-spacing:-0.8px;">
        <span style="color:#253545;">Starlink </span><span style="color:#265868;">Advisor</span>
    </div>
    <div style="color:#6B7B83;font-size:0.84rem;margin-top:0.45rem;font-family:'IBM Plex Mono',monospace;">
        AI suitability assessment · KPI forecasting · live measurements
    </div>
</div>
""", unsafe_allow_html=True)

# Progress
steps_labels = ["1. Conditions", "2. AI advisory", "3. Live personalised forecast"]
cols = st.columns(3)
for i, (col, name) in enumerate(zip(cols, steps_labels)):
    n = i + 1
    active = n == st.session_state.step
    done = n < st.session_state.step
    cls = "step-active" if active else "step-done" if done else "step-wait"
    label = name if not done else "✓ " + name.split(". ", 1)[1]
    col.markdown(f'<div class="step-card {cls}">{label}</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

# ============================================================
# STEP 1
# ============================================================
if st.session_state.step == 1:
    st.markdown("### Step 1, Your conditions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="q-label q-orange" style="margin-top:0;">1. Where are you located?</div>', unsafe_allow_html=True)
        location = st.selectbox("loc", list(LOCATION_ADJ.keys()), index=None, placeholder="City (Muscat or major urban area)", label_visibility="collapsed", key="q_loc")
        if not location:
            overlay_placeholder("City (Muscat or major urban area)")
        st.markdown('<div class="q-label q-sage">2. How clear is the sky view for the dish?</div>', unsafe_allow_html=True)
        sky = st.selectbox("sky", list(SKY_ADJ.keys()), index=None, placeholder="Completely clear, no obstructions", label_visibility="collapsed", key="q_sky")
        if not sky:
            overlay_placeholder("Completely clear, no obstructions")
        st.markdown('<div class="q-label q-yellow">3. What time of day will you mainly use it?</div>', unsafe_allow_html=True)
        time_of_day = st.selectbox("time", list(TIME_ADJ.keys()), index=None, placeholder="Morning (6 AM – 12 PM)", label_visibility="collapsed", key="q_time")
        if not time_of_day:
            overlay_placeholder("Morning (6 AM – 12 PM)")
    with c2:
        st.markdown('<div class="q-label q-sage" style="margin-top:0;">4. What is the weather usually like?</div>', unsafe_allow_html=True)
        weather = st.selectbox("weather", list(WEATHER_ADJ.keys()), index=None, placeholder="Clear / sunny", label_visibility="collapsed", key="q_weather")
        if not weather:
            overlay_placeholder("Clear / sunny")
        st.markdown('<div class="q-label q-yellow">5. What is the usual temperature?</div>', unsafe_allow_html=True)
        temp = st.selectbox("temp", list(TEMP_ADJ.keys()), index=None, placeholder="Below 20°C", label_visibility="collapsed", key="q_temp")
        if not temp:
            overlay_placeholder("Below 20°C")
        st.markdown('<div class="q-label q-orange">6. What will you use it for?</div>', unsafe_allow_html=True)
        uses = st.multiselect("uses", list(USE_CASES.keys()), default=[], placeholder="e.g. Video calls, Streaming", label_visibility="collapsed", key="q_uses")
        if not uses:
            overlay_placeholder("e.g. Video calls, Streaming", top="-3.40rem")

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    if st.button("Get AI Advisory Result →"):
        if not all([location, sky, time_of_day, weather, temp]):
            st.warning("Please choose all conditions first.")
        elif not uses:
            st.warning("Please select at least one use case.")
        else:
            perf_rule = apply_conditions(base, location, sky, time_of_day, weather, temp)
            predictions, intervals = predict_all_kpis(perf_rule, time_of_day, weather, temp, models, features_map, model_metrics)
            conf, conf_color = confidence_level(intervals, live_count=0, weather=weather, location=location)
            ratings = {uc: traffic_light(uc, predictions["latency"], predictions["jitter"], predictions["download"], predictions["upload"]) for uc in uses}
            st.session_state.advisory = {
                "location": location,
                "sky": sky,
                "time": time_of_day,
                "weather": weather,
                "temp": temp,
                "uses": uses,
                "rule_perf": perf_rule,
                "ai_perf": predictions,
                "intervals": intervals,
                "ratings": ratings,
                "confidence": conf,
                "confidence_color": conf_color,
                "models_loaded": advisor_models_loaded,
                "model_metrics": model_metrics,
            }
            st.session_state.step = 2
            st.rerun()


# ============================================================
# STEP 2
# ============================================================
elif st.session_state.step == 2:
    adv = st.session_state.advisory
    perf = adv["ai_perf"]
    intervals = adv["intervals"]

    st.markdown("### Step 2, AI advisory result")

    if adv["models_loaded"]:
        st.markdown(f"""
        <div class="advisor-card" style="border-left:4px solid #5EA38F;background:#EEF8F4;">
            <strong style="color:#3D8B65;">AI forecasting active</strong><br>
            Latency is forecast using the validated combined-dataset model (MAE 4.3 ms, RMSE 6.5 ms).
            Jitter, download, and upload are forecast using models trained on both experiment datasets.
            <br><br>
            <strong style="color:{adv['confidence_color']};">Forecast confidence: {adv['confidence']}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="advisor-card" style="border-left:4px solid #D95B4F;">
            <strong style="color:#D95B4F;">AI models not available</strong><br>
            The advisor is using rule-based fallback estimates only. Check that the cleaned datasets exist.
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Latency", f"{perf['latency']:.1f}", "ms", "#D95B24", intervals.get("latency")), unsafe_allow_html=True)
    c2.markdown(metric_card("Jitter", f"{perf['jitter']:.1f}", "ms", "#D99A20", intervals.get("jitter")), unsafe_allow_html=True)
    c3.markdown(metric_card("Download", f"{perf['download']:.1f}", "Mbps", "#8DB87A", intervals.get("download")), unsafe_allow_html=True)
    c4.markdown(metric_card("Upload", f"{perf['upload']:.1f}", "Mbps", "#3F7F78", intervals.get("upload")), unsafe_allow_html=True)

    st.markdown("### Use case ratings")
    for uc_name, color in adv["ratings"].items():
        uc = USE_CASES[uc_name]
        st.markdown(rating_card(color, uc_name, uc["icon"], uc["desc"]), unsafe_allow_html=True)

    st.markdown("### Explainability")
    imp = top_feature_importance(models.get("latency"), features_map.get("latency", []), limit=6) if adv["models_loaded"] else []
    if imp:
        imp_df = pd.DataFrame(imp, columns=["Feature", "Importance"])
        imp_df["Importance"] = imp_df["Importance"].round(4)
        render_pretty_table(imp_df, max_height=320)
        st.caption("Top model inputs for the latency prediction. This can be discussed as model interpretability in the thesis.")
    else:
        st.caption("Feature importance unavailable — ML model not loaded.")

    st.markdown("### Model validation summary")
    if adv["model_metrics"]:
        rows = []
        for k, m in adv["model_metrics"].items():
            rows.append({"Target": k, "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3), "Residual Std": round(m["residual_std"], 3), "Test Rows": m["n_test"]})
        render_pretty_table(pd.DataFrame(rows), max_height=320)

    b1, b2 = st.columns([1, 2])
    with b1:
        if st.button("← Change answers"):
            st.session_state.step = 1
            st.rerun()
    with b2:
        if st.button("Use live measurements →"):
            st.session_state.measurements = []
            st.session_state.step = 3
            st.rerun()

# ============================================================
# STEP 3
# ============================================================
elif st.session_state.step == 3:
    adv = st.session_state.advisory
    st.markdown("### Step 3, Live personalised forecast")
    st.markdown("<div class='small-note'>Enter real speed-test measurements while connected to Starlink. The advisor will build real lag features from your sequence instead of repeating one synthetic estimate.</div>", unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    in_lat = i1.number_input("Latency (ms)", min_value=0.0, max_value=500.0, value=0.0, step=0.5, key="in_lat")
    in_dl = i2.number_input("Download (Mbps)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, key="in_dl")
    in_ul = i3.number_input("Upload (Mbps)", min_value=0.0, max_value=500.0, value=0.0, step=1.0, key="in_ul")
    in_jit = i4.number_input("Jitter (ms)", min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="in_jit")

    ac, rc, _ = st.columns([1, 1, 3])
    with ac:
        if st.button("Add measurement"):
            if in_lat > 0 and in_dl > 0 and in_ul >= 0:
                st.session_state.measurements.append({"latency": in_lat, "download": in_dl, "upload": in_ul, "jitter": in_jit})
                st.rerun()
            else:
                st.warning("Enter at least latency and download before adding.")
    with rc:
        if st.button("Clear tests"):
            st.session_state.measurements = []
            st.rerun()

    measurements = st.session_state.measurements
    if measurements:
        mdf = pd.DataFrame(measurements)
        mdf.index = [f"Test {i+1}" for i in range(len(mdf))]
        render_pretty_table(mdf.rename(columns={"latency":"Latency (ms)", "download":"Download (Mbps)", "upload":"Upload (Mbps)", "jitter":"Jitter (ms)"}), max_height=320)

        perf_rule = adv["rule_perf"]
        live_pred, live_intervals = predict_all_kpis(perf_rule, adv["time"], adv["weather"], adv["temp"], models, features_map, model_metrics, live_measurements=measurements)
        conf, conf_color = confidence_level(live_intervals, live_count=len(measurements), weather=adv["weather"], location=adv["location"])

        st.markdown("### AI forecast from your real measurements")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Next Latency", f"{live_pred['latency']:.1f}", "ms", "#D95B24", live_intervals.get("latency")), unsafe_allow_html=True)
        c2.markdown(metric_card("Next Jitter", f"{live_pred['jitter']:.1f}", "ms", "#D99A20", live_intervals.get("jitter")), unsafe_allow_html=True)
        c3.markdown(metric_card("Next Download", f"{live_pred['download']:.1f}", "Mbps", "#8DB87A", live_intervals.get("download")), unsafe_allow_html=True)
        c4.markdown(metric_card("Next Upload", f"{live_pred['upload']:.1f}", "Mbps", "#3F7F78", live_intervals.get("upload")), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advisor-card" style="border-left:4px solid {conf_color};background:{conf_color}0d;">
            <strong style="color:{conf_color};">Live forecast confidence: {conf}</strong><br>
            <span style="color:#475569;">Confidence improves as you add more measurements — real lag features replace synthetic estimates.</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(trend_chart([m["latency"] for m in measurements], "Measured Latency", "#265868"), use_container_width=True)
        with col2:
            st.plotly_chart(trend_chart([m["download"] for m in measurements], "Measured Download", "#3F7F78"), use_container_width=True)

        st.markdown("### Updated use-case ratings")
        rank = {"red": 0, "amber": 1, "green": 2}
        for uc_name in adv["uses"]:
            uc = USE_CASES[uc_name]
            new_color = traffic_light(uc_name, live_pred["latency"], live_pred["jitter"], live_pred["download"], live_pred["upload"])
            old_color = adv["ratings"][uc_name]
            change = ""
            if new_color != old_color:
                change = f"Changed from {old_color} estimated rating to {new_color} measured forecast rating."
                if rank[new_color] > rank[old_color]:
                    change = f"Improved from {old_color} estimated rating to {new_color} measured forecast rating."
            st.markdown(rating_card(new_color, uc_name, uc["icon"], uc["desc"], change), unsafe_allow_html=True)
    else:
        st.info("Add at least one measurement. Add 3–4 measurements for a stronger live forecast.")

    st.markdown("---")
    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("← Back to advisory"):
            st.session_state.step = 2
            st.rerun()
    with b2:
        if st.button("Start over"):
            for k in ["step", "advisory", "measurements"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="color:#6B7B83;font-size:0.75rem;text-align:center;padding-bottom:1rem;">
    Based on Starlink data collected in Muscat, Oman · GUtech AI Thesis · Predictions are estimates, not guarantees
</div>
""", unsafe_allow_html=True)