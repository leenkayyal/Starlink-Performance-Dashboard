import warnings
warnings.filterwarnings("ignore")

import os
import hmac
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
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
    page_title="Starlink Performance Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

refresh_seconds = 60
st_autorefresh(interval=refresh_seconds * 1000, key="thesis_refresh")


# ============================================================
# EARLY LIGHT THEME FOR LOGIN SCREEN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --ink:#183B4A;
    --ink2:#265868;
    --cream:#FFF8EE;
    --paper:#FFFFFF;
    --orange:#FF7330;
    --yellow:#FFC845;
    --sage:#A7C983;
    --teal:#5EA38F;
    --muted:#6B7B83;
}

/* MAIN LIGHT BACKGROUND */
html, body, .stApp {
    background:
        radial-gradient(circle at 10% 8%, rgba(255,115,48,.13) 0, transparent 26%),
        radial-gradient(circle at 88% 4%, rgba(94,163,143,.20) 0, transparent 30%),
        radial-gradient(circle at 75% 92%, rgba(255,200,69,.13) 0, transparent 28%),
        linear-gradient(135deg,#FFF9EF 0%,#F6F9ED 45%,#EEF8F4 100%) fixed !important;
    color: var(--ink) !important;
    font-family: Inter, sans-serif !important;
}

/* TOP STREAMLIT BAR */
header[data-testid="stHeader"] {
    background: rgba(255,248,238,.88) !important;
    backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid rgba(24,59,74,.08) !important;
}

[data-testid="stDecoration"] {
    background: linear-gradient(90deg,var(--ink),var(--orange),var(--yellow),var(--sage),var(--teal)) !important;
    height:5px !important;
}

/* SAME SIZE / SHAPE AS THE DARK LOGIN SCREEN */
.block-container {
    width: 100% !important;
    max-width: 100% !important;
    padding-top: 17vh !important;
    padding-left: 5.5rem !important;
    padding-right: 5.5rem !important;
    padding-bottom: 3rem !important;
}

.main .block-container,
section.main > div,
div[data-testid="stAppViewBlockContainer"] {
    width: 100% !important;
    max-width: 100% !important;
}

/* NO OUTER CARD / NO WHITE TITLE BOX */
div[data-testid="stVerticalBlock"] > div:has(.login-hero) {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 0 !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    overflow: visible !important;
}

/* TITLE AREA LIKE THE SCREENSHOT */
.login-hero {
    width: 100% !important;
    max-width: none !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 0 1.35rem 0 !important;
    box-shadow: none !important;
}

.login-hero h1 {
    color: var(--ink) !important;
    margin: 0 0 1.05rem 0 !important;
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    line-height: 1.1 !important;
}

.login-hero p {
    color: var(--ink) !important;
    margin: 0 !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

/* LABELS */
.stTextInput label p,
[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
}

/* USERNAME INPUT */
.stTextInput input {
    background: rgba(255,255,255,.96) !important;
    background-color: rgba(255,255,255,.96) !important;
    border: 1.5px solid rgba(24,59,74,.18) !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    height: 44px !important;
    box-shadow: none !important;
    padding: .65rem .9rem !important;
}

.stTextInput input:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(255,115,48,.18) !important;
}

/* PASSWORD WRAPPER AND EYE ICON AREA */
.stTextInput div[data-baseweb="input"] {
    background: rgba(255,255,255,.96) !important;
    background-color: rgba(255,255,255,.96) !important;
    border: 1.5px solid rgba(24,59,74,.18) !important;
    border-radius: 8px !important;
    height: 44px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.stTextInput div[data-baseweb="input"] input {
    border: 0 !important;
    box-shadow: none !important;
    height: 42px !important;
    background: transparent !important;
}

.stTextInput div[data-baseweb="input"]:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(255,115,48,.18) !important;
}

.stTextInput div[data-baseweb="base-input"] {
    background: transparent !important;
}

.stTextInput div[data-baseweb="input"] > div,
.stTextInput div[data-baseweb="input"] button,
.stTextInput div[data-baseweb="input"] [role="button"] {
    background: transparent !important;
    color: var(--ink) !important;
    border: 0 !important;
    box-shadow: none !important;
}

.stTextInput div[data-baseweb="input"] svg {
    fill: var(--ink) !important;
    color: var(--ink) !important;
}

/* KEEP INPUTS WHITE AFTER TYPING / AUTOFILL */
.stTextInput input,
.stTextInput input:focus,
.stTextInput input:active,
.stTextInput input:hover,
.stTextInput input:-webkit-autofill,
.stTextInput input:-webkit-autofill:hover,
.stTextInput input:-webkit-autofill:focus,
.stTextInput input:-webkit-autofill:active {
    background-color: rgba(255,255,255,.96) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    caret-color: var(--ink) !important;
    box-shadow: 0 0 0 1000px rgba(255,255,255,.96) inset !important;
}

/* FULL-WIDTH INPUTS */
div[data-testid="stTextInput"],
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] input {
    width: 100% !important;
    max-width: none !important;
}

/* SMALL BUTTON LIKE THE DARK LOGIN SCREEN */
.stButton {
    width: auto !important;
    max-width: none !important;
}

.stButton > button {
    width: auto !important;
    max-width: none !important;
    background: rgba(255,255,255,.75) !important;
    color: var(--ink) !important;
    border: 1px solid rgba(24,59,74,.25) !important;
    border-radius: 8px !important;
    padding: .45rem .85rem !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.stButton > button:hover {
    background: rgba(255,255,255,.95) !important;
    border-color: var(--orange) !important;
    color: var(--orange) !important;
    transform: none !important;
    box-shadow: none !important;
}

.stAlert {
    border-radius: 12px !important;
}

/* MOBILE */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 12vh !important;
    }

    .login-hero h1 {
        font-size: 1.85rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECURITY / LOGIN
# ============================================================
try:
    from security_config import USERS
except Exception as e:
    st.error(
        "Security configuration file was not found or could not be loaded. "
        "Make sure security_config.py is beside this dashboard file."
    )
    st.exception(e)
    st.stop()

def _verify_password(saved_user, entered_password):
    """
    Supports the team's simple USERS structure and safer hashed variants.

    Accepted examples inside security_config.py:
    USERS = {
        "admin": {"password": "1234", "role": "admin"},
        "advisor": {"password_hash": "sha256$...", "role": "advisor"}
    }
    """
    if not isinstance(saved_user, dict):
        return False

    # 1) Backward-compatible plain password check
    if "password" in saved_user:
        return hmac.compare_digest(str(saved_user.get("password", "")), str(entered_password))

    # 2) Optional SHA-256 hash check: password_hash can be either "sha256$hash" or just the hash
    if "password_hash" in saved_user:
        saved_hash = str(saved_user.get("password_hash", ""))
        entered_hash = hashlib.sha256(str(entered_password).encode("utf-8")).hexdigest()
        if saved_hash.startswith("sha256$"):
            saved_hash = saved_hash.split("sha256$", 1)[1]
        return hmac.compare_digest(saved_hash, entered_hash)

    return False

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None

    if not st.session_state.authenticated:
        st.markdown("""
        <div class="login-hero">
            <h1>Starlink Performance Monitor</h1>
            <p>Please log in to continue.</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_record = USERS.get(username)
            if user_record and _verify_password(user_record, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = user_record.get("role", "user")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.stop()

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

check_login()

loading_placeholder = st.empty()
loading_placeholder.markdown("""
<div class="loading-shell">
    <div class="loading-row">
        <div class="loading-spinner"></div>
        <div>
            <div class="loading-title">Loading dashboard data...</div>
            <div class="loading-subtitle">Preparing charts, KPIs, and forecast results. Please wait a moment.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FILE PATHS
# ============================================================
# -- Clean measurement files (one per experiment period) --
PERIOD1_CLEAN_FILE    = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
PERIOD2_CLEAN_FILE    = "Cleaned/experiment_B/Starlink_2_cleaned.csv"
PERIOD1_CLEAN_FALLBACK = "starlink_clean_FIXED.csv"
PERIOD2_CLEAN_FALLBACK = "Starlink_2_cleaned.csv"

# -- Feature-engineered forecast files --
THESIS_FORECAST_FILE       = "Cleaned/combined/starlink_forecast_combined.csv"
FORECAST_PERIOD1_FILE      = "Cleaned/experiment_A/starlink_forecast_v2.csv"
FORECAST_PERIOD2_FILE      = "Cleaned/experiment_B/starlink_2_forecast.csv"
FORECAST_PERIOD1_FALLBACK  = "starlink_forecast_v2.csv"
FORECAST_PERIOD2_FALLBACK  = "starlink_2_forecast.csv"
FORECAST_COMBINED_FALLBACK = "starlink_forecast_combined.csv"

# -- Comparison ISP files --
OMANTEL_FILE = "Cleaned/experiment_A/omantel_clean.csv"
OMANTEL_FALLBACK = "omantel_clean.csv"
AWASR_FILE   = "Cleaned/experiment_B/Awasr_cleaned.csv"
AWASR_FALLBACK = "Awasr_cleaned.csv"

# -- Live / retrain files --
RAW_FILE              = "Raw/experiment_A/starlink_data.csv"
STARLINK_RETRAIN_FILE = "Cleaned/state/starlink_retrain_queue.csv"
STARLINK_RETRAIN_FALLBACK = "starlink_retrain_queue.csv"
LIVE_MONITOR_FILE     = "Cleaned/state/live_monitor_only.csv"

# -- Validated research windows (from actual data inspection) --
PERIOD1_START = pd.Timestamp("2026-03-07 01:00:00")
PERIOD1_END   = pd.Timestamp("2026-03-28 11:00:00")
PERIOD2_START = pd.Timestamp("2026-04-20 11:45:00")
PERIOD2_END   = pd.Timestamp("2026-05-12 07:45:00")

# Backwards-compatible single window for legacy callers
THESIS_START = PERIOD1_START
THESIS_END   = PERIOD2_END

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
    background: linear-gradient(90deg, var(--ink), var(--orange), var(--yellow), var(--sage), var(--teal)) !important;
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
    background: linear-gradient(90deg, var(--ink), var(--orange), var(--yellow), var(--sage), var(--teal));
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

/* stronger divider lines inside sidebar */
[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: rgba(255,255,255,0.28) !important;
    margin: 2.1rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border: none !important;
    height: 1px !important;
    background: rgba(255,255,255,0.28) !important;
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

.stDownloadButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #173C4A 0%, #265868 58%, #5EA38F 125%) !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 16px !important;
    font-weight: 900 !important;
    padding: 0.75rem 1.4rem !important;
    box-shadow: 0 14px 28px rgba(24,59,74,0.22) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 18px 38px rgba(24,59,74,0.30) !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #FFFFFF !important;
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
.pretty-table tbody tr:nth-child(even) td {
    background: rgba(94,163,143,0.05);
}
.pretty-table tbody tr:hover td {
    background: rgba(255,200,69,0.10);
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
    box-shadow: 0 0 0 5px rgba(255,115,48,0.18) !important;
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
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,200,69,0.20) !important; color: var(--orange) !important; }
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

/* final coherence fixes */
.badge { margin-right: 0.65rem !important; }
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs button[aria-selected="true"] *,
.stTabs [data-baseweb="tab"][aria-selected="true"] * {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stToggle {
    margin-bottom: 0.8rem !important;
}
.js-plotly-plot .plotly .modebar { background: rgba(255,255,255,0.72) !important; border-radius: 999px !important; }

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
    background: linear-gradient(135deg, rgba(255,115,48,0.14), rgba(255,200,69,0.16)) !important;
    border: 1px solid rgba(255,115,48,0.35) !important;
    color: var(--orange) !important;
    border-radius: 999px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: var(--orange) !important; font-weight: 800 !important; }

/* alerts / status */
.stAlert {
    border-radius: 18px !important;
    border: 1px solid rgba(24,59,74,0.10) !important;
    box-shadow: 0 10px 25px rgba(24,59,74,0.08) !important;
}
.stSpinner > div { border-color: var(--orange) transparent transparent transparent !important; border-width: 3px !important; }
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] svg { color: var(--orange) !important; fill: var(--orange) !important; stroke: var(--orange) !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--orange), var(--yellow), var(--teal)) !important; }

/* custom loading card */
.loading-shell {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(24,59,74,0.10);
    border-radius: 22px;
    box-shadow: 0 18px 45px rgba(24,59,74,0.10);
    padding: 1rem 1.15rem;
    margin: 0 0 1.15rem 0;
}
.loading-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.loading-spinner {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    border: 3px solid rgba(24,59,74,0.14);
    border-top-color: var(--orange);
    border-right-color: var(--teal);
    animation: thesisSpin 0.9s linear infinite;
    flex: 0 0 auto;
}
.loading-title {
    color: var(--ink);
    font-weight: 800;
    font-size: 0.98rem;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.loading-subtitle {
    color: var(--muted);
    font-size: 0.83rem;
    line-height: 1.45;
}
@keyframes thesisSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOT THEME
# ============================================================
PLOT_BG   = "rgba(255,255,255,0.96)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_CLR  = "rgba(24,59,74,0.13)"
AXIS_CLR  = "#1F4D5C"
FONT_CLR  = "#183B4A"

P1_COLOR  = "#FF7330"   # orange – Period 1
P2_COLOR  = "#5EA38F"   # teal – Period 2
P1_FILL   = "rgba(255,115,48,0.12)"
P2_FILL   = "rgba(94,163,143,0.12)"

BASE_LAYOUT = dict(
    plot_bgcolor=PLOT_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_CLR, family="IBM Plex Mono"),
    margin=dict(l=10, r=10, t=42, b=10),
    xaxis=dict(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False),
    yaxis=dict(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False),
    legend=dict(bgcolor="rgba(255,255,255,0.92)", font=dict(color="#183B4A"), bordercolor="rgba(24,59,74,0.12)", borderwidth=1),
)

# ============================================================
# HELPERS, FORMATTING
# ============================================================
def fmt_temp(v):
    try:    return f"{float(v):.1f} °C" if pd.notna(v) else "—"
    except: return "—"
def fmt_humidity(v):
    try:    return f"{float(v):.0f}%" if pd.notna(v) else "—"
    except: return "—"
def fmt_wind(v):
    try:    return f"{float(v):.2f} m/s" if pd.notna(v) else "—"
    except: return "—"
def fmt_code(v):
    desc = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Cloudy",
            45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Drizzle",
            55:"Heavy drizzle",61:"Slight rain",63:"Rain",65:"Heavy rain",
            71:"Slight snow",73:"Snow",75:"Heavy snow",80:"Rain showers",
            81:"Heavy showers",82:"Violent showers",95:"Thunderstorm",
            96:"Thunderstorm+hail",99:"Heavy thunderstorm+hail"}
    try:    return desc.get(int(float(v)), f"Code {int(float(v))}")
    except: return "—"
def safe_float(row, col):
    try:    return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
    except: return 0.0

# ============================================================
# HELPERS, BUSINESS LOGIC
# ============================================================
def compute_health(latency, jitter, download, upload):
    score = 100
    if latency  > 80:  score -= 30
    elif latency > 50: score -= 15
    if jitter   > 15:  score -= 25
    elif jitter  > 8:  score -= 10
    if download  < 20: score -= 20
    elif download < 50: score -= 10
    if upload    < 10: score -= 15
    elif upload  < 20: score -= 8
    score = max(0, min(score, 100))
    label = ("Excellent" if score >= 85 else "Good" if score >= 70
             else "Fair" if score >= 50 else "Poor")
    return score, label

def detect_outage(latency, download, upload):
    if latency  > 100: return "High Latency Outage"
    if download < 5:   return "Severe Download Drop"
    if upload   < 2:   return "Upload Failure Risk"
    return "No Outage Detected"

def generate_alerts(latency_f, jitter_now, download_f, upload_f):
    alerts = []
    if latency_f  > 50: alerts.append(("warning", "Latency forecast to increase in the next 15 minutes."))
    if jitter_now > 10: alerts.append(("warning", "Current jitter is high. Real-time applications may be affected."))
    if download_f < 50: alerts.append(("error",   "Download speed forecast is low."))
    if upload_f   < 10: alerts.append(("error",   "Upload speed forecast is low."))
    if not alerts:      alerts.append(("success",  "No immediate network issue forecast."))
    return alerts

def usage_recommendation(latency, jitter, download, upload):
    recs = []
    recs.append("Suitable for video calls and online meetings."
                if latency < 40 and jitter < 8
                else "Use caution for video calls and real-time meetings.")
    recs.append("Suitable for streaming and large downloads."
                if download > 100 else
                "General browsing and normal streaming should work well."
                if download > 40 else
                "Heavy streaming or large downloads may be affected.")
    recs.append("Suitable for file uploads and cloud syncing."
                if upload > 20 else
                "Normal uploads should work, but large uploads may be slower."
                if upload > 10 else
                "Large uploads may be unreliable at this time.")
    return recs

def ai_insight(latency, jitter, download, upload):
    if latency < 35 and jitter < 8 and download > 100 and upload > 20:
        return ("Starlink is in a strong operating state and appears suitable "
                "for most academic and administrative activities.")
    if latency > 50 or jitter > 10:
        return ("The network may support general use, but latency-sensitive "
                "activities such as live meetings may be affected.")
    if download < 50 or upload < 10:
        return ("Bandwidth-intensive activities may experience reduced quality. "
                "File transfers and streaming should be scheduled with caution.")
    return "Current conditions appear acceptable for standard university use."

def health_badge_html(score):
    if score >= 85:
        return ('<span class="badge" style="background:#EEF8F4;color:#3D8B65;'
                'border:1px solid #5EA38F66;">Excellent</span>')
    elif score >= 70:
        return ('<span class="badge" style="background:#FFF6D8;color:#B87A13;'
                'border:1px solid #FFC84577;">Good</span>')
    elif score >= 50:
        return ('<span class="badge" style="background:#FFF0E8;color:#D95B24;'
                'border:1px solid #FF733077;">Fair</span>')
    return ('<span class="badge" style="background:#FFF0EE;color:#D95B4F;'
            'border:1px solid #D95B4F55;">Poor</span>')

# ============================================================
# HELPERS, KPI CARD
# ============================================================
def kpi_card(label, value, unit, color, icon):
    return f"""
<div style="
    background:linear-gradient(145deg, rgba(255,255,255,0.94), {color}18);
    border:1px solid {color}55;
    border-top:7px solid {color};
    border-radius:24px;
    padding:1.18rem 1.25rem;
    box-shadow:0 18px 38px {color}22;
    position:relative;
    overflow:hidden;
">
    <div style="position:relative;color:{color};font-size:1.45rem;margin-bottom:0.28rem;">{icon}</div>
    <div style="position:relative;color:#6B7B83;font-size:0.64rem;font-weight:900;text-transform:uppercase;
                letter-spacing:1.7px;font-family:'IBM Plex Mono',monospace;">{label}</div>
    <div style="position:relative;color:#183B4A;font-size:2.0rem;font-weight:900;line-height:1.06;
                margin-top:0.2rem;font-family:'Space Grotesk','IBM Plex Mono',monospace;">
        {value}<span style="font-size:0.82rem;color:#6B7B83;font-weight:700;margin-left:4px;">{unit}</span>
    </div>
</div>"""


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

# ============================================================
# HELPERS, CHARTS
# ============================================================
def _apply_base(fig, title, height=360):
    """
    Shared Plotly styling.
    The larger margins and darker axis-title colours prevent labels from becoming
    pale/cut off in Streamlit columns, especially in ISP Comparison and Forecasting.
    """
    layout = dict(BASE_LAYOUT)
    layout["title"] = dict(
        text=title,
        font=dict(color="#183B4A", size=15, family="Space Grotesk"),
        x=0.02,
        xanchor="left"
    )
    layout["height"] = height
    layout["margin"] = dict(l=86, r=34, t=58, b=82)
    layout["plot_bgcolor"] = "rgba(255,255,255,0.98)"
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["font"] = dict(color="#183B4A", family="IBM Plex Mono")
    fig.update_layout(**layout)
    fig.update_xaxes(
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        title_standoff=18,
        automargin=True,
        zeroline=False,
        showline=False,
        gridcolor="rgba(24,59,74,0.13)",
        color="#1F4D5C"
    )
    fig.update_yaxes(
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        title_standoff=22,
        automargin=True,
        zeroline=False,
        showline=False,
        gridcolor="rgba(24,59,74,0.13)",
        color="#1F4D5C"
    )
    return fig

def make_dual_period_chart(df1, df2, col, title, y_label, height=370):
    """
    Two-panel subplot: Period 1 on the left, Period 2 on the right.
    Both panels share the same y-axis scale so values are directly comparable.
    A wider middle divider labels the 23-day gap without cutting the text.
    """
    combined = pd.concat([df1[col].dropna(), df2[col].dropna()])
    y_min = combined.min() * 0.92
    y_max = combined.max() * 1.08

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.45, 0.10, 0.45],
        shared_yaxes=True,
        horizontal_spacing=0.025,
        subplot_titles=["", "", ""]
    )

    # --- Period 1 (left panel) ---
    fig.add_trace(go.Scatter(
        x=df1["timestamp"],
        y=df1[col],
        mode="lines",
        name="Period 1  Mar 7–28",
        line=dict(color=P1_COLOR, width=1.6),
        fill="tozeroy",
        fillcolor="rgba(255,115,48,0.13)",
        showlegend=True
    ), row=1, col=1)

    # --- Period 2 (right panel) ---
    fig.add_trace(go.Scatter(
        x=df2["timestamp"],
        y=df2[col],
        mode="lines",
        name="Period 2  Apr 20 – May 12",
        line=dict(color=P2_COLOR, width=1.6),
        fill="tozeroy",
        fillcolor="rgba(94,163,143,0.13)",
        showlegend=True
    ), row=1, col=3)

    # --- Middle divider label ---
    # cliponaxis=False prevents Plotly from cutting the text at subplot borders.
    fig.add_trace(go.Scatter(
        x=[0.5],
        y=[(y_min + y_max) / 2],
        mode="text",
        text=["23-day<br>gap"],
        textposition="middle center",
        textfont=dict(
            color="#1F4D5C",
            size=12,
            family="IBM Plex Mono"
        ),
        cliponaxis=False,
        showlegend=False
    ), row=1, col=2)

    # Shared y-axis range applied to both outer panels
    fig.update_yaxes(
        range=[y_min, y_max],
        gridcolor=GRID_CLR,
        color=AXIS_CLR,
        showline=False,
        title_text=y_label,
        title_font=dict(color=AXIS_CLR, size=13, family="IBM Plex Mono"),
        tickfont=dict(color=AXIS_CLR, size=11, family="IBM Plex Mono"),
        title_standoff=12,
        automargin=True,
        zeroline=False,
        row=1,
        col=1
    )

    fig.update_yaxes(
        range=[y_min, y_max],
        gridcolor=GRID_CLR,
        color=AXIS_CLR,
        showline=False,
        showticklabels=False,
        tickfont=dict(color=AXIS_CLR, size=11, family="IBM Plex Mono"),
        automargin=True,
        zeroline=False,
        row=1,
        col=3
    )

    fig.update_yaxes(visible=False, row=1, col=2)

    fig.update_xaxes(
        gridcolor=GRID_CLR,
        color=AXIS_CLR,
        showline=False,
        tickfont=dict(color=AXIS_CLR, size=11, family="IBM Plex Mono"),
        automargin=True,
        nticks=4,
        zeroline=False,
        row=1,
        col=1
    )

    fig.update_xaxes(
        gridcolor=GRID_CLR,
        color=AXIS_CLR,
        showline=False,
        tickfont=dict(color=AXIS_CLR, size=11, family="IBM Plex Mono"),
        automargin=True,
        nticks=4,
        zeroline=False,
        row=1,
        col=3
    )

    fig.update_xaxes(
        visible=False,
        range=[0, 1],
        fixedrange=True,
        row=1,
        col=2
    )

    # Period label annotations above each panel
    fig.add_annotation(
        text="<b>Period 1</b>  Mar 7 – 28",
        xref="paper",
        yref="paper",
        x=0.225,
        y=1.025,
        showarrow=False,
        font=dict(color=P1_COLOR, size=12, family="IBM Plex Mono"),
        xanchor="center"
    )

    fig.add_annotation(
        text="<b>Period 2</b>  Apr 20 – May 12",
        xref="paper",
        yref="paper",
        x=0.775,
        y=1.025,
        showarrow=False,
        font=dict(color=P2_COLOR, size=12, family="IBM Plex Mono"),
        xanchor="center"
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="#183B4A", size=13, family="IBM Plex Mono")
        ),
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="IBM Plex Mono"),
        margin=dict(l=62, r=38, t=58, b=62),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            font=dict(color=FONT_CLR, size=11),
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bordercolor="rgba(24,59,74,0.10)",
            borderwidth=1
        )
    )

    return fig

def make_single_period_chart(df, col, title, y_label, color, height=360):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[col], mode="lines", name=col,
        line=dict(color=color, width=1.8),
        fill="tozeroy",
        fillcolor=("rgba(255,115,48,0.10)" if color == P1_COLOR else "rgba(94,163,143,0.10)" if color == P2_COLOR else "rgba(167,201,131,0.10)")
    ))
    _apply_base(fig, title, height)
    fig.update_layout(yaxis_title=y_label)
    return fig

def make_actual_vs_predicted_chart(y_test, y_pred, points, title):
    actual    = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)
    if points < len(actual):
        actual    = actual.iloc[-points:]
        predicted = predicted.iloc[-points:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(color="#FF7330", width=2.2)
    ))
    fig.add_trace(go.Scatter(
        y=predicted,
        mode="lines",
        name="Predicted",
        line=dict(color="#5EA38F", width=2.2, dash="dot")
    ))

    _apply_base(fig, title, 440)
    fig.update_layout(
        xaxis_title="Test Time Steps",
        yaxis_title="Latency (ms)",
        margin=dict(l=92, r=38, t=62, b=92),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(color="#183B4A", size=11, family="IBM Plex Mono"),
            bordercolor="rgba(24,59,74,0.14)",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=20
    )
    fig.update_yaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=24
    )
    return fig

def make_health_gauge(score):
    bar_color = ("#3D8B65" if score >= 85 else "#D99A20" if score >= 70
                 else "#FF7330" if score >= 50 else "#D95B4F")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 26, "color": bar_color,
                                            "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#E2E8F0",
                     "tickfont": {"color": "#475569", "size": 9}},
            "bar":  {"color": bar_color, "thickness": 0.38},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.10)"},
                {"range": [50, 70], "color": "rgba(249,115,22,0.10)"},
                {"range": [70, 85], "color": "rgba(234,179,8,0.10)"},
                {"range": [85,100], "color": "rgba(34,197,94,0.10)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 3},
                          "thickness": 0.8, "value": score}
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=28, b=8),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#253545", "family": "IBM Plex Mono", "size": 12})
    return fig

def make_isp_comparison_chart(df_sl, df_isp, isp_name, col, title, y_label):
    """Comparison of Starlink vs a terrestrial ISP with clearly readable labels."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sl["timestamp"],
        y=df_sl[col],
        mode="lines",
        name="Starlink",
        line=dict(color="#7EC8FF", width=1.7),
        fill="tozeroy",
        fillcolor="rgba(255,115,48,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=df_isp["timestamp"],
        y=df_isp[col],
        mode="lines",
        name=isp_name,
        line=dict(color="#5EA38F", width=1.7),
        fill="tozeroy",
        fillcolor="rgba(94,163,143,0.12)"
    ))

    _apply_base(fig, title, 390)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=y_label,
        margin=dict(l=92, r=36, t=60, b=88),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(color="#183B4A", size=11, family="IBM Plex Mono"),
            bordercolor="rgba(24,59,74,0.14)",
            borderwidth=1,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    fig.update_xaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=20,
        automargin=True
    )
    fig.update_yaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=24,
        automargin=True
    )
    return fig

def make_scatter_weather(df, weather_col, metric_col, title):
    """Scatter of a weather variable vs a network metric, coloured by period."""
    fig = go.Figure()
    # We may get a combined df with a 'period' column, or just one df
    if "period" in df.columns:
        for period, color, label in [(1, P1_COLOR, "Period 1"), (2, P2_COLOR, "Period 2")]:
            sub = df[df["period"] == period]
            fig.add_trace(go.Scatter(
                x=sub[weather_col], y=sub[metric_col], mode="markers", name=label,
                marker=dict(color=color, size=4, opacity=0.6)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[weather_col], y=df[metric_col], mode="markers",
            marker=dict(color=P1_COLOR, size=4, opacity=0.6)
        ))
    _apply_base(fig, title, 320)
    fig.update_layout(xaxis_title=weather_col, yaxis_title=metric_col)
    return fig

def make_box_comparison(df1, df2, col, title, y_label):
    """Box plot comparing distributions for both periods."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df1[col].dropna(), name="P1 (Mar 7–28)",
        marker_color=P1_COLOR, line_color=P1_COLOR,
        fillcolor=P1_FILL, boxmean=True
    ))
    fig.add_trace(go.Box(
        y=df2[col].dropna(), name="P2 (Apr 20–May 12)",
        marker_color=P2_COLOR, line_color=P2_COLOR,
        fillcolor=P2_FILL, boxmean=True
    ))
    _apply_base(fig, title, 360)
    fig.update_layout(
        yaxis_title=y_label,
        showlegend=False,
        margin=dict(l=58, r=24, t=54, b=72),
        xaxis=dict(tickangle=0, tickfont=dict(color=AXIS_CLR, size=11, family="IBM Plex Mono"))
    )
    return fig

def make_residual_chart(y_test, y_pred, title):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=residuals,
        mode="lines",
        name="Residual",
        line=dict(color="#5EA38F", width=1.6)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)
    _apply_base(fig, title, 340)
    fig.update_layout(
        yaxis_title="Residual (ms)",
        xaxis_title="Test Steps",
        margin=dict(l=92, r=38, t=60, b=88)
    )
    fig.update_xaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=20
    )
    fig.update_yaxes(
        title_font=dict(color="#183B4A", size=13, family="IBM Plex Mono"),
        tickfont=dict(color="#1F4D5C", size=11, family="IBM Plex Mono"),
        title_standoff=24
    )
    return fig

# ============================================================
# LIVE DATA ROUTING
# ============================================================
def classify_and_route_new_rows():
    """Route new rows from the raw file into retrain/monitor queues."""
    if not os.path.exists(RAW_FILE):
        return None
    raw = pd.read_csv(RAW_FILE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], dayfirst=False, errors="coerce")
    new_rows = raw[raw["timestamp"] > THESIS_END].copy()
    if len(new_rows) == 0:
        return None
    starlink_new = new_rows[new_rows["network_type"].str.strip().str.lower() == "starlink"]
    other_new    = new_rows[new_rows["network_type"].str.strip().str.lower() != "starlink"]
    if len(starlink_new) > 0:
        _append_dedup(STARLINK_RETRAIN_FILE, starlink_new)
    if len(other_new) > 0:
        _append_dedup(LIVE_MONITOR_FILE, other_new)
    return new_rows.sort_values("timestamp").iloc[-1]

def load_latest_live_row():
    """
    Return (row, source) for the live KPI overlay.
    Priority: raw file post-thesis rows -> retrain queue -> (None, None).
    """
    # 1. Try raw file
    if os.path.exists(RAW_FILE):
        try:
            raw = pd.read_csv(RAW_FILE)
            raw.columns = [c.lower() for c in raw.columns]
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], dayfirst=False, errors="coerce")
            post = raw[raw["timestamp"] > THESIS_END].dropna(subset=["timestamp"])
            if len(post) > 0:
                return post.sort_values("timestamp").iloc[-1], "raw"
        except Exception:
            pass
    # 2. Fall back to retrain queue
    rpath = STARLINK_RETRAIN_FILE if os.path.exists(STARLINK_RETRAIN_FILE) else (
            STARLINK_RETRAIN_FALLBACK if os.path.exists(STARLINK_RETRAIN_FALLBACK) else None)
    if rpath:
        try:
            df = pd.read_csv(rpath)
            df.columns = [c.lower() for c in df.columns]
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(df) > 0:
                # Prefer most recent row with valid speedtest data (download > 0)
                valid = df[df["download_mbps"].notna() & (df["download_mbps"] > 0)]
                best = valid.iloc[-1] if len(valid) > 0 else df.iloc[-1]
                return best, "retrain_queue"
        except Exception:
            pass
    return None, None

def _append_dedup(path, new_df):
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], dayfirst=False, errors="coerce")
        combined = pd.concat([existing, new_df]).drop_duplicates(subset=["timestamp"])
    else:
        combined = new_df
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    combined.to_csv(path, index=False)

def retrain_queue_status():
    path = STARLINK_RETRAIN_FILE if os.path.exists(STARLINK_RETRAIN_FILE) else (
           STARLINK_RETRAIN_FALLBACK if os.path.exists(STARLINK_RETRAIN_FALLBACK) else None)
    if path is None:
        return 0, None
    df = pd.read_csv(path)
    return len(df), pd.to_datetime(df["timestamp"], format="mixed", errors="coerce").max()

# ============================================================
# FILE PATH RESOLVER
# ============================================================
def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(ttl=120)
def load_period1():
    path = first_existing(PERIOD1_CLEAN_FILE, PERIOD1_CLEAN_FALLBACK)
    if path is None:
        st.error("Cannot find Period 1 clean file (starlink_clean_FIXED.csv). "
                 "Place it at Cleaned/experiment_A/ or beside this script.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]  # normalise temperature_C -> temperature_c
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    df = df[(df["timestamp"] >= PERIOD1_START) & (df["timestamp"] <= PERIOD1_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=120)
def load_period2():
    path = first_existing(PERIOD2_CLEAN_FILE, PERIOD2_CLEAN_FALLBACK)
    if path is None:
        st.error("Cannot find Period 2 clean file (Starlink_2_cleaned.csv). "
                 "Place it at Cleaned/experiment_B/ or beside this script.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    df = df[(df["timestamp"] >= PERIOD2_START) & (df["timestamp"] <= PERIOD2_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=120)
def load_omantel():
    path = first_existing(OMANTEL_FILE, OMANTEL_FALLBACK)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=120)
def load_awasr():
    path = first_existing(AWASR_FILE, AWASR_FALLBACK)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=120)
def load_forecast_combined():
    """
    Loads the combined feature-engineered forecast dataset.
    Falls back to auto-merging the two period forecast files if combined is missing.
    """
    path = first_existing(THESIS_FORECAST_FILE, FORECAST_COMBINED_FALLBACK)
    if path:
        df = pd.read_csv(path)
        return _clean_forecast_df(df), "combined_existing"

    p1 = first_existing(FORECAST_PERIOD1_FILE, FORECAST_PERIOD1_FALLBACK)
    p2 = first_existing(FORECAST_PERIOD2_FILE, FORECAST_PERIOD2_FALLBACK)
    if p1 is None or p2 is None:
        missing = []
        if p1 is None: missing.append("starlink_forecast_v2.csv (Period 1)")
        if p2 is None: missing.append("starlink_2_forecast.csv (Period 2)")
        st.error("Missing forecast dataset(s): " + ", ".join(missing))
        st.stop()
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    common = [c for c in df1.columns if c in set(df2.columns)]
    combined = pd.concat([df1[common], df2[common]], ignore_index=True)
    return _clean_forecast_df(combined), "combined_auto_merged"

def _clean_forecast_df(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    if "target" not in df.columns:
        st.error("Forecast dataset must contain a 'target' column.")
        st.stop()
    if len(df) < 10:
        st.error("Forecast dataset too small after cleaning.")
        st.stop()
    return df

# ============================================================
# ML
# ============================================================
def prepare_xy(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.select_dtypes(include=["number"])
    return df.drop(columns=["target"]), df["target"]

def train_model(df, model_name):
    X, y = prepare_xy(df)
    split = int(len(df) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte  = y.iloc[:split], y.iloc[split:]

    if model_name == "Naive Baseline":
        lags   = [c for c in Xte.columns if c.endswith("_lag_1")]
        y_pred = Xte[lags[0]].values
        model  = None
    elif model_name == "Linear Regression":
        model = LinearRegression(); model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    elif model_name == "XGBoost":
        if not XGB_AVAILABLE: raise ValueError("XGBoost not installed.")
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {"model": model, "X_test": Xte, "y_test": yte, "y_pred": y_pred,
            "mae":  float(mean_absolute_error(yte, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(yte, y_pred)))}

def predict_next(df, model_name):
    X, y = prepare_xy(df)
    if len(X) == 0: return 0.0
    latest = X.iloc[[-1]]
    if model_name == "Naive Baseline":
        lags = [c for c in latest.columns if c.endswith("_lag_1")]
        return float(latest[lags[0]].iloc[0])
    if model_name == "Linear Regression":  m = LinearRegression()
    elif model_name == "Random Forest":    m = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":          m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    else:                                  return 0.0
    m.fit(X, y)
    return float(m.predict(latest)[0])

def build_download_upload_features(clean_df, target_col):
    df = clean_df.copy()
    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False]
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in [target_col, "ping_avg_rtt_ms", "ping_jitter_ms",
              "weather_code", "temperature_c", "humidity_percent", "wind_speed_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df[target_col] > 0].copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    for lag in range(1, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    if "ping_avg_rtt_ms" in df.columns: df["latency_lag_1"]  = df["ping_avg_rtt_ms"].shift(1)
    if "ping_jitter_ms"  in df.columns: df["jitter_lag_1"]   = df["ping_jitter_ms"].shift(1)
    df["target"] = df[target_col].shift(-1)
    df = df.dropna().reset_index(drop=True)
    features = ["hour", "day_of_week", "is_weekend",
                f"{target_col}_lag_1", f"{target_col}_lag_2",
                f"{target_col}_lag_3", f"{target_col}_lag_4",
                "latency_lag_1", "jitter_lag_1", "weather_code",
                "temperature_c", "humidity_percent", "wind_speed_mps"]
    features = [c for c in features if c in df.columns]
    return df[features + ["target"]].copy()

def make_model_comparison_table(df):
    names = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE: names.append("XGBoost")
    rows = []
    for m in names:
        r = train_model(df, m)
        rows.append({"Model": m, "MAE (ms)": round(r["mae"], 3),
                     "RMSE (ms)": round(r["rmse"], 3)})
    cmp_df = pd.DataFrame(rows)
    best_mae = cmp_df["MAE (ms)"].idxmin()
    cmp_df["Best"] = ""
    cmp_df.loc[best_mae, "Best"] = "Selected"
    return cmp_df

# ============================================================
# LOAD DATA
# ============================================================
p1_df = load_period1()   # Period 1 clean
p2_df = load_period2()   # Period 2 clean
omantel_df = load_omantel()
awasr_df   = load_awasr()
forecast_df, forecast_merge_status = load_forecast_combined()

# Combined clean (for any legacy callers that need one df)
combined_clean = pd.concat([p1_df, p2_df], ignore_index=True).sort_values("timestamp")

# Build per-period support datasets for download/upload forecasting (use combined)
dl_feat = build_download_upload_features(combined_clean, "download_mbps")
ul_feat = build_download_upload_features(combined_clean, "upload_mbps")

# Route any new raw rows into queues (best-effort)
# Live logger now writes directly to starlink_retrain_queue.csv.
# Do not auto-route old raw rows here, because it can re-add old/bad timestamps.
# Load most recent available row for live overlay
latest_live_row, latest_live_source = load_latest_live_row()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Controls")

    current_user = st.session_state.get("username", "user")
    current_role = st.session_state.get("role", "user")
    st.caption(f"Logged in as **{current_user}** · Role: **{current_role}**")
    if st.button("Logout", use_container_width=True):
        logout()

    st.markdown("---")
    model_options = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE: model_options.append("XGBoost")
    selected_model = st.selectbox("Forecast model", model_options, index=1)
    chart_points   = st.slider("Prediction chart points", 50, 500, 200, 25)

    st.markdown("---")
    show_live = st.toggle(
        "Show live KPI overlay", value=False,
        help="Displays the latest logged measurement on KPI cards. "
             "Does not affect the model, charts, or research data."
    )

    st.markdown("---")
    st.caption("Study windows: P1 Mar 7–28, 2026 · P2 Apr 20–May 12, 2026")
    q_count, q_latest = retrain_queue_status()
    st.markdown("---")
    if q_count > 0:
        st.markdown(f"**Retrain queue:** {q_count} rows")
        st.markdown(f"Latest: `{q_latest}`")
        st.caption("New Starlink-only rows eligible for future retraining.")
    else:
        st.caption("No new Starlink data queued for retraining.")

# ============================================================
# TRAIN MODELS
# ============================================================
latency_result    = train_model(forecast_df, selected_model)
latency_forecast  = predict_next(forecast_df, selected_model)
download_forecast = predict_next(dl_feat, "Linear Regression")
upload_forecast   = predict_next(ul_feat, "Linear Regression")

loading_placeholder.empty()

# Fallback if feature builder returns 0
_last_p2 = p2_df.iloc[-1]
if download_forecast <= 0:
    download_forecast = safe_float(_last_p2, "download_mbps")
if upload_forecast <= 0:
    upload_forecast   = safe_float(_last_p2, "upload_mbps")

# ============================================================
# DECIDE DISPLAY ROW
# ============================================================
if show_live:
    if latest_live_row is not None:
        display_row = latest_live_row
        _net = str(latest_live_row.get("network_type", "Starlink")
                   if hasattr(latest_live_row, "get")
                   else getattr(latest_live_row, "network_type", "Starlink"))
        _ts  = str(latest_live_row.get("timestamp", "")
                   if hasattr(latest_live_row, "get")
                   else getattr(latest_live_row, "timestamp", ""))[:16]
        if latest_live_source == "raw":
            kpi_note = f"Live data (raw file), {_net} · {_ts}"
            kpi_note_color = "#3D8B65"
        else:
            kpi_note = f"Live overlay, retrain queue · most recent row · {q_latest}"
            kpi_note_color = "#D99A20"
    else:
        display_row    = p2_df.iloc[-1]
        kpi_note       = "Live overlay: no post-thesis data found, showing last Period 2 row"
        kpi_note_color = "#D95B4F"
else:
    display_row    = p2_df.iloc[-1]
    kpi_note       = ("Showing last observation from Period 2 "
                      "(most recent validated clean dataset, May 12, 2026)")
    kpi_note_color = "#6B7B83"

current_latency  = safe_float(display_row, "ping_avg_rtt_ms")
current_jitter   = safe_float(display_row, "ping_jitter_ms")
current_download = safe_float(display_row, "download_mbps")
current_upload   = safe_float(display_row, "upload_mbps")

health_score, health_label = compute_health(
    latency_forecast, current_jitter, download_forecast, upload_forecast)
alerts          = generate_alerts(latency_forecast, current_jitter, download_forecast, upload_forecast)
recommendations = usage_recommendation(latency_forecast, current_jitter, download_forecast, upload_forecast)
system_insight  = ai_insight(latency_forecast, current_jitter, download_forecast, upload_forecast)
outage_status   = detect_outage(latency_forecast, download_forecast, upload_forecast)

# The forecast label should describe the next interval after the latest row being shown.
# In historical mode this is the final validated study row.
# When the live KPI overlay is enabled, this is the latest logged row if available.
try:
    latest_time = pd.to_datetime(display_row["timestamp"], errors="coerce")
except Exception:
    latest_time = combined_clean["timestamp"].max()

if pd.isna(latest_time):
    latest_time = combined_clean["timestamp"].max()

forecast_time = latest_time + pd.Timedelta(minutes=15)

# Weather from display row
def _get(row, col):
    try:    return row[col] if col in row.index else np.nan
    except: return np.nan

weather_code = _get(display_row, "weather_code")
temp_val     = _get(display_row, "temperature_c")
hum_val      = _get(display_row, "humidity_percent")
wind_val     = _get(display_row, "wind_speed_mps")

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f'<div style="margin-bottom:-1.4rem;'
    f'margin-top:-2rem;color:#6B7B83;font-weight:600;'
    f'font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;">'
    f'Last refresh: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}'
    f'</div>',
    unsafe_allow_html=True
)

_now_str = pd.Timestamp.now().strftime("%Y-%m-%d  %H:%M:%S")
live_header_dot = ""
if show_live:
    live_header_dot = (
        '<div style="position:absolute;top:50%;left:14px;transform:translateY(-50%);z-index:4;display:flex;align-items:center;justify-content:center;">'
        '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        'background:#4ADE80;border:2px solid #F5FFF8;'
        'box-shadow:0 0 0 4px rgba(74,222,128,0.16),0 0 10px rgba(74,222,128,0.58),0 0 18px rgba(74,222,128,0.34);"></span>'
        '</div>'
    )
st.markdown(
    f'<div style="background:#FFFFFF;'
    f'border-radius:16px;padding:1.5rem 2.4rem 1.5rem 2.8rem;margin-bottom:0.8rem;'
    f'border:1px solid #DDE7E2;'
    f'border-top:4px solid #265868;'
    f'box-shadow:0 2px 12px rgba(24,59,74,0.08);'
    f'position:relative;overflow:hidden;">'

    f'{live_header_dot}'

    f'<div style="position:absolute;top:0;left:0;right:0;height:4px;'
    f'background:linear-gradient(90deg,#183B4A,#265868,#3F7F78,#5EA38F);'
    f'pointer-events:none;"></div>'

    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:2rem;'
    f'font-weight:800;line-height:1.1;letter-spacing:-0.8px;">'
    f'<span style="color:#253545;">Starlink </span>'
    f'<span style="color:#265868;">Performance Dashboard</span></div>'
    f'<div style="margin-top:0.45rem;color:#6B7B83;font-size:0.9rem;line-height:1.45;'
    f'font-family:\'Inter\',sans-serif;font-weight:500;">'
    f'Historical performance analysis with short-term Starlink forecasting.'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
)

# KPI source note
st.markdown(f"""
<div style="
    background:{kpi_note_color}0d;border:1px solid {kpi_note_color}33;
    border-left:4px solid {kpi_note_color};
    border-radius:7px;padding:0.45rem 1rem;margin-bottom:0.7rem;
    font-size:0.78rem;color:{kpi_note_color};
    font-family:'IBM Plex Mono',monospace;
">{kpi_note}</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
hcolor = ("#3D8B65" if health_score >= 85 else "#D99A20" if health_score >= 70
          else "#FF7330" if health_score >= 50 else "#D95B4F")

with c1: st.markdown(kpi_card("Latency",      f"{current_latency:.1f}",  "ms",   "#FF7330", "📡"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("Jitter",       f"{current_jitter:.1f}",   "ms",   "#FFC845", "〰️"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("Download",     f"{current_download:.1f}", "Mbps", "#A7C983", "⬇"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("Upload",       f"{current_upload:.1f}",   "Mbps", "#5EA38F", "⬆"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("Health Score", f"{health_score}",         "/100", hcolor,    "💚"), unsafe_allow_html=True)

st.markdown(
    f'<div style="margin:0.5rem 0 0.9rem;">'
    f'{health_badge_html(health_score)}'
    f'&nbsp;&nbsp;<span style="color:#6B7B83;font-size:0.88rem;'
    f'font-family:\'IBM Plex Mono\',monospace;">'
    f'Network status: <b style="color:#183B4A;">{outage_status}</b></span></div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Historical Trends",
    "ISP Comparison",
    "Forecasting",
    "Alerts & Recs",
    "Model Evaluation"
])

# ============================================================
# TAB 1, OVERVIEW
# ============================================================
with tab1:
    st.markdown('<div class="card-title">Study Snapshot</div>', unsafe_allow_html=True)

    left, right = st.columns([2.2, 1])
    with left:
        # Dual-period latency chart, the signature chart of the thesis
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "ping_avg_rtt_ms",
                                   "Latency Across the Collected Study Periods",
                                   "Latency (ms)", height=430),
            use_container_width=True
        )

    with right:
        st.markdown('<div class="card-title">Forecast: Next 15 Minutes</div>', unsafe_allow_html=True)
        st.metric("Latency",  f"{latency_forecast:.2f} ms")
        st.metric("Download", f"{download_forecast:.2f} Mbps")
        st.metric("Upload",   f"{upload_forecast:.2f} Mbps")
        st.markdown(
            f'<div style="color:#6B7B83;font-size:0.75rem;margin-top:0.5rem;'
            f'font-family:\'IBM Plex Mono\',monospace;">'
            f'Next interval: {forecast_time.strftime("%Y-%m-%d %H:%M")}<br>'
            f'Based on latest used row: {latest_time.strftime("%Y-%m-%d %H:%M")}<br>'
            f'Model: {selected_model}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1.1, 1])
    with col_a:
        st.plotly_chart(make_health_gauge(health_score), use_container_width=True)
    with col_b:
        st.markdown('<div class="card-title">Recommended Use</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f"- {rec}")
    with col_c:
        st.markdown('<div class="card-title">Weather at the Selected Row</div>', unsafe_allow_html=True)
        st.write(f"Condition: **{fmt_code(weather_code)}**")
        st.write(f"Temperature: **{fmt_temp(temp_val)}**")
        st.write(f"Humidity: **{fmt_humidity(hum_val)}**")
        st.write(f"Wind: **{fmt_wind(wind_val)}**")

    st.markdown('<div class="card-title">Plain-Language Interpretation</div>', unsafe_allow_html=True)
    st.write(system_insight)

# ============================================================
# TAB 2, HISTORICAL TRENDS
# ============================================================
with tab2:
    st.markdown('<div class="card-title">Collected Performance Trends</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:#334155;font-size:0.78rem;margin-bottom:0.8rem;'
        f'font-family:\'IBM Plex Mono\',monospace;">'
        f'<span style="color:{P1_COLOR};">Blue = Period 1 (Mar 7–28, 2026)</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:{P2_COLOR};">Green = Period 2 (Apr 20 – May 12, 2026)</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'The shaded region between them marks the 23-day gap where no data was collected.</div>',
        unsafe_allow_html=True
    )

    # Jitter and download side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "ping_jitter_ms",
                                   "Jitter Across the Collected Periods", "Jitter (ms)"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "download_mbps",
                                   "Download Speed Across the Collected Periods", "Download (Mbps)"),
            use_container_width=True
        )

    # Upload and packet loss
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "upload_mbps",
                                   "Upload Speed Across the Collected Periods", "Upload (Mbps)"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "packet_loss_percent",
                                   "Packet Loss Across the Collected Periods", "Packet Loss (%)"),
            use_container_width=True
        )

    # Distribution comparison (box plots)
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">How the Two Study Periods Compare</div>',
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(
            make_box_comparison(p1_df, p2_df, "ping_avg_rtt_ms",
                                "Latency Distribution", "Latency (ms)"),
            use_container_width=True
        )
    with b2:
        st.plotly_chart(
            make_box_comparison(p1_df, p2_df, "download_mbps",
                                "Download Distribution", "Download (Mbps)"),
            use_container_width=True
        )

    # Per-period summary stats
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Summary of the Collected Data</div>',
                unsafe_allow_html=True)
    cols_of_interest = ["ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps",
                        "upload_mbps", "packet_loss_percent"]
    stats_rows = []
    for col in cols_of_interest:
        for period_label, df in [("Period 1 (Mar 7–28)", p1_df),
                                  ("Period 2 (Apr 20 – May 12)", p2_df)]:
            if col in df.columns:
                s = df[col].dropna()
                stats_rows.append({
                    "Metric": col, "Period": period_label,
                    "Mean": round(s.mean(), 2), "Median": round(s.median(), 2),
                    "Std Dev": round(s.std(), 2), "Min": round(s.min(), 2),
                    "Max": round(s.max(), 2), "Rows": len(s)
                })
    render_pretty_table(pd.DataFrame(stats_rows), max_height=520)

# ============================================================
# TAB 3, ISP COMPARISON
# ============================================================
with tab3:
    st.markdown('<div class="card-title">Provider Comparison During the Same Study Windows</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#6B7B83;font-size:0.78rem;margin-bottom:0.8rem;'
        'font-family:\'IBM Plex Mono\',monospace;">'
        'Period 1 (Mar 7–28): Starlink vs OmanTel &nbsp;|&nbsp; '
        'Period 2 (Apr 20 – May 12): Starlink vs Awasr. '
        'Each comparison uses data from the same time window, so conditions are comparable.</div>',
        unsafe_allow_html=True
    )

    # --- PERIOD 1: Starlink vs OmanTel ---
    st.markdown(
        f'<span class="period-tag" style="background:rgba(232,83,28,0.1);color:{P1_COLOR};'
        f'border:1px solid {P1_COLOR}44;">Period 1</span>'
        f'<span style="color:#183B4A;margin-left:0.6rem;font-weight:700;">'
        f'Starlink vs OmanTel, Mar 7 to Mar 28, 2026</span>',
        unsafe_allow_html=True
    )

    if omantel_df is not None:
        om_p1 = omantel_df[(omantel_df["timestamp"] >= PERIOD1_START) &
                            (omantel_df["timestamp"] <= PERIOD1_END)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "ping_avg_rtt_ms",
                                          "Latency: Starlink vs OmanTel", "Latency (ms)"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "download_mbps",
                                          "Download: Starlink vs OmanTel", "Download (Mbps)"),
                use_container_width=True
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "upload_mbps",
                                          "Upload: Starlink vs OmanTel", "Upload (Mbps)"),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "ping_jitter_ms",
                                          "Jitter: Starlink vs OmanTel", "Jitter (ms)"),
                use_container_width=True
            )

        # Summary table P1
        isp_rows = []
        for metric in ["ping_avg_rtt_ms", "download_mbps", "upload_mbps", "ping_jitter_ms"]:
            if metric in p1_df.columns and metric in om_p1.columns:
                isp_rows.append({
                    "Metric": metric,
                    "Starlink Mean": round(p1_df[metric].mean(), 2),
                    "OmanTel Mean":  round(om_p1[metric].mean(), 2),
                    "Difference":    round(p1_df[metric].mean() - om_p1[metric].mean(), 2)
                })
        if isp_rows:
            render_pretty_table(pd.DataFrame(isp_rows), max_height=360)
    else:
        st.info("OmanTel data not found. Place omantel_clean.csv at Cleaned/experiment_A/ or beside this script.")

    st.markdown("---")

    # --- PERIOD 2: Starlink vs Awasr ---
    st.markdown(
        f'<span class="period-tag" style="background:rgba(141,184,122,0.15);color:{P2_COLOR};'
        f'border:1px solid {P2_COLOR}44;">Period 2</span>'
        f'<span style="color:#183B4A;margin-left:0.6rem;font-weight:700;">'
        f'Starlink vs Awasr, Apr 20 to May 12, 2026</span>',
        unsafe_allow_html=True
    )

    if awasr_df is not None:
        aw_p2 = awasr_df[(awasr_df["timestamp"] >= PERIOD2_START) &
                          (awasr_df["timestamp"] <= PERIOD2_END)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "ping_avg_rtt_ms",
                                          "Latency: Starlink vs Awasr", "Latency (ms)"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "download_mbps",
                                          "Download: Starlink vs Awasr", "Download (Mbps)"),
                use_container_width=True
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "upload_mbps",
                                          "Upload: Starlink vs Awasr", "Upload (Mbps)"),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "ping_jitter_ms",
                                          "Jitter: Starlink vs Awasr", "Jitter (ms)"),
                use_container_width=True
            )

        isp_rows2 = []
        for metric in ["ping_avg_rtt_ms", "download_mbps", "upload_mbps", "ping_jitter_ms"]:
            if metric in p2_df.columns and metric in aw_p2.columns:
                isp_rows2.append({
                    "Metric": metric,
                    "Starlink Mean": round(p2_df[metric].mean(), 2),
                    "Awasr Mean":    round(aw_p2[metric].mean(), 2),
                    "Difference":    round(p2_df[metric].mean() - aw_p2[metric].mean(), 2)
                })
        if isp_rows2:
            render_pretty_table(pd.DataFrame(isp_rows2), max_height=360)
    else:
        st.info("Awasr data not found. Place Awasr_cleaned.csv at Cleaned/experiment_B/ or beside this script.")

# ============================================================
# TAB 4, FORECASTING
# ============================================================
with tab4:
    st.markdown('<div class="card-title">Next 15-Minute Forecast</div>',
                unsafe_allow_html=True)

    st.plotly_chart(
        make_actual_vs_predicted_chart(
            latency_result["y_test"], latency_result["y_pred"],
            chart_points, f"Actual vs Predicted Latency ({selected_model})"),
        use_container_width=True
    )

    f1, f2, f3 = st.columns(3)
    f1.metric("Expected Latency",  f"{latency_forecast:.2f} ms")
    f2.metric("Expected Download", f"{download_forecast:.2f} Mbps")
    f3.metric("Expected Upload",   f"{upload_forecast:.2f} Mbps")

    # Residuals
    st.plotly_chart(
        make_residual_chart(
            latency_result["y_test"], latency_result["y_pred"],
            f"Prediction Residuals ({selected_model})"
        ),
        use_container_width=True
    )

    st.markdown(
        '<div style="color:#6B7B83;font-size:0.8rem;margin-top:0.4rem;'
        'font-family:\'IBM Plex Mono\',monospace;">'
        'These values are not another historical summary. They are short-term estimates for the '
        'next 15-minute interval after the latest measurement used by the dashboard. Latency is '
        'the main forecasting target; download and upload are supporting operational estimates. '
        'The model was trained on the combined validated study dataset (Period 1 + Period 2).</div>',
        unsafe_allow_html=True
    )

    q_count, q_latest = retrain_queue_status()
    if q_count > 0:
        st.markdown("---")
        st.markdown('<div class="card-title">New Starlink Data Available for Retraining</div>',
                    unsafe_allow_html=True)
        st.write(f"**{q_count}** new Starlink rows collected after the thesis window.")
        st.write(f"Latest timestamp: **{q_latest}**")
        st.write("These rows are in `starlink_retrain_queue.csv` and can be merged "
                 "to retrain the model when ready.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 5, ALERTS & RECOMMENDATIONS
# ============================================================
with tab5:
    st.markdown('<div class="card-title">Usage Guidance Based on the Forecast</div>', unsafe_allow_html=True)

    for level, message in alerts:
        if level == "success":  st.success(message)
        elif level == "warning": st.warning(message)
        else:                    st.error(message)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card-title">What This Means Operationally</div>', unsafe_allow_html=True)
        st.write(
            f"Based on the next 15-minute forecast, the expected health status is **{health_label}** with a score of "
            f"**{health_score}/100**. This combines predicted latency, forecast bandwidth, "
            f"and current jitter."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card-title">What Users Can Do Next</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.write(f"- {rec}")

# ============================================================
# TAB 6, MODEL EVALUATION
# ============================================================
with tab6:
    st.markdown('<div class="card-title">Model Check on the Validated Study Dataset</div>',
                unsafe_allow_html=True)

    model_table = make_model_comparison_table(forecast_df)
    render_pretty_table(model_table, max_height=360)

    m1, m2 = st.columns(2)
    m1.metric(f"{selected_model}, MAE",  f"{latency_result['mae']:.3f} ms")
    m2.metric(f"{selected_model}, RMSE", f"{latency_result['rmse']:.3f} ms")

    st.markdown('<div class="card-title">Model Notes</div>', unsafe_allow_html=True)
    st.write(
        "Models were trained and evaluated on the combined validated Starlink dataset "
        "(Period 1: Mar 7–28 2026 + Period 2: Apr 20 – May 12 2026, "
        f"total {len(forecast_df):,} feature-engineered rows). "
        "The naive baseline copies the most recent lag as the prediction, serving as a "
        "minimum benchmark. Linear Regression was selected as the primary model for its "
        "balance of accuracy, stability, and interpretability. More complex models showed "
        "no consistent improvement, indicating that Starlink latency follows primarily "
        "linear patterns driven by recent history and time-of-day. Sudden spike events "
        "caused by satellite handoffs remain difficult to predict from available features "
        "across all models."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance (Random Forest only, since it's available without XGB)
    with st.expander("Random Forest Feature Importance"):
        try:
            X_fi, y_fi = prepare_xy(forecast_df)
            rf_fi = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_fi.fit(X_fi, y_fi)
            fi_df = pd.DataFrame({
                "Feature":   X_fi.columns,
                "Importance": rf_fi.feature_importances_
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color="#265868"
            ))
            _apply_base(fig_fi, "Feature Importance (Random Forest)", 380)
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")

    st.markdown("---")
    st.markdown("**Export Results**")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "Download Model Comparison CSV",
            model_table.to_csv(index=False).encode("utf-8"),
            "model_comparison_thesis.csv", "text/csv"
        )
    with col_e2:
        pred_df = pd.DataFrame({
            "actual":    np.array(latency_result["y_test"]),
            "predicted": np.array(latency_result["y_pred"])
        })
        st.download_button(
            "Download Latency Predictions CSV",
            pred_df.to_csv(index=False).encode("utf-8"),
            "latency_predictions_thesis.csv", "text/csv"
        )