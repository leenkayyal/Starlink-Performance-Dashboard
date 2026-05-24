import warnings
warnings.filterwarnings("ignore")

import os
import hashlib
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Starlink Thesis Dashboard — GUtech 2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

refresh_seconds = 60
st_autorefresh(interval=refresh_seconds * 1000, key="thesis_refresh")

# ======================
# FILE PATHS
# ======================
THESIS_CLEAN_FILE     = "Cleaned/starlink_clean_FIXED.csv"
THESIS_FORECAST_FILE  = "Cleaned/starlink_forecast_v2.csv"
RAW_FILE              = "Raw/starlink_data.csv"
STARLINK_RETRAIN_FILE = "Cleaned/starlink_retrain_queue.csv"
LIVE_MONITOR_FILE     = "Cleaned/live_monitor_only.csv"
HASH_STORE_FILE       = "Cleaned/data_hashes.json"
AUDIT_LOG_FILE        = "Cleaned/audit_log.txt"

THESIS_START = pd.Timestamp("2026-03-07 01:00:00")
THESIS_END   = pd.Timestamp("2026-03-28 11:00:00")

SESSION_TIMEOUT = 15 * 60  # 15 minutes in seconds
MAX_ATTEMPTS    = 3
LOCKOUT_SECONDS = 300       # 5 minutes


# ======================
# SECURITY — AUDIT LOG
# ======================
def write_audit_log(event, username="unknown"):
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] USER={username} EVENT={event}\n")


# ======================
# SECURITY — LOGIN / ACCESS CONTROL
# ======================
from security_config import USERS

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.last_active = None
        st.session_state.failed_attempts = {}

    # Session timeout check
    if st.session_state.authenticated:
        elapsed = time.time() - st.session_state.last_active
        if elapsed > SESSION_TIMEOUT:
            write_audit_log("SESSION_TIMEOUT", st.session_state.username)
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.username = None
            st.session_state.last_active = None
            st.warning("Your session expired after 15 minutes of inactivity. Please log in again.")
        else:
            st.session_state.last_active = time.time()
            return

    st.markdown("## Starlink Thesis Dashboard")
    st.markdown("Please log in to continue.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        now = time.time()
        attempts_info = st.session_state.failed_attempts.get(username, {"count": 0, "locked_until": 0})

        if now < attempts_info["locked_until"]:
            remaining = int(attempts_info["locked_until"] - now)
            write_audit_log(f"LOGIN_BLOCKED lockout {remaining}s remaining", username)
            st.error(f"Account locked. Try again in {remaining} seconds.")
            st.stop()

        if username in USERS and USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.role = USERS[username]["role"]
            st.session_state.username = username
            st.session_state.last_active = time.time()
            st.session_state.failed_attempts.pop(username, None)
            write_audit_log("LOGIN_SUCCESS", username)
            st.rerun()
        else:
            attempts_info["count"] = attempts_info.get("count", 0) + 1
            if attempts_info["count"] >= MAX_ATTEMPTS:
                attempts_info["locked_until"] = now + LOCKOUT_SECONDS
                write_audit_log(f"LOGIN_FAILED account locked after {MAX_ATTEMPTS} attempts", username)
                st.error(f"Too many failed attempts. Account locked for {LOCKOUT_SECONDS // 60} minutes.")
            else:
                remaining_attempts = MAX_ATTEMPTS - attempts_info["count"]
                write_audit_log(f"LOGIN_FAILED attempt {attempts_info['count']}", username)
                st.error(f"Invalid username or password. {remaining_attempts} attempt(s) remaining.")
            st.session_state.failed_attempts[username] = attempts_info

    st.stop()

check_login()

current_user  = st.session_state.username
current_role  = st.session_state.role
is_admin      = str(current_role).strip().lower() == "admin"
is_supervisor = str(current_role).strip().lower() == "supervisor"
is_guest      = str(current_role).strip().lower() == "guest"


# ======================
# SECURITY — SHA-256 DATA INTEGRITY
# ======================
def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_hashes(files):
    hashes = {}
    for fp in files:
        if os.path.exists(fp):
            hashes[fp] = compute_sha256(fp)
    os.makedirs(os.path.dirname(HASH_STORE_FILE), exist_ok=True)
    with open(HASH_STORE_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def verify_hashes(files):
    if not os.path.exists(HASH_STORE_FILE):
        save_hashes(files)
        return {fp: "OK" for fp in files if os.path.exists(fp)}
    with open(HASH_STORE_FILE, "r") as f:
        stored = json.load(f)
    results = {}
    for fp in files:
        if not os.path.exists(fp):
            results[fp] = "MISSING"
        elif fp not in stored:
            results[fp] = "NEW"
        elif compute_sha256(fp) != stored[fp]:
            results[fp] = "TAMPERED"
        else:
            results[fp] = "OK"
    return results


# ======================
# SECURITY — API RESPONSE VALIDATION
# ======================
def validate_weather_row(row):
    issues = []
    try:
        temp = float(row.get("temperature_c", np.nan))
        if not (-5 <= temp <= 55):
            issues.append(f"Temperature out of range: {temp} C")
    except (TypeError, ValueError):
        issues.append("Temperature is not a valid number")
    try:
        hum = float(row.get("humidity_percent", np.nan))
        if not (0 <= hum <= 100):
            issues.append(f"Humidity out of range: {hum}%")
    except (TypeError, ValueError):
        issues.append("Humidity is not a valid number")
    try:
        wind = float(row.get("wind_speed_mps", np.nan))
        if not (0 <= wind <= 60):
            issues.append(f"Wind speed out of range: {wind} m/s")
    except (TypeError, ValueError):
        issues.append("Wind speed is not a valid number")
    valid_codes = {0,1,2,3,45,48,51,53,55,61,63,65,71,73,75,80,81,82,95,96,99}
    try:
        code = int(float(row.get("weather_code", -1)))
        if code not in valid_codes:
            issues.append(f"Unknown weather code: {code}")
    except (TypeError, ValueError):
        issues.append("Weather code is not a valid number")
    return len(issues) == 0, issues


def validate_speedtest_row(row):
    issues = []
    try:
        lat = float(row.get("ping_avg_rtt_ms", np.nan))
        if not (1 <= lat <= 2000):
            issues.append(f"Latency out of range: {lat} ms")
    except (TypeError, ValueError):
        issues.append("Latency is not a valid number")
    try:
        jit = float(row.get("ping_jitter_ms", np.nan))
        if not (0 <= jit <= 500):
            issues.append(f"Jitter out of range: {jit} ms")
    except (TypeError, ValueError):
        issues.append("Jitter is not a valid number")
    try:
        dl = float(row.get("download_mbps", np.nan))
        if not (0.1 <= dl <= 5000):
            issues.append(f"Download speed out of range: {dl} Mbps")
    except (TypeError, ValueError):
        issues.append("Download speed is not a valid number")
    try:
        ul = float(row.get("upload_mbps", np.nan))
        if not (0.1 <= ul <= 1000):
            issues.append(f"Upload speed out of range: {ul} Mbps")
    except (TypeError, ValueError):
        issues.append("Upload speed is not a valid number")
    return len(issues) == 0, issues


def validate_latest_row(row):
    if row is None:
        return True, []
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    w_ok, w_issues = validate_weather_row(row_dict)
    s_ok, s_issues = validate_speedtest_row(row_dict)
    all_issues = w_issues + s_issues
    return len(all_issues) == 0, all_issues


# ======================
# RUN INTEGRITY CHECK ON STARTUP
# ======================
PROTECTED_FILES   = [THESIS_CLEAN_FILE, THESIS_FORECAST_FILE]
integrity_results = verify_hashes(PROTECTED_FILES)

tampered = [fp for fp, s in integrity_results.items() if s == "TAMPERED"]
if tampered:
    write_audit_log(f"INTEGRITY_ALERT tampered files: {tampered}", current_user)


# ======================
# STYLE
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .main, .block-container {
    background-color: #080b14 !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0d1117 0%, #080b14 100%) !important;
    border-right: 1px solid #1e2d3d !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%) !important;
    border: 1px solid #1e3a5f !important;
    border-top: 3px solid #3b82f6 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 0 20px rgba(59,130,246,0.15) !important;
}

[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] div,
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #10b981 !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #1e3a5f !important;
    gap: 4px !important;
    width: 100% !important;
}

.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #a78bfa !important;
    background: rgba(139,92,246,0.1) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 16px rgba(37,99,235,0.4) !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

.info-box {
    background: linear-gradient(135deg, #0d1f3c 0%, #0a1628 100%);
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #1e3a5f;
    box-shadow: 0 0 20px rgba(59,130,246,0.1);
}

.section-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
}

hr { border-color: #1e2d3d !important; }

[data-testid="stSidebar"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stSidebar"] strong { color: #e2e8f0 !important; }

div[data-baseweb="select"] > div {
    background: #0d1117 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

p, li, span { color: #e2e8f0; }
h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ======================
# HELPERS
# ======================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def kpi_card(label, value, unit, color, icon):
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border: 1px solid {color}44;
        border-top: 3px solid {color};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 20px {color}22;
    ">
        <div style="color:{color}; font-size:1.3rem; margin-bottom:0.2rem;">{icon}</div>
        <div style="color:#94a3b8; font-size:0.7rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px;">{label}</div>
        <div style="color:#ffffff; font-size:1.9rem; font-weight:900;
                    line-height:1.1; margin-top:0.2rem;">
            {value} <span style="font-size:0.9rem; color:#94a3b8; font-weight:500;">{unit}</span>
        </div>
    </div>"""


def make_time_chart(df, x_col, y_col, title, y_label, color="#8b5cf6"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], mode="lines", name=y_col,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.1,)}"
            if color.startswith("#") else "rgba(139,92,246,0.1)"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=14)),
        xaxis_title="Time", yaxis_title=y_label, height=360,
        plot_bgcolor="rgba(13,31,60,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def make_actual_vs_predicted_chart(y_test, y_pred, points_to_show, title):
    actual    = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)
    if points_to_show < len(actual):
        actual    = actual.iloc[-points_to_show:]
        predicted = predicted.iloc[-points_to_show:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual,    mode="lines", name="Actual",
                             line=dict(color="#06b6d4", width=2)))
    fig.add_trace(go.Scatter(y=predicted, mode="lines", name="Predicted",
                             line=dict(color="#f472b6", width=2, dash="dot")))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=14)),
        xaxis_title="Test Time Steps", yaxis_title="Latency (ms)", height=420,
        plot_bgcolor="rgba(13,31,60,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def make_health_gauge(score):
    if score >= 85:
        bar_color = "#10b981"
    elif score >= 70:
        bar_color = "#f59e0b"
    elif score >= 50:
        bar_color = "#f97316"
    else:
        bar_color = "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 28, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
            "bar":  {"color": bar_color, "thickness": 0.4},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.12)"},
                {"range": [50, 70], "color": "rgba(249,115,22,0.12)"},
                {"range": [70, 85], "color": "rgba(245,158,11,0.12)"},
                {"range": [85,100], "color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 3},
                          "thickness": 0.8, "value": score}
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"}
    )
    return fig


def health_badge_html(score):
    if score >= 85:
        return '<span class="badge" style="background:#052e16;color:#10b981;border:1px solid #10b98133;">Excellent</span>'
    elif score >= 70:
        return '<span class="badge" style="background:#1c1a08;color:#f59e0b;border:1px solid #f59e0b33;">Good</span>'
    elif score >= 50:
        return '<span class="badge" style="background:#1c0f05;color:#f97316;border:1px solid #f9731633;">Fair</span>'
    return '<span class="badge" style="background:#1c0505;color:#ef4444;border:1px solid #ef444433;">Poor</span>'


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
    label = ("Excellent" if score >= 85 else
             "Good"      if score >= 70 else
             "Fair"      if score >= 50 else "Poor")
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
                if download > 100
                else "General browsing and normal streaming should work well."
                if download > 40
                else "Heavy streaming or large downloads may be affected.")
    recs.append("Suitable for file uploads and cloud syncing."
                if upload > 20
                else "Normal uploads should work, but large uploads may be slower."
                if upload > 10
                else "Large uploads may be unreliable at this time.")
    return recs


def ai_insight(latency, jitter, download, upload):
    if latency < 35 and jitter < 8 and download > 100 and upload > 20:
        return "Starlink is in a strong operating state and appears suitable for most academic and administrative activities."
    if latency > 50 or jitter > 10:
        return "The network may support general use, but latency-sensitive activities such as live meetings may be affected."
    if download < 50 or upload < 10:
        return "Bandwidth-intensive activities may experience reduced quality. File transfers and streaming should be scheduled with caution."
    return "Current conditions appear acceptable for standard university use."


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


# ======================
# DATA SEPARATION LOGIC
# ======================
def classify_and_route_new_rows():
    if not os.path.exists(RAW_FILE):
        return None
    raw = pd.read_csv(RAW_FILE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed")
    new_rows = raw[raw["timestamp"] > THESIS_END].copy()
    if len(new_rows) == 0:
        return None
    starlink_new = new_rows[new_rows["network_type"].str.strip().str.lower() == "starlink"]
    other_new    = new_rows[new_rows["network_type"].str.strip().str.lower() != "starlink"]
    if len(starlink_new) > 0:
        if os.path.exists(STARLINK_RETRAIN_FILE):
            existing = pd.read_csv(STARLINK_RETRAIN_FILE)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], format="mixed")
            combined = pd.concat([existing, starlink_new]).drop_duplicates(subset=["timestamp"])
        else:
            combined = starlink_new
        combined.to_csv(STARLINK_RETRAIN_FILE, index=False)
    if len(other_new) > 0:
        if os.path.exists(LIVE_MONITOR_FILE):
            existing = pd.read_csv(LIVE_MONITOR_FILE)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], format="mixed")
            combined = pd.concat([existing, other_new]).drop_duplicates(subset=["timestamp"])
        else:
            combined = other_new
        combined.to_csv(LIVE_MONITOR_FILE, index=False)
    latest = new_rows.sort_values("timestamp").iloc[-1]
    return latest


def retrain_queue_status():
    if not os.path.exists(STARLINK_RETRAIN_FILE):
        return 0, None
    df = pd.read_csv(STARLINK_RETRAIN_FILE)
    return len(df), pd.to_datetime(df["timestamp"]).max()


# ======================
# THESIS DATASET LOADER
# ======================
@st.cache_data(ttl=60)
def load_thesis_clean():
    df = pd.read_csv(THESIS_CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= THESIS_START) & (df["timestamp"] <= THESIS_END)]
    return df


@st.cache_data(ttl=60)
def load_thesis_forecast():
    return pd.read_csv(THESIS_FORECAST_FILE)


# ======================
# ML HELPERS
# ======================
def build_metric_dataset(clean_df, target_col):
    df = clean_df.copy()
    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["ping_avg_rtt_ms","ping_jitter_ms","download_mbps","upload_mbps",
              "weather_code","temperature_c","humidity_percent","wind_speed_mps",target_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ping_avg_rtt_ms" in df.columns: df = df[(df["ping_avg_rtt_ms"]>0)&(df["ping_avg_rtt_ms"]<500)]
    if "ping_jitter_ms"  in df.columns: df = df[(df["ping_jitter_ms"]>=0)&(df["ping_jitter_ms"]<100)]
    if "download_mbps"   in df.columns: df = df[(df["download_mbps"]>0)&(df["download_mbps"]<1000)]
    if "upload_mbps"     in df.columns: df = df[(df["upload_mbps"]>0)&(df["upload_mbps"]<500)]
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"]= df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["hour_sin"]   = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]   = np.cos(2*np.pi*df["hour"]/24)
    for lag in range(1,5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    if "ping_avg_rtt_ms" in df.columns: df["latency_lag_1"]     = df["ping_avg_rtt_ms"].shift(1)
    if "ping_jitter_ms"  in df.columns: df["jitter_lag_1"]      = df["ping_jitter_ms"].shift(1)
    if "download_mbps"   in df.columns: df["download_lag_1_ctx"]= df["download_mbps"].shift(1)
    if "upload_mbps"     in df.columns: df["upload_lag_1_ctx"]  = df["upload_mbps"].shift(1)
    df[f"{target_col}_roll_mean_3"] = df[target_col].rolling(3).mean()
    df[f"{target_col}_roll_std_3"]  = df[target_col].rolling(3).std()
    df["target"] = df[target_col].shift(-1)
    df = df.dropna().reset_index(drop=True)
    features = ["hour","day_of_week","is_weekend","hour_sin","hour_cos",
                f"{target_col}_lag_1",f"{target_col}_lag_2",
                f"{target_col}_lag_3",f"{target_col}_lag_4",
                f"{target_col}_roll_mean_3",f"{target_col}_roll_std_3",
                "latency_lag_1","jitter_lag_1","download_lag_1_ctx","upload_lag_1_ctx",
                "weather_code","temperature_c","humidity_percent","wind_speed_mps"]
    features = [c for c in features if c in df.columns]
    return df[["timestamp"]+features+["target"]].copy(), features


def prepare_xy(df):
    df = df.copy()
    if "timestamp" in df.columns: df = df.drop(columns=["timestamp"])
    df = df.select_dtypes(include=["number"])
    return df.drop(columns=["target"]), df["target"]


def train_model(df, model_name):
    X, y = prepare_xy(df)
    split = int(len(df)*0.8)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    if model_name == "Naive Baseline":
        lags = [c for c in Xte.columns if c.endswith("_lag_1")]
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
        raise ValueError("Unknown model.")
    return {"model":model,"X_test":Xte,"y_test":yte,"y_pred":y_pred,
            "mae":float(mean_absolute_error(yte,y_pred)),
            "rmse":float(np.sqrt(mean_squared_error(yte,y_pred)))}


def predict_next_step(df, model_name):
    X, y = prepare_xy(df)
    if len(X) == 0: return 0.0
    latest = X.iloc[[-1]]
    if model_name == "Naive Baseline":
        lags = [c for c in latest.columns if c.endswith("_lag_1")]
        return float(latest[lags[0]].iloc[0])
    if model_name == "Linear Regression":
        m = LinearRegression()
    elif model_name == "Random Forest":
        m = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    else:
        return 0.0
    m.fit(X, y)
    return float(m.predict(latest)[0])


def make_model_comparison_table(df):
    names = ["Naive Baseline","Linear Regression","Random Forest"]
    if XGB_AVAILABLE: names.append("XGBoost")
    rows = []
    for m in names:
        r = train_model(df, m)
        rows.append({"Model":m,"MAE":round(r["mae"],3),"RMSE":round(r["rmse"],3)})
    return pd.DataFrame(rows)


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown(f"Logged in as **{current_user}** `({current_role})`")
    if st.button("Logout"):
        write_audit_log("LOGOUT", current_user)
        for key in ["authenticated", "role", "username", "last_active", "failed_attempts"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")
    st.markdown("### Controls")

    model_options = ["Naive Baseline","Linear Regression","Random Forest"]
    if XGB_AVAILABLE: model_options.append("XGBoost")
    selected_model = st.selectbox("Forecast model", model_options, index=1)
    chart_points   = st.slider("Prediction chart points", 50, 400, 200, 25)

    st.markdown("---")

    show_live = st.toggle("Show live KPI overlay", value=False,
                          help="Displays latest logged measurement on KPI cards. Does NOT affect the model or charts.")

    st.markdown("---")
    st.markdown("**GUtech AI Thesis · 2026**")
    st.markdown("Starlink Performance Analysis")
    st.markdown("Muscat, Oman · March 7–28 dataset")

    q_count, q_latest = retrain_queue_status()
    if q_count > 0:
        st.markdown("---")
        st.markdown(f"**Starlink retrain queue:** {q_count} rows")
        st.markdown(f"Latest: `{q_latest}`")
        st.caption("These are new Starlink-only measurements eligible for future retraining.")
    else:
        st.markdown("---")
        st.caption("No new Starlink data queued for retraining yet.")

    # ---- DATA INTEGRITY ----
    st.markdown("---")
    st.markdown("### Data Integrity")
    all_ok = all(v == "OK" for v in integrity_results.values())
    if all_ok:
        st.success("All CSV files verified")
    else:
        for fp, status in integrity_results.items():
            fname = os.path.basename(fp)
            if status == "OK":
                st.success(f"{fname}: OK")
            elif status == "TAMPERED":
                st.error(f"{fname}: TAMPERED")
            elif status == "MISSING":
                st.warning(f"{fname}: MISSING")
            else:
                st.info(f"{fname}: {status}")

    if is_admin:
        if st.button("Reset hash baseline"):
            save_hashes(PROTECTED_FILES)
            write_audit_log("HASH_BASELINE_RESET", current_user)
            st.success("Hashes updated.")
    else:
        st.caption("Hash reset available to admin only.")




# ======================
# LOAD THESIS DATA
# ======================
thesis_clean_df  = load_thesis_clean()
thesis_forecast  = load_thesis_forecast()
download_df, _   = build_metric_dataset(thesis_clean_df, "download_mbps")
upload_df, _     = build_metric_dataset(thesis_clean_df, "upload_mbps")

latest_live_row = classify_and_route_new_rows()
api_valid, api_issues = validate_latest_row(latest_live_row)

latency_result    = train_model(thesis_forecast, selected_model)
latency_forecast  = predict_next_step(thesis_forecast, selected_model)
download_forecast = predict_next_step(download_df, "Linear Regression")
upload_forecast   = predict_next_step(upload_df,   "Linear Regression")

if download_forecast <= 0:
    download_forecast = float(thesis_clean_df["download_mbps"].dropna().iloc[-1]) if "download_mbps" in thesis_clean_df.columns else 0.0
if upload_forecast <= 0:
    upload_forecast = float(thesis_clean_df["upload_mbps"].dropna().iloc[-1]) if "upload_mbps" in thesis_clean_df.columns else 0.0

if show_live and latest_live_row is not None:
    display_row       = latest_live_row
    live_source_label = latest_live_row.get("network_type", "Unknown")
    is_live_starlink  = str(live_source_label).strip().lower() == "starlink"
    kpi_note = ("Live Starlink data — consistent with research dataset"
                if is_live_starlink
                else f"Live monitor only ({live_source_label}) — not part of research dataset")
else:
    display_row = thesis_clean_df.iloc[-1]
    kpi_note    = "Showing last observation from validated March 7–28 research dataset"

def safe_float(row, col):
    try: return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
    except: return 0.0

current_latency  = safe_float(display_row, "ping_avg_rtt_ms")
current_jitter   = safe_float(display_row, "ping_jitter_ms")
current_download = safe_float(display_row, "download_mbps")
current_upload   = safe_float(display_row, "upload_mbps")

health_score, health_label = compute_health(latency_forecast, current_jitter,
                                            download_forecast, upload_forecast)
alerts          = generate_alerts(latency_forecast, current_jitter, download_forecast, upload_forecast)
recommendations = usage_recommendation(latency_forecast, current_jitter, download_forecast, upload_forecast)
system_insight  = ai_insight(latency_forecast, current_jitter, download_forecast, upload_forecast)
outage_status   = detect_outage(latency_forecast, download_forecast, upload_forecast)

latest_time    = thesis_clean_df["timestamp"].max()
forecast_time  = latest_time + pd.Timedelta(minutes=15)

weather_row = display_row
latest_weather_code = weather_row.get("weather_code",  np.nan) if hasattr(weather_row,"get") else weather_row["weather_code"] if "weather_code" in weather_row.index else np.nan
latest_temp         = weather_row.get("temperature_c", np.nan) if hasattr(weather_row,"get") else weather_row["temperature_c"] if "temperature_c" in weather_row.index else np.nan
latest_humidity     = weather_row.get("humidity_percent", np.nan) if hasattr(weather_row,"get") else weather_row["humidity_percent"] if "humidity_percent" in weather_row.index else np.nan
latest_wind         = weather_row.get("wind_speed_mps", np.nan) if hasattr(weather_row,"get") else weather_row["wind_speed_mps"] if "wind_speed_mps" in weather_row.index else np.nan

# ======================
# HEADER
# ======================
st.caption(f"Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

if show_live and not api_valid and api_issues:
    st.warning("API Validation Warning: Live data contains suspicious values — " + " | ".join(api_issues))

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1f3c 0%, #1a0533 50%, #0d1f3c 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1rem;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 0 40px rgba(99,102,241,0.15);
    display: flex; justify-content: space-between; align-items: center;
">
    <div>
        <div style="
            font-size: 2.2rem; font-weight: 900; letter-spacing: -1px; line-height: 1.1;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        ">Starlink Performance<br>Forecast System</div>
        <div style="color:#64748b; font-size:0.9rem; margin-top:0.5rem; font-weight:500;">
            University Network Intelligence Dashboard &nbsp;·&nbsp;
            Monitoring, Forecasting &amp; Decision Support
        </div>
    </div>
    <div style="text-align:right;">
        <div style="
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            padding: 0.4rem 1rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 700; color: #ffffff;
            letter-spacing: 0.5px; margin-bottom: 0.4rem;
        ">GUtech AI Thesis Prototype</div>
        <div style="color:#64748b; font-size:0.78rem;">Starlink · Muscat, Oman · 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(90deg, #052e16, #0d2818);
    border: 1px solid #10b98133; border-left: 4px solid #10b981;
    border-radius: 10px; padding: 0.7rem 1.2rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.8rem;
">
    <span style="font-size:1.2rem;">🛰️</span>
    <div>
        <span style="color:#10b981; font-weight:700; font-size:0.85rem;">
            VALIDATED RESEARCH DATASET
        </span>
        <span style="color:#64748b; font-size:0.82rem; margin-left:0.8rem;">
            All charts, models, and forecasts use Starlink data from
            March 7–28, 2026 only. Live logging is routed separately
            and does not affect research results.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="
    background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
    border-radius:8px; padding:0.5rem 1rem; margin-bottom:0.8rem;
    font-size:0.82rem; color:#94a3b8;
">{kpi_note}</div>
""", unsafe_allow_html=True)

# ======================
# KPI CARDS
# ======================
c1, c2, c3, c4, c5 = st.columns(5)
hcolor = ("#10b981" if health_score>=85 else "#f59e0b" if health_score>=70
          else "#f97316" if health_score>=50 else "#ef4444")

with c1: st.markdown(kpi_card("Latency",      f"{current_latency:.1f}",  "ms",   "#60a5fa", "📡"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("Jitter",       f"{current_jitter:.1f}",   "ms",   "#f472b6", "〰️"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("Download",     f"{current_download:.1f}", "Mbps", "#34d399", "⬇"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("Upload",       f"{current_upload:.1f}",   "Mbps", "#fb923c", "⬆"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("Health Score", f"{health_score}",         "/100", hcolor,    "💚"), unsafe_allow_html=True)

st.markdown(
    f'<div style="margin-top:0.6rem; margin-bottom:1rem;">'
    f'{health_badge_html(health_score)}'
    f'&nbsp;&nbsp;<span style="color:#94a3b8; font-size:0.9rem;">'
    f'<b>Network status:</b> {outage_status}</span></div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Historical Trends", "Forecasting",
    "Alerts & Recommendations", "Model Evaluation"
])

# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.markdown('<div class="section-title">Network Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(
            make_time_chart(thesis_clean_df, "timestamp", "ping_avg_rtt_ms",
                            "Latency Over Time (Thesis Dataset — March 7–28)", "Latency (ms)",
                            color="#8b5cf6"),
            use_container_width=True)
    with right:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Current Forecast")
        st.metric("Next 15-min Latency",  f"{latency_forecast:.2f} ms")
        st.metric("Next 15-min Download", f"{download_forecast:.2f} Mbps")
        st.metric("Next 15-min Upload",   f"{upload_forecast:.2f} Mbps")
        st.write(f"Forecast time: **{forecast_time}**")
        st.write(f"Model: **{selected_model}**")
        st.markdown('</div>', unsafe_allow_html=True)

    a, b, c = st.columns([1, 1.15, 1])
    with a:
        st.plotly_chart(make_health_gauge(health_score), use_container_width=True)
    with b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Recommended Usage")
        for rec in recommendations: st.write(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Latest Weather Context")
        st.write(f"Condition: **{fmt_code(latest_weather_code)}**")
        st.write(f"Temperature: **{fmt_temp(latest_temp)}**")
        st.write(f"Humidity: **{fmt_humidity(latest_humidity)}**")
        st.write(f"Wind: **{fmt_wind(latest_wind)}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### AI Insight")
    st.write(system_insight)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB 2: HISTORICAL TRENDS ----------
with tab2:
    st.markdown('<div class="section-title">Historical Performance — Validated Starlink Dataset</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_time_chart(thesis_clean_df, "timestamp", "ping_jitter_ms",
                            "Jitter Over Time", "Jitter (ms)", "#f472b6"),
            use_container_width=True)
    with col2:
        st.plotly_chart(
            make_time_chart(thesis_clean_df, "timestamp", "download_mbps",
                            "Download Speed Over Time", "Download (Mbps)", "#34d399"),
            use_container_width=True)
    st.plotly_chart(
        make_time_chart(thesis_clean_df, "timestamp", "upload_mbps",
                        "Upload Speed Over Time", "Upload (Mbps)", "#fb923c"),
        use_container_width=True)

# ---------- TAB 3: FORECASTING ----------
with tab3:
    st.markdown('<div class="section-title">Forecasting — Model trained on March 7–28 Starlink data</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        make_actual_vs_predicted_chart(
            latency_result["y_test"], latency_result["y_pred"],
            chart_points, f"Actual vs Predicted Latency ({selected_model})"),
        use_container_width=True)

    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast Latency",  f"{latency_forecast:.2f} ms")
    f2.metric("Forecast Download", f"{download_forecast:.2f} Mbps")
    f3.metric("Forecast Upload",   f"{upload_forecast:.2f} Mbps")

    st.markdown(
        '<div class="small-note" style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">'
        'Latency is the primary research forecasting target. Download and upload are supporting '
        'operational forecasts. All predictions are derived from the validated March 7–28 '
        'Starlink experiment dataset exclusively.</div>',
        unsafe_allow_html=True)

    q_count, q_latest = retrain_queue_status()
    if q_count > 0:
        st.markdown("---")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### New Starlink Data Available for Retraining")
        st.write(f"**{q_count} new Starlink rows** have been collected after the thesis experiment window.")
        st.write(f"Latest timestamp: **{q_latest}**")
        st.write("These rows are stored in `Cleaned/starlink_retrain_queue.csv` and can be merged "
                 "with the thesis dataset to retrain the model when you are ready.")
        st.code("python build_dataset.py --include-retrain-queue\nstreamlit run dashboard_thesis.py",
                language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB 4: ALERTS ----------
with tab4:
    st.markdown('<div class="section-title">Alerts &amp; Recommendations</div>', unsafe_allow_html=True)
    for level, message in alerts:
        if level == "success":  st.success(message)
        elif level == "warning": st.warning(message)
        else:                    st.error(message)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Operational Interpretation")
        st.write(f"Forecast-based health status is **{health_label}** with a score of "
                 f"**{health_score}/100**. This combines predicted latency, forecast bandwidth, "
                 f"and current jitter.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Suggested Use Cases Right Now")
        for rec in recommendations: st.write(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- TAB 5: MODEL EVALUATION ----------
with tab5:
    if is_guest:
        st.warning("Access restricted. This tab is not available to guest users.")
    else:
        st.markdown('<div class="section-title">Model Evaluation — Thesis Dataset</div>',
                    unsafe_allow_html=True)

        model_table = make_model_comparison_table(thesis_forecast)
        st.dataframe(model_table, use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Selected Model MAE",  f"{latency_result['mae']:.3f}")
        m2.metric("Selected Model RMSE", f"{latency_result['rmse']:.3f}")

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Research Notes")
        st.write(
            "All models were trained and evaluated exclusively on the validated Starlink dataset "
            "from March 7–28, 2026. The naive baseline serves as benchmark. Linear Regression was "
            "selected as the final model due to the best balance of accuracy, stability, and "
            "interpretability. More complex models did not add predictive value, confirming that "
            "Starlink latency follows primarily linear patterns driven by recent history. All models "
            "struggled to predict sudden latency spikes, which are caused by satellite handoffs and "
            "environmental factors not captured in the available features."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if is_admin:
            st.markdown("### Export Results")
            st.download_button(
                "Download Model Comparison CSV",
                model_table.to_csv(index=False).encode("utf-8"),
                "model_comparison_thesis.csv", "text/csv")

            pred_df = pd.DataFrame({
                "actual": latency_result["y_test"].values,
                "predicted": latency_result["y_pred"]
            })
            st.download_button(
                "Download Latency Predictions CSV",
                pred_df.to_csv(index=False).encode("utf-8"),
                "latency_predictions_thesis.csv", "text/csv")

            st.markdown("---")
            st.markdown("### Audit Log")
            if os.path.exists(AUDIT_LOG_FILE):
                with open(AUDIT_LOG_FILE, "r") as f:
                    lines = f.readlines()
                last_lines = lines[-20:] if len(lines) > 20 else lines
                st.text("".join(last_lines))
            else:
                st.caption("No audit log entries yet.")
        else:
            st.caption("CSV export and audit log are available to admin only.")
