import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Starlink Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================
# FILE PATHS (same pipeline as dashboard_thesis.py)
# ======================
THESIS_CLEAN_FILE = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
THESIS_START      = pd.Timestamp("2026-03-07 01:00:00")
THESIS_END        = pd.Timestamp("2026-03-28 11:00:00")

# Trained ML latency forecast model created by train_model_combined.py
AI_MODEL_FILE    = "Models/starlink_latency_forecast_model.pkl"
AI_FEATURES_FILE = "Models/starlink_latency_features.pkl"

# ======================
# STYLE
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp, .main, .block-container {
    background-color: #05080f !important;
    color: #dde3ed !important;
    font-family: 'DM Sans', sans-serif !important;
}

.block-container {
    padding-top: 3.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }

div[data-baseweb="select"] > div {
    background: #0c1423 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    color: #dde3ed !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] svg { color: #4a7fc1 !important; }

[data-testid="stMultiSelect"] > div > div {
    background: #0c1423 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
}

[data-testid="stNumberInput"] input {
    background: #0c1423 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    color: #dde3ed !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #6366f1) !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.4) !important;
}

h1, h2, h3 { color: #ffffff !important; font-family: 'DM Sans', sans-serif !important; }
p, li, span { color: #dde3ed; }
hr { border-color: #1e3050 !important; }

.step-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #4a7fc1; margin-bottom: 0.6rem;
}
.section-card {
    background: linear-gradient(135deg, #0c1423 0%, #080e1c 100%);
    border: 1px solid #1e3050; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.2rem;
}
.q-label {
    font-size: 0.78rem; font-weight: 700; color: #4a7fc1;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 0.25rem; margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ======================
# DATA — load thesis medians as base
# ======================
@st.cache_data(ttl=300)
def load_base():
    try:
        df = pd.read_csv(THESIS_CLEAN_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= THESIS_START) & (df["timestamp"] <= THESIS_END)]
        for c in ["ping_avg_rtt_ms","ping_jitter_ms","download_mbps","upload_mbps"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[
            (df["ping_avg_rtt_ms"]  > 0) & (df["ping_avg_rtt_ms"]  < 500) &
            (df["ping_jitter_ms"]  >= 0) & (df["ping_jitter_ms"]   < 100) &
            (df["download_mbps"]    > 0) & (df["download_mbps"]    < 1000) &
            (df["upload_mbps"]      > 0) & (df["upload_mbps"]      < 500)
        ]
        return {
            "latency":  float(df["ping_avg_rtt_ms"].median()),
            "jitter":   float(df["ping_jitter_ms"].median()),
            "download": float(df["download_mbps"].median()),
            "upload":   float(df["upload_mbps"].median()),
            "loaded": True
        }
    except Exception:
        return {"latency": 38.0, "jitter": 6.5, "download": 145.0, "upload": 22.0, "loaded": False}


@st.cache_resource
def load_ai_latency_model():
    """Load the trained latency forecasting model and expected feature order."""
    try:
        model = joblib.load(AI_MODEL_FILE)
        features = joblib.load(AI_FEATURES_FILE)
        return model, features, True, "Loaded"
    except Exception as e:
        return None, None, False, str(e)


# ======================
# CONDITION TABLES
# ======================
LOCATION_ADJ = {
    "City (Muscat or major urban area)":   {"lat": 0,  "dl": 1.00},
    "Suburb (outside city centre)":        {"lat": 3,  "dl": 0.95},
    "Rural area":                          {"lat": 6,  "dl": 0.90},
    "Desert / very remote area":           {"lat": 10, "dl": 0.82},
}
SKY_ADJ = {
    "Completely clear, no obstructions":  {"lat": 0,  "dl": 1.00},
    "Mostly clear, minor trees or walls": {"lat": 3,  "dl": 0.95},
    "Partially blocked, buildings nearby":{"lat": 8,  "dl": 0.88},
    "Heavily obstructed":                  {"lat": 18, "dl": 0.70},
}
TIME_ADJ = {
    "Morning (6 AM – 12 PM)":             {"lat": 0,  "dl": 1.00},
    "Afternoon (12 PM – 6 PM)":           {"lat": 2,  "dl": 0.97},
    "Evening (6 PM – 10 PM)":            {"lat": 8,  "dl": 0.88},
    "Night (10 PM – 6 AM)":               {"lat": -2, "dl": 1.05},
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


# ======================
# CORE LOGIC
# ======================
def weather_to_code(weather):
    """Convert the advisor weather choice into the numeric weather_code used by the ML model."""
    mapping = {
        "Clear / sunny": 0,
        "Partly cloudy": 2,
        "Overcast / cloudy": 3,
        "Light rain or drizzle": 51,
        "Heavy rain or thunderstorm": 95,
        "Sandstorm / dust": 3,
    }
    return mapping.get(weather, 0)


def time_to_hour(time_of_day):
    """Use a representative hour for each advisor time period."""
    mapping = {
        "Morning (6 AM – 12 PM)": 9,
        "Afternoon (12 PM – 6 PM)": 15,
        "Evening (6 PM – 10 PM)": 20,
        "Night (10 PM – 6 AM)": 23,
    }
    return mapping.get(time_of_day, 12)


def ai_predict_latency(model, features, perf, time_of_day, weather):
    """
    Predict next-step latency using the trained ML model.

    The training dataset uses lag features. In this advisor we only have one
    condition-adjusted estimate, so the same estimated latency is used for all
    latency lag columns. Live measurements in Step 3 still use the user's real
    measured values.
    """
    hour = time_to_hour(time_of_day)
    day_of_week = pd.Timestamp.now().dayofweek
    is_weekend = int(day_of_week in [5, 6])

    row = {
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "ping_avg_rtt_ms_lag_1": perf["latency"],
        "ping_avg_rtt_ms_lag_2": perf["latency"],
        "ping_avg_rtt_ms_lag_3": perf["latency"],
        "ping_avg_rtt_ms_lag_4": perf["latency"],
        "download_lag_1": perf["download"],
        "upload_lag_1": perf["upload"],
        "jitter_lag_1": perf["jitter"],
        "weather_code": weather_to_code(weather),
    }

    X = pd.DataFrame([row])

    # Guarantee exact training feature order and fill any missing columns safely.
    for f in features:
        if f not in X.columns:
            X[f] = 0
    X = X[features]

    pred = float(model.predict(X)[0])
    return max(10.0, pred)


def apply_conditions(base, location, sky, time_of_day, weather, temp):
    adjs = [LOCATION_ADJ[location], SKY_ADJ[sky], TIME_ADJ[time_of_day],
            WEATHER_ADJ[weather], TEMP_ADJ[temp]]
    lat_delta = sum(a["lat"] for a in adjs)
    dl_factor = 1.0
    for a in adjs:
        dl_factor *= a["dl"]
    return {
        "latency":  max(10.0, base["latency"]  + lat_delta),
        "jitter":   max(2.0,  base["jitter"]   + lat_delta * 0.25),
        "download": max(2.0,  base["download"] * dl_factor),
        "upload":   max(1.0,  base["upload"]   * dl_factor),
    }


def traffic_light(uc_name, lat, jitter, dl, ul):
    uc = USE_CASES[uc_name]
    score = 0
    if lat    > uc["lat_max"]:    score += 2
    if jitter > uc["jitter_max"]: score += 1
    if dl     < uc["dl_min"]:     score += 2
    if ul     < uc["ul_min"]:     score += 2
    return "green" if score == 0 else "amber" if score <= 2 else "red"


def metric_card(label, value, unit, color):
    return f"""
    <div style="background:{color}11;border:1px solid {color}33;border-top:3px solid {color};
        border-radius:12px;padding:0.9rem 1rem;text-align:center;">
        <div style="color:#64748b;font-size:0.68rem;font-weight:700;
                    text-transform:uppercase;letter-spacing:1px;">{label}</div>
        <div style="color:#fff;font-size:1.6rem;font-weight:800;line-height:1.2;margin-top:0.2rem;">
            {value}<span style="font-size:0.8rem;color:#94a3b8;font-weight:400;"> {unit}</span>
        </div>
    </div>"""


def rating_card(color, uc_name, icon, desc, change_note=""):
    bg     = {"green":"#031a0e","amber":"#171203","red":"#1a0303"}[color]
    border = {"green":"#10b981","amber":"#f59e0b","red":"#ef4444"}[color]
    label  = {"green":"Suitable","amber":"Limited","red":"Not Recommended"}[color]
    tc     = border
    change_html = (
        f'<div style="color:#94a3b8;font-size:0.75rem;margin-top:0.2rem;">{change_note}</div>'
        if change_note else ""
    )
    return (
        f'<div style="background:{bg};border:1px solid {border}44;border-left:5px solid {border};'
        f'border-radius:12px;padding:1rem 1.2rem;'
        f'display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">'
        f'<div style="font-size:1.6rem;">{icon}</div>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:600;color:#e2e8f0;">{uc_name}</div>'
        f'<div style="color:#64748b;font-size:0.8rem;margin-top:0.1rem;">{desc}</div>'
        f'{change_html}'
        f'</div>'
        f'<div style="background:{tc}22;border:1px solid {tc}66;border-radius:999px;'
        f'padding:0.3rem 0.9rem;color:{tc};font-weight:700;font-size:0.82rem;'
        f'white-space:nowrap;">{label}</div>'
        f'</div>'
    )


def linear_forecast(values):
    """Return next predicted value using linear regression on a list of floats."""
    if len(values) < 2:
        return None
    x = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    m = LinearRegression().fit(x, y)
    return max(0.0, float(m.predict([[len(values)]])[0]))


def trend_chart(values, label, color):
    series = pd.Series(values)
    next_val = linear_forecast(values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))), y=series.values,
        mode="lines+markers", name="Measured",
        line=dict(color=color, width=2), marker=dict(size=6)
    ))
    if next_val is not None:
        fig.add_trace(go.Scatter(
            x=[len(series)-1, len(series)],
            y=[series.iloc[-1], next_val],
            mode="lines+markers", name="Next forecast",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=9, symbol="diamond")
        ))
    fig.update_layout(
        title=dict(text=label, font=dict(color="#ffffff", size=13)),
        height=200,
        plot_bgcolor="rgba(12,20,35,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#64748b", title="Test #"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
        margin=dict(l=10, r=10, t=36, b=10)
    )
    return fig, next_val


# ======================
# SESSION STATE
# ======================
defaults = {"step": 1, "advisory": None, "measurements": []}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

base = load_base()
ai_latency_model, ai_features, ai_model_loaded, ai_model_message = load_ai_latency_model()


# ======================
# HEADER
# ======================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1f3c 0%, #0a0d1f 50%, #0d1f3c 100%);
    border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; overflow: hidden;
    border: 1px solid rgba(99,102,241,0.5);
    box-shadow: 0 0 40px rgba(99,102,241,0.1);
    display: flex; justify-content: space-between; align-items: center;
">
    <div>
        <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:2px;color:#4a7fc1;margin-bottom:0.5rem;">
            Starlink Suitability Advisor · Oman
        </div>
        <div style="
            font-size: 2.2rem; font-weight: 900; letter-spacing: -1px; line-height: 1.1;
            font-family: 'DM Sans', system-ui, sans-serif;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        ">Will Starlink work for you?</div>
        <div style="color:#64748b; font-size:0.9rem; margin-top:0.5rem;
                    font-family:'DM Sans', system-ui, sans-serif;">
            Answer six questions. Get an honest, data-backed answer.
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

# Progress bar
steps_labels = ["1. Your conditions", "2. Advisory result", "3. Live forecast"]
prog_cols = st.columns(3)
for i, (col, name) in enumerate(zip(prog_cols, steps_labels)):
    n = i + 1
    active = n == st.session_state.step
    done   = n <  st.session_state.step
    if active:
        bg = "linear-gradient(135deg,#1d4ed8,#4f46e5)"
        tc = "#ffffff"
        border = "none"
        label = name
    elif done:
        bg = "#052e16"
        tc = "#10b981"
        border = "1px solid #10b98144"
        label = "✓ " + name.split(". ",1)[1]
    else:
        bg = "#0c1423"
        tc = "#334155"
        border = "1px solid #1e3050"
        label = name
    col.markdown(
        f'<div style="background:{bg};border:{border};border-radius:10px;'
        f'padding:0.6rem 1rem;text-align:center;font-size:0.8rem;'
        f'font-weight:{"700" if active or done else "400"};color:{tc};">'
        f'{label}</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)


# ======================
# STEP 1 — SIX QUESTIONS
# ======================
if st.session_state.step == 1:

    st.markdown("""
    <div style="border-left:4px solid #3b82f6;padding-left:1rem;margin-bottom:1.2rem;">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:1.8px;color:#3b82f6;margin-bottom:0.2rem;">Step 1 of 3</div>
        <div style="font-size:1.15rem;font-weight:700;color:#ffffff;">Your conditions</div>
        <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">
            Tell us about your location, environment, and what you need Starlink for.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="q-label" style="margin-top:0;">1. Where are you located?</div>', unsafe_allow_html=True)
        location = st.selectbox("loc", list(LOCATION_ADJ.keys()), label_visibility="collapsed", key="q_loc")

        st.markdown('<div class="q-label">2. How clear is the sky view for the dish?</div>', unsafe_allow_html=True)
        sky = st.selectbox("sky", list(SKY_ADJ.keys()), label_visibility="collapsed", key="q_sky")

        st.markdown('<div class="q-label">3. What time of day will you mainly use it?</div>', unsafe_allow_html=True)
        time_of_day = st.selectbox("time", list(TIME_ADJ.keys()), label_visibility="collapsed", key="q_time")

    with c2:
        st.markdown('<div class="q-label" style="margin-top:0;">4. What is the weather usually like?</div>', unsafe_allow_html=True)
        weather = st.selectbox("weather", list(WEATHER_ADJ.keys()), label_visibility="collapsed", key="q_weather")

        st.markdown('<div class="q-label">5. What is the usual temperature?</div>', unsafe_allow_html=True)
        temp = st.selectbox("temp", list(TEMP_ADJ.keys()), label_visibility="collapsed", key="q_temp")

        st.markdown('<div class="q-label">6. What will you use it for?</div>', unsafe_allow_html=True)
        uses = st.multiselect("uses", list(USE_CASES.keys()),
                              default=["Video calls / online meetings", "General browsing / social media"],
                              label_visibility="collapsed", key="q_uses")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button("Get My Advisory Result →"):
        if not uses:
            st.warning("Please select at least one use case.")
        else:
            # 1) Start with the existing rule/condition-based performance estimate.
            perf = apply_conditions(base, location, sky, time_of_day, weather, temp)
            rule_latency = perf["latency"]
            ai_latency = None

            # 2) Improve the latency estimate using the trained ML forecasting model.
            if ai_model_loaded:
                ai_latency = ai_predict_latency(
                    ai_latency_model,
                    ai_features,
                    perf,
                    time_of_day,
                    weather
                )

                perf["latency"] = ai_latency

                # Keep jitter consistent with the AI latency change.
                latency_change = ai_latency - rule_latency
                perf["jitter"] = max(2.0, perf["jitter"] + latency_change * 0.25)

            # 3) Classify each use case using the AI-enhanced performance values.
            ratings = {
                uc: traffic_light(
                    uc,
                    perf["latency"],
                    perf["jitter"],
                    perf["download"],
                    perf["upload"]
                )
                for uc in uses
            }

            st.session_state.advisory = {
                "location": location, "sky": sky, "time": time_of_day,
                "weather": weather, "temp": temp, "uses": uses,
                "perf": perf, "ratings": ratings,
                "rule_latency": rule_latency,
                "ai_latency": ai_latency,
                "ai_model_loaded": ai_model_loaded,
                "ai_model_message": ai_model_message,
            }
            st.session_state.step = 2
            st.rerun()


# ======================
# STEP 2 — ADVISORY RESULT
# ======================
elif st.session_state.step == 2:
    adv = st.session_state.advisory
    perf = adv["perf"]

    st.markdown("""
    <div style="border-left:4px solid #a78bfa;padding-left:1rem;margin-bottom:1.2rem;">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:1.8px;color:#a78bfa;margin-bottom:0.2rem;">Step 2 of 3</div>
        <div style="font-size:1.15rem;font-weight:700;color:#ffffff;">Advisory result</div>
        <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">
            Estimated performance based on your conditions and Muscat data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if adv.get("ai_model_loaded") and adv.get("ai_latency") is not None:
        st.markdown(f"""
        <div style="background:#0c1423;border:1px solid #2563eb55;border-left:4px solid #2563eb;
            border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;
            color:#94a3b8;font-size:0.83rem;line-height:1.7;">
            <span style="color:#60a5fa;font-weight:700;">AI latency forecast applied</span><br>
            Rule-based estimated latency:
            <strong style="color:#e2e8f0;">{adv['rule_latency']:.1f} ms</strong><br>
            AI-predicted latency from the retrained ML model:
            <strong style="color:#e2e8f0;">{adv['ai_latency']:.1f} ms</strong><br>
            The suitability rating below uses the AI-predicted latency.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#1a0303;border:1px solid #ef444455;border-left:4px solid #ef4444;
            border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;
            color:#94a3b8;font-size:0.83rem;line-height:1.7;">
            <span style="color:#ef4444;font-weight:700;">AI model not loaded</span><br>
            The advisor is using the original rule-based estimate only.<br>
            Model message: <code>{adv.get('ai_model_message', 'Unknown')}</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Estimated performance for your conditions**")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(metric_card("Latency",  f"{perf['latency']:.1f}",  "ms",   "#60a5fa"), unsafe_allow_html=True)
    mc2.markdown(metric_card("Jitter",   f"{perf['jitter']:.1f}",   "ms",   "#f472b6"), unsafe_allow_html=True)
    mc3.markdown(metric_card("Download", f"{perf['download']:.1f}", "Mbps", "#34d399"), unsafe_allow_html=True)
    mc4.markdown(metric_card("Upload",   f"{perf['upload']:.1f}",   "Mbps", "#fb923c"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("**Use case ratings**")
    for uc_name, color in adv["ratings"].items():
        uc = USE_CASES[uc_name]
        st.markdown(rating_card(color, uc_name, uc["icon"], uc["desc"]), unsafe_allow_html=True)

    # Confidence note
    st.markdown("""
    <div style="background:#0f1117;border:1px solid #f59e0b33;border-left:4px solid #f59e0b;
        border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;
        color:#94a3b8;font-size:0.83rem;line-height:1.7;">
        <span style="color:#f59e0b;font-weight:700;">About these estimates</span><br>
        The advisor first applies condition-based adjustments, then uses the retrained
        machine-learning latency forecast model when available. Results at your location
        may still differ due to satellite beam assignment, local obstructions, hardware,
        or congestion not captured during the experiment. Treat amber ratings as a reason
        to test before committing, and red ratings as a genuine risk.
    </div>
    """, unsafe_allow_html=True)

    # Prompt to continue to live step
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1f3c,#0a0d1f);
        border:1px solid #1e3a5f;border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;">
        <div style="color:#ffffff;font-weight:700;font-size:1rem;margin-bottom:0.4rem;">
            Want a forecast based on your actual connection?
        </div>
        <div style="color:#64748b;font-size:0.85rem;line-height:1.6;">
            Connect to Starlink, run a few speed tests, and enter the results below.
            The system will use your real measurements to personalise the prediction
            and update the ratings based on what your connection is actually doing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    bc, nc = st.columns([1, 2])
    with bc:
        if st.button("← Change my answers"):
            st.session_state.step = 1
            st.rerun()
    with nc:
        if st.button("I am connected to Starlink, get my forecast →"):
            st.session_state.measurements = []
            st.session_state.step = 3
            st.rerun()


# ======================
# STEP 3 — LIVE MEASUREMENTS + FORECAST
# ======================
elif st.session_state.step == 3:
    adv = st.session_state.advisory

    st.markdown("""
    <div style="border-left:4px solid #34d399;padding-left:1rem;margin-bottom:1.2rem;">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:1.8px;color:#34d399;margin-bottom:0.2rem;">Step 3 of 3</div>
        <div style="font-size:1.15rem;font-weight:700;color:#ffffff;">Live connection forecast</div>
        <div style="font-size:0.82rem;color:#64748b;margin-top:0.2rem;">
            Enter your real speed test results to get a personalised forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;margin-bottom:1.2rem;">
        Make sure you are connected to Starlink. Run a speed test at
        <strong style="color:#60a5fa;">fast.com</strong> or
        <strong style="color:#60a5fa;">speedtest.net</strong> and type in the results.
        Add at least 3 measurements, a few minutes apart, to get a reliable forecast.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Enter your speed test result**")
    i1, i2, i3, i4 = st.columns(4)
    in_lat = i1.number_input("Latency (ms)",    min_value=0.0, max_value=500.0,  value=0.0, step=0.5, key="in_lat")
    in_dl  = i2.number_input("Download (Mbps)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0, key="in_dl")
    in_ul  = i3.number_input("Upload (Mbps)",   min_value=0.0, max_value=500.0,  value=0.0, step=1.0, key="in_ul")
    in_jit = i4.number_input("Jitter (ms)",     min_value=0.0, max_value=200.0,  value=0.0, step=0.5, key="in_jit")

    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("Add measurement"):
            if in_lat > 0 or in_dl > 0:
                st.session_state.measurements.append(
                    {"latency": in_lat, "download": in_dl, "upload": in_ul, "jitter": in_jit}
                )
                st.rerun()
            else:
                st.warning("Enter at least latency or download speed before adding.")

    measurements = st.session_state.measurements

    if measurements:
        mdf = pd.DataFrame(measurements)
        mdf.index = [f"Test {i+1}" for i in range(len(mdf))]
        st.markdown(
            f"<div style='color:#64748b;font-size:0.82rem;margin-top:0.6rem;'>"
            f"{len(measurements)} measurement(s) recorded</div>",
            unsafe_allow_html=True
        )
        st.dataframe(mdf.rename(columns={
            "latency":"Latency (ms)","download":"Download (Mbps)",
            "upload":"Upload (Mbps)","jitter":"Jitter (ms)"
        }), use_container_width=True)

        if len(measurements) >= 2:
            st.markdown("---")
            st.markdown("**Trend and next-step forecast from your measurements**")

            lats = [m["latency"]  for m in measurements]
            dls  = [m["download"] for m in measurements]
            uls  = [m["upload"]   for m in measurements]
            jits = [m["jitter"]   for m in measurements]

            tc1, tc2 = st.columns(2)
            with tc1:
                fig_lat, fc_lat = trend_chart(lats, "Latency (ms)", "#60a5fa")
                st.plotly_chart(fig_lat, use_container_width=True)
            with tc2:
                fig_dl, fc_dl = trend_chart(dls, "Download (Mbps)", "#34d399")
                st.plotly_chart(fig_dl, use_container_width=True)

            tc3, tc4 = st.columns(2)
            with tc3:
                fig_ul, fc_ul = trend_chart(uls, "Upload (Mbps)", "#fb923c")
                st.plotly_chart(fig_ul, use_container_width=True)
            with tc4:
                fig_jit, fc_jit = trend_chart(jits, "Jitter (ms)", "#f472b6")
                st.plotly_chart(fig_jit, use_container_width=True)

            # Updated ratings based on live forecast
            st.markdown("---")
            st.markdown("**Updated ratings — based on your actual Starlink connection**")

            RANK = {"red": 0, "amber": 1, "green": 2}
            for uc_name in adv["uses"]:
                uc = USE_CASES[uc_name]
                new_color = traffic_light(uc_name, fc_lat, fc_jit, fc_dl, fc_ul)
                old_color = adv["ratings"][uc_name]
                change = ""
                if new_color != old_color:
                    if RANK[new_color] > RANK[old_color]:
                        change = f"Improved from {old_color} (estimated) to {new_color} (measured)"
                    else:
                        change = f"Changed from {old_color} (estimated) to {new_color} (measured)"
                st.markdown(
                    rating_card(new_color, uc_name, uc["icon"], uc["desc"], change),
                    unsafe_allow_html=True
                )

            avg_lat = float(np.mean(lats))
            avg_dl  = float(np.mean(dls))
            st.markdown(f"""
            <div style="background:#0c1423;border:1px solid #1e3050;border-radius:12px;
                padding:1rem 1.4rem;margin-top:1rem;
                color:#94a3b8;font-size:0.83rem;line-height:1.7;">
                <span style="color:#60a5fa;font-weight:700;">What this means</span><br>
                Your {len(measurements)} tests show an average latency of
                <strong style="color:#e2e8f0;">{avg_lat:.1f} ms</strong> and
                average download of
                <strong style="color:#e2e8f0;">{avg_dl:.1f} Mbps</strong>.
                The forecast projects the trend from your test sequence.
                For a more reliable picture, add more measurements over 15 to 30 minutes,
                especially during the time of day you plan to use it most.
            </div>
            """, unsafe_allow_html=True)

        elif len(measurements) == 1:
            st.info("Add at least one more measurement to generate a forecast.")

    st.markdown("---")

    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("← Back to results"):
            st.session_state.step = 2
            st.rerun()
    with b2:
        if st.button("Start over"):
            for k in ["step", "advisory", "measurements"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("""
<div style="color:#334155;font-size:0.75rem;text-align:center;padding-bottom:1rem;">
    Based on Starlink data collected in Muscat, Oman · March 7–28, 2026 ·
    GUtech AI Thesis · Predictions are estimates, not guarantees
</div>
""", unsafe_allow_html=True)
