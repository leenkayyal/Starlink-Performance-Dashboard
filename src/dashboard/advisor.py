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
    page_title="Starlink AI Advisor",
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp, .main, .block-container {
    background-color: #05080f !important;
    color: #dde3ed !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container {
    padding-top: 3rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }

h1, h2, h3 { color: #ffffff !important; }
p, li, span { color: #dde3ed; }
hr { border-color: #1e3050 !important; }

div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stNumberInput"] input {
    background: #0c1423 !important;
    border: 1px solid #1e3050 !important;
    border-radius: 10px !important;
    color: #dde3ed !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.65rem 2rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #6366f1) !important;
    box-shadow: 0 4px 22px rgba(79,70,229,0.45) !important;
}
.step-label {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4a7fc1;
    margin-bottom: 0.6rem;
}
.q-label {
    font-size: 0.78rem;
    font-weight: 800;
    color: #4a7fc1;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.25rem;
    margin-top: 1.2rem;
}
.advisor-card {
    background: linear-gradient(135deg, #0c1423 0%, #080e1c 100%);
    border: 1px solid #1e3050;
    border-radius: 16px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
}
.small-note {
    color:#64748b;
    font-size:0.82rem;
    line-height:1.65;
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
        return "High", "#10b981"

    if live_count >= 2 and latency_uncertainty <= 10 and condition_penalty <= 1:
        return "Medium", "#f59e0b"

    if latency_uncertainty <= 8 and condition_penalty == 0:
        return "Medium", "#f59e0b"

    return "Low", "#ef4444"

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
    uncertainty = f"<div style='color:#64748b;font-size:0.72rem;margin-top:0.2rem;'>± {interval:.1f} {unit}</div>" if interval is not None and interval > 0 else ""
    return f"""
    <div style="background:{color}11;border:1px solid {color}33;border-top:3px solid {color};border-radius:12px;padding:0.9rem 1rem;text-align:center;">
        <div style="color:#64748b;font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:1px;">{label}</div>
        <div style="color:#fff;font-size:1.6rem;font-weight:900;line-height:1.2;margin-top:0.2rem;">
            {value}<span style="font-size:0.8rem;color:#94a3b8;font-weight:400;"> {unit}</span>
        </div>
        {uncertainty}
    </div>"""

def rating_card(color, uc_name, icon, desc, change_note=""):
    bg = {"green":"#031a0e", "amber":"#171203", "red":"#1a0303"}[color]
    border = {"green":"#10b981", "amber":"#f59e0b", "red":"#ef4444"}[color]
    label = {"green":"Suitable", "amber":"Limited", "red":"Not Recommended"}[color]
    change_html = f'<div style="color:#94a3b8;font-size:0.75rem;margin-top:0.2rem;">{change_note}</div>' if change_note else ""
    return (
        f'<div style="background:{bg};border:1px solid {border}44;border-left:5px solid {border};border-radius:12px;padding:1rem 1.2rem;display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">'
        f'<div style="font-size:1.6rem;">{icon}</div>'
        f'<div style="flex:1;"><div style="font-weight:700;color:#e2e8f0;">{uc_name}</div><div style="color:#64748b;font-size:0.8rem;margin-top:0.1rem;">{desc}</div>{change_html}</div>'
        f'<div style="background:{border}22;border:1px solid {border}66;border-radius:999px;padding:0.3rem 0.9rem;color:{border};font-weight:800;font-size:0.82rem;white-space:nowrap;">{label}</div>'
        f'</div>'
    )

def trend_chart(values, label, color):
    series = pd.Series(values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(series) + 1)), y=series.values, mode="lines+markers", name="Measured", line=dict(color=color, width=2), marker=dict(size=6)))
    fig.update_layout(
        title=dict(text=label, font=dict(color="#ffffff", size=13)),
        height=210,
        plot_bgcolor="rgba(12,20,35,0.9)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#64748b", title="Test #"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#64748b"),
        margin=dict(l=10, r=10, t=38, b=10),
    )
    return fig

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
<div style="background:linear-gradient(135deg,#0d1f3c 0%,#0a0d1f 50%,#0d1f3c 100%);border-radius:16px;padding:2.4rem 3rem;margin-bottom:1.6rem;border:1px solid rgba(99,102,241,0.5);box-shadow:0 0 40px rgba(99,102,241,0.1);display:flex;justify-content:space-between;align-items:center;">
    <div>
        <div style="font-size:0.7rem;font-weight:800;text-transform:uppercase;letter-spacing:2px;color:#4a7fc1;margin-bottom:0.5rem;">Starlink AI Suitability Advisor · Oman</div>
        <div style="font-size:2.25rem;font-weight:900;letter-spacing:-1px;line-height:1.1;background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Will Starlink work for you?</div>
        <div style="color:#64748b;font-size:0.9rem;margin-top:0.5rem;">AI-enhanced recommendation using thesis data, multi-KPI forecasting, uncertainty, and live measurements.</div>
    </div>
    <div style="text-align:right;">
        <div style="background:linear-gradient(135deg,#2563eb,#7c3aed);padding:0.4rem 1rem;border-radius:999px;font-size:0.75rem;font-weight:800;color:#ffffff;margin-bottom:0.4rem;">GUtech AI Thesis Prototype</div>
        <div style="color:#64748b;font-size:0.78rem;">Starlink · Muscat, Oman · 2026</div>
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
    bg = "linear-gradient(135deg,#1d4ed8,#4f46e5)" if active else "#052e16" if done else "#0c1423"
    tc = "#ffffff" if active else "#10b981" if done else "#334155"
    border = "none" if active else "1px solid #10b98144" if done else "1px solid #1e3050"
    label = name if not done else "✓ " + name.split(". ", 1)[1]
    col.markdown(f'<div style="background:{bg};border:{border};border-radius:10px;padding:0.65rem 1rem;text-align:center;font-size:0.82rem;font-weight:800;color:{tc};">{label}</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

# ============================================================
# STEP 1
# ============================================================
if st.session_state.step == 1:
    st.markdown("### Step 1, Your conditions")
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
        uses = st.multiselect("uses", list(USE_CASES.keys()), default=["Video calls / online meetings", "General browsing / social media"], label_visibility="collapsed", key="q_uses")

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    if st.button("Get AI Advisory Result →"):
        if not uses:
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
        <div class="advisor-card" style="border-left:4px solid #2563eb;">
            <strong style="color:#60a5fa;">AI forecasting active</strong><br>
            Latency is forecast using the validated combined-dataset model (MAE 4.3 ms, RMSE 6.5 ms).
            Jitter, download, and upload are forecast using models trained on both experiment datasets.
            <br><br>
            <strong style="color:{adv['confidence_color']};">Forecast confidence: {adv['confidence']}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="advisor-card" style="border-left:4px solid #ef4444;">
            <strong style="color:#ef4444;">AI models not available</strong><br>
            The advisor is using rule-based fallback estimates only. Check that the cleaned datasets exist.
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Latency", f"{perf['latency']:.1f}", "ms", "#60a5fa", intervals.get("latency")), unsafe_allow_html=True)
    c2.markdown(metric_card("Jitter", f"{perf['jitter']:.1f}", "ms", "#f472b6", intervals.get("jitter")), unsafe_allow_html=True)
    c3.markdown(metric_card("Download", f"{perf['download']:.1f}", "Mbps", "#34d399", intervals.get("download")), unsafe_allow_html=True)
    c4.markdown(metric_card("Upload", f"{perf['upload']:.1f}", "Mbps", "#fb923c", intervals.get("upload")), unsafe_allow_html=True)

    st.markdown("### Use case ratings")
    for uc_name, color in adv["ratings"].items():
        uc = USE_CASES[uc_name]
        st.markdown(rating_card(color, uc_name, uc["icon"], uc["desc"]), unsafe_allow_html=True)

    st.markdown("### Explainability")
    imp = top_feature_importance(models.get("latency"), features_map.get("latency", []), limit=6) if adv["models_loaded"] else []
    if imp:
        imp_df = pd.DataFrame(imp, columns=["Feature", "Importance"])
        imp_df["Importance"] = imp_df["Importance"].round(4)
        st.dataframe(imp_df, use_container_width=True)
        st.caption("These are the strongest model inputs for the latency prediction. This can be discussed as model interpretability in the thesis.")
    else:
        st.caption("Feature importance is unavailable because the ML model did not load or does not expose importance values.")

    st.markdown("### Model validation summary")
    if adv["model_metrics"]:
        rows = []
        for k, m in adv["model_metrics"].items():
            rows.append({"Target": k, "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3), "Residual Std": round(m["residual_std"], 3), "Test Rows": m["n_test"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
        st.dataframe(mdf.rename(columns={"latency":"Latency (ms)", "download":"Download (Mbps)", "upload":"Upload (Mbps)", "jitter":"Jitter (ms)"}), use_container_width=True)

        perf_rule = adv["rule_perf"]
        live_pred, live_intervals = predict_all_kpis(perf_rule, adv["time"], adv["weather"], adv["temp"], models, features_map, model_metrics, live_measurements=measurements)
        conf, conf_color = confidence_level(live_intervals, live_count=len(measurements), weather=adv["weather"], location=adv["location"])

        st.markdown("### AI forecast from your real measurements")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Next Latency", f"{live_pred['latency']:.1f}", "ms", "#60a5fa", live_intervals.get("latency")), unsafe_allow_html=True)
        c2.markdown(metric_card("Next Jitter", f"{live_pred['jitter']:.1f}", "ms", "#f472b6", live_intervals.get("jitter")), unsafe_allow_html=True)
        c3.markdown(metric_card("Next Download", f"{live_pred['download']:.1f}", "Mbps", "#34d399", live_intervals.get("download")), unsafe_allow_html=True)
        c4.markdown(metric_card("Next Upload", f"{live_pred['upload']:.1f}", "Mbps", "#fb923c", live_intervals.get("upload")), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advisor-card" style="border-left:4px solid {conf_color};">
            <strong style="color:{conf_color};">Live forecast confidence: {conf}</strong><br>
            Confidence improves as more measurements are added because the advisor can build real lag features instead of relying on synthetic lags.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(trend_chart([m["latency"] for m in measurements], "Measured Latency", "#60a5fa"), use_container_width=True)
        with col2:
            st.plotly_chart(trend_chart([m["download"] for m in measurements], "Measured Download", "#34d399"), use_container_width=True)

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
<div style="color:#334155;font-size:0.75rem;text-align:center;padding-bottom:1rem;">
    Based on Starlink data collected in Muscat, Oman · GUtech AI Thesis · Predictions are estimates, not guarantees
</div>
""", unsafe_allow_html=True)