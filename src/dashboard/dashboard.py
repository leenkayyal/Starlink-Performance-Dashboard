import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
st.caption(f"Last refresh: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Starlink University Forecast Dashboard",
    layout="wide"
)
refresh_seconds = 60
st_autorefresh(interval=refresh_seconds * 1000, key="starlink_live_refresh")
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
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #10b981 !important;
    font-weight: 700 !important;
}

.stTabs {
    background: transparent !important;
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
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 8px 20px !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    flex: 1 !important;
    text-align: center !important;
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

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

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
    letter-spacing: 0.3px;
}

hr { border-color: #1e2d3d !important; }

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

[data-testid="stSidebar"] strong {
    color: #e2e8f0 !important;
}

div[data-baseweb="select"] > div {
    background: #0d1117 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.stSlider [data-testid="stMarkdownContainer"] p {
    color: #3b82f6 !important;
    font-weight: 800 !important;
}

p, li, span, div {
    color: #e2e8f0;
}

h1, h2, h3 {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1f3c 0%, #1a0533 50%, #0d1f3c 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 0 0 40px rgba(99,102,241,0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div>
        <div style="
            font-size: 2.2rem;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: -1px;
            line-height: 1.1;
            background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">Starlink Performance<br>Forecast System</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem; font-weight: 500;">
            University Network Intelligence Dashboard &nbsp;·&nbsp; Monitoring, Forecasting & Decision Support
        </div>
    </div>
    <div style="text-align:right;">
        <div style="
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
            margin-bottom: 0.4rem;
        ">GUtech AI Thesis Prototype</div>
        <div style="color: #64748b; font-size: 0.78rem;">Starlink · Muscat, Oman · 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================
# FILES
# ======================
CLEAN_FILE = "Cleaned/experiment_A/starlink_clean.csv"
FORECAST_FILE = "Cleaned/experiment_A/starlink_forecast.csv"


# ======================
# HELPERS
# ======================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_clean_data():
    df = pd.read_csv(CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Append newer rows from raw file for live display
    raw_file = "Raw/experiment_A/starlink_data.csv"
    if os.path.exists(raw_file):
        try:
            raw = pd.read_csv(raw_file)
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="mixed")
            raw = raw.rename(columns={
                "temperature_C": "temperature_c",
                "packet_loss_percent": "packet_loss_pct",
                "speedtest_server_name": "server_name",
                "speedtest_server_location": "isp",
                "ping_avg_rtt_ms": "ping_avg_rtt_ms",
                "ping_jitter_ms": "ping_jitter_ms"
            })
            latest_clean = df["timestamp"].max()
            new_rows = raw[raw["timestamp"] > latest_clean].copy()
            if len(new_rows) > 0:
                new_rows["was_estimated_row"] = False
                df = pd.concat([df, new_rows], ignore_index=True)
                df = df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass

    return df


def load_forecast_data():
    return pd.read_csv(FORECAST_FILE)


def build_metric_dataset(clean_df, target_col):
    df = clean_df.copy()

    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    candidate_cols = [
        "ping_avg_rtt_ms",
        "ping_jitter_ms",
        "download_mbps",
        "upload_mbps",
        "weather_code",
        "temperature_c",
        "humidity_percent",
        "wind_speed_mps",
        target_col
    ]
    for c in candidate_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ping_avg_rtt_ms" in df.columns:
        df = df[(df["ping_avg_rtt_ms"] > 0) & (df["ping_avg_rtt_ms"] < 500)]

    if "ping_jitter_ms" in df.columns:
        df = df[(df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100)]

    if "download_mbps" in df.columns:
        df = df[(df["download_mbps"] > 0) & (df["download_mbps"] < 1000)]

    if "upload_mbps" in df.columns:
        df = df[(df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)]

    if target_col in df.columns:
        if target_col == "upload_mbps":
            df = df[(df[target_col] > 0) & (df[target_col] < 500)]
        elif target_col == "download_mbps":
            df = df[(df[target_col] > 0) & (df[target_col] < 1000)]
        elif target_col == "ping_avg_rtt_ms":
            df = df[(df[target_col] > 0) & (df[target_col] < 500)]

    # time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # target lags
    for lag in range(1, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # context
    if "ping_avg_rtt_ms" in df.columns:
        df["latency_lag_1"] = df["ping_avg_rtt_ms"].shift(1)
    if "ping_jitter_ms" in df.columns:
        df["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)
    if "download_mbps" in df.columns:
        df["download_lag_1_ctx"] = df["download_mbps"].shift(1)
    if "upload_mbps" in df.columns:
        df["upload_lag_1_ctx"] = df["upload_mbps"].shift(1)

    # rolling
    df[f"{target_col}_roll_mean_3"] = df[target_col].rolling(3).mean()
    df[f"{target_col}_roll_std_3"] = df[target_col].rolling(3).std()

    # future target
    df["target"] = df[target_col].shift(-1)
    df = df.dropna().reset_index(drop=True)

    features = [
        "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
        f"{target_col}_lag_1", f"{target_col}_lag_2", f"{target_col}_lag_3", f"{target_col}_lag_4",
        f"{target_col}_roll_mean_3", f"{target_col}_roll_std_3",
        "latency_lag_1", "jitter_lag_1",
        "download_lag_1_ctx", "upload_lag_1_ctx",
        "weather_code", "temperature_c", "humidity_percent", "wind_speed_mps"
    ]
    features = [c for c in features if c in df.columns]

    return df[["timestamp"] + features + ["target"]].copy(), features


def prepare_xy(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.select_dtypes(include=["number"])
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_model(df, model_name):
    X, y = prepare_xy(df)

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if model_name == "Naive Baseline":
        lag_candidates = [c for c in X_test.columns if c.endswith("_lag_1")]
        y_pred = X_test[lag_candidates[0]].values
        model = None

    elif model_name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "XGBoost":
        if not XGB_AVAILABLE:
            raise ValueError("XGBoost is not installed.")
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    else:
        raise ValueError("Unknown model selected.")

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred)
    }


def fit_full_model(df, model_name):
    X, y = prepare_xy(df)

    if model_name == "Naive Baseline":
        return None, X

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        if not XGB_AVAILABLE:
            raise ValueError("XGBoost is not installed.")
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError("Unknown model selected.")

    model.fit(X, y)
    return model, X


def predict_next_step(df, model_name):
    X, y = prepare_xy(df)
    if len(X) == 0:
        return 0.0
    latest_row = X.iloc[[-1]]

    if model_name == "Naive Baseline":
        lag_candidates = [c for c in latest_row.columns if c.endswith("_lag_1")]
        return float(latest_row[lag_candidates[0]].iloc[0])

    model, _ = fit_full_model(df, model_name)
    return float(model.predict(latest_row)[0])


def compute_health(latency, jitter, download, upload):
    score = 100

    if latency > 80:
        score -= 30
    elif latency > 50:
        score -= 15

    if jitter > 15:
        score -= 25
    elif jitter > 8:
        score -= 10

    if download < 20:
        score -= 20
    elif download < 50:
        score -= 10

    if upload < 10:
        score -= 15
    elif upload < 20:
        score -= 8

    score = max(0, min(score, 100))

    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return score, label


def health_badge_html(score):
    if score >= 85:
        return '<span class="badge" style="background:#143d24;color:#7CFC9F;">Excellent</span>'
    elif score >= 70:
        return '<span class="badge" style="background:#3d3614;color:#ffe082;">Good</span>'
    elif score >= 50:
        return '<span class="badge" style="background:#4a2d12;color:#ffb74d;">Fair</span>'
    return '<span class="badge" style="background:#4a1717;color:#ff8a80;">Poor</span>'


def detect_outage(latency, download, upload):
    if latency > 100:
        return "High Latency Outage"
    if download < 5:
        return "Severe Download Drop"
    if upload < 2:
        return "Upload Failure Risk"
    return "No Outage Detected"


def generate_alerts(latency_forecast, jitter_now, download_forecast, upload_forecast):
    alerts = []

    if latency_forecast > 50:
        alerts.append(("warning", "Latency is forecast to increase in the next 15 minutes."))

    if jitter_now > 10:
        alerts.append(("warning", "Current jitter is high. Real-time applications may be affected."))

    if download_forecast < 50:
        alerts.append(("error", "Download speed forecast is low."))

    if upload_forecast < 10:
        alerts.append(("error", "Upload speed forecast is low."))

    if not alerts:
        alerts.append(("success", "No immediate network issue is forecast."))

    return alerts


def usage_recommendation(latency, jitter, download, upload):
    recommendations = []

    if latency < 40 and jitter < 8:
        recommendations.append("Suitable for video calls and online meetings.")
    else:
        recommendations.append("Use caution for video calls and real-time meetings.")

    if download > 100:
        recommendations.append("Suitable for streaming and large downloads.")
    elif download > 40:
        recommendations.append("General browsing and normal streaming should work well.")
    else:
        recommendations.append("Heavy streaming or large downloads may be affected.")

    if upload > 20:
        recommendations.append("Suitable for file uploads and cloud syncing.")
    elif upload > 10:
        recommendations.append("Normal uploads should work, but large uploads may be slower.")
    else:
        recommendations.append("Large uploads may be unreliable at this time.")

    return recommendations


def ai_insight(latency, jitter, download, upload):
    if latency < 35 and jitter < 8 and download > 100 and upload > 20:
        return "Starlink is currently in a strong operating state and appears suitable for most academic and administrative activities."
    if latency > 50 or jitter > 10:
        return "The network may support general use, but latency-sensitive activities such as live meetings may be affected."
    if download < 50 or upload < 10:
        return "Bandwidth-intensive activities may experience reduced quality. File transfers and streaming should be scheduled with caution."
    return "Current conditions appear acceptable for standard university use."


def make_time_chart(df, x_col, y_col, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], mode="lines", name=y_col,
        line=dict(color="#8b5cf6", width=2),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.1)"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=14)),
        xaxis_title="Time", yaxis_title=y_label, height=360,
        plot_bgcolor="rgba(15,12,41,0.8)",
        paper_bgcolor="rgba(15,12,41,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def make_actual_vs_predicted_chart(y_test, y_pred, points_to_show, title):
    actual = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)

    if points_to_show < len(actual):
        actual = actual.iloc[-points_to_show:]
        predicted = predicted.iloc[-points_to_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual",
        line=dict(color="#06b6d4", width=2)))
    fig.add_trace(go.Scatter(y=predicted, mode="lines", name="Predicted",
        line=dict(color="#f472b6", width=2, dash="dot")))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=14)),
        xaxis_title="Test Time Steps", yaxis_title="Latency (ms)", height=420,
        plot_bgcolor="rgba(15,12,41,0.8)",
        paper_bgcolor="rgba(15,12,41,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff")),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def make_health_gauge(score):
    if score >= 85:
        bar_color = "#4CAF50"
    elif score >= 70:
        bar_color = "#8BC34A"
    elif score >= 50:
        bar_color = "#FF9800"
    else:
        bar_color = "#F44336"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 28, "color": bar_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
            "bar": {"color": bar_color, "thickness": 0.4},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(244,67,54,0.15)"},
                {"range": [50, 70], "color": "rgba(255,152,0,0.15)"},
                {"range": [70, 85], "color": "rgba(139,195,74,0.15)"},
                {"range": [85, 100], "color": "rgba(76,175,80,0.15)"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.8,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"}
    )
    return fig


def make_model_comparison_table(df):
    model_names = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        model_names.append("XGBoost")

    rows = []
    for m in model_names:
        result = train_model(df, m)
        rows.append({
            "Model": m,
            "MAE": round(result["mae"], 3),
            "RMSE": round(result["rmse"], 3)
        })
    return pd.DataFrame(rows)


def fmt_temp(value):
    try:
        return f"{float(value):.1f} °C" if pd.notna(value) else "Not available"
    except (ValueError, TypeError):
        return "Not available"


def fmt_humidity(value):
    return f"{value:.0f}%" if pd.notna(value) else "Not available"


def fmt_wind(value):
    return f"{value:.2f} m/s" if pd.notna(value) else "Not available"


def fmt_code(value):
    weather_descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Cloudy",
        45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Heavy drizzle", 61: "Slight rain", 63: "Rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
        81: "Heavy showers", 82: "Violent showers", 95: "Thunderstorm",
        96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail"
    }
    try:
        code = int(float(value))
        return weather_descriptions.get(code, f"Code {code}")
    except (ValueError, TypeError):
        return "Not available"


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("Dashboard Controls")

    model_options = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        model_options.append("XGBoost")

    selected_model = st.selectbox("Forecast model", model_options, index=1)

    chart_points = st.slider("Prediction chart points", 50, 400, 200, 25)

    history_window = st.selectbox("Historical view", ["Last 24 hours", "Last 7 days", "Full data"], index=2)

    st.markdown("---")
    st.markdown("**Deployment mode:** Starlink operational forecasting")
    st.markdown("**Primary model:** Linear Regression")


# ======================
# LOAD DATA
# ======================
clean_df = load_clean_data()
latency_df = load_forecast_data()
download_df, _ = build_metric_dataset(clean_df, "download_mbps")
upload_df, _ = build_metric_dataset(clean_df, "upload_mbps")

if history_window == "Last 24 hours":
    hist_df = clean_df[clean_df["timestamp"] >= clean_df["timestamp"].max() - pd.Timedelta(hours=24)].copy()
elif history_window == "Last 7 days":
    hist_df = clean_df[clean_df["timestamp"] >= clean_df["timestamp"].max() - pd.Timedelta(days=7)].copy()
else:
    hist_df = clean_df.copy()

latency_result = train_model(latency_df, selected_model)

latency_forecast = predict_next_step(latency_df, selected_model)
download_forecast = predict_next_step(download_df, "Linear Regression")
upload_forecast = predict_next_step(upload_df, "Linear Regression")

# Fall back to last known real value if forecast is 0 or invalid
if download_forecast <= 0:
    download_forecast = float(clean_df["download_mbps"].dropna().iloc[-1]) if "download_mbps" in clean_df.columns else 0.0
if upload_forecast <= 0:
    upload_forecast = float(clean_df["upload_mbps"].dropna().iloc[-1]) if "upload_mbps" in clean_df.columns else 0.0

latest_time = clean_df["timestamp"].max()
forecast_time = latest_time + pd.Timedelta(minutes=15)

latest_row = clean_df.iloc[-1]
current_latency = float(latest_row["ping_avg_rtt_ms"])
current_jitter = float(latest_row["ping_jitter_ms"])
current_download = float(latest_row["download_mbps"])
current_upload = float(latest_row["upload_mbps"])

latest_weather_code = latest_row["weather_code"] if "weather_code" in latest_row.index else np.nan
latest_temp = latest_row["temperature_c"] if "temperature_c" in latest_row.index else np.nan
latest_humidity = latest_row["humidity_percent"] if "humidity_percent" in latest_row.index else np.nan
latest_wind = latest_row["wind_speed_mps"] if "wind_speed_mps" in latest_row.index else np.nan

health_score, health_label = compute_health(
    latency_forecast,
    current_jitter,
    download_forecast,
    upload_forecast
)

alerts = generate_alerts(
    latency_forecast,
    current_jitter,
    download_forecast,
    upload_forecast
)

recommendations = usage_recommendation(
    latency_forecast,
    current_jitter,
    download_forecast,
    upload_forecast
)

system_insight = ai_insight(
    latency_forecast,
    current_jitter,
    download_forecast,
    upload_forecast
)

outage_status = detect_outage(
    latency_forecast,
    download_forecast,
    upload_forecast
)

# ======================
# TOP KPI CARDS
# ======================
def kpi_card(label, value, unit, color, icon):
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border: 1px solid {color}44;
        border-top: 3px solid {color};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 4px 20px {color}22;
        height: 100%;
    ">
        <div style="color: {color}; font-size: 1.4rem; margin-bottom: 0.3rem;">{icon}</div>
        <div style="color: #94a3b8; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
        <div style="color: #ffffff; font-size: 1.9rem; font-weight: 900; line-height: 1.1; margin-top: 0.2rem;">{value} <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 500;">{unit}</span></div>
    </div>"""

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(kpi_card("Latency", f"{current_latency:.1f}", "ms", "#60a5fa", "📡"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("Jitter", f"{current_jitter:.1f}", "ms", "#f472b6", "〰️"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Download", f"{current_download:.1f}", "Mbps", "#34d399", "⬇"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Upload", f"{current_upload:.1f}", "Mbps", "#fb923c", "⬆"), unsafe_allow_html=True)
with c5:
    hcolor = "#34d399" if health_score >= 85 else "#fbbf24" if health_score >= 70 else "#fb923c" if health_score >= 50 else "#f87171"
    st.markdown(kpi_card("Health Score", f"{health_score}", "/100", hcolor, "💚"), unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="margin-top:0.6rem; margin-bottom:1rem;">
        {health_badge_html(health_score)}
        &nbsp;&nbsp;
        <span class="small-note"><b>Network status:</b> {outage_status}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Historical Trends",
    "Forecasting",
    "Alerts & Recommendations",
    "Model Evaluation"
])

with tab1:
    st.markdown('<div class="section-title">Network Overview</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.plotly_chart(
            make_time_chart(hist_df, "timestamp", "ping_avg_rtt_ms", "Latency Over Time", "Latency (ms)"),
            use_container_width=True
        )

    with right:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Current Forecast")
        st.metric("Next 15-min Latency", f"{latency_forecast:.2f} ms")
        st.metric("Next 15-min Download", f"{download_forecast:.2f} Mbps")
        st.metric("Next 15-min Upload", f"{upload_forecast:.2f} Mbps")
        st.write(f"Forecast time: **{forecast_time}**")
        st.write(f"Selected model: **{selected_model}**")
        st.markdown('</div>', unsafe_allow_html=True)

    a, b, c = st.columns([1, 1.15, 1])

    with a:
        st.plotly_chart(make_health_gauge(health_score), use_container_width=True)

    with b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Recommended Usage")
        for rec in recommendations:
            st.write(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Latest Weather Context")
        st.write(f"Weather code: **{fmt_code(latest_weather_code)}**")
        st.write(f"Temperature: **{fmt_temp(latest_temp)}**")
        st.write(f"Humidity: **{fmt_humidity(latest_humidity)}**")
        st.write(f"Wind speed: **{fmt_wind(latest_wind)}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### AI Insight")
    st.write(system_insight)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Historical Performance Trends</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            make_time_chart(hist_df, "timestamp", "ping_jitter_ms", "Jitter Over Time", "Jitter (ms)"),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            make_time_chart(hist_df, "timestamp", "download_mbps", "Download Speed Over Time", "Download (Mbps)"),
            use_container_width=True
        )

    st.plotly_chart(
        make_time_chart(hist_df, "timestamp", "upload_mbps", "Upload Speed Over Time", "Upload (Mbps)"),
        use_container_width=True
    )

with tab3:
    st.markdown('<div class="section-title">Forecasting</div>', unsafe_allow_html=True)

    st.plotly_chart(
        make_actual_vs_predicted_chart(
            latency_result["y_test"],
            latency_result["y_pred"],
            chart_points,
            f"Actual vs Predicted Latency ({selected_model})"
        ),
        use_container_width=True
    )

    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast Latency", f"{latency_forecast:.2f} ms")
    f2.metric("Forecast Download", f"{download_forecast:.2f} Mbps")
    f3.metric("Forecast Upload", f"{upload_forecast:.2f} Mbps")

    st.markdown(
        '<div class="small-note">Latency is the main evaluated forecasting target. '
        'Download and upload are included as supporting operational forecasts.</div>',
        unsafe_allow_html=True
    )

with tab4:
    st.markdown('<div class="section-title">Alerts & Recommendations</div>', unsafe_allow_html=True)

    for level, message in alerts:
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.error(message)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Operational Interpretation")
        st.write(
            f"The current forecast-based health status is **{health_label}** with a score of **{health_score}/100**. "
            "This score combines predicted latency, forecast bandwidth, and current jitter into a single operational indicator."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Suggested Use Cases Right Now")
        for rec in recommendations:
            st.write(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)

    model_table = make_model_comparison_table(latency_df)
    st.dataframe(model_table, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Selected Model MAE", f"{latency_result['mae']:.3f}")
    m2.metric("Selected Model RMSE", f"{latency_result['rmse']:.3f}")

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### Notes")
    st.write(
        "The naive baseline is used as a benchmark. Linear Regression is the main forecasting model because it "
        "provided the best balance of accuracy, stability, and interpretability. More complex models were kept as advanced comparisons."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Export Results")
    model_csv = model_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Model Comparison CSV",
        data=model_csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    pred_df = pd.DataFrame({
        "actual": latency_result["y_test"].values,
        "predicted": latency_result["y_pred"]
    })
    pred_csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Latency Predictions CSV",
        data=pred_csv,
        file_name="latency_predictions.csv",
        mime="text/csv"
    )