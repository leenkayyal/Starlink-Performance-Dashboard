import warnings
warnings.filterwarnings("ignore")

import time
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

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Starlink Live Forecast Dashboard",
    layout="wide"
)

# ======================
# STYLE
# ======================
st.markdown("""
<style>
.main-title {
    font-size: 2.15rem;
    font-weight: 760;
    margin-bottom: 0.15rem;
    line-height: 1.1;
    white-space: nowrap;
}
.sub-title {
    font-size: 1rem;
    color: #a0a8b5;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0.8rem;
}
.info-box {
    padding: 1rem;
    border-radius: 0.9rem;
    background-color: rgba(49, 51, 63, 0.28);
    border: 1px solid rgba(250,250,250,0.08);
}
.small-note {
    color: #9aa4b2;
    font-size: 0.94rem;
}
.badge {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
}
.notice-box {
    padding: 0.85rem 1rem;
    border-radius: 0.8rem;
    background-color: rgba(25, 118, 210, 0.12);
    border: 1px solid rgba(144, 202, 249, 0.28);
    margin-top: 0.4rem;
    margin-bottom: 1rem;
}
.timeline-box {
    padding: 0.9rem 1rem;
    border-radius: 0.8rem;
    background-color: rgba(49, 51, 63, 0.22);
    border: 1px solid rgba(250,250,250,0.08);
}
.timeline-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
.timeline-chip {
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 600;
}
.timeline-recorded {
    background: rgba(76, 175, 80, 0.16);
    color: #8be28f;
}
.timeline-forecast {
    background: rgba(255, 193, 7, 0.16);
    color: #ffd54f;
}
.arrow {
    color: #9aa4b2;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:20px;">
    <div>
        <div class="main-title">Starlink Live Forecast Dashboard</div>
        <div class="sub-title">University network monitoring, forecasting, and decision support prototype</div>
    </div>
    <div style="text-align:right;">
        <div style="font-weight:700;">GUtech AI Thesis Prototype</div>
        <div style="color:#9aa4b2;">Live-ready Starlink monitoring interface</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================
# FILE PATHS
# ======================
CLEAN_FILE = "Cleaned/starlink_clean_FIXED.csv"
FORECAST_FILE = "Cleaned/starlink_forecast_v2.csv"


# ======================
# HELPERS
# ======================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_clean_data():
    df = pd.read_csv(CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
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

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    for lag in range(1, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    if "ping_avg_rtt_ms" in df.columns:
        df["latency_lag_1"] = df["ping_avg_rtt_ms"].shift(1)
    if "ping_jitter_ms" in df.columns:
        df["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)
    if "download_mbps" in df.columns:
        df["download_lag_1_ctx"] = df["download_mbps"].shift(1)
    if "upload_mbps" in df.columns:
        df["upload_lag_1_ctx"] = df["upload_mbps"].shift(1)

    df[f"{target_col}_roll_mean_3"] = df[target_col].rolling(3).mean()
    df[f"{target_col}_roll_std_3"] = df[target_col].rolling(3).std()

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
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_label, height=360)
    return fig


def make_actual_vs_predicted_chart(y_test, y_pred, points_to_show, title):
    actual = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)

    if points_to_show < len(actual):
        actual = actual.iloc[-points_to_show:]
        predicted = predicted.iloc[-points_to_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(y=predicted, mode="lines", name="Predicted"))
    fig.update_layout(title=title, xaxis_title="Test Time Steps", yaxis_title="Latency (ms)", height=420)
    return fig


def make_health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, 50], "color": "#8b1e1e"},
                {"range": [50, 70], "color": "#a36a00"},
                {"range": [70, 85], "color": "#2c6e49"},
                {"range": [85, 100], "color": "#1b5e20"},
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=15, r=15, t=25, b=15))
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
    return f"{value:.1f} °C" if pd.notna(value) else "Not available"


def fmt_humidity(value):
    return f"{value:.0f}%" if pd.notna(value) else "Not available"


def fmt_wind(value):
    return f"{value:.2f} m/s" if pd.notna(value) else "Not available"


def fmt_code(value):
    return f"{value}" if pd.notna(value) else "Not available"


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
    history_window = st.selectbox("Historical view", ["Last 24 hours", "Last 7 days", "Full data"], index=1)

    refresh_seconds = st.selectbox("Refresh interval", [15, 30, 60, 120], index=2)

    refresh_now = st.button("Refresh now")

    st.markdown("---")
    st.markdown("**Data source:** Recorded Starlink dataset")
    st.markdown("**Forecast dataset:** v2")
    st.markdown("**Primary model:** Linear Regression")

# trigger manual rerun
if refresh_now:
    st.rerun()

# optional auto refresh
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds * 1000, key="live_refresh")
else:
    st.info("Optional: install `streamlit-autorefresh` for automatic live refreshing.")


# ======================
# LOAD DATA FRESH ON EVERY RERUN
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
# LIVE NOTICE
# ======================
st.markdown(f"""
<div class="notice-box">
<b>Latest data timestamp:</b> {latest_time}
<br>
<b>Forecast horizon:</b> Next 15 minutes ({forecast_time})
<br>
<b>Refresh mode:</b> Every {refresh_seconds} seconds
<br>
<b>Note:</b> This dashboard reads the latest available Starlink CSV data and updates forecasts whenever new rows are added by the logging pipeline.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="timeline-box">
    <div class="timeline-row">
        <span class="timeline-chip timeline-recorded">Latest recorded data</span>
        <span>{latest_time}</span>
        <span class="arrow">→</span>
        <span class="timeline-chip timeline-forecast">Forecasted next step</span>
        <span>{forecast_time}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================
# TOP KPI CARDS
# ======================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Current Latency", f"{current_latency:.2f} ms")
c2.metric("Current Jitter", f"{current_jitter:.2f} ms")
c3.metric("Current Download", f"{current_download:.2f} Mbps")
c4.metric("Current Upload", f"{current_upload:.2f} Mbps")
c5.metric("Health Score", f"{health_score}/100", delta=health_label)
c6.metric("Rows in Dataset", f"{len(clean_df)}")

st.markdown(
    f"""
    <div style="margin-top:0.6rem; margin-bottom:1rem;">
        {health_badge_html(health_score)}
        &nbsp;&nbsp;
        <span class="small-note"><b>Network status:</b> {outage_status}</span>
        &nbsp;&nbsp;
        <span class="small-note"><b>Last refresh:</b> {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Historical Trends",
    "Forecasting",
    "Alerts & Recommendations",
    "Model Evaluation",
    "Data Quality"
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

with tab6:
    st.markdown('<div class="section-title">Data Quality</div>', unsafe_allow_html=True)

    dq1, dq2, dq3 = st.columns(3)

    estimated_count = int(clean_df["was_estimated_row"].sum()) if "was_estimated_row" in clean_df.columns else 0
    missing_timestamp_count = int(clean_df["was_missing_timestamp_row"].sum()) if "was_missing_timestamp_row" in clean_df.columns else 0
    empty_measurement_count = int(clean_df["had_empty_measurement_in_raw"].sum()) if "had_empty_measurement_in_raw" in clean_df.columns else 0

    dq1.metric("Estimated Rows", estimated_count)
    dq2.metric("Missing Timestamp Rows", missing_timestamp_count)
    dq3.metric("Empty Raw Measurements", empty_measurement_count)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write(
        "This section summarizes data quality indicators carried forward from the cleaning stage. "
        "It helps distinguish between real recorded measurements and rows affected by estimation, missing timestamps, or empty raw measurements."
    )
    st.markdown('</div>', unsafe_allow_html=True)