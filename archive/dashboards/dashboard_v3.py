import warnings
warnings.filterwarnings("ignore")

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


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Starlink University Forecast Dashboard",
    layout="wide"
)

# ======================
# CUSTOM STYLE
# ======================
st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #9aa4b2;
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}
.info-box {
    padding: 1rem;
    border-radius: 0.8rem;
    background-color: rgba(49, 51, 63, 0.35);
    border: 1px solid rgba(250,250,250,0.08);
}
.small-note {
    color: #9aa4b2;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Starlink University Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Monitoring, forecasting, health scoring, and recommendation tool for practical university use</div>',
    unsafe_allow_html=True
)

# ======================
# FILES
# ======================
CLEAN_FILE = "Cleaned/starlink_clean.csv"
FORECAST_V1 = "Cleaned/starlink_forecast.csv"
FORECAST_V2 = "Cleaned/starlink_forecast_v2.csv"


# ======================
# HELPERS
# ======================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@st.cache_data
def load_clean_data():
    df = pd.read_csv(CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_forecast_data(version):
    path = FORECAST_V2 if version == "v2" else FORECAST_V1
    return pd.read_csv(path)


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
        "weather_code"
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


def make_time_chart(df, x_col, y_col, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines", name=y_col))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_label, height=380)
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
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
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


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("Dashboard Controls")

    dataset_version = st.selectbox("Forecast dataset version", ["v1", "v2"], index=1)

    model_options = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        model_options.append("XGBoost")

    selected_model = st.selectbox("Forecast model", model_options, index=1)

    chart_points = st.slider("Prediction chart points", 50, 400, 200, 25)

    history_window = st.selectbox("Historical view", ["Last 24 hours", "Last 7 days", "Full data"], index=1)


# ======================
# LOAD DATA
# ======================
clean_df = load_clean_data()
latency_df = load_forecast_data(dataset_version)
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

# ======================
# TOP CARDS
# ======================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Latency", f"{current_latency:.2f} ms")
c2.metric("Current Jitter", f"{current_jitter:.2f} ms")
c3.metric("Current Download", f"{current_download:.2f} Mbps")
c4.metric("Current Upload", f"{current_upload:.2f} Mbps")
c5.metric("Health Score", f"{health_score}/100", delta=health_label)

st.markdown("---")

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Historical Trends",
    "Forecasting",
    "Alerts",
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
        st.write(f"Dataset version: **{dataset_version}**")
        st.markdown('</div>', unsafe_allow_html=True)

    a, b = st.columns([1, 1.2])

    with a:
        st.plotly_chart(make_health_gauge(health_score), use_container_width=True)

    with b:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Recommended Usage")
        for rec in recommendations:
            st.write(f"- {rec}")
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

    st.markdown('<div class="small-note">Download and upload forecasts are provided using lightweight supporting models, while latency remains the main evaluated forecasting target.</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-title">Operational Alerts</div>', unsafe_allow_html=True)

    for level, message in alerts:
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.error(message)

    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### Health Interpretation")
    st.write(
        f"The forecast-based health status is **{health_label}** with a score of **{health_score}/100**. "
        "This score combines predicted latency, forecast bandwidth, and current jitter into a single operational indicator."
    )
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
        "The naive baseline is used as a benchmark. Linear Regression is the main forecasting model "
        "because it provided the best balance of accuracy, stability, and interpretability. "
        "More complex models were included as advanced comparisons."
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.write("Dashboard V3 focuses on Starlink performance monitoring and forecasting for practical university use.")