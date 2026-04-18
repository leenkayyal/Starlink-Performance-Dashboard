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
    page_title="Starlink Performance Forecast Dashboard",
    layout="wide"
)

st.title("Starlink Performance Forecast Dashboard")
st.caption("Monitoring, analysis, and short-term forecasting tool for university use")


# ======================
# FILE PATHS
# ======================
CLEAN_FILE = "Cleaned/starlink_clean.csv"
FORECAST_V1 = "Cleaned/starlink_forecast.csv"
FORECAST_V2 = "Cleaned/starlink_forecast_v2.csv"

TARGET = "ping_avg_rtt_ms"


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
    if version == "v1":
        path = FORECAST_V1
    else:
        path = FORECAST_V2

    df = pd.read_csv(path)
    return df


def train_model(df, model_name):
    X = df.drop(columns=["target"])
    y = df["target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if model_name == "Naive Baseline":
        model = None
        y_pred = X_test[f"{TARGET}_lag_1"].values

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

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": rmse(y_test, y_pred)
    }

    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred
    })

    return model, X_train, X_test, y_train, y_test, y_pred, metrics, results_df


def fit_full_model(df, model_name):
    X = df.drop(columns=["target"])
    y = df["target"]

    if model_name == "Naive Baseline":
        return None

    elif model_name == "Linear Regression":
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

    model.fit(X, y)
    return model


def predict_next_step(df_forecast, model_name):
    X = df_forecast.drop(columns=["target"])

    latest_features = X.iloc[[-1]]

    if model_name == "Naive Baseline":
        pred = float(latest_features[f"{TARGET}_lag_1"].iloc[0])
    else:
        model = fit_full_model(df_forecast, model_name)
        pred = float(model.predict(latest_features)[0])

    return pred


def health_status(latency, jitter, download, upload):
    score = 100.0

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

    score = max(0, min(100, score))

    if score >= 85:
        label = "Excellent"
    elif score >= 70:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Poor"

    return round(score, 1), label


def make_line_chart(df, x_col, y_col, title, y_label, color_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines",
        name=color_name
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        height=400
    )
    return fig


def make_actual_vs_predicted_chart(y_test, y_pred, points_to_show, model_name):
    actual = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)

    if points_to_show < len(actual):
        actual = actual.iloc[-points_to_show:]
        predicted = predicted.iloc[-points_to_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=actual,
        mode="lines",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        y=predicted,
        mode="lines",
        name=f"Predicted ({model_name})"
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Latency ({model_name})",
        xaxis_title="Test Time Steps",
        yaxis_title="Latency (ms)",
        height=450
    )
    return fig


def make_model_comparison_table(version):
    df = load_forecast_data(version)
    models = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        models.append("XGBoost")

    rows = []
    for model_name in models:
        _, _, _, _, _, _, metrics, _ = train_model(df, model_name)
        rows.append({
            "Model": model_name,
            "MAE": round(metrics["mae"], 3),
            "RMSE": round(metrics["rmse"], 3)
        })

    return pd.DataFrame(rows)


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("Dashboard Controls")

    dataset_version = st.selectbox(
        "Forecast dataset version",
        ["v1", "v2"],
        index=1
    )

    model_choices = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        model_choices.append("XGBoost")

    selected_model = st.selectbox(
        "Forecast model",
        model_choices,
        index=1
    )

    chart_points = st.slider(
        "Prediction chart points",
        min_value=50,
        max_value=400,
        value=200,
        step=25
    )

    history_window = st.selectbox(
        "Historical view",
        ["Last 24 hours", "Last 7 days", "Full data"],
        index=1
    )


# ======================
# LOAD DATA
# ======================
clean_df = load_clean_data()
forecast_df = load_forecast_data(dataset_version)

# History filter
if history_window == "Last 24 hours":
    filtered_clean = clean_df[clean_df["timestamp"] >= clean_df["timestamp"].max() - pd.Timedelta(hours=24)].copy()
elif history_window == "Last 7 days":
    filtered_clean = clean_df[clean_df["timestamp"] >= clean_df["timestamp"].max() - pd.Timedelta(days=7)].copy()
else:
    filtered_clean = clean_df.copy()

# Train selected model
_, _, _, _, y_test, y_pred, metrics, pred_df = train_model(forecast_df, selected_model)

# Latest values
latest_row = clean_df.iloc[-1]
latest_latency = float(latest_row["ping_avg_rtt_ms"]) if "ping_avg_rtt_ms" in latest_row else np.nan
latest_jitter = float(latest_row["ping_jitter_ms"]) if "ping_jitter_ms" in latest_row else np.nan
latest_download = float(latest_row["download_mbps"]) if "download_mbps" in latest_row else np.nan
latest_upload = float(latest_row["upload_mbps"]) if "upload_mbps" in latest_row else np.nan

# Forecast
next_latency_forecast = predict_next_step(forecast_df, selected_model)
forecast_time = clean_df["timestamp"].max() + pd.Timedelta(minutes=15)

# Health
health_score, health_label = health_status(
    latest_latency,
    latest_jitter,
    latest_download,
    latest_upload
)

# Alerts
alerts = []
if next_latency_forecast > 50:
    alerts.append("Latency is forecast to be elevated in the next 15 minutes.")
if latest_jitter > 10:
    alerts.append("Jitter is currently high. Real-time applications may be affected.")
if latest_download < 30:
    alerts.append("Download speed is currently low.")
if latest_upload < 10:
    alerts.append("Upload speed is currently low.")
if not alerts:
    alerts.append("No immediate performance alert detected.")


# ======================
# TOP KPI CARDS
# ======================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Current Latency", f"{latest_latency:.2f} ms")
c2.metric("Current Jitter", f"{latest_jitter:.2f} ms")
c3.metric("Current Download", f"{latest_download:.2f} Mbps")
c4.metric("Current Upload", f"{latest_upload:.2f} Mbps")
c5.metric("Health Score", f"{health_score} / 100", delta=health_label)

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


# ======================
# TAB 1: OVERVIEW
# ======================
with tab1:
    st.subheader("Network Overview")

    left, right = st.columns([2, 1])

    with left:
        fig_latency = make_line_chart(
            filtered_clean,
            "timestamp",
            "ping_avg_rtt_ms",
            "Latency Over Time",
            "Latency (ms)",
            "Latency"
        )
        st.plotly_chart(fig_latency, use_container_width=True)

    with right:
        st.markdown("### Current Forecast")
        st.metric("Next 15-min Latency Forecast", f"{next_latency_forecast:.2f} ms")
        st.write(f"Forecast time: **{forecast_time}**")
        st.write(f"Selected model: **{selected_model}**")
        st.write(f"Dataset version: **{dataset_version}**")

        st.markdown("### Quick Interpretation")
        st.write(
            f"""
            The current health state is **{health_label}** with a score of **{health_score}/100**.
            This combines recent latency, jitter, download, and upload behavior into a simple operational indicator.
            """
        )


# ======================
# TAB 2: HISTORICAL TRENDS
# ======================
with tab2:
    st.subheader("Historical Performance Trends")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_jitter = make_line_chart(
            filtered_clean,
            "timestamp",
            "ping_jitter_ms",
            "Jitter Over Time",
            "Jitter (ms)",
            "Jitter"
        )
        st.plotly_chart(fig_jitter, use_container_width=True)

    with col_b:
        fig_download = make_line_chart(
            filtered_clean,
            "timestamp",
            "download_mbps",
            "Download Speed Over Time",
            "Download (Mbps)",
            "Download"
        )
        st.plotly_chart(fig_download, use_container_width=True)

    fig_upload = make_line_chart(
        filtered_clean,
        "timestamp",
        "upload_mbps",
        "Upload Speed Over Time",
        "Upload (Mbps)",
        "Upload"
    )
    st.plotly_chart(fig_upload, use_container_width=True)


# ======================
# TAB 3: FORECASTING
# ======================
with tab3:
    st.subheader("Latency Forecasting")

    fig_pred = make_actual_vs_predicted_chart(
        y_test,
        y_pred,
        chart_points,
        selected_model
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown("### Forecast Summary")
    st.write(
        f"""
        The selected model, **{selected_model}**, was used to estimate the next 15-minute latency value.
        The predicted latency for **{forecast_time}** is **{next_latency_forecast:.2f} ms**.
        """
    )

    if next_latency_forecast <= 35:
        st.success("Forecast suggests normal latency conditions.")
    elif next_latency_forecast <= 50:
        st.warning("Forecast suggests moderate latency increase.")
    else:
        st.error("Forecast suggests potentially poor latency conditions.")


# ======================
# TAB 4: ALERTS
# ======================
with tab4:
    st.subheader("Operational Alerts")

    for alert in alerts:
        if "No immediate" in alert:
            st.success(alert)
        elif "elevated" in alert or "high" in alert:
            st.warning(alert)
        else:
            st.info(alert)

    st.markdown("### Suggested Operational Use")
    st.write(
        """
        This section is intended to support university use cases such as:
        - checking whether the network is suitable for online lectures
        - monitoring performance before live sessions or uploads
        - identifying periods of degraded Starlink behavior
        """
    )


# ======================
# TAB 5: MODEL EVALUATION
# ======================
with tab5:
    st.subheader("Model Evaluation")

    model_table = make_model_comparison_table(dataset_version)
    st.dataframe(model_table, use_container_width=True)

    st.markdown("### Selected Model Metrics")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{metrics['mae']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.3f}")

    st.markdown("### Interpretation")
    st.write(
        """
        Lower MAE and RMSE indicate better forecasting performance.
        The naive baseline is included as a benchmark, while Linear Regression, Random Forest,
        and XGBoost serve as machine learning comparisons.
        """
    )

st.markdown("---")
st.write("Version 1 focuses on Starlink latency forecasting. Future versions can add Omantel and Awasr comparison, multi-metric forecasting, and advanced alert logic.")