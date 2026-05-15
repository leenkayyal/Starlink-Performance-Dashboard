import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost is optional
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Internet Performance Forecast Dashboard",
    layout="wide"
)

CLEAN_FILE = "Cleaned/starlink_clean.csv"
FORECAST_FILE_V1 = "Cleaned/starlink_forecast.csv"
FORECAST_FILE_V2 = "Cleaned/starlink_forecast_v2.csv"

TARGET = "ping_avg_rtt_ms"


# ======================
# HELPERS
# ======================
def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_features_from_cleaned(clean_path, improved=True):
    """
    Rebuild forecasting dataset from cleaned latency file.
    This lets us:
    1) evaluate historical predictions
    2) generate the latest row for future forecast
    """
    df = pd.read_csv(clean_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep only real rows for modeling
    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False].copy()

    # Basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if improved:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Numeric cleanup
    numeric_cols = [TARGET, "download_mbps", "upload_mbps", "ping_jitter_ms", "weather_code"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic cleaning
    if TARGET in df.columns:
        df = df[(df[TARGET] > 0) & (df[TARGET] < 500)]
    if "download_mbps" in df.columns:
        df = df[(df["download_mbps"] > 0) & (df["download_mbps"] < 1000)]
    if "upload_mbps" in df.columns:
        df = df[(df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)]
    if "ping_jitter_ms" in df.columns:
        df = df[(df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100)]

    # Lag features
    max_lag = 8 if improved else 4
    for lag in range(1, max_lag + 1):
        df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)

    if "download_mbps" in df.columns:
        df["download_lag_1"] = df["download_mbps"].shift(1)
    if "upload_mbps" in df.columns:
        df["upload_lag_1"] = df["upload_mbps"].shift(1)
    if "ping_jitter_ms" in df.columns:
        df["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)

    # Improved engineered features
    if improved:
        df["latency_roll_mean_3"] = df[TARGET].rolling(window=3).mean()
        df["latency_roll_std_3"] = df[TARGET].rolling(window=3).std()
        df["latency_roll_mean_6"] = df[TARGET].rolling(window=6).mean()
        df["latency_roll_std_6"] = df[TARGET].rolling(window=6).std()

        df["latency_delta_1"] = df[TARGET] - df[TARGET].shift(1)
        df["latency_delta_2"] = df[TARGET].shift(1) - df[TARGET].shift(2)

        threshold = df[TARGET].mean() + 2 * df[TARGET].std()
        df["is_spike_recent"] = (df[TARGET].shift(1) > threshold).astype(int)

    # Future target
    df["target"] = df[TARGET].shift(-1)

    # Drop NaN rows created by feature engineering
    df = df.dropna().reset_index(drop=True)

    # Feature list
    if improved:
        features = [
            "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
            f"{TARGET}_lag_1", f"{TARGET}_lag_2", f"{TARGET}_lag_3", f"{TARGET}_lag_4",
            f"{TARGET}_lag_5", f"{TARGET}_lag_6", f"{TARGET}_lag_7", f"{TARGET}_lag_8",
            "download_lag_1", "upload_lag_1", "jitter_lag_1",
            "latency_roll_mean_3", "latency_roll_std_3",
            "latency_roll_mean_6", "latency_roll_std_6",
            "latency_delta_1", "latency_delta_2",
            "is_spike_recent",
            "weather_code"
        ]
    else:
        features = [
            "hour", "day_of_week", "is_weekend",
            f"{TARGET}_lag_1", f"{TARGET}_lag_2", f"{TARGET}_lag_3", f"{TARGET}_lag_4",
            "download_lag_1", "upload_lag_1", "jitter_lag_1",
            "weather_code"
        ]

    features = [c for c in features if c in df.columns]

    final_df = df[["timestamp"] + features + ["target"]].copy()
    return final_df, features


def get_dataset_choice():
    if st.session_state.get("feature_version", "Improved (v2)") == "Improved (v2)":
        return True
    return False


def train_and_evaluate(df, features, model_name):
    X = df[features]
    y = df["target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    t_test = df["timestamp"].iloc[split_index:]

    if model_name == "Naive Baseline":
        y_pred = X_test[f"{TARGET}_lag_1"].values
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
            raise RuntimeError("XGBoost is not installed.")
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

    mae = mean_absolute_error(y_test, y_pred)
    rmse = safe_rmse(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "timestamps_test": t_test,
        "mae": mae,
        "rmse": rmse
    }


def fit_full_model(df, features, model_name):
    X = df[features]
    y = df["target"]

    if model_name == "Naive Baseline":
        return None

    elif model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    elif model_name == "XGBoost":
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is not installed.")
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

    model.fit(X, y)
    return model


def get_latest_forecast_input(clean_path, improved=True):
    df_features, features = build_features_from_cleaned(clean_path, improved=improved)
    latest_row = df_features.iloc[[-1]].copy()
    latest_timestamp = latest_row["timestamp"].iloc[0]
    X_latest = latest_row[features]
    return X_latest, latest_timestamp


def predict_next_value(df_features, features, model_name, clean_path, improved=True):
    X_latest, latest_timestamp = get_latest_forecast_input(clean_path, improved=improved)

    if model_name == "Naive Baseline":
        next_pred = float(X_latest[f"{TARGET}_lag_1"].iloc[0])
    else:
        model = fit_full_model(df_features, features, model_name)
        next_pred = float(model.predict(X_latest)[0])

    next_time = latest_timestamp + pd.Timedelta(minutes=15)
    return next_pred, next_time


def plot_actual_vs_predicted(timestamps, actual, predicted, model_name, points_to_show):
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    timestamps = pd.Series(timestamps).reset_index(drop=True)

    if points_to_show < len(actual):
        actual = actual.iloc[-points_to_show:]
        predicted = predicted.iloc[-points_to_show:]
        timestamps = timestamps.iloc[-points_to_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=actual,
        mode="lines",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=predicted,
        mode="lines",
        name=f"Predicted ({model_name})"
    ))

    fig.update_layout(
        title=f"Actual vs Predicted Latency ({model_name})",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        height=500
    )
    return fig


def plot_residuals(timestamps, actual, predicted, points_to_show):
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    timestamps = pd.Series(timestamps).reset_index(drop=True)
    residuals = actual - predicted

    if points_to_show < len(actual):
        residuals = residuals.iloc[-points_to_show:]
        timestamps = timestamps.iloc[-points_to_show:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=residuals,
        mode="lines",
        name="Residual (Actual - Predicted)"
    ))
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title="Time",
        yaxis_title="Residual (ms)",
        height=350
    )
    return fig


# ======================
# STREAMLIT UI
# ======================
st.title("Internet Performance Forecast Dashboard")
st.caption("Latency forecasting dashboard using historical network data")

with st.sidebar:
    st.header("Controls")

    feature_version = st.selectbox(
        "Dataset version",
        ["Original (v1)", "Improved (v2)"],
        index=1,
        key="feature_version"
    )

    model_options = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE:
        model_options.append("XGBoost")

    selected_model = st.selectbox("Model", model_options, index=1)

    points_to_show = st.slider(
        "Points to display",
        min_value=50,
        max_value=400,
        value=200,
        step=25
    )

    show_table = st.checkbox("Show latest feature row", value=False)

# Choose dataset version
improved = (feature_version == "Improved (v2)")
df_features, features = build_features_from_cleaned(CLEAN_FILE, improved=improved)

# Evaluate selected model
result = train_and_evaluate(df_features, features, selected_model)

# Future forecast
next_pred, next_time = predict_next_value(
    df_features=df_features,
    features=features,
    model_name=selected_model,
    clean_path=CLEAN_FILE,
    improved=improved
)

# Latest actual
latest_actual = float(df_features["target"].iloc[-1])

# ======================
# METRIC CARDS
# ======================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", selected_model)
c2.metric("MAE", f"{result['mae']:.3f}")
c3.metric("RMSE", f"{result['rmse']:.3f}")
c4.metric("Next 15-min Forecast", f"{next_pred:.2f} ms")

c5, c6 = st.columns(2)
c5.metric("Latest Actual Latency", f"{latest_actual:.2f} ms")
c6.metric("Forecast Timestamp", str(next_time))

st.markdown("---")

# ======================
# MAIN CHART
# ======================
fig_main = plot_actual_vs_predicted(
    result["timestamps_test"],
    result["y_test"],
    result["y_pred"],
    selected_model,
    points_to_show
)
st.plotly_chart(fig_main, use_container_width=True)

# ======================
# RESIDUALS
# ======================
fig_res = plot_residuals(
    result["timestamps_test"],
    result["y_test"],
    result["y_pred"],
    points_to_show
)
st.plotly_chart(fig_res, use_container_width=True)

# ======================
# TEXT INTERPRETATION
# ======================
st.subheader("Interpretation")
st.write(
    f"""
This dashboard evaluates **{selected_model}** on the selected forecasting dataset.
The chart compares actual latency values with predicted values over the test period.
The next 15-minute forecast is generated using the latest available historical window.
"""
)

# ======================
# OPTIONAL TABLE
# ======================
if show_table:
    st.subheader("Latest Feature Row Used for Forecast")
    latest_X, _ = get_latest_forecast_input(CLEAN_FILE, improved=improved)
    st.dataframe(latest_X)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.write("Current focus: Starlink latency forecasting. The dashboard structure can later be extended to Omantel and Awasr.")