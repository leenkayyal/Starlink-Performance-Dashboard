"""
combined_robustness_check.py
----------------------------
Runs a combined-dataset robustness check for the Starlink latency forecasting pipeline.

This script compares:
1. The selected lag 1-4 setup used in the final combined model.
2. An extended lag 1-12 setup.
3. A small LSTM sequence model using 12 previous intervals.

Outputs:
- Cleaned/combined/combined_robustness_results_with_lstm.csv

Expected main conclusion:
The extended lag window and LSTM do not provide a meaningful practical improvement
over the selected lag 1-4 Linear Regression model. Linear Regression remains preferred
because it provides the best balance of accuracy, stability, simplicity, and interpretability.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("WARNING: xgboost is not installed. XGBoost will be skipped.")

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("WARNING: PyTorch is not installed. LSTM will be skipped.")


# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE_A_FORECAST = "Cleaned/experiment_A/starlink_forecast_v2.csv"
DATA_FILE_B_FORECAST = "Cleaned/experiment_B/starlink_2_forecast.csv"

DATA_FILE_A_CLEAN = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
DATA_FILE_B_CLEAN = "Cleaned/experiment_B/Starlink_2_cleaned.csv"

OUTPUT_DIR = "Cleaned/combined"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined_robustness_results_with_lstm.csv")

TARGET_COL = "target"
MAIN_LAG = "ping_avg_rtt_ms_lag_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    """Return MAE, RMSE, and R2 for a prediction vector."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def add_result(rows, setup, model, y_true, y_pred):
    """Append rounded model results to the result list."""
    metrics = evaluate(y_true, y_pred)
    rows.append({
        "Setup": setup,
        "Model": model,
        "MAE": round(metrics["MAE"], 4),
        "RMSE": round(metrics["RMSE"], 4),
        "R2": round(metrics["R2"], 4),
    })


def filter_forecast_dataset(df, max_lag=4):
    """Apply the same sanity filters used in the combined training pipeline."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    for col in [TARGET_COL] + [f"ping_avg_rtt_ms_lag_{i}" for i in range(1, max_lag + 1)]:
        df = df[(df[col] > 0) & (df[col] < 500)]

    df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
    df = df[(df["upload_lag_1"] > 0) & (df["upload_lag_1"] < 500)]
    df = df[(df["jitter_lag_1"] >= 0) & (df["jitter_lag_1"] < 100)]

    return df.reset_index(drop=True)


def train_classical_models(df, setup_name, rows):
    """Train/evaluate Naive, LR, RF, and XGBoost using chronological 80/20 split."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Naive baseline: next value = previous latency value
    add_result(rows, setup_name, "Naive Baseline", y_test, X_test[MAIN_LAG].values)

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    add_result(rows, setup_name, "Linear Regression", y_test, lr.predict(X_test))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=150,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)
    add_result(rows, setup_name, "Random Forest", y_test, rf.predict(X_test))

    # XGBoost
    if XGB_OK:
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        ).fit(X_train, y_train)
        add_result(rows, setup_name, "XGBoost", y_test, xgb.predict(X_test))

    return len(df), len(y_test)


def build_lag12_dataset(clean_file):
    """
    Build a lag 1-12 forecasting dataset from one cleaned Starlink time-series file.
    This is done separately for Experiment A and B so lag features do not cross
    experiment boundaries.
    """
    df = pd.read_csv(clean_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = [
        "ping_avg_rtt_ms",
        "download_mbps",
        "upload_mbps",
        "ping_jitter_ms",
        "weather_code",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {clean_file}: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame()
    out["hour"] = df["timestamp"].dt.hour
    out["day_of_week"] = df["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    for lag in range(1, 13):
        out[f"ping_avg_rtt_ms_lag_{lag}"] = df["ping_avg_rtt_ms"].shift(lag)

    out["download_lag_1"] = df["download_mbps"].shift(1)
    out["upload_lag_1"] = df["upload_mbps"].shift(1)
    out["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)
    out["weather_code"] = df["weather_code"]
    out["target"] = df["ping_avg_rtt_ms"]

    return out.dropna().reset_index(drop=True)


def load_lstm_sequences(clean_file, sequence_length=12):
    """
    Create LSTM sequences from one cleaned Starlink time-series file.
    This is also done separately for each experiment to avoid crossing boundaries.
    """
    df = pd.read_csv(clean_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = [
        "ping_avg_rtt_ms",
        "download_mbps",
        "upload_mbps",
        "ping_jitter_ms",
        "weather_code",
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {clean_file}: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Temporal context for the LSTM sequence model
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    sequence_features = [
        "ping_avg_rtt_ms",
        "download_mbps",
        "upload_mbps",
        "ping_jitter_ms",
        "weather_code",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
    ]

    df = df.dropna(subset=sequence_features).reset_index(drop=True)

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df.loc[i - sequence_length:i - 1, sequence_features].values.astype("float32"))
        y.append(float(df.loc[i, "ping_avg_rtt_ms"]))

    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


if TORCH_OK:
    class TinyLSTM(nn.Module):
        """Small LSTM used only as a robustness reference model."""
        def __init__(self, input_size, hidden_size=8):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)


def run_lstm(rows, sequence_length=12):
    """Train/evaluate a small LSTM sequence model."""
    if not TORCH_OK:
        return None, None

    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    XA, yA = load_lstm_sequences(DATA_FILE_A_CLEAN, sequence_length)
    XB, yB = load_lstm_sequences(DATA_FILE_B_CLEAN, sequence_length)

    X = np.concatenate([XA, XB], axis=0)
    y = np.concatenate([yA, yB], axis=0)

    mask = (y > 0) & (y < 500)
    X, y = X[mask], y[mask]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_features = X_train.shape[-1]

    feature_scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train_scaled = feature_scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).ravel()

    model = TinyLSTM(input_size=n_features, hidden_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    # Full-batch training is used for speed and reproducibility.
    for _ in range(35):
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    add_result(rows, f"LSTM sequence {sequence_length}", "LSTM", y_test, pred)
    return len(X), len(y_test)


def main():
    rows = []

    # ── Lag 1-4 official combined setup ────────────────────────────────────────
    df_a4 = pd.read_csv(DATA_FILE_A_FORECAST)
    df_b4 = pd.read_csv(DATA_FILE_B_FORECAST)
    common_cols = [c for c in df_a4.columns if c in set(df_b4.columns)]

    df4 = pd.concat([df_a4[common_cols], df_b4[common_cols]], ignore_index=True).drop_duplicates()
    df4 = filter_forecast_dataset(df4, max_lag=4)
    lag4_rows, lag4_test = train_classical_models(df4, "Lag 1-4", rows)

    # ── Lag 1-12 extended setup ────────────────────────────────────────────────
    df_a12 = build_lag12_dataset(DATA_FILE_A_CLEAN)
    df_b12 = build_lag12_dataset(DATA_FILE_B_CLEAN)
    df12 = pd.concat([df_a12, df_b12], ignore_index=True).drop_duplicates()
    df12 = filter_forecast_dataset(df12, max_lag=12)
    lag12_rows, lag12_test = train_classical_models(df12, "Lag 1-12", rows)

    # ── LSTM sequence reference model ──────────────────────────────────────────
    lstm_rows, lstm_test = run_lstm(rows, sequence_length=12)

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nRobustness check complete.")
    print(f"Lag 1-4 rows:  {lag4_rows} | Test rows: {lag4_test}")
    print(f"Lag 1-12 rows: {lag12_rows} | Test rows: {lag12_test}")
    if lstm_rows is not None:
        print(f"LSTM rows:     {lstm_rows} | Test rows: {lstm_test}")

    print(f"\nSaved results to: {OUTPUT_FILE}\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
