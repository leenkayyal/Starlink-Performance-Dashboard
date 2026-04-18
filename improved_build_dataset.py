import pandas as pd
import numpy as np

# ======================
# CONFIG
# ======================
INPUT_FILE = "Cleaned/starlink_clean_FIXED.csv"
OUTPUT_FILE = "Cleaned/starlink_forecast_v2.csv"

TARGET = "ping_avg_rtt_ms"


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# Keep only real rows
df = df[df["was_estimated_row"] == False].copy()


# ======================
# BASIC TIME FEATURES
# ======================
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Better time encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)


# ======================
# CLEAN IMPORTANT NUMERIC COLUMNS
# ======================
numeric_cols = [
    "download_mbps",
    "upload_mbps",
    "ping_jitter_ms",
    TARGET
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove impossible values
df = df[(df[TARGET] > 0) & (df[TARGET] < 500)]
df = df[(df["download_mbps"] > 0) & (df["download_mbps"] < 1000)]
df = df[(df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)]
df = df[(df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100)]


# ======================
# LAG FEATURES
# ======================
for lag in range(1, 9):   # 8 lags = 2 hours
    df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)

df["download_lag_1"] = df["download_mbps"].shift(1)
df["upload_lag_1"] = df["upload_mbps"].shift(1)
df["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)


# ======================
# ROLLING FEATURES
# ======================
df["latency_roll_mean_3"] = df[TARGET].rolling(window=3).mean()
df["latency_roll_std_3"] = df[TARGET].rolling(window=3).std()

df["latency_roll_mean_6"] = df[TARGET].rolling(window=6).mean()
df["latency_roll_std_6"] = df[TARGET].rolling(window=6).std()


# ======================
# CHANGE FEATURES
# ======================
df["latency_delta_1"] = df[TARGET] - df[TARGET].shift(1)
df["latency_delta_2"] = df[TARGET].shift(1) - df[TARGET].shift(2)


# ======================
# SPIKE FLAG
# ======================
threshold = df[TARGET].mean() + 2 * df[TARGET].std()
df["is_spike_recent"] = (df[TARGET].shift(1) > threshold).astype(int)


# ======================
# FUTURE TARGET
# ======================
# Predict next 15-minute latency
df["target"] = df[TARGET].shift(-1)


# ======================
# DROP NA ROWS
# ======================
df = df.dropna().reset_index(drop=True)


# ======================
# FEATURE LIST
# ======================
features = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",

    # lag features
    f"{TARGET}_lag_1",
    f"{TARGET}_lag_2",
    f"{TARGET}_lag_3",
    f"{TARGET}_lag_4",
    f"{TARGET}_lag_5",
    f"{TARGET}_lag_6",
    f"{TARGET}_lag_7",
    f"{TARGET}_lag_8",

    # network features
    "download_lag_1",
    "upload_lag_1",
    "jitter_lag_1",

    # rolling features
    "latency_roll_mean_3",
    "latency_roll_std_3",
    "latency_roll_mean_6",
    "latency_roll_std_6",

    # change features
    "latency_delta_1",
    "latency_delta_2",

    # spike info
    "is_spike_recent",

    # weather
    "weather_code"
]

# Keep only existing columns
features = [col for col in features if col in df.columns]

final_df = df[features + ["target"]]


# ======================
# SAVE
# ======================
final_df.to_csv(OUTPUT_FILE, index=False)

print("Improved dataset created successfully!")
print("Rows:", len(final_df))
print("Columns:", final_df.columns.tolist())
print("Saved to:", OUTPUT_FILE)