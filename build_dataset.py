import pandas as pd

# ======================
# CONFIG
# ======================
INPUT_FILE = "Cleaned/starlink_clean.csv"
OUTPUT_FILE = "Cleaned/starlink_forecast.csv"

TARGET = "ping_avg_rtt_ms"
LAGS = [1, 2, 3, 4]   # 15, 30, 45, 60 minutes


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(INPUT_FILE)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# Keep only real data
df = df[df["was_estimated_row"] == False].copy()


# ======================
# TIME FEATURES
# ======================
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# ======================
# LAG FEATURES
# ======================
for lag in LAGS:
    df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)

df["download_lag_1"] = df["download_mbps"].shift(1)
df["upload_lag_1"] = df["upload_mbps"].shift(1)
df["jitter_lag_1"] = df["ping_jitter_ms"].shift(1)


# ======================
# FUTURE TARGET
# ======================
# Predict next 15-minute latency
df["target"] = df[TARGET].shift(-1)


# ======================
# DROP MISSING ROWS
# ======================
df = df.dropna().reset_index(drop=True)


# ======================
# FEATURE LIST
# ======================
features = [
    "hour",
    "day_of_week",
    "is_weekend",

    f"{TARGET}_lag_1",
    f"{TARGET}_lag_2",
    f"{TARGET}_lag_3",
    f"{TARGET}_lag_4",

    "download_lag_1",
    "upload_lag_1",
    "jitter_lag_1",

    # weather columns based on your actual cleaned file
    "temperature",
    "humidity",
    "wind_speed",
    "weather_code"
]

# Keep only columns that actually exist
features = [col for col in features if col in df.columns]

final_df = df[features + ["target"]]


# ======================
# SAVE
# ======================
final_df.to_csv(OUTPUT_FILE, index=False)

print("Dataset created successfully!")
print("Rows:", len(final_df))
print("Columns:", final_df.columns.tolist())
print("Saved to:", OUTPUT_FILE)