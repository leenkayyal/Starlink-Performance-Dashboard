import pandas as pd

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Cleaned/starlink_forecast.csv")

print("Original rows:", len(df))


# ======================
# REMOVE IMPOSSIBLE VALUES
# ======================

# latency must be positive and reasonable
df = df[(df["target"] > 0) & (df["target"] < 500)]

# download speed (realistic bounds)
df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]

# upload speed (fix your corrupted value here)
df = df[(df["upload_lag_1"] > 0) & (df["upload_lag_1"] < 500)]

# jitter should be small
df = df[(df["jitter_lag_1"] >= 0) & (df["jitter_lag_1"] < 100)]


# ======================
# REMOVE EXTREME OUTLIERS (OPTIONAL BUT STRONG)
# ======================

# remove top 1% extreme values in latency
upper_limit = df["target"].quantile(0.99)
df = df[df["target"] <= upper_limit]


# ======================
# RESET INDEX
# ======================
df = df.reset_index(drop=True)

print("Cleaned rows:", len(df))


# ======================
# SAVE CLEAN VERSION
# ======================
df.to_csv("Cleaned/starlink_forecast_clean.csv", index=False)

print("Saved cleaned dataset!")