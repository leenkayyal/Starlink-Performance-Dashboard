import pandas as pd

# Load data
df = pd.read_csv("Cleaned/starlink_clean.csv")

print("Original rows:", len(df))

# Remove bad upload values
df = df[(df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)]

# (Optional but recommended) clean other columns too
df = df[(df["download_mbps"] > 0) & (df["download_mbps"] < 1000)]
df = df[(df["ping_avg_rtt_ms"] > 0) & (df["ping_avg_rtt_ms"] < 500)]
df = df[(df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100)]

print("Cleaned rows:", len(df))

# Save fixed file
df.to_csv("Cleaned/starlink_clean_FIXED.csv", index=False)

print("Saved cleaned dataset ✅")