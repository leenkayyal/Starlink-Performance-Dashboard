import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
starlink = pd.read_csv("Cleaned/starlink_clean.csv")
omantel = pd.read_csv("Cleaned/omantel_clean.csv")

# Convert timestamp
starlink["timestamp"] = pd.to_datetime(starlink["timestamp"])
omantel["timestamp"] = pd.to_datetime(omantel["timestamp"])

# Correct download column
download_col = "download_mbps"

# REAL DATA ONLY
starlink_real = starlink[starlink["was_estimated_row"] == False]
omantel_real = omantel[omantel["was_estimated_row"] == False]

# -----------------------------
# PLOT 1: FULL DATA
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink["timestamp"], starlink[download_col], label="Starlink (Full)")
plt.plot(omantel["timestamp"], omantel[download_col], label="Omantel (Full)")

plt.title("Download Speed Comparison (Full Data)")
plt.xlabel("Time")
plt.ylabel("Download Speed (Mbps)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("download_full.png")
plt.show()

# -----------------------------
# PLOT 2: REAL DATA ONLY
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink_real["timestamp"], starlink_real[download_col], label="Starlink (Real)")
plt.plot(omantel_real["timestamp"], omantel_real[download_col], label="Omantel (Real)")

plt.title("Download Speed Comparison (Real Data Only)")
plt.xlabel("Time")
plt.ylabel("Download Speed (Mbps)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("download_real.png")
plt.show()