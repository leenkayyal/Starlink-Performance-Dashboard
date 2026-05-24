import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
starlink = pd.read_csv("Cleaned/starlink_clean.csv")
omantel = pd.read_csv("Cleaned/omantel_clean.csv")

# Convert timestamp
starlink["timestamp"] = pd.to_datetime(starlink["timestamp"])
omantel["timestamp"] = pd.to_datetime(omantel["timestamp"])

# Correct latency column
latency_col = "ping_avg_rtt_ms"

# REAL DATA ONLY
starlink_real = starlink[starlink["was_estimated_row"] == False]
omantel_real = omantel[omantel["was_estimated_row"] == False]

# -----------------------------
# PLOT 1: FULL DATA
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink["timestamp"], starlink[latency_col], label="Starlink (Full)")
plt.plot(omantel["timestamp"], omantel[latency_col], label="Omantel (Full)")

plt.title("Latency Comparison (Full Data)")
plt.xlabel("Time")
plt.ylabel("Latency (ms)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("latency_full.png")
plt.show()

# -----------------------------
# PLOT 2: REAL DATA ONLY
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink_real["timestamp"], starlink_real[latency_col], label="Starlink (Real)")
plt.plot(omantel_real["timestamp"], omantel_real[latency_col], label="Omantel (Real)")

plt.title("Latency Comparison (Real Data Only)")
plt.xlabel("Time")
plt.ylabel("Latency (ms)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("latency_real.png")
plt.show()