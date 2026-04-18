import pandas as pd
import matplotlib.pyplot as plt

# Load data
starlink = pd.read_csv("Cleaned/starlink_clean.csv")
omantel = pd.read_csv("Cleaned/omantel_clean.csv")

# Convert timestamp
starlink["timestamp"] = pd.to_datetime(starlink["timestamp"])
omantel["timestamp"] = pd.to_datetime(omantel["timestamp"])

# Fix upload column
upload_col = "upload_mbps"

# Convert to numeric safely
starlink[upload_col] = pd.to_numeric(starlink[upload_col], errors="coerce")
omantel[upload_col] = pd.to_numeric(omantel[upload_col], errors="coerce")

# Remove impossible values
starlink = starlink[(starlink[upload_col] > 0) & (starlink[upload_col] < 1000)]
omantel = omantel[(omantel[upload_col] > 0) & (omantel[upload_col] < 1000)]

# REAL DATA ONLY
starlink_real = starlink[starlink["was_estimated_row"] == False]
omantel_real = omantel[omantel["was_estimated_row"] == False]

# -----------------------------
# PLOT FULL
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink["timestamp"], starlink[upload_col], label="Starlink (Full)")
plt.plot(omantel["timestamp"], omantel[upload_col], label="Omantel (Full)")

plt.title("Upload Speed Comparison (Fixed)")
plt.xlabel("Time")
plt.ylabel("Upload Speed (Mbps)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT REAL ONLY
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(starlink_real["timestamp"], starlink_real[upload_col], label="Starlink (Real)")
plt.plot(omantel_real["timestamp"], omantel_real[upload_col], label="Omantel (Real)")

plt.title("Upload Speed Comparison (Real Data - Fixed)")
plt.xlabel("Time")
plt.ylabel("Upload Speed (Mbps)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()