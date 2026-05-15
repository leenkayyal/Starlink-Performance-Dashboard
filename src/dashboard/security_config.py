# Roles and credentials for login system
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "viewer": {"password": "view456", "role": "Viewer"},
}

# CSV files that will be integrity-checked with SHA-256
CSV_FILES_TO_HASH = [
    "Cleaned/experiment_A/starlink_clean_FIXED.csv",
    "Cleaned/experiment_A/starlink_forecast.csv",
    "Cleaned/experiment_B/Starlink_2_cleaned.csv",
    "Cleaned/experiment_B/starlink_2_forecast.csv",
    "Cleaned/combined/starlink_forecast_combined.csv",
]
