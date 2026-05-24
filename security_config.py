# Roles and credentials for login system
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "supervisor": {"password": "super456", "role": "supervisor"},
    "guest": {"password": "guest789", "role": "guest"},
}

# CSV files that will be integrity-checked with SHA-256
CSV_FILES_TO_HASH = [
    "Cleaned/starlink_clean_FIXED.csv",
    "Cleaned/starlink_forecast_v2.csv",
]
