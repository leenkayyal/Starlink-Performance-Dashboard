import pandas as pd
import os

STARLINK_PATH = "Raw/starlink_data.csv"
OMANTEL_PATH = "Raw/omantel_data.csv"

OUTPUT_DIR = "Cleaned"
OUTPUT_STARLINK = os.path.join(OUTPUT_DIR, "starlink_clean.csv")
OUTPUT_OMANTEL = os.path.join(OUTPUT_DIR, "omantel_clean.csv")


def clean_dataset(path, label):
    print(f"\nCleaning {label} dataset...")
    print("Trying to open:", path)

    df = pd.read_csv(path, on_bad_lines="skip")

    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {path}")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Normalize timestamps to 15-minute slots
    df["timestamp"] = df["timestamp"].dt.floor("15min")

    # Sort by time
    df = df.sort_values("timestamp")

    # Remove duplicates by timestamp
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    after_dedup = len(df)

    print(f"{label}: removed {before_dedup - after_dedup} duplicate rows")

    # Define columns that count as actual measured values
    measurement_cols = [
        "ping_min", "ping_avg", "ping_max", "ping_mdev",
        "packet_loss", "speedtest_download", "upload_m",
        "weather_code", "temperature_2m", "cloud_cover", "wind_speed_10m",
        "ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps", "upload_mbps",
        "temperature_c", "humidity_percent", "wind_speed_mps"
    ]
    measurement_cols = [c for c in measurement_cols if c in df.columns]


    if measurement_cols:
        df["had_empty_measurement_in_raw"] = df[measurement_cols].isna().any(axis=1)
    else:
        df["had_empty_measurement_in_raw"] = False


    # Build full 15-minute timeline
    full_timeline = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="15min"
    )
    full_df = pd.DataFrame({"timestamp": full_timeline})

    # Merge with indicator so we know which timestamps were missing entirely
    merged = pd.merge(
        full_df,
        df,
        on="timestamp",
        how="left",
        indicator=True
    )

    # Flag 1: timestamp missing entirely from raw data
    merged["was_missing_timestamp_row"] = merged["_merge"] == "left_only"

    # Raw rows that existed but had empty measurements
    merged["had_empty_measurement_in_raw"] = merged["had_empty_measurement_in_raw"].fillna(False)

    # Flag 2: final row is estimated if:
    # - timestamp was missing, OR
    # - row existed but had empty measurement values
    merged["was_estimated_row"] = (
        merged["was_missing_timestamp_row"] |
        merged["had_empty_measurement_in_raw"]
    )

    # Drop helper merge column
    merged = merged.drop(columns=["_merge"])

    # Print stats before filling
    missing_timestamp_count = int(merged["was_missing_timestamp_row"].sum())
    missing_timestamp_pct = merged["was_missing_timestamp_row"].mean() * 100

    estimated_count = int(merged["was_estimated_row"].sum())
    estimated_pct = merged["was_estimated_row"].mean() * 100

    print(f"{label}: missing timestamp rows = {missing_timestamp_count}")
    print(f"{label}: missing timestamp percentage = {missing_timestamp_pct:.2f}%")
    print(f"{label}: estimated rows = {estimated_count}")
    print(f"{label}: estimated percentage = {estimated_pct:.2f}%")

    # Preserve flags
    missing_flag = merged["was_missing_timestamp_row"].copy()
    empty_flag = merged["had_empty_measurement_in_raw"].copy()
    estimated_flag = merged["was_estimated_row"].copy()

    # Fill only data columns, not flag columns
    flag_cols = [
        "was_missing_timestamp_row",
        "had_empty_measurement_in_raw",
        "was_estimated_row"
    ]
    cols_to_fill = [c for c in merged.columns if c not in flag_cols]
    merged[cols_to_fill] = merged[cols_to_fill].ffill().bfill()

    # Restore exact flag values
    merged["was_missing_timestamp_row"] = missing_flag
    merged["had_empty_measurement_in_raw"] = empty_flag
    merged["was_estimated_row"] = estimated_flag

    print(f"{label}: final cleaned rows = {len(merged)}")

    return merged


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    starlink_clean = clean_dataset(STARLINK_PATH, "Starlink")
    omantel_clean = clean_dataset(OMANTEL_PATH, "Omantel")

    starlink_clean.to_csv(OUTPUT_STARLINK, index=False)
    omantel_clean.to_csv(OUTPUT_OMANTEL, index=False)

    print("\nCleaning complete!")
    print(f"Saved: {OUTPUT_STARLINK}")
    print(f"Saved: {OUTPUT_OMANTEL}")


if __name__ == "__main__":
    main()