"""
clean_both_datasets_v2.py
=========================
Cleans starlink_data2.csv and Awasr_data.csv.

Outputs:
  Starlink_2_cleaned.csv
  Awasr_cleaned.csv
  Starlink_2_cleaning_report.txt
  Awasr_cleaning_report.txt

Flag columns (True/False, matching previous clean data convention):
  had_empty_measurement_in_raw  -- row existed in raw data but had at least one NaN
  was_missing_timestamp_row     -- row did NOT exist in raw data; created to fill a gap
  was_estimated_row             -- any numeric value in this row was interpolated/filled

Rules:
  - was_missing_timestamp_row=True  implies was_estimated_row=True
  - A row with had_empty_measurement_in_raw=True is only also was_estimated_row=True
    if the missing value was actually filled (i.e. interpolation succeeded)
  - Real rows where all values were present get all three flags = False
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
STARLINK_IN  = "Raw/experiment_B/starlink_data2.csv"
AWASR_IN     = "Raw/experiment_B/Awasr_data.csv"
STARLINK_OUT = "Cleaned/experiment_B/Starlink_2_cleaned.csv"
AWASR_OUT    = "Cleaned/experiment_B/Awasr_cleaned.csv"
SL_REPORT    = "data_reports/Starlink_2_cleaning_report.txt"
AW_REPORT    = "data_reports/Awasr_cleaning_report.txt"

CORRUPT_UPLOAD_THRESHOLD = -1e10   # known speedtest-cli artefact

NUMERIC_COLS = [
    "ping_avg_rtt_ms", "ping_jitter_ms", "packet_loss_percent",
    "speedtest_ping_ms", "download_mbps", "upload_mbps",
    "temperature_c", "humidity_percent", "wind_speed_mps",
    "precipitation_mm", "weather_code"
]


# -----------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------

def coerce_numerics(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fix_corrupt_upload(df):
    """Replace known corrupt upload_mbps artefact with NaN."""
    if "upload_mbps" not in df.columns:
        return df, 0
    mask = df["upload_mbps"] < CORRUPT_UPLOAD_THRESHOLD
    count = int(mask.sum())
    df.loc[mask, "upload_mbps"] = np.nan
    return df, count


def interpolate_cols(df, cols):
    """
    Time-based linear interpolation on a timestamp-indexed DataFrame.
    Returns the filled DataFrame AND a boolean Series marking which rows
    had values actually filled (were NaN before, not NaN after).
    """
    df = df.set_index("timestamp")
    filled_mask = pd.Series(False, index=df.index)

    for c in cols:
        if c not in df.columns:
            continue
        before = df[c].isna()
        df[c] = df[c].interpolate(method="time", limit_direction="both")
        after  = df[c].isna()
        # rows that were NaN before and are now filled
        filled_mask = filled_mask | (before & ~after)

    df = df.reset_index()
    return df, filled_mask.values


def make_15min_grid(start, end):
    return pd.date_range(start=start, end=end, freq="15min")


# -----------------------------------------------------------------------
# CLEAN STARLINK
# -----------------------------------------------------------------------

def clean_starlink():
    rep = ["STARLINK CLEANING REPORT", "=" * 60]

    # -- Load
    df = pd.read_csv(STARLINK_IN)
    orig_rows = len(df)
    orig_cols = len(df.columns)
    rep.append(f"\nOriginal rows    : {orig_rows}")
    rep.append(f"Original columns : {orig_cols}")

    # -- Standardise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # -- Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        rep.append(f"Rows dropped (unparseable timestamp): {bad_ts}")
        df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # -- Round timestamps to nearest 15-min
    # The raw data has a ~:03s offset from the scheduler; rounding is correct.
    df["timestamp"] = df["timestamp"].dt.round("15min")

    # -- Handle the one duplicate that rounding creates
    # Row 0 (11:49:08) and row 1 (11:52:28) both round to 11:45:00.
    # Row 1 is the closer one; row 0 is from a startup burst -- drop it.
    dup_mask = df.duplicated(subset=["timestamp"], keep="last")
    dupes_removed = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    rep.append(f"Duplicate timestamps after rounding removed : {dupes_removed}")

    # -- Fix corrupt upload values
    df, corrupt_upload = fix_corrupt_upload(df)
    rep.append(f"Corrupt upload_mbps values (< {CORRUPT_UPLOAD_THRESHOLD:.0e}) replaced with NaN: {corrupt_upload}")

    # -- Coerce numerics
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df = coerce_numerics(df, numeric_present)

    # -- Record which original rows had any NaN
    had_empty_in_raw = df[numeric_present].isna().any(axis=1)
    rep.append(f"Original rows with at least one NaN numeric value: {int(had_empty_in_raw.sum())}")

    # -- Build 15-min grid from actual data range
    grid_start = df["timestamp"].min()
    grid_end   = df["timestamp"].max()
    grid = pd.DataFrame({"timestamp": make_15min_grid(grid_start, grid_end)})
    rep.append(f"\nGrid start : {grid_start}")
    rep.append(f"Grid end   : {grid_end}")
    rep.append(f"Grid slots : {len(grid)}")

    # -- Merge data onto grid
    df = df.set_index("timestamp")
    grid = grid.set_index("timestamp")
    merged = grid.join(df, how="left")

    # -- Identify which rows were created (timestamp gap)
    was_missing_ts = merged["network_type"].isna()   # no data at all for this slot
    rep.append(f"Missing timestamp rows created (interpolated): {int(was_missing_ts.sum())}")
    missing_slots = merged.index[was_missing_ts].tolist()
    for s in missing_slots:
        rep.append(f"  {s}")

    merged = merged.reset_index()

    # -- Forward/back fill categoricals for inserted rows only
    for cat_col in ["network_type", "ping_target"]:
        if cat_col in merged.columns:
            merged[cat_col] = merged[cat_col].ffill().bfill()

    # -- Interpolate numeric values
    merged, filled_mask = interpolate_cols(merged, numeric_present)

    # -- Build flag columns
    # had_empty_measurement_in_raw: row existed AND had NaN in raw
    # (inserted rows don't count here)
    merged["had_empty_measurement_in_raw"] = False
    orig_idx = merged["timestamp"].isin(
        pd.DatetimeIndex(merged["timestamp"]) .difference(pd.DatetimeIndex(missing_slots))
    )
    # Rebuild had_empty from the pre-merge record
    had_empty_series = dict(zip(
        df.reset_index()["timestamp"].tolist(),
        had_empty_in_raw.tolist()
    ))
    merged["had_empty_measurement_in_raw"] = merged["timestamp"].map(
        lambda ts: had_empty_series.get(ts, False)
    )

    # was_missing_timestamp_row: True for created rows
    merged["was_missing_timestamp_row"] = merged["timestamp"].isin(missing_slots)

    # was_estimated_row: True if interpolation actually filled a value in this row
    # OR if this row was a created (missing) row
    merged["was_estimated_row"] = (
        pd.Series(filled_mask, index=merged.index) | merged["was_missing_timestamp_row"]
    )

    # -- Ensure bool dtype
    for fc in ["had_empty_measurement_in_raw", "was_missing_timestamp_row", "was_estimated_row"]:
        merged[fc] = merged[fc].astype(bool)

    # -- Remove fully-empty columns
    removed = [c for c in merged.columns if merged[c].isna().all()
               and c not in ["had_empty_measurement_in_raw","was_missing_timestamp_row","was_estimated_row"]]
    if removed:
        merged.drop(columns=removed, inplace=True)
        rep.append(f"\nFully-empty columns removed: {removed}")

    # -- Final report numbers
    rep.append(f"\nRows with had_empty_measurement_in_raw=True : {merged['had_empty_measurement_in_raw'].sum()}")
    rep.append(f"Rows with was_missing_timestamp_row=True    : {merged['was_missing_timestamp_row'].sum()}")
    rep.append(f"Rows with was_estimated_row=True            : {merged['was_estimated_row'].sum()}")
    rep.append(f"\nFinal rows    : {len(merged)}")
    rep.append(f"Final columns : {len(merged.columns)}")
    rep.append(f"\nRemaining NaN per column after interpolation:")
    for c in numeric_present:
        n = int(merged[c].isna().sum())
        if n:
            rep.append(f"  {c}: {n}")

    rep.append(f"\nRemaining limitations:")
    rep.append(f"  - 13 consecutive slots missing (2026-05-07 08:30 to 11:30) filled by interpolation")
    rep.append(f"  - Weather data during that gap is also interpolated")
    rep.append(f"  - speedtest_server_name / speedtest_server_location not normalised further")

    # -- Validate
    assert merged["timestamp"].nunique() == len(merged), "Duplicate timestamps remain!"
    assert not merged[numeric_present].isna().any().any(), "NaN values remain after interpolation!"

    # -- Save
    merged.to_csv(STARLINK_OUT, index=False)
    rep.append(f"\nSaved to: {STARLINK_OUT}")
    with open(SL_REPORT, "w") as f:
        f.write("\n".join(rep))

    print(f"Starlink done: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"  was_missing_timestamp_row=True : {merged['was_missing_timestamp_row'].sum()}")
    print(f"  was_estimated_row=True         : {merged['was_estimated_row'].sum()}")
    print(f"  had_empty_measurement_in_raw   : {merged['had_empty_measurement_in_raw'].sum()}")
    return merged


# -----------------------------------------------------------------------
# CLEAN AWASR
# -----------------------------------------------------------------------

def clean_awasr():
    rep = ["AWASR CLEANING REPORT", "=" * 60]

    # -- Load
    df = pd.read_csv(AWASR_IN)
    orig_rows = len(df)
    orig_cols = len(df.columns)
    rep.append(f"\nOriginal rows    : {orig_rows}")
    rep.append(f"Original columns : {orig_cols}")

    # -- Fix known bad timestamps
    ts_fixes = {
        "cat":                None,
        "0":                  None,
        "026-05-06T16:15:02": "2026-05-06T16:15:02",
        "026-05-10T09:30:02": "2026-05-10T09:30:02",
        "026-05-11T13:45:05": "2026-05-11T13:45:05",
    }
    dropped_bad_ts = 0
    for bad, good in ts_fixes.items():
        mask = df["timestamp"] == bad
        if mask.any():
            if good:
                df.loc[mask, "timestamp"] = good
                rep.append(f"  Fixed: '{bad}' -> '{good}' ({int(mask.sum())} row(s))")
            else:
                df.loc[mask, "timestamp"] = np.nan
                dropped_bad_ts += int(mask.sum())
                rep.append(f"  Dropped (unfixable): '{bad}' ({int(mask.sum())} row(s))")

    df = df.dropna(subset=["timestamp"])
    rep.append(f"Rows dropped (unfixable timestamp): {dropped_bad_ts}")

    # -- Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -- Standardise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # -- Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    exact_dupes = before - len(df)
    rep.append(f"Exact duplicate rows removed: {exact_dupes}")

    # -- Fix corrupt upload
    df, corrupt_upload = fix_corrupt_upload(df)
    rep.append(f"Corrupt upload_mbps values replaced with NaN: {corrupt_upload}")

    # -- Coerce numerics
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df = coerce_numerics(df, numeric_present)

    # -- Round each sub-minute timestamp to nearest 15-min slot
    df["slot"] = df["timestamp"].dt.round("15min")

    rows_before_agg = len(df)
    unique_slots = df["slot"].nunique()
    rep.append(f"\nRows before aggregation : {rows_before_agg}")
    rep.append(f"Unique 15-min slots     : {unique_slots}")

    # -- Track which slots had any NaN in raw (before aggregation)
    # A slot had an empty measurement if any of its contributing rows had NaN
    slot_had_empty = df.groupby("slot")[numeric_present].apply(
        lambda g: g.isna().any().any()
    )

    # -- Aggregate: median per slot for numerics, mode for categoricals
    agg_num = df.groupby("slot")[numeric_present].median()

    cat_cols = [c for c in ["network_type", "ping_target"] if c in df.columns]
    def safe_mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) > 0 else np.nan
    agg_cat = df.groupby("slot")[cat_cols].agg(safe_mode)

    agg = agg_num.join(agg_cat)
    agg.index.name = "timestamp"

    rep.append(f"Rows after aggregation (one per slot): {len(agg)}")
    rep.append(f"Aggregation: numeric=median, categorical=mode")

    # -- Slots where speedtest was absent entirely (not truly 'empty', just ping-only)
    slots_no_speedtest = int(agg["download_mbps"].isna().sum()) if "download_mbps" in agg.columns else 0
    rep.append(f"Slots with no speedtest measurement (ping-only): {slots_no_speedtest}")
    rep.append(f"  (These are treated as had_empty_measurement_in_raw=True since speedtest cols are NaN)")

    # -- Build 15-min grid aligned to Starlink range
    grid_start = agg.index.min()
    grid_end   = agg.index.max()
    grid = pd.DataFrame({"timestamp": make_15min_grid(grid_start, grid_end)})
    grid = grid.set_index("timestamp")

    merged = grid.join(agg, how="left")

    # -- Identify truly missing slots (no rows at all in raw)
    was_missing_ts = merged["ping_avg_rtt_ms"].isna()
    missing_slots = merged.index[was_missing_ts].tolist()
    rep.append(f"\nMissing timestamp rows created: {len(missing_slots)}")
    for s in missing_slots:
        rep.append(f"  {s}")

    merged = merged.reset_index()

    # -- Fill categoricals for inserted rows
    for cat_col in cat_cols:
        if cat_col in merged.columns:
            merged[cat_col] = merged[cat_col].ffill().bfill()

    # -- Interpolate numeric values
    merged, filled_mask = interpolate_cols(merged, numeric_present)

    # -- Build flag columns
    # had_empty_measurement_in_raw:
    #   True if the slot existed in raw AND (any raw row for that slot had NaN
    #   OR the slot had no speedtest data at all)
    slot_had_empty_dict = slot_had_empty.to_dict()
    merged["had_empty_measurement_in_raw"] = merged["timestamp"].map(
        lambda ts: slot_had_empty_dict.get(ts, False)
    )

    # was_missing_timestamp_row: created rows
    merged["was_missing_timestamp_row"] = merged["timestamp"].isin(missing_slots)

    # was_estimated_row: interpolation actually ran on this row OR it was created
    merged["was_estimated_row"] = (
        pd.Series(filled_mask, index=merged.index) | merged["was_missing_timestamp_row"]
    )

    # Ensure bool dtype
    for fc in ["had_empty_measurement_in_raw", "was_missing_timestamp_row", "was_estimated_row"]:
        merged[fc] = merged[fc].astype(bool)

    # -- Remove fully-empty columns
    removed = [c for c in merged.columns if merged[c].isna().all()
               and c not in ["had_empty_measurement_in_raw","was_missing_timestamp_row","was_estimated_row"]]
    if removed:
        merged.drop(columns=removed, inplace=True)
        rep.append(f"\nFully-empty columns removed: {removed}")

    # -- Final report
    rep.append(f"\nRows with had_empty_measurement_in_raw=True : {merged['had_empty_measurement_in_raw'].sum()}")
    rep.append(f"Rows with was_missing_timestamp_row=True    : {merged['was_missing_timestamp_row'].sum()}")
    rep.append(f"Rows with was_estimated_row=True            : {merged['was_estimated_row'].sum()}")
    rep.append(f"\nFinal rows    : {len(merged)}")
    rep.append(f"Final columns : {len(merged.columns)}")
    rep.append(f"\nRemaining NaN per column after interpolation:")
    for c in numeric_present:
        n = int(merged[c].isna().sum())
        if n:
            rep.append(f"  {c}: {n}")

    rep.append(f"\nRemaining limitations:")
    rep.append(f"  - Speedtest was absent in {slots_no_speedtest} of {len(merged)} slots (~{slots_no_speedtest*100//len(merged)}%)")
    rep.append(f"    Values filled by time-based interpolation across those gaps")
    rep.append(f"  - network_type labelled 'Awaser' in source data; preserved as-is")
    rep.append(f"  - upload_mbps had {corrupt_upload} corrupt artefact values removed before aggregation")

    # -- Validate
    assert merged["timestamp"].nunique() == len(merged), "Duplicate timestamps remain!"
    assert not merged[numeric_present].isna().any().any(), "NaN values remain after interpolation!"

    # -- Save
    merged.to_csv(AWASR_OUT, index=False)
    rep.append(f"\nSaved to: {AWASR_OUT}")
    with open(AW_REPORT, "w") as f:
        f.write("\n".join(rep))

    print(f"Awasr done: {len(merged)} rows, {len(merged.columns)} columns")
    print(f"  was_missing_timestamp_row=True : {merged['was_missing_timestamp_row'].sum()}")
    print(f"  was_estimated_row=True         : {merged['was_estimated_row'].sum()}")
    print(f"  had_empty_measurement_in_raw   : {merged['had_empty_measurement_in_raw'].sum()}")
    return merged


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Cleaning Starlink...")
    sl = clean_starlink()

    print("\nCleaning Awasr...")
    aw = clean_awasr()

    print()
    print("=" * 50)
    print("CROSS-VALIDATION")
    print("=" * 50)
    print(f"Starlink rows : {len(sl)}")
    print(f"Awasr rows    : {len(aw)}")
    print(f"Row counts match : {len(sl) == len(aw)}")
    print(f"Starlink start   : {sl['timestamp'].iloc[0]}")
    print(f"Awasr start      : {aw['timestamp'].iloc[0]}")
    print(f"Starlink end     : {sl['timestamp'].iloc[-1]}")
    print(f"Awasr end        : {aw['timestamp'].iloc[-1]}")
    print(f"Timestamps identical : {(sl['timestamp'].values == aw['timestamp'].values).all()}")

    # Sample rows to verify flags make sense
    print()
    print("Starlink - sample estimated rows:")
    print(sl[sl["was_estimated_row"]][["timestamp","ping_avg_rtt_ms","was_missing_timestamp_row","was_estimated_row"]].head(5).to_string())
    print()
    print("Starlink - sample non-estimated rows (should be all False):")
    print(sl[~sl["was_estimated_row"]][["timestamp","ping_avg_rtt_ms","had_empty_measurement_in_raw","was_missing_timestamp_row","was_estimated_row"]].head(5).to_string())
