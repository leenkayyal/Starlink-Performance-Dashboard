"""
train_latency_model_A.py
--------------------------
Trains and evaluates Naive Baseline, Linear Regression, Random Forest,
and XGBoost on the Experiment A Starlink forecast dataset only.

Verified Experiment A results (starlink_forecast_v2.csv, 1988 rows):
  Naive Baseline    MAE=5.152  RMSE=11.494  R2=+0.019
  Linear Regression MAE=4.741  RMSE=11.906  R2=-0.052  <- selected primary
  Random Forest     MAE=5.364  RMSE=12.436  R2=-0.148
  XGBoost           MAE=6.045  RMSE=14.261  R2=-0.510  <- EXCLUDED (exceeds naive baseline)

XGBoost is evaluated here for documentation purposes but is excluded from
the Experiment A pipeline because its MAE (6.045 ms) exceeds the naive
baseline (5.152 ms). It is re-evaluated on the combined dataset in
train_latency_model_combined.py where it recovers below the naive baseline.

Note on R2: Negative R2 does not indicate model failure for strongly
autocorrelated time series (thesis Section 2.7.5).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("WARNING: xgboost not installed. XGBoost evaluation will be skipped.")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = "Cleaned/experiment_A/starlink_forecast_v2.csv"
OUTPUT_DIR = "Cleaned/experiment_A"
MODEL_DIR  = "Models"
TARGET_COL = "target"
MAIN_LAG   = "ping_avg_rtt_ms_lag_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# Sanity filters
for col in [TARGET_COL, "ping_avg_rtt_ms_lag_1", "ping_avg_rtt_ms_lag_2",
            "ping_avg_rtt_ms_lag_3", "ping_avg_rtt_ms_lag_4"]:
    df = df[(df[col] > 0) & (df[col] < 500)]
df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
df = df[(df["upload_lag_1"]   > 0) & (df["upload_lag_1"]   < 500)]
df = df[(df["jitter_lag_1"]  >= 0) & (df["jitter_lag_1"]   < 100)].reset_index(drop=True)

# Save verified dataset
df.to_csv(os.path.join(OUTPUT_DIR, "starlink_forecast_A_verified.csv"), index=False)

# ── Split ──────────────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Total rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")


# ── Train and evaluate ─────────────────────────────────────────────────────────
predictions = {}
models = {}

# Naive baseline
predictions["Naive Baseline"] = X_test[MAIN_LAG].values

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
models["Linear Regression"] = lr
predictions["Linear Regression"] = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)
models["Random Forest"] = rf
predictions["Random Forest"] = rf.predict(X_test)

# XGBoost - evaluated for comparison, excluded from pipeline on this dataset
if XGB_OK:
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    ).fit(X_train, y_train)
    models["XGBoost"] = xgb
    predictions["XGBoost"] = xgb.predict(X_test)


# ── Results ────────────────────────────────────────────────────────────────────
rows = []
for name, y_pred in predictions.items():
    m = evaluate(y_test, y_pred)
    rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})

comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
print("\nExperiment A Model Comparison:")
print(comparison.to_string(index=False))

# Check if XGBoost exceeds naive baseline (expected: yes)
naive_mae = comparison.loc[comparison["Model"] == "Naive Baseline", "MAE"].values[0]
if XGB_OK:
    xgb_mae = comparison.loc[comparison["Model"] == "XGBoost", "MAE"].values[0]
    if xgb_mae > naive_mae:
        print(f"\nNOTE: XGBoost MAE ({xgb_mae:.4f}) > Naive Baseline MAE ({naive_mae:.4f}).")
        print("      XGBoost is EXCLUDED from the Experiment A pipeline.")
        print("      See train_latency_model_combined.py for combined-dataset re-evaluation.")

comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_experiment_A.csv"), index=False)


# ── Save predictions ───────────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "actual":                   y_test.values,
    "naive_baseline_pred":      predictions["Naive Baseline"],
    "linear_regression_pred":   predictions["Linear Regression"],
    "random_forest_pred":       predictions["Random Forest"],
})
if XGB_OK:
    pred_df["xgboost_pred"] = predictions["XGBoost"]
pred_df.to_csv(os.path.join(OUTPUT_DIR, "starlink_predictions_experiment_A.csv"), index=False)


# ── Save best ML model (LR or RF, NOT XGBoost on this dataset) ────────────────
# Exclude both Naive Baseline and XGBoost when selecting the pipeline model
exclude = ["Naive Baseline"]
if XGB_OK:
    xgb_mae = comparison.loc[comparison["Model"] == "XGBoost", "MAE"].values[0]
    if xgb_mae > naive_mae:
        exclude.append("XGBoost")

ml_comparison = comparison[~comparison["Model"].isin(exclude)].reset_index(drop=True)
best_name  = ml_comparison.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "starlink_latency_forecast_model_experiment_A.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "starlink_latency_features_experiment_A.pkl"))

pd.DataFrame([{
    "dataset":          "Experiment A only",
    "best_ml_model":    best_name,
    "rows_total":       len(df),
    "train_rows":       len(X_train),
    "test_rows":        len(X_test),
    "split_type":       "chronological_80_20",
    "xgboost_excluded": XGB_OK and ("XGBoost" in exclude),
}]).to_csv(os.path.join(MODEL_DIR, "starlink_latency_model_metadata_experiment_A.csv"), index=False)

print(f"\nSaved Experiment A model: {best_name}")