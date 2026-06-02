"""
train_latency_model_combined.py
--------------------------------
Trains and evaluates Naive Baseline, Linear Regression, Random Forest,
and XGBoost on the combined Starlink forecast dataset (Exp A + Exp B).
Saves the best ML model (by RMSE) as the advisor/dashboard model.

Verified combined-dataset results (starlink_forecast_combined.csv, 4050 rows):
  Naive Baseline  MAE=5.9457  RMSE=9.0529  R2=-0.9763
  Linear Regression MAE=4.3367  RMSE=6.4660  R2=-0.0082  <- selected
  Random Forest   MAE=4.5213  RMSE=6.5591  R2=-0.0375
  XGBoost         MAE=4.6288  RMSE=6.8265  R2=-0.1237

Note on R2: Negative R2 does not indicate model failure for strongly
autocorrelated time series (see thesis Section 2.7.5). Even the naive
persistence model produces near-zero or negative R2 when test-set
variance is low relative to prediction error.

XGBoost note: XGBoost exceeded the naive baseline MAE on Experiment A
alone (6.045 ms vs 5.152 ms) and was excluded from that pipeline.
On the combined dataset it recovers to 4.629 ms, below the naive
baseline of 5.946 ms, and is retained as a dashboard model option.
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
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    print("XGBoost will be skipped in this run.")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE_1 = "Cleaned/experiment_A/starlink_forecast_v2.csv"
DATA_FILE_2 = "Cleaned/experiment_B/starlink_2_forecast.csv"
OUTPUT_DIR  = "Cleaned/combined"
MODEL_DIR   = "Models"
TARGET_COL  = "target"
MAIN_LAG    = "ping_avg_rtt_ms_lag_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


# ── Load and combine ───────────────────────────────────────────────────────────
df1 = pd.read_csv(DATA_FILE_1)
df2 = pd.read_csv(DATA_FILE_2)
common_cols = [c for c in df1.columns if c in set(df2.columns)]
df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True).drop_duplicates()

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# Sanity filters (same as Experiment A training script)
for col in [TARGET_COL, "ping_avg_rtt_ms_lag_1", "ping_avg_rtt_ms_lag_2",
            "ping_avg_rtt_ms_lag_3", "ping_avg_rtt_ms_lag_4"]:
    df = df[(df[col] > 0) & (df[col] < 500)]
df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
df = df[(df["upload_lag_1"]   > 0) & (df["upload_lag_1"]   < 500)]
df = df[(df["jitter_lag_1"]  >= 0) & (df["jitter_lag_1"]   < 100)].reset_index(drop=True)

# Save the combined forecast dataset for reproducibility
combined_path = os.path.join(OUTPUT_DIR, "starlink_forecast_combined.csv")
df.to_csv(combined_path, index=False)
print(f"Combined dataset saved: {combined_path}  ({len(df)} rows)")

# ── Split ──────────────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\nTotal rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Features:   {list(X.columns)}\n")


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

# XGBoost (excluded from Exp A pipeline; recovered on combined dataset)
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
    print(f"  {name:25s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_combined.csv"), index=False)
print(f"\nModel comparison saved to {OUTPUT_DIR}/model_comparison_combined.csv")


# ── Save predictions ───────────────────────────────────────────────────────────
pred_df = pd.DataFrame({"actual": y_test.values})
for name, y_pred in predictions.items():
    pred_df[name.lower().replace(" ", "_") + "_pred"] = y_pred
pred_df.to_csv(os.path.join(OUTPUT_DIR, "starlink_predictions_combined.csv"), index=False)
print(f"Predictions saved to   {OUTPUT_DIR}/starlink_predictions_combined.csv")


# ── Save best ML model (by RMSE, excluding naive baseline) ────────────────────
ml_comparison = comparison[comparison["Model"] != "Naive Baseline"].reset_index(drop=True)
best_name  = ml_comparison.iloc[0]["Model"]
best_model = models[best_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "starlink_latency_forecast_model.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "starlink_latency_features.pkl"))

pd.DataFrame([{
    "best_model_for_advisor": best_name,
    "rows_combined":          len(df),
    "train_rows":             len(X_train),
    "test_rows":              len(X_test),
    "split_type":             "chronological_80_20",
    "xgboost_available":      XGB_OK,
}]).to_csv(os.path.join(MODEL_DIR, "starlink_latency_model_metadata.csv"), index=False)

print(f"\nSaved advisor model:  {best_name}")
print(f"Model file:           {MODEL_DIR}/starlink_latency_forecast_model.pkl")