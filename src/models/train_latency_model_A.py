import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "Cleaned/experiment_A/starlink_forecast_v2.csv"

OUTPUT_DIR = "Cleaned/experiment_A"
MODEL_DIR = "Models"

TARGET_COL = "target"
MAIN_LAG_COL = "ping_avg_rtt_ms_lag_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# Load Experiment A only
df = pd.read_csv(DATA_FILE)

# Convert all columns to numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Remove missing rows
df = df.dropna().reset_index(drop=True)

# Basic sanity filters
for col in [
    "target",
    "ping_avg_rtt_ms_lag_1",
    "ping_avg_rtt_ms_lag_2",
    "ping_avg_rtt_ms_lag_3",
    "ping_avg_rtt_ms_lag_4",
]:
    df = df[(df[col] > 0) & (df[col] < 500)]

df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
df = df[(df["upload_lag_1"] > 0) & (df["upload_lag_1"] < 500)]
df = df[(df["jitter_lag_1"] >= 0) & (df["jitter_lag_1"] < 100)].reset_index(drop=True)

# Save cleaned Experiment A forecasting dataset used for verification
df.to_csv(os.path.join(OUTPUT_DIR, "starlink_forecast_A_verified.csv"), index=False)


# Features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Total rows:", len(df))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))


# Predictions dictionary
predictions = {}
models = {}

# Naive baseline: previous latency predicts next latency
predictions["Naive Baseline"] = X_test[MAIN_LAG_COL].values


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

models["Linear Regression"] = lr
predictions["Linear Regression"] = lr.predict(X_test)


# Random Forest
rf = RandomForestRegressor(
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

models["Random Forest"] = rf
predictions["Random Forest"] = rf.predict(X_test)


# Evaluate models
rows = []

for name, y_pred in predictions.items():
    metrics = evaluate(y_test, y_pred)
    rows.append({
        "Model": name,
        "MAE": round(metrics["MAE"], 4),
        "RMSE": round(metrics["RMSE"], 4),
        "R2": round(metrics["R2"], 4),
    })

comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

print("\nExperiment A Model Comparison:")
print(comparison)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison_experiment_A.csv"),
    index=False
)


# Save predictions
pred_df = pd.DataFrame({
    "actual": y_test.values,
    "naive_baseline_pred": predictions["Naive Baseline"],
    "linear_regression_pred": predictions["Linear Regression"],
    "random_forest_pred": predictions["Random Forest"],
})

pred_df.to_csv(
    os.path.join(OUTPUT_DIR, "starlink_predictions_experiment_A.csv"),
    index=False
)


# Save best ML model only, excluding Naive Baseline
ml_comparison = comparison[comparison["Model"] != "Naive Baseline"].reset_index(drop=True)

best_ml_name = ml_comparison.iloc[0]["Model"]
best_model = models[best_ml_name]

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "starlink_latency_forecast_model_experiment_A.pkl")
)

joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR, "starlink_latency_features_experiment_A.pkl")
)

pd.DataFrame([{
    "dataset": "Experiment A only",
    "best_ml_model": best_ml_name,
    "rows_total": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "split_type": "chronological_80_20",
}]).to_csv(
    os.path.join(MODEL_DIR, "starlink_latency_model_metadata_experiment_A.csv"),
    index=False
)

print("\nSaved Experiment A model:", best_ml_name)