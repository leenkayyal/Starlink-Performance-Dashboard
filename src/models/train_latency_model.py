import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE_1 = "Cleaned/experiment_A/starlink_forecast_v2.csv"
DATA_FILE_2 = "Cleaned/experiment_B/starlink_2_forecast.csv"
OUTPUT_DIR = "Cleaned/combined"
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

df1 = pd.read_csv(DATA_FILE_1)
df2 = pd.read_csv(DATA_FILE_2)
common_cols = [c for c in df1.columns if c in set(df2.columns)]
df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True).drop_duplicates()
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

# Basic sanity filters
for col in ["target", "ping_avg_rtt_ms_lag_1", "ping_avg_rtt_ms_lag_2", "ping_avg_rtt_ms_lag_3", "ping_avg_rtt_ms_lag_4"]:
    df = df[(df[col] > 0) & (df[col] < 500)]
df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
df = df[(df["upload_lag_1"] > 0) & (df["upload_lag_1"] < 500)]
df = df[(df["jitter_lag_1"] >= 0) & (df["jitter_lag_1"] < 100)].reset_index(drop=True)

df.to_csv(os.path.join(OUTPUT_DIR, "starlink_forecast_combined.csv"), index=False)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

predictions = {}
models = {}
predictions["Naive Baseline"] = X_test[MAIN_LAG_COL].values

lr = LinearRegression().fit(X_train, y_train)
models["Linear Regression"] = lr
predictions["Linear Regression"] = lr.predict(X_test)

rf = RandomForestRegressor(
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
).fit(X_train, y_train)
models["Random Forest"] = rf
predictions["Random Forest"] = rf.predict(X_test)

rows = []
for name, y_pred in predictions.items():
    metrics = evaluate(y_test, y_pred)
    rows.append({"Model": name, **{k: round(v, 4) for k, v in metrics.items()}})
comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_combined.csv"), index=False)
print(comparison)

pred_df = pd.DataFrame({"actual": y_test.values})
for name, y_pred in predictions.items():
    pred_df[name.lower().replace(" ", "_") + "_pred"] = y_pred
pred_df.to_csv(os.path.join(OUTPUT_DIR, "starlink_predictions_combined.csv"), index=False)

best_ml_name = comparison[comparison["Model"] != "Naive Baseline"].iloc[0]["Model"]
best_model = models[best_ml_name]
joblib.dump(best_model, os.path.join(MODEL_DIR, "starlink_latency_forecast_model.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "starlink_latency_features.pkl"))

pd.DataFrame([{
    "best_model_for_advisor": best_ml_name,
    "rows_combined": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
}]).to_csv(os.path.join(MODEL_DIR, "starlink_latency_model_metadata.csv"), index=False)
print("Saved advisor model:", best_ml_name)
