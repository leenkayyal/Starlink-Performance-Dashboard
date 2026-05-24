import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ======================
# CONFIG
# ======================
DATA_FILE = "Cleaned/starlink_forecast_v2.csv"
OUTPUT_DIR = "Cleaned"

TARGET_COL = "target"
MAIN_LAG_COL = "ping_avg_rtt_ms_lag_1"


# ======================
# HELPERS
# ======================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred)
    }


def save_dataframe(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ======================
# TRAIN / TEST SPLIT
# ======================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ======================
# 1. NAIVE BASELINE
# ======================
y_pred_naive = X_test[MAIN_LAG_COL].values
metrics_naive = evaluate_model(y_test, y_pred_naive)


# ======================
# 2. LINEAR REGRESSION
# ======================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
metrics_lr = evaluate_model(y_test, y_pred_lr)


# ======================
# 3. RANDOM FOREST
# ======================
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
metrics_rf = evaluate_model(y_test, y_pred_rf)


# ======================
# 4. XGBOOST (OPTIONAL)
# ======================
if XGB_AVAILABLE:
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    metrics_xgb = evaluate_model(y_test, y_pred_xgb)
else:
    y_pred_xgb = None
    metrics_xgb = None


# ======================
# PRINT RESULTS
# ======================
print("\n=== MODEL RESULTS ===")

print("\nNaive Baseline")
print("MAE :", round(metrics_naive["MAE"], 3))
print("RMSE:", round(metrics_naive["RMSE"], 3))

print("\nLinear Regression")
print("MAE :", round(metrics_lr["MAE"], 3))
print("RMSE:", round(metrics_lr["RMSE"], 3))

print("\nRandom Forest")
print("MAE :", round(metrics_rf["MAE"], 3))
print("RMSE:", round(metrics_rf["RMSE"], 3))

if XGB_AVAILABLE:
    print("\nXGBoost")
    print("MAE :", round(metrics_xgb["MAE"], 3))
    print("RMSE:", round(metrics_xgb["RMSE"], 3))
else:
    print("\nXGBoost")
    print("Not installed in this environment.")


# ======================
# SAVE MODEL COMPARISON TABLE
# ======================
comparison_rows = [
    {
        "Model": "Naive Baseline",
        "MAE": round(metrics_naive["MAE"], 3),
        "RMSE": round(metrics_naive["RMSE"], 3)
    },
    {
        "Model": "Linear Regression",
        "MAE": round(metrics_lr["MAE"], 3),
        "RMSE": round(metrics_lr["RMSE"], 3)
    },
    {
        "Model": "Random Forest",
        "MAE": round(metrics_rf["MAE"], 3),
        "RMSE": round(metrics_rf["RMSE"], 3)
    }
]

if XGB_AVAILABLE:
    comparison_rows.append(
        {
            "Model": "XGBoost",
            "MAE": round(metrics_xgb["MAE"], 3),
            "RMSE": round(metrics_xgb["RMSE"], 3)
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_path = save_dataframe(comparison_df, "model_comparison_v2.csv")


# ======================
# SAVE PREDICTIONS
# ======================
predictions_df = pd.DataFrame({
    "actual": y_test.values,
    "naive_pred": y_pred_naive,
    "linear_regression_pred": y_pred_lr,
    "random_forest_pred": y_pred_rf
})

if XGB_AVAILABLE:
    predictions_df["xgboost_pred"] = y_pred_xgb

predictions_path = save_dataframe(predictions_df, "starlink_predictions_v2.csv")


# ======================
# PLOT 1: ALL MODELS
# ======================
plt.figure(figsize=(14, 6))

plt.plot(y_test.values, label="Actual", color="black", linewidth=2, alpha=0.6)
plt.plot(y_pred_lr, label="Linear Regression", color="blue", linewidth=2)
plt.plot(y_pred_rf, label="Random Forest", color="green", linewidth=2)

if XGB_AVAILABLE:
    plt.plot(y_pred_xgb, label="XGBoost", color="red", linewidth=2)

plt.title("Actual vs Predicted Latency (Model Comparison)")
plt.xlabel("Test Time Steps")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_all_path = os.path.join(OUTPUT_DIR, "latency_model_comparison_v2.png")
plt.savefig(plot_all_path, dpi=300)
plt.show()


# ======================
# PLOT 2: LINEAR REGRESSION ONLY
# ======================
plt.figure(figsize=(14, 6))

plt.plot(y_test.values, label="Actual", color="black", linewidth=2, alpha=0.6)
plt.plot(y_pred_lr, label="Linear Regression", color="blue", linewidth=2)

plt.title("Actual vs Predicted Latency (Linear Regression)")
plt.xlabel("Test Time Steps")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_lr_path = os.path.join(OUTPUT_DIR, "latency_linear_regression_v2.png")
plt.savefig(plot_lr_path, dpi=300)
plt.show()


# ======================
# PLOT 3: XGBOOST ONLY (IF AVAILABLE)
# ======================
if XGB_AVAILABLE:
    plt.figure(figsize=(14, 6))

    plt.plot(y_test.values, label="Actual", color="black", linewidth=2, alpha=0.6)
    plt.plot(y_pred_xgb, label="XGBoost", color="red", linewidth=2)

    plt.title("Actual vs Predicted Latency (XGBoost)")
    plt.xlabel("Test Time Steps")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_xgb_path = os.path.join(OUTPUT_DIR, "latency_xgboost_v2.png")
    plt.savefig(plot_xgb_path, dpi=300)
    plt.show()


# ======================
# FINAL OUTPUT
# ======================
print("\nSaved files:")
print("-", comparison_path)
print("-", predictions_path)
print("-", plot_all_path)
print("-", plot_lr_path)
if XGB_AVAILABLE:
    print("-", plot_xgb_path)