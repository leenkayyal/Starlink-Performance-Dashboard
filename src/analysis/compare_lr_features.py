import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_linear_regression(df):
    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return y_test.reset_index(drop=True), y_pred, mae, rmse


# ======================
# LOAD DATASETS
# ======================
# "Before FE" uses the archived pre-feature-engineering snapshot.
# "After FE" uses the current feature-engineered set from Experiment A.
df_old = pd.read_csv("archive/old_data/starlink_forecast_clean.csv")
df_new = pd.read_csv("Cleaned/experiment_A/starlink_forecast.csv")

# ======================
# TRAIN BOTH MODELS
# ======================
y_test_old, y_pred_old, mae_old, rmse_old = train_linear_regression(df_old)
y_test_new, y_pred_new, mae_new, rmse_new = train_linear_regression(df_new)

# ======================
# PRINT RESULTS
# ======================
print("=== LINEAR REGRESSION COMPARISON ===")
print("\nBefore Feature Engineering")
print("MAE :", round(mae_old, 3))
print("RMSE:", round(rmse_old, 3))

print("\nAfter Feature Engineering")
print("MAE :", round(mae_new, 3))
print("RMSE:", round(rmse_new, 3))

# ======================
# ALIGN LENGTHS FOR PLOTTING
# ======================
min_len = min(len(y_test_old), len(y_test_new))

y_test_plot = y_test_new.iloc[:min_len]
y_pred_old_plot = y_pred_old[:min_len]
y_pred_new_plot = y_pred_new[:min_len]

# ======================
# PLOT
# ======================
plt.figure(figsize=(14, 6))

plt.plot(y_test_plot.values, label="Actual", color="black", linewidth=2, alpha=0.6)
plt.plot(y_pred_old_plot, label="Linear Regression (Before FE)", color="blue", linestyle="--", linewidth=2)
plt.plot(y_pred_new_plot, label="Linear Regression (After FE)", color="red", linewidth=2)

plt.title("Linear Regression: Before vs After Feature Engineering")
plt.xlabel("Test Time Steps")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("figures/lr_before_vs_after_fe.png")
plt.show()