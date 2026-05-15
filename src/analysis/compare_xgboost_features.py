import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ======================
# LOAD DATASETS
# ======================
# Before Feature Engineering — archived snapshot
df_old = pd.read_csv("archive/old_data/starlink_forecast_clean.csv")

# After Feature Engineering — current Experiment A feature set
df_new = pd.read_csv("Cleaned/experiment_A/starlink_forecast.csv")


# ======================
# SPLIT DATA (TIME SERIES)
# ======================
def split_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


X_train_old, X_test_old, y_train_old, y_test_old = split_data(df_old)
X_train_new, X_test_new, y_train_new, y_test_new = split_data(df_new)


# ======================
# TRAIN XGBOOST MODELS
# ======================
model_old = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model_old.fit(X_train_old, y_train_old)

model_new = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model_new.fit(X_train_new, y_train_new)


# ======================
# PREDICTIONS
# ======================
y_pred_old = model_old.predict(X_test_old)
y_pred_new = model_new.predict(X_test_new)


# ======================
# PLOT
# ======================
plt.figure(figsize=(14, 6))

# Actual

plt.plot(y_test_old.values, label="Actual", color="black", linewidth=2, alpha=0.6)

# Before FE
plt.plot(y_pred_old, label="XGBoost (Before FE)", color="blue", linestyle="--", linewidth=2)

# After FE

plt.plot(y_pred_new, label="XGBoost (After FE)", color="red", linewidth=2)

plt.xlabel("Test Time Steps")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("figures/xgboost_before_vs_after_fe.png")
plt.show()