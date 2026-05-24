import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================
# LOAD DATA
# ======================
df = pd.read_csv("Cleaned/experiment_A/starlink_forecast_v2.csv")

# Remove missing values
df = df.dropna().reset_index(drop=True)

# Make sure target exists
if "target" not in df.columns:
    raise ValueError("The dataset must contain a column named 'target'.")


# ======================
# FEATURES AND TARGET
# ======================
X = df.drop(columns=["target"])
y = df["target"]

# Keep numeric columns only
X = X.select_dtypes(include=[np.number])

if X.empty:
    raise ValueError("No numeric feature columns found.")


# ======================
# TIME-BASED SPLIT
# ======================
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ======================
# HELPER FUNCTION
# ======================
def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.4f}")

    return mae, rmse, r2


# ======================
# 1. NAIVE BASELINE
# ======================
# Naive baseline means: predict the next latency using the most recent previous latency.
# This uses the previous actual y value as prediction.

naive_pred = y_test.shift(1)

# First test row has no previous test value, so use last training value
naive_pred.iloc[0] = y_train.iloc[-1]

evaluate_model("Naive Baseline", y_test, naive_pred)


# ======================
# 2. LINEAR REGRESSION
# ======================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

evaluate_model("Linear Regression", y_test, lr_pred)


# ======================
# 3. RANDOM FOREST
# ======================
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

evaluate_model("Random Forest", y_test, rf_pred)


# ======================
# 4. OPTIONAL XGBOOST
# ======================
try:
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)

    evaluate_model("XGBoost", y_test, xgb_pred)

except ModuleNotFoundError:
    print("\nXGBoost skipped because xgboost is not installed.")
    print("This is okay if you are not using XGBoost in the final table.")