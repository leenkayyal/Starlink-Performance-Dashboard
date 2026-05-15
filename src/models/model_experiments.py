"""
model_experiments.py
====================
Extended model evaluation for Starlink latency forecasting.
Runs three experiments on top of the existing baseline:

  1. Extended lag window  — lags 1-12 (3 hours) vs original 1-4 (1 hour)
  2. LSTM                 — sequence model trained on a sliding window of 12 steps
  3. Spike classifier     — binary model: will latency exceed SPIKE_THRESHOLD in next step?

Run from the project root:
    python model_experiments.py

Outputs:
    results/model_comparison_extended.csv   — MAE / RMSE table for all models
    results/spike_classifier_report.txt     — classification report + confusion matrix
    results/lstm_predictions.csv            — actual vs predicted (LSTM)
    results/extended_lag_predictions.csv    — actual vs predicted (extended lag LR)

Author: Thesis experiment script — GUtech 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_OK = True
except ImportError:
    KERAS_OK = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLEAN_FILE      = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
FORECAST_FILE   = "Cleaned/experiment_A/starlink_forecast.csv"
RESULTS_DIR     = "results"
SPIKE_THRESHOLD = 50.0   # ms — latency above this is a "spike"
SHORT_LAGS      = 4      # original lag window
LONG_LAGS       = 12     # extended lag window (3 hours at 15-min intervals)
LSTM_WINDOW     = 12     # look-back steps for LSTM
LSTM_EPOCHS     = 60
LSTM_BATCH      = 32
TRAIN_RATIO     = 0.80

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_clean():
    df = pd.read_csv(CLEAN_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False].copy()

    for col in ["ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps",
                "upload_mbps", "weather_code", "temperature_c",
                "humidity_percent", "wind_speed_mps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["ping_avg_rtt_ms"] > 0) & (df["ping_avg_rtt_ms"] < 500)]
    df = df[(df["ping_jitter_ms"] >= 0) & (df["ping_jitter_ms"] < 100)]
    df = df[(df["download_mbps"] > 0) & (df["download_mbps"] < 1000)]
    df = df[(df["upload_mbps"] > 0) & (df["upload_mbps"] < 500)]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df, n_lags=4):
    """
    Build a supervised dataset from the clean time series.
    n_lags controls how many lag steps to include.
    Target = latency at next 15-min interval.
    """
    d = df.copy()
    lat = "ping_avg_rtt_ms"

    # Temporal
    d["hour"]        = d["timestamp"].dt.hour
    d["day_of_week"] = d["timestamp"].dt.dayofweek
    d["is_weekend"]  = d["day_of_week"].isin([5, 6]).astype(int)
    d["hour_sin"]    = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"]    = np.cos(2 * np.pi * d["hour"] / 24)

    # Lag features for latency
    for lag in range(1, n_lags + 1):
        d[f"lat_lag_{lag}"] = d[lat].shift(lag)

    # Context lags
    d["jitter_lag_1"]   = d["ping_jitter_ms"].shift(1)
    d["dl_lag_1"]       = d["download_mbps"].shift(1)
    d["ul_lag_1"]       = d["upload_mbps"].shift(1)

    # Rolling statistics (window = 3 steps)
    d["roll_mean_3"]    = d[lat].rolling(3).mean()
    d["roll_std_3"]     = d[lat].rolling(3).std()

    # Delta features
    d["delta_1"]        = d[lat].diff(1)
    d["delta_2"]        = d[lat].diff(2)

    # Spike indicator (was there a spike in the last 3 steps?)
    d["recent_spike"]   = (d[lat].rolling(3).max() > SPIKE_THRESHOLD).astype(int)

    # Weather
    for col in ["weather_code", "temperature_c", "humidity_percent", "wind_speed_mps"]:
        if col in d.columns:
            d[col] = d[col].shift(0)   # same-step weather is known

    # Target
    d["target"] = d[lat].shift(-1)

    d = d.dropna().reset_index(drop=True)

    feature_cols = (
        ["hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"]
        + [f"lat_lag_{i}" for i in range(1, n_lags + 1)]
        + ["jitter_lag_1", "dl_lag_1", "ul_lag_1"]
        + ["roll_mean_3", "roll_std_3", "delta_1", "delta_2", "recent_spike"]
        + [c for c in ["weather_code", "temperature_c",
                        "humidity_percent", "wind_speed_mps"] if c in d.columns]
    )
    feature_cols = [c for c in feature_cols if c in d.columns]
    return d, feature_cols


def chronological_split(df, feature_cols, ratio=TRAIN_RATIO):
    split = int(len(df) * ratio)
    X = df[feature_cols].values
    y = df["target"].values
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────
# EXPERIMENT 1 — EXTENDED LAG COMPARISON
# ─────────────────────────────────────────────
def experiment_extended_lags(df):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Extended lag window (4 lags vs 12 lags)")
    print("="*60)

    results = []

    for n_lags, label in [(SHORT_LAGS, "Linear Regression (4 lags — original)"),
                          (LONG_LAGS,  "Linear Regression (12 lags — extended)")]:
        feat_df, feat_cols = build_features(df, n_lags=n_lags)
        Xtr, Xte, ytr, yte = chronological_split(feat_df, feat_cols)

        model = LinearRegression()
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)

        mae  = mean_absolute_error(yte, y_pred)
        rmse = float(np.sqrt(mean_squared_error(yte, y_pred)))

        print(f"  {label}")
        print(f"    MAE : {mae:.4f} ms")
        print(f"    RMSE: {rmse:.4f} ms")
        results.append({"Model": label, "MAE": round(mae, 4), "RMSE": round(rmse, 4)})

        if n_lags == LONG_LAGS:
            pd.DataFrame({"actual": yte, "predicted": y_pred}).to_csv(
                f"{RESULTS_DIR}/extended_lag_predictions.csv", index=False)

    return results


# ─────────────────────────────────────────────
# EXPERIMENT 2 — LSTM
# ─────────────────────────────────────────────
def build_lstm_sequences(series, window=LSTM_WINDOW):
    """
    Converts a 1-D latency series into (X, y) sequences.
    X shape: (n_samples, window, 1)
    y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X)[..., np.newaxis], np.array(y)


def experiment_lstm(df):
    print("\n" + "="*60)
    print("EXPERIMENT 2: LSTM sequence model")
    print("="*60)

    if not KERAS_OK:
        print("  TensorFlow/Keras not available — skipping LSTM.")
        return []

    lat = df["ping_avg_rtt_ms"].values.astype(float)

    # Scale to [0,1] — helps LSTM converge faster on this value range
    scaler = StandardScaler()
    lat_scaled = scaler.fit_transform(lat.reshape(-1, 1)).flatten()

    X, y = build_lstm_sequences(lat_scaled, window=LSTM_WINDOW)

    # Chronological split
    split = int(len(X) * TRAIN_RATIO)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    print(f"  Training sequences : {len(Xtr)}")
    print(f"  Test sequences     : {len(Xte)}")

    # Build model — kept small deliberately because of limited data
    model = Sequential([
        LSTM(32, input_shape=(LSTM_WINDOW, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    history = model.fit(
        Xtr, ytr,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0
    )
    print(f"  Training stopped at epoch {len(history.history['loss'])}")

    y_pred_scaled = model.predict(Xte, verbose=0).flatten()

    # Inverse transform
    yte_orig   = scaler.inverse_transform(yte.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(yte_orig, y_pred_orig)
    rmse = float(np.sqrt(mean_squared_error(yte_orig, y_pred_orig)))

    print(f"  MAE : {mae:.4f} ms")
    print(f"  RMSE: {rmse:.4f} ms")

    pd.DataFrame({"actual": yte_orig, "predicted": y_pred_orig}).to_csv(
        f"{RESULTS_DIR}/lstm_predictions.csv", index=False)

    # Training curve
    plt.figure(figsize=(8, 3))
    plt.plot(history.history["loss"],     label="Train loss", color="#60a5fa")
    plt.plot(history.history["val_loss"], label="Val loss",   color="#f472b6")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (scaled)")
    plt.title("LSTM Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lstm_training_curve.png", dpi=150)
    plt.close()

    return [{"Model": "LSTM (window=12)", "MAE": round(mae, 4), "RMSE": round(rmse, 4)}]


# ─────────────────────────────────────────────
# EXPERIMENT 3 — SPIKE CLASSIFIER
# ─────────────────────────────────────────────
def experiment_spike_classifier(df):
    print("\n" + "="*60)
    print(f"EXPERIMENT 3: Spike classifier (threshold = {SPIKE_THRESHOLD} ms)")
    print("="*60)

    feat_df, feat_cols = build_features(df, n_lags=LONG_LAGS)

    # Binary label: will next-step latency exceed SPIKE_THRESHOLD?
    feat_df["spike_label"] = (feat_df["target"] > SPIKE_THRESHOLD).astype(int)

    spike_rate = feat_df["spike_label"].mean()
    print(f"  Spike rate in dataset: {spike_rate:.1%} of observations")

    X = feat_df[feat_cols].values
    y = feat_df["spike_label"].values
    split = int(len(X) * TRAIN_RATIO)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    classifiers = [
        ("Logistic Regression",       LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Random Forest Classifier",  RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ]
    if XGB_OK:
        # XGBoost handles imbalance via scale_pos_weight
        neg = int((ytr == 0).sum())
        pos = int((ytr == 1).sum())
        ratio = neg / pos if pos > 0 else 1.0
        classifiers.append(
            ("XGBoost Classifier",
             xgb.XGBClassifier(n_estimators=100, max_depth=4, scale_pos_weight=ratio,
                               use_label_encoder=False, eval_metric="logloss", random_state=42))
        )

    report_lines = []
    report_lines.append(f"Spike Classifier Report — threshold = {SPIKE_THRESHOLD} ms\n")
    report_lines.append(f"Spike rate: {spike_rate:.1%} | Train: {len(Xtr)} | Test: {len(Xte)}\n")
    report_lines.append("=" * 60 + "\n")

    best_f1   = 0
    best_name = ""

    for name, clf in classifiers:
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)

        print(f"\n  {name}")
        report = classification_report(yte, y_pred, target_names=["Normal", "Spike"],
                                       zero_division=0)
        print(report)

        report_lines.append(f"\n{name}\n{report}\n")

        cm = confusion_matrix(yte, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Spike"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(f"{RESULTS_DIR}/cm_{safe_name}.png", dpi=150)
        plt.close()

        # Track best by spike recall (we care most about catching real spikes)
        from sklearn.metrics import f1_score
        f1 = f1_score(yte, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1   = f1
            best_name = name

    report_lines.append(f"\nBest spike F1-score: {best_name} ({best_f1:.3f})\n")

    with open(f"{RESULTS_DIR}/spike_classifier_report.txt", "w") as f:
        f.writelines(report_lines)

    print(f"\n  Best spike F1: {best_name} ({best_f1:.3f})")
    return []   # no MAE/RMSE for classifier — separate metric space


# ─────────────────────────────────────────────
# COMPARISON CHART
# ─────────────────────────────────────────────
def plot_comparison(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison — Extended Experiments", fontsize=13, fontweight="bold")

    for ax, metric, color in zip(axes, ["MAE", "RMSE"], ["#60a5fa", "#f472b6"]):
        bars = ax.barh(results_df["Model"], results_df[metric], color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(f"{metric} (ms)")
        ax.set_title(metric)
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison_extended.png", dpi=150)
    plt.close()
    print(f"\n  Saved comparison chart → {RESULTS_DIR}/model_comparison_extended.png")


# ─────────────────────────────────────────────
# ACTUAL VS PREDICTED PLOTS
# ─────────────────────────────────────────────
def plot_predictions(csv_path, title, filename):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["actual"].values,    label="Actual",    color="#06b6d4", linewidth=1.2, alpha=0.9)
    ax.plot(df["predicted"].values, label="Predicted", color="#f472b6", linewidth=1.2,
            linestyle="--", alpha=0.85)
    ax.axhline(SPIKE_THRESHOLD, color="#ef4444", linestyle=":", linewidth=1, label=f"Spike threshold ({SPIKE_THRESHOLD} ms)")
    ax.set_xlabel("Test time steps (15-min intervals)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}", dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading clean dataset ...")
    df = load_clean()
    print(f"  Rows after cleaning: {len(df)}")

    # Run all three experiments
    results = []
    results += experiment_extended_lags(df)
    results += experiment_lstm(df)
    experiment_spike_classifier(df)

    # Also include original baselines for comparison
    print("\n" + "="*60)
    print("Adding original baselines for reference table ...")
    print("="*60)

    orig_df, orig_cols = build_features(df, n_lags=SHORT_LAGS)
    Xtr, Xte, ytr, yte = chronological_split(orig_df, orig_cols)

    # Naive baseline
    lat_lag_col_idx = orig_cols.index("lat_lag_1")
    y_naive = Xte[:, lat_lag_col_idx]
    results.insert(0, {
        "Model": "Naive Baseline",
        "MAE":   round(float(mean_absolute_error(yte, y_naive)), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(yte, y_naive))), 4)
    })

    # RF with original features
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    y_rf = rf.predict(Xte)
    results.insert(2, {
        "Model": "Random Forest (4 lags — original)",
        "MAE":   round(float(mean_absolute_error(yte, y_rf)), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(yte, y_rf))), 4)
    })

    # Summary table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULTS_DIR}/model_comparison_extended.csv", index=False)

    print("\n" + "="*60)
    print("FULL MODEL COMPARISON TABLE")
    print("="*60)
    print(results_df.to_string(index=False))

    # Charts
    plot_comparison(results_df)
    plot_predictions(f"{RESULTS_DIR}/extended_lag_predictions.csv",
                     "Extended Lag Linear Regression — Actual vs Predicted",
                     "extended_lag_actual_vs_pred.png")
    plot_predictions(f"{RESULTS_DIR}/lstm_predictions.csv",
                     "LSTM — Actual vs Predicted",
                     "lstm_actual_vs_pred.png")

    print("\n" + "="*60)
    print("All results saved to ./results/")
    print("  model_comparison_extended.csv")
    print("  model_comparison_extended.png")
    print("  lstm_predictions.csv")
    print("  lstm_actual_vs_pred.png")
    print("  lstm_training_curve.png")
    print("  extended_lag_predictions.csv")
    print("  extended_lag_actual_vs_pred.png")
    print("  spike_classifier_report.txt")
    print("  cm_logistic_regression.png")
    print("  cm_random_forest_classifier.png")
    if XGB_OK:
        print("  cm_xgboost_classifier.png")
    print("="*60)


if __name__ == "__main__":
    main()