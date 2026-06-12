"""
run_forecasting_validation.py

Reproduces the forecasting validation results for the Starlink latency study.
It mirrors the training pipeline used elsewhere in the project: the same
sanity filters, the same chronological 80/20 split, and the same naive
baseline (predict the previous latency value).

It produces three CSV files:
  1. crosssite_generalization_results.csv
       within-site and cross-site transfer performance (Linear Regression vs naive)
  2. forecast_confidence_and_limits.csv
       combined-model error, the 45.7% RMSE reduction check,
       conformal prediction interval coverage, and the honest spike-detection metrics
  3. permutation_importance.csv
       which inputs drive the next-step latency forecast

Inputs (forecasting feature files, one row per 15-minute slot):
  Experiment A: starlink_forecast_v2.csv
  Experiment B: starlink_2_forecast.csv

Usage:
  Put this script in the same folder as the two input CSVs (or edit INPUT_DIR),
  then run:  python run_forecasting_validation.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
INPUT_DIR = "."                       # folder holding the two input CSVs
OUTPUT_DIR = "."                      # folder where the result CSVs are written
FILE_A = "starlink_forecast_v2.csv"   # Experiment A forecasting features
FILE_B = "starlink_2_forecast.csv"    # Experiment B forecasting features

TARGET = "target"
LAG1 = "ping_avg_rtt_ms_lag_1"        # also used as the naive baseline
SPIKE_THRESHOLD_MS = 50.0             # latency above this counts as a spike
CONFORMAL_LEVEL = 0.90                # target coverage for the prediction interval
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_and_filter(path):
    """Load one forecasting file and apply the same sanity filters used in training."""
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # latency target and its lags must be realistic (0 to 500 ms)
    for col in [TARGET, "ping_avg_rtt_ms_lag_1", "ping_avg_rtt_ms_lag_2",
                "ping_avg_rtt_ms_lag_3", "ping_avg_rtt_ms_lag_4"]:
        df = df[(df[col] > 0) & (df[col] < 500)]
    # speed and jitter ranges
    df = df[(df["download_lag_1"] > 0) & (df["download_lag_1"] < 1000)]
    df = df[(df["upload_lag_1"] > 0) & (df["upload_lag_1"] < 500)]
    df = df[(df["jitter_lag_1"] >= 0) & (df["jitter_lag_1"] < 100)]
    return df.reset_index(drop=True)


def evaluate(y_true, y_pred):
    """Return MAE, RMSE, R2 for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def chrono_split(X, y, frac=0.8):
    """Chronological split: first frac for training, the rest for testing."""
    cut = int(len(X) * frac)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def within_site(X, y, label):
    """Train and test on the same site using a chronological split."""
    Xtr, Xte, ytr, yte = chrono_split(X, y)
    model = LinearRegression().fit(Xtr, ytr)
    m = evaluate(yte, model.predict(Xte))
    n = evaluate(yte, Xte[LAG1].values)          # naive baseline
    return [label, "Linear Regression", len(Xte),
            round(m[0], 3), round(m[1], 3), round(m[2], 3),
            round(n[0], 3), round(n[1], 3)]


def transfer(X_train, y_train, X_test, y_test, label):
    """Train on one whole site, test on the other whole site."""
    model = LinearRegression().fit(X_train, y_train)
    m = evaluate(y_test, model.predict(X_test))
    n = evaluate(y_test, X_test[LAG1].values)     # naive baseline at the new site
    return [label, "Linear Regression", len(X_test),
            round(m[0], 3), round(m[1], 3), round(m[2], 3),
            round(n[0], 3), round(n[1], 3)]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    A = load_and_filter(os.path.join(INPUT_DIR, FILE_A))
    B = load_and_filter(os.path.join(INPUT_DIR, FILE_B))
    print(f"Loaded Experiment A: {len(A)} rows | Experiment B: {len(B)} rows")

    XA, yA = A.drop(columns=[TARGET]), A[TARGET]
    XB, yB = B.drop(columns=[TARGET]), B[TARGET]

    # --- 1. Cross-site generalization -------------------------------------
    cross_rows = [
        within_site(XA, yA, "Within A (train A, test A)"),
        within_site(XB, yB, "Within B (train B, test B)"),
        transfer(XA, yA, XB, yB, "Transfer A->B (train A, test B)"),
        transfer(XB, yB, XA, yA, "Transfer B->A (train B, test A)"),
    ]
    cross = pd.DataFrame(cross_rows, columns=[
        "scenario", "model", "test_rows",
        "model_MAE_ms", "model_RMSE_ms", "model_R2",
        "naive_MAE_ms", "naive_RMSE_ms"])
    cross.to_csv(os.path.join(OUTPUT_DIR, "crosssite_generalization_results.csv"), index=False)

    # Experiment A alone RMSE (used for the 45.7% reduction check)
    expA_alone_rmse = within_site(XA, yA, "_")[4]

    # --- Combined model (concatenate both sites, drop duplicate slots) ----
    common = [c for c in A.columns if c in set(B.columns)]
    C = pd.concat([A[common], B[common]], ignore_index=True).drop_duplicates().reset_index(drop=True)
    XC, yC = C.drop(columns=[TARGET]), C[TARGET]
    Xtr, Xte, ytr, yte = chrono_split(XC, yC)
    combined = LinearRegression().fit(Xtr, ytr)
    cmb = evaluate(yte, combined.predict(Xte))
    rmse_reduction = (expA_alone_rmse - cmb[1]) / expA_alone_rmse * 100.0

    # --- 2. Conformal prediction intervals --------------------------------
    # carve a calibration block out of the training data the model never fits on
    cal_cut = int(len(Xtr) * 0.8)
    cal_model = LinearRegression().fit(Xtr.iloc[:cal_cut], ytr.iloc[:cal_cut])
    residuals = np.abs(ytr.iloc[cal_cut:].values - cal_model.predict(Xtr.iloc[cal_cut:]))
    half_width = np.quantile(residuals, CONFORMAL_LEVEL)   # interval is pred +/- half_width
    preds = cal_model.predict(Xte)
    inside = (yte.values >= preds - half_width) & (yte.values <= preds + half_width)
    coverage = inside.mean() * 100.0

    # --- Honest spike-detection metrics at the 50 ms threshold ------------
    actual_spike = (yte.values > SPIKE_THRESHOLD_MS).astype(int)
    pred_spike = (combined.predict(Xte) > SPIKE_THRESHOLD_MS).astype(int)

    summary = pd.DataFrame([
        ["combined_LR_MAE_ms", round(cmb[0], 3)],
        ["combined_LR_RMSE_ms", round(cmb[1], 3)],
        ["combined_LR_R2", round(cmb[2], 3)],
        ["expA_alone_RMSE_ms", round(expA_alone_rmse, 3)],
        ["RMSE_reduction_pct_vs_expA_alone", round(rmse_reduction, 1)],
        ["conformal_interval_halfwidth_ms", round(half_width, 2)],
        ["conformal_target_coverage_pct", round(CONFORMAL_LEVEL * 100, 1)],
        ["conformal_empirical_coverage_pct", round(coverage, 1)],
        ["conformal_mean_width_ms", round(2 * half_width, 1)],
        ["spike_threshold_ms", SPIKE_THRESHOLD_MS],
        ["spikes_in_test", int(actual_spike.sum())],
        ["test_points", int(len(actual_spike))],
        ["spike_share_pct", round(actual_spike.mean() * 100, 1)],
        ["overall_accuracy_pct", round(accuracy_score(actual_spike, pred_spike) * 100, 1)],
        ["spike_precision_pct", round(precision_score(actual_spike, pred_spike, zero_division=0) * 100, 1)],
        ["spike_recall_pct", round(recall_score(actual_spike, pred_spike, zero_division=0) * 100, 1)],
        ["spike_f1_pct", round(f1_score(actual_spike, pred_spike, zero_division=0) * 100, 1)],
    ], columns=["metric", "value"])
    summary.to_csv(os.path.join(OUTPUT_DIR, "forecast_confidence_and_limits.csv"), index=False)

    # --- 3. Permutation importance ----------------------------------------
    pi = permutation_importance(
        combined, Xte, yte, n_repeats=20,
        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({
        "feature": XC.columns,
        "importance_MAE_ms": np.round(pi.importances_mean, 4),
    }).sort_values("importance_MAE_ms", ascending=False).reset_index(drop=True)
    importance.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)

    # --- Console summary --------------------------------------------------
    print("\nCross-site generalization:")
    print(cross.to_string(index=False))
    print("\nForecast confidence and limits:")
    print(summary.to_string(index=False))
    print("\nPermutation importance (top drivers):")
    print(importance.head(6).to_string(index=False))
    print("\nWrote 3 CSV files to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()