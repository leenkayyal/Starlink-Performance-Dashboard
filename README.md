# Bachelor Thesis: AI-Based Internet Performance Analysis and Forecasting of Starlink Connectivity in Oman

**Author:** Leen Al Kayyal
**Student ID:** 22-0024
**Institution:** German University of Technology in Oman (GUtech)
**Year:** 2026

## Project Overview

This repository contains the code, datasets, machine learning models, and dashboard files developed for the bachelor thesis project. The project analyzes Starlink internet performance in Muscat, Oman, compares it with terrestrial internet service providers, and uses machine learning to forecast short-term Starlink latency.

The repository includes scripts for data cleaning, feature engineering, model training, dashboard visualization, live monitoring, and the Starlink suitability advisor.

## Repository Structure

```text
Starlink-Performance-Dashboard/
│
├── Raw/
│   ├── experiment_A/
│   │   ├── omantel_data.csv
│   │   └── starlink_data1.csv
│   │
│   └── experiment_B/
│       ├── Awasr_data.csv
│       └── starlink_data2.csv
│
├── Cleaned/
│   ├── experiment_A/
│   │   ├── omantel_clean.csv
│   │   ├── starlink_clean_FIXED.csv
│   │   └── starlink_predictions_experiment_A.csv
│   │
│   ├── experiment_B/
│   │   ├── Awasr_cleaned.csv
│   │   ├── Starlink_2_cleaned.csv
│   │   └── starlink_2_forecast.csv
│   │
│   ├── combined/
│   │   ├── starlink_forecast_combined.csv
│   │   ├── starlink_predictions_combined.csv
│   │   ├── model_comparison_combined.csv
│   │   └── combined_robustness_results_with_lstm.csv
│   │
│   └── state/
│       └── starlink_retrain_queue.csv
│
├── Models/
│   ├── starlink_latency_forecast_model.pkl
│   ├── starlink_latency_features.pkl
│   └── starlink_latency_model_metadata.csv
│
├── src/
│   ├── analysis/
│   ├── cleaning/
│   ├── collection/
│   ├── dashboard/
│   ├── features/
│   └── models/
│
├── archive/
├── data_reports/
├── results/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not used, install the main required libraries manually:

```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit plotly streamlit-autorefresh requests
```

## Running the Project

### 1. Data Collection

The live logger collects Starlink performance measurements and appends new observations to the retraining queue.

```bash
python src/collection/live_logger.py
```

### 2. Data Cleaning

The cleaning script processes the raw Starlink and terrestrial ISP datasets.

```bash
python src/cleaning/clean_both_datasets.py
```

### 3. Feature Engineering

The feature engineering script builds the forecasting dataset used for machine learning.

```bash
python src/features/build_forecast_dataset.py
```

### 4. Model Training

Train the Experiment A model:

```bash
python src/models/train_latency_model_A.py
```

Train the combined dataset model:

```bash
python src/models/train_latency_model_combined.py
```

### 5. Dashboard

Run the Streamlit dashboard:

```bash
streamlit run src/dashboard/dashboard_thesis.py
```

## Dashboard and Advisor

The dashboard presents the cleaned data, ISP comparison results, forecasting results, live monitoring output, model evaluation, and advisor module.

The advisor is included as a dashboard tab. It uses the trained Linear Regression model saved in the `Models/` folder to estimate short-term Starlink latency and provide a simple suitability recommendation for different user needs.

## Security and Access Control

The `security_config.py` file defines role-based access profiles for the dashboard. The role profiles include:

| Role               | Access                  |
| ------------------ | ----------------------- |
| Network Manager    | Full dashboard access   |
| Prospective User   | Advisor access          |
| Technical Reviewer | Model evaluation access |

The security implementation is maintained in the `security-login-roles` branch.

## Branch Structure

| Branch                 | Description                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `main`                 | Main thesis codebase, including data processing, model training, dashboard, and advisor files                                                   |
| `security-login-roles` | Backup branch created during development; the security features were later integrated into the main branch and retained here as a fallback copy |

## Key Findings

- Starlink showed stable median latency across both experiments in Muscat.
- Starlink latency generally remained suitable for the tested application categories.
- Terrestrial ISPs showed stronger upload instability compared with Starlink in the collected datasets.
- Latency spikes were difficult to predict because they appeared irregular and were likely affected by network routing or satellite-side factors.
- Linear Regression was selected as the final forecasting model because it provided the best balance of accuracy, stability, and interpretability.
- The dashboard and advisor provide a practical way to visualize the results and explain Starlink suitability to non-technical users.

## Thesis Reference

Leen Al Kayyal, _AI-Based Internet Performance Analysis and Forecasting of Starlink Connectivity in Oman_, Bachelor Thesis, German University of Technology in Oman, 2026.
