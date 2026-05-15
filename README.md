# Starlink vs Terrestrial Broadband in Oman

Empirical measurement and machine-learning forecasting of Starlink Gen 3 satellite internet performance in Muscat, Oman, compared against Omantel fibre and Awasr broadband. This repository contains the data, code, models, and written thesis for the project.

## Project at a glance

Two field experiments were carried out at residential sites in Muscat. Each site ran a Starlink Gen 3 terminal alongside a terrestrial provider. Latency, jitter, packet loss, download and upload throughput, and weather context were logged every fifteen minutes through automated Python scripts. The pipeline cleans the raw measurements, builds a feature set with time-of-day features and lagged latency values, trains short-horizon forecasting models, and serves the results through a Streamlit dashboard and a Suitability Advisor.

| | Experiment A | Experiment B |
| --- | --- | --- |
| Site | Al Mawaleh | Al Hail South |
| Satellite provider | Starlink Gen 3 (Data A) | Starlink Gen 3 (Data B) |
| Terrestrial provider | Omantel fibre | Awasr broadband |
| Collection window | 7 Mar - 28 Mar 2026 | 20 Apr - 12 May 2026 |
| Cadence | 15 minutes | 15 minutes |
| Raw rows (Starlink) | ~2,000 | 2,085 |
| Raw rows (terrestrial) | ~1,500 | 21,620 (oversampled, aggregated to 15-min slots) |

Both experiments serve two different analytical purposes. The provider-comparison analysis treats each experiment independently. The Starlink latency forecasting model was first trained on Experiment A only (`starlink_forecast.csv`), then retrained on the combined feature set built from both experiments (`starlink_forecast_combined.csv`). Terrestrial provider data is never used as training input for the forecaster.

The current production model is a Linear Regression on four lagged latency values, with test-set MAE around 4.7 ms. Random Forest, XGBoost, an extended-lag LSTM, and a binary spike classifier were all evaluated as alternatives; their reports live under `results/`.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
│
├── Raw/
│   ├── experiment_A/        starlink_data.csv, omantel_data.csv
│   └── experiment_B/        starlink_data2.csv, Awasr_data.csv
│
├── Cleaned/
│   ├── experiment_A/        starlink_clean.csv, starlink_clean_FIXED.csv,
│   │                        omantel_clean.csv, starlink_forecast.csv
│   ├── experiment_B/        Starlink_2_cleaned.csv, Awasr_cleaned.csv,
│   │                        starlink_2_forecast.csv
│   ├── combined/            starlink_forecast_combined.csv  (training set for the final model)
│   └── state/               starlink_retrain_queue.csv      (live retrain queue)
│
├── Models/                  starlink_latency_forecast_model.pkl,
│                            starlink_latency_features.pkl,
│                            starlink_latency_model_metadata.csv
│
├── results/                 Model comparison tables, confusion matrices, prediction CSVs,
│                            and plots used in Chapter 4
│
├── data_reports/            Cleaning reports describing every drop, fix, and interpolation
│                            for the Experiment B datasets
│
├── figures/                 Standalone figures used in the thesis chapters
│
├── logs/                    Run-time logs from the live logger (gitignored)
│
├── src/
│   ├── collection/          live_logger.py            cron-driven 15-minute logger
│   ├── cleaning/            clean_both_datasets.py    raw → cleaned, Experiment B
│   ├── features/            build_forecast_dataset.py time features + lags
│   ├── models/              train_latency_model.py    LR / RF training on A, B, A+B
│   │                        model_experiments.py      extended lags, LSTM, spike classifier
│   ├── dashboard/           dashboard.py              live Starlink dashboard
│   │                        dashboard_thesis.py       thesis-focused dashboard,
│   │                                                  reads both experiments + combined
│   │                        advisor.py                Suitability Advisor
│   │                        security_config.py       login + CSV integrity manifest
│   └── analysis/            compare_lr_features.py    before/after feature engineering
│                            compare_xgboost_features.py
│
├── archive/                 Earlier iterations of every pipeline stage, kept for traceability
│                            (do not run; paths inside refer to older filenames)
│
└── thesis/                  Chapter 1-4, Table of Contents, and an archived Chapter 3 draft
```

## How to run

All commands assume the working directory is the project root, since every script uses paths relative to it.

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Re-run the Experiment B cleaning pipeline (Experiment A clean files are already in Cleaned/experiment_A/)
python src/cleaning/clean_both_datasets.py

# 3. Build the feature set for Experiment B
python src/features/build_forecast_dataset.py

# 4. Train the latency model on Experiment A only, on Experiment B only, and on the combined set
python src/models/train_latency_model.py

# 5. Run the extended experiments: long lag window, LSTM, spike classifier
python src/models/model_experiments.py

# 6. Live dashboard (rolling-window view, advisor, model evaluation tab)
streamlit run src/dashboard/dashboard.py

# 7. Thesis dashboard (uses Experiment A, Experiment B, and the combined dataset)
streamlit run src/dashboard/dashboard_thesis.py

# 8. Suitability Advisor
streamlit run src/dashboard/advisor.py

# 9. Schedule the live logger every 15 minutes
crontab -e
# */15 * * * * /path/to/venv/bin/python /path/to/src/collection/live_logger.py
```

## Datasets

Every cleaned row carries three audit flags so analysis can ignore interpolated values when needed: `had_empty_measurement_in_raw`, `was_missing_timestamp_row`, `was_estimated_row`. Full cleaning reports for Experiment B are in `data_reports/`.

Experiment A (Starlink + Omantel)
* `Raw/experiment_A/starlink_data.csv` raw 15-min Starlink measurements
* `Raw/experiment_A/omantel_data.csv` raw 15-min Omantel measurements
* `Cleaned/experiment_A/starlink_clean.csv` initial cleaned version
* `Cleaned/experiment_A/starlink_clean_FIXED.csv` canonical clean file used by the dashboards and the advisor
* `Cleaned/experiment_A/omantel_clean.csv` cleaned Omantel measurements
* `Cleaned/experiment_A/starlink_forecast.csv` feature-engineered training set (hour, day-of-week, lags 1-4)

Experiment B (Starlink + Awasr)
* `Raw/experiment_B/starlink_data2.csv` raw 15-min Starlink measurements
* `Raw/experiment_B/Awasr_data.csv` raw 15-min Awasr measurements (oversampled, aggregated to 15-min slots during cleaning)
* `Cleaned/experiment_B/Starlink_2_cleaned.csv`
* `Cleaned/experiment_B/Awasr_cleaned.csv`
* `Cleaned/experiment_B/starlink_2_forecast.csv` feature-engineered training set

Combined
* `Cleaned/combined/starlink_forecast_combined.csv` Experiment A + Experiment B feature sets concatenated chronologically, used for the final model retrain

State
* `Cleaned/state/starlink_retrain_queue.csv` rows captured by the live logger that are eligible for future retraining (Starlink only)

## Model

* `Models/starlink_latency_forecast_model.pkl` Linear Regression on four lags (final model)
* `Models/starlink_latency_features.pkl` ordered feature names expected by the model
* `Models/starlink_latency_model_metadata.csv` training split sizes and model choice

`train_latency_model.py` loads `experiment_A/starlink_forecast.csv` and `experiment_B/starlink_2_forecast.csv`, evaluates Linear Regression and Random Forest on each dataset individually and on the concatenated set, and saves the model that wins on combined-set MAE plus the matching feature list. Run order recorded in the metadata CSV.

## Reproducibility notes

* Ping target was Cloudflare DNS (1.1.1.1). Weather context was pulled from Open-Meteo. Speedtest used `speedtest-cli`.
* Both sites are residential subscribers in Muscat. Starlink uses inter-satellite-link routing because there is no Starlink gateway in Oman as of the measurement period; that routing variability is the main driver of latency spikes in the Starlink data and is discussed in Chapter 4.
* The dataset is too small for deep models to outperform Linear Regression. The thesis discusses why, and `results/spike_classifier_report.txt` shows that rare spikes are not predictable from the current feature set without Starlink routing telemetry.

## Author

Leen Kayal. Thesis submitted in 2026.
