# Archive

This folder contains earlier iterations of the project scripts. They are kept for traceability and reproducibility of figures that appear in earlier drafts of the thesis. The active versions live under `src/`.

| Folder | Replaced by |
| --- | --- |
| `build_dataset/` | `src/features/build_forecast_dataset.py` |
| `cleaning/` | `src/cleaning/clean_both_datasets.py` |
| `dashboards/` (v0 through v4) | `src/dashboard/dashboard.py` (formerly `dashboard_v5.py`) |
| `plots/` | Plots are now produced by the dashboard and by `src/analysis/` |
| `training/` | `src/models/train_latency_model.py` |

Do not run anything from this folder. Paths and dataset names inside these scripts refer to older filenames that no longer exist in `Cleaned/`.
