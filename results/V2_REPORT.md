# v2: cleanup, feature engineering and LightGBM results (validation 7 Sep to 4 Oct 2026, evaluated once)
Data audit and cleanup: ../docs/DATA_AUDIT.md. Leakage mutation test: src/leak_test.py (PASSED).

| Model | Direction Brier | Direction Brier, steps 1-4 | Spike Brier | P50 MAE (EUR/MWh) | Pinball P10 | Pinball P90 | 80% coverage |
|---|---|---|---|---|---|---|---|
| Climatology | 0.451 | – | 0.428 | 44.29 | – | – | – |
| v1 best (base+rt+wind) | 0.409 | 0.309 | 0.354 | 42.08 | 8.65 | 17.61 | 0.54 |
| **v2 final (selected on folds, tuned, horizon split, conformal)** | **0.393** | **0.246** | **0.322** | 42.13 | **8.26** | **16.72** | **0.72** |
| v2 + all weather (sensitivity only, not selected) | 0.392 | 0.246 | 0.318 | 41.46 | 8.25 | 16.66 | 0.73 |

Day-block bootstrap on validation, v2 vs climatology: direction -0.058 [-0.084, -0.031]; spike -0.105 [-0.147, -0.068]; P50 absolute error -2.16 EUR [-3.30, -0.97].
Files: v2_ladder_rolling.csv, v2_ladder_bootstrap.csv, v2_optuna_best.json, v2_final_validation.csv, v2_val_by_horizon.csv, v2_shap_mean_abs.csv
