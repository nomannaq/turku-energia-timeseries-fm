# Turku GridCast dashboard

**[Setup and launch](../RUNNING.md#1-set-up-and-launch-the-dashboard) · [Walkthrough](../RUNNING.md#2-explore-the-dashboard) · [Troubleshooting](../RUNNING.md#7-troubleshooting-and-verification)**

The main run guide contains the installation steps, required files and demo walkthrough. From the repository root, with the existing environment:

```sh
.venv/bin/python dashboard/server.py
```

Open **http://127.0.0.1:8765**. Choose another port with `--port 8766`.

## Implementation and data provenance

The backend uses Python's HTTP server with pandas, NumPy and PyArrow. The frontend is HTML, CSS, JavaScript and SVG, with no CDN dependencies or frontend package installation. This is not a Streamlit app.

| Dashboard content | Source |
|---|---|
| Final forecasts | `submission/final_ensemble/` |
| Previous-model comparison | Root `forecast_backtest.csv` and `submission/validation_forecast_B.csv` |
| Reference drivers and freshness | `submission/dashboard/test_B.csv` and `validation_B.csv` |
| Validation outcomes | `data/actuals_validation.csv` |
| Local test outcomes | `work_private/final_ensemble/local_actuals_test.csv`, or reconstruction in memory from starter input and spot files |
| Input publication times | `data/inputs.parquet` |
| Opening full-period tables | Saved comparison CSVs in `results/final_ensemble/` |

Paths in this table are relative to the repository root. Reference drivers are **previous single-seed LightGBM explanations**, not SHAP attributions of the final blend. No causal explanation is claimed.

The opening metric tables are retrospective and independent of the replay origin. Historical performance defaults to outcomes available at the selected origin. Full-period scoring intentionally repeats delivery quarters across forecast horizons, matching the organiser scorer's row-based evaluation. Test evaluation is local, not organiser-certified.

The server binds only to localhost and serves an explicit static-route allowlist. It does not serve repository files, keys or raw spot-price data. Viewing the dashboard does not train models, change forecasts, make external requests or upload data. The default alerts (30% spike risk and EUR 20/MWh forecast revision) are illustrative review thresholds, not tuned operating policies.

## Verification

From the repository root:

```sh
.venv/bin/python dashboard/test_dashboard.py
```

Optional frontend syntax check, if Node is installed:

```sh
node --check dashboard/static/app.js
```

Tests cover forecast preservation, complete metric parity, absence of raw spot columns, future-outcome invariance for as-of evaluation, empty initial windows and local test-label reconstruction. Desktop Chrome checks passed for result tables, replay, actual overlays, risk panels and history-window changes. Mobile and external deployment remain unverified.
