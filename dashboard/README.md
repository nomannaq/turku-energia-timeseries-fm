# Turku GridCast interactive dashboard

From the project root:

```sh
.venv/bin/python dashboard/server.py
```

Open **http://127.0.0.1:8765**. Choose another port with `--port 8766`.
Uses the existing pandas/numpy/pyarrow environment and Python's HTTP server. The frontend is plain HTML, CSS, JavaScript and SVG; there are no CDN dependencies or frontend packages to install.

## Demo walkthrough

1. Review the full-period metric tables at the top: validation baseline, previous LightGBM and final ensemble, plus the local test comparison. These results are retrospective and independent of the replay origin.
2. Choose validation or local test and move the origin slider. Play replay advances one hourly forecast at a time.
3. Inspect the next ten hours: P50, P10–P90 and the previous hourly forecast aligned to the same delivery quarters.
4. Select a quarter on the chart or dropdown to inspect direction/spike probabilities.
5. Review threshold-based alerts and reference input freshness. Alerts suggest expert review, not automated action.
6. Toggle retrospective actuals explicitly to inspect misses. These realised outcomes were not available at forecast issuance.
7. Compare final ensemble and previous LightGBM metrics over one day, seven days, 28 days or the whole available period. The default restricts outcomes to those published by the selected origin. Select retrospective period end plus whole available period to reproduce complete saved scores.
8. Download the selected origin's original 40 forecast rows.

## Files and scope

The final forecasts come from `submission/final_ensemble/`. Previous-model comparisons use root `forecast_backtest.csv` and `submission/validation_forecast_B.csv`. The driver/freshness panel uses `submission/dashboard/{period}_B.csv`: these are **previous single-seed LightGBM reference explanations**, not SHAP attributions of the final blend. No causal explanation is claimed.

Evaluation uses official validation actuals and `work_private/final_ensemble/local_actuals_test.csv`. If the latter is missing, run `.venv/bin/python src/pkg/score_local_test.py` with the starter data available. Input publication times are read from `data/inputs.parquet`. Test evaluation remains local, not organiser-certified. Historical scoring intentionally repeats delivery quarters across forecast horizons, matching the scorer's row-based evaluation.

The server binds only to localhost and serves an explicit static-route allowlist. It does not serve repository files, keys, or raw spot-price data. No model training, forecast mutation, external requests or data uploads occur. The demo thresholds (30% spike risk and €20/MWh revision) are illustrative, not tuned operating policies.

## Verification

```sh
.venv/bin/python dashboard/test_dashboard.py
node --check dashboard/static/app.js
```

Tests check exact forecast preservation, complete metric parity, absence of raw spot columns, future-outcome invariance for as-of evaluation, and empty initial windows. Visual browser QA was unavailable in the build environment; review desktop and mobile layouts locally before recording the demo.
