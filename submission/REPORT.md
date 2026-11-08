# Turku Energia imbalance-price forecast: submission build (starter package v1)

## Files
| File | What |
|---|---|
| `forecast_backtest.csv` | **Primary submission**, variant B (package + external data). 673 origins x 40 steps = 26,920 rows. Validator: PASSED (0 errors, 0 warnings) |
| `forecast_backtest_packageonly.csv` | Fallback, variant A (package only). Validator: PASSED |
| `submission/validation_forecast_{A,B}.csv` | Validation-period forecasts (672 origins, 7 Sep - 4 Oct), scored with the official `score.py` |
| `submission/score_validation_{A,B}/` | Official `score.py` output: score_summary.csv/.md and score_by_horizon.csv (with baseline comparison) |
| `submission/dashboard/{validation,test}_{A,B}.{parquet,csv}` | Dashboard/replay format: origin, delivery, step, p10/p50/p90, direction and spike probabilities, top-3 p50 drivers with signed SHAP contribution, data age per source at the origin, spot / DA-wind known-until, ECMWF run used, interval width, most likely direction |
| `submission/run_{A,B}.json` | Selected feature families per output, fold losses, conformal adjustments, training sizes |

`work_private/` holds the feature matrices. **They contain Nord Pool spot values: never upload or share them.** Raw spot data never goes into `submission/`.

## Official validation metrics (score.py 0.1.1, spike margin 50)
| Metric (lower is better) | Baseline | A: package only | **B: package + external** |
|---|---|---|---|
| MAE p50 (accuracy, 30 pts) | 66.20 | 42.75 | **42.03** |
| Mean pinball p10/p90 (calibration, 20 pts) | 24.18 | 12.75 | **12.42** |
| P10-P90 coverage (information, target 0.80) | 0.683 | 0.742 | 0.732 |
| Mean Brier spikes (20 pts) | 0.1132 | 0.0933 | **0.0910** |
| 3-class Brier direction (10 pts) | 0.4509 | 0.4124 | **0.3940** |

B by horizon hour: MAE goes from 36.0 (hour 1) to 43.5 (hour 10); direction Brier from 0.305 (hour 1) to 0.416; spike Brier from 0.077 to 0.096.

## Availability rules used (binding: `first_usable_origin_utc <= origin`)
- All package series use their row-level `first_usable_origin_utc`. Missing package rows (sparse aFRR, about 1 gap per series) get the availability.csv rule (lag after period end, rounded up to the next full hour). This rule matches the column exactly on every existing row (asserted in code).
- A feature built from several series and rolling windows is usable from the running maximum of its sources' first_usable times.
- Delivery-time inputs (spot, DA wind/consumption/solar forecasts, FI-EE capacity) are used only where their own first_usable time is <= origin.
- External, variant B (all documented, conservative): Fingrid 75 (wind 15-min), 397, 181, 188, 194, 177, 260, 400. Usable from quarter end + 30 min, rounded up to the full hour; 1/3-min data is first aggregated to 15-min means (slot start label, at least 2 samples, forward fill at most 1 quarter). 400 is not used before its quarter ends. Open-Meteo ECMWF IFS single runs: usable from run time + 6 h, rounded up to the full hour; no Historical Forecast API.
- Training: validation run uses rows whose target sources are usable at 2026-09-06T21:00Z. Test run uses rows usable at 2026-10-04T21:00Z, i.e. data up to 4 Oct only; nothing is fitted on the test period. Test-period inputs enter only as lagged features after availability.
- Leakage test `src/pkg/pkg_leak_test.py` scrambles every value (package, external, weather) whose availability is later than cutoff c, then requires identical features for all origins <= c (3 cutoffs x 2 variants). PASSED, and the mutation does show up for later origins.

## Method (short)
- Target: spread = imbalance price - spot. Price quantiles = spot(t) + spread quantiles.
- LightGBM: 3-class direction model; 3-class spike model (high / normal / low); quantile models for p10, p50, p90.
- Feature families (E1 spot + basic spread history, E2 spread lags/volatility, E3 load/wind/solar/residual, E4 regulation/prices/flows/neighbours, EX external Fingrid real-time + wind error, EW weather) are added one at a time with fixed parameters on 5 rolling-origin folds inside training. Each output keeps the best rung (B: direction EW, spikes EX, p10 E4, p50 EX, p90 EX).
- Hyperparameters come from a light Optuna run on training folds only. Steps 1-4 and 5-8 get dedicated models.
- Post-processing: quantiles sorted so they never cross; P(spike high) <= P(up) and P(spike low) <= P(down).
- Intervals: split-conformal per horizon bucket, scaled by recent spread volatility. The validation run calibrates on training folds; the test run calibrates on the validation run's out-of-sample validation forecasts.

## Data sources and licences
- Starter package: Fingrid Open Data (CC BY 4.0), ENTSO-E Transparency (CC BY 4.0), calendar.
- Nord Pool spot (`spot_nordpool.csv`, spot column of the actuals): licensed to Turku Energia, hackathon use only, no redistribution or publication, delete after the event.
- External: Fingrid Open Data API datasets 75, 177, 181, 188, 194, 260, 397, 400 (CC BY 4.0, API key); Open-Meteo Single Runs API with ECMWF IFS (Open-Meteo data CC BY 4.0; free API tier is for non-commercial use; ECMWF open data CC BY 4.0).
- Downloaded but not used in the submission: eSett open data (duplicates package series), FMI observations.

## How to run

Run these commands from the submission repository root after installing
`requirements.txt` in an activated Python virtual environment. The imported data
is already under `data/`; `FINGRID_API_KEY` is only needed to re-download Fingrid data.

```sh
python src/pkg/pkg_features.py A
python src/pkg/pkg_features.py B
python src/pkg/pkg_leak_test.py
python src/pkg/pkg_model.py A
python src/pkg/pkg_model.py B
python src/pkg/export_dashboard.py A
python src/pkg/export_dashboard.py B
python score.py --submission submission/validation_forecast_B.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir submission/score_validation_B --baseline-summary baseline_output/validation/score/score_summary.csv
python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv
```

External fetchers: `src/fetch_fingrid.py`, `src/fetch_openmeteo.py`. The imported
baseline was reproduced in the original project by running the code cells of
`baseline.ipynb`, giving MAE 66.2033, pinball 24.1776, coverage 0.6829, spikes 0.1132,
direction 0.4509. See `README.md` for setup and a package-only rebuild.

Paths in this report are relative to the repository root. Training refreshes both
`submission/forecast_backtest.csv` and the root `forecast_backtest.csv` for B, and
both package-only copies for A. The supplied forecast values are preserved during
integration; path and output-copy changes allow future runs in this layout.
