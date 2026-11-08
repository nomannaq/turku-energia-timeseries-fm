# Final ensemble submission

**Upload the `forecast_backtest.csv` in this directory.** It contains 673 hourly test origins × 40 quarter-hour steps = 26,920 rows from the five-seed LightGBM + Chronos-2 ensemble.

| File | Purpose |
|---|---|
| [forecast_backtest.csv](forecast_backtest.csv) | Final test forecast to upload |
| [validation_forecast.csv](validation_forecast.csv) | Validation forecast for scoring, not the test submission |

The root `forecast_backtest.csv` is the preserved older single-seed LightGBM fallback.

## Check or reproduce

From the repository root, with the main environment active:

```sh
python validate_submission.py --backtest submission/final_ensemble/forecast_backtest.csv --origins forecast_origins.csv
```

- [Complete setup, scoring and reconstruction guide](../../RUNNING.md)
- [Current evaluation report](../../EVALUATION_REPORT.md)
- [Seven-page PDF](../../output/pdf/Turku_GridCast_Evaluation_Report.pdf)
- [Detailed ensemble audit and provenance](../../results/final_ensemble/REPORT.md)
- [Data sources and licences](../../DATA_SOURCES_AND_LICENSES.md)

Only LightGBM (MIT) and Chronos-2 (Apache-2.0) are model members. Quantiles are blended without conformal adjustment; class probabilities come from the five-seed LightGBM average. No test-period targets are used for fitting or selection.

Final validation MAE is **41.5988** and mean P10/P90 pinball is **11.8864**. Frozen test forecasts have also been backtested locally against reconstructed package outcomes: MAE **26.7028**, pinball **6.8302**. The local test backtest is separate from the pending official organiser evaluation.
