# Final ensemble submission

Use `forecast_backtest.csv` in this folder for the five-seed LightGBM + Chronos-2
ensemble: 673 hourly test origins, 40 quarter-hour steps each. The separate
`validation_forecast.csv` is for evaluation, not submission as the test forecast.

The original single-model fallback remains unchanged at the repository root.
The full method, validation results, weight-selection audit, source hashes and
checks are in [the ensemble report](../../results/final_ensemble/REPORT.md).

Reproduce from the repository root:

```sh
.venv/bin/python src/pkg/run_final_ensemble.py
```

Only LightGBM (MIT) and Chronos-2 (Apache-2.0) are model members. The blend uses
raw quantiles without conformal adjustment; all class probabilities come from
the five-seed LightGBM average. No test-period targets are used for fitting or
selection, and no test accuracy is claimed.
