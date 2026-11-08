# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: validation_forecast_B.csv
- Actuals: actuals_validation.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26880
- Rows scored (price metrics): 26700; skipped (no actual): 180
- Rows scored (exceptional prices): 26700
- Rows scored (direction): 26700

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | 42.0286 | Level 1: accuracy |
| Pinball loss p10 | 8.1621 | |
| Pinball loss p90 | 16.6724 | |
| Mean pinball loss p10 and p90 | 12.4172 | Level 1: calibration |
| P10-P90 coverage | 0.7322 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.0953 | |
| Brier p_spike_low | 0.0867 | |
| Mean Brier, exceptional prices | 0.0910 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.3940 | Level 1: direction |
| Pinball loss p50 | 21.0143 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | 66.2033 | for information |
| MAE difference vs baseline (team - baseline) | -24.1747 | for information |
| MAE ratio vs baseline (team / baseline) | 0.6348 | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 36.0367 | 10.8737 | 0.7574 | 0.0772 | 0.3054 |
| 2 | 5-8 | 2684 | 40.3298 | 11.8249 | 0.7135 | 0.0890 | 0.3769 |
| 3 | 9-12 | 2680 | 42.7561 | 12.3873 | 0.7425 | 0.0908 | 0.4027 |
| 4 | 13-16 | 2676 | 42.7649 | 12.4776 | 0.7414 | 0.0914 | 0.4040 |
| 5 | 17-20 | 2672 | 42.7064 | 12.5866 | 0.7305 | 0.0921 | 0.4030 |
| 6 | 21-24 | 2668 | 42.8343 | 12.6442 | 0.7320 | 0.0926 | 0.4034 |
| 7 | 25-28 | 2664 | 42.9855 | 12.7085 | 0.7312 | 0.0927 | 0.4049 |
| 8 | 29-32 | 2660 | 43.1555 | 12.7872 | 0.7263 | 0.0936 | 0.4102 |
| 9 | 33-36 | 2656 | 43.3135 | 12.9045 | 0.7244 | 0.0948 | 0.4144 |
| 10 | 37-40 | 2652 | 43.4711 | 13.0001 | 0.7225 | 0.0958 | 0.4162 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.
