# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: validation_forecast_A.csv
- Actuals: actuals_validation.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26880
- Rows scored (price metrics): 26700; skipped (no actual): 180
- Rows scored (exceptional prices): 26700
- Rows scored (direction): 26700

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | 42.7450 | Level 1: accuracy |
| Pinball loss p10 | 8.1868 | |
| Pinball loss p90 | 17.3163 | |
| Mean pinball loss p10 and p90 | 12.7515 | Level 1: calibration |
| P10-P90 coverage | 0.7420 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.0975 | |
| Brier p_spike_low | 0.0890 | |
| Mean Brier, exceptional prices | 0.0933 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.4124 | Level 1: direction |
| Pinball loss p50 | 21.3725 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | 66.2033 | for information |
| MAE difference vs baseline (team - baseline) | -23.4584 | for information |
| MAE ratio vs baseline (team / baseline) | 0.6457 | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 38.8775 | 11.4407 | 0.7507 | 0.0857 | 0.3600 |
| 2 | 5-8 | 2684 | 41.4423 | 12.0641 | 0.7381 | 0.0913 | 0.3962 |
| 3 | 9-12 | 2680 | 43.3571 | 12.6610 | 0.7463 | 0.0923 | 0.4267 |
| 4 | 13-16 | 2676 | 43.2366 | 12.7601 | 0.7485 | 0.0928 | 0.4235 |
| 5 | 17-20 | 2672 | 43.1813 | 12.9287 | 0.7414 | 0.0937 | 0.4223 |
| 6 | 21-24 | 2668 | 43.1587 | 12.9853 | 0.7425 | 0.0941 | 0.4175 |
| 7 | 25-28 | 2664 | 43.2892 | 13.0163 | 0.7432 | 0.0943 | 0.4171 |
| 8 | 29-32 | 2660 | 43.4620 | 13.1058 | 0.7387 | 0.0953 | 0.4190 |
| 9 | 33-36 | 2656 | 43.6604 | 13.2426 | 0.7357 | 0.0963 | 0.4206 |
| 10 | 37-40 | 2652 | 43.8299 | 13.3317 | 0.7349 | 0.0971 | 0.4212 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.
