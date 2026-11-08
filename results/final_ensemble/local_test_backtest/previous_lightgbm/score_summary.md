# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: forecast_backtest.csv
- Actuals: local_actuals_test.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26920
- Rows scored (price metrics): 26920; skipped (no actual): 0
- Rows scored (exceptional prices): 26920
- Rows scored (direction): 26920

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | 26.6909 | Level 1: accuracy |
| Pinball loss p10 | 4.8372 | |
| Pinball loss p90 | 9.3360 | |
| Mean pinball loss p10 and p90 | 7.0866 | Level 1: calibration |
| P10-P90 coverage | 0.8940 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.0713 | |
| Brier p_spike_low | 0.0418 | |
| Mean Brier, exceptional prices | 0.0565 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.3947 | Level 1: direction |
| Pinball loss p50 | 13.3455 | for information |
| Observed share of high / low quarter-hours | 0.0818 / 0.0668 | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2692 | 20.9983 | 5.7477 | 0.8711 | 0.0425 | 0.3133 |
| 2 | 5-8 | 2692 | 24.9914 | 6.6115 | 0.8811 | 0.0514 | 0.3876 |
| 3 | 9-12 | 2692 | 26.8805 | 7.0750 | 0.8978 | 0.0558 | 0.3998 |
| 4 | 13-16 | 2692 | 27.3094 | 7.1647 | 0.8956 | 0.0574 | 0.4041 |
| 5 | 17-20 | 2692 | 27.5194 | 7.2750 | 0.9008 | 0.0586 | 0.4061 |
| 6 | 21-24 | 2692 | 27.6700 | 7.3342 | 0.8993 | 0.0595 | 0.4083 |
| 7 | 25-28 | 2692 | 27.6749 | 7.3700 | 0.8997 | 0.0601 | 0.4096 |
| 8 | 29-32 | 2692 | 27.8491 | 7.3845 | 0.8982 | 0.0601 | 0.4078 |
| 9 | 33-36 | 2692 | 27.9560 | 7.4314 | 0.8990 | 0.0599 | 0.4069 |
| 10 | 37-40 | 2692 | 28.0601 | 7.4718 | 0.8975 | 0.0599 | 0.4031 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.
