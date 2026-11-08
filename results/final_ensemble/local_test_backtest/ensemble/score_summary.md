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
| MAE (p50) | 26.7028 | Level 1: accuracy |
| Pinball loss p10 | 4.3594 | |
| Pinball loss p90 | 9.3010 | |
| Mean pinball loss p10 and p90 | 6.8302 | Level 1: calibration |
| P10-P90 coverage | 0.8072 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.0717 | |
| Brier p_spike_low | 0.0417 | |
| Mean Brier, exceptional prices | 0.0567 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.3937 | Level 1: direction |
| Pinball loss p50 | 13.3514 | for information |
| Observed share of high / low quarter-hours | 0.0818 / 0.0668 | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2692 | 21.0206 | 5.6330 | 0.7827 | 0.0427 | 0.3115 |
| 2 | 5-8 | 2692 | 24.8679 | 6.4811 | 0.7842 | 0.0509 | 0.3873 |
| 3 | 9-12 | 2692 | 26.4709 | 6.8440 | 0.8035 | 0.0555 | 0.3985 |
| 4 | 13-16 | 2692 | 27.3582 | 6.8799 | 0.8076 | 0.0573 | 0.4032 |
| 5 | 17-20 | 2692 | 27.4760 | 6.9274 | 0.8143 | 0.0586 | 0.4055 |
| 6 | 21-24 | 2692 | 27.6651 | 7.0119 | 0.8143 | 0.0597 | 0.4078 |
| 7 | 25-28 | 2692 | 27.7189 | 7.0842 | 0.8154 | 0.0605 | 0.4087 |
| 8 | 29-32 | 2692 | 28.0047 | 7.0793 | 0.8198 | 0.0606 | 0.4069 |
| 9 | 33-36 | 2692 | 28.2009 | 7.1453 | 0.8146 | 0.0604 | 0.4058 |
| 10 | 37-40 | 2692 | 28.2449 | 7.2160 | 0.8158 | 0.0605 | 0.4023 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.
