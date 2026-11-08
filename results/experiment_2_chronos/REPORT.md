# Experiment 2: Chronos-2 foundation-model benchmark

**Conclusion:** Keep LightGBM B as the primary submission. None of the three
Chronos-2 zero-shot configurations beat it on validation MAE or mean P10/P90
pinball loss. The spread-only Chronos model has coverage closest to 80%, but its
intervals are substantially wider and its scored quantile loss is worse.

## Scope and checkpoint

This is the first Chronos experiment in this project. It tests the public
`amazon/chronos-2` checkpoint (120M parameters), identified as the latest Chronos
model family in the official documentation at the time of this experiment.

- Model revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`.
- Hub revision last modified: 2026-06-05, before the evaluated validation period.
- Checkpoint and inference code licence: Apache-2.0.
- Local CPU inference, float32, six PyTorch threads; no project data uploaded.
- No fine-tuning, parameter updates, validation-based recalibration, or test scoring.
- No direction or spike-probability head was fitted. These are **price-quantile
  experiments**, not complete replacement competition submissions.
- The existing LightGBM model is task-trained and calibrated. This comparison
  measures out-of-the-box replacement performance; it does not establish the
  best performance achievable through foundation-model fine-tuning.

Primary model references:
[official repository](https://github.com/amazon-science/chronos-forecasting),
[model card](https://huggingface.co/amazon/chronos-2),
[pinned checkpoint](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c).

## Evaluation design

The three configurations and seven-day context were fixed before the full
validation comparison. Each is evaluated over all **672 hourly origins** from
2026-09-06 21:00 UTC through 2026-10-04 20:00 UTC (7 September-4 October, Finnish
time). Each origin produces 40 quarter-hour forecasts, giving 26,880 rows.
Exactly **26,700 rows** have official validation actuals; the 180 horizon-tail
rows without actuals are excluded, identically for every model and comparator.

The context consists of 672 quarter-hours (seven days), refreshed at every
origin using only information available then. There is no model fitting on
validation observations. Earlier validation observations may enter the context
once published, as allowed by the backtest availability rule.

The latest usable imbalance target is 75 minutes before the origin. The model
therefore forecasts 44 quarters: four unpublished intervening quarters, followed
by the 40 required quarters. Only the last 40 are evaluated. For spread models,
the delivery-time spot price is added to all three predicted quantiles after
checking that spot was already published at the origin.

Package source values are masked using each row's `first_usable_origin_utc`.
External Fingrid data uses the existing package pipeline's documented rule:
quarter end plus 30 minutes, rounded up to the next hour. Nine mutation checks
(three origins x three configurations) verify that changing unavailable values
cannot change model inputs. Cross-learning between origins is explicitly disabled
so one batch item cannot use another origin's later observations.

Forecast identifiers and all 40 steps match the official validation template.
MAE, pinball and coverage were independently checked against `score.py`'s
`compute_metrics` implementation. Direction/spike columns from LightGBM were used
only in memory to satisfy that function's schema; their scores are not reported
as Chronos results.

## Configurations

1. **Price:** forecast historical imbalance prices directly, without covariates.
2. **Spread:** forecast imbalance price minus spot, then add the known future spot.
3. **Spread + covariates:** the same spread target, with 16 covariate channels:
   - Known into the future: spot, day-ahead wind/consumption/solar, residual load,
     local-hour sine/cosine and weekend indicator (eight channels).
   - Past only: mFRR direction, actual wind minus day-ahead wind, calculated
     imbalance, nuclear generation, net import, frequency deviation, kinetic
     energy and scheduled mFRR price minus spot (eight channels).

The covariate experiment uses package inputs and external Fingrid datasets
75, 397, 188, 194, 177, 260 and 400. It does not feed the existing LightGBM feature
matrix, ECMWF weather forecasts or every external signal into Chronos.
Chronos' P10/P50/P90 outputs are sorted to prevent quantile crossing. There is no
post-hoc tuning of interval width on validation results.

## Results

Lower MAE and pinball loss are better. Coverage should be close to 80%.

| Model | MAE EUR/MWh | Mean pinball P10/P90 | Coverage | Mean interval width EUR/MWh |
|---|---:|---:|---:|---:|
| Official baseline | 66.2033 | 24.1776 | 68.29% | 111.51 |
| LightGBM A: package only | 42.7450 | 12.7515 | 74.20% | 105.89 |
| **LightGBM B: current primary** | **42.0286** | **12.4172** | 73.22% | **95.81** |
| Chronos-2: price | 52.4441 | 14.5975 | 73.04% | 137.40 |
| Chronos-2: spread | 44.9159 | 14.4699 | **80.16%** | 145.47 |
| Chronos-2: spread + covariates | 47.6458 | 13.6460 | 75.65% | 128.95 |

The best Chronos MAE is **2.8873 EUR/MWh worse (6.87%)** than LightGBM B. Adding the
chosen covariates improves pinball loss relative to spread-only Chronos, but
worsens its median forecast error. Matching the nominal coverage alone does not
win the challenge's calibration criterion, which scores pinball loss. The
spread-only model's intervals are **51.84% wider** than LightGBM B's.

A paired bootstrap resampling 28 Finnish forecast-origin days, with 5,000
resamples and seed 7, estimates these MAE differences relative to LightGBM B:

| Chronos configuration | MAE difference EUR/MWh | Approximate 95% bootstrap interval |
|---|---:|---:|
| Price | +10.4155 | [+4.6066, +16.7271] |
| Spread | +2.8873 | [+1.5096, +4.2166] |
| Spread + covariates | +5.6172 | [+2.4535, +9.6027] |

Positive differences favour LightGBM. These intervals are validation diagnostics,
not guarantees about the hidden test period. They preserve within-day dependence
but may not capture all dependence across days.

Model inference, input assembly and output scoring took approximately 6.7 seconds
for price, 6.7 seconds for spread, and 63.0 seconds for spread with covariates, after
loading the checkpoint. Downloads, environment setup and the initial smoke check
are excluded from those timings.

## Artifacts and reproduction

- `comparison.csv`: full-precision metrics for all six models.
- `validation_chronos2_{price,spread,spread_covariates}.csv`: validation price
  forecasts, including the latest observed target timestamp.
- `horizon_*.csv`: scores by forecast hour.
- `paired_mae_bootstrap.csv`, `daily_mae_differences.csv`: paired comparison.
- `run.json`: checkpoint metadata, parameters, availability checks, timings and
  SHA256 hashes verifying that the original forecasts are unchanged.
- `environment.txt`: exact installed package versions.
- `run.log`: execution progress and final metrics.

From the repository root:

```sh
python3.12 -m venv .venv-chronos
.venv-chronos/bin/python -m pip install -r experiments/requirements-chronos.txt
.venv-chronos/bin/python experiments/download_chronos_2.py
.venv-chronos/bin/python experiments/experiment_2_chronos.py
.venv-chronos/bin/python experiments/analyze_experiment_2.py
```

The checkpoint is cached under ignored `work_private/models/chronos-2/`.
The supplied data must be available locally under `data/`.
The experiment uses `baseline_output/validation/` templates and baseline forecasts;
these were imported with the project. On a fresh checkout, run `baseline.ipynb`
to reproduce them first.

For a smoke run, append `--limit 2 --output-dir work_private/experiment_2_smoke`.
Do not compare that limited run with full-month metrics.

Original submission forecasts, validation forecasts and model outputs were not
replaced. A separate future experiment could test training-only fine-tuning or
predeclared ensemble weights; those approaches were not tested here.
