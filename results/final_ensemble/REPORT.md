# Five-seed LightGBM + Chronos-2 ensemble

> Update: the frozen test forecasts have now been scored locally against realised
> package inputs. See [the local test backtest](local_test_backtest/README.md).
> Earlier references below to unknown test accuracy describe the pre-evaluation
> state; only the official organiser score remains pending.


Implemented the supplied ensemble as a separate submission candidate, retaining
the original submission byte-for-byte. Both exports pass the official validator;
validation scores reproduce with `score.py` 0.1.1. No hidden test accuracy is claimed.

Uploadable test forecast: `submission/final_ensemble/forecast_backtest.csv`
(673 origins, 26,920 rows). Validation forecast:
`submission/final_ensemble/validation_forecast.csv` (672 origins, 26,880 rows;
26,700 rows with available actuals).

## Reproduced validation results

| model | mae_p50 | calibration_pinball_mean | coverage_p10_p90 | brier_spike_mean | brier_direction | S |
| --- | --- | --- | --- | --- | --- | --- |
| organizer_baseline | 66.203304 | 24.177618 | 0.682884 | 0.113176 | 0.450907 | 80.000000 |
| current_lightgbm_single | 42.028598 | 12.417250 | 0.732210 | 0.090976 | 0.393988 | 54.131532 |
| lightgbm_5seed_raw | 42.011360 | 12.492785 | 0.646442 | 0.090792 | 0.392898 | 54.129380 |
| chronos2_covariates | 42.996477 | 12.370898 | 0.753745 | — | — | — |
| final_blend | 41.598828 | 11.886367 | 0.732809 | 0.090792 | 0.392898 | 53.440806 |

Chronos has no probability heads here; its standalone probability metrics are
intentionally blank. The final blend uses only averaged LightGBM probabilities.
`S` uses the supplied 30/20/20/10 normalized-loss formula (baseline S = 80).

The supplied rounded final-blend values were MAE 41.60, pinball 11.89, coverage
0.733, spike Brier 0.0908 and direction Brier 0.3928. Local reproduction is
MAE **41.598828**, pinball **11.886367**,
coverage **0.732809**, spike **0.090792**,
direction **0.392898**. Differences are recorded explicitly
in `reported_result_comparison.csv`; the remote implementation was not available
for byte-for-byte parity checking.

The standalone Chronos raw-quantile result has pinball 12.370898 and coverage
0.753745, differing from the supplied standalone figures 12.53 and 0.803, despite
matching its approximately 43.00 MAE. The supplied standalone postprocessing was
not available to audit. No extra calibration was introduced to force a match;
the requested final blend uses uncalibrated member quantiles and closely
reproduces the supplied final-blend metrics.

## Weight selection and training-only scores

The requested candidate keeps the supplied LightGBM weights: validation
**[0.7, 0.4, 0.5]**, test **[0.7, 0.4, 0.6]**, for steps 1–4, 5–16 and 17–40.
Chronos receives the remaining weight, identically for all three quantiles.
The independently recomputed grid optima are validation
**[0.7, 0.4, 0.5]**, test
**[0.7, 0.4, 0.6]**. Agreement with the supplied
weights: validation **True**, test
**True**. Differences do not silently change the
requested candidate. Full candidate losses are in `weight_grid.csv`.

Validation weight selection uses only rolling OOS training targets usable by
2026-09-06 21:00 UTC. Test selection may additionally use validation OOS targets,
but only those usable by 2026-10-04 21:00 UTC. The latter is a final-training
update; its settings are not retroactively credited with the validation score.
Weight selection minimizes total pinball over p10/p50/p90; the official scored
pinball metric averages p10/p90 only.

| model | mae_p50 | calibration_pinball_mean | coverage_p10_p90 | brier_spike_mean | brier_direction | S |
| --- | --- | --- | --- | --- | --- | --- |
| organizer_baseline | 30.493996 | 9.429438 | 0.836094 | 0.057891 | 0.453569 | 80.000000 |
| lightgbm_5seed_raw | 22.838372 | 5.630556 | 0.751875 | 0.048582 | 0.413211 | 60.304840 |
| specified_blend | 22.516102 | 5.283963 | 0.773664 | 0.048582 | 0.413211 | 59.252662 |

Organizer baseline training-fold predictions use only function definitions
extracted from the shipped `baseline.ipynb`; no notebook workflow or side effects
are executed. The supplied wider model-selection search is not reconstructed:
this audits the requested blend against its five-seed raw LightGBM member.

## Implemented specification

- Five seeds: 7, 11, 23, 37, 51, with explicit bagging/feature/data seed offsets
  +1/+2/+3. Raw quantile and class predictions are averaged before sorting,
  normalization and spike-direction capping. Global family assignments and the
  separate steps 1–4 / 5–8 specialists match the supplied specification.
- Global parameters use the stored full-precision Optuna values underlying the
  rounded settings in the spec. Exact settings are in `lgb_completed.json`.
- Chronos-2, `chronos-forecasting==2.3.2`, CPU, batch size 64, 2,048-quarter
  context ending at origin minus 45 minutes. Last two target quarters are NaN.
  Forecast length 42; discard its first two positions and retain 40 quarters.
- Twelve covariates: seven mFRR/real-time/SA channels, RT-wind-minus-DA-wind,
  and the four known-future series. Every source is masked separately before
  subtraction or forward filling. Only future covariate gaps are forward-filled
  from already visible history. No backward filling or cross-origin learning.
- The same delivery spot is added after spread blending. No conformal adjustment
  is applied. Quantiles are sorted, prices rounded to four decimals and
  probabilities to six. No outage or new ICON/HARMONIE features are added.
- Model licenses recorded locally: LightGBM MIT; Chronos code/checkpoint
  Apache-2.0. No TimesFM or Timestral model is used. Dependency and input-data
  licenses remain their own; this is not a claim that every dependency is MIT/Apache.

## Availability and verification

Package `first_usable_origin_utc` controls visibility. Fingrid external series use
quarter end +30 minutes, rounded up to the hour. ECMWF runs use run time +6 hours.
Every LightGBM fit enforces origin before cutoff and target availability by cutoff.
Five training folds start Jul 5/19, Aug 2/16/30 at 21:00 UTC; validation and test
fits use Sep 6 and Oct 4 at 21:00 UTC respectively. Official templates determine
the exact forecast origin grids, including the 673-origin test period.

Future-value mutation checks passed at Jul 10 09:00, Sep 20 14:00 and Oct 20
03:00 UTC for all eligible LightGBM feature rows. All 225 fitted global/specialist
models checked at these origins produced bit-identical outputs after mutation.
Chronos inputs were identical for every required historical origin through each
of the Sep 20/Oct 20 cutoffs; cutoff predictions were bit-identical. Separate
solo-versus-batch checks pass within numerical tolerance for every period.

Paired changes versus the current single-seed submission, with 5,000 resamples
of the 28 local forecast-origin days (negative loss differences favor the blend):

| metric | difference | ci_95_lower | ci_95_upper |
| --- | --- | --- | --- |
| mae_p50 | -0.429770 | -1.505218 | 0.674700 |
| calibration_pinball_mean | -0.530883 | -1.312134 | 0.154598 |
| brier_spike_mean | -0.000185 | -0.000581 | 0.000185 |
| brier_direction | -0.001090 | -0.002096 | -0.000008 |

Day resampling retains within-day overlap, but not every multi-day dependence.
Repeated validation inspection and upstream training-fold model selection limit
the independence of these estimates. Feature availability tests do not audit
foundation-model pretraining overlap. ECMWF's six-hour publication delay is the
specified assumption. The separate test forecast has no reported actuals or score.

## Reproduction

```sh
.venv/bin/python src/pkg/run_final_ensemble.py
```

The orchestrator uses `.venv` for LightGBM/scoring and `.venv-chronos` for Chronos.
The existing pinned checkpoint and source data are required. Intermediate members,
model files and resumable Chronos batches are in ignored `work_private/final_ensemble`.
The original submission remains at `forecast_backtest.csv`; the new uploadable
candidate is in the separate path above.
