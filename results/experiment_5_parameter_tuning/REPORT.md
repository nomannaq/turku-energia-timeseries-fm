# Experiment 5 — parameter tuning

Completed four equal-budget searches: existing features, outages, archived weather,
and both. The feature set nominated **before official validation scoring** was
**both_tuned**, using pre-validation confirmation MAE. Its validation MAE is
**41.951015 EUR/MWh**, versus
**42.028598** for the saved submission.
The lowest validation MAE among tuned runs is **both_tuned** at
**41.951015**; that retrospective ranking did not choose parameters.
Existing submission files are unchanged.

**Tuning itself reduced validation median MAE in 0 of four matched comparisons.**
The feature-set comparison and the effect of parameter changes are distinct;
a model beating the saved submission does not by itself show a tuning benefit.

## Validation results

All losses below are lower-is-better. Coverage targets approximately 0.80.
Each forecast has 672 origins × 40 steps = 26,880 rows; the official scorer
uses 26,700 rows with available actuals. No test-period targets enter any fit.

| model | mae_p50 | calibration_pinball_mean | brier_spike_mean | brier_direction | coverage_p10_p90 |
| --- | --- | --- | --- | --- | --- |
| baseline_control | 42.028606 | 12.844585 | 0.090986 | 0.394063 | 0.801610 |
| baseline_tuned | 42.065239 | 12.854078 | 0.092083 | 0.397183 | 0.792846 |
| outages_control | 41.954152 | 12.894123 | 0.090816 | 0.394118 | 0.811948 |
| outages_tuned | 42.021002 | 12.947370 | 0.091720 | 0.394118 | 0.801760 |
| weather_control | 42.064246 | 12.933793 | 0.091245 | 0.392319 | 0.804794 |
| weather_tuned | 42.119022 | 12.888328 | 0.092407 | 0.395607 | 0.798614 |
| both_control | 41.951015 | 12.922178 | 0.091200 | 0.393859 | 0.804270 |
| both_tuned | 41.951015 | 12.922178 | 0.091167 | 0.393859 | 0.804270 |
| submitted_B | 42.028598 | 12.417250 | 0.090976 | 0.393988 | 0.732210 |

Tuned minus matched-control losses (negative means tuning helped):

| feature_set | mae_p50 | calibration_pinball_mean | brier_spike_mean | brier_direction |
| --- | --- | --- | --- | --- |
| baseline | 0.036632 | 0.009493 | 0.001097 | 0.003120 |
| outages | 0.066850 | 0.053248 | 0.000904 | 0.000000 |
| weather | 0.054777 | -0.045466 | 0.001162 | 0.003288 |
| both | 0.000000 | 0.000000 | -0.000033 | 0.000000 |

`control` preserves the inherited global and short-horizon settings, using the
same single-week calibration and final seed as `tuned`. The saved submission
used a different, longer calibration history. Compare tuned versus its matching
control to isolate tuning; compare against `submitted_B` for practical benefit.

The existing pipeline sorts independently predicted quantiles, then sorts again
after interval calibration. Thus changing p10/p90 can move the exported p50 even
when the raw median head is unchanged. `quantile_sorting_diagnostics.csv` records
crossings and median shifts; unchanged median-head selections are verified to
produce exactly identical raw medians. This postprocessing was retained for the
controlled experiment, not changed in response to validation outcomes.

## Search and temporal controls

- Eight candidates per output head and feature set: inherited settings plus seven
  reproducible random configurations; 160 candidate/head/feature-set combinations.
- Tuned learning rate, tree count, number of leaves, minimum leaf observations,
  feature fraction and L2 regularization. Global and first-two-hour specialist
  models are both included. Bagging and feature-family assignments remain fixed.
- Search folds in UTC: Jul 19 21:00–Aug 2 21:00 and Aug 2 21:00–Aug 16 21:00.
- Confirm the two best search candidates plus the incumbent on Aug 16 21:00–
  Aug 30 21:00, using seeds 7 and 19; select by mean loss. If the incumbent is
  already in the top two, only two distinct candidates need confirmation.
- Choose each output head independently: p50 MAE, p10/p90 pinball, direction
  multiclass Brier, and mean high/low spike Brier. Head-level selection precedes
  final quantile sorting, direction capping of spikes and interval calibration.
- Freeze choices in `frozen_selection.json`. Calibrate intervals on Aug 30
  21:00–Sep 6 21:00 only, then refit using all labels available at Sep 6 21:00.
  Final predictions use seed 7; seed 19 checks selection robustness.
- For every fold, training origins precede its cutoff and label `target_fu` is
  at or before that cutoff. Scoring/calibration labels must be available by the
  window end. Forecast targets crossing a boundary are purged accordingly.
- Official validation uses the package's UTC origin template: Sep 6 21:00–
  Oct 4 21:00 exclusive. Its actuals are used for scoring only after parameter freezing.

## Uncertainty and limitations

For the pre-nominated model versus the saved submission, the paired MAE change
is -0.077584 EUR/MWh, with a day-bootstrap 95% interval
[-0.413428, +0.272862]. Negative favors the nominated model.
`paired_day_bootstrap.csv` includes matched-control and tuned-feature contrasts
for every scoring objective, using 5,000 resamples of the 28 local origin days.
These intervals preserve within-day overlap but not all multi-day dependence;
they do not correct for repeated experimentation or multiple comparisons.

This is a bounded search, not proof of globally optimal hyperparameters. The
official validation has been reviewed in previous experiments, so it is not a
fresh final test. The inherited parameters and feature families also come from
earlier work on the training period; these new temporal partitions isolate the
new search, not the entire historical research process. The calibration week
is short, and finite-sample interval adjustment is not a coverage guarantee
under time dependence or changing market regimes. Weather availability retains
Experiment 4's conservative run-plus-six-hours assumption, not measured
historical publication timestamps.

## Verification and reproduction

All eight CSVs pass the official submission validator; independently reloaded
forecasts reproduce every saved score. SHA-256 checks confirm all three protected
submission artifacts are unchanged. Training input hashes, library versions,
parameter values, seed losses and selections are recorded alongside this report.
`selected_parameters.json` expands the frozen candidate IDs into reusable settings.
Runtime: 26.3 minutes.

```sh
.venv/bin/python experiments/experiment_5_parameter_tuning.py
.venv/bin/python experiments/analyze_experiment_5.py
```

Predictions are resumable from `work_private/experiment5`; the protocol must
match to reuse them. Use new output/cache directories for a different search.
