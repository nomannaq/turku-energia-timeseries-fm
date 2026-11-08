# Forecasting experiments

## Experiment 2: Chronos-2

A completed zero-shot benchmark of `amazon/chronos-2` over all 672 official
validation origins. It compares direct price, spread, and spread with covariates
against the baseline and both existing LightGBM variants.

**Result:** best Chronos MAE 44.9159 EUR/MWh versus 42.0286 for LightGBM B.
The current submission remains unchanged.

See [the experiment report](../results/experiment_2_chronos/REPORT.md) for the
complete comparison, availability controls, limitations and reproduction commands.

- `download_chronos_2.py`: download the pinned public model checkpoint locally.
- `experiment_2_chronos.py`: produce forecasts and evaluate price metrics.
- `analyze_experiment_2.py`: verify scores with the official scoring function and
  estimate paired daily-bootstrap differences.
- `requirements-chronos.txt`: isolated experiment dependencies.

## Experiment 3: TimesFM 3 and Timestral 209M

Completed six zero-shot runs across all 672 validation origins, using the same
three input configurations as Experiment 2. Best MAE: TimesFM 3 **43.5332**,
Timestral **43.8162**, compared with LightGBM B **42.0286** EUR/MWh.
Both new models improve on Chronos's best MAE but do not beat the submission.

See [the full report](../results/experiment_3_foundation_models/REPORT.md) for
scores, paired uncertainty estimates, provenance and licence limitations.
Timestral was released after the validation period; its benchmark is retrospective.
TimesFM 3 outputs remain research-only.

- `download_experiment_3.py`: pinned local checkpoints.
- `experiment_3_foundation_models.py`: model adapters and full validation runs.
- `requirements-foundation-models.txt`: pinned official inference code.
- `analyze_experiment_2.py --output-dir results/experiment_3_foundation_models`:
  official score verification and paired day-bootstrap comparison.

## Experiment 4: additional external data

Completed a source audit and four matched LightGBM runs: retrained control,
outage notices, archived ICON/HARMONIE weather, and both. Uses full UMM revision
histories, including cancellations, and publication-aware forecast-run selection.

Validation MAE: control **42.0286**, outages **41.9542**, weather **42.0642**,
both **41.9510** EUR/MWh. The best point estimate is only 0.18% better; all
paired day-bootstrap MAE intervals include zero. The control wins pre-validation
fold MAE, so the current submission remains unchanged.

See [the report](../results/experiment_4_external_data/REPORT.md) for all scored
criteria, availability assumptions, source-access limitations and reproduction.

- `download_experiment_4_outages.py`: full public notice revision histories.
- `download_experiment_4_weather.py`: individual archived weather runs.
- `experiment_4_features.py`: as-of features and leakage checks.
- `experiment_4_external_data.py`: fixed-settings ablations and calibration.
- `analyze_experiment_4.py`: score checks and paired daily uncertainty estimates.

## Experiment 5: parameter tuning

Equal-budget LightGBM searches for the existing features, outage features,
archived weather, and both. Each output head gets eight parameter candidates,
including the inherited settings; short-horizon specialists are included.
Search folds, two-seed shortlist confirmation, and interval calibration occupy
successive pre-validation windows. Choices are frozen before validation scoring.

Matched untuned controls use the same calibration window as tuned models.
Original submission files are protected by SHA-256 checks.

**Result:** no matched run improved median-price MAE. Baseline tuning changed
MAE from **42.0286 to 42.0652**, outages **41.9542 to 42.0210**, weather
**42.0642 to 42.1190**, and both remained **41.9510** EUR/MWh. The combined
model's tiny spike-Brier gain was not statistically clear. Keep the current
submission. See [the full report](../results/experiment_5_parameter_tuning/REPORT.md)
for interval scores, seed checks, quantile-sorting effects and uncertainty.

- `experiment_5_parameter_tuning.py`: reproducible search, confirmation,
  calibration and validation exports; resumes cached fits.
- `analyze_experiment_5.py`: official format checks, independent rescoring,
  paired day-bootstrap comparisons and report generation.
- [Protocol and results](../results/experiment_5_parameter_tuning/): candidates,
  trial scores, frozen selections, forecast CSVs and the generated report.

## Final ensemble: five-seed LightGBM + Chronos-2

The separately supplied ensemble specification is implemented under `src/pkg/`.
It uses Chronos-2 with 2,048 quarters of context and the exact supplied horizon
weights, independently reproduced on rolling OOS predictions. No conformal
adjustment or TimesFM/Timestral member is included.

Validation MAE **41.5988**, pinball **11.8864**. The separate test candidate is
[`submission/final_ensemble/forecast_backtest.csv`](../submission/final_ensemble/forecast_backtest.csv).
The original single-model forecasts remain unchanged.

- `src/pkg/run_final_ensemble.py`: complete reproducible workflow.
- [Full report](../results/final_ensemble/REPORT.md): all metrics, weight-grid
  reproduction, training-only selection scores, mutation checks and uncertainty.
