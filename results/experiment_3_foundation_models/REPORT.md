# Experiment 3: latest TimesFM and Timestral

Research benchmark run on 7 November 2026. Results below compare zero-shot price
forecasts with the existing task-trained LightGBM B and Experiment 2 Chronos-2.
The original competition forecasts are preserved.

## Results

**Conclusion:** Keep LightGBM B as the primary submission. TimesFM 3 and Timestral
both improve on the best Chronos-2 MAE in this experiment, but neither beats
LightGBM B on MAE or mean P10/P90 pinball loss.

Lower MAE and pinball loss are better. Coverage targets 80%.

| Model | MAE EUR/MWh | Mean P10/P90 pinball | Coverage | Mean interval width EUR/MWh |
|---|---:|---:|---:|---:|
| Official baseline | 66.2033 | 24.1776 | 68.29% | 111.51 |
| LightGBM B (current submission) | 42.0286 | 12.4172 | 73.22% | 95.81 |
| Chronos-2: price | 52.4441 | 14.5975 | 73.04% | 137.40 |
| Chronos-2: spread | 44.9159 | 14.4699 | 80.16% | 145.47 |
| Chronos-2: spread + covariates | 47.6458 | 13.6460 | 75.65% | 128.95 |
| TimesFM 3: price | 53.0689 | 15.5711 | 77.74% | 161.20 |
| TimesFM 3: spread | 45.3728 | 14.7111 | 81.13% | 148.66 |
| TimesFM 3: spread + covariates | 43.5332 | 12.7567 | 80.88% | 119.05 |
| Timestral 209M: price | 52.3042 | 14.7788 | 78.17% | 156.72 |
| Timestral 209M: spread | 44.3230 | 14.3182 | 80.56% | 140.18 |
| Timestral 209M: spread + covariates | 43.8162 | 12.7579 | 81.04% | 128.30 |

The best new models both use spread targets with covariates. Relative to LightGBM B:

- TimesFM 3: spread + covariates: MAE is 1.5046 EUR/MWh worse (3.58%).
- Timestral 209M: spread + covariates: MAE is 1.7876 EUR/MWh worse (4.25%).

Their coverage is closer to 80%, but their intervals are wider and their scored
quantile loss remains worse. Coverage alone does not establish a better forecast.

Paired MAE differences versus LightGBM B, resampling all 28 Finnish origin days
5,000 times with seed 7 (positive differences favour LightGBM):

| Configuration | MAE difference EUR/MWh | Approximate 95% bootstrap interval |
|---|---:|---:|
| TimesFM 3: price | +11.0404 | [5.1268, 17.8265] |
| TimesFM 3: spread | +3.3442 | [1.7606, 5.0053] |
| TimesFM 3: spread + covariates | +1.5046 | [0.0648, 2.9919] |
| Timestral 209M: price | +10.2756 | [4.7600, 16.3086] |
| Timestral 209M: spread | +2.2944 | [1.1759, 3.4401] |
| Timestral 209M: spread + covariates | +1.7876 | [0.3282, 3.4511] |

All six configurations passed the availability and batch-independence checks.
All price metrics matched the official scoring function. Original forecast
hashes were verified unchanged after both model runs.

Elapsed inference/input/scoring time after loading (seconds): TimesFM 3: price 23.9, TimesFM 3: spread 23.7, TimesFM 3: spread + covariates 169.0, Timestral 209M: price 13.5, Timestral 209M: spread 13.4, Timestral 209M: spread + covariates 206.0.
These timings exclude checkpoint downloads, installation and smoke checks; they
are local-run measurements, not a controlled hardware benchmark.

## Models and scope

- **Google TimesFM 3.0**, released 31 August 2026, using the official PyTorch
  checkpoint. [Release](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/),
  [model card](https://huggingface.co/google/timesfm-3.0-pytorch).
- **Mistral Timestral 209M**, the largest checkpoint of the family released
  15 October 2026. [Release](https://mistral.ai/news/timestral/),
  [model card](https://huggingface.co/mistralai/Timestral-209M).

Both support past-only and known-future covariates. Exact checkpoint revisions
are pinned in `experiments/download_experiment_3.py` and recorded in `run.json`.
Official inference code is commit- and archive-hash pinned in
`experiments/requirements-foundation-models.txt`; archive provenance is saved in
`source_packages.json`. Python dependencies are recorded in `environment.txt`.
All 65 installed Python source files from the two model packages were compared
byte-for-byte with the pinned source archives and matched.

These are local CPU runs with six PyTorch threads, eight independent origins per
batch, symmetric averaging enabled and nonnegative clipping disabled. Energy
prices and spreads can be negative. Native P10/P50/P90 outputs are extracted;
quantiles are sorted. No parameters are fitted, no validation recalibration is
applied, and no test-period actuals are scored. The models do not produce the
challenge's direction or spike-probability outputs in this experiment.

## Evaluation and availability

The runner reuses Experiment 2's input builder and scoring helpers. Each model
is evaluated in three configurations: direct imbalance price, imbalance price
minus spot, and the same spread with 16 covariates. Seven days (672 quarters) of
context are refreshed at each hourly origin. Spot is added back to spread
quantiles using prices already available at that origin.

The covariates are the same eight known-future and eight past-only channels
documented in [Experiment 2](../experiment_2_chronos/REPORT.md#configurations).
Only the covariate configuration uses external Fingrid inputs; the other two use
package price data. Every source is masked using the existing availability rule
before derived features are calculated. Missing inputs remain NaN for the model.

All 672 official validation origins are used: 6 September 2026 21:00 UTC through
4 October 2026 20:00 UTC, corresponding to the advertised 7 September–4 October
local-date period. Each origin produces 40 quarter-hour rows. The models forecast
44 quarters from the last published target (75 minutes before origin), discard
the four intervening quarters and score the required 40 quarters beginning at
the origin. There are 26,880 forecast rows and 26,700 rows with actuals. The 180
unlabelled horizon-tail rows are excluded identically for all models.

Nine mutation checks verify that changing unavailable source values cannot
change model inputs. For the first batch of every model/configuration, a solo
forecast is compared with the same origin's batched forecast to check batch
independence. Identifiers and steps are matched against the official template.
The official scorer independently checks price metrics; existing LightGBM
probabilities are used only in memory to satisfy its schema, and are never
reported as foundation-model predictions. Submission SHA256 hashes are checked
before and after the run.

## Interpretation limits

This measures out-of-the-box replacement performance, not the best attainable
performance after fine-tuning, context selection or ensembling. Selecting the
best configuration on this validation set is exploratory; it is not a new
independent test score. Paired daily bootstrap intervals preserve within-day
dependence but may miss dependence across days.

Timestral was released after validation ended. Its result is a retrospective
benchmark and cannot establish historical model availability. The public model
descriptions do not establish a complete, independently verified pretraining
overlap audit for this challenge; input availability checks do not audit
pretraining data. No claim of a strictly historical deployment is made.

TimesFM 3's downloaded checkpoint has a
[non-commercial, non-production licence](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
Its outputs here are for research evaluation and have not been promoted into
the competition submission or a commercial decision process. Timestral weights
use [Modified MIT terms](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE)
with a company-revenue restriction; they are not unrestricted MIT weights.
Neither checkpoint is included in the repository; both remain under ignored
`work_private/models/`.

## Reproduction

After reproducing Experiment 2 and its reference predictions, run from the
repository root:

```sh
.venv-chronos/bin/python -m pip install --no-deps -r experiments/requirements-foundation-models.txt
.venv-chronos/bin/python experiments/download_experiment_3.py
.venv-chronos/bin/python experiments/experiment_3_foundation_models.py
.venv-chronos/bin/python experiments/analyze_experiment_2.py --output-dir results/experiment_3_foundation_models
```

For a smoke run add `--limit 2 --output-dir work_private/experiment3_smoke`.
`--models timesfm3` or `--models timestral209m` selects one checkpoint; `--append`
adds a model to a completed compatible run. Full-precision results are in
`comparison.csv`, forecasts in `validation_*.csv`, horizon scores in
`horizon_*.csv`, and paired uncertainty estimates in `paired_mae_bootstrap.csv`.
