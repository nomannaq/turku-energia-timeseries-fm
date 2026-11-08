# Experiment 4: additional data for imbalance forecasting

This experiment tests the hypothesis that earlier information about supply,
interconnector availability and changing wind forecasts improves the existing
LightGBM system. It is an additional-data experiment; no foundation model is
fine-tuned. Downloads and inference run locally, and no observations are uploaded.

## Results and decision

**Keep the existing submission.** The added data produced small validation gains
in some metrics, but none of the paired day-bootstrap intervals excludes no
change. The retrained baseline has the best pre-validation fold MAE. The data
hypothesis remains plausible; this experiment does not establish a substantial
or reliable improvement for this model and feature design.

All scored losses below are better when lower. Coverage is informational, with
an 80% nominal target.

| Variant | MAE EUR/MWh | Mean P10/P90 pinball | Spike Brier | Direction Brier | Coverage |
|---|---:|---:|---:|---:|---:|
| Saved submission B | 42.0286 | 12.4172 | 0.090976 | 0.393988 | 73.22% |
| Retrained control | 42.0286 | 12.4172 | 0.090986 | 0.394063 | 73.24% |
| + outage notices | 41.9542 | 12.3570 | 0.090816 | 0.394118 | 73.58% |
| + ICON/HARMONIE weather | 42.0642 | 12.4707 | 0.091245 | 0.392319 | 72.88% |
| + outages and weather | 41.9510 | 12.4225 | 0.091200 | 0.393859 | 73.42% |

The combined variant improves validation MAE by 0.0776 EUR/MWh (0.18%).
Its pinball and spike losses are slightly worse than the control. Outages alone
improve MAE, pinball and spike Brier modestly, while direction Brier is slightly
worse. Weather alone improves direction Brier but worsens price, interval and
spike losses. There is no consistent winner across the challenge criteria.

Pre-validation fold results (60,260 scored rows):

| Variant | MAE EUR/MWh | Mean P10/P90 pinball | Spike Brier | Direction Brier |
|---|---:|---:|---:|---:|
| Retrained control | 22.8341 | 5.6505 | 0.048732 | 0.413225 |
| + outage notices | 22.8511 | 5.6385 | 0.048935 | 0.411319 |
| + ICON/HARMONIE weather | 23.0125 | 5.6418 | 0.048902 | 0.413838 |
| + outages and weather | 22.9475 | 5.5585 | 0.048953 | 0.412684 |

These fold forecasts precede interval calibration. Selecting by the predeclared
fold-MAE criterion retains the baseline, not the lowest validation-MAE variant.

Paired validation MAE differences versus the retrained control, resampling 28
Finnish forecast-origin days 5,000 times (seed 7):

| Addition | MAE difference EUR/MWh | Approximate 95% interval |
|---|---:|---:|
| + outage notices | -0.0745 | [-0.3626, +0.2284] |
| + ICON/HARMONIE weather | +0.0356 | [-0.2028, +0.2890] |
| + outages and weather | -0.0776 | [-0.4134, +0.2729] |

Negative differences favour the added-data model. Every interval crosses zero.
The intervals preserve within-day dependence, but do not fully address
dependence across days, repeated validation use or model-selection uncertainty.
Intervals for all four scored criteria are in `paired_day_bootstrap.csv`.

All four generated forecasts pass the official format validator. Independently
running the official scoring CLI reproduces the reported metrics. Original
submission hashes remain unchanged. No test forecasts were replaced.

The archive supplied 1,516 of 1,524 requested weather runs: 1,009 DMI and 507 ICON.
Some ICON rows have missing 100m wind values; the maximum missing fraction of
a derived weather feature is 1%. Missing values are retained for LightGBM.

The imported saved submission and retrained control differ negligibly in MAE
(42.028598 versus 42.028606), but some probability and interval values differ.
Paired comparisons consistently use the retrained control.

## Scope of the source audit

| Proposed source | Decision for this experiment |
|---|---|
| Outage notices | Included: public Nord Pool UMM revision histories with publication timestamps. |
| Updated Fingrid wind forecasts | Excluded: no historical forecast-vintage archive established for dataset 245 or an alternative. |
| Additional weather models | Included: individual DMI HARMONIE and ICON-EU forecast runs. |
| Continuous intraday prices | Not tested: the required historical-trade subscription is not configured. |
| Additional neighbour measurements | Not tested: historical as-of publication/version provenance was not established. |
| Reserve bid/capacity data | Not tested: additional API access and publication-rule verification are needed. |

See [source_audit.json](source_audit.json) for evidence and primary-source links
for all six families. These exclusions do not mean the data has no predictive
value, nor that no suitable archive exists elsewhere.

Fingrid's [245 feed](https://data.fingrid.fi/en/datasets/245) updates every 15
minutes, but its local archive contains only dataset ID, delivery start/end and
value, with no issue timestamp or multiple versions. It must not be interpreted
as the forecast historically known before delivery. The public catalogue also
contains [246](https://data.fingrid.fi/en/datasets/246), the once-daily forecast,
which is not an intraday revision archive.

The current submission already uses actual wind errors, nuclear generation
changes, cross-border flows, neighbouring imbalance prices, calculated imbalance,
frequency and scheduled activation prices. Those remain in the control model.
The additions below provide different information; they do not merely relabel
existing columns as new data.

## Outage features

The [public UMM platform](https://umm.nordpoolgroup.com/) supplies historical
message revisions. Discovery combines an event-overlap query with a publication
date query, both including outdated versions. Complete histories are fetched
for every relevant message ID, including revisions outside the discovery window.
There are 442 notices and 1,511 revisions, including 47 cancellation revisions.

At each forecast origin, only versions with publication timestamp plus a
five-minute processing buffer at or before the origin are visible. The latest
visible version replaces the prior version; a cancellation removes that notice.
The archive's present-day `isOutdated` flag is never used to reconstruct past
state. Outage start time alone never makes a notice available.

Fourteen features describe planned/unplanned unavailable generation, nuclear and
wind capacity, unavailable consumption, recently announced unplanned generation,
announced restoration across the horizon, net generation-minus-load loss, and
directional FI–EE, FI–SE1 and FI–SE3 transmission reductions. Repeated overlapping
notices for the same asset use the maximum reduction, avoiding double counting.
For transmission, the data represents a directional corridor: these values must
not be presented as independently measured Estlink/Fenno-Skan cable outages.

The synthetic checks cover cancellation timing and duplicate-asset notices.
Real-history checks remove all future revisions at three origins and require
the resulting features to remain exactly unchanged. Listing totals are checked,
but archive completeness has not been independently certified.

## Weather features

The [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
preserves individual model runs. The experiment requests DMI HARMONIE AROME
Europe every three hours and ICON-EU every six hours, from 31 May through
4 October 2026. No reanalysis or seamless historical forecast is substituted for
a missing run.

The three locations are Raahe, Oulu and Vaasa, drawn from the existing six-region
ECMWF proxy. Their existing weights (0.30, 0.20, 0.15) are normalised to sum to one.
The subset was fixed before model comparison to keep archive requests within the
free API's hourly allowance. It is a regional wind proxy, not a complete fleet
forecast or a replacement for a revised official Fingrid production forecast.

Features include weighted 100m wind speed, a fixed crude power curve, wind
direction components, revision versus the previous available run, run age, and
disagreement with the existing ECMWF proxy and between the two added models.
Hourly values are interpolated to quarter-hours within the same visible run.

Run initialisation is not publication. A **six-hour delay** is imposed before a
run can be used; runs older than 18 hours are not used for forecast values.
This is a conservative operational assumption, not a measured historical
ingestion timestamp. [Provider timing documentation](https://open-meteo.com/en/docs/model-updates)
explains that these timestamps differ. Weather results are conditional on that
assumption. At three origins, removing all runs not yet usable must leave every
weather feature unchanged.

## Controlled comparison

The four variants are the freshly retrained baseline, baseline plus outages,
baseline plus weather, and baseline plus both. All use the submission pipeline's
existing selected feature families, hyperparameters, seeds, short-horizon
specialists and output constraints. New features are appended to each output's
selected columns; there is no new hyperparameter search.

Five chronological folds before the official validation period generate
out-of-fold forecasts. Their boundaries are 5 July, 19 July, 2 August, 16 August,
30 August and 6 September at 21:00 UTC. A training row is usable only when its
origin is before the cutoff and every target source is available by the cutoff.
Each variant's quantile interval adjustments are estimated from these folds.
Training-fold comparisons use uncalibrated out-of-fold forecasts; they should
not be interpreted as an independent estimate of the interval calibration
fitted using those same folds. Existing feature-family selection also used these
folds before this experiment.

Final validation models use data available by **6 September 2026 21:00 UTC** and
are evaluated on the same 672 official origins, ending 4 October 20:00 UTC.
Every origin has 40 quarter-hour steps, starting at the origin. There are 26,880
forecast rows; 26,700 have price actuals. The official `score.py` computes MAE,
P10/P90 pinball loss, coverage, direction Brier and spike Brier scores. Feature
selection and calibration do not use the validation labels. The hidden test
period is not trained on, scored, or regenerated in this experiment.

The saved imported submission is reported as a separate reference. The first
control check found differences in some interval/probability outputs when
retraining with the local feature matrix. A diagnostic directly compared the
production training function with the experiment implementation: all six
feature-family column lists matched, and the steps 1–4 direction model produced
identical predictions between the two functions. Thus paired ablations use the
fresh control, without attributing saved-versus-retrained differences to new
data. See `control_reproduction.json` and `model_parity_check.json`.

This remains an exploratory validation comparison. Gains, if any, require
confirmation on independent data; outage scarcity, seasonality and weather
availability assumptions limit generalisation. Original submission files are
protected by SHA256 checks and remain unchanged.

## Reproduction and artifacts

Use the existing project `.venv` and local starter-package/external data:

```sh
.venv/bin/python experiments/download_experiment_4_outages.py
.venv/bin/python experiments/download_experiment_4_weather.py
.venv/bin/python experiments/experiment_4_features.py
.venv/bin/python experiments/experiment_4_external_data.py
.venv/bin/python experiments/analyze_experiment_4.py
```

Downloads are cached under ignored `work_private/experiment4/`. The public UMM
endpoint is discovered from the platform's own application configuration. The
weather downloader caches unavailable runs, respects request limits and stops
on HTTP 429; it does not switch endpoints to evade a rate limit.

The scripts also support running `--outages-only` feature generation and
`--variants baseline,outages`, then appending `--variants weather,both --append`
after weather downloads complete. These are equivalent to the single full run.

`comparison.csv` contains full-precision validation scores;
`training_fold_comparison.csv` contains pre-validation fold scores;
`validation_*.csv` and `horizon_*.csv` contain predictions and horizon results;
`run.json` records settings and protected artifact hashes;
`availability_checks.json` records leakage checks. Raw-data manifests and
out-of-fold predictions remain in the ignored work directory.

Weather data attribution: Open-Meteo, DMI and DWD; derived wind proxies and
interpolation are this project's transformations. Outage data attribution:
Nord Pool UMM and the message publishers. This run uses public research access
and does not obtain commercial redistribution rights or licensed intraday data.
