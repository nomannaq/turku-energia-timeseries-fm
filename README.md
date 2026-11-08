# Turku Energia imbalance-price forecast

Submission project for Since AI 2026. The final ensemble candidate is
[`submission/final_ensemble/forecast_backtest.csv`](submission/final_ensemble/forecast_backtest.csv):
five-seed LightGBM B plus Chronos-2 spread quantiles. It contains 26,920 forecasts
(673 hourly origins, 40 quarter-hour steps per origin). The original single-model
variant B remains unchanged at [`forecast_backtest.csv`](forecast_backtest.csv)
as the fallback.

The completed source code, results, reports, dashboard exports and external data
have been imported from the project handoff. The official starter scripts and
local starter data are retained. The original package instructions are in
[`docs/STARTER_README.md`](docs/STARTER_README.md).

## Submission files

| Path | Purpose |
|---|---|
| `submission/final_ensemble/forecast_backtest.csv` | Final five-seed LightGBM + Chronos-2 ensemble test forecast |
| `submission/final_ensemble/validation_forecast.csv` | Ensemble forecasts on the official validation origins |
| `results/final_ensemble/REPORT.md` | Ensemble method, reproduced scores, weights, provenance and checks |
| `forecast_backtest.csv` | Preserved original variant B test forecast; fallback |
| `forecast_backtest_packageonly.csv` | Alternative variant A, using package data only |
| `REPORT.md` | Method, evaluation, availability rules and data-source licences |
| `src/pkg/` | Final feature, model, leakage-check and dashboard-export pipeline |
| `submission/` | Both variants, validation forecasts, run metadata and official scores |
| `submission/dashboard/` | CSV and Parquet replay data with probabilities, SHAP drivers and source freshness |
| `results/` | Earlier experiments, selected hyperparameters and ablation results |
| `src/` | External-data fetchers and earlier research scripts |
| `docs/` | Data audit, background material and original starter instructions |
| `data/` | Local starter and external datasets plus cached API responses; ignored by Git |
| `work_private/` | Generated feature matrices containing licensed spot values; ignored by Git |

`src/pkg/` is the final submission pipeline. Earlier experiments in `src/` use
older availability assumptions, documented in their source and reports.
The dashboard files are data exports; the handoff did not contain a dashboard
application, presentation slides or a demo video.

## Final LightGBM + Chronos-2 ensemble

The ensemble uses seeds 7/11/23/37/51, Chronos-2 with 2,048 quarters of context,
and the supplied horizon-dependent quantile weights. There is no conformal
adjustment; direction and spike probabilities come from LightGBM only. Both
validation and test weight grids reproduce the supplied settings. Model licenses
are MIT (LightGBM) and Apache-2.0 (Chronos); TimesFM and Timestral are not used.

Validation MAE is **41.5988** and mean p10/p90 pinball is **11.8864**, compared
with **42.0286** and **12.4172** for the original single-model submission.
See [the ensemble report](results/final_ensemble/REPORT.md) for coverage,
probability scores, training-fold selection, uncertainty and availability checks.

```sh
.venv/bin/python src/pkg/run_final_ensemble.py
.venv/bin/python validate_submission.py --backtest submission/final_ensemble/forecast_backtest.csv --origins forecast_origins.csv
```

The runner uses the existing `.venv` and `.venv-chronos` environments, local
source data, `work_private/features_B.parquet`, and the pinned Chronos-2 checkpoint.
Chronos dependencies are in `experiments/requirements-chronos.txt`; the checkpoint
can be restored with `experiments/download_chronos_2.py` in that environment.
Intermediate fits and forecasts are cached under `work_private/final_ensemble`.
This runner writes the ensemble subdirectory and preserves the original forecasts
and their existing dashboard exports. The older rebuild commands below still
refer to the original single-model variants.

## Setup and validate the delivered forecasts

Run commands from this repository's root. Python 3.11 or newer is recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv
python validate_submission.py --backtest forecast_backtest_packageonly.csv --origins forecast_origins.csv
```

On Windows, create the environment with `py -m venv .venv` and activate it using
`.venv\Scripts\Activate.ps1`. On macOS, LightGBM requires the OpenMP runtime;
install it with `brew install libomp` if LightGBM reports that `libomp.dylib` is missing.

Both delivered test forecasts pass the official validator with zero errors and
zero warnings. Checking or scoring the delivered forecasts does not require
retraining or API keys.

## Reproduce validation scores

```sh
python score.py --submission submission/validation_forecast_A.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir submission/score_validation_A
python score.py --submission submission/validation_forecast_B.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir submission/score_validation_B
```

The imported baseline scores are locally available under
`baseline_output/validation/score/`. Add
`--baseline-summary baseline_output/validation/score/score_summary.csv` to include
that comparison. On a fresh checkout, run `baseline.ipynb` to regenerate them.

| Validation metric | Baseline | A: package only | B: primary |
|---|---:|---:|---:|
| MAE p50 | 66.20 | 42.75 | 42.03 |
| Mean pinball p10/p90 | 24.18 | 12.75 | 12.42 |
| P10–P90 coverage (target 0.80) | 0.683 | 0.742 | 0.732 |
| Mean Brier spikes | 0.1132 | 0.0933 | 0.0910 |
| Three-class Brier direction | 0.4509 | 0.4124 | 0.3940 |

## Rebuild features, models and dashboard exports

The imported forecasts are ready to validate. The following commands rebuild
both variants and replace their outputs; full rolling-fold training takes longer
than validation. Scripts resolve paths relative to this repository, so the old
`/workspace/imbalance` layout is not needed.

```sh
python src/pkg/pkg_features.py A
python src/pkg/pkg_features.py B
python src/pkg/pkg_leak_test.py
python src/pkg/pkg_model.py A
python src/pkg/pkg_model.py B
python src/pkg/export_dashboard.py A
python src/pkg/export_dashboard.py B
python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv
```

The model script creates the validation template if needed and updates the
variant forecast, its named copy in `submission/`, and the root submission file.
The leakage check exits with a nonzero status if future-data mutations change
features at or before a cutoff.

Use only the `A` feature/model/export commands for a package-only rebuild.
Variant B additionally requires `data/fingrid_{75,177,181,188,194,260,397,400}.parquet`
and `data/openmeteo_ecmwf_runs.parquet`. All are included locally from
`data_external.zip`. To restore the external data on another machine, extract
that archive at the repository root (it already contains the `data/` prefix).
Restore the official starter data separately into `data/`, including
`spot_nordpool.csv` and `actuals_validation.csv`.

To fetch the external inputs again, set `FINGRID_API_KEY` in your environment and run:

```sh
python src/fetch_fingrid.py 75 177 181 188 194 260 397 400
python src/fetch_openmeteo.py
```

The fetchers target the original competition dates and reuse cached responses.
No API key is needed when using the delivered local datasets.

## Data handling and submission scope

Nord Pool spot data is licensed for hackathon use only. Keep `data/`,
`work_private/`, API keys and local environments out of public repositories.
The `.gitignore` covers these locations. See [the report](REPORT.md) and
[the official licence terms](docs/STARTER_README.md#9-sources-and-licences).

Submit through the Since AI platform. The challenge also requests presentation
slides and a demo video; those were not present in the supplied project archives.

## Experiment 2: Chronos-2

The completed [foundation-model experiment](results/experiment_2_chronos/REPORT.md)
compares three zero-shot Chronos-2 configurations on all 672 validation origins.
The best Chronos price MAE is 44.9159 EUR/MWh, versus 42.0286 for the current
LightGBM B submission. Chronos spread-only coverage is closer to 80%, but its
pinball loss is worse. The original submission remains unchanged.
Scripts and isolated dependencies are in [experiments/](experiments/README.md).

## Experiment 3: TimesFM 3 and Timestral

The [latest-model benchmark](results/experiment_3_foundation_models/REPORT.md)
tests TimesFM 3 and Timestral 209M on the same 672 validation origins. Their best
MAEs are 43.5332 and 43.8162 EUR/MWh respectively, both using spread targets with
covariates. Both improve on Chronos but remain behind LightGBM B (42.0286).
The original submission is unchanged. TimesFM 3 is research-only under its
checkpoint licence; Timestral's release postdates validation, so its result is
retrospective. See the report for complete metrics and reproduction commands.

## Experiment 4: outage notices and additional weather

The [external-data experiment](results/experiment_4_external_data/REPORT.md)
adds publication-timestamped outage revisions and archived ICON/HARMONIE runs
to the existing LightGBM model. Validation MAE is 41.9542 with outages and
41.9510 with both additions, versus 42.0286 for the matched retrained control.
These small gains are not statistically clear under the paired day bootstrap,
and the control has the best pre-validation fold MAE. The submission is unchanged.
The report audits all six proposed source families, including those not tested
because access or historical publication provenance was not established.
