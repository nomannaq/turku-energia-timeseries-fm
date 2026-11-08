# Since AI 2026 – Turku Energia challenge: starter package

**Challenge:** forecast the Finnish imbalance price for the next 10 hours in 15-minute resolution
(40 quarter-hours), with uncertainty (p10, p50, p90), the direction of balancing and the probability of
exceptional prices.

Schema version 1.0. All timestamps are UTC (ISO 8601, suffix `Z`).

---

## 1. Quick start

1. Read sections 3 (schedule) and 4 (data availability rule).
2. Open `baseline.ipynb`. It loads the data, shows the helper function `rows_available_at()`, builds the
   baseline forecast for the validation period and runs the validator and the scoring script.
3. Build your model. Train it only on data up to 4 October 2026.
4. Fill in `sample_submission.csv` for the test period and save it as `forecast_backtest.csv`.
5. Run `validate_submission.py` (section 2) and correct every error.
6. Submit `forecast_backtest.csv` and the other deliverables through the Since AI platform (section 10).

New to electricity markets? See section 12 (Glossary) for the key terms.

### Package contents

| File | Content |
|---|---|
| `.gitignore` | keeps API keys, `baseline_output/` and the `data/` folder out of git (section 9) |
| `README.md` | this file |
| `forecast_origins.csv` | the 673 forecast origins of the test period |
| `sample_submission.csv` | submission template: 26,920 rows, identifiers filled in, forecast columns empty |
| `availability.csv` | publication delay of each source; `first_usable_origin_utc` is calculated from it |
| `data/inputs.parquet`, `data/inputs.csv` | input data 2026-05-31T21:00:00Z – 2026-11-02T06:45:00Z, long format (same data in both files) |
| `data/spot_nordpool.csv` | Nord Pool day-ahead spot price, long format. **Licence terms apply (section 9).** |
| `data/actuals_validation.csv` | actual values of the validation period for your own scoring |
| `data/quality_report.csv`, `data/quality_report.md` | row counts, gaps and removed identical duplicates per series |
| `validate_submission.py` | validator: checks the format of the submission |
| `score.py` | scoring script: calculates the metrics and converts them into points |
| `baseline.ipynb` | baseline forecast, helper function `rows_available_at()`, validation and scoring example |
| `build_dataset.py` | organiser script that built the data files (optional for you, section 8) |
| `make_templates.py` | organiser script that built the templates; also builds templates for other periods |
| `requirements.txt` | Python dependencies |

`baseline.ipynb` creates the folder `baseline_output/` when you run it. The folder is not part of the
distributed package.

### What to submit

One forecast file, `forecast_backtest.csv` (673 origins × 40 steps = 26,920 rows), and the other
deliverables listed in the challenge description. No example forecast is submitted.

---

## 2. Submission validation

The validator checks the structure and internal consistency of the submission. It does not evaluate
forecast quality or prove that the model is free from future-data leakage.

### 2.1 Before submitting

Complete these steps in order:

1. Generate the backtest forecast using `sample_submission.csv` as the template.
2. At every forecast origin, use only data rows with `first_usable_origin_utc <= forecast_origin_utc`
   (section 4).
3. Run the validation command.
4. Correct every error reported by the validator.
5. Submit `forecast_backtest.csv` through the Since AI platform together with the other deliverables
   (section 10).

A warning does not necessarily prevent submission. An error prevents the file from being accepted for
official scoring.

### 2.2 Validate the backtest submission

```bash
python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv
```

The validator checks that:

- all 13 columns are present and the values have the correct data types
- all 673 forecast origins are present, with exactly 40 rows each and 26,920 rows in total
- no unexpected forecast origins are included
- the combination of `forecast_origin_utc` and `step` is unique
- required values are not empty (`data_cutoff_utc` and `top_drivers` are optional and may be empty)
- `step` is an integer from 1 to 40 and `delivery_start_utc` matches the forecast origin and step
- `p10 <= p50 <= p90`
- every probability is between 0 and 1
- `p_up + p_down + p_none = 1`, allowing a tolerance of 0.001
- `p_spike_high + p_spike_low <= 1`, allowing a tolerance of 0.001
- if `data_cutoff_utc` is given, `data_cutoff_utc <= forecast_origin_utc`
- timestamps are UTC in the format `YYYY-MM-DDTHH:MM:SSZ`; the suffix `+00:00` is also accepted.
  Timestamps without a time zone, with another time zone, with fractional seconds or in another format
  (for example `04.10.2026 21:00` saved by a spreadsheet program) are rejected (`INVALID_TIMESTAMP`)
- the file is UTF-8 with a comma separator and a decimal point; a UTF-8 byte order mark is removed and
  CRLF line endings are accepted

### 2.3 Successful validation

Example output:

```text
VALIDATION PASSED

File: forecast_backtest.csv
Validator version: 0.1.0
Schema version: 1.0
Forecast origins: 673 / 673
Rows: 26920 / 26920
File: OK
Required columns: OK
Data types: OK
Forecast origins: OK
Rows: OK
Duplicate rows: OK
Missing required values: OK
Steps: OK
Timestamps: OK
Quantiles: OK
Probabilities: OK
Direction probabilities: OK
Spike probabilities: OK
Data cut-off rule: OK (optional column, 26920 values given)
Errors: 0
Warnings: 0

The submission is ready for scoring.
```

The validator exits with return code `0` when all required checks pass.

### 2.4 Failed validation

Example output:

```text
VALIDATION FAILED

File: forecast_backtest.csv
Validator version: 0.1.0
Schema version: 1.0
Forecast origins: 670 / 673
Rows: 26814 / 26920
File: OK
Required columns: OK
Data types: OK
Forecast origins: FAILED
Rows: FAILED
Duplicate rows: FAILED
Missing required values: FAILED
Steps: OK
Timestamps: OK
Quantiles: OK
Probabilities: OK
Direction probabilities: FAILED
Spike probabilities: FAILED
Data cut-off rule: OK (optional column, 26814 values given)
Errors: 6
Warnings: 0

ERROR [MISSING_VALUES]
p50 contains 80 empty value(s).
First empty value: line 14722 (forecast_origin_utc=2026-10-20T08:00:00Z, step=1)

ERROR [DUPLICATE_ROWS]
14 duplicate forecast_origin_utc + step combination(s) found.
First duplicate: line 26802 (forecast_origin_utc=2026-10-10T03:00:00Z, step=1)

ERROR [MISSING_ORIGINS]
3 forecast origin(s) are missing.
First missing origin: 2026-10-08T14:00:00Z

ERROR [ROW_COUNT]
The file has 26814 data rows; expected 26920 (673 origins x 40 steps).

ERROR [INVALID_PROBABILITY_SUM]
27 row(s) have p_up + p_down + p_none outside 1 +/- 0.001.
First invalid row: line 4088 (forecast_origin_utc=2026-10-09T06:00:00Z, step=7), sum=1.150926

ERROR [INVALID_SPIKE_SUM]
5 row(s) have p_spike_high + p_spike_low greater than 1 (tolerance 0.001).
First invalid row: line 7343 (forecast_origin_utc=2026-10-12T15:00:00Z, step=22), sum=1.100000

The submission is not ready for scoring.
```

The validator exits with return code `1` when one or more errors are found. The report gives an error
code, a count and the first example for each error category.

**Follow-up errors:** one problem can cause several errors. For example, an invalid timestamp also
causes `MISSING_ROWS` or `MISSING_ORIGINS`, because the row cannot be matched to an origin. Correct the
first error first and run the validator again.

### 2.5 Official validation and scoring

- The organiser downloads the submissions from the Since AI platform and runs the same locked validator
  version. Only a file that passes the structural checks proceeds to official scoring.
- All accepted files are scored with the same locked `score.py` version, the same test-period actual
  values and the same spike margin (50 EUR/MWh).
- Passing the validator does not mean that the forecast follows the data availability rule or reaches a
  good score. Top solutions' source code is reviewed to verify that no data was used before its
  `first_usable_origin_utc`. The review may use AI tools.

---

## 3. Schedule

| Item | Value |
|---|---|
| Data in the package | 1 Jun 2026 00:00 – 2 Nov 2026 08:45 Finnish time (`2026-05-31T21:00:00Z` – `2026-11-02T06:45:00Z`, quarter-hour starts) |
| Training data | 1 Jun – 4 Oct 2026. Train your model only on this data. |
| Validation period | 7 Sep – 4 Oct 2026, part of the training data, for your own testing. Actual values in `data/actuals_validation.csv`. |
| Test period | 5 Oct – 1 Nov 2026 (Finnish time). Input data is in the package but must not be used for training. |
| Forecast origins | every full hour, 673 origins (`2026-10-04T21:00:00Z` – `2026-11-01T21:00:00Z`; daylight saving time ends on 25 Oct) |
| Horizon | 10 hours = 40 quarter-hours; `step 1` starts at the forecast origin |
| Last forecast quarter-hour | `2026-11-02T06:45:00Z`; quarter-hours after the test period are scored too |

Local time (Europe/Helsinki) is used only for presentation and publication times. On 25 Oct 2026 the
local hour 03:00–04:00 occurs twice; UTC timestamps are unambiguous.

---

## 4. Data availability rule

```text
Data availability rule

Every data row in inputs.parquet / inputs.csv and spot_nordpool.csv has the column
first_usable_origin_utc: the first full-hour forecast origin at or after the time
the value was published. At forecast origin t, your model may use a row only if
first_usable_origin_utc <= t. If data is used before it is available, the
submission is not a forecast and its automatic score is 0.

availability.csv lists the publication delay of each source. first_usable_origin_utc
has been calculated from it, and the row-level column is binding.

The package contains input data up to 2026-11-02T06:45:00Z, including the test
period. Train your model only on data up to the training cut-off date, 4 October 2026.

Helper function: rows_available_at(data, origin_utc)  (in baseline.ipynb)
- Input: long-format rows from inputs.csv / inputs.parquet and/or spot_nordpool.csv,
  with timestamp_utc and first_usable_origin_utc parsed as UTC timestamps, and a
  forecast origin in UTC.
- Output: a wide table with one row per timestamp_utc and one column per series_id,
  containing only rows with first_usable_origin_utc <= origin_utc. Values that are
  not yet usable are missing.
- The table can contain future timestamps, for example day-ahead forecasts and spot
  prices that had already been published at the origin.
- Use it to build the model inputs for each forecast origin. It is the simplest way
  to follow the data availability rule.
```

**Exception: calendar series.** The calendar is known in advance, so all calendar rows have
`first_usable_origin_utc = 2026-05-31T21:00:00Z` (start of the data period).

**Hourly source data.** Some ENTSO-E series are published per hour (currently the Estonian load
forecast). They are repeated for each of the four quarter-hours, and `first_usable_origin_utc` is
calculated from the end of the hour. All four quarter-hours therefore become usable at the same time.

The validator cannot detect a breach of this rule. Top solutions' source code is reviewed to verify that
no data was used before its `first_usable_origin_utc`. The review may use AI tools.

---

## 5. Schema

### 5.1 Submission file `forecast_backtest.csv` (schema version 1.0)

| Column | Type | Required | Rule |
|---|---|---|---|
| `forecast_origin_utc` | UTC timestamp | yes | full hour listed in `forecast_origins.csv` |
| `delivery_start_utc` | UTC timestamp | yes | origin + (step − 1) × 15 min |
| `step` | integer 1–40 | yes | quarter-hour number |
| `p10`, `p50`, `p90` | decimal, EUR/MWh | yes | `p10 <= p50 <= p90`; p50 is the point forecast |
| `p_up`, `p_down`, `p_none` | decimal 0–1 | yes | probabilities of the dominating mFRR direction (Fingrid dataset 369): up, down, none; sum 1 (±0.001) |
| `p_spike_high`, `p_spike_low` | decimal 0–1 | yes | probability that the imbalance price is above spot + 50 EUR/MWh (`high`) or below spot − 50 EUR/MWh (`low`); sum <= 1 |
| `data_cutoff_utc` | UTC timestamp | no | timestamp of the latest data used; if given, `<= forecast_origin_utc` |
| `top_drivers` | text | no | semicolon-separated names of the main drivers, e.g. `da_price;wind_fc_change`; not scored |

All 13 columns are in the template, in this order. The cells of the optional columns may be empty. The
forecast columns of the template are empty cells; fill all of them. Example rows (values invented):

```csv
forecast_origin_utc,delivery_start_utc,step,p10,p50,p90,p_up,p_down,p_none,p_spike_high,p_spike_low,data_cutoff_utc,top_drivers
2026-10-05T06:00:00Z,2026-10-05T06:00:00Z,1,42.10,61.30,88.50,0.55,0.30,0.15,0.04,0.01,2026-10-05T05:00:00Z,da_price;mfrr_up_lag
2026-10-05T06:00:00Z,2026-10-05T06:15:00Z,2,40.80,60.10,92.00,0.52,0.31,0.17,0.05,0.01,,
```

**Change policy:** schema version 1.0 is locked when the package is published. If a change is
unavoidable, it is announced to all teams at the same time with a new version number.

### 5.2 Data files

**How the imbalance price is formed.** The imbalance price is set separately for each quarter-hour, and
the regulation direction decides how it is calculated: the dominating direction is determined from mFRR
activations (`fi_mfrr_direction`), and in up-regulation the price is above the day-ahead spot price, in
down-regulation below it. Since 1 June 2026, when activated mFRR and the fulfilled aFRR demand in Finland
are both in the dominating direction, the price is the volume-weighted average of their energy prices,
not their sum. If a quarter-hour has no dominating mFRR direction, the price is based on the spot price,
even if aFRR was activated; other situations follow separate pricing rules.

`data/inputs.parquet` / `data/inputs.csv` and `data/spot_nordpool.csv` are in long format: one row per
series and quarter-hour.

| Column | Content |
|---|---|
| `series_id` | series identifier, see the table below |
| `source` | `fingrid`, `entsoe`, `nordpool` or `calendar` |
| `dataset_id` | Fingrid dataset number, ENTSO-E article and area (e.g. `17.1.g:SE1`), or empty |
| `timestamp_utc` | start of the quarter-hour (UTC); for forecasts and the spot price the delivery quarter-hour |
| `value` | value |
| `unit` | unit, e.g. `EUR/MWh`, `MW` |
| `first_usable_origin_utc` | first forecast origin at which the row may be used (section 4) |

`series_id` + `timestamp_utc` is unique.

| series_id | Source | dataset_id | Unit | Note |
|---|---|---|---|---|
| `fi_imbalance_price` | Fingrid | 319 | EUR/MWh | imbalance price (the forecast target) |
| `fi_mfrr_up_price` / `fi_mfrr_down_price` | Fingrid | 244 / 106 | EUR/MWh | mFRR energy marginal price |
| `fi_mfrr_up_activated` / `fi_mfrr_down_activated` | Fingrid | 375 / 376 | MW | activated mFRR balancing regulation (max power of the quarter-hour) |
| `fi_mfrr_direction` | Fingrid | 369 | -1/0/1 | dominating mFRR direction: 1 up, -1 down, 0 none |
| `fi_afrr_up_price` / `fi_afrr_down_price` | Fingrid | 347 / 348 | EUR/MWh | **sparse:** no row when there was no aFRR activation |
| `fi_consumption` / `fi_production` | Fingrid | 124 / 74 | MWh/h | measured |
| `fi_consumption_fc_da`, `fi_wind_fc_da`, `fi_solar_fc_da` | Fingrid | 165, 246, 247 | MW, MWh/h | day-ahead forecasts, usable from D-1 13:00 Finnish time |
| `fi_se3_flow`, `fi_no4_flow`, `fi_se1_flow`, `fi_ee_flow` | Fingrid | 61, 57, 60, 55 | MW, MWh/h | measured flow, positive = export from Finland |
| `se1_/se3_/ee_imbalance_price_long`, `_short` | ENTSO-E | `17.1.g:<area>` | EUR/MWh | `_long` = category A04 (excess balance), `_short` = A05 (insufficient balance); identical under single pricing |
| `se1_load_fc_da`, `se3_load_fc_da`, `ee_load_fc_da` | ENTSO-E | `6.1.b:<area>` | MW | day-ahead total load forecast; EE is hourly at source |
| `fi_ee_capacity_da` / `ee_fi_capacity_da` | ENTSO-E | `11.1:FI-EE` / `11.1:EE-FI` | MW | offered day-ahead capacity, usable from D-1 14:00 Finnish time |
| `cal_weekday_local`, `cal_hour_local`, `cal_is_holiday_fi`, `cal_is_dst` | calendar | – | – | local Finnish time (`cal_weekday_local` 0 = Monday); holidays = official public holidays + Midsummer Eve |
| `fi_spot_price` (in `spot_nordpool.csv`) | Nord Pool | – | EUR/MWh | day-ahead spot price, Finland; licence terms in section 9 |

**Missing source values are left missing (no row). They are not filled.** `data/quality_report.md`
lists the missing quarter-hours and gaps of each series.

Not in the package: ENTSO-E outages (10.1), ENTSO-E data for NO4 (not published), transfer capacities on
the Nordic borders (flow-based, not published as capacities), ENTSO-E day-ahead prices (not on the free
re-use list), continuously updated forecasts (only the latest version is stored), weather data and
Fingrid commercial flows.

### 5.3 Actual values

`data/actuals_validation.csv` (validation period) has the columns `delivery_start_utc`,
`imbalance_price`, `spot_price`, `direction` (`up`, `down`, `none`; source Fingrid 369) and
`first_usable_origin_utc` (the latest of the row's sources). The test-period actual values are kept by
the organiser.

---

## 6. Metrics and scoring

```text
Metrics and scoring

The organiser calculates the official results for the test period with score.py.
Total score = 0.8 x Level 1 + 0.2 x Level 2.

Level 1 (0-100 points):
- Accuracy, 30 points: mean absolute error (MAE) of p50.
- Calibration, 20 points: mean pinball loss of p10 and p90. The coverage of the
  P10-P90 interval is reported as a supporting metric (target about 80 %).
- Exceptional prices, 20 points: mean of the Brier scores of p_spike_high and
  p_spike_low. A quarter-hour is exceptionally high when the imbalance price is
  above the spot price + 50 EUR/MWh, and exceptionally low when it is below the
  spot price - 50 EUR/MWh.
- Direction, 10 points: three-class Brier score of p_up, p_down and p_none.
- Reasoning chain, 20 points: the jury scores the description of the reasoning
  chain on a scale of 0-5, based on the evaluation report, presentation slides and
  demo video. The score is divided by 5.

For the four automatic sub-criteria (lower metric value is better):
  points = weight x clip((B - T) / (B - best), 0, 1)
  B = metric of the baseline forecast, T = metric of the team,
  best = best metric among the accepted submissions.
The baseline level gives 0 points, the best team gets full points, and a team
that is not better than the baseline gets 0 points. If no team beats the baseline
in a sub-criterion, all teams get 0 points for it.

Level 2 (0-100 points): the jury scores the situational view and usability on a
scale of 0-5. Points = 100 x score / 5.

You can run score.py on the validation period (actuals_validation.csv) to compare
your model with the baseline. Your final points depend on the best submission, so
they cannot be calculated before all submissions have been scored.
```

Additional rules:

- **No minimum requirements.** A result weaker than the baseline gives 0 points for that sub-criterion.
  The MAE compared with the baseline (difference and ratio) and the P10–P90 coverage are reported for
  information only.
- **Ties with the baseline:** if `B - best <= 1e-6 x |B|`, the difference is treated as zero and every
  team gets 0 points for that sub-criterion. The tolerance is the `score.py --points` parameter
  `--zero-tolerance` (default `1e-6`).
- **Spike margin:** 50 EUR/MWh in the official scoring (`score.py --spike-margin`, default 50).
- **Baseline:** the reference level B is the baseline of `baseline.ipynb` for the test period, scored
  by the organiser with the same `score.py` version.
- Formulas: pinball loss for quantile q is `mean(max(q × (y − ŷ), (q − 1) × (y − ŷ)))`; two-class Brier
  `mean((p − o)²)`; three-class Brier `mean(Σ_k (p_k − o_k)²)` over up, down and none (range 0–2).

---

## 7. Run instructions

### 7.1 Environment

Python 3.11 or newer is recommended (tested on Windows with 3.11, 3.14 and 3.15, and on Linux with 3.10); 3.10 also works but has reached end of life.

On Windows, extract the package to a short path, for example `C:\sinceai\`. Deep folders (for example
inside OneDrive) can exceed the Windows path length limit of 260 characters.

Optional virtual environment, created in the package folder:

| Shell | Create | Activate |
|---|---|---|
| Windows PowerShell | `py -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |
| Windows cmd | `py -m venv .venv` | `.venv\Scripts\activate.bat` |
| macOS / Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |

If PowerShell reports that running scripts is disabled, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
```

The second command is needed only for `baseline.ipynb`; Jupyter is not in `requirements.txt`.
On Windows without an activated virtual environment, use `py` instead of `python`.
Write every command on one line; the commands in this README work in PowerShell, cmd and bash.

`validate_submission.py`, `score.py` and `make_templates.py` need only pandas and numpy. `pyarrow` is
needed for `inputs.parquet` (use `inputs.csv` without it) and `requests` only for `build_dataset.py`.
The scripts make no network connections, except `build_dataset.py`.

### 7.2 Commands

Run the commands in the package folder.

Format check of the test-period submission:

```bash
python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv
```

Metrics for the validation period (compare with the baseline with `--baseline-summary`):

```bash
python score.py --submission my_validation.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir results/my_model --baseline-summary baseline_output/validation/score/score_summary.csv
```

Templates for another period, for example the validation period (write them to another folder):

```bash
python make_templates.py --test-start 2026-09-07 --test-end 2026-10-04 --output-dir my_validation_templates
```

Help and version of each script:

```bash
python validate_submission.py --help
python score.py --version
```

Return codes: `0` = success, `1` = error (validation failed or invalid input), `2` = invalid
command-line arguments. `score.py` writes `score_summary.csv`, `score_summary.md` and
`score_by_horizon.csv` (results by forecast hour 1–10). Rows without actual values are skipped and the
counts are reported.

### 7.3 Baseline notebook

Open `baseline.ipynb` in Jupyter in the package folder and run all cells. To start Jupyter, run
`python -m notebook` in the package folder. The notebook:

- loads `data/inputs.parquet` (or `inputs.csv`) and `data/spot_nordpool.csv`
- builds the validation-period origins and template with `make_templates.py` into
  `baseline_output/validation/` (the package templates are not changed; the notebook checks this with
  SHA-256)
- writes the baseline forecast `baseline_output/validation/forecast_backtest.csv`, validates it and
  scores it into `baseline_output/validation/score/`.

The notebook creates the folder `baseline_output/`; it is not part of the distributed package
(it is listed in `.gitignore`).

Baseline method: p50 = the latest imbalance price usable at the origin, the same for all 40 steps;
p10/p90 = p50 plus the 10 % and 90 % quantiles of historical persistence errors for each step;
direction and exceptional price probabilities = historical shares. The history contains only rows and
(origin, target) pairs whose `first_usable_origin_utc` is at or before the first origin of the period.

The validation period's last origins reach into the test period; their last steps have no actual
values and are not scored. Do not print or publish rows of `spot_nordpool.csv`.

---

## 8. APIs and external data

```text
APIs and external data

- Fingrid Open Data: an API key is required. Rate limits apply.
- ENTSO-E Transparency Platform: request API access early. Processing can take
  several working days.
- Weather forecasts: the package contains only Fingrid's day-ahead consumption,
  wind power and solar power forecasts. You may use the Open-Meteo Single Runs API
  at your own responsibility. Add the publication delay to the model initialisation
  time before using a run. The Open-Meteo Historical Forecast API is not
  recommended, because it leaks future information.
- Finnish Meteorological Institute data is not included in the package. You may
  fetch it yourself.
- Commercial cross-border flows (Fingrid) include day-ahead and intraday trades and
  are updated hourly. If you use them, use them only as realised values.
- You are responsible for the licences of any data you add, and you must list all
  sources in your submission.
```

### Running build_dataset.py yourself (optional)

You do not need to run `build_dataset.py`; the data files are ready.

- API keys come from the environment variables `FINGRID_API_KEY` and `ENTSOE_API_TOKEN`, or from a file
  given with `--env-file`. There is no default key file. Never commit key files: `.gitignore` excludes
  `.secrets/` and `*.env`.
- Example: `python build_dataset.py --start 2026-06-01 --end 2026-11-02T06:45:00Z --env-file keys.env`
- Raw responses are cached in `../data_cache/`, so an interrupted run can be restarted.
- By default, `build_dataset.py` writes the organiser file `actuals_test.csv` to `../organizer_only/`,
  outside the package folder. Change the folder with `--organizer-dir`.
- The Nord Pool part needs the organiser's spot file (`--spot-file`) and does not work without it.
- Duplicate source rows with identical values are removed and counted in the quality report; duplicates
  with different values stop the run.

---

## 9. Sources and licences

```text
Sources and licences

- Fingrid Open Data, licensed under CC BY 4.0.
- ENTSO-E Transparency Platform, licensed under CC BY 4.0. Only data on ENTSO-E's
  list of data available for free re-use is included. Modifications: converted to
  UTC and to 15-minute series.
- Nord Pool spot price data is licensed to Turku Energia. It is shared with the
  teams only for use during the hackathon. Terms: do not share the data further and
  do not upload it to public repositories; your code may be public, the data may
  not; delete the data after the event.
  These terms apply to spot_nordpool.csv and to the spot_price column of
  actuals_validation.csv.
- Other data: you are responsible for the licences of any data you use.
```

---

## 10. Submission

- Submit through the **Since AI platform**.
- Submit one forecast file, `forecast_backtest.csv`. No example forecast is submitted.
- Submit the other deliverables listed in section 07 of the challenge description: source code and
  start-up instructions, evaluation report, list of data sources and components with their licences,
  presentation slides and demo video.
- Run the validator before submitting.
- The deadline and file size limits are given in the Since AI event instructions.

---

## 11. Questions

Ask questions in the Turku Energia channel of the Since AI Discord.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| Imbalance price | The price at which the difference between a market party's planned and actual electricity production or consumption is settled, set separately for each quarter-hour. It is the forecast target (`fi_imbalance_price`, Fingrid dataset 319). |
| mFRR | Manual frequency restoration reserve: reserve capacity that is activated manually to balance the power system. mFRR activations determine the dominating regulation direction, and mFRR energy prices affect the imbalance price. |
| aFRR | Automatic frequency restoration reserve: reserve capacity that is activated automatically. aFRR energy prices can affect the imbalance price; the aFRR price series have no row when there was no activation. |
| Regulation direction (up / down / none) | Up-regulation means more production or less consumption, down-regulation the opposite. The dominating direction of each quarter-hour is determined from mFRR activations (Fingrid dataset 369: 1 up, −1 down, 0 none); none means that there is no dominating mFRR direction. |
| ENTSO-E long / short imbalance price | Imbalance prices of the neighbouring areas SE1, SE3 and EE (ENTSO-E article 17.1.g): `_long` = category A04 (excess balance), `_short` = A05 (insufficient balance). Under single pricing the two values are identical. |
| Spot price | The Finnish area price formed on the day-ahead market (Nord Pool) for the next day; not to be confused with intraday or imbalance prices. It is the reference level for exceptional prices and the basis of the imbalance price when there is no dominating mFRR direction. |
| Cross-border flow | Measured electricity flow between Finland and a neighbouring area (SE1, SE3, NO4, EE), from Fingrid Open Data. |
| Forecast origin | The full hour at which a forecast is made (`forecast_origin_utc`). The test period has 673 hourly origins, and step 1 is the quarter-hour that starts at the origin. |
| Quarter-hour | The 15-minute period for which the imbalance price is set and forecast. Timestamps give the start of the quarter-hour in UTC; each forecast covers 40 quarter-hours (10 hours). |
| P10 / P50 / P90 | Quantile forecasts of the imbalance price; p50 is the point forecast. In a well-calibrated forecast, about 80 % of the actual prices fall within the P10–P90 interval. |
| `first_usable_origin_utc` | Column on every data row: the first full-hour forecast origin at or after the time the value was published. At origin t, a row may be used only if `first_usable_origin_utc <= t` (section 4). |
| Backtest | Forecasts made for every forecast origin of a past period, using at each origin only data that would have been available at that time. `forecast_backtest.csv` contains the backtest for the test period. |
| Baseline | The simple reference forecast in `baseline.ipynb` (p50 = latest imbalance price usable at the origin). Its test-period metric is the reference level B; a team that is not better than the baseline gets 0 points for that sub-criterion. |
| Spike (exceptional price) | A quarter-hour where the imbalance price is above the spot price + 50 EUR/MWh (high) or below the spot price − 50 EUR/MWh (low). Forecast as the probabilities `p_spike_high` and `p_spike_low`. |
