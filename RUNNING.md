# Run Turku GridCast

[Project overview](README.md) · [Evaluation report](EVALUATION_REPORT.md) · [Data sources and licences](DATA_SOURCES_AND_LICENSES.md)

## Choose a workflow

| Goal | Sections | Training required? |
|---|---|---|
| View the delivered forecasts and metrics | [1. Setup](#1-set-up-and-launch-the-dashboard), [2. Dashboard](#2-explore-the-dashboard) | No |
| Review experiments and export charts | [1. Setup](#1-set-up-and-launch-the-dashboard), [3. Notebook](#3-review-the-notebook-and-generate-charts) | No, by default |
| Check submission format and reproduce scores | [1. Setup](#1-set-up-and-launch-the-dashboard), [4. Scoring](#4-validate-and-score-the-final-forecasts) | No |
| Rebuild the final model outputs | [5. Training](#5-rebuild-the-final-models-optional) | Yes; substantial CPU time |
| Regenerate the evaluation PDF | [6. Report](#6-regenerate-the-evaluation-pdf) | No |

Run all commands from the **repository root**, the directory containing `score.py`, `README.md` and `dashboard/`. Commands after setup use `python` from the activated `.venv`, except the training workflow, which names each environment explicitly.

## 1. Set up and launch the dashboard

### Existing environment on the supplied computer

No installation is needed if the delivered `.venv` still works:

```sh
source .venv/bin/activate
python dashboard/server.py
```

### Fresh computer

Use Python 3.11 or newer for the dashboard. Copy the project with its saved forecast and result files, then install the dashboard dependencies.

macOS / Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r dashboard/requirements.txt
```

Windows PowerShell (activation without changing PowerShell policy):

```powershell
py -m venv .venv
$env:Path = "$PWD\.venv\Scripts;$env:Path"
python -m pip install -r dashboard/requirements.txt
```

Check `python --version` and `python -m pip --version` if unsure which environment is active. In a new terminal, activate/select this environment again.

### Restore local data

Obtain the official starter data separately; raw datasets are not shipped in the source-code archive. Put these files in `data/`:

| File | Used for |
|---|---|
| `data/inputs.parquet` | Input availability and reconstructed realised outcomes |
| `data/actuals_validation.csv` | Official validation outcomes |
| `data/spot_nordpool.csv` | Reconstructing local test outcomes if the private label file is absent |

Retain the following saved project artifacts when copying the code:

- `submission/final_ensemble/forecast_backtest.csv` and `validation_forecast.csv`
- Root `forecast_backtest.csv` and `submission/validation_forecast_B.csv` for the previous-model comparison
- `submission/dashboard/test_B.csv` and `submission/dashboard/validation_B.csv` for reference explanations
- `results/final_ensemble/validation_comparison.csv` and `results/final_ensemble/local_test_backtest/comparison.csv`
- Root `score.py`; keep `forecast_origins.csv` for the submission validator

For the notebook, also retain the complete saved `results/` tree. Keep `data/`, `work_private/` and credentials private; see [licence terms](DATA_SOURCES_AND_LICENSES.md).

### Launch

```sh
python dashboard/server.py
```

Open **http://127.0.0.1:8765**. Keep the terminal running; **Ctrl+C** stops the server. If the port is busy:

```sh
python dashboard/server.py --port 8766
```

Then open http://127.0.0.1:8766. No API keys, model downloads, training, Node packages or Streamlit installation are needed. The server binds to localhost only; public deployment is not included.

## 2. Explore the dashboard

1. Read the opening validation and local-test tables. These are full-period retrospective metrics, independent of the replay origin.
2. Choose September validation or October test. Select an origin, move the slider or press **Play replay**.
3. Inspect the 40-quarter forecast: P50 and the P10–P90 interval. The dashed curve is the previous hourly forecast aligned to the same delivery times.
4. Click a quarter in the chart or select it from the dropdown to update risk probabilities and reference explanations.
5. Toggle **Show retrospective actuals** to inspect realised prices after the fact.
6. Review spike-risk and forecast-revision prompts. These suggest expert review, not automatic trading or plant control.
7. Under performance, choose **Known at selected origin** to include only already-published outcomes. To reproduce complete scores, choose **Retrospective period end** and **Whole available period**.
8. Use **Download forecast** to export the selected origin's original 40 rows.

The final model is five-seed LightGBM plus Chronos-2. Driver/freshness panels use the older single-seed LightGBM export and are labelled accordingly; they do not explain the full ensemble. If `work_private/final_ensemble/local_actuals_test.csv` is missing, the dashboard reconstructs test outcomes in memory from local starter data. It does not serve raw spot prices.

## 3. Review the notebook and generate charts

Complete section 1's environment and data setup, then install notebook support:

```sh
python -m pip install -r experiments/requirements-notebook.txt
```

Create the local test-label file needed by the notebook if it is absent:

```sh
python -c "from pathlib import Path; Path('work_private/final_ensemble').mkdir(parents=True, exist_ok=True)"
python src/pkg/score_local_test.py
```

This checks reconstructed labels against official validation labels, scores the frozen test forecasts and refreshes local test reports. It does not fit or select models.

Open [Complete_Experiment.ipynb](Complete_Experiment.ipynb) in PyCharm or an existing Jupyter frontend and select this project's `.venv` kernel. If the kernel is not listed, register it:

```sh
python -m ipykernel install --user --name turku-gridcast --display-name "Turku GridCast (.venv)"
```

For a standalone JupyterLab interface, install and launch it:

```sh
python -m pip install jupyterlab
python -m jupyter lab Complete_Experiment.ipynb
```

Use **Run All** to review saved experiments, independently score predictions and generate eight inline figures plus PNGs under `results/notebook_charts/`. The charts cover baseline results, foundation models, external data, tuning, the final blend, local test results, horizon diagnostics and forecast uncertainty.

Optional training and inference flags default to false. Leave them false for the review workflow. Full notebook execution has **not been verified** because the IDE kernel failed to start during creation; dashboard/scorer verification is separate. See [troubleshooting](#7-troubleshooting-and-verification) for kernel issues.

## 4. Validate and score the final forecasts

Use the environment from section 1. **The final upload is `submission/final_ensemble/forecast_backtest.csv`.** The root file of the same name is the older fallback.

### Check the test-file format

```sh
python validate_submission.py --backtest submission/final_ensemble/forecast_backtest.csv --origins forecast_origins.csv
```

Expected: 26,920 rows across 673 origins, with zero errors and warnings. This validates format and constraints; it does not measure accuracy.

### Score validation against official actuals

```sh
python score.py --submission submission/final_ensemble/validation_forecast.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir results/recheck_final_validation
```

Results are written under `results/recheck_final_validation/`. The baseline comparison and experiment history are saved in `results/final_ensemble/validation_comparison.csv`.

### Score the historical test period locally

```sh
python -c "from pathlib import Path; Path('work_private/final_ensemble').mkdir(parents=True, exist_ok=True)"
python src/pkg/score_local_test.py
```

Outputs: private reconstructed labels under `work_private/final_ensemble/` and score reports under `results/final_ensemble/local_test_backtest/`. The script checks that forecast hashes remain unchanged.

| Metric | Final validation | Final local test |
|---|---:|---:|
| MAE, EUR/MWh | 41.5988 | 26.7028 |
| Mean P10/P90 pinball | 11.8864 | 6.8302 |
| P10–P90 coverage | 73.28% | 80.72% |
| Spike Brier | 0.0908 | 0.0567 |
| Direction Brier | 0.3929 | 0.3937 |

The test period has been backtested locally. Its labels are reconstructed from package inputs; the official organiser score remains pending. Different-period loss levels are not evidence of model improvement. See the [evaluation report](EVALUATION_REPORT.md) for matched comparisons.

## 5. Rebuild the final models (optional)

This is separate from viewing and scoring saved forecasts. The orchestrator expects macOS/Linux paths `.venv/bin/python` and `.venv-chronos/bin/python`; use WSL on Windows or adapt the source paths. Keep a copy of delivered artifacts before rebuilding: this workflow writes ensemble outputs and uses cached intermediate results.

### A. Install the two environments

Reuse `.venv` if it already exists; otherwise create it. Python 3.12 must be installed separately for the isolated Chronos environment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.12 -m venv .venv-chronos
.venv-chronos/bin/python -m pip install -r experiments/requirements-chronos.txt
.venv-chronos/bin/python experiments/download_chronos_2.py
```

The last command downloads the pinned public checkpoint to `work_private/models/chronos-2/`; it requires network access. On macOS, if LightGBM reports a missing OpenMP runtime, install it with `brew install libomp`.

### B. Restore training inputs

Keep the official starter files from section 1 and restore all of these external inputs:

- `data/fingrid_75.parquet`, `fingrid_177.parquet`, `fingrid_181.parquet`, `fingrid_188.parquet`
- `data/fingrid_194.parquet`, `fingrid_260.parquet`, `fingrid_397.parquet`, `fingrid_400.parquet`
- `data/openmeteo_ecmwf_runs.parquet`

If you have the original handoff's `data_external.zip`, extract it at the repository root; the archive already contains the `data/` prefix. This archive is not part of the source-code distribution. Restoring the original data is preferable for exact reproduction; later provider revisions can change a fresh download.

Alternatively, with `FINGRID_API_KEY` set in your shell environment, fetch the original competition date range:

```sh
.venv/bin/python src/fetch_fingrid.py 75 177 181 188 194 260 397 400
.venv/bin/python src/fetch_openmeteo.py
```

These commands require network/provider access and reuse cached responses. Credentials are not included in the project.

### C. Build features and run the ensemble

```sh
.venv/bin/python src/pkg/pkg_features.py B
.venv/bin/python src/pkg/run_final_ensemble.py
```

The feature builder creates `work_private/features_B.parquet`. The runner trains five LightGBM seeds, runs Chronos-2, evaluates rolling folds, selects blend weights and runs leakage/verification checks. It uses both environments and may take substantial CPU time. Final forecasts are written to `submission/final_ensemble/`; evidence is written to `results/final_ensemble/`. Repeat section 4 after completion.

Historical A/B rebuilds and research experiments have different entry points. See [experiments/README.md](experiments/README.md) and the [original single-model report](REPORT.md). Older `pkg_model.py` commands can overwrite the root fallback forecasts; they are not required for the final ensemble workflow.

## 6. Regenerate the evaluation PDF

With the main environment active and saved result CSVs present:

```sh
python -m pip install reportlab matplotlib pandas numpy
python scripts/build_evaluation_report.py
```

Outputs: `output/pdf/Turku_GridCast_Evaluation_Report.pdf` (seven pages), charts under `output/pdf/charts/` and [EVALUATION_REPORT.md](EVALUATION_REPORT.md). The builder overwrites these generated artifacts; make narrative changes in the builder before regenerating. [DATA_SOURCES_AND_LICENSES.md](DATA_SOURCES_AND_LICENSES.md) is maintained separately.

## 7. Troubleshooting and verification

| Symptom | Action |
|---|---|
| `ModuleNotFoundError` | Activate `.venv` and install the requirements for the workflow you are running. |
| Connection refused | Start `dashboard/server.py`, keep its terminal open and use the printed port. |
| Port already in use | Add `--port 8766` and open that port in your browser. |
| Missing input file | Restore the named file using section 1 or section 5; viewing forecasts does not require training. |
| Empty historical metrics | Move to a later origin or choose retrospective evaluation; early windows may have no published outcomes. |
| No previous forecast at the first origin | Expected: there is no earlier hourly forecast in that period. |
| Replay metrics differ from the headline | Select retrospective period end and whole available period. A 28-day window omits one of the 673 test origins. |
| PyCharm kernel fails / `JupyterIgnoreNotificationException` | Select `.venv`, configure its local Jupyter server or use the standalone JupyterLab instructions in section 3. |
| `libomp.dylib` missing during training | Install the macOS OpenMP runtime as described in section 5. |

Run dashboard checks from the activated environment:

```sh
python dashboard/test_dashboard.py
```

Optional JavaScript syntax check, if Node is already installed:

```sh
node --check dashboard/static/app.js
```

Dashboard tests check forecast preservation, full-period metric parity, future-outcome invariance for as-of evaluation, empty initial windows and test-label reconstruction. Desktop Chrome checks covered metric tables, advancing origins, actual overlays, risk panels and history-window changes. Mobile layouts and external deployment remain unverified.
