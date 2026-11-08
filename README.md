# Turku GridCast

Ten-hour forecasts of Finnish electricity imbalance prices, with uncertainty intervals and direction/spike probabilities. Built for the Turku Energia challenge at Since AI 2026 using five-seed LightGBM and Chronos-2.

**Start with [RUNNING.md](RUNNING.md): setup, dashboard, notebook, scoring and model reproduction in one guide.**

## Choose your task

| I want to… | Start here |
|---|---|
| Open the interactive dashboard | [Setup and launch](RUNNING.md#1-set-up-and-launch-the-dashboard) |
| Explore forecasts and actual results | [Dashboard walkthrough](RUNNING.md#2-explore-the-dashboard) |
| Review experiments and generate charts | [Notebook instructions](RUNNING.md#3-review-the-notebook-and-generate-charts) |
| Validate or rescore the final forecasts | [Evaluation commands](RUNNING.md#4-validate-and-score-the-final-forecasts) |
| Rebuild the final models | [Training instructions](RUNNING.md#5-rebuild-the-final-models-optional) |
| Read the method and findings | [Evaluation report](EVALUATION_REPORT.md) · [Seven-page PDF](output/pdf/Turku_GridCast_Evaluation_Report.pdf) |
| Check sources and usage terms | [Data sources and licences](DATA_SOURCES_AND_LICENSES.md) |

On the supplied computer, run from this directory:

```sh
.venv/bin/python dashboard/server.py
```

Open **http://127.0.0.1:8765**. For a fresh computer, follow the setup guide first. The dashboard uses Python and HTML/JavaScript; it is not a Streamlit app. Viewing saved forecasts requires local starter data, but no training or API keys.

## Final submission and results

**Upload [`submission/final_ensemble/forecast_backtest.csv`](submission/final_ensemble/forecast_backtest.csv)**: 673 hourly origins × 40 quarter-hours = 26,920 rows. The similarly named root `forecast_backtest.csv` is the older single-seed LightGBM fallback.

| Metric | Validation baseline | Final validation | Final local test |
|---|---:|---:|---:|
| MAE, EUR/MWh ↓ | 66.20 | 41.60 | 26.70 |
| Mean P10/P90 pinball ↓ | 24.18 | 11.89 | 6.83 |
| P10–P90 coverage, target 80% | 68.29% | 73.28% | 80.72% |
| Spike Brier ↓ | 0.1132 | 0.0908 | 0.0567 |
| Direction Brier ↓ | 0.4509 | 0.3929 | 0.3937 |

Validation MAE is **37.17% lower** and pinball loss **50.84% lower** than the organiser baseline. Test results are a local historical backtest against reconstructed actuals, not an organiser-certified score. Compare models within the same period; validation and test have different market conditions. See the [evaluation report](EVALUATION_REPORT.md) for the full comparison, including the modest incremental blend gains and limitations.

## Project map

| Location | Purpose |
|---|---|
| [RUNNING.md](RUNNING.md) | Canonical setup and execution instructions |
| [Complete_Experiment.ipynb](Complete_Experiment.ipynb) | Experiment walkthrough and chart generation; full notebook execution remains unverified |
| [dashboard/](dashboard/README.md) | Local replay dashboard and technical notes |
| [submission/final_ensemble/](submission/final_ensemble/README.md) | Final test and validation forecasts |
| [EVALUATION_REPORT.md](EVALUATION_REPORT.md) | Current consolidated evaluation |
| [output/pdf/](output/pdf/) | Seven-page report and its charts |
| [demo/Turku_Energia_Research_Story.mp4](demo/Turku_Energia_Research_Story.mp4) | Current demo video |
| [src/pkg/](src/pkg/) | Final features, training, blending and leakage checks |
| [experiments/](experiments/README.md) and [results/](results/) | Historical experiments, saved metrics and provenance |
| [docs/STARTER_README.md](docs/STARTER_README.md) | Original organiser rules and file format |
| `data/`, `work_private/` | Local datasets, checkpoints and generated private matrices; excluded from source distribution |

Earlier A/B reports and experiment recommendations describe their original stage of development. Use the consolidated evaluation report for the final project status. The dashboard's driver explanations are reference exports from the previous single-seed LightGBM, explicitly labelled in the interface.

Keep raw datasets, licensed Nord Pool spot data, private matrices, credentials and virtual environments out of the source-code archive and public repository. See [data sources and licences](DATA_SOURCES_AND_LICENSES.md) for attribution, redistribution restrictions and cleanup requirements.
