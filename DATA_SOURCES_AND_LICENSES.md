# Data sources, models, services and licences

Turku GridCast / Since AI 2026. Inventory prepared 8 November 2026 from the project source, saved run manifests, local model cards and the linked provider terms. This is a scoped dependency/data inventory, not a certification of all transitive rights. Final forecasting members, research experiments and presentation tooling are distinguished below.

## Final forecasting inputs

| Source | Use and transformation | Licence / handling | Availability control |
|---|---|---|---|
| Fingrid Open Data, via the organiser package | Imbalance prices, mFRR/aFRR, direction, consumption/production, flows; lagged values, rolling statistics and labels. Day-ahead forecast datasets 165/246/247. | Fingrid-produced open data: CC BY 4.0. Credit Fingrid and identify derived features. [Provider terms](https://beta-data.fingrid.fi/about), [catalogue](https://data.fingrid.fi/en/datasets). | Package `first_usable_origin_utc` is binding. |
| Fingrid external datasets 75, 177, 181, 188, 194, 260, 397, 400 | Wind actuals and wind error, nuclear generation, net import, frequency, calculated imbalance and other real-time system signals; aggregate to quarter-hours. | CC BY 4.0 for Fingrid-produced datasets; API access conditions also apply. Dataset IDs are retained in the fetcher and source files. | Quarter end + 30 minutes, rounded up to a full hour. |
| ENTSO-E Transparency, via the organiser package | Neighbour imbalance prices/load forecasts and FI-EE transfer capacity; lagged and known-future features. | Package documentation records CC BY 4.0. ENTSO-E's reuse permission is dataset-specific: do not generalise it to every platform field. [Reuse list](https://m-transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/220218_List_of_Data_available_for_reuse.pdf). Preserve the package's source attributions. | Package row-level publication times. |
| Nord Pool day-ahead spot, supplied by Turku Energia | Spread target, known-ahead spot features, conversion of spread quantiles back to prices, retrospective scoring. | Restricted hackathon licence, per organiser briefing page 25: no redistribution/public repository; delete after the event. This includes spot columns inside actuals and derived private feature matrices. Do not infer redistribution permission from public accessibility. | Day-ahead publication at 14:00 Finnish local time; package timestamps are authoritative. |
| Open-Meteo Single Runs API / ECMWF IFS | Archived weather runs transformed into regional wind proxies; not the seamless Historical Forecast API. | Weather API data CC BY 4.0; API service access has separate terms. Free API access is non-commercial; commercial API use requires the relevant service entitlement. [Terms](https://open-meteo.com/en/terms), [pricing/data attribution](https://open-meteo.com/en/pricing). ECMWF open data: CC BY 4.0 plus [ECMWF terms](https://www.ecmwf.int/en/forecasts/datasets/open-data). | Run initialization + 6 hours, rounded up; conservative assumption, not verified historical API-ingestion timestamps. |
| Calendar / local-time calculations | UTC, Europe/Helsinki, hour/day/holiday and seasonal features. | Python zoneinfo/tzdata and holidays package licences apply; no third-party market observations. | Deterministic calendar information, known in advance. |

Attribution for derived data: **Source: Fingrid / data.fingrid.fi, CC BY 4.0; ENTSO-E Transparency via the challenge package; weather data by Open-Meteo and ECMWF, CC BY 4.0.** This project performs aggregation, availability masking, lagging, interpolation and feature derivation. No provider endorsement is implied. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Research-only inputs and investigated sources

| Source | Actual status | Licence / limitation |
|---|---|---|
| Nord Pool UMM and notice publishers | Outage-history experiment: 442 notice histories, 1,511 revisions. Not in final ensemble. | Public access under [UMM platform terms](https://umm.nordpoolgroup.com/) and [REMIT UMM service terms](https://www.nordpoolgroup.com/en/services/compliance/umm/). No open redistribution licence or commercial rights were established by this project. Raw notices remain private. Publication time + 5 minutes rounded up was the experiment's visibility rule. |
| Open-Meteo / DMI HARMONIE / DWD ICON-EU | Archived extra-weather experiments; not in final ensemble. | Attribute Open-Meteo, DMI and DWD. API data/service terms are distinct; retain source-provider conditions. Derived wind proxies are project transformations. Run + 6 h is an assumption. See `results/experiment_4_external_data/source_audit.json`. |
| eSett open data; FMI observations | Earlier fetches/exploration, not final inputs. | No redistribution rights are asserted for cached material in this inventory; verify the specific dataset terms before reuse. Local code does not bundle their raw archives for publication. |
| Fingrid 245 intraday wind | Inspected and excluded: historical issue/version timestamps not established. | CC BY status alone does not make overwritten historical forecasts safe for a backtest. |
| Elering, Svenska kraftnat, Statnett | Catalogues/samples investigated; additional real-time feeds not adopted. | No verified historical publication archive established; no blanket reuse licence asserted. Existing organiser-package neighbour series remain separately in use. |
| Fingrid 373/374 reserve bids | Identified, deferred; not benchmarked. | Historical publication rules and accessible archive not established. |
| Nord Pool continuous intraday prices | Investigated, not acquired or used. | Licensed subscription required; no entitlement obtained. |

## Models and exact experiment scope

| Model | Status | Recorded licence and provenance |
|---|---|---|
| LightGBM | Final task-trained model; five seeds and short-horizon specialists. | MIT. [Source/licence](https://github.com/microsoft/LightGBM/blob/master/LICENSE). Runtime/settings recorded in `results/final_ensemble/lgb_completed.json`. |
| Amazon Chronos-2 | Final zero-shot covariate model; also earlier Experiment 2. | Apache-2.0 for checkpoint as recorded in local model card; forecasting code Apache-2.0. [Model card](https://huggingface.co/amazon/chronos-2), [source](https://github.com/amazon-science/chronos-forecasting). Pinned revision `29ec3766d36d6f73f0696f85560a422f50e8498c`; `chronos-forecasting==2.3.2`. Final context differs from the first benchmark. |
| Google TimesFM 3.0 | Experiment 3 only; not final. | Local checkpoint card declares **TimesFM Non-Commercial License v1.0**, not Apache/MIT. [Checkpoint licence](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE). Code and checkpoint rights are distinct; pinned source evidence is in `results/experiment_3_foundation_models/source_packages.json`. |
| Mistral Timestral 209M | Experiment 3 only; not final. | Local card declares **Modified MIT**, not unmodified MIT. [Checkpoint licence](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE). Preserve its exact conditions for research reuse. |

Only the final model members are described as MIT/Apache-2.0. That statement does not apply to all data, experiments, dependencies or media services. The team's own source does not acquire a licence automatically from its dependencies; no new blanket project licence is granted by this inventory.

## Software components

These are direct/key components, not an exhaustive transitive SBOM. Preserve installed distributions' licence and notice files; exact training environments are recorded in the experiment manifests, which may differ from the current report-building environment.

| Component | Purpose | Licence family / evidence |
|---|---|---|
| Python / standard library, zoneinfo, local HTTP server | Pipeline and dashboard backend | PSF licence; bundled components can carry additional notices. |
| NumPy | Numerical computation | BSD-3-Clause core; installed wheel also lists bundled MIT/0BSD/Zlib/CC0 notices. |
| pandas | Dataframes, CSV and scoring | BSD-3-Clause. |
| Apache Arrow / PyArrow | Parquet storage | Apache-2.0. |
| LightGBM | Quantile and multiclass training | MIT. |
| Optuna | Historical hyperparameter search | MIT. |
| holidays | Finnish calendar features | MIT. |
| requests | Data retrieval | Apache-2.0. |
| tzdata | Time-zone database | Apache-2.0 package; IANA database includes its own notices. |
| PyTorch | Foundation-model inference | BSD-style, with bundled third-party notices. |
| Hugging Face Transformers / huggingface_hub / safetensors | Model loading, checkpoint distribution | Apache-2.0; retain package notices. |
| chronos-forecasting | Chronos inference | Apache-2.0. |
| Matplotlib | Notebook and report charts | Matplotlib licence, based on PSF; preserve included notices. |
| Pillow | Image composition | MIT-CMU (installed metadata). |
| ReportLab | This evaluation PDF | BSD licence (installed metadata). |
| Jupyter / IPython / ipykernel / jupyter-server | Notebook execution | BSD-family; exact component licences apply. |
| Plain HTML/CSS/JavaScript/SVG | Local dashboard frontend | Project-authored code; no frontend framework or CDN dependencies. |
| PyMuPDF | Local PDF inspection/rendering during authoring | AGPL/commercial dual licensing; authoring tool only, not a dashboard/model runtime dependency. No proprietary redistribution entitlement asserted. |
| FFmpeg / libx264 | Demo video encoding | FFmpeg build-dependent LGPL/GPL; libx264 GPL. Tool use does not mean generated video itself is GPL software. Do not redistribute binaries without their notices/terms. |
| Arial / Georgia system fonts | Local report/demo typography | Proprietary system fonts; no font files included in the source package. PDF uses standard built-in Helvetica/Times faces rather than bundling these fonts. |

## External services and media

- **ElevenLabs:** existing demo narration used stock voice Will and model ID `eleven_v4`, per production records. Proprietary service; account/plan-specific output rights apply. The account plan was not independently audited. Free access is non-commercial; paid subscriptions permit commercial use subject to terms. [EU terms](https://elevenlabs.io/terms-of-use-eu), [output rights after subscription](https://elevenlabs.io/docs/help-center/account/general/what-happens-to-my-content-after-my-subscription-ends). AI narration is disclosed in the video. No voice model is bundled or used in price forecasting.
- **Hugging Face Hub:** checkpoint distribution only; each checkpoint's own licence governs model reuse. No project observations were uploaded for inference.
- **AI coding assistants:** used for research, implementation, documentation and presentation assistance. They are development services, not inference members of the submitted forecast. Human/team review remains necessary. No licence or model-revision claim is made for an unrecorded external assistant session.
- **Challenge briefing:** visual inspiration for orange/blue report and demo styling. Branding does not imply organiser endorsement; original briefing is not republished as a project-owned work.

## Packaging boundaries and outstanding rights checks

Ship code, forecasts, aggregate metrics, reports and generated charts. Exclude `.env`, credentials, raw starter/external data, `work_private/`, virtual environments and model checkpoints. In particular, do not upload `spot_nordpool.csv`, actuals containing spot prices, or private matrices. Check specific UMM reuse terms and ElevenLabs plan rights before uses beyond the hackathon. Provider/API terms and dataset permissions can change; the linked sources and local model cards are the evidence, not a universal grant of rights.

Local evidence: `REPORT.md`, `docs/STARTER_README.md`, `docs/DATA_AUDIT.md`, `results/final_ensemble/REPORT.md`, `results/experiment_2_chronos/run.json`, `results/experiment_3_foundation_models/source_packages.json`, `results/experiment_4_external_data/source_audit.json`, and the three pinned local model cards.
