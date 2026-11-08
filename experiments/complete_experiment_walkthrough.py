# %% [markdown]
# # Turku GridCast — complete experiment record
# An executable walkthrough of the Turku Energia imbalance-price forecasting project.
# 
# **Read this first:** Run All reviews saved experiments, recomputes validation/test metrics and checks forecast integrity. It does not download data, retrain models or change the submission. Optional model training/inference cells are disabled. Original experiment implementations are linked rather than duplicated; long rebuild commands are documentation only.
# 
# The major gain is against the organiser's **validation baseline**. The extra Chronos blend adds modest improvements over LightGBM. Local October test results are not the official organiser evaluation.
# 
# Contents: setup → task/splits → data audit → baseline/LightGBM → Chronos → TimesFM/Timestral → external data → tuning → final blend → scoring/plots → leakage/provenance → optional reruns → dashboard/handoff.
# 
# Do not distribute local data, spot-price files, feature matrices or API keys with this notebook. Outputs intentionally avoid printing raw spot prices.

# %%
from pathlib import Path
import sys, json, hashlib, importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / 'score.py').exists() and (p / 'results/final_ensemble').exists()), None)
assert ROOT is not None, 'Open this notebook from the project root.'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/pkg'))
METRICS = ['mae_p50', 'calibration_pinball_mean', 'coverage_p10_p90', 'brier_spike_mean', 'brier_direction']
def csv(path): return pd.read_csv(ROOT / path)
def evidence(path):
    frame = csv(path)
    return frame[[c for c in ['model', *METRICS] if c in frame]]
def report(path): display(Markdown((ROOT / path).read_text()))
versions = {}
for name in ['numpy', 'pandas', 'matplotlib', 'lightgbm', 'chronos-forecasting', 'torch']:
    try: versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError: versions[name] = 'not in this kernel'
display(pd.Series(versions, name='Current kernel package version'))
print('Root:', ROOT)
print('Saved result review enabled. Training and inference disabled by default.')

# %% [markdown]
# ## 1. Task, target and temporal protocol
# Each full-hour origin produces 40 quarter-hours; step 1 begins at the origin. Predict spread = imbalance price − spot, then add delivery spot back for price quantiles. Direction is up/down/none; high/low spikes mean spread > +50 or < −50 EUR/MWh.
# 
# The package defines exact UTC origins, even though date labels describe Finnish calendar days. Validation origins: **6 Sep 21:00Z to 4 Oct 20:00Z** (672). Test origins: **4 Oct 21:00Z to 1 Nov 21:00Z** (673, including DST effects).
# Train only on origins before each fit cutoff and targets usable by that cutoff. Validation cutoff: 6 Sep 21:00Z; final test fit: 4 Oct 21:00Z. Rolling folds begin Jul 5/19, Aug 2/16/30 at 21:00Z.
# 
# Use a package row only if first_usable_origin_utc ≤ origin. External Fingrid observations: quarter end + 30 minutes, rounded up to a full hour. ECMWF archived runs: run time + 6 hours. These external publication delays are documented assumptions. No random train/test split.

# %%
val_pred = pd.read_csv(ROOT / 'submission/final_ensemble/validation_forecast.csv', parse_dates=['forecast_origin_utc','delivery_start_utc'])
test_pred = pd.read_csv(ROOT / 'submission/final_ensemble/forecast_backtest.csv', parse_dates=['forecast_origin_utc','delivery_start_utc'])
summary = []
for label, frame, expected in [('validation', val_pred, 672), ('test', test_pred, 673)]:
    assert frame.forecast_origin_utc.nunique() == expected
    assert frame.groupby('forecast_origin_utc').size().eq(40).all()
    assert not frame.duplicated(['forecast_origin_utc','step']).any()
    assert (frame.delivery_start_utc == frame.forecast_origin_utc + pd.to_timedelta((frame.step-1)*15, unit='min')).all()
    assert (frame.p10 <= frame.p50).all() and (frame.p50 <= frame.p90).all()
    assert np.allclose(frame[['p_up','p_down','p_none']].sum(axis=1), 1, atol=2e-6)
    summary.append(dict(period=label, rows=len(frame), origins=expected, first=frame.forecast_origin_utc.min(), last=frame.forecast_origin_utc.max()))
display(pd.DataFrame(summary))
training_audit = json.loads((ROOT / 'results/final_ensemble/lgb_training_audit.json').read_text())
display(pd.DataFrame(training_audit))

# %% [markdown]
# ## 2. Local input audit
# Package sources include Fingrid, ENTSO-E and licensed Nord Pool spot. Selected external signals include Fingrid real-time system measurements and archived Open-Meteo ECMWF forecasts. Outages and ICON/HARMONIE were later experiments, not members of the final model.
# 
# This cell prints metadata only. Raw spot values remain local. See [data audit](docs/DATA_AUDIT.md) and [source/licence inventory](REPORT.md).

# %%
inputs = pd.read_parquet(ROOT / 'data/inputs.parquet')
required = {'series_id','timestamp_utc','value','first_usable_origin_utc'}
assert required.issubset(inputs.columns)
input_audit = inputs.groupby('series_id').agg(rows=('value','size'), missing=('value',lambda x: x.isna().sum()), first=('timestamp_utc','min'), last=('timestamp_utc','max'))
display(input_audit)
print('Input dtypes:', inputs.dtypes.astype(str).to_dict())
print('Duplicate series/timestamp rows:', inputs.duplicated(['series_id','timestamp_utc']).sum())
print('Missing publication times:', inputs.first_usable_origin_utc.isna().sum())
del inputs

# %% [markdown]
# ## 3. Baseline and task-trained LightGBM
# The organiser baseline persists the latest available imbalance price and uses historical errors/proportions for intervals and probabilities. Our LightGBM uses calendar, price/spread history, supply/demand forecasts, regulation/flows, external system signals and weather.
# 
# Final feature families: direction EW, spike EX, P10 E4, P50/P90 EX. Separate short-horizon specialists cover steps 1–4 and 5–8. Research motivated gradient-boosted trees as a starting point; task-specific evaluation, not a universal superiority claim, supports this choice.
# 
# Sources: [LightGBM paper](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [feature builder](src/pkg/pkg_features.py), [original model](src/pkg/pkg_model.py).

# %%
final_comparison = csv('results/final_ensemble/validation_comparison.csv')
display(final_comparison[['model', *METRICS]])
base = final_comparison.set_index('model').loc['organizer_baseline']
final = final_comparison.set_index('model').loc['final_blend']
losses = [c for c in METRICS if c != 'coverage_p10_p90']
display(pd.DataFrame({'baseline':base[losses].astype(float), 'final':final[losses].astype(float), 'relative_reduction_percent':100*(1-final[losses].astype(float)/base[losses].astype(float))}))
print('Coverage is not a loss: its nominal target is 80%.')

# %% [markdown]
# ## 4. Experiment 2 — initial Chronos-2 zero-shot benchmark
# Price, spread and spread-plus-covariates were tested with a seven-day context. This initial configuration differs from the later final ensemble's longer context and covariate processing. Do not merge their results under a single Chronos label. No task-specific fine-tuning was performed.
# 
# Outcome: none of these initial variants beat the saved LightGBM on MAE and official pinball. [Implementation](experiments/experiment_2_chronos.py) · [full report](results/experiment_2_chronos/REPORT.md).

# %%
display(evidence('results/experiment_2_chronos/comparison.csv'))
display(csv('results/experiment_2_chronos/paired_mae_bootstrap.csv'))

# %% [markdown]
# ## 5. Experiment 3 — TimesFM and Timestral
# Zero-shot price, spread and covariate configurations benchmarked alternative pretrained model families. The local runs did not displace LightGBM on price error. Coverage alone is not evidence of superior forecasts: wide intervals can cover more outcomes while receiving worse pinball loss.
# 
# [Implementation](experiments/experiment_3_foundation_models.py) · [report and exact model revisions](results/experiment_3_foundation_models/REPORT.md). Neither TimesFM nor Timestral is in the final submission.

# %%
display(evidence('results/experiment_3_foundation_models/comparison.csv'))
display(csv('results/experiment_3_foundation_models/paired_mae_bootstrap.csv'))

# %% [markdown]
# ## 6. Experiment 4 — earlier external information
# Tested outage notices, archived ICON/HARMONIE weather, and both against a matched control. Some metrics improved slightly; all paired MAE intervals included zero. Pre-validation fold selection retained the baseline. Downloaded or investigated data is not automatically a final-model feature.
# 
# [Implementation](experiments/experiment_4_external_data.py) · [report](results/experiment_4_external_data/REPORT.md) · source and availability audits are retained below.

# %%
display(evidence('results/experiment_4_external_data/comparison.csv'))
display(csv('results/experiment_4_external_data/training_fold_comparison.csv'))
display(csv('results/experiment_4_external_data/paired_day_bootstrap.csv'))

# %% [markdown]
# ## 7. Experiment 5 — controlled parameter tuning
# Four equal-budget searches covered existing features, outages, weather, and both. Candidate tuning covered learning rate, rounds, leaves, leaf size, feature fraction and L2. Selection used chronological pre-validation search/confirmation folds.
# 
# **Tuning reduced median MAE in zero of four matched comparisons.** The saved submission had a different calibration history, so compare tuned runs to their matched controls to isolate the tuning effect. Quantile sorting can change the exported median even if the raw median head is unchanged.
# 
# [Implementation](experiments/experiment_5_parameter_tuning.py) · [protocol](results/experiment_5_parameter_tuning/protocol.json) · [report](results/experiment_5_parameter_tuning/REPORT.md).

# %%
display(evidence('results/experiment_5_parameter_tuning/comparison.csv'))
display(csv('results/experiment_5_parameter_tuning/tuning_effect.csv'))
display(csv('results/experiment_5_parameter_tuning/selection_summary.csv'))
print('Protocol:')
display(json.loads((ROOT / 'results/experiment_5_parameter_tuning/protocol.json').read_text()))

# %% [markdown]
# ## 8. Final five-seed LightGBM + Chronos-2 blend
# Seeds 7, 11, 23, 37, 51; bagging/feature/data seed offsets +1/+2/+3. Average raw predictions first. Blend price quantiles with Chronos-2; retain LightGBM direction/spike probabilities. Sort quantiles and cap spike probabilities by corresponding direction probability. No final conformal adjustment.
# 
# Chronos: 2,048 context quarters, 42 predicted positions, discard first two bridge positions; keep 40 delivery quarters. Context/each covariate is availability-masked. Zero-shot CPU inference uses the pinned local checkpoint.
# 
# LightGBM weights for steps 1–4 / 5–16 / 17–40:
# - Validation: 0.7 / 0.4 / 0.5, selected from training OOS predictions.
# - Test: 0.7 / 0.4 / 0.6, allowing validation OOS data usable at the final cutoff.
# Remaining weights belong to Chronos. Selection optimizes P10/P50/P90 pinball; the official interval score averages **P10/P90 only**.
# 
# The stored normalized selection score S is an internal objective, **not official competition points**. Official points also depend on accepted competitors and jury assessment.

# %%
display(csv('results/final_ensemble/weight_grid.csv').head(12))
display(json.loads((ROOT / 'results/final_ensemble/weight_selection_audit.json').read_text()))
display(csv('results/final_ensemble/paired_day_bootstrap.csv'))
fig, axes = plt.subplots(1,2,figsize=(12,4))
selected = final_comparison[final_comparison.model.isin(['organizer_baseline','current_lightgbm_single','final_blend'])].set_index('model')
for ax, metric in zip(axes,['mae_p50','calibration_pinball_mean']):
    selected[metric].plot.barh(ax=ax,color=['#9ebbc9','#ed9600','#124f74'])
    ax.set_title(metric + ' (lower is better)'); ax.set_ylabel('')
fig.suptitle('Validation only — same evaluation period');plt.tight_layout();plt.show()

# %% [markdown]
# ## 9. Independent scoring of frozen forecasts
# Recompute the official metrics in this kernel using score.py. Validation uses supplied actuals. Test uses the locally reconstructed actuals; their construction matched the supplied validation snapshot. This is retrospective evaluation after the forecasts were frozen, never training on test targets.
# 
# This cell deliberately does not display actuals or spot-price rows.

# %%
from score import load_actuals, load_submission, compute_metrics
val_actual = load_actuals(ROOT / 'data/actuals_validation.csv')
test_actual_path = ROOT / 'work_private/final_ensemble/local_actuals_test.csv'
assert test_actual_path.exists(), 'Prepare local test actuals using the documented score_local_test.py workflow.'
test_actual = load_actuals(test_actual_path)
scores = []
joined_frames = {}
for label,pred,actual in [('validation',val_pred,val_actual),('local_test',test_pred,test_actual)]:
    joined = pred.merge(actual,on='delivery_start_utc',how='left',validate='many_to_one')
    joined_frames[label] = joined
    scores.append({'period':label,**compute_metrics(joined,50)})
recomputed = pd.DataFrame(scores).set_index('period')
display(recomputed[[*METRICS,'rows_scored_price']])
np.testing.assert_allclose(recomputed.loc['validation',METRICS].astype(float), final[METRICS].astype(float), rtol=0,atol=1e-8)
local_saved = csv('results/final_ensemble/local_test_backtest/comparison.csv').set_index('model')
np.testing.assert_allclose(recomputed.loc['local_test',METRICS].astype(float), local_saved.loc['ensemble',METRICS].astype(float), rtol=0,atol=1e-8)
print('Saved scores independently reproduced. Validation tail rows without actuals are excluded by the official scorer.')
display(local_saved[METRICS])
print('Local test: essentially unchanged MAE versus previous LightGBM, improved pinball and coverage closer to 80%.')

# %% [markdown]
# ## 10. Horizon diagnostics and a real forecast
# Separate validation/test charts avoid implying that differences between periods are model improvements. The historical example defaults to the middle validation origin; it is illustrative, not selected as the best forecast. Actuals are added after forecasting for evaluation.

# %%
# Generate the complete chart gallery inline and save each figure.
CHART_DIR = ROOT / 'results/notebook_charts'
CHART_DIR.mkdir(parents=True, exist_ok=True)
chart_paths = []
BLUE, ORANGE, GREY = '#124f74', '#ed9600', '#9ebbc9'
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.titleweight':'bold'})
def finish_chart(fig, name):
    fig.tight_layout()
    path = CHART_DIR / (name + '.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    chart_paths.append(path)
    plt.show()
    plt.close(fig)
def compare_chart(frame, name, title):
    frame = frame.set_index('model')
    fig, axes = plt.subplots(1,2,figsize=(14,max(4,len(frame)*0.45)))
    labels = [str(v).replace('_',' ') for v in frame.index]
    for ax, metric, label in zip(axes,
        ['mae_p50','calibration_pinball_mean'],
        ['MAE (EUR/MWh) — lower is better','P10/P90 pinball — lower is better']):
        bars = ax.barh(labels,frame[metric],color=BLUE)
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=9)
        ax.set_xlim(0,float(frame[metric].max())*1.2)
        ax.invert_yaxis();ax.set_xlabel(label)
    fig.suptitle(title,fontsize=15)
    finish_chart(fig,name)

# 1. Same-period baseline comparison.
compare_chart(final_comparison[final_comparison.model.isin(
    ['organizer_baseline','current_lightgbm_single','final_blend'])],
    '01_validation_baseline','Validation: organiser baseline → LightGBM → final blend')

# 2. Initial zero-shot runs (distinct from final Chronos configuration).
fm = csv('results/experiment_3_foundation_models/comparison.csv')
compare_chart(fm[fm.model != 'baseline'],'02_foundation_models',
              'Initial zero-shot experiments — validation; configurations are not final ensemble members')

# 3. Controlled additional-data experiment.
external = csv('results/experiment_4_external_data/comparison.csv')
compare_chart(external[external.model != 'submitted_B'],'03_external_data',
              'External-data experiment — validation; small, mixed changes')

# 4. Isolate tuning from feature-set and calibration differences.
tuning = csv('results/experiment_5_parameter_tuning/tuning_effect.csv')
fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, metric, title in zip(axes,['mae_p50','calibration_pinball_mean'],
                             ['MAE change','Pinball change']):
    delta=tuning[metric]
    ax.bar(tuning.feature_set,delta,color=[BLUE if v<0 else ORANGE for v in delta])
    ax.axhline(0,color='#444444',linewidth=1)
    ax.set_title(title);ax.set_ylabel('Tuned minus matched control')
fig.suptitle('Parameter tuning: below zero helps; no median-MAE improvement in four matched comparisons')
finish_chart(fig,'04_parameter_tuning')

# 5. Final model progression and interval calibration.
members=final_comparison.set_index('model')
names=['current_lightgbm_single','lightgbm_5seed_raw','chronos2_covariates','final_blend']
fig, axes=plt.subplots(1,2,figsize=(13,5))
labels=['Previous LightGBM','5-seed raw LightGBM','Final-config Chronos alone','Final blend']
for ax,metric,title in zip(axes,['calibration_pinball_mean','coverage_p10_p90'],
                           ['Pinball loss — lower is better','P10–P90 coverage — target 80%']):
    values=members.loc[names,metric].astype(float)
    if metric=='coverage_p10_p90':values=values*100
    bars=ax.barh(labels,values,color=[GREY,GREY,ORANGE,BLUE])
    ax.bar_label(bars,fmt='%.2f',padding=4)
    ax.set_xlim(0,max(values)*1.2);ax.invert_yaxis();ax.set_title(title)
    if metric=='coverage_p10_p90':ax.axvline(80,color=ORANGE,linestyle='--',label='80% target');ax.legend()
fig.suptitle('Final ensemble progression — validation only')
finish_chart(fig,'05_final_blend')

# 6. Test comparison: all metrics, without conflating test and validation.
test_table=csv('results/final_ensemble/local_test_backtest/comparison.csv').set_index('model')
fig, axes=plt.subplots(1,5,figsize=(17,4))
titles=['MAE ↓','Pinball ↓','Coverage %','Spike Brier ↓','Direction Brier ↓']
for ax,metric,title in zip(axes,METRICS,titles):
    values=test_table.loc[['previous_lightgbm','ensemble'],metric].astype(float)
    if metric=='coverage_p10_p90':values=values*100
    bars=ax.bar(['Previous','Blend'],values,color=[GREY,BLUE])
    ax.bar_label(bars,fmt='%.4f' if 'brier' in metric else '%.2f',padding=4,fontsize=9)
    ax.set_ylim(0,max(values)*1.3);ax.set_title(title)
    if metric=='coverage_p10_p90':ax.axhline(80,color=ORANGE,linestyle='--')
fig.suptitle('Local October backtest — official organiser score pending; results are mixed')
finish_chart(fig,'06_local_test')

# 7. Horizon diagnostics. Curves are different periods, not successive improvements.
fig, axes=plt.subplots(1,2,figsize=(12,4))
for label,joined in joined_frames.items():
    valid=joined.dropna(subset=['imbalance_price'])
    by_step=valid.assign(error=(valid.p50-valid.imbalance_price).abs()).groupby('step').error.mean()
    widths=valid.assign(width=valid.p90-valid.p10).groupby('step').width.mean()
    axes[0].plot(by_step.index/4,by_step.values,label=label)
    axes[1].plot(widths.index/4,widths.values,label=label)
axes[0].set_title('Price MAE by horizon');axes[1].set_title('Mean interval width by horizon')
for ax in axes:
    ax.set_xlabel('End of delivery quarter (hours from origin)');ax.set_ylabel('EUR/MWh');ax.legend()
fig.suptitle('Horizon diagnostics — validation and test are separate evaluation periods')
finish_chart(fig,'07_horizon_diagnostics')

# 8. Real forecast selected by position, not by lowest error.
origins=sorted(val_pred.forecast_origin_utc.unique())
example_origin=origins[len(origins)//2]
example=joined_frames['validation'].query('forecast_origin_utc == @example_origin').sort_values('step')
fig,ax=plt.subplots(figsize=(12,4))
ax.fill_between(example.delivery_start_utc,example.p10,example.p90,color='#d8eaf2',label='P10–P90')
ax.plot(example.delivery_start_utc,example.p50,color=BLUE,label='Predicted P50')
ax.plot(example.delivery_start_utc,example.imbalance_price,color=ORANGE,label='Actual — retrospective')
ax.set(title=f'Validation forecast issued {example_origin}',ylabel='EUR/MWh',xlabel='UTC delivery time')
ax.legend()
finish_chart(fig,'08_forecast_with_uncertainty')
display(pd.DataFrame({'chart':[p.name for p in chart_paths],'saved_to':[str(p) for p in chart_paths]}))
print('Generated and saved',len(chart_paths),'charts. No raw spot-price charts are exported.')

# %% [markdown]
# ## 11. Leakage and provenance evidence
# Read the saved mutation-check reports and verify the exported CSV hashes. This verifies existing evidence and immutable forecasts; it does not rerun every expensive mutation/inference test. These tests do not prove absence of foundation-model pretraining overlap.
# 
# Full implementations: [LightGBM checks](src/pkg/pkg_leak_test.py), [final pipeline](src/pkg/run_final_ensemble.py), and recorded JSON audits under results/final_ensemble.

# %%
verification = json.loads((ROOT / 'results/final_ensemble/verification.json').read_text())
for relative, expected in verification['output_sha256'].items():
    observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    assert observed == expected, f'Forecast changed: {relative}'
display({k:v for k,v in verification.items() if not isinstance(v,dict)})
for name in ['feature_leak_checks.json','lgb_output_leak_checks.json','fm_leak_checks.json']:
    payload = json.loads((ROOT / 'results/final_ensemble' / name).read_text())
    print(name)
    display(payload)
print('Final forecast hashes match recorded verification.')

# %% [markdown]
# ## 12. Optional model execution, directly in the kernel
# The next two cells are opt-in. They call the production functions, retaining returned models/arrays in the notebook kernel. Default Run All does not train or infer.
# 
# LightGBM smoke training requires the local feature matrix. Chronos requires the pinned checkpoint and the dependencies in experiments/requirements-chronos.txt; select a suitable kernel before enabling it. A single-model smoke fit is not the five-seed final ensemble and does not overwrite it.

# %%
RUN_LIGHTGBM_SMOKE = False
if RUN_LIGHTGBM_SMOKE:
    from ensemble_spec import load_frame, params, SELECTED
    from ens_members import fit, family_columns
    x = load_frame()
    cutoff = pd.Timestamp('2026-09-06T21:00:00Z')
    train_idx = np.flatnonzero((x.o < cutoff) & (x.target_fu <= cutoff))
    predict_idx = np.flatnonzero(x.o == cutoff)
    columns = family_columns(x)[SELECTED['0.5']]
    smoke_model = fit(x,0.5,train_idx,columns,params()['0.5'],7,4)
    smoke_spread = smoke_model.predict(x.loc[predict_idx,columns])
    print('Single global median model trained; predictions:',smoke_spread.shape)
else:
    print('Optional LightGBM smoke training disabled.')

# %%
RUN_CHRONOS_SMOKE = False
if RUN_CHRONOS_SMOKE:
    from fm_run import load_pipeline, predict
    from fm_data import load_sources, make_input
    values, availability = load_sources()
    smoke_origin = pd.Timestamp('2026-09-06T21:00:00Z')
    task, known_spot, filled = make_input(values,availability,smoke_origin)
    pipeline = load_pipeline(4)
    smoke_quantiles = predict(pipeline,[task])
    assert smoke_quantiles.shape == (1,40,3)
    print('Chronos spread-quantile shape:',smoke_quantiles.shape,'forward-filled covariates:',filled)
else:
    print('Optional Chronos inference disabled.')

# %% [markdown]
# ## 13. Full rerun map and environment boundaries
# The saved experiments are complete; the cells above review and independently score them. Retraining all historical runs is a separate, expensive operation. Use the original source implementations to preserve exact cache, calibration and environment behavior.
# 
# Run these in a terminal from the repository root (not as notebook shell cells):
# ```sh
# # Main feature matrix / original model
# .venv/bin/python src/pkg/pkg_features.py B
# .venv/bin/python src/pkg/pkg_model.py B
# # Final ensemble orchestrator: uses main and isolated Chronos environments
# .venv/bin/python src/pkg/run_final_ensemble.py
# # Local post-hoc test scoring after freezing forecasts
# .venv/bin/python src/pkg/score_local_test.py
# # Local dashboard
# .venv/bin/python dashboard/server.py
# ```
# 
# **Original-model rebuilds can overwrite the root fallback forecast.** Preserve your submission before deliberately rerunning them. The ensemble orchestrator writes its own final_ensemble directory. Check source CLI options before individual research runs:
# - Experiment 2: experiments/experiment_2_chronos.py
# - Experiment 3: experiments/experiment_3_foundation_models.py
# - Experiment 4: experiments/experiment_4_external_data.py
# - Experiment 5: experiments/experiment_5_parameter_tuning.py
# 
# Environment specifications: requirements.txt, experiments/requirements-chronos.txt and experiments/requirements-foundation-models.txt. Actual run environments and checkpoint revisions are stored alongside experiment reports. Raw input archives/checkpoints are deliberately not bundled in a public code ZIP.
# 
# ## 14. Dashboard and submission
# Open http://127.0.0.1:8765 after starting the dashboard. Its opening section shows full validation and local-test metric tables; the replay provides ten-hour quantiles, risk probabilities, retrospective actuals, earlier-forecast comparisons and publication-aware historical metrics.
# 
# Reference drivers/freshness describe the earlier single-seed LightGBM only, not the complete final ensemble. Alert thresholds are illustrative expert-review prompts. Browser visual QA remains separate from API tests.
# 
# Final upload: submission/final_ensemble/forecast_backtest.csv. Keep reports, source/licence inventory, start-up instructions, presentation slides and a ≤5-minute demo together. Notebook existence does not imply the slide deck, official submission or deployment is complete.
# 
# ## Conclusion
# Strong validation improvement over the organiser baseline; modest additional benefit from the final blend; mixed local test results. Further work: interval calibration, additional seasons, verifiable outage/forecast revisions and operational evaluation. No unsupported claim of statistical significance, live profitability or official test ranking.

