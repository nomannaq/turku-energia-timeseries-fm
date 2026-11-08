"""Build the seven-page report and matching Markdown from frozen scoring evidence."""
from pathlib import Path
import json, hashlib, os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/pdfs/mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader
from xml.sax.saxutils import escape
R=Path(__file__).resolve().parents[1]; O=R/'output/pdf'; A=O/'charts';A.mkdir(parents=True,exist_ok=True)
BLUE='#124f74';ORANGE='#ed9600';GREY='#95b1c1';PALE='#eaf5f8'
v=pd.read_csv(R/'results/final_ensemble/validation_comparison.csv').set_index('model')
t=pd.read_csv(R/'results/final_ensemble/local_test_backtest/comparison.csv').set_index('model')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
def save(fig,name):
 fig.tight_layout();fig.savefig(A/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig);return name
fig,ax=plt.subplots(1,2,figsize=(9,3))
for a,k,title in zip(ax,['mae_p50','calibration_pinball_mean'],['Price MAE (EUR/MWh)','P10/P90 pinball loss']):
 vals=v.loc[['organizer_baseline','current_lightgbm_single','final_blend'],k]
 bars=a.barh(['Baseline','LightGBM','Final blend'],vals,color=[GREY,ORANGE,BLUE]);a.invert_yaxis();a.bar_label(bars,fmt='%.2f',padding=4);a.set_xlim(0,max(vals)*1.22);a.set_title(title+' | lower is better')
save(fig,'01_validation.png')
fig,ax=plt.subplots(figsize=(9,2.9));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,4))
for x,y,w,h,title,body,col in [(0,2.2,3,1.2,'Availability-masked inputs','Prices / system / weather',BLUE),(3.6,2.2,2.8,1.2,'Five-seed LightGBM','Quantiles + probabilities',BLUE),(3.6,.3,2.8,1.2,'Zero-shot Chronos-2','Spread quantiles',BLUE),(7,1.2,2.9,1.2,'Horizon-weighted blend','P10 / P50 / P90',ORANGE)]:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=PALE,edgecolor=col));ax.text(x+w/2,y+.8,title,ha='center',color=col,weight='bold');ax.text(x+w/2,y+.35,body,ha='center',fontsize=9)
for start,end in [((3,2.8),(3.55,2.8)),((2.4,2.15),(3.55,.9)),((6.45,2.8),(7,2)),((6.45,.9),(7,1.5))]:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.7))
save(fig,'02_method.png')
fig,ax=plt.subplots(figsize=(9,2.8));origin=pd.Timestamp('2026-06-01',tz='UTC')
def day(s):return (pd.Timestamp(s)-origin).total_seconds()/86400
for y,a,b,label,col in [(2,'2026-06-01T00Z','2026-09-06T21Z','Pre-validation history',BLUE),(1,'2026-09-06T21Z','2026-10-04T21Z','Validation: 672 origins',ORANGE),(0,'2026-10-04T21Z','2026-11-01T22Z','Test: 673 origins',BLUE)]:
 ax.barh(y,day(b)-day(a),left=day(a),height=.6,color=col);ax.text(day(a)+1,y,label,va='center',color='white',fontsize=9)
ax.set_yticks([]);ticks=['2026-06-01T00Z','2026-07-01T00Z','2026-08-01T00Z','2026-09-01T00Z','2026-10-01T00Z','2026-11-01T00Z'];ax.set_xticks([day(x) for x in ticks],['Jun 1','Jul 1','Aug 1','Sep 1','Oct 1','Nov 1']);ax.set_title('Chronological evaluation; boundaries shown in UTC');save(fig,'03_split.png')
e=pd.read_csv(R/'results/experiment_4_external_data/comparison.csv').set_index('model');e=e.drop('submitted_B');base=e.loc['baseline','mae_p50'];fig,ax=plt.subplots(figsize=(9,2.7));names=[x for x in e.index if x!='baseline'];delta=e.loc[names,'mae_p50']-base
bars=ax.barh(names,delta,color=[BLUE if x<0 else ORANGE for x in delta]);ax.bar_label(bars,fmt='%+.3f',padding=5);ax.axvline(0,color='grey');ax.set_xlim(-.12,.085);ax.set_title('External-data experiment: validation MAE change vs matched control');ax.set_xlabel('EUR/MWh; below zero improves MAE. All paired 95% MAE intervals include zero.');save(fig,'04_external_data.png')
fig,axes=plt.subplots(1,3,figsize=(9,3));names=['previous_lightgbm','ensemble']
for ax,k,title in zip(axes,['mae_p50','calibration_pinball_mean','coverage_p10_p90'],['MAE (EUR/MWh)','Pinball loss','Coverage (%)']):
 vals=t.loc[names,k].astype(float)*(100 if k=='coverage_p10_p90' else 1);bars=ax.bar(['Previous','Blend'],vals,color=[GREY,BLUE]);ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_ylim(0,max(vals)*1.25);ax.set_title(title)
 if k=='coverage_p10_p90':ax.axhline(80,color=ORANGE,ls='--');ax.text(.5,.94,'80% target',transform=ax.transAxes,ha='center',fontsize=9,color=ORANGE)
save(fig,'05_local_test.png')
# Content is shared by both output formats. References point to repository evidence.
pages=[]
def page(title,subtitle,items):pages.append((title,subtitle,items))
def P(s):return ('p',s)
def H(s):return ('h',s)
def C(file,caption):return ('chart',file,caption)
def T(head,rows):return ('table',head,rows)
def N(s):return ('note',s)
page('Evaluation report','Turku GridCast | Since AI 2026 | 8 November 2026',[
 N('A substantial validation gain over the organiser baseline; a smaller, mixed incremental gain from blending.'),
 P('We forecast Finnish imbalance prices every hour for the next ten hours: 40 quarter-hours, P10/P50/P90 prices, balancing-direction probabilities and high/low spike probabilities. The final system blends five-seed LightGBM spread quantiles with zero-shot Chronos-2.'),
 C('01_validation.png','Figure 1. Same validation period and official scoring definition for all three models.'),
 T(['Validation metric','Baseline','Final blend','Reduction'],[['MAE, EUR/MWh','66.2033','41.5988','37.17%'],['P10/P90 pinball','24.1776','11.8864','50.84%'],['Spike Brier','0.113176','0.090792','19.78%'],['Direction Brier','0.450907','0.392898','12.87%'],['Interval coverage','68.29%','73.28%','80% target']]),
 P('Most of the baseline improvement comes from task-trained LightGBM and feature engineering. Relative to the previous LightGBM submission, the final blend improves validation MAE by 1.02% and pinball by 4.28%. These smaller gains should not be described as the full 37%/51% baseline improvement.'),
 P('Scope: validation results use organiser-supplied actuals; October test results are local post-hoc scores of frozen forecasts. Official organiser test scores and competition ranking remain pending. Source: results/final_ensemble/validation_comparison.csv.')])
page('Method and reasoning','Model the imbalance spread, then restore the known delivery spot price',[
 C('02_method.png','Figure 2. Only quantiles are blended. Direction and spike probabilities remain LightGBM-only.'),
 H('LightGBM: structured power-system information'),
 P('The target is imbalance price minus spot (EUR/MWh). Cumulative feature families cover calendar/spot/history (E1), spread lags/volatility/spike history (E2), supply-demand forecasts and actuals (E3), regulation/flows/neighbour prices (E4), external Fingrid real-time measurements (EX), and ECMWF weather (EW). Selected families are direction EW, spike EX, P10 E4 and P50/P90 EX.'),
 P('Each output is trained with seeds 7, 11, 23, 37 and 51; seed offsets for bagging, feature fraction and data sampling are +1/+2/+3. Average raw predictions before post-processing. Dedicated models override steps 1-4 and 5-8. Global quantiles use 250 rounds; global classifiers 400; short-horizon specialists 300. Full-precision settings are retained in lgb_completed.json.'),
 H('Chronos-2: a complementary historical representation'),
 P('Zero-shot Chronos-2 uses 2,048 quarter-hours of context ending at origin minus 45 minutes. Unavailable target/covariate values are masked. It predicts 42 positions; two bridge positions are discarded, leaving the 40 required quarters. Past inputs include regulation prices/direction/activation, real-time imbalance and wind error; known-future inputs include spot and day-ahead wind, load and solar.'),
 T(['Steps','Validation: LGBM / Chronos','Test: LGBM / Chronos'],[['1-4','0.7 / 0.3','0.7 / 0.3'],['5-16','0.4 / 0.6','0.4 / 0.6'],['17-40','0.5 / 0.5','0.6 / 0.4']]),
 P('Weights use a 0.1 simplex grid on out-of-sample training predictions. The same weight applies to P10/P50/P90. Add delivery spot after blending spreads, sort quantiles, normalise direction probabilities and cap each spike probability by its matching direction probability. No conformal adjustment is applied to the final blend. Sources: ensemble_spec.py, ens_members.py, fm_data.py, fm_run.py and fm_blend3.py under src/pkg/.')])
page('Validation design','Chronological folds, publication-aware fitting and separate evaluation periods',[
 C('03_split.png','Figure 3. Exact origin grids come from the starter templates, including the autumn clock change.'),
 P('Training data begin 1 June 2026. The named validation days are 7 September-4 October in Finnish local time. Exact UTC validation origins run 6 September 21:00 to 4 October 20:00: 672 origins and 26,880 rows. The scorer uses 26,700 price rows; 180 horizon-tail rows have no supplied validation actuals.'),
 P('The test period has 673 origins, 4 October 21:00 to 1 November 21:00 UTC inclusive, producing 26,920 rows. Step 1 starts at the origin; delivery time is origin + (step-1) x 15 minutes. There is no shuffled train/test split.'),
 T(['Rolling OOS fold','Origin interval, UTC (end exclusive)'],[['1','5 Jul 21:00 - 19 Jul 21:00'],['2','19 Jul 21:00 - 2 Aug 21:00'],['3','2 Aug 21:00 - 16 Aug 21:00'],['4','16 Aug 21:00 - 30 Aug 21:00'],['5','30 Aug 21:00 - 6 Sep 21:00']]),
 P('Every fit uses origin < fold start and target first-usable time <= fold start. The validation fit/selection cutoff is 6 September 21:00 UTC. The final test cutoff is 4 October 21:00 UTC: validation OOS observations may then enter final training/weight selection only once their targets are usable. Test-period targets never enter fitting or selection.'),
 P('Official metrics: MAE of P50; mean pinball at P10 and P90; mean high/low spike Brier at a +/-50 EUR/MWh spread threshold; and three-class direction Brier. Coverage is descriptive, with an 80% nominal target. Blend selection uses P10/P50/P90 pinball, distinct from the official P10/P90 metric.'),
 N('The stored normalized selection score S is an internal comparison objective, not official competition points. Actual points also depend on other accepted submissions and jury assessment.')])
page('Availability and leakage checks','Evidence of as-of forecasting, with explicit boundaries on what was tested',[
 P('A package value is visible only when first_usable_origin_utc <= forecast origin. Composite features inherit the latest source availability. External Fingrid measurements become usable at quarter end +30 minutes, rounded up to a full hour. ECMWF runs become usable at initialization +6 hours, rounded up. The latter is an assumed conservative delay, not measured historical API ingestion.'),
 T(['Check','Recorded result'],[['Final CSV format','673 origins; 26,920 rows; 0 errors, 0 warnings'],['LightGBM feature mutations','Passed at Jul 10, Sep 20 and Oct 20 cutoffs'],['LightGBM output mutations','75 models per cutoff; 225 model checks; bit-identical'],['Chronos input mutations','1,842 and 2,551 origins checked at two cutoffs'],['Chronos cutoff predictions','Bit-identical after unavailable-value mutation'],['Solo versus batch inference','All saved periods passed within numerical tolerance'],['Independent validation scoring','Matches saved metrics'],['Final forecast file integrity','Hashes match saved verification evidence']]),
 H('What the mutation test establishes'),
 P('The tests scramble values whose first-usable time is later than a cutoff and rebuild earlier features or contexts. Outputs at checked origins stay unchanged; mutations must not influence forecasts before publication. Chronos checks cover all required input contexts through the September/October cutoffs and cutoff predictions. These are recorded test results, not a new exhaustive rerun performed for this report.'),
 H('What the checks do not establish'),
 P('A format validator does not prove absence of leakage. Feature mutation tests do not audit foundation-model pretraining overlap, provider archive completeness or actual historical API arrival times. Those remain separate limitations. Repeated validation inspection also reduces the strength of claims that validation was wholly untouched.'),
 P('Evidence: results/final_ensemble/feature_leak_checks.json, lgb_output_leak_checks.json, fm_leak_checks.json, fm_batch_check_*.json, lgb_training_audit.json and verification.json. The original single-model fallback is retained separately from the final candidate.'),
 N('Final file SHA-256: '+hashlib.sha256((R/'submission/final_ensemble/forecast_backtest.csv').read_bytes()).hexdigest())])
page('Experiments and decisions','What we tried, what we retained, and what we dropped',[
 C('04_external_data.png','Figure 4. Small external-data MAE changes do not establish a reliable overall improvement.'),
 P('Initial Chronos-2 benchmarks tested price, spread and covariate targets with a seven-day context. None beat the existing LightGBM on both validation MAE and official pinball. The later final Chronos configuration uses a longer context and different covariate processing; its results must not be conflated with the initial experiment.'),
 P('TimesFM 3.0 and Timestral 209M were tested zero-shot on price/spread/covariate configurations. Their best local MAEs were 43.53 and 43.82, versus LightGBM 42.03. Neither model is in the final submission. No foundation-model fine-tuning was performed.'),
 P('Outages and archived ICON/HARMONIE weather produced small, mixed gains. Combined features reached MAE 41.9510 versus control 42.0286, but all paired MAE intervals included zero and pre-validation fold selection retained the baseline. These additions were not adopted into the final blend.'),
 P('Parameter tuning tested four feature sets with equal budgets, inherited settings plus seven random candidates per output, chronological search/confirmation, and matched calibration controls. It reduced median MAE in zero of four matched comparisons. The both_tuned result equalled its matched control on MAE; it is not evidence that tuning caused the gain over the saved submission.'),
 P('Rejected or deferred: Fingrid 245 without historical forecast vintages; licensed continuous intraday prices without access; extra neighbour measurements without verified publication history; new reserve bid/capacity series without accessible as-of archives. Final-blend conformal adjustment was omitted because extra calibration hurt training-fold selection.'),
 P('Sources: results/experiment_2_chronos/REPORT.md; experiment_3_foundation_models/REPORT.md; experiment_4_external_data/REPORT.md and source_audit.json; experiment_5_parameter_tuning/REPORT.md. The final blend adds a modest price/quantile gain; it does not reverse every negative experiment.')])
page('Historical test backtest','Frozen October forecasts scored locally, after model selection',[
 C('05_local_test.png','Figure 5. Local test comparison: similar price error, improved pinball, coverage closer to target.'),
 T(['Local test metric','Previous LightGBM','Final ensemble'],[['MAE, EUR/MWh','26.690916','26.702820'],['P10/P90 pinball','7.086564','6.830206'],['P10-P90 coverage','89.40%','80.72%'],['Spike Brier','0.056534','0.056664'],['Direction Brier','0.394655','0.393744']]),
 P('The organiser actuals_test.csv was not supplied locally. Realised imbalance price and mFRR direction were reconstructed from package inputs and joined with local spot for scoring. Reconstructed labels matched every supplied validation actual before this procedure was applied to test. All 26,920 test rows were scored using score.py 0.1.1; frozen forecast hashes remained unchanged.'),
 P('The blend reduces local test pinball by 3.62%, but MAE worsens by 0.0119 EUR/MWh (about 0.045%). Direction Brier improves slightly and spike Brier worsens slightly. Coverage is closer to 80%, but coverage alone is not the scored calibration objective. This is a mixed result, not an across-the-board win.'),
 H('Uncertainty in the incremental validation gain'),
 P('Paired day-bootstrap analysis (5,000 resamples of 28 Finnish origin days) gives blend-minus-LightGBM MAE -0.4298 with 95% interval [-1.5052, +0.6747], and pinball -0.5309 with interval [-1.3121, +0.1546]. Both include zero. Within-day forecast overlap is retained; longer dependence and repeated validation inspection limit inference.'),
 P('Do not interpret the drop from validation MAE 41.60 to test MAE 26.70 as a model upgrade: the periods and conditions differ. The official organiser may use a different locked actuals snapshot. Sources: results/final_ensemble/local_test_backtest/{README.md,comparison.csv,audit.json}; paired_day_bootstrap.csv.')])
page('Limitations and handoff','Operational prototype, reproducible evidence, and the next evaluation steps',[
 H('Limitations that materially affect the conclusion'),
 P('The sample is short and season-specific; rare spikes and stressed regimes remain difficult. Validation coverage is 73.28%, below the 80% target. The ensemble improvements over LightGBM are small with broad bootstrap intervals. There is no demonstrated trading P&L, production-control benefit or competition ranking.'),
 P('External publication delays are assumptions; archive completeness and model pretraining overlap are not certified. The final experiment audit verifies the requested blend and recomputes its weight grid, but does not reconstruct every upstream model-selection decision. Changes in market rules or data quality can invalidate learned relationships.'),
 H('User-facing situational view'),
 P('The local dashboard provides result tables, origin replay, ten-hour quantiles, direction/spike risks, previous-hour forecast revisions, retrospective actual overlays and historical scoring windows. Publication-aware history excludes outcomes not yet usable at the selected origin. Desktop Chrome interactions and data tests passed; mobile and public deployment remain unverified.'),
 P('Driver and freshness panels are explicitly labelled as references from the previous single-seed LightGBM, not explanations of the complete ensemble. Alert thresholds are illustrative expert-review prompts, not optimized trading or operating policies. Notebook execution through the IDE was blocked by kernel startup; its chart-generation cells are not claimed as execution-verified.'),
 H('Reproduction and submission'),
 P('Read RUNNING.md. Start the dashboard with .venv/bin/python dashboard/server.py. Validate submission/final_ensemble/forecast_backtest.csv against forecast_origins.csv. The root forecast_backtest.csv is the older fallback. Complete_Experiment.ipynb documents all experiments and optional reruns. Retraining requires the two Python environments, private feature matrix, archived inputs and pinned Chronos checkpoint.'),
 H('Data rights and next steps'),
 P('DATA_SOURCES_AND_LICENSES.md records final/research inputs, model licences, dependencies and media services. Final model members are LightGBM (MIT) and Chronos-2 (Apache-2.0); this is not a blanket licence for all assets. Nord Pool spot and private matrices must not be redistributed. TimesFM, Timestral, UMM and ElevenLabs have separate conditions.'),
 P('Next: evaluate more seasons and stress events; test calibration choices only on predeclared folds; obtain verifiable intraday wind/outage vintages; validate alerts with energy experts; assess operational costs and benefits in a controlled pilot. A forecast prototype is complete enough to inspect, not proven production-ready.'),
 N('Evidence index: results/final_ensemble/REPORT.md and local_test_backtest/; experiment reports 2-5; docs/STARTER_README.md; RUNNING.md; DATA_SOURCES_AND_LICENSES.md. Charts in this report use aggregate saved metrics only. No raw spot-price data is embedded.')])
# Render matching Markdown.
md=['# Turku GridCast - Evaluation report','Prepared 8 November 2026. Seven-section companion to the seven-page PDF.']
for number,(title,sub,items) in enumerate(pages,1):
 md += [f'\n## {number}. {title}',sub]
 for item in items:
  if item[0] in ['p','note']:md.append(('> ' if item[0]=='note' else '')+item[1])
  elif item[0]=='h':md.append('### '+item[1])
  elif item[0]=='chart':md += [f'![{item[2]}](output/pdf/charts/{item[1]})',item[2]]
  else:
   _,head,rows=item;md.append('\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in rows]))
(R/'EVALUATION_REPORT.md').write_text('\n\n'.join(md)+'\n')
W,H=595.28,841.89;M=42;CW=W-2*M
c=canvas.Canvas(str(O/'Turku_GridCast_Evaluation_Report.pdf'),pagesize=(W,H));c.setTitle('Turku GridCast - Evaluation report');c.setAuthor('Turku GridCast team')
style=ParagraphStyle('body',fontName='Helvetica',fontSize=9.7,leading=13.2,textColor=colors.HexColor('#233c48'),spaceAfter=8)
small=ParagraphStyle('small',parent=style,fontSize=8,leading=10.4)
headstyle=ParagraphStyle('heading',parent=style,fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=colors.HexColor(BLUE))
def para(text,sty,y,x=M,width=CW):
 p=Paragraph(escape(text),sty);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return y-height-8
for number,(title,sub,items) in enumerate(pages,1):
 c.setFillColor(colors.HexColor(BLUE));c.setFont('Helvetica-Bold',9);c.drawString(M,H-35,'TURKU GRIDCAST  /  SINCE AI 2026')
 c.setFillColor(colors.HexColor(ORANGE));c.rect(W-M-54,H-41,54,4,fill=1,stroke=0)
 c.setFont('Times-Roman',27);c.drawString(M,H-79,title)
 c.setFillColor(colors.HexColor('#617681'));c.setFont('Helvetica',9);c.drawString(M,H-99,sub)
 y=H-122
 for item in items:
  kind=item[0]
  if kind in ['p','h']:y=para(item[1],headstyle if kind=='h' else style,y)
  elif kind=='note':
   pp=Paragraph(escape(item[1]),small);_,hh=pp.wrap(CW-20,H);c.setFillColor(colors.HexColor(PALE));c.roundRect(M,y-hh-18,CW,hh+18,5,fill=1,stroke=0);pp.drawOn(c,M+10,y-hh-9);y-=hh+28
  elif kind=='chart':
   img=ImageReader(str(A/item[1]));iw,ih=img.getSize();height=CW*ih/iw;c.drawImage(img,M,y-height,width=CW,height=height);y-=height+5;y=para(item[2],small,y)
  else:
   _,header,rows=item;data=[[Paragraph(escape(x),small) for x in row] for row in [header,*rows]]
   widths=([CW*.31]+[CW*.69/(len(header)-1)]*(len(header)-1)) if len(header)>2 else [CW*.35,CW*.65]
   tbl=Table(data,colWidths=widths);tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(PALE)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#dbe7ec'))]));_,hh=tbl.wrap(CW,H);tbl.drawOn(c,M,y-hh);y-=hh+12
 assert y>48,(number,y)
 c.setStrokeColor(colors.HexColor('#dbe7ec'));c.line(M,40,W-M,40);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#617681'));c.drawString(M,27,'Frozen forecast evaluation | Local test score is not organiser-certified');c.drawRightString(W-M,27,f'{number} / 7');c.showPage()
c.save()
print('Built seven-page PDF and matching Markdown.')
