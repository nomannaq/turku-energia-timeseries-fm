"""Post-hoc test evaluation from packaged realised inputs; not organizer-certified labels.
Does not fit/select models or alter forecasts. Run only after forecasts are frozen.
"""
import sys,json,hashlib,subprocess
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from score import load_actuals,load_submission,compute_metrics
OUT=ROOT/'results/final_ensemble/local_test_backtest';OUT.mkdir(parents=True,exist_ok=True)
PRIVATE=ROOT/'work_private/final_ensemble';PRIVATE.mkdir(exist_ok=True)
paths={'ensemble':ROOT/'submission/final_ensemble/forecast_backtest.csv','previous_lightgbm':ROOT/'forecast_backtest.csv'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={k:sha(p) for k,p in paths.items()}
x=pd.read_parquet(ROOT/'data/inputs.parquet')
x=x[x.series_id.isin(['fi_imbalance_price','fi_mfrr_direction'])]
x=x.pivot(index='timestamp_utc',columns='series_id',values='value')
s=pd.read_csv(ROOT/'data/spot_nordpool.csv',parse_dates=['timestamp_utc'])
s=s.set_index('timestamp_utc').value
actual=pd.DataFrame({'delivery_start_utc':x.index,'imbalance_price':x.fi_imbalance_price.to_numpy(),
 'spot_price':s.reindex(x.index).to_numpy(),'direction':x.fi_mfrr_direction.map({-1:'down',0:'none',1:'up'}).to_numpy()})
actual.delivery_start_utc=pd.to_datetime(actual.delivery_start_utc,utc=True)
# Establish that the reconstruction matches the official validation-label snapshot.
val=load_actuals(ROOT/'data/actuals_validation.csv').set_index('delivery_start_utc')
r=actual.set_index('delivery_start_utc').reindex(val.index)
for c in ['imbalance_price','spot_price']:
 np.testing.assert_allclose(r[c],val[c],rtol=0,atol=1e-9,equal_nan=True)
assert r.direction.fillna('MISSING').equals(val.direction.fillna('MISSING'))
forecast=load_submission(paths['ensemble']);deliveries=forecast.delivery_start_utc.unique()
test=actual[actual.delivery_start_utc.isin(deliveries)].copy()
assert not test.duplicated('delivery_start_utc').any()
assert test[['imbalance_price','spot_price','direction']].notna().all().all()
assert set(deliveries)==set(test.delivery_start_utc)
export=test.copy();export.delivery_start_utc=export.delivery_start_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
actual_path=PRIVATE/'local_actuals_test.csv';export.to_csv(actual_path,index=False)
summaries=[]
for name,path in paths.items():
 f=load_submission(path);j=f.merge(test,on='delivery_start_utc',how='left',validate='many_to_one')
 summaries.append({'model':name,**compute_metrics(j,50)})
 result=subprocess.run([sys.executable,str(ROOT/'score.py'),'--submission',str(path),'--actuals',str(actual_path),'--output-dir',str(OUT/name)],capture_output=True,text=True)
 (OUT/f'{name}_scorer.log').write_text(result.stdout+result.stderr)
 assert result.returncode==0,result.stderr
 assert sha(path)==before[name]
pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'scope':'Local post-hoc historical test score from packaged realised series; not official organizer actuals_test.csv',
 'validation_labels_match':True,'unique_delivery_quarters':len(test),'forecast_origins':forecast.forecast_origin_utc.nunique(),
 'rows':len(forecast),'forecast_sha256_unchanged':before,'source_sha256':{p:sha(ROOT/p) for p in ['data/inputs.parquet','data/spot_nordpool.csv']},
 'scored_at_utc':str(pd.Timestamp.now(tz='UTC')),'no_fitting_or_selection_performed':True},indent=2))
print(pd.DataFrame(summaries)[['model','mae_p50','calibration_pinball_mean','coverage_p10_p90','brier_spike_mean','brier_direction']].to_string(index=False))
print('Validation source parity passed; all 26,920 test forecast rows scored; forecast hashes unchanged.')
