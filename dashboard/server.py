"""Local-only historical forecast dashboard. Run: .venv/bin/python dashboard/server.py"""
import argparse, json, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from score import compute_metrics
KEYS=['forecast_origin_utc','delivery_start_utc','step']

def records(df):
    return json.loads(df.to_json(orient='records',date_format='iso'))

class Data:
    def __init__(self):
        self.frames={};self.previous={};self.explanations={};self.joined={}
        inputs=pd.read_parquet(ROOT/'data/inputs.parquet',columns=['series_id','timestamp_utc','first_usable_origin_utc','value'])
        availability=inputs[inputs.series_id=='fi_imbalance_price'].copy()
        availability['delivery_start_utc']=pd.to_datetime(availability.timestamp_utc,utc=True)
        availability['usable']=pd.to_datetime(availability.first_usable_origin_utc,utc=True)
        availability=availability[['delivery_start_utc','usable']].drop_duplicates('delivery_start_utc')
        for period in ['validation','test']:
            path='validation_forecast.csv' if period=='validation' else 'forecast_backtest.csv'
            f=pd.read_csv(ROOT/'submission/final_ensemble'/path,parse_dates=KEYS[:2]);self.frames[period]=f
            old=ROOT/('submission/validation_forecast_B.csv' if period=='validation' else 'forecast_backtest.csv')
            self.previous[period]=pd.read_csv(old,parse_dates=KEYS[:2])
            self.explanations[period]=pd.read_csv(ROOT/f'submission/dashboard/{period}_B.csv',parse_dates=KEYS[:2])
            ap=ROOT/('data/actuals_validation.csv' if period=='validation' else 'work_private/final_ensemble/local_actuals_test.csv')
            if ap.exists():
                actual=pd.read_csv(ap,parse_dates=['delivery_start_utc'])
            elif period=='test':
                # Fresh checkouts need no private generated actuals file. Reconstruct in memory.
                series=inputs[inputs.series_id.isin(['fi_imbalance_price','fi_mfrr_direction'])]
                wide=series.pivot(index='timestamp_utc',columns='series_id',values='value')
                spot=pd.read_csv(ROOT/'data/spot_nordpool.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc').value
                wide.index=pd.to_datetime(wide.index,utc=True)
                actual=pd.DataFrame({'delivery_start_utc':wide.index,'imbalance_price':wide.fi_imbalance_price.to_numpy(),
                    'spot_price':spot.reindex(wide.index).to_numpy(),'direction':wide.fi_mfrr_direction.map({-1:'down',0:'none',1:'up'}).to_numpy()})
            else:raise FileNotFoundError(f'Missing starter data: {ap}')
            actual=actual.drop(columns=['first_usable_origin_utc'],errors='ignore').merge(availability,on='delivery_start_utc',how='left',validate='one_to_one')
            self.joined[period]=f.merge(actual,on='delivery_start_utc',how='left',validate='many_to_one')
            self.previous[period]=self.previous[period].merge(actual,on='delivery_start_utc',how='left',validate='many_to_one')
    def origins(self,p):return [t.isoformat() for t in sorted(self.frames[p].forecast_origin_utc.unique())]
    def forecast(self,p,o):
        f=self.frames[p]; rows=f[f.forecast_origin_utc==o].sort_values('step')
        if len(rows)!=40:raise ValueError('Select an available forecast origin')
        prev=f[f.forecast_origin_utc==o-pd.Timedelta(hours=1)][['delivery_start_utc','p50']].rename(columns={'p50':'previous_p50'})
        actual=self.joined[p][self.joined[p].forecast_origin_utc==o][['delivery_start_utc','imbalance_price','usable']]
        rows=rows.merge(prev,on='delivery_start_utc',how='left').merge(actual,on='delivery_start_utc',how='left')
        e=self.explanations[p]; e=e[e.forecast_origin_utc==o].sort_values('step')
        explanation=e[['step','top_drivers_p50']+[c for c in e if c.startswith('age_min_')]+['weather_run_utc']]
        return dict(rows=records(rows),reference_explanations=records(explanation),explanation_scope='Previous single-seed LightGBM B reference only; not an attribution of the final ensemble.',actuals_scope='Retrospective realised prices; not available when these forecasts were issued.')
    def performance(self,p,o,days,mode):
        end=o if mode=='asof' else self.frames[p].forecast_origin_utc.max()+pd.Timedelta(hours=1)
        start=self.frames[p].forecast_origin_utc.min() if days==0 else end-pd.Timedelta(days=days)
        def select(j):
            mask=(j.forecast_origin_utc>=start)&(j.forecast_origin_utc<end)
            if mode=='asof':mask &= (j.usable<=o)&(j.delivery_start_utc<o)
            return j[mask].dropna(subset=['imbalance_price','spot_price','direction'])
        j=select(self.joined[p]);old=select(self.previous[p])
        def metrics(x):
            if x.empty:return None
            m=compute_metrics(x,50)
            return {k:m[k] for k in ['mae_p50','calibration_pinball_mean','coverage_p10_p90','brier_spike_mean','brier_direction']}
        daily=[]
        if len(j):
            for day,g in j.groupby(j.forecast_origin_utc.dt.strftime('%Y-%m-%d')):daily.append({'day':day,'mae':float((g.p50-g.imbalance_price).abs().mean()),'rows':len(g)})
        return dict(final=metrics(j),previous=metrics(old),rows=len(j),origins=int(j.forecast_origin_utc.nunique()),daily=daily,start=start.isoformat(),end=end.isoformat(),mode=mode,scope='Official validation labels' if p=='validation' else 'Local reconstructed test actuals; official organiser score pending')

def make_handler(data):
 class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u=urlparse(self.path);q={k:v[0] for k,v in parse_qs(u.query).items()}
        try:
            if u.path.startswith('/api/'):
                p=q.get('period','test')
                if p not in data.frames:raise ValueError('Invalid period')
                if u.path=='/api/results':
                    val=pd.read_csv(ROOT/'results/final_ensemble/validation_comparison.csv')
                    val=val[val.model.isin(['organizer_baseline','current_lightgbm_single','final_blend'])]
                    test=pd.read_csv(ROOT/'results/final_ensemble/local_test_backtest/comparison.csv')
                    payload={'validation':records(val),'test':records(test)}
                elif u.path=='/api/origins':payload={'origins':data.origins(p)}
                else:
                    o=pd.Timestamp(q['origin'])
                    if o.tzinfo is None:raise ValueError('Origin must include timezone')
                    if o.isoformat() not in data.origins(p):raise ValueError('Unknown origin')
                    if u.path=='/api/forecast':payload=data.forecast(p,o)
                    elif u.path=='/api/performance':
                        days=int(q.get('days','7'));mode=q.get('mode','asof')
                        if days not in [0,1,7,28] or mode not in ['asof','full']:raise ValueError('Invalid evaluation setting')
                        payload=data.performance(p,o,days,mode)
                    elif u.path=='/api/download':
                        f=data.frames[p];body=f[f.forecast_origin_utc==o].to_csv(index=False).encode()
                        return self.send(body,'text/csv','forecast_origin.csv')
                    else:raise ValueError('Unknown endpoint')
                return self.send(json.dumps(payload,allow_nan=False).encode(),'application/json')
            routes={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if u.path not in routes:self.send_error(404);return
            name=routes[u.path];self.send((ROOT/'dashboard/static'/name).read_bytes(),{'html':'text/html','js':'text/javascript','css':'text/css'}[name.split('.')[-1]])
        except (ValueError,KeyError) as e:self.send(json.dumps({'error':str(e)}).encode(),'application/json',status=400)
    def send(self,body,mime,filename=None,status=200):
        self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass
 return Handler
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    try:
        data=Data()
    except FileNotFoundError as e:
        parser.exit(1,f'Missing dashboard input: {e.filename or e}\nSee RUNNING.md for the local starter data and forecast files required.\n')
    try:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(data))
    except OSError as e:
        parser.exit(1,f'Cannot start dashboard on port {args.port}: {e}. Try --port 8766.\n')
    print(f'Turku GridCast: http://127.0.0.1:{args.port}',flush=True);server.serve_forever()
