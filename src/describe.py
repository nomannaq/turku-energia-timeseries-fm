import numpy as np, pandas as pd
from features import *
q = quarter_table(); q = q[TRAIN[0]:TEST[1]]
q["split"] = np.select([q.index < pd.Timestamp(TRAIN[1], tz="UTC"), q.index < pd.Timestamp(VAL[1], tz="UTC")], ["train", "val"], "test")
q["up"] = (q.dir == 1); q["spk_hi"] = q.spread > 50; q["spk_lo"] = q.spread < -50; q["werr"] = q.wind_act - q.wind_da; q["cerr"] = q.cons_act - q.cons_da
print(q.groupby("split").agg(n=("imb", "size"), imb_mean=("imb", "mean"), imb_p99=("imb", lambda x: x.quantile(.99)), spread_med=("spread", "median"), spread_mad=("spread", lambda x: (x - x.median()).abs().mean()),
      up=("up", "mean"), spk_hi=("spk_hi", "mean"), spk_lo=("spk_lo", "mean"), spread0=("spread", lambda x: (x.abs() < 0.01).mean())).round(3).to_string())
print("\nmissing share:"); print(q.isna().mean().round(4)[q.isna().mean() > 0].to_string())
print("\nContemporaneous (NOT as-of) relation: wind error decile vs P(up), P(spike hi/lo)")
q["werr_dec"] = pd.qcut(q.werr, 10, labels=False)
print(q.groupby("werr_dec").agg(werr=("werr", "mean"), p_up=("up", "mean"), spk_hi=("spk_hi", "mean"), spk_lo=("spk_lo", "mean"), spread=("spread", "mean")).round(3).to_string())
print("\ncorr(sign) up vs werr:", round(q.up.astype(float).corr(q.werr), 3), " cerr:", round(q.up.astype(float).corr(q.cerr), 3))
up = q.up.astype(float)
print("direction persistence: corr(up_t, up_{t-k}) k=1,2,4,8,16,40:", [round(up.autocorr(k), 3) for k in (1, 2, 4, 8, 16, 40)])
print("wind error persistence: autocorr k=1,4,16,40:", [round(q.werr.autocorr(k), 3) for k in (1, 4, 16, 40)])
print("\nP(up) by Helsinki hour:"); print(q.groupby(q.index.tz_convert("Europe/Helsinki").hour).up.mean().round(2).to_dict())
