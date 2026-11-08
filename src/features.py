"""As-of feature builder. Every feature for origin o uses only data with first_usable_utc <= o (assumed lags, see LAGS)."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"
Q = pd.Timedelta("15min")
LAGS = {"q15": pd.Timedelta("30min"), "rt": pd.Timedelta("15min"), "fmi": pd.Timedelta("30min")}  # after interval END
TRAIN = ("2026-05-31 21:00", "2026-09-06 21:00"); VAL = ("2026-09-06 21:00", "2026-10-04 21:00"); TEST = ("2026-10-04 21:00", "2026-11-01 22:00")
FI_WIND = ["vaasa", "kristiinankaupunki", "pori", "raahe", "oulu", "kemi_tornio"]

def fg(i):
    d = pd.read_parquet(D / f"fingrid_{i}.parquet").set_index("startTime")["value"].astype(float)
    d = d[~d.index.duplicated()].sort_index()
    return d.resample("15min").mean() if (d.index[1] - d.index[0]) < Q else d

def quarter_table():
    e = pd.read_parquet(D / "esett_EXP14.parquet"); e["t"] = pd.to_datetime(e.timestampUTC, utc=True)
    fi = e[e.mba_code == "FI"].set_index("t").sort_index()
    q = pd.DataFrame(index=pd.date_range("2026-05-25", "2026-11-02", freq="15min", tz="UTC", inclusive="left"))
    q["imb"] = fg(319); q["dir"] = fg(369); q["spot"] = fi.imblSalesPrice - fi.imblSpotDifferencePrice
    q["spread"] = q.imb - q.spot
    q["wind_da"] = fg(246); q["cons_da"] = fg(165); q["wind_act"] = fg(75); q["cons_act"] = fg(124)
    for i, n in [(397, "calc_imb"), (181, "wind_rt"), (188, "nuc_rt"), (194, "netimp_rt"), (177, "freq"), (260, "kin"), (400, "sa_price"), (244, "up_price")]:
        try: q[n] = fg(i)
        except Exception as ex: print("missing", i, ex)
    for m in ["SE1", "SE3"]:
        s = e[e.mba_code == m].set_index("t").sort_index()
        q[f"imb_{m}"] = s.imblSalesPrice[~s.index.duplicated()]
    return q

def state_features(q):
    """Features describing the latest known state; index = time at which they become usable."""
    s = pd.DataFrame(index=q.index)
    up = (q.dir == 1).astype(float).where(q.dir.notna())
    s["last_dir"] = q.dir; s["up_share_1h"] = up.rolling(4).mean(); s["up_share_4h"] = up.rolling(16).mean(); s["up_share_24h"] = up.rolling(96).mean()
    s["last_spread"] = q.spread; s["spread_mean_4h"] = q.spread.rolling(16).mean(); s["spread_std_4h"] = q.spread.rolling(16).std()
    s["spread_med_24h"] = q.spread.rolling(96).median(); s["spike_share_24h"] = (q.spread.abs() > 50).astype(float).rolling(96).mean()
    s["wind_err_1h"] = (q.wind_act - q.wind_da).rolling(4).mean(); s["wind_err_4h"] = (q.wind_act - q.wind_da).rolling(16).mean()
    s["cons_err_1h"] = (q.cons_act - q.cons_da).rolling(4).mean()
    s["last_imb_SE1"] = q.imb_SE1; s["last_imb_SE3"] = q.imb_SE3
    s.index = s.index + Q + LAGS["q15"]           # usable at end + lag
    r = pd.DataFrame(index=q.index)
    r["calc_imb_15m"] = q.calc_imb; r["calc_imb_1h"] = q.calc_imb.rolling(4).mean()
    r["wind_rt_err"] = q.wind_rt - q.wind_da; r["nuc_drop"] = q.nuc_rt - q.nuc_rt.rolling(96).max()
    r["netimp"] = q.netimp_rt; r["freq_dev_1h"] = (q.freq - 50).rolling(4).mean(); r["kin"] = q.kin
    r["last_sa_price_minus_spot"] = q.sa_price - q.spot
    r.index = r.index + Q + LAGS["rt"]
    return s, r

def om_table(path, locs=None):
    w = pd.read_parquet(path)
    if locs is not None: w = w[w.location.isin(locs)]
    return w

def weather_features(origins, deliveries, w, prefix, groups):
    """For each (origin, delivery): latest OM run with first_usable<=origin; value at delivery hour; revision vs previous run."""
    w = w.copy(); w["vh"] = w.valid_time_utc
    runs = pd.DatetimeIndex(np.sort(w.first_usable_utc.unique())).tz_convert("UTC") if pd.DatetimeIndex(w.first_usable_utc.unique()).tz else pd.DatetimeIndex(np.sort(w.first_usable_utc.unique())).tz_localize("UTC")
    od = pd.DataFrame({"o": pd.DatetimeIndex(origins), "t": pd.DatetimeIndex(deliveries)}); od["vh"] = od.t.dt.floor("h")
    idx = runs.searchsorted(pd.DatetimeIndex(od.o), side="right") - 1
    od["fu"] = runs[np.clip(idx, 0, None)]; od.loc[idx < 0, "fu"] = pd.NaT
    od["fu_prev"] = runs[np.clip(idx - 1, 0, None)]
    out = pd.DataFrame(index=od.index)
    for g, locs in groups.items():
        ww = w[w.location.isin(locs)].copy()
        ww["pw"] = np.clip(ww.wind_speed_100m, 3, 13) ** 3 / 13**3     # crude power-curve proxy
        agg = ww.groupby(["first_usable_utc", "vh"]).agg(ws=("wind_speed_100m", "mean"), pw=("pw", "mean"), t2=("temperature_2m", "mean"), sw=("shortwave_radiation", "mean"), ws10=("wind_speed_10m", "mean"))
        cur = od.join(agg, on=["fu", "vh"]); prv = od.join(agg, on=["fu_prev", "vh"])
        for c in ["ws", "pw", "t2", "sw", "ws10"]: out[f"{prefix}{g}_{c}"] = cur[c].values
        out[f"{prefix}{g}_ws_rev"] = (cur.ws - prv.ws).values; out[f"{prefix}{g}_pw_rev"] = (cur.pw - prv.pw).values
        # wind-speed ramp over next hour
        nxt = od.assign(vh=od.vh + pd.Timedelta("1h")).join(agg, on=["fu", "vh"])
        out[f"{prefix}{g}_pw_ramp"] = (nxt.pw - cur.pw).values
    return out
