"""V2 as-of feature builder. Output: data/feature_matrix_v2.parquet (one row per origin x step).
Availability (assumed): 15-min Fingrid/eSett series usable at slot end + 30 min; real-time Fingrid (1/3-min -> 15-min mean,
labelled by slot START, i.e. [t, t+15)) usable at slot end + 15 min; day-ahead 246/165 and spot known for all 40 steps;
ECMWF run usable at run + 6 h; 400 for slot t usable at t - 7.5 min (step 1 only)."""
import numpy as np, pandas as pd, holidays
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"
Q = pd.Timedelta("15min"); L15 = pd.Timedelta("30min"); LRT = pd.Timedelta("15min")
TRAIN = ("2026-05-31 21:00", "2026-09-06 21:00"); VAL = ("2026-09-06 21:00", "2026-10-04 21:00"); TEST = ("2026-10-04 21:00", "2026-11-01 22:00")
# rough installed-capacity weights per region (approximate, Finnish Wind Power Association regional split; to refine)
FI_W = {"raahe": 0.30, "oulu": 0.20, "vaasa": 0.15, "kristiinankaupunki": 0.15, "kemi_tornio": 0.10, "pori": 0.10}
IDX = pd.date_range("2026-05-20", "2026-11-03", freq="15min", tz="UTC", inclusive="left")

def fg(i, how="mean"):
    d = pd.read_parquet(D / f"fingrid_{i}.parquet").drop_duplicates("startTime").set_index("startTime")["value"].astype(float).sort_index()
    if (d.index[1] - d.index[0]) < Q:
        g = d.resample("15min")                       # label = slot start, covers [t, t+15)
        cnt = g.count(); d = getattr(g, how)()
        d[cnt < 2] = np.nan                           # require >=2 samples in quarter, else missing (no blind fill)
    return d.reindex(IDX)

def quarter_table():
    e = pd.read_parquet(D / "esett_EXP14.parquet"); e["t"] = pd.to_datetime(e.timestampUTC, utc=True)
    def es(m, c): s = e[e.mba_code == m].drop_duplicates("t").set_index("t")[c].astype(float); return s.reindex(IDX)
    q = pd.DataFrame(index=IDX)
    q["imb"] = fg(319); q["dir"] = fg(369)
    q["spot"] = es("FI", "imblSalesPrice") - es("FI", "imblSpotDifferencePrice")
    q["imb"] = q.imb.fillna(es("FI", "imblSalesPrice"))
    q["spread"] = q.imb - q.spot
    q["afrr_up"] = es("FI", "upRegPriceFrrA"); q["afrr_dn"] = es("FI", "downRegPriceFrrA")
    # eSett codes "no aFRR activation" as 0.00 (28% down / 33% up quarters, no true nulls): treat 0 as missing + flag, never as a price
    q["afrr_up_noact"] = ((q.afrr_up == 0) | q.afrr_up.isna()).astype(float); q["afrr_dn_noact"] = ((q.afrr_dn == 0) | q.afrr_dn.isna()).astype(float)
    q["afrr_missing"] = q.afrr_up_noact * q.afrr_dn_noact
    q.loc[q.afrr_up_noact == 1, "afrr_up"] = np.nan; q.loc[q.afrr_dn_noact == 1, "afrr_dn"] = np.nan
    q["mfrr_up"] = fg(244); q["sa_price"] = fg(400)
    q["wind_da"] = fg(246); q["cons_da"] = fg(165); q["wind_act"] = fg(75); q["cons_act"] = fg(124)
    q["calc_imb"] = fg(397); q["wind_rt"] = fg(181); q["nuc"] = fg(188); q["netimp"] = fg(194); q["freq"] = fg(177); q["kin"] = fg(260)
    q["freq_std"] = fg(177, "std"); q["calc_imb_min"] = fg(397, "min"); q["calc_imb_max"] = fg(397, "max")
    for m in ["SE1", "SE3"]: q[f"imb_{m}"] = es(m, "imblSalesPrice")
    # short gaps in real-time series: ffill max 1 quarter (still respects availability because done before shifting)
    for c in ["calc_imb", "wind_rt", "nuc", "netimp", "freq", "kin", "freq_std", "calc_imb_min", "calc_imb_max"]: q[c] = q[c].ffill(limit=1)
    return q

def roll(s, n, f): return getattr(s.rolling(n, min_periods=max(1, n // 2)), f)()

def state_features(q):
    s = {}
    sp = q.spread
    s["sp_last"] = sp
    for k in [1, 2, 4, 8, 24, 96]: s[f"sp_lag{k}"] = sp.shift(k)
    for n in [4, 12, 24, 96]:
        s[f"sp_mean{n}"] = roll(sp, n, "mean"); s[f"sp_std{n}"] = roll(sp, n, "std"); s[f"sp_min{n}"] = roll(sp, n, "min"); s[f"sp_max{n}"] = roll(sp, n, "max")
    for n in [4, 16, 96]: s[f"sp_ewm{n}"] = sp.ewm(span=n).mean()
    s["sp_med96"] = roll(sp, 96, "median"); s["sp_absmed672"] = sp.abs().rolling(672, min_periods=96).median()
    s["spike_hi_96"] = roll((sp > 50).astype(float), 96, "mean"); s["spike_lo_96"] = roll((sp < -50).astype(float), 96, "mean")
    s["spike_any_672"] = (sp.abs() > 50).astype(float).rolling(672, min_periods=96).mean(); s["sp_std672"] = sp.rolling(672, min_periods=96).std()
    up = (q.dir == 1).astype(float).where(q.dir.notna())
    s["dir_last"] = q.dir
    run = q.dir.ne(q.dir.shift()).cumsum(); s["dir_runlen"] = q.dir.groupby(run).cumcount() + 1
    for n in [4, 12, 24, 96]: s[f"up_share{n}"] = roll(up, n, "mean")
    s["up_ewm8"] = up.ewm(span=8).mean()
    we = q.wind_act - q.wind_da; ce = q.cons_act - q.cons_da
    s["werr_last"] = we
    for k in [4, 8]: s[f"werr_lag{k}"] = we.shift(k)
    for n in [4, 12, 24, 96]: s[f"werr_mean{n}"] = roll(we, n, "mean")
    s["werr_ewm8"] = we.ewm(span=8).mean(); s["werr_rel"] = we / q.wind_da.clip(lower=200)
    s["cerr_last"] = ce; s["cerr_mean4"] = roll(ce, 4, "mean"); s["cerr_mean12"] = roll(ce, 12, "mean")
    s["wind_act_last"] = q.wind_act
    s["afrr_up_minus_spot"] = q.afrr_up - q.spot; s["afrr_dn_minus_spot"] = q.afrr_dn - q.spot; s["afrr_missing"] = q.afrr_missing
    s["afrr_missing_4"] = roll(q.afrr_missing, 4, "mean"); s["afrr_up_act_share16"] = 1 - roll(q.afrr_up_noact, 16, "mean"); s["afrr_dn_act_share16"] = 1 - roll(q.afrr_dn_noact, 16, "mean")
    s["mfrr_up_minus_spot"] = q.mfrr_up - q.spot
    for m in ["SE1", "SE3"]: s[f"fi_minus_{m}_imb"] = q.imb - q[f"imb_{m}"]; s[f"{m}_imb_mean4"] = roll(q[f"imb_{m}"], 4, "mean")
    # --- adapted from Saleh et al. (2026) Table A.1: aFRR price lags/rolling, spot & consumption history, log transforms
    for c in ["afrr_up", "afrr_dn"]:
        v = q[c] - q.spot
        for k in [48, 96]: s[f"{c}_lag{k}"] = v.shift(k)
        for n in [48, 96]: s[f"{c}_mean{n}"] = v.rolling(n, min_periods=4).mean(); s[f"{c}_std{n}"] = v.rolling(n, min_periods=4).std()
        s[f"{c}_imp96"] = v.fillna(v.rolling(96, min_periods=1).mean())       # backward-looking 24h-mean imputation (paper); flag kept separately
    s["spot_past_mean96"] = roll(q.spot, 96, "mean"); s["spot_past_mean192"] = roll(q.spot, 192, "mean"); s["spot_past_std48"] = roll(q.spot, 48, "std"); s["spot_past_std96"] = roll(q.spot, 96, "std")
    s["cons_lag96"] = q.cons_act.shift(96); s["cons_mean96"] = roll(q.cons_act, 96, "mean")
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x))
    s["sp_slog_last"] = slog(sp); s["sp_slog_mean4"] = slog(roll(sp, 4, "mean")); s["sp_slog_lag96"] = slog(sp.shift(96))
    S = pd.DataFrame(s); S.index = S.index + Q + L15
    r = {}
    ci = q.calc_imb
    r["ci_last"] = ci
    for k in [1, 2, 4]: r[f"ci_lag{k}"] = ci.shift(k)
    for n in [4, 12, 24, 96]: r[f"ci_mean{n}"] = roll(ci, n, "mean")
    r["ci_std4"] = roll(ci, 4, "std"); r["ci_min_q"] = q.calc_imb_min; r["ci_max_q"] = q.calc_imb_max; r["ci_ewm4"] = ci.ewm(span=4).mean()
    wre = q.wind_rt - q.wind_da
    r["wrt_err_last"] = wre; r["wrt_err_mean4"] = roll(wre, 4, "mean"); r["wrt_err_mean12"] = roll(wre, 12, "mean"); r["wrt_err_d4"] = wre - wre.shift(4)
    r["nuc_last"] = q.nuc; r["nuc_drop24h"] = q.nuc - roll(q.nuc, 96, "max"); r["nuc_d4"] = q.nuc - q.nuc.shift(4)
    r["netimp_last"] = q.netimp; r["netimp_mean4"] = roll(q.netimp, 4, "mean"); r["netimp_mean24"] = roll(q.netimp, 24, "mean"); r["netimp_d4"] = q.netimp - q.netimp.shift(4)
    r["freq_dev4"] = roll(q.freq - 50, 4, "mean"); r["freq_std_q"] = q.freq_std; r["kin_last"] = q.kin
    r["sa_last_minus_spot"] = q.sa_price - q.spot; r["sa_mean4_minus_spot"] = roll(q.sa_price - q.spot, 4, "mean")
    Rt = pd.DataFrame(r); Rt.index = Rt.index + Q + LRT
    return S, Rt

def delivery_features(q, X):
    loc = X.t.dt.tz_convert("Europe/Helsinki"); oloc = X.o.dt.tz_convert("Europe/Helsinki")
    fh = holidays.Finland(years=[2026])
    F = pd.DataFrame(index=X.index)
    F["hour"] = loc.dt.hour; F["qh"] = loc.dt.minute // 15; F["dow"] = loc.dt.day_of_week; F["weekend"] = (F.dow >= 5).astype(int)
    hf = loc.dt.hour + loc.dt.minute / 60
    F["hour_sin"] = np.sin(2 * np.pi * hf / 24); F["hour_cos"] = np.cos(2 * np.pi * hf / 24)
    F["dow_sin"] = np.sin(2 * np.pi * loc.dt.day_of_week / 7); F["dow_cos"] = np.cos(2 * np.pi * loc.dt.day_of_week / 7)
    F["holiday"] = loc.dt.date.map(lambda d: d in fh).astype(int); F["origin_hour"] = oloc.dt.hour; F["hour_x_step"] = F.hour * 100 + X.step
    spot = q.spot; day = spot.index.tz_convert("Europe/Helsinki").date
    g = spot.groupby(day)
    rank = g.rank(pct=True); dmean = g.transform("mean"); dmax = g.transform("max"); dmin = g.transform("min")
    t = X.t
    F["spot"] = spot.reindex(t).values; F["spot_rank_day"] = rank.reindex(t).values; F["spot_minus_daymean"] = (spot - dmean).reindex(t).values
    F["spot_range_day"] = (dmax - dmin).reindex(t).values
    for k in [1, 4]: F[f"spot_ramp{k}"] = (spot - spot.shift(k)).reindex(t).values; F[f"spot_fwd{k}"] = (spot.shift(-k) - spot).reindex(t).values
    F["spot_minus_spot_o"] = F.spot - spot.reindex(X.o).values
    res = q.cons_da - q.wind_da
    F["wind_da"] = q.wind_da.reindex(t).values; F["cons_da"] = q.cons_da.reindex(t).values; F["resid_da"] = res.reindex(t).values
    for k in [1, 4]: F[f"resid_ramp{k}"] = (res - res.shift(k)).reindex(t).values; F[f"wind_da_ramp{k}"] = (q.wind_da - q.wind_da.shift(k)).reindex(t).values
    F["wind_da_minus_o"] = F.wind_da - q.wind_da.reindex(X.o).values; F["resid_minus_o"] = F.resid_da - res.reindex(X.o).values
    F["resid_rank_day"] = res.groupby(day).rank(pct=True).reindex(t).values
    F["sa_price_t_minus_spot"] = np.where(X.step == 1, (q.sa_price - q.spot).reindex(t).values, np.nan)
    return F

def weather_features(X, w, groups, prefix, weights=None):
    """Latest ECMWF run usable at origin; values at delivery hour; rolling/ramps along valid time; revision vs previous run."""
    runs = pd.DatetimeIndex(np.sort(w.first_usable_utc.unique()))
    uo = pd.DatetimeIndex(X.o.unique()); idx = runs.searchsorted(uo, side="right") - 1
    fu = pd.Series(runs[np.clip(idx, 0, None)], index=uo); fu_prev = pd.Series(runs[np.clip(idx - 1, 0, None)], index=uo)
    od = pd.DataFrame({"fu": fu.reindex(X.o).values, "fu_prev": fu_prev.reindex(X.o).values, "vh": X.t.dt.floor("h").values}, index=X.index)
    od["fu"] = pd.to_datetime(od.fu, utc=True); od["fu_prev"] = pd.to_datetime(od.fu_prev, utc=True); od["vh"] = pd.to_datetime(od.vh, utc=True)
    out = pd.DataFrame(index=X.index)
    for gname, locs in groups.items():
        ww = w[w.location.isin(locs)].copy()
        ww["pw"] = np.clip((ww.wind_speed_100m - 3) / 10, 0, 1) ** 1.5 * (ww.wind_speed_100m < 25)   # crude normalised power curve
        wt = ww.location.map(weights or {l: 1 / len(locs) for l in locs}).fillna(0)
        ww["wd_sin"] = np.sin(np.deg2rad(ww.wind_direction_100m)); ww["wd_cos"] = np.cos(np.deg2rad(ww.wind_direction_100m))
        ww["ws2"] = ww.wind_speed_100m ** 2
        V = ww.wind_speed_10m * 3.6; T = ww.temperature_2m
        ww["wchill"] = np.where((T <= 10) & (V > 4.8), 13.12 + 0.6215 * T - 11.37 * V ** 0.16 + 0.3965 * T * V ** 0.16, T)
        for c in ["wind_speed_100m", "pw", "temperature_2m", "shortwave_radiation", "wd_sin", "wd_cos", "ws2", "wchill", "wind_speed_10m"]: ww[c + "_w"] = ww[c] * wt
        ww["wt"] = wt
        a = ww.groupby(["first_usable_utc", "valid_time_utc"]).agg(ws=("wind_speed_100m_w", "sum"), pw=("pw_w", "sum"), t2=("temperature_2m_w", "sum"), sw=("shortwave_radiation_w", "sum"), wd_sin=("wd_sin_w", "sum"), wd_cos=("wd_cos_w", "sum"),
                                                                  ws2=("ws2_w", "sum"), wchill=("wchill_w", "sum"), ws10=("wind_speed_10m_w", "sum"), wsum=("wt", "sum"), ws_sd=("wind_speed_100m", "std"))
        for c in ["ws", "pw", "t2", "sw", "wd_sin", "wd_cos", "ws2", "wchill", "ws10"]: a[c] = a[c] / a.wsum
        a = a.drop(columns="wsum").sort_index()
        gr = a.groupby(level=0)
        a["pw_roll3c"] = gr.pw.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())   # +-3h along valid time, same run
        a["pw_ramp1"] = gr.pw.diff(); a["pw_ramp3"] = gr.pw.diff(3); a["pw_fwd3"] = gr.pw.shift(-3) - a.pw
        a["t2_lag3"] = gr.t2.shift(3); a["t2_lag6"] = gr.t2.shift(6); a["t2_roll6"] = gr.t2.transform(lambda s: s.rolling(6, min_periods=1).mean())
        a["t2_roll12"] = gr.t2.transform(lambda s: s.rolling(12, min_periods=1).mean()); a["t2_diff6"] = a.t2 - a.t2_lag6
        a["ws_lag3"] = gr.ws.shift(3); a["ws_roll6"] = gr.ws.transform(lambda s: s.rolling(6, min_periods=1).mean()); a["is_high_wind"] = (a.ws > 12).astype(float)
        a["log_ws"] = np.log1p(a.ws)
        a["ws_max6"] = gr.ws.transform(lambda s: s.rolling(7, center=True, min_periods=1).max())
        cur = od.join(a, on=["fu", "vh"]); prv = od.join(a[["pw", "ws"]], on=["fu_prev", "vh"])
        for c in a.columns: out[f"{prefix}{gname}_{c}"] = cur[c].values
        out[f"{prefix}{gname}_pw_rev"] = (cur.pw - prv.pw).values; out[f"{prefix}{gname}_ws_rev"] = (cur.ws - prv.ws).values
        out[f"{prefix}{gname}_lead_h"] = ((od.vh - od.fu) / pd.Timedelta("1h") + 6).values
    return out

def build():
    q = quarter_table(); S, Rt = state_features(q)
    origins = pd.date_range(TRAIN[0], TEST[1], freq="1h", tz="UTC", inclusive="left")
    X = pd.DataFrame({"o": np.repeat(origins, 40), "step": np.tile(np.arange(1, 41), len(origins))})
    X["t"] = X.o + (X.step - 1) * Q
    def asof(F):
        F = F[~F.index.duplicated(keep="last")].sort_index()
        uo = pd.DataFrame({"o": origins}); m = pd.merge_asof(uo, F, left_on="o", right_index=True, direction="backward").set_index("o")
        return m.reindex(X.o).reset_index(drop=True)
    Sx = asof(S).add_prefix("st_"); Rx = asof(Rt).add_prefix("rt_")
    F = delivery_features(q, X).add_prefix("dl_")
    fi = pd.read_parquet(D / "openmeteo_ecmwf_runs.parquet")
    W = weather_features(X, fi, {"fiw": list(FI_W)}, "wx_", FI_W)
    Wh = weather_features(X, fi, {"hel": ["helsinki", "tampere"]}, "wx_")
    W = pd.concat([W, Wh[["wx_hel_t2", "wx_hel_sw", "wx_hel_ws", "wx_hel_wchill", "wx_hel_t2_roll12", "wx_hel_t2_diff6"]]], axis=1)
    hr = X.t.dt.tz_convert("Europe/Helsinki").dt.hour
    W["wx_ws_x_hour"] = W.wx_fiw_ws * hr; W["wx_temp_x_hour"] = W.wx_hel_t2 * hr; W["wx_temp_x_consda"] = W.wx_hel_t2 * F.dl_cons_da / 1000
    W["wx_om_mw_minus_da"] = W.wx_fiw_pw * 9000 - F.dl_wind_da          # OM-implied MW (9 GW guess) vs Fingrid DA
    # weather-model bias now: OM pw for the last available actual-wind hour vs actual wind (as-of)
    Xn = pd.DataFrame({"o": X.o, "step": 1, "t": X.o - pd.Timedelta("1h")})
    Wn = weather_features(Xn, fi, {"fiw": list(FI_W)}, "now_", FI_W)
    wa_h = q.wind_act.rolling(4).mean(); wa_h.index = wa_h.index + Q + L15      # mean of last hour, usable +30m
    wa = asof(wa_h.to_frame("wa"))["wa"]
    W["wx_bias_now"] = wa.values - Wn.now_fiw_pw.values * 9000
    W["wx_om_mw_minus_da_corr"] = W.wx_om_mw_minus_da + W.wx_bias_now
    # temperature anomaly (vs trailing 7-day mean of forecast at lead<=6h) to remove seasonal drift
    t2 = fi[(fi.location.isin(["helsinki", "tampere"])) & (fi.lead_h < 12)].groupby("valid_time_utc").temperature_2m.mean().sort_index()
    t2m = t2.rolling("7D").mean(); t2m.index = t2m.index + pd.Timedelta("6h")
    W["wx_hel_t2_anom"] = W.wx_hel_t2 - asof(t2m.to_frame("m"))["m"].values
    X = pd.concat([X, Sx, Rx, F, W], axis=1)
    X["y_dir"] = q.dir.reindex(X.t).values; X["spread"] = q.spread.reindex(X.t).values; X["imb"] = q.imb.reindex(X.t).values; X["spot"] = q.spot.reindex(X.t).values
    X["split"] = np.select([X.o < pd.Timestamp(TRAIN[1], tz="UTC"), X.o < pd.Timestamp(VAL[1], tz="UTC")], ["train", "val"], "test")
    X.to_parquet(D / "feature_matrix_v2.parquet", index=False)
    return X

if __name__ == "__main__":
    X = build(); print(X.shape); print(X.isna().mean().sort_values(ascending=False).head(15).round(3))
