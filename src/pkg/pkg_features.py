"""Feature builder on the OFFICIAL starter package. Binding rule: a value is used at origin o only if its
first_usable_origin_utc <= o. Missing package rows (sparse aFRR, rare gaps) get a rule-based first_usable from
availability.csv (lag after period end, rounded UP to the next full hour), which matches the row-level column on existing rows.
Variant A = package only. Variant B = package + external (documented conservative availability, see EXT_RULES).
Usage: python pkg_features.py [A|B]"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; PKG = ROOT; D = ROOT / "data"; OUT = ROOT / "work_private"; OUT.mkdir(parents=True, exist_ok=True)
Q = pd.Timedelta("15min")
IDX = pd.date_range("2026-05-31T21:00Z", "2026-11-02T06:45Z", freq="15min")
FIRST_ORIGIN, LAST_ORIGIN = pd.Timestamp("2026-06-01T21:00Z"), pd.Timestamp("2026-11-01T21:00Z")
LAG = {"fi_imbalance_price": 60, "fi_mfrr_up_price": 60, "fi_mfrr_down_price": 60, "fi_mfrr_up_activated": 60, "fi_mfrr_down_activated": 60, "fi_mfrr_direction": 60,
       "fi_afrr_up_price": 60, "fi_afrr_down_price": 60, "fi_consumption": 30, "fi_production": 30, "fi_se3_flow": 30, "fi_no4_flow": 30, "fi_se1_flow": 30, "fi_ee_flow": 30,
       **{f"{a}_imbalance_price_{b}": 120 for a in ["se1", "se3", "ee"] for b in ["long", "short"]}, "se1_load_fc_da": 120, "se3_load_fc_da": 120, "ee_load_fc_da": 120}
# External series (variant B): Fingrid realised data, availability = period end + 30 min, rounded up to the full hour (conservative;
# real-time 3-min data is published within minutes). Fingrid 400 is treated the same way (no use before the quarter ends).
EXT_RULES = {75: ("x_wind_act", 30), 397: ("x_calc_imb", 30), 181: ("x_wind_rt", 30), 188: ("x_nuc", 30), 194: ("x_netimp", 30), 177: ("x_freq", 30), 260: ("x_kin", 30), 400: ("x_sa_price", 30)}
def ceil_h(ts): return ts.ceil("h")

def load_package():
    d = pd.read_parquet(PKG / "data/inputs.parquet")
    s = pd.read_csv(PKG / "data/spot_nordpool.csv"); s["timestamp_utc"] = pd.to_datetime(s.timestamp_utc, utc=True); s["first_usable_origin_utc"] = pd.to_datetime(s.first_usable_origin_utc, utc=True)
    d = pd.concat([d, s], ignore_index=True)
    V = d.pivot(index="timestamp_utc", columns="series_id", values="value").reindex(IDX)
    FU = d.pivot(index="timestamp_utc", columns="series_id", values="first_usable_origin_utc").reindex(IDX)
    for c, L in LAG.items():   # rule-based availability for missing rows
        rule = pd.Series(ceil_h(IDX + Q + pd.Timedelta(minutes=L)), index=IDX)
        ok = FU[c].notna(); assert (FU.loc[ok, c] == rule[ok]).all(), c
        FU[c] = FU[c].fillna(rule)
    return V, FU

def load_external(V, FU):
    for i, (name, L) in EXT_RULES.items():
        x = pd.read_parquet(D / f"fingrid_{i}.parquet").drop_duplicates("startTime").set_index("startTime")["value"].astype(float).sort_index()
        if (x.index[1] - x.index[0]) < Q:
            g = x.resample("15min"); cnt = g.count(); x = g.mean(); x[cnt < 2] = np.nan; x = x.ffill(limit=1)
            if i == 397:
                gg = pd.read_parquet(D / "fingrid_397.parquet").set_index("startTime").value.resample("15min")
                V["x_calc_imb_min"] = gg.min().reindex(IDX); V["x_calc_imb_max"] = gg.max().reindex(IDX)
                for c in ["x_calc_imb_min", "x_calc_imb_max"]: FU[c] = ceil_h(IDX + Q + pd.Timedelta(minutes=L))
            if i == 177:
                V["x_freq_std"] = pd.read_parquet(D / "fingrid_177.parquet").set_index("startTime").value.resample("15min").std().reindex(IDX); FU["x_freq_std"] = ceil_h(IDX + Q + pd.Timedelta(minutes=L))
        V[name] = x.reindex(IDX); FU[name] = ceil_h(IDX + Q + pd.Timedelta(minutes=L))
    return V, FU

def roll(s, n, f): return getattr(s.rolling(n, min_periods=max(1, n // 2)), f)()

def state_groups(V, variant):
    """Returns list of (frame_of_features_indexed_by_t, list_of_source_series)."""
    G = []
    sp = V.fi_imbalance_price - V.fi_spot_price
    s = {"sp_last": sp}
    for k in [1, 2, 4, 8, 24, 96]: s[f"sp_lag{k}"] = sp.shift(k)
    for n in [4, 12, 24, 96]: s[f"sp_mean{n}"] = roll(sp, n, "mean"); s[f"sp_std{n}"] = roll(sp, n, "std"); s[f"sp_min{n}"] = roll(sp, n, "min"); s[f"sp_max{n}"] = roll(sp, n, "max")
    for n in [4, 16, 96]: s[f"sp_ewm{n}"] = sp.ewm(span=n).mean()
    s["sp_med96"] = roll(sp, 96, "median"); s["sp_absmed672"] = sp.abs().rolling(672, min_periods=96).median()
    s["spike_hi_96"] = roll((sp > 50).astype(float), 96, "mean"); s["spike_lo_96"] = roll((sp < -50).astype(float), 96, "mean")
    s["spike_any_672"] = (sp.abs() > 50).astype(float).rolling(672, min_periods=96).mean(); s["sp_std672"] = sp.rolling(672, min_periods=96).std()
    slog = lambda x: np.sign(x) * np.log1p(np.abs(x)); s["sp_slog_last"] = slog(sp); s["sp_slog_mean4"] = slog(roll(sp, 4, "mean")); s["sp_slog_lag96"] = slog(sp.shift(96))
    s["imb_last"] = V.fi_imbalance_price
    G.append(("spread", pd.DataFrame(s), ["fi_imbalance_price", "fi_spot_price"]))
    dr = V.fi_mfrr_direction; up = (dr == 1).astype(float).where(dr.notna())
    s = {"dir_last": dr, "dir_runlen": dr.groupby(dr.ne(dr.shift()).cumsum()).cumcount() + 1, "up_ewm8": up.ewm(span=8).mean()}
    for n in [4, 12, 24, 96]: s[f"up_share{n}"] = roll(up, n, "mean")
    for c in ["up", "down"]:
        a = V[f"fi_mfrr_{c}_activated"]; s[f"mfrr_{c}_act_last"] = a; s[f"mfrr_{c}_act_mean4"] = roll(a, 4, "mean"); s[f"mfrr_{c}_act_mean12"] = roll(a, 12, "mean")
    s["mfrr_net_act_mean4"] = s["mfrr_up_act_mean4"] - s["mfrr_down_act_mean4"]
    G.append(("regulation", pd.DataFrame(s), ["fi_mfrr_direction", "fi_mfrr_up_activated", "fi_mfrr_down_activated"]))
    s = {}
    for c in ["up", "down"]:
        p = V[f"fi_mfrr_{c}_price"] - V.fi_spot_price; s[f"mfrr_{c}_minus_spot"] = p; s[f"mfrr_{c}_minus_spot_mean4"] = roll(p, 4, "mean")
        a = V[f"fi_afrr_{c}_price"]; noact = a.isna().astype(float); v = a - V.fi_spot_price
        s[f"afrr_{c}_minus_spot"] = v; s[f"afrr_{c}_act_share16"] = 1 - roll(noact, 16, "mean")
        for k in [48, 96]: s[f"afrr_{c}_lag{k}"] = v.shift(k)
        for n in [48, 96]: s[f"afrr_{c}_mean{n}"] = v.rolling(n, min_periods=4).mean(); s[f"afrr_{c}_std{n}"] = v.rolling(n, min_periods=4).std()
        s[f"afrr_{c}_imp96"] = v.fillna(v.rolling(96, min_periods=1).mean())
    G.append(("prices", pd.DataFrame(s), ["fi_mfrr_up_price", "fi_mfrr_down_price", "fi_afrr_up_price", "fi_afrr_down_price", "fi_spot_price"]))
    ce = V.fi_consumption - V.fi_consumption_fc_da; net = V.fi_production - V.fi_consumption
    s = {"cerr_last": ce, "cerr_mean4": roll(ce, 4, "mean"), "cerr_mean12": roll(ce, 12, "mean"), "cerr_mean96": roll(ce, 96, "mean"),
         "cons_last": V.fi_consumption, "cons_lag96": V.fi_consumption.shift(96), "cons_mean96": roll(V.fi_consumption, 96, "mean"),
         "prod_last": V.fi_production, "net_prod_last": net, "net_prod_mean4": roll(net, 4, "mean"), "net_prod_d4": net - net.shift(4)}
    flows = ["fi_se3_flow", "fi_no4_flow", "fi_se1_flow", "fi_ee_flow"]; tot = V[flows].sum(axis=1, min_count=3)
    for f_ in flows: s[f"{f_}_last"] = V[f_]; s[f"{f_}_mean4"] = roll(V[f_], 4, "mean")
    s["flow_total_last"] = tot; s["flow_total_d4"] = tot - tot.shift(4); s["flow_total_mean24"] = roll(tot, 24, "mean")
    G.append(("system", pd.DataFrame(s), ["fi_consumption", "fi_production", "fi_consumption_fc_da"] + flows))
    s = {}
    for a in ["se1", "se3", "ee"]:
        p = V[f"{a}_imbalance_price_long"]; s[f"{a}_imb_last"] = p; s[f"{a}_imb_mean4"] = roll(p, 4, "mean"); s[f"fi_minus_{a}_imb"] = V.fi_imbalance_price - p
        s[f"{a}_long_minus_short"] = V[f"{a}_imbalance_price_long"] - V[f"{a}_imbalance_price_short"]
    s["se3_load_err_proxy"] = V.se3_load_fc_da.diff(4)
    G.append(("neighbours", pd.DataFrame(s), ["se1_imbalance_price_long", "se3_imbalance_price_long", "ee_imbalance_price_long", "se1_imbalance_price_short",
                                              "se3_imbalance_price_short", "ee_imbalance_price_short", "fi_imbalance_price", "se3_load_fc_da"]))
    sp_ = V.fi_spot_price
    G.append(("spot_past", pd.DataFrame({"spot_past_mean96": roll(sp_, 96, "mean"), "spot_past_mean192": roll(sp_, 192, "mean"), "spot_past_std48": roll(sp_, 48, "std"), "spot_past_std96": roll(sp_, 96, "std")}), ["fi_spot_price"]))
    if variant == "B":
        we = V.x_wind_act - V.fi_wind_fc_da
        s = {"werr_last": we, "werr_lag4": we.shift(4), "werr_lag8": we.shift(8), "werr_ewm8": we.ewm(span=8).mean(), "werr_rel": we / V.fi_wind_fc_da.clip(lower=200), "wind_act_last": V.x_wind_act}
        for n in [4, 12, 24, 96]: s[f"werr_mean{n}"] = roll(we, n, "mean")
        G.append(("x_wind_err", pd.DataFrame(s), ["x_wind_act", "fi_wind_fc_da"]))
        ci = V.x_calc_imb; wre = V.x_wind_rt - V.fi_wind_fc_da
        s = {"ci_last": ci, "ci_ewm4": ci.ewm(span=4).mean(), "ci_std4": roll(ci, 4, "std"), "ci_min_q": V.x_calc_imb_min, "ci_max_q": V.x_calc_imb_max,
             "wrt_err_last": wre, "wrt_err_mean4": roll(wre, 4, "mean"), "wrt_err_mean12": roll(wre, 12, "mean"), "wrt_err_d4": wre - wre.shift(4),
             "nuc_last": V.x_nuc, "nuc_drop24h": V.x_nuc - roll(V.x_nuc, 96, "max"), "nuc_d4": V.x_nuc - V.x_nuc.shift(4),
             "netimp_last": V.x_netimp, "netimp_mean4": roll(V.x_netimp, 4, "mean"), "netimp_d4": V.x_netimp - V.x_netimp.shift(4),
             "freq_dev4": roll(V.x_freq - 50, 4, "mean"), "freq_std_q": V.x_freq_std, "kin_last": V.x_kin,
             "sa_last_minus_spot": V.x_sa_price - V.fi_spot_price, "sa_mean4_minus_spot": roll(V.x_sa_price - V.fi_spot_price, 4, "mean")}
        for k in [1, 2, 4]: s[f"ci_lag{k}"] = ci.shift(k)
        for n in [4, 12, 24, 96]: s[f"ci_mean{n}"] = roll(ci, n, "mean")
        G.append(("x_realtime", pd.DataFrame(s), ["x_calc_imb", "x_calc_imb_min", "x_calc_imb_max", "x_wind_rt", "fi_wind_fc_da", "x_nuc", "x_netimp", "x_freq", "x_freq_std", "x_kin", "x_sa_price", "fi_spot_price"]))
    return G

def asof_group(F, srcs, FU, origins):
    A = FU[srcs].max(axis=1).ffill()
    A = pd.Series(np.maximum.accumulate(A.values.astype("datetime64[ns]").astype("int64")), index=A.index)   # availability of a row = max over its window (monotone)
    pos = np.searchsorted(A.values, origins.values.astype("datetime64[ns]").astype("int64"), side="right") - 1
    out = F.iloc[np.clip(pos, 0, None)].reset_index(drop=True); out[pos < 0] = np.nan
    return out

def build(variant="A", mutate_cutoff=None, save=True):
    V, FU = load_package()
    if variant == "B": V, FU = load_external(V, FU)
    if mutate_cutoff is not None:   # leakage test: scramble every value not yet usable at the cutoff origin
        rng = np.random.default_rng(0)
        for c in V.columns:
            if c.startswith("cal_"): continue
            m = (FU[c] > mutate_cutoff).values; V.loc[m, c] = rng.normal(0, 500, m.sum())
    origins = pd.Series(pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq="h"))
    X = pd.DataFrame({"o": np.repeat(origins.values, 40), "step": np.tile(np.arange(1, 41), len(origins))}); X["o"] = pd.to_datetime(X.o, utc=True)
    X["t"] = X.o + (X.step - 1) * Q
    parts = []
    for name, F, srcs in state_groups(V, variant):
        S = asof_group(F, srcs, FU, origins).add_prefix("st_"); S.index = origins.values
        parts.append(S.reindex(X.o.values).reset_index(drop=True))
    # delivery-time features: value at t used only if its own first_usable <= o
    def at_t(series, src=None):
        src = src or [series.name] if isinstance(series, pd.Series) else src
        v = series.reindex(X.t).values; fu = FU[src].max(axis=1).reindex(X.t).values
        return np.where(fu <= X.o.values, v, np.nan)
    F = pd.DataFrame(index=X.index)
    for c in ["cal_hour_local", "cal_weekday_local", "cal_is_holiday_fi", "cal_is_dst"]: F[c] = V[c].reindex(X.t).values
    mins = X.t.dt.tz_convert("Europe/Helsinki"); hf = mins.dt.hour + mins.dt.minute / 60
    F["qh"] = mins.dt.minute // 15; F["hour_sin"] = np.sin(2 * np.pi * hf / 24); F["hour_cos"] = np.cos(2 * np.pi * hf / 24)
    F["dow_sin"] = np.sin(2 * np.pi * F.cal_weekday_local / 7); F["dow_cos"] = np.cos(2 * np.pi * F.cal_weekday_local / 7); F["weekend"] = (F.cal_weekday_local >= 5).astype(float)
    F["origin_hour"] = X.o.dt.tz_convert("Europe/Helsinki").dt.hour; F["hour_x_step"] = F.cal_hour_local * 100 + X.step
    spot = V.fi_spot_price; day = spot.index.tz_convert("Europe/Helsinki").date; g = spot.groupby(day)
    def tf(name, ser, src): F[name] = np.where(FU[src].max(axis=1).reindex(X.t).values <= X.o.values, ser.reindex(X.t).values, np.nan)
    tf("spot", spot, ["fi_spot_price"]); tf("spot_rank_day", g.rank(pct=True), ["fi_spot_price"]); tf("spot_minus_daymean", spot - g.transform("mean"), ["fi_spot_price"])
    tf("spot_range_day", g.transform("max") - g.transform("min"), ["fi_spot_price"])
    for k in [1, 4]:
        tf(f"spot_ramp{k}", spot - spot.shift(k), ["fi_spot_price"])
        F[f"spot_fwd{k}"] = np.where(FU.fi_spot_price.shift(-k).reindex(X.t).values <= X.o.values, (spot.shift(-k) - spot).reindex(X.t).values, np.nan)
    so = spot.reindex(X.o).values; F["spot_minus_spot_o"] = F.spot - so
    res = V.fi_consumption_fc_da - V.fi_wind_fc_da - V.fi_solar_fc_da; das = ["fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da"]
    tf("wind_da", V.fi_wind_fc_da, ["fi_wind_fc_da"]); tf("cons_da", V.fi_consumption_fc_da, ["fi_consumption_fc_da"]); tf("solar_da", V.fi_solar_fc_da, ["fi_solar_fc_da"])
    tf("resid_da", res, das); tf("resid_rank_day", res.groupby(day).rank(pct=True), das)
    for k in [1, 4]:
        tf(f"resid_ramp{k}", res - res.shift(k), das); tf(f"wind_da_ramp{k}", V.fi_wind_fc_da - V.fi_wind_fc_da.shift(k), ["fi_wind_fc_da"])
    F["wind_da_minus_o"] = F.wind_da - V.fi_wind_fc_da.reindex(X.o).values; F["resid_minus_o"] = F.resid_da - res.reindex(X.o).values
    tf("fi_ee_cap", V.fi_ee_capacity_da, ["fi_ee_capacity_da"]); tf("ee_fi_cap", V.ee_fi_capacity_da, ["ee_fi_capacity_da"])

    parts.append(F.add_prefix("dl_"))
    if variant == "B":   # Open-Meteo ECMWF IFS single runs: usable from run + 6 h, rounded up to the full hour
        sys.path.insert(0, str(ROOT / "src")); import features_v2 as f2
        w = pd.read_parquet(D / "openmeteo_ecmwf_runs.parquet"); w["first_usable_utc"] = (w.run_time_utc + pd.Timedelta("6h")).dt.ceil("h")
        if mutate_cutoff is not None:
            m = w.first_usable_utc > mutate_cutoff; w.loc[m, ["wind_speed_100m", "temperature_2m", "shortwave_radiation", "wind_speed_10m", "wind_direction_100m"]] = 99.0
        Wx = f2.weather_features(X, w, {"fiw": list(f2.FI_W)}, "wx_", f2.FI_W)
        Wx["wx_om_mw_minus_da"] = Wx.wx_fiw_pw * 9000 - F.wind_da
        parts.append(Wx.drop(columns=[c for c in Wx.columns if c.endswith(("_t2", "_sw", "t2_lag3", "t2_lag6", "t2_roll6", "t2_roll12", "t2_diff6", "wchill"))]))
    X = pd.concat([X] + parts, axis=1)
    X["imb"] = V.fi_imbalance_price.reindex(X.t).values; X["spot_t"] = V.fi_spot_price.reindex(X.t).values; X["y_dir"] = V.fi_mfrr_direction.reindex(X.t).values
    X["target_fu"] = FU[["fi_imbalance_price", "fi_mfrr_direction", "fi_spot_price"]].max(axis=1).reindex(X.t).values
    X["spread"] = X.imb - X.spot_t
    if save: X.to_parquet(OUT / f"features_{variant}.parquet", index=False)
    return X

if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "A"; X = build(v); print(v, X.shape); print(X.isna().mean().sort_values(ascending=False).head(8).round(3).to_string())
