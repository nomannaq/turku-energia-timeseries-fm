"""Usefulness test: LightGBM direction / spike / spread-quantile models with nested feature sets, train -> validation.
Hyperparameters fixed a priori (no tuning on validation)."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from features import *
import warnings; warnings.filterwarnings('ignore')
q = quarter_table(); s, r = state_features(q)
origins = pd.date_range(TRAIN[0], TEST[1], freq="1h", tz="UTC", inclusive="left")
o = np.repeat(origins.values, 40); step = np.tile(np.arange(1, 41), len(origins))
X = pd.DataFrame({"o": pd.to_datetime(o, utc=True), "step": step}); X["t"] = X.o + (X.step - 1) * Q
loc = X.t.dt.tz_convert("Europe/Helsinki")
X["hour"] = loc.dt.hour; X["qh"] = loc.dt.minute // 15; X["dow"] = loc.dt.dayofweek
for c in ["spot", "wind_da", "cons_da"]: X[c] = q[c].reindex(X.t).values
X["resid_da"] = X.cons_da - X.wind_da
X["resid_da_ramp"] = (q.cons_da - q.wind_da).diff().reindex(X.t).values
X["spot_ramp"] = q.spot.diff().reindex(X.t).values
X["spot_minus_daymean"] = X.spot - q.spot.rolling(96, center=True, min_periods=1).mean().reindex(X.t).values
def asof(feat):
    f = feat.sort_index(); f = f[~f.index.duplicated(keep="last")]
    return pd.merge_asof(X[["o"]], f, left_on="o", right_index=True, direction="backward").drop(columns="o")
S = asof(s); R = asof(r)
R["sa_price_t_minus_spot"] = np.where(X.step == 1, (q.sa_price - q.spot).reindex(X.t).values, np.nan)  # published t-7.5min = o-7.5min
X["wind_da_now_diff"] = X.wind_da - q.wind_da.reindex(X.o).values
# Weather
fi = pd.read_parquet(D / "openmeteo_ecmwf_runs.parquet")
NBF = D / "openmeteo_ecmwf_runs_neighbours.parquet"; nb = pd.read_parquet(NBF if NBF.exists() else D / "openmeteo_ecmwf_runs.parquet")
W_FI = weather_features(X.o, X.t, fi, "om_", {"fiwind": FI_WIND, "hel": ["helsinki", "tampere"]})
W_NB = weather_features(X.o, X.t, nb, "om_", {"se1": ["se1_lulea", "se1_pitea"], "se3": ["se3_gavle", "se3_goteborg"], "ee": ["ee_parnu", "ee_hiiumaa"], "no": ["no_tromso", "no_finnmark"]})
for g in ["se1", "se3", "ee", "no"]:
    W_NB[f"fi_minus_{g}_pw"] = W_FI.om_fiwind_pw - W_NB[f"om_{g}_pw"]; W_NB[f"fi_minus_{g}_t2"] = W_FI.om_hel_t2 - W_NB[f"om_{g}_t2"]
W_FI["om_fiwind_pw_x_cap_minus_da"] = W_FI.om_fiwind_pw * 9000 - X.wind_da  # crude: OM-implied wind MW vs Fingrid DA forecast (9 GW nominal guess)
# FMI current obs vs OM forecast for "now" (forecast-error proxy)
fm = pd.read_parquet(D / "fmi_obs_10min.parquet"); fm = fm[fm.station.isin(["vaasa", "kaskinen", "pori", "raahe", "oulu", "kemi"])]
obs = fm.groupby("obs_time_utc").ws_10min.mean().resample("1h").mean()   # mean of 6 coastal stations, hourly
uo = pd.DatetimeIndex(X.o.drop_duplicates())
hr = (uo - LAGS["fmi"]).floor("h") - pd.Timedelta("1h")                       # last full hour whose obs are usable at o
fc = weather_features(pd.Series(uo), pd.Series(hr), fi, "now_", {"fi": FI_WIND})
tmp = pd.DataFrame({"o": uo, "fmi_ws10_last": obs.reindex(hr).values})
tmp["fmi_minus_om_ws10"] = tmp.fmi_ws10_last - fc["now_fi_ws10"].values      # current weather-forecast error proxy
W_FI_OBS = X[["o"]].merge(tmp, on="o", how="left").drop(columns="o")
X = pd.concat([X, S, R, W_FI, W_NB, W_FI_OBS], axis=1)
# targets
X["y_dir"] = q.dir.reindex(X.t).values; X["spread"] = q.spread.reindex(X.t).values; X["imb"] = q.imb.reindex(X.t).values
X = X[X.y_dir.notna() & X.spread.notna()]
X["y_spk"] = np.select([X.spread > 50, X.spread < -50], [2, 0], 1)
X["y_dir3"] = X.y_dir.map({-1: 0, 0: 1, 1: 2}).astype(int)
tr = (X.o >= TRAIN[0]) & (X.t < TRAIN[1]); va = (X.o >= VAL[0]) & (X.t < VAL[1])
X.to_parquet(D / "feature_matrix.parquet", index=False)

CAL = ["step", "hour", "qh", "dow"]
BASE = CAL + ["spot", "wind_da", "cons_da", "resid_da", "resid_da_ramp", "spot_ramp", "spot_minus_daymean", "wind_da_now_diff"] + list(S.columns)
RT = list(R.columns); WFI = list(W_FI.columns) + list(W_FI_OBS.columns); WNB = list(W_NB.columns)
WFI_W = [c for c in WFI if not c.endswith(("_t2", "_sw"))]; WNB_W = [c for c in WNB if not c.endswith(("_t2", "_sw"))]
SETS = {"base+wx_fi_wind": BASE + WFI_W, "base+wx_fi_wind+nb_wind": BASE + WFI_W + WNB_W, "base+rt+wx_fi_wind": BASE + RT + WFI_W, "base+rt+wx_fi_wind+nb_wind": BASE + RT + WFI_W + WNB_W,
        "base": BASE, "base+wx_fi": BASE + WFI, "base+wx_fi+nb": BASE + WFI + WNB, "base+rt": BASE + RT, "base+rt+wx_fi": BASE + RT + WFI, "all": BASE + RT + WFI + WNB}
P = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
def brier(p, y, k=3): Y = np.eye(k)[y]; return float(((p - Y) ** 2).sum(1).mean())
def pinball(y, qhat, a): d = y - qhat; return float(np.mean(np.maximum(a * d, (a - 1) * d)))
res, imp, preds = [], {}, {}
Xtr, Xva = X[tr], X[va]
clim_dir = np.bincount(Xtr.y_dir3, minlength=3) / len(Xtr); clim_spk = np.bincount(Xtr.y_spk, minlength=3) / len(Xtr)
res.append(dict(set="climatology", dir_brier=brier(np.tile(clim_dir, (len(Xva), 1)), Xva.y_dir3.values), spk_brier=brier(np.tile(clim_spk, (len(Xva), 1)), Xva.y_spk.values),
                p50_mae=float(np.abs(Xva.spread - Xtr.spread.median()).mean()), spot_only_mae=float(np.abs(Xva.spread).mean())))
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SETS)
for name, cols in SETS.items():
    if name not in ONLY: continue
    row = dict(set=name, n_feat=len(cols))
    m = lgb.train({**P, "objective": "multiclass", "num_class": 3}, lgb.Dataset(Xtr[cols], Xtr.y_dir3), 400)
    pd_ = m.predict(Xva[cols]); preds[name] = (((pd_ - np.eye(3)[Xva.y_dir3.values]) ** 2).sum(1), None); row["dir_brier"] = brier(pd_, Xva.y_dir3.values)
    for lo, hi in [(1, 4), (5, 16), (17, 40)]:
        k = Xva.step.between(lo, hi).values; row[f"dir_brier_s{lo}-{hi}"] = brier(pd_[k], Xva.y_dir3.values[k])
    imp[name] = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False).head(15).round(0).to_dict()
    m2 = lgb.train({**P, "objective": "multiclass", "num_class": 3}, lgb.Dataset(Xtr[cols], Xtr.y_spk), 400)
    ps = m2.predict(Xva[cols]); preds[name] = (preds[name][0], ((ps - np.eye(3)[Xva.y_spk.values]) ** 2).sum(1)); row["spk_brier"] = brier(ps, Xva.y_spk.values)
    k = Xva.step.between(1, 4).values; row["spk_brier_s1-4"] = brier(ps[k], Xva.y_spk.values[k])
    for a in (0.1, 0.5, 0.9):
        mq = lgb.train({**P, "objective": "quantile", "alpha": a}, lgb.Dataset(Xtr[cols], Xtr.spread), 400)
        qh = mq.predict(Xva[cols])
        if a == 0.5: row["p50_mae"] = float(np.abs(Xva.spread - qh).mean()); row["p50_mae_s1-4"] = float(np.abs(Xva.spread - qh)[k].mean())
        else: row[f"pinball_{a}"] = pinball(Xva.spread.values, qh, a); row[f"q{a}"] = qh
    row["cov80"] = float(((Xva.spread >= row.pop("q0.1")) & (Xva.spread <= row.pop("q0.9"))).mean())
    res.append(row); print(name, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
out = pd.DataFrame(res); out.to_csv(D.parent / ("results_usefulness.csv" if len(sys.argv) == 1 else "results_usefulness_windonly.csv"), index=False)
json.dump(imp, open(D.parent / ("results_importance.json" if len(sys.argv) == 1 else "results_importance_windonly.json"), "w"), indent=1)
print(out.round(4).to_string()); print("train rows", tr.sum(), "val rows", va.sum())

# paired block bootstrap over validation days (by origin day): P(set improves on base)
day = Xva.o.dt.tz_convert("Europe/Helsinki").dt.date.values; days = np.unique(day); rng = np.random.default_rng(0)
B = [np.isin(day, rng.choice(days, len(days))) for _ in range(0)]
print("\nBootstrap (500 day-block resamples) of Brier difference vs base (negative = better):")
dl = pd.Series(range(len(day))).groupby(day).apply(list)
PAIRS = [("base+wx_fi", "base"), ("base+wx_fi+nb", "base+wx_fi"), ("base+rt", "base"), ("all", "base+rt"), ("all", "base"),
         ("base+wx_fi_wind", "base"), ("base+wx_fi_wind+nb_wind", "base+wx_fi_wind"), ("base+rt+wx_fi_wind", "base+rt"), ("base+rt+wx_fi_wind+nb_wind", "base+rt+wx_fi_wind")]
for comp, ref in [p_ for p_ in PAIRS if p_[0] in preds and p_[1] in preds]:
    for j, lab in [(0, "dir"), (1, "spk")]:
        d = preds[comp][j] - preds[ref][j]; dm = pd.Series(d).groupby(day).agg(["sum", "size"])
        bs = []
        for _ in range(500):
            pick = dm.iloc[rng.integers(0, len(dm), len(dm))]; bs.append(pick["sum"].sum() / pick["size"].sum())
        bs = np.array(bs); print(f"{comp} vs {ref} [{lab}]: mean {d.mean():+.4f}  95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  P(better)={np.mean(bs<0):.2f}")
