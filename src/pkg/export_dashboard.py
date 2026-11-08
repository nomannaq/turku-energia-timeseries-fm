"""Dashboard-friendly forecast files: one row per (origin, step) with quantiles, probabilities, top drivers (TreeSHAP of the
p50 model, top 3 with sign), and data freshness per source at the origin (age of the latest usable quarter-hour in minutes).
Writes submission/dashboard/{validation,test}_{A,B}.parquet + .csv. Contains NO raw spot prices (licence), only forecasts."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
import pkg_features as pf
ROOT = Path(__file__).resolve().parents[2]; SUB = ROOT / "submission"; OUT = SUB / "dashboard"; OUT.mkdir(exist_ok=True)
V = sys.argv[1] if len(sys.argv) > 1 else "B"
Vv, FU = pf.load_package()
if V == "B": Vv, FU = pf.load_external(Vv, FU)
FRESH = {"imbalance_price": "fi_imbalance_price", "mfrr_direction": "fi_mfrr_direction", "mfrr_activated": "fi_mfrr_up_activated", "consumption": "fi_consumption",
         "flows": "fi_se3_flow", "neighbour_imbalance": "se1_imbalance_price_long"}
if V == "B": FRESH.update({"calc_imbalance_rt": "x_calc_imb", "wind_actual": "x_wind_act", "mfrr_sa_price": "x_sa_price"})
def freshness(origins):
    out = pd.DataFrame({"forecast_origin_utc": origins})
    for name, s in FRESH.items():
        fu = FU[s].values.astype("datetime64[ns]"); ts = FU.index.values.astype("datetime64[ns]")
        has = ~np.isnan(Vv[s].values) if s in Vv else np.ones(len(ts), bool)
        last = []
        for o in origins.values.astype("datetime64[ns]"):
            m = (fu <= o) & has; last.append(ts[m].max() if m.any() else np.datetime64("NaT"))
        last = pd.to_datetime(np.array(last), utc=True)
        out[f"age_min_{name}"] = ((origins - (last + pd.Timedelta("15min"))) / pd.Timedelta("1min")).values
    spot_fu = FU.fi_spot_price; wind_fu = FU.fi_wind_fc_da
    out["spot_known_until_utc"] = [FU.index[(spot_fu <= o).values].max() + pd.Timedelta("15min") for o in origins]
    out["wind_da_known_until_utc"] = [FU.index[(wind_fu <= o).values].max() + pd.Timedelta("15min") for o in origins]
    if V == "B":
        w = pd.read_parquet(ROOT / "data/openmeteo_ecmwf_runs.parquet"); runs = np.sort(((w.run_time_utc + pd.Timedelta("6h")).dt.ceil("h")).unique())
        rr = pd.DatetimeIndex(runs); idx = rr.searchsorted(pd.DatetimeIndex(origins), side="right") - 1
        out["weather_run_utc"] = (rr[np.clip(idx, 0, None)] - pd.Timedelta("6h")).floor("6h")
    return out
# drivers: refit p50 global model per period with the run's selected features
exec_ns = {}
X = pd.read_parquet(ROOT / "work_private" / f"features_{V}.parquet"); X["target_fu"] = pd.to_datetime(X.target_fu)
if X.target_fu.dt.tz is None: X["target_fu"] = X.target_fu.dt.tz_localize("UTC")
run = json.load(open(SUB / f"run_{V}.json")); tb = json.load(open(ROOT / "results/v2_optuna_best.json"))["0.5"]["params"]
ladder = ["E1", "E2", "E3", "E4"] + (["EX", "EW"] if V == "B" else [])
import pkg_model_fams as fams
cols = fams.cols_upto(X, V, run["selected_rungs"]["0.5"])
def drivers(train_cut, a, b):
    i = np.where((X.target_fu <= train_cut) & (X.o < train_cut) & X.spread.notna())[0]; j = np.where((X.o >= a) & (X.o < b))[0]
    p = dict(tb); n = p.pop("n_rounds"); p.update(objective="quantile", alpha=0.5, verbose=-1, seed=7, num_threads=8, bagging_fraction=0.8, bagging_freq=1)
    m = lgb.train(p, lgb.Dataset(X.loc[i, cols].astype("float32"), X.spread.iloc[i]), n)
    c = m.predict(X.loc[j, cols].astype("float32"), pred_contrib=True)[:, :-1]; top = np.argsort(-np.abs(c), axis=1)[:, :3]
    import re; nm = np.array([re.sub(r"^(st|dl)_", "", x) for x in cols])
    d = [";".join(f"{nm[k]}({'+' if c[r, k] > 0 else '-'}{abs(c[r, k]):.1f})" for k in top[r]) for r in range(len(j))]
    return pd.DataFrame({"forecast_origin_utc": X.o.values[j], "step": X.step.values[j], "top_drivers_p50": d})
for per, fcst, cut, a, b in [("validation", f"validation_forecast_{V}.csv", "2026-09-06T21:00Z", "2026-09-06T21:00Z", "2026-10-04T21:00Z"),
                              ("test", f"forecast_backtest_{V}.csv", "2026-10-04T21:00Z", "2026-10-04T21:00Z", "2026-11-01T22:00Z")]:
    f = pd.read_csv(SUB / fcst); f["forecast_origin_utc"] = pd.to_datetime(f.forecast_origin_utc, utc=True); f["delivery_start_utc"] = pd.to_datetime(f.delivery_start_utc, utc=True)
    f = f.drop(columns=["top_drivers", "data_cutoff_utc"], errors="ignore")
    dr = drivers(pd.Timestamp(cut), pd.Timestamp(a), pd.Timestamp(b)); dr["forecast_origin_utc"] = pd.to_datetime(dr.forecast_origin_utc, utc=True)
    f = f.merge(dr, on=["forecast_origin_utc", "step"], how="left")
    fr = freshness(pd.Series(f.forecast_origin_utc.unique()))
    f = f.merge(fr, on="forecast_origin_utc", how="left")
    f["interval_width"] = f.p90 - f.p10; f["direction_mode"] = f[["p_up", "p_down", "p_none"]].idxmax(axis=1).str.replace("p_", "")
    f["variant"] = V; f["period"] = per
    f.to_parquet(OUT / f"{per}_{V}.parquet", index=False); f.to_csv(OUT / f"{per}_{V}.csv", index=False)
    print(per, V, f.shape)
