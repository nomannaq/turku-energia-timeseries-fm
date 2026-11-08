"""Train/evaluate LightGBM on the package feature matrix and write submission files.
Usage: python pkg_model.py A|B
- validation run: training rows need every target source usable at the first validation origin (2026-09-06T21:00Z)
- test run: training rows need every target source usable at the first test origin (2026-10-04T21:00Z), i.e. data up to 4 Oct only.
Feature families are chosen per output on rolling-origin folds inside the training data; hyperparameters come from the earlier
Optuna run on training folds only (results/v2_optuna_best.json); steps 1-4 and 5-8 get dedicated models; intervals get split-conformal
adjustments per horizon bucket (validation run: from training folds; test run: from the validation-run's out-of-sample validation forecasts)."""
import sys, json, shutil, subprocess, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; PKG = ROOT; SUB = ROOT / "submission"; W = ROOT / "work_private"
if len(sys.argv) != 2 or sys.argv[1] not in ("A", "B"):
    sys.exit("Usage: python src/pkg/pkg_model.py A|B")
V = sys.argv[1]
SUB.mkdir(parents=True, exist_ok=True)
VT = PKG / "baseline_output/validation/sample_submission.csv"
if not VT.exists():
    subprocess.run([sys.executable, str(PKG / "make_templates.py"), "--test-start", "2026-09-07",
                    "--test-end", "2026-10-04", "--output-dir", str(VT.parent)], check=True)
X = pd.read_parquet(W / f"features_{V}.parquet")
X["target_fu"] = pd.to_datetime(X.target_fu).dt.tz_localize("UTC") if pd.to_datetime(X.target_fu).dt.tz is None else pd.to_datetime(X.target_fu, utc=True)
X["y_dir3"] = X.y_dir.map({-1: 0, 0: 1, 1: 2}); X["y_spk"] = np.select([X.spread > 50, X.spread < -50], [2, 0], 1)
X.loc[X.spread.isna(), "y_spk"] = -1
FEAT = [c for c in X.columns if c.startswith(("st_", "dl_", "wx_"))] + ["step"]; X[FEAT] = X[FEAT].astype("float32")
CAL = ["step"] + [c for c in FEAT if c.startswith("dl_") and any(k in c for k in ["cal_", "qh", "hour_", "dow_", "weekend", "origin_hour"])]
def fam(c):
    if c in CAL or c.startswith(("dl_spot", "st_spot_past")) or c in ("st_sp_last", "st_sp_mean96", "st_sp_med96", "st_imb_last"): return "E1"
    if c.startswith(("st_sp_", "st_spike")): return "E2"
    if c.startswith(("dl_wind", "dl_cons", "dl_solar", "dl_resid", "st_cerr", "st_cons", "st_prod", "st_net_prod")): return "E3"
    if c.startswith(("st_werr", "st_wind_act", "st_ci_", "st_wrt", "st_nuc", "st_netimp", "st_freq", "st_kin", "st_sa_")): return "EX"
    if c.startswith("wx_"): return "EW"
    return "E4"
FAM = pd.Series({c: fam(c) for c in FEAT}); LADDER = ["E1", "E2", "E3", "E4"] + (["EX", "EW"] if V == "B" else [])
def upto(k): keep = LADDER[: LADDER.index(k) + 1]; return [c for c in FEAT if FAM[c] in keep]
tb = json.load(open(ROOT / "results/v2_optuna_best.json"))
P = {"dir": tb["dir"]["params"], "spk": tb["dir"]["params"], 0.1: tb["0.5"]["params"], 0.5: tb["0.5"]["params"], 0.9: tb["0.5"]["params"]}
SHORT = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.5, lambda_l2=10, n_rounds=300)
BASE = dict(bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7, num_threads=8)
KINDS = ["dir", "spk", 0.1, 0.5, 0.9]
def ylab(k): return X.y_dir3 if k == "dir" else X.y_spk if k == "spk" else X.spread
def fit(k, i, cols, params):
    p = {**BASE, **params}; n = p.pop("n_rounds", 300)
    p.update(dict(objective="multiclass", num_class=3) if k in ("dir", "spk") else dict(objective="quantile", alpha=k))
    y = ylab(k).iloc[i]; ok = y.notna() & (y >= 0) if k in ("dir", "spk") else y.notna()
    return lgb.train(p, lgb.Dataset(X.loc[i[ok.values], cols], y[ok].astype(float)), n)
def brier(p, y): return ((p - np.eye(3)[y]) ** 2).sum(1)
def pin(y, q, a): d = y - q; return np.maximum(a * d, (a - 1) * d)
def loss(k, pr, j):
    y = ylab(k).iloc[j].values; ok = ~np.isnan(y) & ((y >= 0) if k in ("dir", "spk") else True)
    return brier(pr[ok], y[ok].astype(int)) if k in ("dir", "spk") else pin(y[ok], pr[ok], k)
def tr_idx(cut): return np.where((X.target_fu <= cut) & (X.o < cut))[0]
def te_idx(a, b): return np.where((X.o >= a) & (X.o < b))[0]
OV, OT, OEND = pd.Timestamp("2026-09-06T21:00Z"), pd.Timestamp("2026-10-04T21:00Z"), pd.Timestamp("2026-11-01T22:00Z")
FOLDS = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in [("2026-07-05T21:00Z", "2026-07-19T21:00Z"), ("2026-07-19T21:00Z", "2026-08-02T21:00Z"),
         ("2026-08-02T21:00Z", "2026-08-16T21:00Z"), ("2026-08-16T21:00Z", "2026-08-30T21:00Z"), ("2026-08-30T21:00Z", "2026-09-06T21:00Z")]]
log = {}
# ---- 1. feature-family selection per output on rolling folds (fixed params)
sel = {}
for k in KINDS:
    best = None
    for rung in LADDER:
        cols = upto(rung); L = []
        for a, b in FOLDS:
            i, j = tr_idx(a), te_idx(a, b); j = j[(X.target_fu.iloc[j] <= OV).values]
            L.append(loss(k, fit(k, i, cols, P[k]).predict(X.loc[j, cols]), j))
        m = float(np.concatenate(L).mean()); log[f"{k}_{rung}"] = m
        if best is None or m < best[0] - 1e-9: best = (m, rung)
    sel[str(k)] = best[1]; print(k, {r: round(log[f"{k}_{r}"], 4) for r in LADDER}, "->", best[1], flush=True)
SCOLS = upto("EX" if V == "B" else "E4")
def predict(i, j):
    out = {k: fit(k, i, upto(sel[str(k)]), P[k]).predict(X.loc[j, upto(sel[str(k)])]) for k in KINDS}
    for lo, hi in [(1, 4), (5, 8)]:
        mi = i[(X.step.values[i] >= lo) & (X.step.values[i] <= hi)]; mj = (X.step.values[j] >= lo) & (X.step.values[j] <= hi)
        for k in KINDS: out[k][mj] = fit(k, mi, SCOLS, SHORT).predict(X.loc[j[mj], SCOLS])
    return out
BUCK = lambda st: np.digitize(st, [4.5, 16.5])
def post(p, j):
    q = np.sort(np.c_[p[0.1], p[0.5], p[0.9]], axis=1); d = p["dir"] / p["dir"].sum(1, keepdims=True); s = p["spk"].copy()
    s[:, 2] = np.minimum(s[:, 2], d[:, 2]); s[:, 0] = np.minimum(s[:, 0], d[:, 0]); return q, d, s
def scale(j): return np.clip(X.st_sp_std96.values[j], 5, None)
def calib(pairs):
    adj = {}
    for b in range(3):
        lo, hi = [], []
        for j, p in pairs:
            q, _, _ = post(p, j); y = X.spread.values[j]; m = (BUCK(X.step.values[j]) == b) & ~np.isnan(y); sc = scale(j)
            lo.append(((q[:, 0] - y) / sc)[m]); hi.append(((y - q[:, 2]) / sc)[m])
        lo, hi = np.concatenate(lo), np.concatenate(hi); lev = min(1, 0.9 * (1 + 1 / len(lo))); adj[b] = (float(np.quantile(lo, lev)), float(np.quantile(hi, lev)))
    return adj
def apply(q, j, adj):
    q = q.copy(); sc = scale(j); b = BUCK(X.step.values[j])
    for k, (al, ah) in adj.items(): m = b == k; q[m, 0] -= al * sc[m]; q[m, 2] += ah * sc[m]
    return np.sort(q, axis=1)
def to_frame(j, q, d, s):
    spot = X.dl_spot.values[j]; spot = np.where(np.isnan(spot), X.st_spot_past_mean96.values[j], spot)   # spot is known at every step; fallback never used in practice
    f = pd.DataFrame({"forecast_origin_utc": X.o.values[j], "delivery_start_utc": X.t.values[j], "step": X.step.values[j].astype(int),
                      "p10": spot + q[:, 0], "p50": spot + q[:, 1], "p90": spot + q[:, 2], "p_up": d[:, 2], "p_down": d[:, 0], "p_none": d[:, 1],
                      "p_spike_high": s[:, 2], "p_spike_low": s[:, 0]})
    f[["p_up", "p_down", "p_none"]] = f[["p_up", "p_down", "p_none"]].round(6); f["p_none"] = (1 - f.p_up - f.p_down).clip(lower=0).round(6)
    f["p_up"] = (1 - f.p_down - f.p_none).round(6)
    for c in ["p10", "p50", "p90"]: f[c] = f[c].round(4)
    f[["p_spike_high", "p_spike_low"]] = f[["p_spike_high", "p_spike_low"]].clip(0, 1).round(6)
    return f
def write(f, template, path, drivers=None):
    t = pd.read_csv(template)[["forecast_origin_utc", "step"]]; t["forecast_origin_utc"] = pd.to_datetime(t.forecast_origin_utc, utc=True)
    f = f.copy(); f["forecast_origin_utc"] = pd.to_datetime(f.forecast_origin_utc, utc=True)
    out = t.merge(f, on=["forecast_origin_utc", "step"], how="left"); assert out.p50.notna().all(), "missing forecasts"
    out["data_cutoff_utc"] = ""; out["top_drivers"] = drivers if drivers is not None else ""
    if drivers is not None: out["top_drivers"] = t.merge(f.assign(td=drivers)[["forecast_origin_utc", "step", "td"]], on=["forecast_origin_utc", "step"], how="left").td.fillna("")
    for c in ["forecast_origin_utc", "delivery_start_utc"]: out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    cols = pd.read_csv(template, nrows=0).columns.tolist(); out[cols].to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
# ---- 2. validation run
oof = []
for a, b in FOLDS:
    i, j = tr_idx(a), te_idx(a, b); j = j[(X.target_fu.iloc[j] <= OV).values]; oof.append((j, predict(i, j)))
ADJ_V = calib(oof)
iv, jv = tr_idx(OV), te_idx(OV, OT)
pv = predict(iv, jv); qv, dv, sv = post(pv, jv)
fv = to_frame(jv, apply(qv, jv, ADJ_V), dv, sv)
VT = PKG / "baseline_output/validation/sample_submission.csv"
write(fv, VT, SUB / f"validation_forecast_{V}.csv")
# ---- 3. test run (training data usable by 4 Oct 21:00Z; conformal from validation-run out-of-sample forecasts)
ADJ_T = calib([(jv[(X.target_fu.iloc[jv] <= OT).values], {k: v[(X.target_fu.iloc[jv] <= OT).values] for k, v in pv.items()})])
it, jt = tr_idx(OT), te_idx(OT, OEND)
pt = predict(it, jt); qt, dt, st = post(pt, jt); qt = apply(qt, jt, ADJ_T)
ft = to_frame(jt, qt, dt, st)
# top drivers (TreeSHAP of the p50 global model, top 3 by |contribution| per row)
k50 = upto(sel["0.5"]); m50 = fit(0.5, it, k50, P[0.5]); c = m50.predict(X.loc[jt, k50], pred_contrib=True)[:, :-1]
top = np.argsort(-np.abs(c), axis=1)[:, :3]; import re; names = np.array([re.sub(r"^(st|dl)_", "", n) for n in k50]); drv = [";".join(names[r]) for r in top]
name = "forecast_backtest.csv" if V == "B" else "forecast_backtest_packageonly.csv"
write(ft, PKG / "sample_submission.csv", SUB / f"forecast_backtest_{V}.csv", drivers=drv)
shutil.copyfile(SUB / f"forecast_backtest_{V}.csv", SUB / name)
shutil.copyfile(SUB / name, ROOT / name)
json.dump(dict(variant=V, selected_rungs=sel, fold_losses=log, conformal_val=ADJ_V, conformal_test=ADJ_T,
               n_train_val=int(len(iv)), n_train_test=int(len(it)), train_last_target_fu_test=str(X.target_fu.iloc[it].max())), open(SUB / f"run_{V}.json", "w"), indent=1, default=str)
print("done", V, "train rows val/test:", len(iv), len(it), "last target_fu in test training:", X.target_fu.iloc[it].max())
