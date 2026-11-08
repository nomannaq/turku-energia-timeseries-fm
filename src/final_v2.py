"""Final v2: tuned params, OOF fold preds for conformal calibration, fit on full training, single evaluation on validation; SHAP via pred_contrib."""
import json, sys
from common_v2 import *
FSFILE = sys.argv[1] if len(sys.argv) > 1 else "v2_selected_features.json"; TAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
SEL = json.load(open(R / FSFILE)); tb = json.load(open(R / "v2_optuna_best.json"))
P = {"dir": tb["dir"]["params"], "spk": tb["dir"]["params"], 0.1: tb["0.5"]["params"], 0.5: tb["0.5"]["params"], 0.9: tb["0.5"]["params"]}
if json.load(open(R / "v2_use_tuned.json"))["use"] is False: P = {k: {} for k in P}
SHORT = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.5, lambda_l2=10, n_rounds=300)
HBUCKETS = [(1, 4), (5, 8)]          # dedicated short-horizon models (chosen on rolling folds); steps 9-40 use the global model
SCOLS = cols_upto("E4rt")
def predict_all(i, j):
    out = {k: fit(k, i, SEL[str(k)], P[k]).predict(X.loc[j, SEL[str(k)]]) for k in KINDS}
    for lo, hi in HBUCKETS:
        mi = i[(X.step.values[i] >= lo) & (X.step.values[i] <= hi)]; mj = (X.step.values[j] >= lo) & (X.step.values[j] <= hi)
        for k in KINDS: out[k][mj] = fit(k, mi, SCOLS, SHORT).predict(X.loc[j[mj], SCOLS])
    return out
# OOF on rolling folds (for calibration)
oof = []
for a, b in FOLDS:
    i, j = fold_idx(a, b); p = predict_all(i, j); oof.append((j, p))
def postprocess(p, j):
    q = np.sort(np.c_[p[0.1], p[0.5], p[0.9]], axis=1)                     # non-crossing
    d = p["dir"]; s = p["spk"].copy()
    s[:, 2] = np.minimum(s[:, 2], d[:, 2]); s[:, 0] = np.minimum(s[:, 0], d[:, 0]); s[:, 1] = 1 - s[:, 0] - s[:, 2]   # spike-high <= P(up), spike-low <= P(down)
    return q, d, s
BUCK = lambda st: np.digitize(st, [4.5, 16.5])
def calib(js, ps, scaled):
    """Split-conformal one-sided adjustments per step bucket (CQR); scaled: residuals / recent spread volatility."""
    adj = {}
    for bkt in range(3):
        lo, hi = [], []
        for j, p in zip(js, ps):
            q, _, _ = postprocess(p, j); y = X.spread.values[j]; m = BUCK(X.step.values[j]) == bkt
            sc = np.clip(X.st_sp_std96.values[j], 5, None) if scaled else np.ones(len(j))
            lo.append(((q[:, 0] - y) / sc)[m]); hi.append(((y - q[:, 2]) / sc)[m])
        lo, hi = np.concatenate(lo), np.concatenate(hi); n = len(lo); lev = min(1, 0.9 * (1 + 1 / n))
        adj[bkt] = (np.quantile(lo, lev), np.quantile(hi, lev))
    return adj
def apply(q, j, adj, scaled):
    q = q.copy(); sc = np.clip(X.st_sp_std96.values[j], 5, None) if scaled else np.ones(len(j)); b = BUCK(X.step.values[j])
    for bkt, (al, ah) in adj.items():
        m = b == bkt; q[m, 0] -= al * sc[m]; q[m, 2] += ah * sc[m]
    return np.sort(q, axis=1)
def metrics(q, d, s, j):
    y = X.spread.values[j]; s14 = X.step.values[j] <= 4
    bd = brier(d, X.y_dir3.values[j]); bs = brier(s, X.y_spk.values[j])
    return dict(dir_brier=bd.mean(), dir_brier_s1_4=bd[s14].mean(), spk_brier=bs.mean(), p50_mae=np.abs(y - q[:, 1]).mean(),
                pin10=pinball(y, q[:, 0], 0.1).mean(), pin90=pinball(y, q[:, 2], 0.9).mean(), cov80=((y >= q[:, 0]) & (y <= q[:, 2])).mean())
# choose conformal variant on training folds only: calibrate on folds 1-4, test on fold 5
choice = {}
for scaled in (False, True):
    adj = calib([o[0] for o in oof[:4]], [o[1] for o in oof[:4]], scaled); j, p = oof[4]; q, d, s = postprocess(p, j); qa = apply(q, j, adj, scaled)
    m = metrics(qa, d, s, j); choice[scaled] = m["pin10"] + m["pin90"] + abs(m["cov80"] - 0.8) * 20; print("conformal scaled" if scaled else "conformal abs", {k: round(v, 3) for k, v in m.items()})
SCALED = min(choice, key=choice.get); ADJ = calib([o[0] for o in oof], [o[1] for o in oof], SCALED)
print("chosen conformal:", "scaled" if SCALED else "absolute", ADJ)
# final fit on full training, evaluate once on validation
i, j = train_idx(), val_idx(); p = predict_all(i, j); q, d, s = postprocess(p, j)
res = {f"{TAG}_raw": metrics(q, d, s, j), f"{TAG}_conformal": metrics(apply(q, j, ADJ, SCALED), d, s, j)}
clim_d = np.bincount(X.y_dir3.values[i], minlength=3) / len(i); clim_s = np.bincount(X.y_spk.values[i], minlength=3) / len(i)
y = X.spread.values[j]; med = np.median(X.spread.values[i])
res["climatology"] = dict(dir_brier=brier(np.tile(clim_d, (len(j), 1)), X.y_dir3.values[j]).mean(), spk_brier=brier(np.tile(clim_s, (len(j), 1)), X.y_spk.values[j]).mean(),
                          p50_mae=np.abs(y - med).mean())
v1 = pd.read_csv(ROOT / "results_usefulness_windonly.csv").set_index("set").loc["base+rt+wx_fi_wind"]
res["v1_best (base+rt+wx_fi_wind)"] = dict(dir_brier=v1.dir_brier, dir_brier_s1_4=v1["dir_brier_s1-4"], spk_brier=v1.spk_brier, p50_mae=v1.p50_mae, pin10=v1["pinball_0.1"], pin90=v1["pinball_0.9"], cov80=v1.cov80)
T = pd.DataFrame(res).T; T.to_csv(R / ("v2_final_validation.csv".replace("v2", TAG))); print(T.round(4).to_string())
# day-block bootstrap on validation: v2 vs climatology
blocks = X.dday.values[j]
bb = {"dir vs climatology": block_boot(brier(d, X.y_dir3.values[j]) - brier(np.tile(clim_d, (len(j), 1)), X.y_dir3.values[j]), blocks),
      "spk vs climatology": block_boot(brier(s, X.y_spk.values[j]) - brier(np.tile(clim_s, (len(j), 1)), X.y_spk.values[j]), blocks),
      "p50 abs err vs train-median": block_boot(np.abs(y - q[:, 1]) - np.abs(y - med), blocks)}
print("bootstrap (mean, 2.5%, 97.5%, P(better)):", {k: tuple(round(x, 4) for x in v) for k, v in bb.items()})
json.dump(bb, open(R / ("v2_val_bootstrap.json".replace("v2", TAG)), "w"), indent=1)
# per-step-bucket table
pb = []
for lo_, hi_ in [(1, 4), (5, 16), (17, 40)]:
    m = (X.step.values[j] >= lo_) & (X.step.values[j] <= hi_); qa = apply(q, j, ADJ, SCALED)
    mm = metrics(qa[m], d[m], s[m], j[m]); mm["steps"] = f"{lo_}-{hi_}"; pb.append(mm)
pd.DataFrame(pb).to_csv(R / ("v2_val_by_horizon.csv".replace("v2", TAG)), index=False); print(pd.DataFrame(pb).round(4).to_string())
# SHAP (TreeSHAP via LightGBM pred_contrib) on 6000 validation rows
rng = np.random.default_rng(0); sj = rng.choice(j, 6000, replace=False); sh = {}
for k, cls in [("dir", 2), ("spk", 2), (0.5, None)]:
    cols = SEL[str(k)]; m = fit(k, i, cols, P[k]); c = m.predict(X.loc[sj, cols], pred_contrib=True)
    if cls is not None: c = c.reshape(len(sj), 3, len(cols) + 1)[:, cls, :]
    sh[str(k)] = pd.Series(np.abs(c[:, :-1]).mean(0), index=cols).sort_values(ascending=False)
mi = i[X.step.values[i] <= 4]; sj4 = sj[X.step.values[sj] <= 4]
for k, cls in [("dir", 2), (0.5, None)]:
    m = fit(k, mi, SCOLS, SHORT); c = m.predict(X.loc[sj4, SCOLS], pred_contrib=True)
    if cls is not None: c = c.reshape(len(sj4), 3, len(SCOLS) + 1)[:, cls, :]
    sh[f"{k}_steps1-4"] = pd.Series(np.abs(c[:, :-1]).mean(0), index=SCOLS).sort_values(ascending=False)
S = pd.concat({f"shap_{k}": v for k, v in sh.items()}, axis=1); S.to_csv(R / ("v2_shap_mean_abs.csv".replace("v2", TAG)))
for k, v in sh.items(): print(f"SHAP top {k}:", [(a, round(b, 3)) for a, b in v.head(12).items()])
