"""Sequential feature-family ladder E1..E5c with FIXED LightGBM params on rolling-origin folds (expanding, purged) inside training.
Day-block bootstrap of the loss change vs the previous rung."""
import time, json
from common_v2 import *
oof = {}; rows = []; t0 = time.time()
configs = [(k, cols_upto(k)) for k in LADDER] + [("E5a_no_rt", cols_upto("E5a", drop=("E4rt",))), ("E5a_t2anom_only", cols_upto("E5a") + ["wx_hel_t2_anom"])]
for name, cols in configs:
    L = {str(k): [] for k in KINDS}; J = []
    for a, b in FOLDS:
        i, j = fold_idx(a, b); J.append(j)
        for k in KINDS:
            m = fit(k, i, cols); L[str(k)].append(loss(k, m.predict(X.loc[j, cols]), j))
    J = np.concatenate(J); oof[name] = {k: np.concatenate(v) for k, v in L.items()}
    j14 = X.step.values[J] <= 4
    rows.append(dict(config=name, n_feat=len(cols), dir=oof[name]["dir"].mean(), dir_s1_4=oof[name]["dir"][j14].mean(), spk=oof[name]["spk"].mean(),
                     mae_p50=2 * oof[name]["0.5"].mean(), pin10=oof[name]["0.1"].mean(), pin90=oof[name]["0.9"].mean()))
    print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
A = pd.DataFrame(rows); blocks = X.dday.values[J]
bs = []
pairs = [(LADDER[i], LADDER[i - 1]) for i in range(1, len(LADDER))] + [("E5a_no_rt", "E5a"), ("E5a_t2anom_only", "E5a")]
for c, r in pairs:
    for k in ["dir", "spk", "0.5", "0.1", "0.9"]:
        m, lo, hi, pb = block_boot(oof[c][k] - oof[r][k], blocks); bs.append(dict(config=c, vs=r, target=k, mean_diff=m, ci_lo=lo, ci_hi=hi, p_better=pb))
B = pd.DataFrame(bs)
A.to_csv(R / "v2_ladder_rolling.csv", index=False); B.to_csv(R / "v2_ladder_bootstrap.csv", index=False)
pd.set_option("display.width", 200); print(A.round(4).to_string()); print(B.round(4).to_string())
