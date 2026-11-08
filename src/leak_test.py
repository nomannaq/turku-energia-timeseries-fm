"""Future-data mutation test: scramble all non-scheduled series from cutoff c onward; every feature for origins o <= c must be unchanged
(sa_price step-1 feature checked for o < c)."""
import numpy as np, pandas as pd, features_v2 as f
q0 = f.quarter_table(); rng = np.random.default_rng(1)
MUT = ["imb", "dir", "spread", "wind_act", "cons_act", "calc_imb", "wind_rt", "nuc", "netimp", "freq", "kin", "freq_std", "calc_imb_min", "calc_imb_max", "afrr_up", "afrr_dn", "afrr_missing", "afrr_up_noact", "afrr_dn_noact", "mfrr_up", "sa_price", "imb_SE1", "imb_SE3"]
bad = 0
for c in pd.to_datetime(["2026-07-15 10:00", "2026-08-20 03:00", "2026-09-30 17:00"], utc=True):
    q1 = q0.copy(); m = q1.index >= c
    for col in MUT: q1.loc[m, col] = rng.normal(0, 1000, m.sum())
    origins = pd.date_range(c - pd.Timedelta("2D"), c, freq="1h")
    for qq, tag in [(q0, 0), (q1, 1)]:
        S, R = f.state_features(qq); F = pd.concat([S, R], axis=1); F = F.groupby(level=0).last().sort_index()
        v = pd.merge_asof(pd.DataFrame({"o": origins}), F, left_on="o", right_index=True)
        if tag == 0: v0 = v
    diff = (v0.drop(columns="o").fillna(-9e9) != v.drop(columns="o").fillna(-9e9)).any()
    print(c, "changed features:", list(diff[diff].index)); bad += diff.sum()
print("LEAK TEST", "PASSED" if bad == 0 else "FAILED")
