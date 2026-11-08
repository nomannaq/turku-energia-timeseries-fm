"""Future-data mutation test on the package pipeline: every value whose first_usable_origin_utc (package or documented external rule)
is later than cutoff c is replaced with noise; all features for origins <= c must be identical. Targets are excluded."""
import numpy as np, pandas as pd, pkg_features as pf
ok = True
for variant in ["A", "B"]:
    base = pf.build(variant, save=False)
    feats = [c for c in base.columns if c.startswith(("st_", "dl_", "wx_"))]
    for c in pd.to_datetime(["2026-07-10T09:00Z", "2026-09-20T14:00Z", "2026-10-20T03:00Z"]):
        mut = pf.build(variant, mutate_cutoff=c, save=False)
        m = base.o <= c
        a = base.loc[m, feats].fillna(-9e9).values; b = mut.loc[m, feats].fillna(-9e9).values
        diff = ~np.isclose(a, b, rtol=0, atol=1e-9); bad = np.array(feats)[diff.any(0)]
        # sanity: the mutation must change something for later origins
        later = (~np.isclose(base.loc[~m, feats].fillna(-9e9).values, mut.loc[~m, feats].fillna(-9e9).values)).any()
        print(variant, c, "changed features at origins<=cutoff:", list(bad), "| mutation visible later:", bool(later)); ok &= len(bad) == 0 and later
print("PACKAGE LEAK TEST", "PASSED" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
