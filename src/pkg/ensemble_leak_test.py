"""Future-value mutation checks for LightGBM features and Chronos inputs/outputs."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import pkg_features as pf
from ensemble_spec import ROOT, WORK, OUT, SEEDS, KINDS, SELECTED, save_json
from fm_data import load_sources, make_input, assert_task_equal

CUTOFFS = pd.to_datetime(["2026-07-10T09:00Z", "2026-09-20T14:00Z", "2026-10-20T03:00Z"])


def mutate(v, fu, cutoff):
    rng = np.random.default_rng(0)
    changed = v.copy()
    for c in v:
        if c.startswith("cal_"):
            continue
        mask = fu[c].isna() | fu[c].gt(cutoff)
        changed.loc[mask, c] = rng.normal(0, 500, int(mask.sum()))
    return changed


def features():
    base = pf.build("B", save=False)
    feats = [c for c in base if c.startswith(("st_", "dl_", "wx_"))] + ["step"]
    cached = pd.read_parquet(ROOT / "work_private/features_B.parquet")
    np.testing.assert_array_equal(base[feats].to_numpy(dtype="float32"), cached[feats].to_numpy(dtype="float32"))
    original_loader = pf.load_external
    checks = []
    for cutoff in CUTOFFS:
        def changed_loader(v, fu):
            v, fu = original_loader(v, fu)
            return mutate(v, fu, cutoff), fu
        pf.load_external = changed_loader
        try:
            changed = pf.build("B", mutate_cutoff=cutoff, save=False)
        finally:
            pf.load_external = original_loader
        mask = base.o <= cutoff
        a, b = base.loc[mask, feats].to_numpy(), changed.loc[mask, feats].to_numpy()
        np.testing.assert_array_equal(a, b)
        later_changed = not np.array_equal(base.loc[~mask, feats].to_numpy(), changed.loc[~mask, feats].to_numpy(), equal_nan=True)
        assert later_changed
        # Preserve exact cutoff rows for inference checks with the fitted seed members.
        base[base.o == cutoff].to_parquet(WORK / f"leak_base_{cutoff.value}.parquet", index=False)
        changed[changed.o == cutoff].to_parquet(WORK / f"leak_mutated_{cutoff.value}.parquet", index=False)
        checks.append(dict(cutoff=cutoff, rows_checked=int(mask.sum()), bit_identical_features=True, mutation_visible_later=True))
        print("LightGBM feature mutation passed:", cutoff, flush=True)
    save_json(OUT / "feature_leak_checks.json", dict(rebuilt_matches_cached=True, checks=checks))


def lgb_outputs():
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT / "experiments"))
    from experiment_4_external_data import family_columns
    checks = []
    for cutoff, period in zip(CUTOFFS, ["fold0", "validation", "test"]):
        a = pd.read_parquet(WORK / f"leak_base_{cutoff.value}.parquet")
        b = pd.read_parquet(WORK / f"leak_mutated_{cutoff.value}.parquet")
        cols = [c for c in a if c.startswith(("st_", "dl_", "wx_"))] + ["step"]
        a[cols], b[cols] = a[cols].astype("float32"), b[cols].astype("float32")
        families = family_columns(a)
        tested = 0
        for seed in SEEDS:
            for kind in KINDS:
                for tag, selected, mask in [("global", SELECTED[str(kind)], np.ones(len(a), dtype=bool)),
                    ("1_4", "EX", a.step.between(1, 4)), ("5_8", "EX", a.step.between(5, 8))]:
                    model = lgb.Booster(model_file=str(WORK / f"models/{period}_{seed}_{kind}_{tag}.txt"))
                    pa = model.predict(a.loc[mask, families[selected]], num_threads=2)
                    pb = model.predict(b.loc[mask, families[selected]], num_threads=2)
                    np.testing.assert_array_equal(pa, pb)
                    tested += 1
        checks.append(dict(cutoff=cutoff, bit_identical_outputs=True, models_checked=tested))
    save_json(OUT / "lgb_output_leak_checks.json", checks)
    print("All 225 fitted LightGBM model mutation checks passed.", flush=True)


def chronos():
    from fm_run import load_pipeline, predict
    values, availability = load_sources()
    pipeline = load_pipeline(4)
    checks = []
    for cutoff in CUTOFFS[1:]:
        changed = mutate(values, availability, cutoff)
        # Every forecast origin in the evaluation span up through the cutoff.
        origins = pd.date_range("2026-07-05T21:00Z", cutoff, freq="h")
        for o in origins:
            a, sa, _ = make_input(values, availability, o)
            b, sb, _ = make_input(changed, availability, o)
            assert_task_equal(a, b)
            np.testing.assert_array_equal(sa, sb)
        # Same execution shape: exact output equality, not a numerical tolerance.
        a = make_input(values, availability, cutoff)[0]
        b = make_input(changed, availability, cutoff)[0]
        np.testing.assert_array_equal(predict(pipeline, [a]), predict(pipeline, [b]))
        checks.append(dict(cutoff=cutoff, origins_checked=len(origins), bit_identical_inputs=True,
            bit_identical_cutoff_output=True))
        print("Chronos input/output mutation passed:", cutoff, flush=True)
    save_json(OUT / "fm_leak_checks.json", checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("part", choices=["features", "lgb", "chronos"])
    args = p.parse_args()
    {"features": features, "lgb": lgb_outputs, "chronos": chronos}[args.part]()
