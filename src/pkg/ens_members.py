"""Five-seed LightGBM members, averaged before sorting/capping; no conformal step."""
import argparse
import importlib.metadata
import sys
import time

import lightgbm as lgb
import numpy as np

from ensemble_spec import (ROOT, WORK, OUT, SEEDS, KINDS, SELECTED, SHORT,
    PERIODS, params, load_frame, fingerprint, guard, save_json)

# Reuse the audited feature-family mapping without executing the production trainer.
sys.path.insert(0, str(ROOT / "experiments"))
from experiment_4_external_data import family_columns


def seed_params(seed):
    return dict(seed=seed, bagging_seed=seed+1, feature_fraction_seed=seed+2, data_random_seed=seed+3)


def fit(x, kind, indices, columns, settings, seed, threads):
    y = x["y_dir3" if kind == "dir" else "y_spk" if kind == "spk" else "spread"].iloc[indices]
    ok = y.notna() & (y >= 0) if isinstance(kind, str) else y.notna()
    p = dict(bagging_fraction=.8, bagging_freq=1, verbosity=-1, num_threads=threads, **seed_params(seed), **settings)
    rounds = p.pop("n_rounds")
    p.update(dict(objective="multiclass", num_class=3) if isinstance(kind, str)
             else dict(objective="quantile", alpha=kind))
    return lgb.train(p, lgb.Dataset(x.loc[indices[ok.to_numpy()], columns], y[ok].astype(float)), rounds)


def predict_member(x, i, j, seed, period, threads):
    families, settings = family_columns(x), params()
    predictions = {}
    for kind in KINDS:
        cols = families[SELECTED[str(kind)]]
        model = fit(x, kind, i, cols, settings[str(kind)], seed, threads)
        predictions[str(kind)] = model.predict(x.loc[j, cols], num_threads=threads)
        if period in ("fold0", "validation", "test"):
            model.save_model(str(WORK / f"models/{period}_{seed}_{kind}_global.txt"))
        cols = families["EX"]
        for lo, hi in [(1, 4), (5, 8)]:
            mi = i[x.step.iloc[i].between(lo, hi).to_numpy()]
            mask = x.step.iloc[j].between(lo, hi).to_numpy()
            model = fit(x, kind, mi, cols, SHORT, seed, threads)
            predictions[str(kind)][mask] = model.predict(x.loc[j[mask], cols], num_threads=threads)
            if period in ("fold0", "validation", "test"):
                model.save_model(str(WORK / f"models/{period}_{seed}_{kind}_{lo}_{hi}.txt"))
    return predictions


def main(args):
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    x = load_frame()
    protocol = dict(seeds=SEEDS, seed_offsets=[0, 1, 2, 3], selected=SELECTED, params=params(), short=SHORT,
        feature_sha256=fingerprint(ROOT / "work_private/features_B.parquet"), periods=PERIODS,
        lightgbm=importlib.metadata.version("lightgbm"), threads=args.threads, conformal=False)
    guard(WORK / "lgb_protocol.json", protocol)
    started, audit = time.monotonic(), []
    for period in args.periods.split(","):
        start, stop = PERIODS[period]
        i = np.flatnonzero((x.o < start) & (x.target_fu <= start))
        j = np.flatnonzero((x.o >= start) & (x.o < stop))
        assert (x.target_fu.iloc[i] <= start).all() and (x.o.iloc[i] < start).all()
        members = []
        for seed in SEEDS:
            cache = WORK / f"lgb_{period}_{seed}.npz"
            if cache.exists():
                with np.load(cache) as stored:
                    np.testing.assert_array_equal(stored["indices"], j)
                    pred = {str(k): stored[str(k)] for k in KINDS}
            else:
                pred = predict_member(x, i, j, seed, period, args.threads)
                assert all(np.isfinite(v).all() for v in pred.values())
                np.savez_compressed(cache, indices=j, **pred)
            members.append(pred)
            print(f"LightGBM {period} seed={seed} complete, {time.monotonic()-started:.0f}s", flush=True)
        mean = {str(k): np.mean([p[str(k)] for p in members], axis=0) for k in KINDS}
        np.savez_compressed(WORK / f"lgb_{period}.npz", indices=j, **mean)
        audit.append(dict(period=period, train_rows=len(i), prediction_rows=len(j), cutoff=start,
            latest_training_origin=x.o.iloc[i].max(), latest_training_target_usable=x.target_fu.iloc[i].max()))
        save_json(OUT / "lgb_training_audit.json", audit)
    save_json(OUT / "lgb_completed.json", dict(runtime_seconds=time.monotonic()-started, **protocol))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--periods", default=",".join(PERIODS))
    main(p.parse_args())
