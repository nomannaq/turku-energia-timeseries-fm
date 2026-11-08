"""Equal-budget, chronological LightGBM tuning; never writes submission files.

Run from the repository: .venv/bin/python experiments/experiment_5_parameter_tuning.py
Search and shortlist confirmation use only pre-validation labels. A separate
last training week calibrates intervals after all parameter choices are frozen.
"""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiment_4_external_data import (
    ROOT, OV, OT, BOUNDARIES, KINDS, SHORT, Experiment, fingerprint,
    compute_metrics, load_actuals,
)

OUT = ROOT / "results/experiment_5_parameter_tuning"
WORK = ROOT / "work_private/experiment5"
VARIANTS = ["baseline", "outages", "weather", "both"]
SEEDS = [7, 19]
N_CANDIDATES = 8
KEYS = ["forecast_origin_utc", "delivery_start_utc", "step"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def load_frame():
    x = pd.read_parquet(ROOT / "work_private/features_B.parquet")
    x = x[x.o < OT].copy()
    assert np.array_equal(x.index, np.arange(len(x)))
    x.target_fu = pd.to_datetime(x.target_fu, utc=True)
    x["y_dir3"] = x.y_dir.map({-1: 0, 0: 1, 1: 2})
    x["y_spk"] = np.select([x.spread > 50, x.spread < -50], [2, 0], 1)
    x.loc[x.spread.isna(), "y_spk"] = -1
    extras = {}
    for group in ["outage", "weather"]:
        f = pd.read_parquet(ROOT / f"work_private/experiment4/{group}_features.parquet")
        pd.testing.assert_index_equal(x.index, f.index)
        x = pd.concat([x, f], axis=1)
        extras[group] = list(f)
    cols = [c for c in x if c.startswith(("st_", "dl_", "wx_", "ex4_"))] + ["step"]
    x[cols] = x[cols].astype("float32")
    return x, {"baseline": [], "outages": extras["outage"],
               "weather": extras["weather"], "both": extras["outage"] + extras["weather"]}


def candidates(experiment):
    # Identical draws and budgets across all feature sets and output heads.
    rng = np.random.default_rng(20261107)
    draws = []
    for _ in range(N_CANDIDATES - 1):
        common = dict(learning_rate=float(np.exp(rng.uniform(np.log(.008), np.log(.06)))),
                      num_leaves=int(rng.choice([4, 7, 15, 23, 31])),
                      feature_fraction=float(rng.uniform(.3, .9)),
                      lambda_l2=float(np.exp(rng.uniform(np.log(1), np.log(80)))),
                      n_rounds=int(rng.choice([150, 250, 400, 600])))
        draws.append({"global": {**common, "min_data_in_leaf": int(rng.integers(200, 3001))},
                      "short": {**common, "min_data_in_leaf": int(rng.integers(60, 801))}})
    return {str(k): [{"global": experiment.params[k], "short": SHORT}, *draws] for k in KINDS}


def fold_indices(experiment, start, stop):
    i, j = experiment.train_idx(start), experiment.test_idx(start, stop)
    x = experiment.x
    j = j[(x.target_fu.iloc[j] <= stop).to_numpy()]
    assert len(i) and len(j)
    assert (x.o.iloc[i] < start).all() and (x.target_fu.iloc[i] <= start).all()
    assert (x.o.iloc[j] >= start).all() and (x.o.iloc[j] < stop).all()
    return i, j


def head_prediction(experiment, kind, i, j, config, seed):
    cols = experiment.cols[experiment.selected[str(kind)]] + experiment.extras
    params = {**config["global"], "seed": seed, "num_threads": 4}
    pred = experiment.fit(kind, i, cols, params).predict(experiment.x.loc[j, cols], num_threads=4)
    cols = experiment.cols["EX"] + experiment.extras
    for lo, hi in [(1, 4), (5, 8)]:
        mi = i[experiment.x.step.iloc[i].between(lo, hi).to_numpy()]
        mask = experiment.x.step.iloc[j].between(lo, hi).to_numpy()
        params = {**config["short"], "seed": seed, "num_threads": 4}
        pred[mask] = experiment.fit(kind, mi, cols, params).predict(experiment.x.loc[j[mask], cols], num_threads=4)
    assert np.isfinite(pred).all()
    return pred


def cached_prediction(experiment, variant, kind, config_id, config, seed, phase, i, j):
    path = WORK / f"{variant}_{kind}_{config_id}_{seed}_{phase}.npz"
    if path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["indices"], j)
            return saved["prediction"]
    pred = head_prediction(experiment, kind, i, j, config, seed)
    np.savez_compressed(path, indices=j, prediction=pred)
    return pred


def head_loss(x, kind, j, pred):
    if kind in ("dir", "spk"):
        y = x["y_dir3" if kind == "dir" else "y_spk"].to_numpy()[j]
        mask = np.isfinite(y) & (y >= 0)
        onehot = np.eye(3)[y[mask].astype(int)]
        squared = (pred[mask] - onehot) ** 2
        loss = squared.sum(1) if kind == "dir" else squared[:, [0, 2]].mean(1)
    else:
        y = x.spread.to_numpy()[j]
        mask = np.isfinite(y)
        residual = y[mask] - pred[mask]
        loss = np.abs(residual) if kind == .5 else np.maximum(kind * residual, (kind - 1) * residual)
    return float(loss.sum()), len(loss)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protected = ["forecast_backtest.csv", "forecast_backtest_packageonly.csv", "submission/validation_forecast_B.csv"]
    preserved = {p: fingerprint(ROOT / p) for p in protected}
    x, extras = load_frame()
    configs = candidates(Experiment(x, []))
    protocol = {"experiment": 5, "candidates_per_head_per_variant": N_CANDIDATES,
        "heads": [str(k) for k in KINDS], "variants": VARIANTS,
        "search_folds": [[str(BOUNDARIES[n]), str(BOUNDARIES[n+1])] for n in [1, 2]],
        "confirmation": [str(BOUNDARIES[3]), str(BOUNDARIES[4])],
        "calibration": [str(BOUNDARIES[4]), str(OV)], "validation": [str(OV), str(OT)],
        "confirmation_seeds": SEEDS, "final_fit_seed": 7,
        "selection": "Two best search candidates plus incumbent; minimum mean confirmation loss over both seeds. Feature-set nomination uses confirmation p50 MAE.",
        "objectives": {"0.5": "MAE", "0.1": "p10 pinball", "0.9": "p90 pinball", "dir": "multiclass Brier", "spk": "mean high/low Brier before direction capping"},
        "notes": ["All training targets must have target_fu <= fit cutoff; evaluation targets must be available by window end.",
                  "Search losses are for individual heads before quantile sorting, interval calibration and spike direction capping.",
                  "Candidate zero preserves inherited global and short-horizon settings. Other candidates tune both.",
                  "Official validation has been inspected in previous experiments and is not an independent final test.",
                  "Matched untuned controls use the same last-week calibration and seed as tuned models."],
        "configs": configs, "unchanged_artifact_sha256": preserved,
        "input_sha256": {p: fingerprint(ROOT / p) for p in ["work_private/features_B.parquet", "work_private/experiment4/outage_features.parquet", "work_private/experiment4/weather_features.parquet"]},
        "packages": {p: importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas"]}}
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol, "Changed protocol: use a new output directory."
    save(OUT / "protocol.json", protocol)
    records, selected, confirmation = [], {}, {}
    for variant in VARIANTS:
        e = Experiment(x, extras[variant])
        selected[variant], confirmation[variant] = {}, {}
        for kind in KINDS:
            scores = []
            for cid, config in enumerate(configs[str(kind)]):
                total, count = 0., 0
                for n in [1, 2]:
                    i, j = fold_indices(e, BOUNDARIES[n], BOUNDARIES[n+1])
                    pred = cached_prediction(e, variant, kind, cid, config, 7, f"search{n}", i, j)
                    loss, size = head_loss(x, kind, j, pred)
                    total += loss
                    count += size
                scores.append(total / count)
                records.append(dict(variant=variant, head=str(kind), phase="search", candidate=cid, seed=7, loss=total/count, rows=count))
                print(f"{variant} head={kind} search {cid+1}/{N_CANDIDATES}: {total/count:.6f} ({time.monotonic()-started:.0f}s)", flush=True)
                pd.DataFrame(records).to_csv(OUT / "trials.csv", index=False)
            shortlist = sorted(set([0, *np.argsort(scores)[:2].tolist()]))
            checked = {}
            i, j = fold_indices(e, BOUNDARIES[3], BOUNDARIES[4])
            for cid in shortlist:
                losses = []
                for seed in SEEDS:
                    pred = cached_prediction(e, variant, kind, cid, configs[str(kind)][cid], seed, "confirm", i, j)
                    loss, count = head_loss(x, kind, j, pred)
                    losses.append(loss/count)
                    records.append(dict(variant=variant, head=str(kind), phase="confirmation", candidate=cid, seed=seed, loss=loss/count, rows=count))
                checked[cid] = float(np.mean(losses))
            best = min(checked, key=checked.get)
            selected[variant][str(kind)] = int(best)
            confirmation[variant][str(kind)] = {"candidate": int(best), "mean_loss": checked[best], "incumbent_mean_loss": checked[0]}
            pd.DataFrame(records).to_csv(OUT / "trials.csv", index=False)
            print(f"{variant} head={kind}: selected {best}; confirmation {checked[best]:.6f} vs incumbent {checked[0]:.6f}", flush=True)
    nominated = min(VARIANTS, key=lambda v: confirmation[v]["0.5"]["mean_loss"])
    # All choices are persisted before any official validation scoring.
    save(OUT / "frozen_selection.json", {"selected_candidates": selected, "confirmation": confirmation,
        "nominated_feature_set": nominated, "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    summaries = []
    for variant in VARIANTS:
        e = Experiment(x, extras[variant])
        for mode in ["control", "tuned"]:
            ids = {str(k): 0 for k in KINDS} if mode == "control" else selected[variant]
            i, j = fold_indices(e, BOUNDARIES[4], OV)
            cal_pred = {k: cached_prediction(e, variant, k, ids[str(k)], configs[str(k)][ids[str(k)]], 7, "calibration", i, j) for k in KINDS}
            adjustments = e.calibrate([(j, cal_pred)])
            i, j = e.train_idx(OV), e.test_idx(OV, OT)
            pred = {k: cached_prediction(e, variant, k, ids[str(k)], configs[str(k)][ids[str(k)]], 7, "validation", i, j) for k in KINDS}
            f = e.to_frame(j, pred, adjustments)
            template = pd.read_csv(ROOT / "baseline_output/validation/sample_submission.csv")
            for c in KEYS[:2]:
                template[c] = pd.to_datetime(template[c], utc=True)
            pd.testing.assert_frame_equal(f[KEYS], template[KEYS], check_dtype=False)
            actuals = load_actuals(ROOT / "data/actuals_validation.csv")
            joined = f.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
            metrics = compute_metrics(joined, 50)
            name = f"{variant}_{mode}"
            summaries.append({"model": name, **metrics})
            exported = f.assign(data_cutoff_utc="", top_drivers="")
            for c in KEYS[:2]:
                exported[c] = pd.to_datetime(exported[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            exported.to_csv(OUT / f"validation_{name}.csv", index=False)
            save(OUT / f"calibration_{name}.json", adjustments)
            joined["horizon_hour"] = (joined.step - 1) // 4 + 1
            pd.DataFrame([{"horizon_hour": h, **compute_metrics(g, 50)} for h, g in joined.groupby("horizon_hour")]).to_csv(OUT / f"horizon_{name}.csv", index=False)
            pd.DataFrame(summaries).to_csv(OUT / "comparison.csv", index=False)
            print(name, json.dumps(metrics), flush=True)
    for p, digest in preserved.items():
        assert fingerprint(ROOT / p) == digest
    save(OUT / "completed.json", {"runtime_seconds": time.monotonic()-started, "submission_files_unchanged": True,
        "nominated_feature_set": nominated, "validation_used_for_parameter_selection": False})


if __name__ == "__main__":
    main()
