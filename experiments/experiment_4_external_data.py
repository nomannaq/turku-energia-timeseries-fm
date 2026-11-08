"""Controlled LightGBM ablations: baseline, outage notices, weather, and both.

Uses the submission's fixed feature families, model settings, short-horizon
specialists and training-fold calibration. No test fitting or submission writes.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score import compute_metrics, load_actuals

WORK = ROOT / "work_private/experiment4"
OUT = ROOT / "results/experiment_4_external_data"
OV, OT = pd.Timestamp("2026-09-06T21:00Z"), pd.Timestamp("2026-10-04T21:00Z")
BOUNDARIES = pd.to_datetime(["2026-07-05T21:00Z", "2026-07-19T21:00Z", "2026-08-02T21:00Z",
                             "2026-08-16T21:00Z", "2026-08-30T21:00Z", "2026-09-06T21:00Z"])
KINDS = ["dir", "spk", 0.1, 0.5, 0.9]
SHORT = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=300,
             feature_fraction=0.5, lambda_l2=10, n_rounds=300)
BASE = dict(bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7, num_threads=8)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_columns(frame):
    feat = [c for c in frame if c.startswith(("st_", "dl_", "wx_"))] + ["step"]
    cal = ["step"] + [c for c in feat if c.startswith("dl_") and
                      any(k in c for k in ["cal_", "qh", "hour_", "dow_", "weekend", "origin_hour"])]
    def family(c):
        if c in cal or c.startswith(("dl_spot", "st_spot_past")) or c in ("st_sp_last", "st_sp_mean96", "st_sp_med96", "st_imb_last"):
            return "E1"
        if c.startswith(("st_sp_", "st_spike")):
            return "E2"
        if c.startswith(("dl_wind", "dl_cons", "dl_solar", "dl_resid", "st_cerr", "st_cons", "st_prod", "st_net_prod")):
            return "E3"
        if c.startswith(("st_werr", "st_wind_act", "st_ci_", "st_wrt", "st_nuc", "st_netimp", "st_freq", "st_kin", "st_sa_")):
            return "EX"
        return "EW" if c.startswith("wx_") else "E4"
    ladder = ["E1", "E2", "E3", "E4", "EX", "EW"]
    return {rung: [c for c in feat if ladder.index(family(c)) <= ladder.index(rung)] for rung in ladder}


class Experiment:
    def __init__(self, frame, extras):
        self.x = frame
        self.cols = family_columns(frame)
        self.selected = json.loads((ROOT / "submission/run_B.json").read_text())["selected_rungs"]
        best = json.loads((ROOT / "results/v2_optuna_best.json").read_text())
        self.params = {k: best["dir" if isinstance(k, str) else "0.5"]["params"] for k in KINDS}
        self.extras = extras

    def train_idx(self, cutoff):
        return np.flatnonzero((self.x.target_fu <= cutoff) & (self.x.o < cutoff))

    def test_idx(self, start, stop):
        return np.flatnonzero((self.x.o >= start) & (self.x.o < stop))

    def fit(self, kind, indices, columns, params):
        y = self.x["y_dir3" if kind == "dir" else "y_spk" if kind == "spk" else "spread"].iloc[indices]
        ok = y.notna() & (y >= 0) if isinstance(kind, str) else y.notna()
        p = {**BASE, **params}
        rounds = p.pop("n_rounds")
        p.update(dict(objective="multiclass", num_class=3) if isinstance(kind, str)
                 else dict(objective="quantile", alpha=kind))
        return lgb.train(p, lgb.Dataset(self.x.loc[indices[ok.to_numpy()], columns], y[ok].astype(float)), rounds)

    def predict(self, i, j):
        predictions = {}
        for kind in KINDS:
            columns = self.cols[self.selected[str(kind)]] + self.extras
            predictions[kind] = self.fit(kind, i, columns, self.params[kind]).predict(self.x.loc[j, columns])
        columns = self.cols["EX"] + self.extras
        for lo, hi in [(1, 4), (5, 8)]:
            mi = i[self.x.step.iloc[i].between(lo, hi).to_numpy()]
            mj = self.x.step.iloc[j].between(lo, hi).to_numpy()
            for kind in KINDS:
                predictions[kind][mj] = self.fit(kind, mi, columns, SHORT).predict(self.x.loc[j[mj], columns])
        return predictions

    def post(self, predictions):
        q = np.sort(np.c_[predictions[0.1], predictions[0.5], predictions[0.9]], axis=1)
        d = predictions["dir"] / predictions["dir"].sum(1, keepdims=True)
        s = predictions["spk"].copy()
        s[:, 2] = np.minimum(s[:, 2], d[:, 2])
        s[:, 0] = np.minimum(s[:, 0], d[:, 0])
        return q, d, s

    def calibrate(self, pairs):
        adjustments = {}
        for bucket in range(3):
            lower, upper = [], []
            for j, predictions in pairs:
                q, _, _ = self.post(predictions)
                y = self.x.spread.to_numpy()[j]
                mask = (np.digitize(self.x.step.to_numpy()[j], [4.5, 16.5]) == bucket) & ~np.isnan(y)
                scale = np.clip(self.x.st_sp_std96.to_numpy()[j], 5, None)
                lower.extend(((q[:, 0] - y) / scale)[mask])
                upper.extend(((y - q[:, 2]) / scale)[mask])
            level = min(1, 0.9 * (1 + 1 / len(lower)))
            adjustments[bucket] = [float(np.quantile(lower, level)), float(np.quantile(upper, level))]
        return adjustments

    def to_frame(self, j, predictions, adjustments=None):
        q, d, s = self.post(predictions)
        if adjustments is not None:
            scale = np.clip(self.x.st_sp_std96.to_numpy()[j], 5, None)
            buckets = np.digitize(self.x.step.to_numpy()[j], [4.5, 16.5])
            for bucket, (lower, upper) in adjustments.items():
                mask = buckets == bucket
                q[mask, 0] -= lower * scale[mask]
                q[mask, 2] += upper * scale[mask]
            q = np.sort(q, axis=1)
        spot = self.x.dl_spot.to_numpy()[j]
        spot = np.where(np.isnan(spot), self.x.st_spot_past_mean96.to_numpy()[j], spot)
        f = pd.DataFrame({"forecast_origin_utc": self.x.o.to_numpy()[j], "delivery_start_utc": self.x.t.to_numpy()[j],
            "step": self.x.step.to_numpy()[j].astype(int), "p10": spot + q[:, 0], "p50": spot + q[:, 1], "p90": spot + q[:, 2],
            "p_up": d[:, 2], "p_down": d[:, 0], "p_none": d[:, 1], "p_spike_high": s[:, 2], "p_spike_low": s[:, 0]})
        f[["p_up", "p_down", "p_none"]] = f[["p_up", "p_down", "p_none"]].round(6)
        f["p_none"] = (1 - f.p_up - f.p_down).clip(lower=0).round(6)
        f["p_up"] = (1 - f.p_down - f.p_none).round(6)
        f[["p10", "p50", "p90"]] = f[["p10", "p50", "p90"]].round(4)
        f[["p_spike_high", "p_spike_low"]] = f[["p_spike_high", "p_spike_low"]].clip(0, 1).round(6)
        return f


def main(args):
    OUT.mkdir(exist_ok=True, parents=True)
    preserved = {p: fingerprint(ROOT / p) for p in ["forecast_backtest.csv", "forecast_backtest_packageonly.csv", "submission/validation_forecast_B.csv"]}
    x = pd.read_parquet(ROOT / "work_private/features_B.parquet")
    x = x[x.o < OT].copy()
    x.target_fu = pd.to_datetime(x.target_fu, utc=True)
    x["y_dir3"] = x.y_dir.map({-1: 0, 0: 1, 1: 2})
    x["y_spk"] = np.select([x.spread > 50, x.spread < -50], [2, 0], 1)
    x.loc[x.spread.isna(), "y_spk"] = -1
    extras = {}
    for group in ["outage", "weather"]:
        path = WORK / f"{group}_features.parquet"
        if path.exists():
            features = pd.read_parquet(path)
            pd.testing.assert_index_equal(x.index, features.index)
            x = pd.concat([x, features], axis=1)
            extras[group] = list(features)
    feat = [c for c in x if c.startswith(("st_", "dl_", "wx_", "ex4_"))] + ["step"]
    x[feat] = x[feat].astype("float32")
    actuals = load_actuals(ROOT / "data/actuals_validation.csv")
    summaries, cv_summaries, metadata = [], [], {}
    if args.append:
        summaries = pd.read_csv(OUT / "comparison.csv").to_dict("records")
        cv_summaries = pd.read_csv(OUT / "training_fold_comparison.csv").to_dict("records")
        metadata = json.loads((OUT / "run.json").read_text())["variants"]
    else:
        reference = pd.read_csv(ROOT / "submission/validation_forecast_B.csv")
        reference.delivery_start_utc = pd.to_datetime(reference.delivery_start_utc, utc=True)
        summaries.append({"model": "submitted_B", **compute_metrics(reference.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one"), 50)})
    for variant in args.variants.split(","):
        added = []
        if variant in ("outages", "both"):
            added += extras["outage"]
        if variant in ("weather", "both"):
            added += extras["weather"]
        assert variant in ("baseline", "outages", "weather", "both")
        experiment = Experiment(x, added)
        started = time.monotonic()
        pairs, oof_frames = [], []
        for start, stop in zip(BOUNDARIES[:-1], BOUNDARIES[1:]):
            i, j = experiment.train_idx(start), experiment.test_idx(start, stop)
            j = j[(x.target_fu.iloc[j] <= OV).to_numpy()]
            predictions = experiment.predict(i, j)
            pairs.append((j, predictions))
            f = experiment.to_frame(j, predictions)
            f["imbalance_price"] = x.imb.to_numpy()[j]
            f["spot_price"] = x.spot_t.to_numpy()[j]
            f["direction"] = x.y_dir.iloc[j].map({-1: "down", 0: "none", 1: "up"}).to_numpy()
            oof_frames.append(f)
            print(f"{variant}: fold {start.date()}–{stop.date()}, {time.monotonic()-started:.1f}s", flush=True)
        oof = pd.concat(oof_frames, ignore_index=True)
        cv_summaries.append({"model": variant, **compute_metrics(oof, 50)})
        oof.to_parquet(WORK / f"oof_{variant}.parquet", index=False)
        adjustments = experiment.calibrate(pairs)
        i, j = experiment.train_idx(OV), experiment.test_idx(OV, OT)
        prediction = experiment.predict(i, j)
        f = experiment.to_frame(j, prediction, adjustments)
        keys = ["forecast_origin_utc", "delivery_start_utc", "step"]
        template = pd.read_csv(ROOT / "baseline_output/validation/sample_submission.csv")
        for c in keys[:2]:
            template[c] = pd.to_datetime(template[c], utc=True)
        pd.testing.assert_frame_equal(f[keys], template[keys], check_dtype=False)
        if variant == "baseline":
            reference = pd.read_csv(ROOT / "submission/validation_forecast_B.csv")
            values = [c for c in f if c.startswith("p")]
            difference = f[values].to_numpy() - reference[values].to_numpy()
            (OUT / "control_reproduction.json").write_text(json.dumps({
                "max_abs_difference_by_column": dict(zip(values, np.max(np.abs(difference), axis=0).tolist())),
                "mean_abs_difference_by_column": dict(zip(values, np.mean(np.abs(difference), axis=0).tolist())),
                "interpretation": "Use the freshly retrained baseline for paired ablations. Saved imported submission is a separate reference; exact production-function parity was checked on the short-horizon direction model.",
            }, indent=2))
        joined = f.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
        metrics = compute_metrics(joined, 50)
        summaries.append({"model": variant, **metrics})
        exported = f.assign(data_cutoff_utc="", top_drivers="")
        for column in keys[:2]:
            exported[column] = pd.to_datetime(exported[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        exported.to_csv(OUT / f"validation_{variant}.csv", index=False)
        joined["horizon_hour"] = (joined.step - 1) // 4 + 1
        pd.DataFrame([{"horizon_hour": h, **compute_metrics(g, 50)} for h, g in joined.groupby("horizon_hour")]).to_csv(OUT / f"horizon_{variant}.csv", index=False)
        metadata[variant] = {"added_features": added, "calibration": adjustments, "train_rows": len(i),
                             "validation_rows": len(j), "runtime_seconds": time.monotonic() - started}
        pd.DataFrame(summaries).drop_duplicates("model", keep="last").to_csv(OUT / "comparison.csv", index=False)
        pd.DataFrame(cv_summaries).drop_duplicates("model", keep="last").to_csv(OUT / "training_fold_comparison.csv", index=False)
        (OUT / "run.json").write_text(json.dumps({"experiment": 4, "variants": metadata,
            "selection": "Compare pre-validation rolling-fold MAE; validation is reported separately, never used for fitting/calibration.",
            "training_cutoff": str(OV), "validation_end_exclusive": str(OT), "fold_boundaries": [str(t) for t in BOUNDARIES],
            "unchanged_artifact_sha256": preserved, "packages": {p: importlib.metadata.version(p) for p in ["lightgbm", "numpy", "pandas"]}}, indent=2))
        print(variant, json.dumps(metrics), flush=True)
    for path, digest in preserved.items():
        assert fingerprint(ROOT / path) == digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="baseline,outages,weather,both")
    parser.add_argument("--append", action="store_true")
    main(parser.parse_args())
