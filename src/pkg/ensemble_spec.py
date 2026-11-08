"""Frozen settings for the requested LightGBM / Chronos-2 ensemble."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work_private/final_ensemble"
OUT = ROOT / "results/final_ensemble"
SUB = ROOT / "submission/final_ensemble"
SEEDS = [7, 11, 23, 37, 51]
KINDS = ["dir", "spk", .1, .5, .9]
SELECTED = {"dir": "EW", "spk": "EX", "0.1": "E4", "0.5": "EX", "0.9": "EX"}
SHORT = dict(learning_rate=.03, num_leaves=15, min_data_in_leaf=300,
             feature_fraction=.5, lambda_l2=10, n_rounds=300)
OV = pd.Timestamp("2026-09-06T21:00Z")
OT = pd.Timestamp("2026-10-04T21:00Z")
END = pd.Timestamp("2026-11-01T22:00Z")
BOUNDARIES = pd.to_datetime(["2026-07-05T21:00Z", "2026-07-19T21:00Z", "2026-08-02T21:00Z",
    "2026-08-16T21:00Z", "2026-08-30T21:00Z", "2026-09-06T21:00Z"])
PERIODS = {**{f"fold{i}": (a, b) for i, (a, b) in enumerate(zip(BOUNDARIES[:-1], BOUNDARIES[1:]))},
           "validation": (OV, OT), "test": (OT, END)}
WEIGHTS = {"validation": [.7, .4, .5], "test": [.7, .4, .6]}
KEYS = ["forecast_origin_utc", "delivery_start_utc", "step"]
MODEL = ROOT / "work_private/models/chronos-2"
PROTECTED = ["forecast_backtest.csv", "forecast_backtest_packageonly.csv", "submission/validation_forecast_B.csv"]


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n")


def guard(path, value):
    canonical = json.loads(json.dumps(value, default=str))
    if path.exists():
        assert json.loads(path.read_text()) == canonical, f"Stale cache protocol: {path}"
    else:
        save_json(path, canonical)


def params():
    # Full-precision values behind the rounded Optuna settings in the supplied spec.
    saved = json.loads((ROOT / "results/v2_optuna_best.json").read_text())
    return {str(k): saved["dir" if isinstance(k, str) else "0.5"]["params"] for k in KINDS}


def load_frame():
    import numpy as np
    x = pd.read_parquet(ROOT / "work_private/features_B.parquet")
    assert np.array_equal(x.index, np.arange(len(x)))
    x.target_fu = pd.to_datetime(x.target_fu, utc=True)
    x["y_dir3"] = x.y_dir.map({-1: 0, 0: 1, 1: 2})
    x["y_spk"] = np.select([x.spread < -50, x.spread > 50], [0, 2], 1)
    x.loc[x.spread.isna(), "y_spk"] = -1
    features = [c for c in x if c.startswith(("st_", "dl_", "wx_"))] + ["step"]
    x[features] = x[features].astype("float32")
    return x
