"""Training-only S scores against the organizer's exact notebook baseline functions."""
import ast
import json
import sys

import numpy as np
import pandas as pd

from ensemble_spec import ROOT, WORK, OUT, PERIODS, WEIGHTS, KEYS, OV, load_frame, save_json
from fm_blend3 import load_members, blend, frame
sys.path.insert(0, str(ROOT))
from score import compute_metrics


def baseline_functions():
    # Extract definitions only; never execute notebook workflow, shell calls or writes.
    wanted = {"rows_available_at", "to_utc", "series", "persistence_at", "calibrate", "build_forecast"}
    notebook = json.loads((ROOT / "baseline.ipynb").read_text())
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            definitions.extend(n for n in ast.parse("".join(cell["source"])).body
                if isinstance(n, ast.FunctionDef) and n.name in wanted)
    assert {n.name for n in definitions} == wanted
    namespace = dict(np=np, pd=pd, HORIZON_STEPS=40, SPIKE_MARGIN=50., IMBALANCE="fi_imbalance_price",
        DIRECTION="fi_mfrr_direction", STEP_OFFSETS=pd.to_timedelta(np.arange(40)*15, unit="min"))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / "baseline.ipynb"), "exec"), namespace)
    return namespace


def main():
    methods = baseline_functions()
    inputs = pd.read_parquet(ROOT / "data/inputs.parquet")
    spot = pd.read_csv(ROOT / "data/spot_nordpool.csv")
    for table in [inputs, spot]:
        for c in ["timestamp_utc", "first_usable_origin_utc"]:
            table[c] = pd.to_datetime(table[c], utc=True)
    x = load_frame()
    frames = {name: [] for name in ["organizer_baseline", "lightgbm_5seed_raw", "specified_blend"]}
    audit = []
    for period, (start, stop) in PERIODS.items():
        if not period.startswith("fold"):
            continue
        m = load_members(x, period)
        j = m["indices"]
        eligible = (x.target_fu.iloc[j] <= OV).to_numpy()
        origins = pd.Series(pd.date_range(start, stop, freq="h", inclusive="left"))
        cache = WORK / f"organizer_{period}.parquet"
        if cache.exists():
            base = pd.read_parquet(cache)
        else:
            calibration = methods["calibrate"](inputs, spot, start)
            assert calibration["max_first_usable_used"] <= start
            base = methods["build_forecast"](inputs, calibration, origins)
            base.to_parquet(cache, index=False)
        f = frame(x, m, m["lgb"])
        pd.testing.assert_frame_equal(base[KEYS], f[KEYS], check_dtype=False)
        candidates = {"organizer_baseline": base, "lightgbm_5seed_raw": f,
            "specified_blend": frame(x, m, blend(m, x.step.to_numpy()[j], WEIGHTS["validation"]))}
        for name, predictions in candidates.items():
            f = predictions.loc[eligible].copy()
            f["imbalance_price"] = x.imb.to_numpy()[j[eligible]]
            f["spot_price"] = x.spot_t.to_numpy()[j[eligible]]
            f["direction"] = x.y_dir.iloc[j[eligible]].map({-1: "down", 0: "none", 1: "up"}).to_numpy()
            frames[name].append(f)
            audit.append(dict(period=period, model=name, **compute_metrics(f, 50)))
        print("Training S-score baseline and forecasts:", period, flush=True)
    scores = pd.DataFrame([dict(model=name, **compute_metrics(pd.concat(parts, ignore_index=True), 50)) for name, parts in frames.items()])
    baseline = scores.set_index("model").loc["organizer_baseline"]
    scores["S"] = sum(weight*scores[metric]/baseline[metric] for weight, metric in
        [(30, "mae_p50"), (20, "calibration_pinball_mean"), (20, "brier_spike_mean"), (10, "brier_direction")])
    scores.to_csv(OUT / "training_fold_comparison.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "training_scores_by_fold.csv", index=False)
    save_json(OUT / "training_selection_scope.json", dict(target_cutoff=OV,
        description="Audits the supplied model choice against five-seed raw LightGBM using only rolling OOS training predictions; does not repeat the remote model's unprovided broader search."))
    print(scores[["model", "mae_p50", "calibration_pinball_mean", "brier_spike_mean", "brier_direction", "S"]].to_string(index=False))


if __name__ == "__main__":
    main()
