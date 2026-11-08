"""Apply supplied weights; audit grid selection using only eligible OOS targets."""
import json
import sys

import numpy as np
import pandas as pd

from ensemble_spec import (ROOT, WORK, OUT, SUB, PERIODS, WEIGHTS, KEYS, OV, OT,
    load_frame, save_json, fingerprint, PROTECTED, guard)
sys.path.insert(0, str(ROOT))
from score import compute_metrics, load_actuals


def load_members(x, period):
    with np.load(WORK / f"lgb_{period}.npz") as stored:
        j = stored["indices"]
        q = np.sort(np.column_stack([stored[str(k)] for k in [.1, .5, .9]]), axis=1)
        d, s = stored["dir"].copy(), stored["spk"].copy()
    d /= d.sum(1, keepdims=True)
    s[:, 0] = np.minimum(s[:, 0], d[:, 0])
    s[:, 2] = np.minimum(s[:, 2], d[:, 2])
    c = pd.read_parquet(WORK / f"chronos_{period}.parquet")
    expected = x.iloc[j][["o", "t", "step"]].rename(columns={"o": KEYS[0], "t": KEYS[1]}).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, c[KEYS], check_dtype=False)
    np.testing.assert_allclose(c.spot, x.dl_spot.iloc[j], atol=1e-7, rtol=1e-7)
    return dict(indices=j, lgb=q, chronos=np.sort(c[["q10", "q50", "q90"]].to_numpy(), axis=1), direction=d, spike=s)


def blend(member, steps, weights):
    bucket = np.digitize(steps, [4.5, 16.5])
    w = np.asarray(weights)[bucket, None]
    return np.sort(w * member["lgb"] + (1-w) * member["chronos"], axis=1)


def frame(x, member, q):
    j, d, s = member["indices"], member["direction"], member["spike"]
    spot = x.dl_spot.to_numpy(dtype=float)[j]
    assert np.isfinite(spot).all()
    f = pd.DataFrame({KEYS[0]: x.o.to_numpy()[j], KEYS[1]: x.t.to_numpy()[j], "step": x.step.to_numpy()[j].astype(int),
        "p10": spot+q[:, 0], "p50": spot+q[:, 1], "p90": spot+q[:, 2],
        "p_down": d[:, 0], "p_none": d[:, 1], "p_up": d[:, 2], "p_spike_low": s[:, 0], "p_spike_high": s[:, 2]})
    f[["p10", "p50", "p90"]] = f[["p10", "p50", "p90"]].round(4)
    f[["p_down", "p_none", "p_up"]] = f[["p_down", "p_none", "p_up"]].round(6)
    f["p_none"] = (1-f.p_up-f.p_down).clip(lower=0).round(6)
    f["p_up"] = (1-f.p_down-f.p_none).round(6)
    f["p_spike_high"] = np.minimum(f.p_spike_high, f.p_up).clip(0, 1).round(6)
    f["p_spike_low"] = np.minimum(f.p_spike_low, f.p_down).clip(0, 1).round(6)
    return f


def write_forecast(f, period, destination):
    template = pd.read_csv(ROOT / ("baseline_output/validation/sample_submission.csv" if period == "validation" else "sample_submission.csv"))
    for c in KEYS[:2]:
        template[c] = pd.to_datetime(template[c], utc=True)
    pd.testing.assert_frame_equal(f[KEYS], template[KEYS], check_dtype=False)
    out = f.assign(data_cutoff_utc="", top_drivers="")
    for c in KEYS[:2]:
        out[c] = pd.to_datetime(out[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    out[template.columns].to_csv(destination, index=False)


def pool(members, names):
    return {key: np.concatenate([members[n][key] for n in names]) for key in members[names[0]]}


def select_weights(x, member, cutoff, stage):
    j = member["indices"]
    eligible = (x.target_fu.iloc[j] <= cutoff).to_numpy() & x.spread.iloc[j].notna().to_numpy()
    assert (x.o.iloc[j[eligible]] < cutoff).all()
    buckets = np.digitize(x.step.to_numpy()[j], [4.5, 16.5])
    y = x.spread.to_numpy()[j]
    selected, rows = [], []
    for b in range(3):
        mask = eligible & (buckets == b)
        candidates = []
        for w in np.arange(11)/10:
            q = np.sort(w*member["lgb"][mask] + (1-w)*member["chronos"][mask], axis=1)
            residual = y[mask, None] - q
            losses = np.maximum(residual * [.1, .5, .9], residual * [-.9, -.5, -.1])
            score = float(losses.sum(axis=1).mean())
            candidates.append(score)
            rows.append(dict(stage=stage, bucket=b, lgb_weight=w, chronos_weight=1-w,
                mean_total_pinball=score, rows=int(mask.sum())))
        selected.append(float(np.argmin(candidates)/10))
    return selected, rows, int(eligible.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    preserved = {p: fingerprint(ROOT / p) for p in PROTECTED}
    guard(WORK / "blend_protocol.json", dict(weights=WEIGHTS, quantile_sort=True, conformal=False,
        selection_objective="sum of pinball at 0.1, 0.5, 0.9", preserved=preserved))
    x = load_frame()
    members = {period: load_members(x, period) for period in PERIODS}
    training_names = [p for p in PERIODS if p.startswith("fold")]
    searches, selections = [], {}
    for stage, names, cutoff in [("validation", training_names, OV), ("test", training_names+["validation"], OT)]:
        selected, rows, n = select_weights(x, pool(members, names), cutoff, stage)
        searches.extend(rows)
        selections[stage] = dict(specified_lgb_weights=WEIGHTS[stage], reproduced_grid_weights=selected,
            weights_match=selected == WEIGHTS[stage], selection_target_cutoff=cutoff, eligible_rows=n,
            latest_eligible_target_fu=x.target_fu.iloc[pool(members, names)["indices"]].loc[lambda s: s <= cutoff].max())
    pd.DataFrame(searches).to_csv(OUT / "weight_grid.csv", index=False)
    # Supplied weights remain the implemented spec even if reproduction differs.
    save_json(OUT / "weight_selection_audit.json", selections)
    save_json(OUT / "frozen_blend.json", dict(weights=WEIGHTS, selections=selections,
        no_validation_based_changes=True, frozen_at_utc=pd.Timestamp.now(tz="UTC")))
    actuals = load_actuals(ROOT / "data/actuals_validation.csv")
    results = []
    for name, path in [("organizer_baseline", ROOT / "baseline_output/validation/forecast_backtest.csv"),
                       ("current_lightgbm_single", ROOT / "submission/validation_forecast_B.csv")]:
        f = pd.read_csv(path)
        f[KEYS[1]] = pd.to_datetime(f[KEYS[1]], utc=True)
        results.append(dict(model=name, **compute_metrics(f.merge(actuals, on=KEYS[1], how="left", validate="many_to_one"), 50)))
    for period in ["validation", "test"]:
        member = members[period]
        steps = x.step.to_numpy()[member["indices"]]
        f = frame(x, member, blend(member, steps, WEIGHTS[period]))
        destination = SUB / ("validation_forecast.csv" if period == "validation" else "forecast_backtest.csv")
        write_forecast(f, period, destination)
        if period == "validation":
            for name, q in [("lightgbm_5seed_raw", member["lgb"]), ("chronos2_covariates", member["chronos"]),
                            ("final_blend", blend(member, steps, WEIGHTS[period]))]:
                predictions = frame(x, member, q)
                joined = predictions.merge(actuals, on=KEYS[1], how="left", validate="many_to_one")
                metrics = compute_metrics(joined, 50)
                if name == "chronos2_covariates":
                    # The foundation model has no probability heads in this implementation.
                    metrics = {k: v for k, v in metrics.items() if not (k.startswith("brier") or "spike" in k or "direction" in k)}
                results.append(dict(model=name, **metrics))
                joined["horizon_hour"] = (joined.step-1)//4+1
                horizon = []
                for h, g in joined.groupby("horizon_hour"):
                    m = compute_metrics(g, 50)
                    if name == "chronos2_covariates":
                        m = {k: v for k, v in m.items() if not (k.startswith("brier") or "spike" in k or "direction" in k)}
                    horizon.append(dict(horizon_hour=h, **m))
                pd.DataFrame(horizon).to_csv(OUT / f"horizon_{name}.csv", index=False)
                if name != "chronos2_covariates":
                    write_forecast(predictions, period, OUT / f"validation_{name}.csv")
    summary = pd.DataFrame(results)
    baseline = summary.set_index("model").loc["organizer_baseline"]
    summary["S"] = sum(weight*summary[metric]/baseline[metric] for weight, metric in
        [(30, "mae_p50"), (20, "calibration_pinball_mean"), (20, "brier_spike_mean"), (10, "brier_direction")])
    summary.to_csv(OUT / "validation_comparison.csv", index=False)
    for p, digest in preserved.items():
        assert fingerprint(ROOT / p) == digest
    save_json(OUT / "blend_completed.json", dict(unchanged_artifact_sha256=preserved,
        output_sha256={str(p.relative_to(ROOT)): fingerprint(p) for p in [SUB / "validation_forecast.csv", SUB / "forecast_backtest.csv"]}))
    print(summary[["model", "mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction", "S"]].to_string(index=False))
    print(json.dumps(selections, indent=2, default=str))


if __name__ == "__main__":
    main()
