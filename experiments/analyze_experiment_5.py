"""Verify Experiment 5 exports, score paired uncertainty and write its report."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from experiment_5_parameter_tuning import ROOT, OUT, WORK, VARIANTS, KEYS, fingerprint
from analyze_experiment_4 import read, losses
from score import compute_metrics, load_actuals

METRICS = ["mae_p50", "calibration_pinball_mean", "brier_spike_mean", "brier_direction"]


def markdown_table(frame):
    def value(v):
        return f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                      *["| " + " | ".join(map(value, row)) + " |" for row in frame.itertuples(index=False, name=None)]])


def main():
    completed = json.loads((OUT / "completed.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    selection = json.loads((OUT / "frozen_selection.json").read_text())
    expected = pd.read_csv(OUT / "comparison.csv").set_index("model")
    actuals = load_actuals(ROOT / "data/actuals_validation.csv")
    frames, verified = {}, []
    for name in expected.index:
        path = OUT / f"validation_{name}.csv"
        f = read(path)
        check = subprocess.run([sys.executable, str(ROOT / "validate_submission.py"), "--backtest", str(path),
            "--origins", str(ROOT / "baseline_output/validation/forecast_origins.csv")], capture_output=True, text=True)
        (OUT / f"validation_check_{name}.txt").write_text(check.stdout + check.stderr)
        assert check.returncode == 0, check.stdout + check.stderr
        joined = f.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
        measured = compute_metrics(joined, 50)
        for metric, value in measured.items():
            np.testing.assert_allclose(value, expected.loc[name, metric], atol=1e-12, rtol=1e-12)
        verified.append(dict(model=name, **measured))
        frames[name] = joined
    reference = read(ROOT / "submission/validation_forecast_B.csv")
    frames["submitted_B"] = reference.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
    verified.append(dict(model="submitted_B", **compute_metrics(frames["submitted_B"], 50)))
    pd.DataFrame(verified).to_csv(OUT / "verified_comparison.csv", index=False)
    contrasts = [(f"{v}_tuned", f"{v}_control") for v in VARIANTS]
    contrasts += [(f"{v}_tuned", "submitted_B") for v in VARIANTS]
    contrasts += [(f"{v}_tuned", "baseline_tuned") for v in VARIANTS if v != "baseline"]
    intervals = []
    for name, reference in contrasts:
        f, base = frames[name], frames[reference]
        pd.testing.assert_frame_equal(f[KEYS], base[KEYS], check_dtype=False)
        difference = losses(f) - losses(base)
        days = f.forecast_origin_utc.dt.tz_convert("Europe/Helsinki").dt.date
        for metric in METRICS:
            grouped = difference[metric].groupby(days).agg(["sum", "count"])
            chosen = np.random.default_rng(7).integers(0, len(grouped), size=(5000, len(grouped)))
            samples = grouped["sum"].to_numpy()[chosen].sum(1) / grouped["count"].to_numpy()[chosen].sum(1)
            lo, hi = np.quantile(samples, [.025, .975])
            intervals.append(dict(model=name, reference=reference, metric=metric, difference=difference[metric].mean(),
                ci_95_lower=lo, ci_95_upper=hi, origin_days=len(grouped), resamples=5000))
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(OUT / "paired_day_bootstrap.csv", index=False)
    trials = pd.read_csv(OUT / "trials.csv", dtype={"head": str})
    search = trials[trials.phase == "search"]
    assert len(search.groupby(["variant", "head"])) == 20
    assert search.groupby(["variant", "head"]).size().eq(8).all()
    selected_parameters = {variant: {head: protocol["configs"][head][cid]
        for head, cid in heads.items()} for variant, heads in selection["selected_candidates"].items()}
    (OUT / "selected_parameters.json").write_text(json.dumps(selected_parameters, indent=2))
    crossings = []
    for variant in VARIANTS:
        medians = {}
        for mode in ["control", "tuned"]:
            ids = {str(k): 0 for k in [.1, .5, .9]} if mode == "control" else selection["selected_candidates"][variant]
            predictions = []
            for kind in [.1, .5, .9]:
                with np.load(WORK / f"{variant}_{kind}_{ids[str(kind)]}_7_validation.npz") as saved:
                    predictions.append(saved["prediction"])
            q = np.column_stack(predictions)
            moved = np.sort(q, axis=1)[:, 1] - q[:, 1]
            medians[mode] = q[:, 1]
            exported_shift = frames[f"{variant}_{mode}"].p50 - frames[f"{variant}_control"].p50
            crossings.append(dict(model=f"{variant}_{mode}",
                raw_quantile_crossing_rows=int(((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])).sum()),
                median_changed_by_initial_sort_rows=int((np.abs(moved) > 1e-8).sum()),
                mean_abs_initial_median_shift=float(np.abs(moved).mean()),
                exported_median_changed_vs_control_rows=int((exported_shift.abs() > 1e-8).sum()),
                mean_abs_exported_median_shift_vs_control=float(exported_shift.abs().mean())))
        if selection["selected_candidates"][variant]["0.5"] == 0:
            np.testing.assert_array_equal(medians["control"], medians["tuned"])
    pd.DataFrame(crossings).to_csv(OUT / "quantile_sorting_diagnostics.csv", index=False)
    selected_rows = []
    for variant, heads in selection["selected_candidates"].items():
        for head, cid in heads.items():
            selected_trial = trials[(trials.variant == variant) & (trials["head"] == head) &
                (trials.phase == "confirmation") & (trials.candidate == cid)]
            incumbent = trials[(trials.variant == variant) & (trials["head"] == head) &
                (trials.phase == "confirmation") & (trials.candidate == 0)]
            assert set(selected_trial.seed) == {7, 19}
            assert selected_trial.loss.mean() <= incumbent.loss.mean() + 1e-12
            selected_rows.append(dict(variant=variant, head=head, candidate=cid,
                confirmation_loss=selected_trial.loss.mean(), incumbent_loss=incumbent.loss.mean(),
                wins_both_seeds=bool((selected_trial.set_index("seed").loss <= incumbent.set_index("seed").loss).all())))
    pd.DataFrame(selected_rows).to_csv(OUT / "selection_summary.csv", index=False)
    for path, digest in protocol["unchanged_artifact_sha256"].items():
        assert fingerprint(ROOT / path) == digest
    nominated = selection["nominated_feature_set"] + "_tuned"
    scores = pd.DataFrame(verified).set_index("model")
    best = scores.loc[[f"{v}_tuned" for v in VARIANTS], "mae_p50"].idxmin()
    comparison = scores[METRICS + ["coverage_p10_p90"]].reset_index()
    deltas = pd.DataFrame([{"feature_set": v, **{metric: scores.loc[f"{v}_tuned", metric] -
        scores.loc[f"{v}_control", metric] for metric in METRICS}} for v in VARIANTS])
    deltas.to_csv(OUT / "tuning_effect.csv", index=False)
    mae_wins = int((deltas.mae_p50 < -1e-9).sum())
    selected_ci = intervals[(intervals.model == nominated) & (intervals.reference == "submitted_B") & (intervals.metric == "mae_p50")].iloc[0]
    report = f"""# Experiment 5 — parameter tuning

Completed four equal-budget searches: existing features, outages, archived weather,
and both. The feature set nominated **before official validation scoring** was
**{nominated}**, using pre-validation confirmation MAE. Its validation MAE is
**{scores.loc[nominated, 'mae_p50']:.6f} EUR/MWh**, versus
**{scores.loc['submitted_B', 'mae_p50']:.6f}** for the saved submission.
The lowest validation MAE among tuned runs is **{best}** at
**{scores.loc[best, 'mae_p50']:.6f}**; that retrospective ranking did not choose parameters.
Existing submission files are unchanged.

**Tuning itself reduced validation median MAE in {mae_wins} of four matched comparisons.**
The feature-set comparison and the effect of parameter changes are distinct;
a model beating the saved submission does not by itself show a tuning benefit.

## Validation results

All losses below are lower-is-better. Coverage targets approximately 0.80.
Each forecast has 672 origins × 40 steps = 26,880 rows; the official scorer
uses 26,700 rows with available actuals. No test-period targets enter any fit.

{markdown_table(comparison)}

Tuned minus matched-control losses (negative means tuning helped):

{markdown_table(deltas)}

`control` preserves the inherited global and short-horizon settings, using the
same single-week calibration and final seed as `tuned`. The saved submission
used a different, longer calibration history. Compare tuned versus its matching
control to isolate tuning; compare against `submitted_B` for practical benefit.

The existing pipeline sorts independently predicted quantiles, then sorts again
after interval calibration. Thus changing p10/p90 can move the exported p50 even
when the raw median head is unchanged. `quantile_sorting_diagnostics.csv` records
crossings and median shifts; unchanged median-head selections are verified to
produce exactly identical raw medians. This postprocessing was retained for the
controlled experiment, not changed in response to validation outcomes.

## Search and temporal controls

- Eight candidates per output head and feature set: inherited settings plus seven
  reproducible random configurations; 160 candidate/head/feature-set combinations.
- Tuned learning rate, tree count, number of leaves, minimum leaf observations,
  feature fraction and L2 regularization. Global and first-two-hour specialist
  models are both included. Bagging and feature-family assignments remain fixed.
- Search folds in UTC: Jul 19 21:00–Aug 2 21:00 and Aug 2 21:00–Aug 16 21:00.
- Confirm the two best search candidates plus the incumbent on Aug 16 21:00–
  Aug 30 21:00, using seeds 7 and 19; select by mean loss. If the incumbent is
  already in the top two, only two distinct candidates need confirmation.
- Choose each output head independently: p50 MAE, p10/p90 pinball, direction
  multiclass Brier, and mean high/low spike Brier. Head-level selection precedes
  final quantile sorting, direction capping of spikes and interval calibration.
- Freeze choices in `frozen_selection.json`. Calibrate intervals on Aug 30
  21:00–Sep 6 21:00 only, then refit using all labels available at Sep 6 21:00.
  Final predictions use seed 7; seed 19 checks selection robustness.
- For every fold, training origins precede its cutoff and label `target_fu` is
  at or before that cutoff. Scoring/calibration labels must be available by the
  window end. Forecast targets crossing a boundary are purged accordingly.
- Official validation uses the package's UTC origin template: Sep 6 21:00–
  Oct 4 21:00 exclusive. Its actuals are used for scoring only after parameter freezing.

## Uncertainty and limitations

For the pre-nominated model versus the saved submission, the paired MAE change
is {selected_ci.difference:+.6f} EUR/MWh, with a day-bootstrap 95% interval
[{selected_ci.ci_95_lower:+.6f}, {selected_ci.ci_95_upper:+.6f}]. Negative favors the nominated model.
`paired_day_bootstrap.csv` includes matched-control and tuned-feature contrasts
for every scoring objective, using 5,000 resamples of the 28 local origin days.
These intervals preserve within-day overlap but not all multi-day dependence;
they do not correct for repeated experimentation or multiple comparisons.

This is a bounded search, not proof of globally optimal hyperparameters. The
official validation has been reviewed in previous experiments, so it is not a
fresh final test. The inherited parameters and feature families also come from
earlier work on the training period; these new temporal partitions isolate the
new search, not the entire historical research process. The calibration week
is short, and finite-sample interval adjustment is not a coverage guarantee
under time dependence or changing market regimes. Weather availability retains
Experiment 4's conservative run-plus-six-hours assumption, not measured
historical publication timestamps.

## Verification and reproduction

All eight CSVs pass the official submission validator; independently reloaded
forecasts reproduce every saved score. SHA-256 checks confirm all three protected
submission artifacts are unchanged. Training input hashes, library versions,
parameter values, seed losses and selections are recorded alongside this report.
`selected_parameters.json` expands the frozen candidate IDs into reusable settings.
Runtime: {completed['runtime_seconds']/60:.1f} minutes.

```sh
.venv/bin/python experiments/experiment_5_parameter_tuning.py
.venv/bin/python experiments/analyze_experiment_5.py
```

Predictions are resumable from `work_private/experiment5`; the protocol must
match to reuse them. Use new output/cache directories for a different search.
"""
    (OUT / "REPORT.md").write_text(report)
    print(comparison.to_string(index=False))
    print("Pre-validation nomination:", nominated)
    print("All eight validators and independent score comparisons passed; submissions unchanged.")


if __name__ == "__main__":
    main()
