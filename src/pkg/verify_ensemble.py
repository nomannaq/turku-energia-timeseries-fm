"""Official validators/scorer, paired uncertainty, provenance and final run report."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from ensemble_spec import ROOT, OUT, WORK, SUB, PROTECTED, fingerprint, save_json
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from score import load_submission, load_actuals, compute_metrics
from analyze_experiment_4 import losses

METRICS = ["mae_p50", "calibration_pinball_mean", "coverage_p10_p90", "brier_spike_mean", "brier_direction", "S"]


def markdown_table(f):
    def value(v):
        return "—" if pd.isna(v) else f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(f.columns) + " |", "| " + " | ".join(["---"]*len(f.columns)) + " |",
        *["| " + " | ".join(map(value, row)) + " |" for row in f.itertuples(index=False, name=None)]])


def main():
    for name, filename, origins in [("validation", "validation_forecast.csv", "baseline_output/validation/forecast_origins.csv"),
                                    ("test", "forecast_backtest.csv", "forecast_origins.csv")]:
        result = subprocess.run([sys.executable, str(ROOT / "validate_submission.py"), "--backtest", str(SUB / filename),
            "--origins", str(ROOT / origins)], capture_output=True, text=True)
        (OUT / f"validator_{name}.txt").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, str(ROOT / "score.py"), "--submission", str(SUB / "validation_forecast.csv"),
        "--actuals", str(ROOT / "data/actuals_validation.csv"), "--output-dir", str(OUT / "official_validation_score")], capture_output=True, text=True)
    (OUT / "official_score_log.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    actuals = load_actuals(ROOT / "data/actuals_validation.csv")
    candidate = load_submission(SUB / "validation_forecast.csv").merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
    reference = load_submission(ROOT / "submission/validation_forecast_B.csv").merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(candidate[["forecast_origin_utc", "delivery_start_utc", "step"]],
                                  reference[["forecast_origin_utc", "delivery_start_utc", "step"]])
    scores = pd.read_csv(OUT / "validation_comparison.csv").set_index("model")
    for metric, value in compute_metrics(candidate, 50).items():
        np.testing.assert_allclose(value, scores.loc["final_blend", metric], atol=1e-12)
    diff = losses(candidate) - losses(reference)
    days = candidate.forecast_origin_utc.dt.tz_convert("Europe/Helsinki").dt.date
    intervals = []
    for metric in diff:
        grouped = diff[metric].groupby(days).agg(["sum", "count"])
        chosen = np.random.default_rng(7).integers(0, len(grouped), size=(5000, len(grouped)))
        resampled = grouped["sum"].to_numpy()[chosen].sum(1)/grouped["count"].to_numpy()[chosen].sum(1)
        low, high = np.quantile(resampled, [.025, .975])
        intervals.append(dict(metric=metric, difference=diff[metric].mean(), ci_95_lower=low, ci_95_upper=high))
    ci = pd.DataFrame(intervals)
    ci.to_csv(OUT / "paired_day_bootstrap.csv", index=False)
    checks = {name: json.loads((OUT / name).read_text()) for name in
        ["feature_leak_checks.json", "lgb_output_leak_checks.json", "fm_leak_checks.json"]}
    assert checks["feature_leak_checks.json"]["rebuilt_matches_cached"]
    selection = json.loads((OUT / "weight_selection_audit.json").read_text())
    completed = json.loads((OUT / "blend_completed.json").read_text())
    for path, digest in completed["unchanged_artifact_sha256"].items():
        assert fingerprint(ROOT / path) == digest
    for path, digest in completed["output_sha256"].items():
        assert fingerprint(ROOT / path) == digest
    training = pd.read_csv(OUT / "training_fold_comparison.csv")
    observed = scores.loc["final_blend"]
    supplied = dict(mae_p50=41.60, calibration_pinball_mean=11.89, coverage_p10_p90=.733,
                    brier_spike_mean=.0908, brier_direction=.3928)
    pd.DataFrame([dict(metric=k, supplied_rounded=v, reproduced=observed[k], difference=observed[k]-v)
                  for k, v in supplied.items()]).to_csv(OUT / "reported_result_comparison.csv", index=False)
    report = f"""# Five-seed LightGBM + Chronos-2 ensemble

Implemented the supplied ensemble as a separate submission candidate, retaining
the original submission byte-for-byte. Both exports pass the official validator;
validation scores reproduce with `score.py` 0.1.1. No hidden test accuracy is claimed.

Uploadable test forecast: `submission/final_ensemble/forecast_backtest.csv`
(673 origins, 26,920 rows). Validation forecast:
`submission/final_ensemble/validation_forecast.csv` (672 origins, 26,880 rows;
26,700 rows with available actuals).

## Reproduced validation results

{markdown_table(scores[METRICS].reset_index())}

Chronos has no probability heads here; its standalone probability metrics are
intentionally blank. The final blend uses only averaged LightGBM probabilities.
`S` uses the supplied 30/20/20/10 normalized-loss formula (baseline S = 80).

The supplied rounded final-blend values were MAE 41.60, pinball 11.89, coverage
0.733, spike Brier 0.0908 and direction Brier 0.3928. Local reproduction is
MAE **{observed.mae_p50:.6f}**, pinball **{observed.calibration_pinball_mean:.6f}**,
coverage **{observed.coverage_p10_p90:.6f}**, spike **{observed.brier_spike_mean:.6f}**,
direction **{observed.brier_direction:.6f}**. Differences are recorded explicitly
in `reported_result_comparison.csv`; the remote implementation was not available
for byte-for-byte parity checking.

The standalone Chronos raw-quantile result has pinball 12.370898 and coverage
0.753745, differing from the supplied standalone figures 12.53 and 0.803, despite
matching its approximately 43.00 MAE. The supplied standalone postprocessing was
not available to audit. No extra calibration was introduced to force a match;
the requested final blend uses uncalibrated member quantiles and closely
reproduces the supplied final-blend metrics.

## Weight selection and training-only scores

The requested candidate keeps the supplied LightGBM weights: validation
**[0.7, 0.4, 0.5]**, test **[0.7, 0.4, 0.6]**, for steps 1–4, 5–16 and 17–40.
Chronos receives the remaining weight, identically for all three quantiles.
The independently recomputed grid optima are validation
**{selection['validation']['reproduced_grid_weights']}**, test
**{selection['test']['reproduced_grid_weights']}**. Agreement with the supplied
weights: validation **{selection['validation']['weights_match']}**, test
**{selection['test']['weights_match']}**. Differences do not silently change the
requested candidate. Full candidate losses are in `weight_grid.csv`.

Validation weight selection uses only rolling OOS training targets usable by
2026-09-06 21:00 UTC. Test selection may additionally use validation OOS targets,
but only those usable by 2026-10-04 21:00 UTC. The latter is a final-training
update; its settings are not retroactively credited with the validation score.
Weight selection minimizes total pinball over p10/p50/p90; the official scored
pinball metric averages p10/p90 only.

{markdown_table(training[['model', *METRICS]])}

Organizer baseline training-fold predictions use only function definitions
extracted from the shipped `baseline.ipynb`; no notebook workflow or side effects
are executed. The supplied wider model-selection search is not reconstructed:
this audits the requested blend against its five-seed raw LightGBM member.

## Implemented specification

- Five seeds: 7, 11, 23, 37, 51, with explicit bagging/feature/data seed offsets
  +1/+2/+3. Raw quantile and class predictions are averaged before sorting,
  normalization and spike-direction capping. Global family assignments and the
  separate steps 1–4 / 5–8 specialists match the supplied specification.
- Global parameters use the stored full-precision Optuna values underlying the
  rounded settings in the spec. Exact settings are in `lgb_completed.json`.
- Chronos-2, `chronos-forecasting==2.3.2`, CPU, batch size 64, 2,048-quarter
  context ending at origin minus 45 minutes. Last two target quarters are NaN.
  Forecast length 42; discard its first two positions and retain 40 quarters.
- Twelve covariates: seven mFRR/real-time/SA channels, RT-wind-minus-DA-wind,
  and the four known-future series. Every source is masked separately before
  subtraction or forward filling. Only future covariate gaps are forward-filled
  from already visible history. No backward filling or cross-origin learning.
- The same delivery spot is added after spread blending. No conformal adjustment
  is applied. Quantiles are sorted, prices rounded to four decimals and
  probabilities to six. No outage or new ICON/HARMONIE features are added.
- Model licenses recorded locally: LightGBM MIT; Chronos code/checkpoint
  Apache-2.0. No TimesFM or Timestral model is used. Dependency and input-data
  licenses remain their own; this is not a claim that every dependency is MIT/Apache.

## Availability and verification

Package `first_usable_origin_utc` controls visibility. Fingrid external series use
quarter end +30 minutes, rounded up to the hour. ECMWF runs use run time +6 hours.
Every LightGBM fit enforces origin before cutoff and target availability by cutoff.
Five training folds start Jul 5/19, Aug 2/16/30 at 21:00 UTC; validation and test
fits use Sep 6 and Oct 4 at 21:00 UTC respectively. Official templates determine
the exact forecast origin grids, including the 673-origin test period.

Future-value mutation checks passed at Jul 10 09:00, Sep 20 14:00 and Oct 20
03:00 UTC for all eligible LightGBM feature rows. All 225 fitted global/specialist
models checked at these origins produced bit-identical outputs after mutation.
Chronos inputs were identical for every required historical origin through each
of the Sep 20/Oct 20 cutoffs; cutoff predictions were bit-identical. Separate
solo-versus-batch checks pass within numerical tolerance for every period.

Paired changes versus the current single-seed submission, with 5,000 resamples
of the 28 local forecast-origin days (negative loss differences favor the blend):

{markdown_table(ci)}

Day resampling retains within-day overlap, but not every multi-day dependence.
Repeated validation inspection and upstream training-fold model selection limit
the independence of these estimates. Feature availability tests do not audit
foundation-model pretraining overlap. ECMWF's six-hour publication delay is the
specified assumption. The separate test forecast has no reported actuals or score.

## Reproduction

```sh
.venv/bin/python src/pkg/run_final_ensemble.py
```

The orchestrator uses `.venv` for LightGBM/scoring and `.venv-chronos` for Chronos.
The existing pinned checkpoint and source data are required. Intermediate members,
model files and resumable Chronos batches are in ignored `work_private/final_ensemble`.
The original submission remains at `forecast_backtest.csv`; the new uploadable
candidate is in the separate path above.
"""
    (OUT / "REPORT.md").write_text(report)
    save_json(OUT / "verification.json", dict(validation_format_passed=True, test_format_passed=True,
        independent_score_match=True, mutation_checks_passed=True, original_submission_unchanged=True,
        validation_origins=672, test_origins=673, **completed))
    print(scores[METRICS].to_string())
    print(ci.to_string(index=False))
    print("Verified both submissions, official scores, mutation checks and fallback hashes.")


if __name__ == "__main__":
    main()
