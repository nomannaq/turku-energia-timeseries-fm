"""Check Experiment 2 scores and estimate paired MAE differences by resampling origin days."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score import compute_metrics


def main(output_dir="results/experiment_2_chronos"):
    out = ROOT / output_dir
    keys = ["forecast_origin_utc", "delivery_start_utc", "step"]
    base = pd.read_csv(ROOT / "submission/validation_forecast_B.csv")
    actuals = pd.read_csv(ROOT / "data/actuals_validation.csv")
    comparison = pd.read_csv(out / "comparison.csv").set_index("model")
    results, daily_results = [], []
    for path in sorted(out.glob("validation_*.csv")):
        name = path.stem.removeprefix("validation_")
        forecasts = pd.read_csv(path)
        for frame in (forecasts, base, actuals):
            for col in ("forecast_origin_utc", "delivery_start_utc"):
                if col in frame:
                    frame[col] = pd.to_datetime(frame[col], utc=True)
        joined = forecasts.merge(base, on=keys, validate="one_to_one", suffixes=("", "_lightgbm"))
        joined = joined.merge(actuals, on="delivery_start_utc", validate="many_to_one", how="left")
        # Existing B probabilities are carried only in memory so the official scorer
        # can independently verify price metrics. They are not foundation-model predictions.
        official = compute_metrics(joined, 50)
        for metric in ("mae_p50", "pinball_p10", "pinball_p90", "calibration_pinball_mean", "coverage_p10_p90"):
            np.testing.assert_allclose(official[metric], comparison.loc[name, metric], rtol=1e-12, atol=1e-12)
        scored = joined.dropna(subset=["imbalance_price"]).copy()
        scored["difference"] = (scored.imbalance_price - scored.p50).abs() - (scored.imbalance_price - scored.p50_lightgbm).abs()
        scored["day"] = scored.forecast_origin_utc.dt.tz_convert("Europe/Helsinki").dt.date
        days = scored.groupby("day").difference.agg(["sum", "size", "mean"])
        rng = np.random.default_rng(7)
        selected = rng.integers(0, len(days), size=(5000, len(days)))
        samples = days["sum"].to_numpy()[selected].sum(axis=1) / days["size"].to_numpy()[selected].sum(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        results.append({"model": name, "reference": "lightgbm_B", "origin_days": len(days),
                        "bootstrap_resamples": 5000, "mae_difference": scored.difference.mean(),
                        "ci_95_lower": lo, "ci_95_upper": hi})
        daily_results.append(days.reset_index().assign(model=name))
    pd.DataFrame(results).to_csv(out / "paired_mae_bootstrap.csv", index=False)
    pd.concat(daily_results, ignore_index=True).to_csv(out / "daily_mae_differences.csv", index=False)
    print("All experiment price metrics match the official scoring function.")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/experiment_2_chronos")
    main(parser.parse_args().output_dir)
