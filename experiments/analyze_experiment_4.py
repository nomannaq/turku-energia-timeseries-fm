"""Independently verify saved forecasts and paired differences against the rebuilt control."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score import compute_metrics, load_actuals
from experiment_4_external_data import fingerprint

OUT = ROOT / "results/experiment_4_external_data"
KEYS = ["forecast_origin_utc", "delivery_start_utc", "step"]


def read(path):
    f = pd.read_csv(path)
    for c in KEYS[:2]:
        f[c] = pd.to_datetime(f[c], utc=True)
    assert len(f) == 26880 and not f.duplicated(KEYS).any()
    assert np.isfinite(f.filter(regex="^p").to_numpy()).all()
    assert ((f.p10 <= f.p50) & (f.p50 <= f.p90)).all()
    np.testing.assert_allclose(f.p_up + f.p_down + f.p_none, 1, atol=1e-6)
    assert ((f.filter(regex="^p_") >= 0) & (f.filter(regex="^p_") <= 1)).all().all()
    return f


def losses(f):
    y = f.imbalance_price
    low, high = y - f.p10, y - f.p90
    result = pd.DataFrame({"mae_p50": (y - f.p50).abs(),
        "calibration_pinball_mean": (np.maximum(.1 * low, -.9 * low) + np.maximum(.9 * high, -.1 * high)) / 2,
        "brier_spike_mean": ((f.p_spike_high - (y > f.spot_price + 50)) ** 2 +
                             (f.p_spike_low - (y < f.spot_price - 50)) ** 2) / 2,
        "brier_direction": sum((f[f"p_{k}"] - (f.direction == k)) ** 2 for k in ["up", "down", "none"])})
    result.loc[y.isna(), ["mae_p50", "calibration_pinball_mean", "brier_spike_mean"]] = np.nan
    result.loc[f.direction.isna(), "brier_direction"] = np.nan
    return result


def main():
    actuals = load_actuals(ROOT / "data/actuals_validation.csv")
    base = read(OUT / "validation_baseline.csv").merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
    expected = pd.read_csv(OUT / "comparison.csv").set_index("model")
    base_loss = losses(base)
    results = []
    for name in ["baseline", "outages", "weather", "both"]:
        f = read(OUT / f"validation_{name}.csv")
        pd.testing.assert_frame_equal(base[KEYS], f[KEYS], check_dtype=False)
        joined = f.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
        official = compute_metrics(joined, 50)
        for metric, value in official.items():
            np.testing.assert_allclose(value, expected.loc[name, metric], atol=1e-12, rtol=1e-12)
        if name == "baseline":
            continue
        differences = losses(joined) - base_loss
        days = f.forecast_origin_utc.dt.tz_convert("Europe/Helsinki").dt.date
        for metric in differences:
            grouped = differences[metric].groupby(days).agg(["sum", "count"])
            rng = np.random.default_rng(7)
            chosen = rng.integers(0, len(grouped), size=(5000, len(grouped)))
            samples = grouped["sum"].to_numpy()[chosen].sum(axis=1) / grouped["count"].to_numpy()[chosen].sum(axis=1)
            lo, hi = np.quantile(samples, [.025, .975])
            results.append({"model": name, "reference": "baseline", "metric": metric,
                            "difference": differences[metric].mean(), "ci_95_lower": lo,
                            "ci_95_upper": hi, "origin_days": len(grouped), "resamples": 5000})
    pd.DataFrame(results).to_csv(OUT / "paired_day_bootstrap.csv", index=False)
    metadata = json.loads((OUT / "run.json").read_text())
    for path, digest in metadata["unchanged_artifact_sha256"].items():
        assert fingerprint(ROOT / path) == digest
    print("All four forecast files pass alignment/probability checks and match official scores; submissions unchanged.")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
