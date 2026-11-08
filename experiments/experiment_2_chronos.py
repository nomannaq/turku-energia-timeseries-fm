"""Experiment 2: local, zero-shot Chronos-2 price forecasting on official validation origins.

No parameter fitting, test scoring, or submission replacement. Each origin is an
independent task; cross-learning between origins is explicitly disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "work_private/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT / "src/pkg"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pkg_features as pf
from score import pinball

QUARTER = pd.Timedelta(minutes=15)
MODES = ("price", "spread", "spread_covariates")
KNOWN = {
    "spot": ["fi_spot_price"],
    "wind_da": ["fi_wind_fc_da"],
    "consumption_da": ["fi_consumption_fc_da"],
    "solar_da": ["fi_solar_fc_da"],
    "residual_load_da": ["fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da"],
}
PAST = {
    "mfrr_direction": ["fi_mfrr_direction"],
    "wind_error": ["x_wind_act", "fi_wind_fc_da"],
    "calculated_imbalance": ["x_calc_imb"],
    "nuclear_generation": ["x_nuc"],
    "net_import": ["x_netimp"],
    "frequency_deviation": ["x_freq"],
    "kinetic_energy": ["x_kin"],
    "scheduled_mfrr_spread": ["x_sa_price", "fi_spot_price"],
}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visible(values, availability, columns, timestamps, origin):
    """Mask every source separately before deriving a feature."""
    v = values.reindex(timestamps)[columns].copy()
    a = availability.reindex(timestamps)[columns]
    return v.where(a.notna() & a.le(origin))


def covariate(name, columns, values, availability, timestamps, origin):
    v = visible(values, availability, columns, timestamps, origin)
    if name == "residual_load_da":
        result = v.iloc[:, 0] - v.iloc[:, 1] - v.iloc[:, 2]
    elif name in ("wind_error", "scheduled_mfrr_spread"):
        result = v.iloc[:, 0] - v.iloc[:, 1]
    elif name == "frequency_deviation":
        result = v.iloc[:, 0] - 50
    else:
        result = v.iloc[:, 0]
    return result.to_numpy(dtype="float32")


def calendar(timestamps):
    local = timestamps.tz_convert("Europe/Helsinki")
    hour = local.hour + local.minute / 60
    return {
        "hour_sin": np.asarray(np.sin(2 * np.pi * hour / 24), dtype="float32"),
        "hour_cos": np.asarray(np.cos(2 * np.pi * hour / 24), dtype="float32"),
        "weekend": np.asarray(local.day_of_week >= 5, dtype="float32"),
    }


def make_input(values, availability, origin, mode, context_length):
    sources = ["fi_imbalance_price"] + ([] if mode == "price" else ["fi_spot_price"])
    known = availability[sources].notna().all(axis=1) & availability[sources].le(origin).all(axis=1)
    known &= values.index < origin
    end = values.index[known][-1]
    history = pd.date_range(end=end, periods=context_length, freq="15min")
    gap = int((origin - end) / QUARTER) - 1
    assert gap >= 0
    future = pd.date_range(end + QUARTER, periods=gap + 40, freq="15min")
    target = visible(values, availability, sources, history, origin)
    target = target.iloc[:, 0] if mode == "price" else target.iloc[:, 0] - target.iloc[:, 1]
    task = {"target": target.to_numpy(dtype="float32")}
    if mode == "spread_covariates":
        past = {name: covariate(name, cols, values, availability, history, origin)
                for name, cols in {**KNOWN, **PAST}.items()}
        future_cov = {name: covariate(name, cols, values, availability, future, origin)
                      for name, cols in KNOWN.items()}
        past.update(calendar(history))
        future_cov.update(calendar(future))
        task.update(past_covariates=past, future_covariates=future_cov)
    spot = visible(values, availability, ["fi_spot_price"], future[gap:], origin).iloc[:, 0].to_numpy()
    assert np.isfinite(spot).all(), f"Future spot not available at {origin}"
    assert np.isfinite(task["target"]).sum() >= context_length // 2
    return task, gap, spot, end


def check_availability(values, availability, context_length):
    """Unusable-value mutation must leave all model inputs exactly unchanged."""
    checks = []
    for cutoff in pd.to_datetime(["2026-09-06T21:00Z", "2026-09-20T14:00Z", "2026-10-04T20:00Z"]):
        changed = values.mask(availability.gt(cutoff) | availability.isna(), 987654.0)
        for mode in MODES:
            a, ga, sa, ea = make_input(values, availability, cutoff, mode, context_length)
            b, gb, sb, eb = make_input(changed, availability, cutoff, mode, context_length)
            assert ga == gb and ea == eb
            np.testing.assert_array_equal(sa, sb)
            for key in a:
                if isinstance(a[key], dict):
                    for c in a[key]:
                        np.testing.assert_array_equal(a[key][c], b[key][c])
                else:
                    np.testing.assert_array_equal(a[key], b[key])
            checks.append({"origin": str(cutoff), "mode": mode, "passed": True})
    return checks


def metrics(frame):
    frame = frame.dropna(subset=["imbalance_price"])
    y = frame.imbalance_price
    return {
        "rows_scored": len(frame),
        "mae_p50": float((y - frame.p50).abs().mean()),
        "pinball_p10": pinball(y, frame.p10, 0.1),
        "pinball_p90": pinball(y, frame.p90, 0.9),
        "calibration_pinball_mean": (pinball(y, frame.p10, 0.1) + pinball(y, frame.p90, 0.9)) / 2,
        "coverage_p10_p90": float(((y >= frame.p10) & (y <= frame.p90)).mean()),
        "below_p10": float((y < frame.p10).mean()),
        "above_p90": float((y > frame.p90).mean()),
        "mean_interval_width": float((frame.p90 - frame.p10).mean()),
    }


def evaluate(forecasts, actuals):
    joined = forecasts.merge(actuals, on="delivery_start_utc", how="left", validate="many_to_one")
    joined["horizon_hour"] = (joined.step - 1) // 4 + 1
    by_horizon = pd.DataFrame([{ "horizon_hour": hour, **metrics(group)}
                               for hour, group in joined.groupby("horizon_hour")])
    return metrics(joined), by_horizon


def run(args):
    import torch
    from chronos import Chronos2Pipeline

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    original_hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in (
        ROOT / "forecast_backtest.csv", ROOT / "forecast_backtest_packageonly.csv",
        ROOT / "submission/validation_forecast_B.csv")}
    values, availability = pf.load_package()
    values, availability = pf.load_external(values, availability)
    checks = check_availability(values, availability, args.context_length)
    print(f"Availability checks: {len(checks)} passed", flush=True)
    template = pd.read_csv(ROOT / "baseline_output/validation/sample_submission.csv")
    template["forecast_origin_utc"] = pd.to_datetime(template.forecast_origin_utc, utc=True)
    template["delivery_start_utc"] = pd.to_datetime(template.delivery_start_utc, utc=True)
    origins = pd.DatetimeIndex(template.forecast_origin_utc.drop_duplicates())
    if args.limit:
        origins = origins[:args.limit]
    assert origins.min() >= pd.Timestamp("2026-09-06T21:00Z")
    assert origins.max() < pd.Timestamp("2026-10-04T21:00Z")

    torch.manual_seed(7)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Chronos-2 on {device}; {len(origins)} origins, context {args.context_length}", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(args.model_path, device_map=device,
                                                dtype=torch.float32, local_files_only=True)
    actuals = pd.read_csv(ROOT / "data/actuals_validation.csv", usecols=["delivery_start_utc", "imbalance_price"])
    actuals["delivery_start_utc"] = pd.to_datetime(actuals.delivery_start_utc, utc=True)
    summary = []
    for name, path in [("baseline", ROOT / "baseline_output/validation/forecast_backtest.csv"),
                       ("lightgbm_A", ROOT / "submission/validation_forecast_A.csv"),
                       ("lightgbm_B", ROOT / "submission/validation_forecast_B.csv")]:
        f = pd.read_csv(path)
        for c in ("forecast_origin_utc", "delivery_start_utc"):
            f[c] = pd.to_datetime(f[c], utc=True)
        f = f[f.forecast_origin_utc.isin(origins)]
        result, horizon = evaluate(f, actuals)
        summary.append({"model": name, **result})
        horizon.to_csv(out / f"horizon_{name}.csv", index=False)

    timings = {}
    bridge_counts = {}
    for mode in args.modes.split(","):
        assert mode in MODES, mode
        started = time.monotonic()
        records = []
        for start in range(0, len(origins), args.origin_batch_size):
            batch_origins = origins[start:start + args.origin_batch_size]
            prepared = [make_input(values, availability, o, mode, args.context_length) for o in batch_origins]
            gaps = [x[1] for x in prepared]
            assert len(set(gaps)) == 1, "Batch has inconsistent publication-delay bridges"
            gap = gaps[0]
            bridge_counts[mode] = gap
            with torch.inference_mode():
                q, _ = pipeline.predict_quantiles(
                    inputs=[x[0] for x in prepared], prediction_length=gap + 40,
                    quantile_levels=[0.1, 0.5, 0.9], batch_size=args.batch_size,
                    context_length=args.context_length, cross_learning=False,
                )
            for origin, prediction, (_, _, spot, end) in zip(batch_origins, q, prepared):
                quantiles = prediction[0, gap:gap + 40, :].cpu().numpy().astype(float)
                if mode != "price":
                    quantiles += spot[:, None]
                quantiles = np.sort(quantiles, axis=1)
                assert quantiles.shape == (40, 3) and np.isfinite(quantiles).all()
                f = pd.DataFrame({"forecast_origin_utc": origin,
                                  "delivery_start_utc": pd.date_range(origin, periods=40, freq="15min"),
                                  "step": np.arange(1, 41), "p10": quantiles[:, 0],
                                  "p50": quantiles[:, 1], "p90": quantiles[:, 2],
                                  "last_observed_target_utc": end})
                records.append(f)
            done = min(start + args.origin_batch_size, len(origins))
            if start == 0 or done % 64 == 0 or done == len(origins):
                print(f"{mode}: {done}/{len(origins)} origins, {time.monotonic() - started:.1f}s", flush=True)
        forecast = pd.concat(records, ignore_index=True)
        identifiers = ["forecast_origin_utc", "delivery_start_utc", "step"]
        expected = template[template.forecast_origin_utc.isin(origins)][identifiers].reset_index(drop=True)
        pd.testing.assert_frame_equal(forecast[identifiers], expected, check_dtype=False)
        forecast.to_csv(out / f"validation_chronos2_{mode}.csv", index=False)
        result, horizon = evaluate(forecast, actuals)
        timings[mode] = time.monotonic() - started
        summary.append({"model": f"chronos2_{mode}", **result})
        horizon.to_csv(out / f"horizon_chronos2_{mode}.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "comparison.csv", index=False)
        print(mode, json.dumps(result), flush=True)

    for name, digest in original_hashes.items():
        assert fingerprint(ROOT / name) == digest, f"Original artifact changed: {name}"
    metadata = {
        "experiment": 2, "model_id": "amazon/chronos-2", "model_path": args.model_path,
        "model_download": json.loads((Path(args.model_path) / "download_metadata.json").read_text()),
        "device": device, "context_quarters": args.context_length,
        "prediction_quarters": 40, "delay_bridge_quarters": bridge_counts,
        "origins": len(origins), "first_origin": str(origins.min()), "last_origin": str(origins.max()),
        "parameter_updates": 0, "cross_learning": False, "availability_checks": checks,
        "past_covariates": PAST, "known_covariates": KNOWN, "calendar_covariates": list(calendar(origins)),
        "runtime_seconds": timings, "original_artifact_sha256": original_hashes,
        "packages": {p: importlib.metadata.version(p) for p in
                     ["chronos-forecasting", "torch", "transformers", "numpy", "pandas", "pyarrow"]},
        "scope": "Price quantiles only; no direction or spike probability head; no test-period scoring.",
    }
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(ROOT / "work_private/models/chronos-2"))
    parser.add_argument("--output-dir", default="results/experiment_2_chronos")
    parser.add_argument("--context-length", type=int, default=672, help="Seven days of 15-minute history")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--origin-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run only; zero evaluates all origins")
    run(parser.parse_args())
