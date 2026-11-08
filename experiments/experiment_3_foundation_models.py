"""Research-only zero-shot TimesFM 3 and Timestral benchmarks; never writes submissions."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiment_2_chronos import (
    ROOT, MODES, pf, make_input, check_availability, evaluate, fingerprint,
)


def covariates(tasks):
    """Separate past-only channels from channels with known future values."""
    if "past_covariates" not in tasks[0]:
        return None, None
    known = list(tasks[0]["future_covariates"])
    past = [k for k in tasks[0]["past_covariates"] if k not in known]
    return (
        np.stack([np.stack([t["past_covariates"][k] for k in past]) for t in tasks]),
        np.stack([np.stack([np.concatenate((t["past_covariates"][k],
                                           t["future_covariates"][k])) for k in known]) for t in tasks]),
    )


def predict(model, name, tasks, horizon):
    past, known = covariates(tasks)
    with torch.inference_mode():
        if name == "timesfm3":
            predictions = list(model.predict_batch(
                contexts=[t["target"] for t in tasks], horizon=horizon,
                past_only_covariates=None if past is None else list(past),
                past_future_covariates=None if known is None else list(known),
                return_quantiles=True, use_symmetric_averaging=True, make_positive=False,
            ))
            return np.stack([p.quantiles[..., [0, 4, 8]] for p in predictions])
        predictions = model.forecast(
            np.stack([t["target"] for t in tasks]), prediction_length=horizon,
            past_covariates=past, known_covariates=known,
            symmetric_averaging=True, clip_nonnegative=False,
        )
        return torch.stack([predictions.quantile(q)[:, 0] for q in (0.1, 0.5, 0.9)], -1).cpu().numpy()


def run(args):
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ["forecast_backtest.csv", "forecast_backtest_packageonly.csv",
                 "submission/validation_forecast_B.csv"]
    hashes = {p: fingerprint(ROOT / p) for p in artifacts}
    values, availability = pf.load_package()
    values, availability = pf.load_external(values, availability)
    checks = check_availability(values, availability, args.context_length)
    print(f"Availability checks: {len(checks)} passed", flush=True)
    template = pd.read_csv(ROOT / "baseline_output/validation/sample_submission.csv")
    for c in ("forecast_origin_utc", "delivery_start_utc"):
        template[c] = pd.to_datetime(template[c], utc=True)
    origins = pd.DatetimeIndex(template.forecast_origin_utc.drop_duplicates())
    if args.limit:
        origins = origins[:args.limit]
    assert origins.min() >= pd.Timestamp("2026-09-06T21:00Z")
    assert origins.max() < pd.Timestamp("2026-10-04T21:00Z")
    actuals = pd.read_csv(ROOT / "data/actuals_validation.csv",
                         usecols=["delivery_start_utc", "imbalance_price"])
    actuals.delivery_start_utc = pd.to_datetime(actuals.delivery_start_utc, utc=True)
    summary = []
    # Recompute reference metrics on precisely the same origins, including smoke runs.
    references = {"baseline": "baseline_output/validation/forecast_backtest.csv",
                  "lightgbm_B": "submission/validation_forecast_B.csv"}
    references.update({f"chronos2_{m}": f"results/experiment_2_chronos/validation_chronos2_{m}.csv" for m in MODES})
    for name, path in references.items():
        f = pd.read_csv(ROOT / path)
        for c in ("forecast_origin_utc", "delivery_start_utc"):
            f[c] = pd.to_datetime(f[c], utc=True)
        result, _ = evaluate(f[f.forecast_origin_utc.isin(origins)], actuals)
        summary.append({"model": name, **result})
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    metadata = {"experiment": 3, "device": "cpu", "origins": len(origins),
                "first_origin": str(origins.min()), "last_origin": str(origins.max()),
                "context_quarters": args.context_length, "parameter_updates": 0,
                "symmetric_averaging": True, "nonnegative_clipping": False,
                "availability_checks": checks, "original_artifact_sha256": hashes,
                "models": {}, "runtime_seconds": {}, "batch_independence_checks": {},
                "scope": "Research price benchmark only; no direction/spike probabilities or test scoring.",
                "packages": {p: importlib.metadata.version(p) for p in
                             ["timesfm", "timestral", "torch", "numpy", "pandas"]}}
    if args.append:
        previous = json.loads((out / "run.json").read_text())
        for key in ("origins", "first_origin", "last_origin", "context_quarters",
                    "symmetric_averaging", "nonnegative_clipping", "original_artifact_sha256"):
            assert previous[key] == metadata[key], f"Incompatible previous run: {key}"
        for key in ("models", "runtime_seconds", "batch_independence_checks"):
            metadata[key].update(previous[key])
        previous_scores = pd.read_csv(out / "comparison.csv")
        summary.extend(previous_scores[previous_scores.model.str.startswith(("timesfm3_", "timestral209m_"))].to_dict("records"))
    for name in args.models.split(","):
        model_path = ROOT / "work_private/models" / {"timesfm3": "timesfm-3", "timestral209m": "timestral-209m"}[name]
        metadata["models"][name] = json.loads((model_path / "download_metadata.json").read_text())
        if name == "timesfm3":
            from timesfm3 import TimesFM3Forecaster, ModelConfig
            model = TimesFM3Forecaster(ModelConfig(checkpoint_path=str(model_path), device="cpu",
                                                   per_core_batch_size=args.batch_size, local_files_only=True))
        else:
            from timestral import TimestralForecaster
            model = TimestralForecaster.from_pretrained(model_path, device="cpu", local_files_only=True)
        for mode in args.modes.split(","):
            assert mode in MODES
            label = f"{name}_{mode}"
            started = time.monotonic()
            records = []
            for start in range(0, len(origins), args.batch_size):
                batch = origins[start:start + args.batch_size]
                prepared = [make_input(values, availability, o, mode, args.context_length) for o in batch]
                tasks = [p[0] for p in prepared]
                gaps = [p[1] for p in prepared]
                assert len(set(gaps)) == 1
                gap = gaps[0]
                quantiles = predict(model, name, tasks, gap + 40)
                if start == 0 and len(tasks) > 1:
                    solo = predict(model, name, tasks[:1], gap + 40)
                    np.testing.assert_allclose(quantiles[:1], solo, rtol=1e-4, atol=0.002)
                    metadata["batch_independence_checks"][label] = True
                assert quantiles.shape == (len(batch), gap + 40, 3), quantiles.shape
                for origin, q, (_, _, spot, end) in zip(batch, quantiles, prepared):
                    q = q[gap:gap + 40].astype(float)
                    if mode != "price":
                        q += spot[:, None]
                    q = np.sort(q, axis=-1)
                    assert np.isfinite(q).all()
                    records.append(pd.DataFrame({"forecast_origin_utc": origin,
                        "delivery_start_utc": pd.date_range(origin, periods=40, freq="15min"),
                        "step": np.arange(1, 41), "p10": q[:, 0], "p50": q[:, 1], "p90": q[:, 2],
                        "last_observed_target_utc": end}))
                done = min(start + args.batch_size, len(origins))
                if start == 0 or done % 64 == 0 or done == len(origins):
                    print(f"{label}: {done}/{len(origins)}, {time.monotonic()-started:.1f}s", flush=True)
            forecast = pd.concat(records, ignore_index=True)
            keys = ["forecast_origin_utc", "delivery_start_utc", "step"]
            expected = template[template.forecast_origin_utc.isin(origins)][keys].reset_index(drop=True)
            pd.testing.assert_frame_equal(forecast[keys], expected, check_dtype=False)
            forecast.to_csv(out / f"validation_{label}.csv", index=False)
            result, horizon = evaluate(forecast, actuals)
            summary.append({"model": label, **result})
            horizon.to_csv(out / f"horizon_{label}.csv", index=False)
            metadata["runtime_seconds"][label] = time.monotonic() - started
            metadata["delay_bridge_quarters"] = gap
            pd.DataFrame(summary).drop_duplicates("model", keep="last").to_csv(out / "comparison.csv", index=False)
            (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(label, json.dumps(result), flush=True)
        del model
        gc.collect()
    for path, digest in hashes.items():
        assert fingerprint(ROOT / path) == digest
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/experiment_3_foundation_models")
    parser.add_argument("--models", default="timesfm3,timestral209m")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--context-length", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="Append a model to a compatible completed run")
    run(parser.parse_args())
