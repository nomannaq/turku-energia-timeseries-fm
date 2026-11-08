"""CPU Chronos-2 forecasts for training folds, validation and test; resumable batches."""
import argparse
import importlib.metadata
import json
import os
import time

from ensemble_spec import ROOT, WORK, OUT, MODEL, PERIODS, fingerprint, guard, save_json
os.environ.setdefault("HF_HOME", str(ROOT / "work_private/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from fm_data import CONTEXT, HORIZON, BRIDGE, FUTURE, PAST, load_sources, make_input


def load_pipeline(threads):
    assert importlib.metadata.version("chronos-forecasting") == "2.3.2"
    torch.set_num_threads(threads)
    torch.manual_seed(7)
    return Chronos2Pipeline.from_pretrained(str(MODEL), device_map="cpu", dtype=torch.float32, local_files_only=True)


def predict(pipeline, tasks):
    with torch.inference_mode():
        q, _ = pipeline.predict_quantiles(inputs=tasks, prediction_length=HORIZON,
            quantile_levels=[.1, .5, .9], context_length=CONTEXT, batch_size=64, cross_learning=False)
    pred = np.stack([x[0, BRIDGE:BRIDGE+40, :].cpu().numpy().astype(float) for x in q])
    assert pred.shape == (len(tasks), 40, 3) and np.isfinite(pred).all()
    return pred


def main(args):
    (WORK / "fm_batches").mkdir(parents=True, exist_ok=True)
    values, availability = load_sources()
    protocol = dict(model=json.loads((MODEL / "download_metadata.json").read_text()),
        model_sha256=fingerprint(MODEL / "model.safetensors"), context=CONTEXT, horizon=HORIZON, bridge=BRIDGE,
        past=PAST, future=FUTURE, wind_error="x_wind_rt - fi_wind_fc_da", batch_size=64,
        origin_batch_size=args.origin_batch_size, cross_learning=False, threads=args.threads,
        packages={p: importlib.metadata.version(p) for p in ["chronos-forecasting", "torch", "numpy", "pandas"]},
        input_hashes={str(p.relative_to(ROOT)): fingerprint(p) for p in
            [ROOT / "data/inputs.parquet", ROOT / "data/spot_nordpool.csv", *sorted((ROOT / "data").glob("fingrid_*.parquet"))]})
    guard(WORK / "fm_protocol.json", protocol)
    pipeline = load_pipeline(args.threads)
    started = time.monotonic()
    for period in args.periods.split(","):
        start, stop = PERIODS[period]
        origins = pd.date_range(start, stop, freq="h", inclusive="left")
        if args.limit:
            origins = origins[:args.limit]
        pieces, filled_total = [], 0
        for n in range(0, len(origins), args.origin_batch_size):
            batch = origins[n:n+args.origin_batch_size]
            cache = WORK / f"fm_batches/{period}_{n}.npz"
            tasks = [make_input(values, availability, o) for o in batch]
            filled_total += sum(t[2] for t in tasks)
            if cache.exists():
                with np.load(cache) as stored:
                    np.testing.assert_array_equal(stored["origins"], batch.asi8)
                    q = stored["quantiles"]
            else:
                q = predict(pipeline, [t[0] for t in tasks])
                np.savez_compressed(cache, origins=batch.asi8, quantiles=q)
            if n == 0:
                solo = predict(pipeline, [tasks[0][0]])[0]
                np.testing.assert_allclose(q[0], solo, rtol=1e-5, atol=1e-4)
                save_json(OUT / f"fm_batch_check_{period}.json", dict(max_abs_difference=float(np.max(np.abs(q[0]-solo))), passed=True))
            for origin, pred, (_, spot, _) in zip(batch, q, tasks):
                pieces.append(pd.DataFrame(dict(forecast_origin_utc=origin,
                    delivery_start_utc=pd.date_range(origin, periods=40, freq="15min"), step=np.arange(1, 41),
                    q10=pred[:, 0], q50=pred[:, 1], q90=pred[:, 2], spot=spot)))
            print(f"Chronos {period}: {min(n+len(batch), len(origins))}/{len(origins)} origins, {time.monotonic()-started:.0f}s", flush=True)
        frame = pd.concat(pieces, ignore_index=True)
        suffix = "_smoke" if args.limit else ""
        frame.to_parquet(WORK / f"chronos_{period}{suffix}.parquet", index=False)
        save_json(OUT / f"fm_audit_{period}{suffix}.json", dict(origins=len(origins), rows=len(frame),
            forward_filled_future_values=filled_total, first_origin=origins[0], last_origin=origins[-1]))
    save_json(OUT / ("fm_smoke_completed.json" if args.limit else "fm_completed.json"),
        dict(runtime_seconds=time.monotonic()-started, **protocol))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--origin-batch-size", type=int, default=16)
    p.add_argument("--periods", default=",".join(PERIODS))
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
