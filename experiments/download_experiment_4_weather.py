"""Cache individual archived weather runs for the additional-data experiment.

No project observations are sent: requests contain public coordinates and dates.
Three existing wind-region locations keep this benchmark below the free API's
5,000 weighted calls/hour. Missing runs remain missing; no seamless replacement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "work_private/experiment4/weather_raw"
LOCS = {"raahe": (64.68, 24.48), "oulu": (65.01, 25.47), "vaasa": (63.10, 21.60)}
MODELS = {"dmi_harmonie_arome_europe": "3h", "icon_eu": "6h"}
URL = "https://single-runs-api.open-meteo.com/v1/forecast"
VARS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m"]
lock = threading.Lock()
last_request = 0.0


def fetch(job):
    global last_request
    model, run = job
    path = RAW / model / f"{run:%Y%m%d%H}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    if path.with_suffix(".missing").exists():
        return None
    params = dict(latitude=",".join(str(v[0]) for v in LOCS.values()),
                  longitude=",".join(str(v[1]) for v in LOCS.values()),
                  hourly=",".join(VARS), models=model, run=f"{run:%Y-%m-%dT%H:%M}",
                  forecast_hours=30, timezone="GMT", wind_speed_unit="ms")
    for attempt in range(3):
        with lock:
            time.sleep(max(0, 0.5 - (time.monotonic() - last_request)))
            last_request = time.monotonic()
        try:
            response = requests.get(URL, params=params, timeout=45)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 429:
            raise RuntimeError("API rate limit reached; stop and resume later from cached files.")
        if response.ok:
            blocks = response.json()
            assert isinstance(blocks, list) and len(blocks) == len(LOCS)
            for block in blocks:
                assert block["hourly"]["time"][0] == f"{run:%Y-%m-%dT%H:%M}"
                assert block["hourly_units"]["wind_speed_100m"] == "m/s"
            path.write_text(json.dumps(blocks))
            return path
        if response.status_code == 400:
            path.with_suffix(".missing").write_text(response.text)
            return None
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Weather fetch failed: {model} {run} HTTP {response.status_code}")


def main(limit=0):
    jobs = [(model, run) for model, freq in MODELS.items()
            for run in pd.date_range("2026-05-31T00:00Z", "2026-10-04T21:00Z", freq=freq)]
    if limit:
        jobs = [j for model in MODELS for j in [j for j in jobs if j[0] == model][:limit]]
    rows, manifest = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for n, (job, path) in enumerate(zip(jobs, pool.map(fetch, jobs)), 1):
            model, run = job
            if path is not None:
                blocks = json.loads(path.read_text())
                for name, block in zip(LOCS, blocks):
                    frame = pd.DataFrame(block["hourly"])
                    frame["location"] = name
                    frame["model"] = model
                    frame["run_time_utc"] = run
                    frame["valid_time_utc"] = pd.to_datetime(frame.pop("time"), utc=True)
                    rows.append(frame)
                manifest.append({"model": model, "run_time_utc": str(run),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if n == 1 or n % 64 == 0 or n == len(jobs):
                print(f"Archived runs: {n}/{len(jobs)}, available {len(manifest)}", flush=True)
    out = ROOT / "work_private/experiment4"
    suffix = "_smoke" if limit else ""
    pd.concat(rows, ignore_index=True).to_parquet(out / f"weather_runs{suffix}.parquet", index=False)
    (out / f"weather_manifest{suffix}.json").write_text(json.dumps(manifest, indent=2))
    print("Weather download complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args().limit)
