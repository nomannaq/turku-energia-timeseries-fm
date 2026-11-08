"""Fetch archived ECMWF IFS single runs from Open-Meteo for Finnish wind/load points.
Leakage rule: first_usable_utc = run_time + 6h (conservative; ECMWF 00z data typically available ~+4-6h)."""
import json, time, os, sys, requests, pandas as pd
from pathlib import Path
RAW = Path(__file__).resolve().parents[1] / "data/raw/openmeteo"; RAW.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parents[1] / "data/openmeteo_ecmwf_runs.parquet"
LOCS = {  # name: (lat, lon)
    "vaasa": (63.10, 21.60), "kristiinankaupunki": (62.27, 21.37), "pori": (61.48, 21.70),
    "raahe": (64.68, 24.48), "oulu": (65.01, 25.47), "kemi_tornio": (65.75, 24.55),
    "helsinki": (60.17, 24.94), "tampere": (61.50, 23.76)}
VARS = "wind_speed_100m,wind_direction_100m,temperature_2m,shortwave_radiation,wind_speed_10m"
URL = "https://single-runs-api.open-meteo.com/v1/forecast"
def fetch(run):
    f = RAW / f"{run:%Y%m%d%H}.json"
    if f.exists(): return json.loads(f.read_text())
    if (RAW / f"{run:%Y%m%d%H}.missing").exists(): return None
    p = dict(latitude=",".join(str(v[0]) for v in LOCS.values()), longitude=",".join(str(v[1]) for v in LOCS.values()),
             hourly=VARS, models="ecmwf_ifs", run=f"{run:%Y-%m-%dT%H:%M}", forecast_hours=30, timezone="GMT", wind_speed_unit="ms")
    for a in range(5):
        try: r = requests.get(URL, params=p, timeout=90)
        except requests.RequestException as ex:
            print(run, "net error", ex, file=sys.stderr); time.sleep(10 * (a + 1)); continue
        if r.status_code == 200:
            try: d = r.json()
            except ValueError: print(run, "bad json", file=sys.stderr); time.sleep(10); continue
            f.write_text(json.dumps(d)); return d
        print(run, r.status_code, r.text[:200], file=sys.stderr)
        if r.status_code == 400 and "not available" in r.text:
            (RAW / f"{run:%Y%m%d%H}.missing").write_text(r.text); return None
        time.sleep(10 * (a + 1))
    return None
rows = []
for run in pd.date_range("2026-05-25", "2026-11-01 18:00", freq="6h"):
    d = fetch(run)
    if d is None: continue
    d = d if isinstance(d, list) else [d]
    for name, blk in zip(LOCS, d):
        h = pd.DataFrame(blk["hourly"]); h["location"] = name; h["run_time_utc"] = run; rows.append(h)
    time.sleep(0.3)
df = pd.concat(rows); df["valid_time_utc"] = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC")
df["run_time_utc"] = df["run_time_utc"].dt.tz_localize("UTC")
df["first_usable_utc"] = df["run_time_utc"] + pd.Timedelta("6h")
df["lead_h"] = (df.valid_time_utc - df.run_time_utc) / pd.Timedelta("1h")
df.to_parquet(OUT, index=False); print(df.shape, df.isna().mean().round(3).to_dict())
