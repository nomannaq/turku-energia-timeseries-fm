"""FMI open-data station observations (10-min). Leakage rule (assumed): first_usable_utc = obs_time + 30 min."""
import re, time, sys, requests, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw/fmi"; RAW.mkdir(parents=True, exist_ok=True)
PLACES = ["Vaasa", "Kaskinen", "Pori", "Raahe", "Oulu", "Kemi", "Helsinki", "Tampere"]
URL = "https://opendata.fmi.fi/wfs"
pat = re.compile(r"<gml:pos>([\d.]+) ([\d.]+) </gml:pos>.*?<BsWfs:Time>(.*?)</BsWfs:Time>.*?<BsWfs:ParameterName>(.*?)</BsWfs:ParameterName>.*?<BsWfs:ParameterValue>(.*?)</BsWfs:ParameterValue>", re.S)
rows = []
for place in PLACES:
    for s in pd.date_range("2026-05-25", "2026-11-01", freq="7D", tz="UTC"):
        e = min(s + pd.Timedelta("7D") - pd.Timedelta("10min"), pd.Timestamp("2026-11-01 23:50", tz="UTC"))
        f = RAW / f"{place}_{s:%Y%m%d}.xml"
        if not f.exists():
            p = dict(service="WFS", version="2.0.0", request="getFeature", storedquery_id="fmi::observations::weather::simple",
                     place=place, parameters="ws_10min,wd_10min,t2m", starttime=f"{s:%Y-%m-%dT%H:%M:%SZ}", endtime=f"{e:%Y-%m-%dT%H:%M:%SZ}", timestep=10)
            r = requests.get(URL, params=p, timeout=120)
            if r.status_code != 200: print(place, s, r.status_code, r.text[:200], file=sys.stderr); continue
            f.write_text(r.text); time.sleep(0.5)
        for lat, lon, t, par, v in pat.findall(f.read_text()):
            rows.append((place.lower(), float(lat), float(lon), t, par, float(v) if v != "NaN" else None))
df = pd.DataFrame(rows, columns=["station", "lat", "lon", "obs_time_utc", "param", "value"])
df["obs_time_utc"] = pd.to_datetime(df.obs_time_utc, utc=True)
df = df.pivot_table(index=["station", "lat", "lon", "obs_time_utc"], columns="param", values="value").reset_index()
df["first_usable_utc"] = df.obs_time_utc + pd.Timedelta("30min")
df.to_parquet(ROOT / "data/fmi_obs_10min.parquet", index=False); print(df.shape); print(df.groupby("station").agg(n=("obs_time_utc","size"), t0=("obs_time_utc","min"), t1=("obs_time_utc","max")))
