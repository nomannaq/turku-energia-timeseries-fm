"""Fingrid open data API. Requires env FINGRID_API_KEY (never printed). 1 req / 2.1 s, paginated, raw JSON cached.
Leakage notes (assumed until starter availability table): realtime 1/3-min data usable at end_time + 15 min;
15-min datasets (369, 319, 75, 124) usable at end_time + 30 min; 246/165 day-ahead forecasts usable from 12:00 EET (10:00 UTC)
of the previous day (conservative: D-1 11:00 UTC); 245 is overwritten every 15 min (no archive) -> NOT usable as forecast feature."""
import os, sys, json, time, requests, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw/fingrid"; RAW.mkdir(parents=True, exist_ok=True)
H = {"x-api-key": os.environ["FINGRID_API_KEY"]}
IDS = [int(x) for x in sys.argv[1:]] or [369, 319, 75, 246, 165, 124, 245, 400, 244, 260, 188, 181, 194, 177, 397]
START, END = "2026-05-25T00:00:00Z", "2026-11-02T00:00:00Z"
def get(params, i):
    for a in range(6):
        r = requests.get(f"https://data.fingrid.fi/api/datasets/{i}/data", headers=H, params=params, timeout=120); time.sleep(2.1)
        if r.status_code == 200: return r.json()
        print(i, params.get("page"), r.status_code, r.text[:150], file=sys.stderr); time.sleep(5 * (a + 1))
    raise RuntimeError(i)
for i in IDS:
    page, rows, last = 1, [], None
    while True:
        f = RAW / f"{i}_p{page}.json"
        if f.exists(): d = json.loads(f.read_text())
        else:
            d = get(dict(startTime=START, endTime=END, format="json", pageSize=20000, page=page, sortOrder="asc"), i); f.write_text(json.dumps(d))
        rows += d["data"]
        if last is None: last = d.get("pagination", {}).get("lastPage") or 1
        if page >= last: break
        page += 1
    df = pd.DataFrame(rows)
    for c in ("startTime", "endTime"): df[c] = pd.to_datetime(df[c], utc=True)
    df.to_parquet(ROOT / f"data/fingrid_{i}.parquet", index=False)
    print(i, len(df), df.startTime.min(), df.startTime.max(), flush=True)
