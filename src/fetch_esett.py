"""eSett Open Data (no key): single imbalance prices (EXP14), imbalance volumes (EXP13), production by type (EXP16), consumption (EXP15).
Leakage (assumed, document): EXP14 price/direction for quarter q usable at q_end + 2h (conservative; preliminary prices
are typically public within ~1h). EXP13/15/16 settlement volumes: assume usable only 14 days after delivery (settlement lag) ->
use only for analysis/labels, NOT as live features unless the starter package availability table says otherwise."""
import json, sys, time, requests, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw/esett"; RAW.mkdir(parents=True, exist_ok=True)
MBA = {"FI": "10YFI_1________U", "SE1": "10Y1001A1001A44P", "SE3": "10Y1001A1001A46L"}
EXPS = {"EXP14/Prices": list(MBA), "EXP13/ImbalancePowerVolume": ["FI"], "EXP16/Volumes": ["FI"], "EXP15/Consumption": ["FI"]}
for exp, mbas in EXPS.items():
    out = []
    for m in mbas:
        for s in pd.date_range("2026-05-25", "2026-11-02", freq="7D", tz="UTC"):
            e = s + pd.Timedelta("7D")
            f = RAW / f"{exp.replace('/','_')}_{m}_{s:%Y%m%d}.json"
            if not f.exists():
                r = requests.get(f"https://api.opendata.esett.com/{exp}", params=dict(mba=MBA[m], start=f"{s:%Y-%m-%dT%H:%M:%S.000Z}", end=f"{e:%Y-%m-%dT%H:%M:%S.000Z}"), timeout=120)
                if r.status_code != 200: print(exp, m, s, r.status_code, r.text[:200], file=sys.stderr); continue
                f.write_text(r.text); time.sleep(0.3)
            d = json.loads(f.read_text())
            if d: df = pd.json_normalize(d); df["mba_code"] = m; out.append(df)
    if out:
        df = pd.concat(out).drop_duplicates()
        df.to_parquet(ROOT / f"data/esett_{exp.split('/')[0]}.parquet", index=False)
        print(exp, df.shape, list(df.columns)[:20], df.get("timestampUTC", pd.Series()).min(), df.get("timestampUTC", pd.Series()).max())
