"""Data audit: gaps, duplicates, resolution, DST, outliers, revisions, aFRR missingness. Writes results/audit_*.csv + prints summary."""
import numpy as np, pandas as pd, glob
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data"; R = D.parent / "results"
rows = []
for f in sorted(glob.glob(str(D / "fingrid_*.parquet"))):
    i = int(Path(f).stem.split("_")[1]); d = pd.read_parquet(f)
    st = d.startTime.sort_values(); dt = st.diff().dropna()
    res = dt.mode().iloc[0]; dup = st.duplicated().sum()
    gaps = dt[dt > res * 1.5]
    v = d.value.astype(float)
    z = (v - v.median()).abs() / (1.4826 * (v - v.median()).abs().median() + 1e-9)
    rows.append(dict(series=f"fingrid_{i}", n=len(d), resolution=str(res), sec_offset=int(st.dt.second.mode().iloc[0]), dups=int(dup), gaps=len(gaps),
                     max_gap=str(gaps.max()) if len(gaps) else "", missing_val=int(v.isna().sum()), min=v.min(), p50=v.median(), max=v.max(), robust_z_gt10=int((z > 10).sum()),
                     constant_runs_gt2h=int((v.diff().eq(0).astype(int).groupby(v.diff().ne(0).cumsum()).transform("size") > (pd.Timedelta("2h") / res)).sum())))
e = pd.read_parquet(D / "esett_EXP14.parquet")
for m, g in e.groupby("mba_code"):
    t = pd.to_datetime(g.timestampUTC, utc=True).sort_values(); dt = t.diff().dropna()
    loc = pd.to_datetime(g.timestamp); off = (loc.values - t.dt.tz_localize(None).values) / np.timedelta64(1, "h")
    rows.append(dict(series=f"esett14_{m}", n=len(g), resolution=str(dt.mode().iloc[0]), dups=int(t.duplicated().sum()), gaps=int((dt > pd.Timedelta("15min")).sum()),
                     missing_val=int(g.imblSalesPrice.isna().sum()), min=g.imblSalesPrice.min(), p50=g.imblSalesPrice.median(), max=g.imblSalesPrice.max(),
                     note=f"local-UTC offsets seen: {sorted(set(np.round(off,2)))}; aFRR up price null {g.upRegPriceFrrA.isna().mean():.1%}, down null {g.downRegPriceFrrA.isna().mean():.1%}, purchase!=sales {(g.imblSalesPrice!=g.imblPurchasePrice).mean():.2%}"))
for name in ["openmeteo_ecmwf_runs", "openmeteo_ecmwf_runs_neighbours"]:
    w = pd.read_parquet(D / f"{name}.parquet"); runs = w.run_time_utc.drop_duplicates().sort_values()
    rows.append(dict(series=name, n=len(w), resolution="1h, runs 6h", dups=int(w.duplicated(["location", "run_time_utc", "valid_time_utc"]).sum()), gaps=int((runs.diff() > pd.Timedelta("6h")).sum()),
                     missing_val=int(w.wind_speed_100m.isna().sum()), min=w.wind_speed_100m.min(), p50=w.wind_speed_100m.median(), max=w.wind_speed_100m.max(), note=f"missing runs: {[str(x) for x in runs[runs.diff() > pd.Timedelta('6h')]]}"))
fm = pd.read_parquet(D / "fmi_obs_10min.parquet")
for s, g in fm.groupby("station"):
    rows.append(dict(series=f"fmi_{s}", n=len(g), resolution="10min", dups=int(g.obs_time_utc.duplicated().sum()), missing_val=int(g.ws_10min.isna().sum()), min=g.ws_10min.min(), p50=g.ws_10min.median(), max=g.ws_10min.max()))
A = pd.DataFrame(rows); A.to_csv(R / "audit_series.csv", index=False)
pd.set_option("display.width", 250); print(A.drop(columns=["note"]).to_string()); [print(r.series, r.note) for r in A.itertuples() if isinstance(r.note, str)]
# price revisions: Fingrid 319 vs eSett FI
p = pd.read_parquet(D / "fingrid_319.parquet").set_index("startTime").value; fi = e[e.mba_code == "FI"].assign(t=lambda x: pd.to_datetime(x.timestampUTC, utc=True)).set_index("t").imblSalesPrice
dd = (p - fi).dropna(); bad = dd[dd.abs() > 0.01]
print(f"\n319 vs eSett: {len(bad)} of {len(dd)} quarters differ; examples:"); print(pd.DataFrame({"fingrid": p[bad.index], "esett": fi[bad.index]}).head(10))
# extreme prices
q = pd.DataFrame({"imb": fi, "spread": fi - (fi - e[e.mba_code=='FI'].set_index(pd.to_datetime(e[e.mba_code=='FI'].timestampUTC, utc=True)).imblSpotDifferencePrice)})
print("\nTop 10 imbalance prices (eSett FI):"); print(fi.sort_values().tail(10))
ex = fi[fi > 1000]; print("quarters >1000 EUR:", len(ex), "days:", sorted(set(ex.index.tz_convert('Europe/Helsinki').date)))
