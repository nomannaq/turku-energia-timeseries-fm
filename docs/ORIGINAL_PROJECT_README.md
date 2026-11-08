# Turku Energia imbalance-price hackathon: data and usefulness test (box workspace)
Scripts are in src/, data in data/ (raw API responses are cached under data/raw/). Python venv: .venv
- fetch_openmeteo.py / fetch_openmeteo_neighbours.py: ECMWF IFS single runs, 00/06/12/18Z from 2026-05-25 to 2026-11-01, 30 h each. first_usable_utc = run + 6 h.
- fetch_fmi.py: FMI 10-min station observations (ws_10min, wd_10min, t2m). first_usable = obs + 30 min (assumed).
- fetch_esett.py: eSett open data (no key needed). EXP14 single imbalance price, direction and imbalance-minus-spot for FI/SE1/SE3. Spot = imblSalesPrice - imblSpotDifferencePrice.
  EXP13/15/16 settlement volumes lag about 2 weeks, so use them for analysis only.
- fetch_fingrid.py: Fingrid API (needs env FINGRID_API_KEY). 15-min datasets: usable at end + 30 min. Real-time 1/3-min datasets: resampled to 15 min, usable at end + 15 min.
  246/165 day-ahead forecasts are treated as known for every delivery slot (issued D-1 12:00 EET). 245 is overwritten in place with no archive, so it is EXCLUDED as a feature.
  400 (scheduled mFRR price) for slot t is published at t-7.5 min, so it is used only at step 1.
- features.py: as-of feature builder. usefulness.py: nested feature-set LightGBM test, train -> validation, with a day-block bootstrap.
- describe.py: descriptive statistics.
All lags are ASSUMPTIONS until checked against the starter package availability table (first_usable_origin_utc).
