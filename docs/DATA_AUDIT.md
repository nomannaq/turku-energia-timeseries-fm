# Data audit, per series (Turku Energia imbalance-price challenge, box workspace)
Period downloaded: 2026-05-25 to 2026-11-01 UTC (+ eSett to 2026-11-07). All timestamps are stored in UTC. Calendar features use Europe/Helsinki
(DST change on 2026-10-25 handled by tz conversion; eSett `timestamp` local vs `timestampUTC` offsets of +2/+1 h confirmed).
Availability lags are ASSUMED until checked against the starter package `first_usable_origin_utc` table. Outliers are deliberately NOT removed
(same choice as Saleh et al. 2026): price-cap spikes are real market outcomes.
Machine-readable stats: `results/audit_series.csv` (from `src/audit.py`). Leakage check: `src/leak_test.py` (future-mutation test, PASSED; the sanity variant that includes future origins FAILS as expected).

| Series | Source / ID | Unit | Native resolution | Sign / encoding | Duplicates | Missingness / gaps | Outliers / notes | Assumed availability | Transform used |
|---|---|---|---|---|---|---|---|---|---|
| Imbalance price | Fingrid 319 (eSett EXP14 `imblSalesPrice` as fallback) | EUR/MWh | 15 min | single price | 0 | 0 missing in period | min -115.04, median 9.97, max 4000 (cap, 1 quarter); 8 quarters > 1000 EUR on 29 May, 9 Sep and 15 Sep, all kept. 12 of 15,456 quarters differ from eSett (revisions, max 78.6 EUR on 18 Aug; Fingrid value used) | slot end + 30 min | spread = imb - spot (target); signed log1p of lagged spread as a feature |
| Spot price | derived: eSett `imblSalesPrice - imblSpotDifferencePrice` | EUR/MWh | 15 min | — | 0 | 0 | min -10, max 399.6 | known for every delivery slot (day-ahead, published about D-1 14:00 EET) | level, within-day rank, ramps, day mean/range; past rolling mean/std |
| mFRR dominating direction | Fingrid 369 (identical to eSett `mainDirRegPowerPerMBA`, 15,456/15,456) | — | 15 min | 1 = up, -1 = down, 0 = none (24 quarters, 0.16%) | 0 | 0 | — | slot end + 30 min | label (3-class); last value, run length, up share over 4/12/24/96 quarters, EWM |
| Calculated imbalance (ACE open loop) | Fingrid 397 | MW | 1 min (sec :52) | positive = surplus | 0 | 222 gaps (max 8 min); 0 quarters with < 2 samples | — | slot end + 15 min | 15-min mean/min/max labelled by slot start; lags, rolling, EWM |
| Wind power, real time | Fingrid 181 | MW | 3 min | — | 0 | 2,111 short gaps (max 24 min); 2 quarters < 2 samples -> NaN, ffill <= 1 quarter | min -35 (station consumption) | slot end + 15 min | minus DA forecast (error), rolling |
| Nuclear, real time | Fingrid 188 | MW | 3 min | — | 0 | 510 gaps (max 24 min) | drops show unit trips/maintenance | slot end + 15 min | drop vs 24-h max, 1-h change |
| Net import, real time | Fingrid 194 | MW | 3 min | positive = net import (per Fingrid; median -504 MW = net export) | 0 | 2,032 gaps (max 18 min) | — | slot end + 15 min | level, rolling, 1-h change |
| Frequency | Fingrid 177 | Hz | 1 min | — | 0 | 190 gaps (max 61 min); 4 quarters < 2 samples | — | slot end + 15 min | 15-min mean deviation from 50 Hz, intra-quarter std |
| Kinetic energy | Fingrid 260 | GWs | 15 min | — | 0 | 1 missing quarter | — | slot end + 15 min | last value |
| mFRR scheduled-activation price | Fingrid 400 | EUR/MWh | 15 min | — | 0 | 1 missing | 4000 cap seen | slot t published at t - 7.5 min | step-1 only: price(t) - spot; last value - spot for all steps |
| mFRR up price | Fingrid 244 | EUR/MWh | 15 min | — | 0 | 0 | 145 robust-z > 10 (real spikes) | slot end + 30 min | minus spot |
| aFRR up/down energy price | eSett EXP14 `upRegPriceFrrA` / `downRegPriceFrrA` | EUR/MWh | 15 min | **0.00 = no activation** (33% of up, 28% of down quarters; no true nulls) | 0 | — | — | slot end + 30 min (assumed) | 0 -> NaN + no-activation flags; lags 48/96, rolling mean/std 48/96, 24-h backward-mean imputed variant (paper), activation share |
| Wind DA forecast | Fingrid 246 | MW | 15 min | — | 0 | 0 | — | issued D-1 12:00 EET -> known for all 40 steps | level, ramps, change vs origin, residual load |
| Consumption DA forecast | Fingrid 165 | MW | 15 min | — | 0 | 0 | — | as above | residual load = cons_da - wind_da (no solar forecast downloaded) |
| Wind actual (15 min) | Fingrid 75 | MW | 15 min | — | 0 | 1 missing | — | slot end + 30 min | error vs DA: last, lags 4/8, rolling 4/12/24/96, EWM, relative |
| Consumption actual | Fingrid 124 | MW | 15 min | — | 0 | 1 missing | — | slot end + 30 min | error vs DA; lag 96; 24-h mean |
| Wind forecast updated every 15 min | Fingrid 245 | MW | 15 min | — | 0 | 0 | **overwritten in place, no vintages** | not usable | EXCLUDED |
| SE1 / SE3 imbalance price | eSett EXP14 | EUR/MWh | 15 min | — | 0 | 0 in period | SE1 min -1500, max 750 | slot end + 30 min (assumed) | FI - SE difference, 1-h mean |
| ECMWF IFS single runs, FI 6 wind points + Helsinki/Tampere; 8 neighbour points | Open-Meteo | m/s, deg, degC, W/m2 | 1 h, runs every 6 h, 30 h ahead | — | 0 | 1 missing run (2026-06-10 12Z); < 0.4% NaN (3.5% radiation) | — | run + 6 h | capacity-weighted FI aggregate (approximate regional weights), normalised power curve, ws^2, log ws, wd sin/cos, ramps, +-3 h rolling, revision vs previous run, wind chill, temperature lags/rolling/diff, interactions with hour/consumption |
| FMI station obs (10 min) | FMI WFS | m/s, deg, degC | 10 min | — | 0 | **Oulu: no wind at all; Kaskinen: 60% NaN**; Vaasa 5%, Tampere 3% | — | obs + 30 min | v1 only (not in v2 matrix) |
| eSett EXP13/15/16 (imbalance volume, consumption, production by type) | eSett | MWh | 15 min | — | 0 | ends 24-25 Oct (settlement lag of about 2 weeks) | — | ~14 days | analysis only, NOT used as features |

Resolution handling: 1/3-min series are aggregated to 15-min means labelled by slot START, covering [t, t+15 min). A quarter with fewer than 2 samples is set to NaN,
and forward-fill is limited to 1 quarter. All of this happens before the availability shift, so a filled value is never visible earlier than its source.
