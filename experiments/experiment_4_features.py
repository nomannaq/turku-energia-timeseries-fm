"""Publication-aware outage notices and archived forecast-run features."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work_private/experiment4"
OUT = ROOT / "results/experiment_4_external_data"
FI = "10YFI-1--------U"
Q = pd.Timedelta("15min")
WEIGHTS = {"raahe": 0.30 / 0.65, "oulu": 0.20 / 0.65, "vaasa": 0.15 / 0.65}
OUTAGE_COLUMNS = ["gen_unplanned_mw", "gen_planned_mw", "nuclear_mw", "wind_mw",
                  "load_unplanned_mw", "recent_unplanned_mw", "gen_restoration_mw",
                  "net_unplanned_supply_mw"] + [f"cable_{a}_{b}_mw" for a, b in
                  [("FI", "EE"), ("EE", "FI"), ("FI", "SE3"), ("SE3", "FI"), ("FI", "SE1"), ("SE1", "FI")]]


def prepare_notices(messages):
    first = {}
    for message in messages:
        published = pd.Timestamp(message["publicationDate"]).value
        first[message["messageId"]] = min(published, first.get(message["messageId"], published))
    prepared = []
    for message in messages:
        published = pd.Timestamp(message["publicationDate"])
        record = {"id": message["messageId"], "version": message["version"],
                  "usable": (published + pd.Timedelta("5min")).ceil("h").value,
                  "first": first[message["messageId"]], "periods": []}
        if message["eventStatus"] == 3:  # A cancellation replaces the previous notice.
            prepared.append(record)
            continue
        assert message["eventStatus"] == 1
        unplanned = message.get("unavailabilityType") == 1
        # Some messages describe an aggregate production unit; others its generators.
        # Prefer generator detail when both are supplied to avoid double counting.
        generation = message.get("generationUnits") or message.get("productionUnits", [])
        groups = [("generation", generation), ("consumption", message.get("consumptionUnits", [])),
                  ("transmission", message.get("transmissionUnits", []))]
        for kind, units in groups:
            for unit in units:
                categories = []
                if kind == "transmission":
                    a, b = unit.get("inAreaName"), unit.get("outAreaName")
                    category = f"cable_{a}_{b}_mw"
                    if category not in OUTAGE_COLUMNS:
                        continue
                    categories = [category]
                    # Transmission units describe a directional corridor, not necessarily
                    # an individual physical cable. Use its maximum reported reduction.
                    asset = category
                else:
                    if unit.get("areaEic") != FI:
                        continue
                    asset = unit.get("eic") or unit["name"]
                    if kind == "generation":
                        categories = ["gen_unplanned_mw" if unplanned else "gen_planned_mw"]
                        if unit.get("fuelType") == 14:
                            categories.append("nuclear_mw")
                        if unit.get("fuelType") in (18, 19):
                            categories.append("wind_mw")
                        if unplanned:
                            categories.append("recent_unplanned_mw")
                    elif unplanned:
                        categories = ["load_unplanned_mw"]
                for period in unit.get("timePeriods", []):
                    start = pd.Timestamp(period["eventStart"]).value
                    stop = pd.Timestamp(period["eventStop"]).value
                    capacity = float(period["unavailableCapacity"])
                    assert stop >= start and np.isfinite(capacity) and capacity >= 0
                    record["periods"].append((asset, categories, start, stop, capacity))
        prepared.append(record)
    return sorted(prepared, key=lambda r: (r["usable"], r["id"], r["version"]))


def outage_features(frame, messages):
    prepared = prepare_notices(messages)
    latest, pointer, blocks = {}, 0, []
    for origin, group in frame.groupby("o", sort=True):
        now = origin.value
        while pointer < len(prepared) and prepared[pointer]["usable"] <= now:
            notice = prepared[pointer]
            latest[notice["id"]] = notice
            pointer += 1
        targets = group.t.to_numpy(dtype="datetime64[ns]").astype("int64")
        per_asset = {}
        for notice in latest.values():
            for asset, categories, start, stop, capacity in notice["periods"]:
                active = (targets >= start) & (targets < stop)
                if not active.any():
                    continue
                for category in categories:
                    if category == "recent_unplanned_mw" and now - notice["first"] > pd.Timedelta("3h").value:
                        continue
                    key = (asset, category)
                    old = per_asset.setdefault(key, np.zeros(len(group)))
                    np.maximum(old, active * capacity, out=old)
        result = {c: np.zeros(len(group)) for c in OUTAGE_COLUMNS}
        for (_, category), vector in per_asset.items():
            result[category] += vector
        total = result["gen_unplanned_mw"] + result["gen_planned_mw"]
        result["gen_restoration_mw"] = total[0] - total
        result["net_unplanned_supply_mw"] = result["gen_unplanned_mw"] - result["load_unplanned_mw"]
        blocks.append(pd.DataFrame(result, index=group.index))
    return pd.concat(blocks).sort_index().add_prefix("ex4_out_").astype("float32")


def weather_features(frame, weather, delay_hours=6):
    output = pd.DataFrame(index=frame.index)
    for model, prefix in [("dmi_harmonie_arome_europe", "dmi"), ("icon_eu", "icon")]:
        w = weather[weather.model == model].copy()
        w["weight"] = w.location.map(WEIGHTS)
        ws = w.wind_speed_100m
        variables = {"ws": ws, "power": np.clip((ws - 3) / 10, 0, 1) ** 1.5 * (ws < 25),
                     "wd_sin": np.sin(np.deg2rad(w.wind_direction_100m)),
                     "wd_cos": np.cos(np.deg2rad(w.wind_direction_100m))}
        for name, value in variables.items():
            w[name] = value * w.weight
        table = w.groupby(["run_time_utc", "valid_time_utc"])[list(variables)].sum(min_count=3)
        runs = pd.DatetimeIndex(table.index.get_level_values(0).unique()).sort_values()
        run_ns = runs.to_numpy(dtype="datetime64[ns]").astype("int64")
        origin_ns = frame.o.to_numpy(dtype="datetime64[ns]").astype("int64")
        idx = np.searchsorted(run_ns + pd.Timedelta(hours=delay_hours).value, origin_ns, side="right") - 1
        assert (idx >= 0).all(), "Archive does not cover the earliest origin"
        chosen = runs[np.maximum(idx, 0)]
        previous = runs[np.maximum(idx - 1, 0)]
        targets = pd.DatetimeIndex(frame.t)
        hour = targets.floor("h")
        fraction = np.asarray((targets - hour) / pd.Timedelta("1h"))
        def lookup(selected, valid, column):
            keys = pd.MultiIndex.from_arrays([selected, valid], names=table.index.names)
            return table[column].reindex(keys).to_numpy()
        age = (origin_ns - run_ns[idx]) / pd.Timedelta("1h").value
        for name in variables:
            v0 = lookup(chosen, hour, name)
            v1 = lookup(chosen, hour + pd.Timedelta("1h"), name)
            current = np.where(fraction == 0, v0, v0 * (1 - fraction) + v1 * fraction)
            current[age > 18] = np.nan
            output[f"ex4_wx_{prefix}_{name}"] = current
            if name in ("ws", "power"):
                old0 = lookup(previous, hour, name)
                old1 = lookup(previous, hour + pd.Timedelta("1h"), name)
                old = np.where(fraction == 0, old0, old0 * (1 - fraction) + old1 * fraction)
                output[f"ex4_wx_{prefix}_{name}_revision"] = current - old
        output[f"ex4_wx_{prefix}_age_h"] = age
        # Differences from the current model's six-region ECMWF proxy are features,
        # not a claim that these three-region power proxies are measured fleet output.
        output[f"ex4_wx_{prefix}_power_minus_ecmwf"] = output[f"ex4_wx_{prefix}_power"] - frame.wx_fiw_pw
        output[f"ex4_wx_{prefix}_ws_minus_ecmwf"] = output[f"ex4_wx_{prefix}_ws"] - frame.wx_fiw_ws
    output["ex4_wx_model_power_disagreement"] = output.ex4_wx_dmi_power - output.ex4_wx_icon_power
    output["ex4_wx_model_ws_disagreement"] = output.ex4_wx_dmi_ws - output.ex4_wx_icon_ws
    return output.astype("float32")


def check_outages(frame, messages):
    checks = []
    for cutoff in pd.to_datetime(["2026-07-05T21:00Z", "2026-09-06T21:00Z", "2026-09-20T14:00Z"]):
        sample = frame[frame.o == cutoff]
        expected = outage_features(sample, messages)
        # Removing all future revisions must exactly match the full archived history.
        past = [m for m in messages if (pd.Timestamp(m["publicationDate"]) + pd.Timedelta("5min")).ceil("h") <= cutoff]
        pd.testing.assert_frame_equal(expected, outage_features(sample, past))
        checks.append({"origin": str(cutoff), "unavailable_revision_removal": True})
    # A future revision/cancellation must not rewrite an earlier known outage.
    m = {"messageId": "test", "version": 1, "publicationDate": "2026-09-06T19:00Z", "eventStatus": 1,
         "unavailabilityType": 1, "generationUnits": [{"areaEic": FI, "eic": "test-asset", "name": "test",
         "fuelType": 14, "timePeriods": [{"eventStart": "2026-09-06T20:00Z", "eventStop": "2026-09-07T08:00Z", "unavailableCapacity": 100}]}]}
    cancelled = {**deepcopy(m), "version": 2, "publicationDate": "2026-09-06T22:00Z", "eventStatus": 3}
    sample = frame[frame.o.isin(pd.to_datetime(["2026-09-06T21:00Z", "2026-09-06T23:00Z"]))]
    f = outage_features(sample, [m, cancelled])
    assert (f.loc[sample.o == pd.Timestamp("2026-09-06T21:00Z"), "ex4_out_gen_unplanned_mw"] == 100).all()
    assert (f.loc[sample.o == pd.Timestamp("2026-09-06T23:00Z"), "ex4_out_gen_unplanned_mw"] == 0).all()
    duplicate = {**deepcopy(m), "messageId": "duplicate"}
    pd.testing.assert_frame_equal(outage_features(sample, [m]), outage_features(sample, [m, duplicate]))
    return checks


def main(outages_only=False):
    frame = pd.read_parquet(ROOT / "work_private/features_B.parquet")
    frame = frame[frame.o < pd.Timestamp("2026-10-04T21:00Z")].copy()
    messages = json.loads((WORK / "outage_revisions.json").read_text())
    checks = {"outages": check_outages(frame, messages)}
    of = outage_features(frame, messages)
    of.to_parquet(WORK / "outage_features.parquet")
    print("Outage features", of.shape, flush=True)
    if not outages_only:
        w = pd.read_parquet(WORK / "weather_runs.parquet")
        wf = weather_features(frame, w)
        checks["weather"] = []
        for cutoff in pd.to_datetime(["2026-07-05T21:00Z", "2026-09-06T21:00Z", "2026-09-20T14:00Z"]):
            sample = frame[frame.o == cutoff]
            past = w[w.run_time_utc + pd.Timedelta("6h") <= cutoff]
            pd.testing.assert_frame_equal(wf.loc[sample.index], weather_features(sample, past))
            checks["weather"].append({"origin": str(cutoff), "unavailable_run_removal": True})
        wf.to_parquet(WORK / "weather_features.parquet")
        print("Weather features", wf.shape, "missing fraction", wf.isna().mean().max(), flush=True)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / "availability_checks.json").write_text(json.dumps(checks, indent=2))
    stats = of.describe().T
    if not outages_only:
        stats = pd.concat([stats, wf.describe().T])
    stats.to_csv(OUT / "feature_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outages-only", action="store_true")
    main(parser.parse_args().outages_only)
