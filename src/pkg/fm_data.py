"""Chronos inputs: fixed 2048-quarter history through origin minus 45 minutes."""
import numpy as np
import pandas as pd

import pkg_features as pf

CONTEXT, HORIZON, BRIDGE = 2048, 42, 2
FUTURE = {"spot": "fi_spot_price", "wind_da": "fi_wind_fc_da",
          "consumption_da": "fi_consumption_fc_da", "solar_da": "fi_solar_fc_da"}
PAST = {"mfrr_up_price": "fi_mfrr_up_price", "mfrr_down_price": "fi_mfrr_down_price",
        "mfrr_direction": "fi_mfrr_direction", "mfrr_up_activated": "fi_mfrr_up_activated",
        "mfrr_down_activated": "fi_mfrr_down_activated", "calculated_imbalance": "x_calc_imb",
        "sa_price": "x_sa_price"}


def load_sources():
    return pf.load_external(*pf.load_package())


def make_input(values, availability, origin):
    end = origin - pd.Timedelta(minutes=45)
    history = pd.date_range(end=end, periods=CONTEXT, freq="15min")
    future = pd.date_range(end + pf.Q, periods=HORIZON, freq="15min")
    assert future[BRIDGE] == origin and future[-1] == origin + 39 * pf.Q
    timestamps = history.append(future)
    cols = list(dict.fromkeys(["fi_imbalance_price", *FUTURE.values(), *PAST.values(), "x_wind_rt"]))
    v = values.reindex(timestamps)[cols]
    a = availability.reindex(timestamps)[cols]
    visible = v.where(a.notna() & a.le(origin))
    past = {name: visible[c].iloc[:CONTEXT].to_numpy(dtype="float32") for name, c in PAST.items()}
    past["wind_rt_minus_da"] = (visible.x_wind_rt - visible.fi_wind_fc_da).iloc[:CONTEXT].to_numpy(dtype="float32")
    known = {}
    filled = 0
    for name, c in FUTURE.items():
        past[name] = visible[c].iloc[:CONTEXT].to_numpy(dtype="float32")
        filled += int(visible[c].iloc[CONTEXT:].isna().sum())
        # Forward fill only AFTER masking. History seeds the unknown future tail.
        known[name] = visible[c].ffill().iloc[CONTEXT:].to_numpy(dtype="float32")
        assert np.isfinite(known[name]).all(), (origin, name)
    target = (visible.fi_imbalance_price - visible.fi_spot_price).iloc[:CONTEXT].to_numpy(dtype="float32")
    assert np.isnan(target[-2:]).all(), origin
    assert np.isfinite(target).sum() >= CONTEXT // 2
    spot = visible.fi_spot_price.iloc[CONTEXT + BRIDGE:].to_numpy(dtype=float)
    assert len(spot) == 40 and np.isfinite(spot).all(), f"Unavailable delivery spot at {origin}"
    return {"target": target, "past_covariates": past, "future_covariates": known}, spot, filled


def assert_task_equal(a, b):
    for key in a:
        if isinstance(a[key], dict):
            assert a[key].keys() == b[key].keys()
            for c in a[key]:
                np.testing.assert_array_equal(a[key][c], b[key][c])
        else:
            np.testing.assert_array_equal(a[key], b[key])
