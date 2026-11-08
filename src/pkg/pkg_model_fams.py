"""Feature-family definitions shared with pkg_model.py (kept identical)."""
def cols_upto(X, V, rung):
    FEAT = [c for c in X.columns if c.startswith(("st_", "dl_", "wx_"))] + ["step"]
    CAL = ["step"] + [c for c in FEAT if c.startswith("dl_") and any(k in c for k in ["cal_", "qh", "hour_", "dow_", "weekend", "origin_hour"])]
    def fam(c):
        if c in CAL or c.startswith(("dl_spot", "st_spot_past")) or c in ("st_sp_last", "st_sp_mean96", "st_sp_med96", "st_imb_last"): return "E1"
        if c.startswith(("st_sp_", "st_spike")): return "E2"
        if c.startswith(("dl_wind", "dl_cons", "dl_solar", "dl_resid", "st_cerr", "st_cons", "st_prod", "st_net_prod")): return "E3"
        if c.startswith(("st_werr", "st_wind_act", "st_ci_", "st_wrt", "st_nuc", "st_netimp", "st_freq", "st_kin", "st_sa_")): return "EX"
        if c.startswith("wx_"): return "EW"
        return "E4"
    L = ["E1", "E2", "E3", "E4"] + (["EX", "EW"] if V == "B" else []); keep = L[: L.index(rung) + 1]
    return [c for c in FEAT if fam(c) in keep]
