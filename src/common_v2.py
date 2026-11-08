import warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"; R = ROOT / "results"; R.mkdir(exist_ok=True)
X = pd.read_parquet(D / "feature_matrix_v2.parquet")
X = X[X.y_dir.notna() & X.spread.notna()].reset_index(drop=True)
X["y_dir3"] = X.y_dir.map({-1: 0, 0: 1, 1: 2}).astype(int); X["y_spk"] = np.select([X.spread > 50, X.spread < -50], [2, 0], 1)
X["dday"] = X.t.dt.tz_convert("Europe/Helsinki").dt.date
FEAT = [c for c in X.columns if c.startswith(("st_", "rt_", "dl_", "wx_"))] + ["step"]
X[FEAT] = X[FEAT].astype("float32")
CAL = ["step"] + [f"dl_{c}" for c in ["hour", "qh", "dow", "weekend", "holiday", "origin_hour", "hour_x_step", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]]
def fam(c):
    if c in CAL: return "E1"
    if c.startswith(("dl_spot", "st_spot_past")) or c in ("st_sp_last", "st_sp_mean96", "st_sp_med96"): return "E1"
    if c.startswith(("st_sp_", "st_spike")): return "E2"
    if c.startswith(("dl_wind_da", "dl_cons_da", "dl_resid", "st_werr", "st_cerr", "st_wind_act", "st_cons_")): return "E3"
    if c.startswith(("st_dir", "st_up", "st_afrr", "st_mfrr", "st_fi_minus", "st_SE")): return "E4"
    if c.startswith("rt_") or c == "dl_sa_price_t_minus_spot": return "E4rt"
    if c.startswith("wx_"):
        if "_sw" in c: return "E5c"
        if any(k in c for k in ("_t2", "wchill", "temp_")): return "E5b"
        return "E5a"
    return "other"
FAM = pd.Series({c: fam(c) for c in FEAT})
LADDER = ["E1", "E2", "E3", "E4", "E4rt", "E5a", "E5b", "E5c"]
def cols_upto(k, drop=()):
    keep = LADDER[: LADDER.index(k) + 1]; return [c for c in FEAT if FAM[c] in keep and FAM[c] not in drop]
FOLDS = [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in [("2026-07-05 21:00", "2026-07-19 21:00"), ("2026-07-19 21:00", "2026-08-02 21:00"),
         ("2026-08-02 21:00", "2026-08-16 21:00"), ("2026-08-16 21:00", "2026-08-30 21:00"), ("2026-08-30 21:00", "2026-09-06 21:00")]]
TR_END = pd.Timestamp("2026-09-06 21:00", tz="UTC"); VA_END = pd.Timestamp("2026-10-04 21:00", tz="UTC")
def fold_idx(a, b): return np.where(X.t < a)[0], np.where((X.o >= a) & (X.o < b) & (X.t < b))[0]
def train_idx(): return np.where(X.t < TR_END)[0]
def val_idx(): return np.where((X.o >= TR_END) & (X.t < VA_END))[0]
BASEP = dict(learning_rate=0.02, num_leaves=7, min_data_in_leaf=2000, feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=30.0, n_rounds=200, verbose=-1, seed=7, num_threads=8)  # conservative fixed model (small data)
KINDS = ["dir", "spk", 0.1, 0.5, 0.9]
def ylab(k): return X.y_dir3 if k == "dir" else X.y_spk if k == "spk" else X.spread
def fit(k, i, cols, params=None, n=300):
    p = {**BASEP, **(params or {})}; n = p.pop("n_rounds", n)
    p.update(dict(objective="multiclass", num_class=3) if k in ("dir", "spk") else dict(objective="quantile", alpha=k))
    return lgb.train(p, lgb.Dataset(X.loc[i, cols], ylab(k).iloc[i]), n)
def brier(p, y): return ((p - np.eye(3)[y]) ** 2).sum(1)
def pinball(y, q, a): d = y - q; return np.maximum(a * d, (a - 1) * d)
def loss(k, pred, j):
    y = ylab(k).iloc[j].values
    return brier(pred, y) if k in ("dir", "spk") else pinball(y, pred, k)
def block_boot(d, blocks, B=1000, seed=0):
    """Paired bootstrap of mean loss difference, resampling delivery days (blocks)."""
    g = pd.DataFrame({"d": d, "b": blocks}).groupby("b").d.agg(["sum", "size"]); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (B, len(g))); s = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), float((s < 0).mean())
