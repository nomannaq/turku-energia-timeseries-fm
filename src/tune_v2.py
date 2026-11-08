"""Light Optuna tuning on the LAST TWO training folds only (no validation data). dir: 30 trials, p50: 20 trials."""
import json, optuna
from common_v2 import *
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEL = json.load(open(R / "v2_selected_features.json")); F2 = FOLDS[3:]
def cvloss(k, params):
    out = []
    for a, b in F2:
        cols = SEL[str(k)]; i, j = fold_idx(a, b); m = fit(k, i, cols, params); out.append(loss(k, m.predict(X.loc[j, cols]), j))
    return float(np.concatenate(out).mean())
best = {}
for k, n in [("dir", 30), (0.5, 20)]:
    def obj(tr):
        p = dict(learning_rate=tr.suggest_float("learning_rate", 0.01, 0.08, log=True), num_leaves=tr.suggest_int("num_leaves", 3, 31, log=True),
                 min_data_in_leaf=tr.suggest_int("min_data_in_leaf", 300, 4000, log=True), feature_fraction=tr.suggest_float("feature_fraction", 0.2, 0.8),
                 lambda_l2=tr.suggest_float("lambda_l2", 1, 100, log=True), n_rounds=tr.suggest_int("n_rounds", 100, 600, step=50))
        return cvloss(k, p)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial(dict(learning_rate=0.02, num_leaves=7, min_data_in_leaf=2000, feature_fraction=0.4, lambda_l2=30.0, n_rounds=200))  # default as trial 0
    st.optimize(obj, n_trials=n)
    best[str(k)] = dict(params=st.best_params, cv=st.best_value, default_cv=st.trials[0].value); print(k, best[str(k)], flush=True)
json.dump(best, open(R / "v2_optuna_best.json", "w"), indent=1)
