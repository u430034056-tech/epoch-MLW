"""Optuna search on the V2 (no-TE) pipeline with a tight, conservative space.

We deliberately limit the search around the STRONG_PARAMS anchor so Optuna
cannot wander into OOF-overfitted corners (as in the first Optuna run that
produced ``lr=0.0446, depth=4, gamma=0.86``).  We also shorten the trial
count because with a clean pipeline the OOF is more volatile per-trial than
per-seed (std ~0.001).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import optuna
import pandas as pd

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("v2_optuna")

ANCHOR = dict(config.STRONG_PARAMS)


def objective(trial: optuna.Trial) -> float:
    params = dict(ANCHOR)
    params.update(dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.06, log=True),
        max_depth=trial.suggest_int("max_depth", 5, 7),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
        subsample=trial.suggest_float("subsample", 0.75, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.0),
    ))
    stage = StageConfig(
        name=f"V2_opt_{trial.number}",
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=(),
        target_encode_mode="none",
        use_common_features=True,
        params_override=params,
        use_early_stopping=True,
    )
    rep = run_cv(stage, verbose=False)
    return rep.oof_acc


t0 = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False, timeout=240)
elapsed = time.perf_counter() - t0

log.info("Best OOF accuracy: %.5f after %d trials (%.0fs)", study.best_value, len(study.trials), elapsed)
log.info("Best params: %s", study.best_params)

# Also record top-5 trials with their params
rows = sorted(
    [{"trial": t.number, "acc": t.value, **t.params} for t in study.trials if t.value is not None],
    key=lambda r: -r["acc"],
)[:10]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("reports/xgboost/logs/v2_optuna_top10.csv", index=False)

# Save best for reuse
import json
best_out = {"best_value": float(study.best_value), "best_params": study.best_params, "elapsed": elapsed}
with open("reports/xgboost/logs/v2_best_params.json", "w") as f:
    json.dump(best_out, f, indent=2)
log.info("Saved best params -> reports/xgboost/logs/v2_best_params.json")
