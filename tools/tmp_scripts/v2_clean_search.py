"""V2 clean pipeline with different parameter recipes.

Goal: find the recipe that is both competitive on OOF *and* has a small
OOF→LB gap (inferred from our priors: closer to A0 params should track LB).
"""
from __future__ import annotations

import logging
import pandas as pd

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("v2_clean")


def make(name: str, params: dict) -> StageConfig:
    return StageConfig(
        name=name,
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=(),  # No TE
        target_encode_mode="none",
        use_surname_rate=False,
        use_common_features=True,
        params_override=params,
        use_early_stopping=True,
        description=str(params),
    )


RECIPES = {
    "V2_A0like": dict(  # Closest to the legacy LB=0.80804 config, plus early stopping
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=1, gamma=0.0,
    ),
    "V2_strong": dict(  # Our STRONG_PARAMS baseline
        **{k: v for k, v in config.STRONG_PARAMS.items() if k not in ("enable_categorical", "tree_method", "eval_metric", "random_state", "n_jobs")}
    ),
    "V2_deeper": dict(
        n_estimators=2000, learning_rate=0.03, max_depth=7,
        subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
        min_child_weight=2, gamma=0.5,
    ),
    "V2_shallow": dict(
        n_estimators=2000, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=3, gamma=0.0,
    ),
    "V2_dart": dict(
        booster="dart", n_estimators=800, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        rate_drop=0.1, skip_drop=0.5, sample_type="uniform",
    ),
}

rows = []
for name, p in RECIPES.items():
    rep = run_cv(make(name, p), verbose=False)
    mean_iter = float(sum(rep.best_iterations)) / len(rep.best_iterations)
    row = dict(
        name=name,
        oof_acc=round(rep.oof_acc, 5),
        oof_logloss=round(rep.oof_logloss, 5),
        oof_auc=round(rep.oof_auc, 5),
        mean_best_iter=round(mean_iter, 1),
        elapsed=round(rep.elapsed_seconds, 1),
        n_features=len(rep.feature_names),
        params=str(p),
    )
    log.info(
        "[%s] OOF acc=%.4f logloss=%.4f auc=%.4f best_iter=%.0f feats=%d  %.1fs",
        name, rep.oof_acc, rep.oof_logloss, rep.oof_auc, mean_iter, len(rep.feature_names), rep.elapsed_seconds,
    )
    rows.append(row)

df = pd.DataFrame(rows).sort_values("oof_acc", ascending=False)
print("\n" + df.to_string(index=False))
df.to_csv("reports/xgboost/logs/v2_clean_search.csv", index=False)
