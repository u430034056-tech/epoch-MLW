"""SKF vs SGK probe on the V2 (no-TE) pipeline.

The legacy LB=0.80804 model was trained under a StratifiedKFold (SKF).  Our
V2 uses StratifiedGroupKFold (SGK), which is theoretically more honest but
also reports systematically lower OOF (because no group leaks between
train/valid).  If the public LB's group distribution is closer to SKF's
train/test split, SGK might be *over-pessimistic* and we are leaving points
on the table.

This script trains V2 under both CV protocols and reports OOF plus the
per-seed stddev to decide whether a SKF submission is worth trying.
"""
from __future__ import annotations

import logging
import pandas as pd

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cv_probe")


def make(name: str, group_aware: bool) -> StageConfig:
    return StageConfig(
        name=name,
        group_aware_cv=group_aware,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=(),
        target_encode_mode="none",
        use_common_features=True,
        params_override=dict(config.STRONG_PARAMS),
        use_early_stopping=True,
    )


for name, gc in [("V2_SGK", True), ("V2_SKF", False)]:
    rep = run_cv(make(name, gc), verbose=False)
    log.info(
        "[%s] OOF acc=%.4f logloss=%.4f auc=%.4f best_iter=%.1f",
        name, rep.oof_acc, rep.oof_logloss, rep.oof_auc,
        float(sum(rep.best_iterations)) / len(rep.best_iterations),
    )
    # save test probas for comparison
    pd.DataFrame({"PassengerId": rep.passenger_id_test, "y_proba": rep.test_proba}).to_csv(
        f"reports/xgboost/logs/{name}_test_proba.csv", index=False
    )
