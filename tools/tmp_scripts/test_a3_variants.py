"""Isolate which of {OOF target encoding, group aggregates} hurts A3."""
from __future__ import annotations

import logging
import sys

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("a3_variants")


def make(name: str, oof: bool, grp: bool) -> StageConfig:
    return StageConfig(
        name=name,
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=config.TARGET_ENCODE_COLUMNS,
        use_oof_target_encoding=oof,
        use_group_aggregates=grp,
        use_common_features=True,
        params_override={},
        use_early_stopping=True,
        description=f"A3 + OOF-TE={oof}, GrpAgg={grp}",
    )


for name, oof, grp in [("A3_OOFte", True, False), ("A3_grpAgg", False, True), ("A3_both", True, True)]:
    rep = run_cv(make(name, oof, grp), verbose=False)
    log.info("[%s] OOF acc=%.4f logloss=%.4f auc=%.4f mean_best_iter=%.0f elapsed=%.1fs",
             name, rep.oof_acc, rep.oof_logloss, rep.oof_auc,
             float(sum(rep.best_iterations))/len(rep.best_iterations), rep.elapsed_seconds)
