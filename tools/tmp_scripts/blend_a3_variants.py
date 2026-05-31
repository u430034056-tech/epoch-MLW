"""Blend A3 (plain target encoding) with A3_OOFte (OOF target encoding).

Idea: they produce slightly different probability distributions, so their
average often lands at a higher accuracy and logloss than either alone.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("blend_a3")


def make(name: str, oof: bool) -> StageConfig:
    return StageConfig(
        name=name,
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=config.TARGET_ENCODE_COLUMNS,
        use_oof_target_encoding=oof,
        use_group_aggregates=False,
        use_common_features=True,
        params_override={},
        use_early_stopping=True,
        description=f"A3 OOF-TE={oof}",
    )


def _best_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best = (0.5, accuracy_score(y_true, (p >= 0.5).astype(int)))
    for t in np.arange(0.30, 0.70 + 1e-9, 0.005):
        acc = accuracy_score(y_true, (p >= t).astype(int))
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


a3 = run_cv(make("A3_plain", False), verbose=False)
a3_oof = run_cv(make("A3_OOFte", True), verbose=False)

for name, rep in [("A3_plain", a3), ("A3_OOFte", a3_oof)]:
    t, acc = _best_threshold(rep.oof_true, rep.oof_proba)
    log.info("[%s] acc@0.5=%.4f | best_t=%.3f acc@best=%.4f | logloss=%.4f | auc=%.4f",
             name, accuracy_score(rep.oof_true, (rep.oof_proba >= 0.5).astype(int)),
             t, acc, rep.oof_logloss, rep.oof_auc)

blend = (a3.oof_proba + a3_oof.oof_proba) / 2.0
t, acc_best = _best_threshold(a3.oof_true, blend)
log.info("[blend] acc@0.5=%.4f | best_t=%.3f acc@best=%.4f | logloss=%.4f | auc=%.4f",
         accuracy_score(a3.oof_true, (blend >= 0.5).astype(int)),
         t, acc_best,
         log_loss(a3.oof_true, np.clip(blend, 1e-6, 1 - 1e-6)),
         roc_auc_score(a3.oof_true, blend))

# Weighted blend search
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    b = w * a3.oof_proba + (1 - w) * a3_oof.oof_proba
    t, acc = _best_threshold(a3.oof_true, b)
    log.info("  w(A3_plain)=%.1f -> acc@best=%.4f (t=%.3f)", w, acc, t)
