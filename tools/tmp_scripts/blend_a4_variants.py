"""Blend A4 (plain TE, tuned) with A4_OOFte (OOF TE, tuned)."""
from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("blend_a4")

best = json.loads((config.LOGS_DIR / "best_params.json").read_text())["best_params"]


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
        params_override=best,
        use_early_stopping=True,
        description="A4 tuned, OOF-TE variants",
    )


def _best_threshold(y_true, p):
    best_pair = (0.5, accuracy_score(y_true, (p >= 0.5).astype(int)))
    for t in np.arange(0.30, 0.70 + 1e-9, 0.005):
        a = accuracy_score(y_true, (p >= t).astype(int))
        if a > best_pair[1]:
            best_pair = (float(t), float(a))
    return best_pair


a4 = run_cv(make("A4_plain", False), verbose=False)
a4_oof = run_cv(make("A4_OOFte", True), verbose=False)

for name, rep in [("A4_plain", a4), ("A4_OOFte", a4_oof)]:
    t, acc = _best_threshold(rep.oof_true, rep.oof_proba)
    log.info(
        "[%s] acc@0.5=%.4f | best_t=%.3f acc@best=%.4f | logloss=%.4f | auc=%.4f",
        name,
        accuracy_score(rep.oof_true, (rep.oof_proba >= 0.5).astype(int)),
        t, acc, rep.oof_logloss, rep.oof_auc,
    )

for w in np.arange(0.2, 0.81, 0.1):
    b = w * a4.oof_proba + (1 - w) * a4_oof.oof_proba
    t, acc = _best_threshold(a4.oof_true, b)
    log.info("  w(A4_plain)=%.1f -> acc@best=%.4f (t=%.3f) logloss=%.4f auc=%.4f",
             w, acc, t, log_loss(a4.oof_true, np.clip(b, 1e-6, 1 - 1e-6)), roc_auc_score(a4.oof_true, b))
