"""Quick smoke test of target-encoding mode × surname-rate.

Goal: prove the self-leak hypothesis by swapping plain TE for LOO/OOF/none
and see if the OOF number becomes more conservative (closer to the honest
A2 baseline of 0.8143) while the LB should move up because the train-test
distribution shift is reduced.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.xgboost import config
from src.xgboost.train import StageConfig, run_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("smoke_v2")


def stage(name: str, te_mode: str, use_rate: bool, params: dict | None = None) -> StageConfig:
    return StageConfig(
        name=name,
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=config.TARGET_ENCODE_COLUMNS if te_mode != "none" else (),
        target_encode_mode=te_mode,
        use_surname_rate=use_rate,
        use_common_features=True,
        params_override=params or {},
        use_early_stopping=True,
        description=f"te_mode={te_mode}, surname_rate={use_rate}",
    )


CONFIGS = [
    ("V2a", "none", False),   # no TE, no surname rate = clean baseline
    ("V2b", "loo",  False),   # LOO TE only
    ("V2c", "oof",  False),   # OOF TE only
    ("V2d", "none", True),    # no TE, surname rate
    ("V2e", "loo",  True),    # LOO TE + surname rate
    ("V2f", "oof",  True),    # OOF TE + surname rate
]

rows = []
for name, mode, rate in CONFIGS:
    rep = run_cv(stage(name, mode, rate), verbose=False)
    row = dict(
        stage=name,
        te_mode=mode,
        surname_rate=rate,
        oof_acc=rep.oof_acc,
        oof_logloss=rep.oof_logloss,
        oof_auc=rep.oof_auc,
        mean_best_iter=float(np.mean(rep.best_iterations)),
        elapsed=rep.elapsed_seconds,
        n_features=len(rep.feature_names),
    )
    log.info(
        "[%s] te_mode=%-5s surname_rate=%-5s  OOF acc=%.4f  logloss=%.4f  auc=%.4f  feats=%d  %.1fs",
        name, mode, str(rate), rep.oof_acc, rep.oof_logloss, rep.oof_auc, len(rep.feature_names), rep.elapsed_seconds,
    )
    # dump test probas so we can inspect disagreement with A4
    pd.DataFrame({"PassengerId": rep.passenger_id_test, "y_proba": rep.test_proba}).to_csv(
        Path("reports/xgboost/logs") / f"{name}_test_proba.csv", index=False
    )
    pd.DataFrame({"feature": rep.feature_names, "gain": rep.importance_gain}).sort_values("gain", ascending=False).to_csv(
        Path("reports/xgboost/logs") / f"{name}_importance.csv", index=False
    )
    rows.append(row)

df = pd.DataFrame(rows)
out = Path("reports/xgboost/logs/smoke_v2.csv")
df.to_csv(out, index=False)
log.info("\n%s", df.to_string(index=False))
log.info("saved: %s", out)
