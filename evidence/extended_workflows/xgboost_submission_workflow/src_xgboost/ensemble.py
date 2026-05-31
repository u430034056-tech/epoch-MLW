"""Multi-seed bagging for the tuned XGBoost stage.

A single seed already gives ~0.005 fold-to-fold wiggle; averaging the test
probabilities across five seeds usually buys another 0.001-0.003 on the public
leaderboard at near-zero risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .train import CVReport, StageConfig, run_cv


@dataclass
class EnsembleReport:
    members: list[CVReport]
    oof_proba: np.ndarray
    oof_true: np.ndarray
    test_proba: np.ndarray
    passenger_id_test: list[str]
    seeds_used: list[int]


def run_multi_seed(
    stage: StageConfig,
    seeds: Iterable[int] | None = None,
    verbose: bool = False,
) -> EnsembleReport:
    """Run ``stage`` once per seed and average probabilities.

    We only vary ``random_state`` between seeds; everything else (folds,
    preprocessing, params) is controlled by ``stage``.  The CV splitter itself
    uses ``config.RANDOM_SEED`` so the folds are stable across seeds, which
    gives a clean variance reduction interpretation.
    """
    seeds = list(seeds or config.SEED_POOL)
    members: list[CVReport] = []
    for s in seeds:
        params = dict(stage.params_override)
        params["random_state"] = int(s)
        this_stage = StageConfig(
            name=f"{stage.name}-s{s}",
            group_aware_cv=stage.group_aware_cv,
            fold_aware_surname=stage.fold_aware_surname,
            fold_aware_cabin_bin=stage.fold_aware_cabin_bin,
            target_encode_cols=stage.target_encode_cols,
            target_encode_mode=stage.target_encode_mode,
            te_smoothing=stage.te_smoothing,
            use_surname_rate=stage.use_surname_rate,
            surname_rate_smoothing=stage.surname_rate_smoothing,
            use_oof_target_encoding=stage.use_oof_target_encoding,
            use_group_aggregates=stage.use_group_aggregates,
            use_common_features=stage.use_common_features,
            params_override=params,
            use_early_stopping=stage.use_early_stopping,
            early_stopping_rounds=stage.early_stopping_rounds,
            n_estimators_override=stage.n_estimators_override,
            description=f"{stage.description} [seed={s}]",
        )
        members.append(run_cv(this_stage, verbose=verbose))

    oof = np.mean(np.stack([m.oof_proba for m in members]), axis=0).astype("float32")
    test = np.mean(np.stack([m.test_proba for m in members]), axis=0).astype("float32")
    return EnsembleReport(
        members=members,
        oof_proba=oof,
        oof_true=members[0].oof_true,
        test_proba=test,
        passenger_id_test=members[0].passenger_id_test,
        seeds_used=[int(s) for s in seeds],
    )
