"""XGBoost model factory and per-fold fitting helper.

We use the sklearn-style ``XGBClassifier`` interface for easy cross-validation,
early stopping and multi-seed bagging.  ``enable_categorical=True`` lets the
booster handle pandas ``category`` columns without one-hot expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from . import config


@dataclass
class FoldResult:
    fold: int
    best_iteration: int
    best_score: float
    valid_proba: np.ndarray
    valid_true: np.ndarray
    test_proba: np.ndarray
    train_seconds: float
    feature_names: list[str] = field(default_factory=list)
    feature_importance: dict[str, np.ndarray] = field(default_factory=dict)


def build_model(params: dict | None = None, random_state: int | None = None) -> XGBClassifier:
    """Construct a fresh XGBClassifier from ``params``.

    Parameters
    ----------
    params
        Overrides for the default hyper-parameters.  If ``None``, we use
        ``STRONG_PARAMS`` — a sensible starting point given early stopping and
        our richer feature matrix.
    random_state
        Seed override used by multi-seed bagging.
    """
    merged: dict[str, Any] = dict(config.STRONG_PARAMS) if params is None else dict(config.STRONG_PARAMS)
    if params:
        merged.update(params)
    if random_state is not None:
        merged["random_state"] = random_state
    merged.setdefault("enable_categorical", True)
    merged.setdefault("verbosity", 0)
    return XGBClassifier(**merged)


def fit_one_fold(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    *,
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS,
    fold_idx: int = 0,
    verbose: bool = False,
) -> FoldResult:
    """Train on one fold, produce validation + test probabilities."""
    import time

    t0 = time.perf_counter()
    # XGBoost 2.x / 3.x expose early stopping via a callback when using the
    # scikit-learn wrapper.  We attach a fresh callback per fit to pick up the
    # latest ``eval_metric`` value.
    callbacks = [EarlyStopping(rounds=early_stopping_rounds, save_best=True)]

    model.set_params(callbacks=callbacks)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=verbose,
    )
    dt = time.perf_counter() - t0

    best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else model.get_params().get("n_estimators", 0)
    best_score = float(model.best_score) if hasattr(model, "best_score") else float("nan")

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    booster = model.get_booster()
    feature_names = list(booster.feature_names) if booster.feature_names else list(X_train.columns)

    importance = {}
    for imp_type in ("gain", "weight", "cover"):
        raw = booster.get_score(importance_type=imp_type)
        arr = np.zeros(len(feature_names), dtype="float64")
        for i, name in enumerate(feature_names):
            arr[i] = raw.get(name, 0.0)
        importance[imp_type] = arr

    return FoldResult(
        fold=fold_idx,
        best_iteration=best_iter,
        best_score=best_score,
        valid_proba=valid_proba.astype("float32"),
        valid_true=np.asarray(y_valid).astype("int8"),
        test_proba=test_proba.astype("float32"),
        train_seconds=float(dt),
        feature_names=feature_names,
        feature_importance=importance,
    )
