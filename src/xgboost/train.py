"""Top-level CV trainer for the Spaceship Titanic XGBoost pipeline.

``run_cv`` is the single entry point used by both the ablation script and the
tuning loop.  It is parameterised so all ablation stages (A0 → A6) share the
same code path; only the ``StageConfig`` changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config, cv, data, model
from .features import apply_fold_features


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Knobs that differ between the ablation stages (A0 … A6)."""

    name: str = "A?"
    group_aware_cv: bool = True
    fold_aware_surname: bool = True
    fold_aware_cabin_bin: bool = True
    target_encode_cols: tuple[str, ...] = ()
    target_encode_mode: str = "plain"  # plain | loo | oof | none
    te_smoothing: float = 20.0
    use_surname_rate: bool = False
    surname_rate_smoothing: float = 10.0
    use_oof_target_encoding: bool = False  # legacy, kept for A3b
    use_group_aggregates: bool = False
    use_common_features: bool = True  # if False, fall back to legacy 107-dim bundle
    params_override: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = config.EARLY_STOPPING_ROUNDS
    n_estimators_override: int | None = None
    use_early_stopping: bool = True
    description: str = ""

    def resolved_params(self) -> dict[str, Any]:
        p = dict(config.STRONG_PARAMS)
        p.update(self.params_override)
        if self.n_estimators_override is not None:
            p["n_estimators"] = self.n_estimators_override
        if not self.use_early_stopping:
            # Replace the very large default with something smaller when we
            # intentionally disable early stopping (e.g. A0 baseline).
            p["n_estimators"] = self.n_estimators_override or 400
        return p


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CVReport:
    stage: str
    n_folds: int
    oof_acc: float
    oof_logloss: float
    oof_auc: float
    fold_scores: list[dict]
    oof_proba: np.ndarray
    oof_true: np.ndarray
    test_proba: np.ndarray
    feature_names: list[str]
    importance_gain: np.ndarray
    importance_weight: np.ndarray
    importance_cover: np.ndarray
    best_iterations: list[int]
    description: str = ""
    passenger_id_test: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary_row(self) -> dict:
        return dict(
            stage=self.stage,
            n_folds=self.n_folds,
            oof_acc=self.oof_acc,
            oof_logloss=self.oof_logloss,
            oof_auc=self.oof_auc,
            mean_best_iter=float(np.mean(self.best_iterations)) if self.best_iterations else float("nan"),
            elapsed_seconds=self.elapsed_seconds,
            description=self.description,
        )


# ---------------------------------------------------------------------------
# Legacy 107-dim bundle loader (used by A0 to reproduce the *current* baseline)
# ---------------------------------------------------------------------------


def _load_legacy_xgb_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load `processed/xgboost/preprocessed_xgboost.joblib` and return usable
    (train, test, y, groups, passenger_id_test) DataFrames.

    The object-dtype ndarray is coerced to float32 where possible — this is
    exactly what the current benchmark script does, so the resulting score is
    directly comparable to the progress-report number.
    """
    b = joblib.load(config.XGB_BUNDLE)
    X_train_obj = b["X_train"]
    X_test_obj = b["X_test"]
    feat = b["feature_names"]
    X_train = pd.DataFrame(np.asarray(X_train_obj, dtype="float32"), columns=feat)
    X_test = pd.DataFrame(np.asarray(X_test_obj, dtype="float32"), columns=feat)
    y = pd.Series(b["y_train"]).reset_index(drop=True).astype("int8")

    common = joblib.load(config.COMMON_BUNDLE)
    groups = common["common_train"][config.GROUP_COLUMN].reset_index(drop=True).astype("string")
    pid_test = common["common_test"][config.ID_COLUMN].reset_index(drop=True).astype("string")
    return X_train, X_test, y, groups, pid_test


# ---------------------------------------------------------------------------
# Core CV loop
# ---------------------------------------------------------------------------


def run_cv(stage: StageConfig, verbose: bool = False) -> CVReport:
    """Execute one ablation stage end-to-end and return aggregated metrics."""
    import time

    t0 = time.perf_counter()

    # 1. Load the appropriate feature matrix
    if stage.use_common_features:
        common = data.load_common()
        X, X_test, y, groups = data.build_xgb_features(common)
        raw_train_full = common.train.reset_index(drop=True)
        raw_test_full = common.test.reset_index(drop=True)
        passenger_id_test = common.passenger_id_test.astype("string").tolist()
    else:
        X, X_test, y, groups, pid_test = _load_legacy_xgb_matrix()
        raw_train_full = None
        raw_test_full = None
        passenger_id_test = pid_test.astype("string").tolist()

    # 2. Build folds
    if stage.group_aware_cv:
        folds = cv.make_folds(y=y, groups=groups, seed=config.RANDOM_SEED)
    else:
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
        folds = [cv.FoldSlice(fold=i, train_idx=tr, valid_idx=va) for i, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y))]

    # 3. Loop over folds
    oof_proba = np.zeros(len(y), dtype="float32")
    test_proba_sum = np.zeros(len(X_test), dtype="float32")
    fold_scores: list[dict] = []
    best_iterations: list[int] = []
    importance_gain = None
    importance_weight = None
    importance_cover = None
    feature_names: list[str] = []

    for f in folds:
        Xt = X.iloc[f.train_idx].reset_index(drop=True)
        Xv = X.iloc[f.valid_idx].reset_index(drop=True)
        yt = y.iloc[f.train_idx].reset_index(drop=True)
        yv = y.iloc[f.valid_idx].reset_index(drop=True)

        if stage.use_common_features and (
            stage.fold_aware_surname
            or stage.fold_aware_cabin_bin
            or stage.target_encode_cols
            or stage.use_group_aggregates
            or stage.use_surname_rate
        ):
            raw_tr = raw_train_full.iloc[f.train_idx].reset_index(drop=True)
            raw_va = raw_train_full.iloc[f.valid_idx].reset_index(drop=True)
            Xt, Xv, Xte_fold = apply_fold_features(
                Xt,
                Xv,
                X_test,
                yt,
                raw_tr,
                raw_va,
                raw_test_full,
                target_encode_cols=stage.target_encode_cols,
                target_encode_mode=stage.target_encode_mode,
                te_smoothing=stage.te_smoothing,
                use_surname_refit=stage.fold_aware_surname,
                use_surname_rate=stage.use_surname_rate,
                surname_rate_smoothing=stage.surname_rate_smoothing,
                use_cabin_bin_refit=stage.fold_aware_cabin_bin,
                use_oof_target_encoding=stage.use_oof_target_encoding,
                use_group_aggregates=stage.use_group_aggregates,
                oof_te_seed=int(stage.params_override.get("random_state", config.RANDOM_SEED)),
            )
        else:
            Xte_fold = X_test

        # Keep column order aligned across train / valid / test
        Xv = Xv[Xt.columns]
        Xte_fold = Xte_fold[Xt.columns]

        booster = model.build_model(params=stage.resolved_params())

        if stage.use_early_stopping:
            fr = model.fit_one_fold(
                booster,
                Xt,
                yt,
                Xv,
                yv,
                Xte_fold,
                early_stopping_rounds=stage.early_stopping_rounds,
                fold_idx=f.fold,
                verbose=verbose,
            )
        else:
            # No early stopping → no eval_set either, to keep behaviour identical
            # to the A0 baseline script.
            t_fit0 = time.perf_counter()
            booster.set_params(callbacks=None)
            booster.fit(Xt, yt, verbose=False)
            fit_dt = time.perf_counter() - t_fit0
            valid_proba = booster.predict_proba(Xv)[:, 1].astype("float32")
            test_p = booster.predict_proba(Xte_fold)[:, 1].astype("float32")
            xgb_booster = booster.get_booster()
            importance = {
                k: np.asarray([xgb_booster.get_score(importance_type=k).get(n, 0.0) for n in xgb_booster.feature_names])
                for k in ("gain", "weight", "cover")
            }
            fr = model.FoldResult(
                fold=f.fold,
                best_iteration=booster.get_params().get("n_estimators", 400),
                best_score=float("nan"),
                valid_proba=valid_proba,
                valid_true=np.asarray(yv).astype("int8"),
                test_proba=test_p,
                train_seconds=fit_dt,
                feature_names=list(xgb_booster.feature_names or Xt.columns),
                feature_importance=importance,
            )

        oof_proba[f.valid_idx] = fr.valid_proba
        test_proba_sum += fr.test_proba
        best_iterations.append(fr.best_iteration)
        acc = accuracy_score(fr.valid_true, (fr.valid_proba >= 0.5).astype(int))
        ll = log_loss(fr.valid_true, fr.valid_proba.clip(1e-6, 1 - 1e-6))
        auc = roc_auc_score(fr.valid_true, fr.valid_proba)
        fold_scores.append(
            dict(
                fold=fr.fold,
                acc=acc,
                logloss=ll,
                auc=auc,
                best_iter=fr.best_iteration,
                train_seconds=fr.train_seconds,
                n_train=len(f.train_idx),
                n_valid=len(f.valid_idx),
            )
        )
        # Accumulate importance (average across folds at the end)
        if importance_gain is None:
            feature_names = fr.feature_names
            importance_gain = fr.feature_importance["gain"].astype("float64").copy()
            importance_weight = fr.feature_importance["weight"].astype("float64").copy()
            importance_cover = fr.feature_importance["cover"].astype("float64").copy()
        else:
            importance_gain += fr.feature_importance["gain"]
            importance_weight += fr.feature_importance["weight"]
            importance_cover += fr.feature_importance["cover"]

    test_proba = test_proba_sum / float(len(folds))
    importance_gain /= float(len(folds))
    importance_weight /= float(len(folds))
    importance_cover /= float(len(folds))

    oof_acc = accuracy_score(y, (oof_proba >= 0.5).astype(int))
    oof_ll = log_loss(y, oof_proba.clip(1e-6, 1 - 1e-6))
    oof_auc = roc_auc_score(y, oof_proba)
    elapsed = float(time.perf_counter() - t0)

    return CVReport(
        stage=stage.name,
        n_folds=len(folds),
        oof_acc=oof_acc,
        oof_logloss=oof_ll,
        oof_auc=oof_auc,
        fold_scores=fold_scores,
        oof_proba=oof_proba,
        oof_true=np.asarray(y).astype("int8"),
        test_proba=test_proba.astype("float32"),
        feature_names=feature_names,
        importance_gain=importance_gain,
        importance_weight=importance_weight,
        importance_cover=importance_cover,
        best_iterations=best_iterations,
        description=stage.description,
        passenger_id_test=passenger_id_test,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Canonical ablation stage registry
# ---------------------------------------------------------------------------


def ablation_stages() -> list[StageConfig]:
    """Return the canonical A0 → A5 ablation configurations.

    A6 (multi-seed ensemble) is handled separately because it re-uses the A5
    stage across ``SEED_POOL``.
    """
    return [
        StageConfig(
            name="A0",
            group_aware_cv=False,
            fold_aware_surname=False,
            fold_aware_cabin_bin=False,
            use_common_features=False,
            params_override=dict(config.BASELINE_PARAMS),
            use_early_stopping=False,
            n_estimators_override=400,
            description="Legacy bundle + StratifiedKFold + default params (current team baseline)",
        ),
        StageConfig(
            name="A1",
            group_aware_cv=True,
            fold_aware_surname=False,
            fold_aware_cabin_bin=False,
            use_common_features=False,
            params_override=dict(config.BASELINE_PARAMS),
            use_early_stopping=True,
            description="Legacy bundle + StratifiedGroupKFold + early stopping (honest CV of the status quo)",
        ),
        StageConfig(
            name="A2",
            group_aware_cv=True,
            fold_aware_surname=False,
            fold_aware_cabin_bin=False,
            use_common_features=True,
            params_override={},
            description="Common features + native category dtype + STRONG_PARAMS + early stopping",
        ),
        StageConfig(
            name="A3",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            use_common_features=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            params_override={},
            description="A2 + fold-aware SurnameFreq/CabinNumBin + fold-safe target encoding",
        ),
        StageConfig(
            name="A3b",
            group_aware_cv=True,
            fold_aware_surname=True,
            fold_aware_cabin_bin=True,
            use_common_features=True,
            target_encode_cols=config.TARGET_ENCODE_COLUMNS,
            use_oof_target_encoding=True,
            use_group_aggregates=True,
            params_override={},
            description="A3 + OOF target encoding + group-level aggregates",
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers for downstream scripts
# ---------------------------------------------------------------------------


def save_report(report: CVReport, out_dir: Path, tag: str) -> dict[str, Path]:
    """Serialise a :class:`CVReport` to CSVs + npz for later re-use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    oof_df = pd.DataFrame({"y_true": report.oof_true, "y_proba": report.oof_proba})
    oof_path = out_dir / f"{tag}_oof.csv"
    oof_df.to_csv(oof_path, index=False)
    paths["oof"] = oof_path

    test_df = pd.DataFrame({"PassengerId": report.passenger_id_test, "y_proba": report.test_proba})
    test_path = out_dir / f"{tag}_test_proba.csv"
    test_df.to_csv(test_path, index=False)
    paths["test_proba"] = test_path

    imp_df = pd.DataFrame(
        {
            "feature": report.feature_names,
            "gain": report.importance_gain,
            "weight": report.importance_weight,
            "cover": report.importance_cover,
        }
    ).sort_values("gain", ascending=False)
    imp_path = out_dir / f"{tag}_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    paths["importance"] = imp_path

    fold_df = pd.DataFrame(report.fold_scores)
    fold_path = out_dir / f"{tag}_fold_scores.csv"
    fold_df.to_csv(fold_path, index=False)
    paths["fold_scores"] = fold_path

    return paths
