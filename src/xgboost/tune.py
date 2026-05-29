"""Optuna Bayesian search for XGBoost hyper-parameters.

The search reuses :func:`train.run_cv` so every trial enjoys the same honest
group-aware cross-validation + fold-aware preprocessing + early stopping.  The
trial objective is OOF accuracy (the Kaggle metric).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler

from . import config
from .train import StageConfig, run_cv


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one XGBoost configuration from the search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 7),
    }


def _objective_factory(base_stage: StageConfig):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        stage = StageConfig(
            name=f"tune-{trial.number}",
            group_aware_cv=base_stage.group_aware_cv,
            fold_aware_surname=base_stage.fold_aware_surname,
            fold_aware_cabin_bin=base_stage.fold_aware_cabin_bin,
            target_encode_cols=base_stage.target_encode_cols,
            use_common_features=base_stage.use_common_features,
            params_override=params,
            use_early_stopping=True,
            early_stopping_rounds=base_stage.early_stopping_rounds,
            description=f"Optuna trial {trial.number}",
        )
        report = run_cv(stage, verbose=False)
        trial.set_user_attr("logloss", report.oof_logloss)
        trial.set_user_attr("auc", report.oof_auc)
        trial.set_user_attr("mean_best_iter", float(np.mean(report.best_iterations)))
        return report.oof_acc

    return objective


def run_optuna(
    base_stage: StageConfig,
    n_trials: int = config.OPTUNA_TRIALS,
    timeout: int | None = config.OPTUNA_TIMEOUT_S,
    study_dir: Path | None = None,
) -> dict:
    """Run the Optuna study and persist the best config / study dump."""
    study_dir = study_dir or config.LOGS_DIR
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'optuna_study.db').resolve()}"

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_SEED, multivariate=True),
        study_name="spaceship_xgb",
        storage=storage,
        load_if_exists=True,
    )
    t0 = time.perf_counter()
    study.optimize(_objective_factory(base_stage), n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    best = study.best_trial
    out = {
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "best_trial_number": int(best.number),
        "n_trials": int(len(study.trials)),
        "elapsed_seconds": float(elapsed),
        "logloss": float(best.user_attrs.get("logloss", float("nan"))),
        "auc": float(best.user_attrs.get("auc", float("nan"))),
        "mean_best_iter": float(best.user_attrs.get("mean_best_iter", float("nan"))),
    }
    (study_dir / "best_params.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Full trial log (useful in the report)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_dir / "optuna_trials.csv", index=False)
    return out
