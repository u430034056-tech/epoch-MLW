from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from preprocess import get_project_paths, load_preprocessed_bundle, run_all_preprocessing


@dataclass(frozen=True)
class RFRunOutputs:
    submission_path: Path
    model_path: Path | None
    tuning_summary_path: Path | None


def load_data(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load the Random Forest preprocessing bundle produced by `preprocess.py`."""
    paths = get_project_paths(project_root)
    bundle_path = paths["random_forest_dir"] / "preprocessed_random_forest.joblib"
    if bundle_path.exists():
        return load_preprocessed_bundle("random_forest", paths["processed_root"])
    # If the bundle isn't on disk yet, run preprocessing to generate it.
    run_all_preprocessing(project_root=paths["project_root"], save_outputs=True)
    return load_preprocessed_bundle("random_forest", paths["processed_root"])


def evaluate_cv_accuracy(model: RandomForestClassifier, X: Any, y: np.ndarray, *, seed: int = 42) -> float:
    """Return mean CV accuracy using out-of-fold predictions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=1)
    return float(accuracy_score(y, oof))


def _find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Find probability threshold that maximizes accuracy on given predictions."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    # Dense sweep is fast enough for this dataset size.
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr = 0.5
    best_acc = -1.0
    for thr in thresholds:
        acc = float(accuracy_score(y_true, (proba >= thr).astype(int)))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def tune_hyperparams(X: Any, y: np.ndarray, *, seed: int = 42, n_iter: int = 90) -> RandomizedSearchCV:
    """Randomized hyperparameter search for RandomForestClassifier."""
    base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Split the space by bootstrap mode so we never sample invalid combinations.
    common_space: dict[str, Any] = {
        "n_estimators": list(range(800, 3401, 200)),
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 8, 10, 12, 14, 16, 20, 24, 30, 40],
        "min_samples_split": [2, 3, 4, 5, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", None, 0.2, 0.3, 0.35, 0.4, 0.5, 0.65],
        "max_leaf_nodes": [None, 96, 128, 160, 220, 320, 480],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "ccp_alpha": [0.0, 1e-5, 3e-5, 1e-4, 3e-4],
    }
    param_distributions: list[dict[str, Any]] = [
        {
            **common_space,
            "bootstrap": [True],
            "max_samples": [None, 0.55, 0.7, 0.85, 0.95],
        },
        {
            **common_space,
            "bootstrap": [False],
        },
    ]

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Best configuration found from the latest RF tuning run.
BEST_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 2600,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_leaf_nodes": 128,
    "max_features": 0.5,
    "max_depth": 16,
    "criterion": "log_loss",
    "class_weight": "balanced",
    "ccp_alpha": 3e-05,
    "bootstrap": False,
}
BEST_RF_THRESHOLD = 0.51


def run(
    *,
    project_root: str | Path | None = None,
    tune: bool = False,
    seed: int = 42,
    n_iter: int = 90,
) -> RFRunOutputs:
    data = load_data(project_root)

    X_train = data["X_train"]
    y_train = np.asarray(data["y_train"], dtype=int)
    X_test = data["X_test"]
    test_ids = data["test_ids"]

    outputs_dir = Path("outputs") / "models"
    _ensure_dir(outputs_dir)

    if tune:
        search = tune_hyperparams(X_train, y_train, seed=seed, n_iter=n_iter)
        best_model: RandomForestClassifier = search.best_estimator_
        cv_acc = float(search.best_score_)

        # Post-tuning: compute OOF probabilities and pick an accuracy-optimal threshold.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        best_thr, thr_acc = _find_best_threshold(y_train, oof_proba)

        tuning_summary = {
            "model": "random_forest",
            "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True)",
            "scoring": "accuracy",
            "best_cv_accuracy": cv_acc,
            "oof_best_threshold": best_thr,
            "oof_accuracy_at_best_threshold": thr_acc,
            "best_params": search.best_params_,
        }
        tuning_summary_path = outputs_dir / "random_forest_tuning_summary.json"
        tuning_summary_path.write_text(json.dumps(tuning_summary, indent=2), encoding="utf-8")
        print(f"[random_forest] best CV accuracy={cv_acc:.5f}")
        print(f"[random_forest] best OOF threshold={best_thr:.3f}, OOF accuracy={thr_acc:.5f}")
        print(f"[random_forest] best params={search.best_params_}")
    else:
        best_model = RandomForestClassifier(
            **BEST_RF_PARAMS,
            random_state=seed,
            n_jobs=-1,
        )
        cv_acc = evaluate_cv_accuracy(best_model, X_train, y_train, seed=seed)
        tuning_summary_path = None
        print(f"[random_forest] fixed-params CV accuracy={cv_acc:.5f}")
        best_model.fit(X_train, y_train)
        best_thr = BEST_RF_THRESHOLD

    proba = best_model.predict_proba(X_test)[:, 1]
    preds = (proba >= best_thr)

    submission = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Transported": preds.astype(bool),
        }
    )
    submission_path = Path("submission_rf.csv")
    submission.to_csv(submission_path, index=False)
    print(f"[random_forest] wrote {submission_path} (True/False)")

    return RFRunOutputs(
        submission_path=submission_path,
        model_path=None,
        tuning_summary_path=tuning_summary_path,
    )


if __name__ == "__main__":
    run(tune=False)
