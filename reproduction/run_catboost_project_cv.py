from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP3 = Path(__file__).resolve().parents[1]
SOURCE_DIR = GROUP3 / "sources" / "github_upload_project_kevinhe_full"
RAW_DATA_DIR = GROUP3 / "data" / "raw"
OUT_DIR = GROUP3 / "reproduction" / "outputs" / "catboost_project_cv"
WORK_PROJECT_ROOT = GROUP3 / "reproduction" / "work" / "catboost_project_cv_project"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_source() -> None:
    required = [
        "model_catboost.py",
        "preprocess.py",
        "inference_utils.py",
        "run_utils.py",
        "selftrain_utils.py",
    ]
    missing = [name for name in required if not (SOURCE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing upload/project-kevinhe source files: {missing}")


def prepare_project_root() -> None:
    reset_dir(WORK_PROJECT_ROOT)
    data_dir = WORK_PROJECT_ROOT / "spaceship-titanic"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        shutil.copy2(RAW_DATA_DIR / name, data_dir / name)


def validate_submission(path: Path) -> dict[str, Any]:
    sample = pd.read_csv(RAW_DATA_DIR / "sample_submission.csv")
    submission = pd.read_csv(path)
    id_match = submission["PassengerId"].astype(str).tolist() == sample["PassengerId"].astype(str).tolist()
    bool_like = set(submission["Transported"].dropna().astype(str).unique()).issubset({"True", "False", "0", "1"})
    return {
        "path": str(path),
        "shape": [int(submission.shape[0]), int(submission.shape[1])],
        "columns": submission.columns.tolist(),
        "id_order_match": bool(id_match),
        "boolean_like": bool(bool_like),
        "true_count": int(submission["Transported"].astype(str).isin(["True", "1"]).sum()),
    }


def main() -> None:
    ensure_source()
    reset_dir(OUT_DIR)
    prepare_project_root()

    sys.path.insert(0, str(SOURCE_DIR))
    import model_catboost as catboost_module  # noqa: PLC0415

    raw_train = pd.read_csv(RAW_DATA_DIR / "train.csv").reset_index(drop=True)
    params = {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
    }
    threshold_grid = catboost_module.build_threshold_grid(0.30, 0.70, 0.01)

    cv_start = time.perf_counter()
    trial_result = catboost_module._evaluate_cv_trial(
        raw_train,
        trial_index=1,
        params=params,
        cv_folds=5,
        random_state=42,
        early_stopping_rounds=100,
        threshold_grid=threshold_grid,
        threshold_metric="accuracy",
    )
    cv_seconds = time.perf_counter() - cv_start

    best_threshold = float(trial_result["recommended_threshold"])
    cv_results = pd.DataFrame([trial_result["trial_row"]])
    cv_results.to_csv(OUT_DIR / "cv_results.csv", index=False)
    pd.DataFrame(trial_result["fold_metrics"]).to_csv(OUT_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(trial_result["threshold_search"]["threshold_results"]).to_csv(
        OUT_DIR / "threshold_search.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "PassengerId": raw_train["PassengerId"].astype(str),
            "y_true": raw_train["Transported"].astype(int),
            "oof_proba": np.asarray(trial_result["oof_probabilities"], dtype=float),
            "recommended_pred": (
                np.asarray(trial_result["oof_probabilities"], dtype=float) >= best_threshold
            ).astype(int),
            "fold": np.asarray(trial_result["oof_folds"], dtype=int),
        }
    ).to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    final_start = time.perf_counter()
    final_result = catboost_module.run_training_pipeline(
        project_root=WORK_PROJECT_ROOT,
        artifacts_dir=OUT_DIR / "artifacts",
        submissions_dir=OUT_DIR / "submissions",
        processed_root=WORK_PROJECT_ROOT / "processed",
        config=params,
        mode="final_train",
        threshold=best_threshold,
    )
    final_seconds = time.perf_counter() - final_start

    submission_path = Path(final_result["submission_path"])
    submission_validation = validate_submission(submission_path)

    summary = {
        "model": "CatBoost",
        "source_branch": "origin/upload/project-kevinhe",
        "source_dir": str(SOURCE_DIR),
        "protocol": "fixed report project-side config, 5-fold StratifiedKFold OOF threshold search",
        "params": params,
        "cv_folds": 5,
        "random_state": 42,
        "early_stopping_rounds": 100,
        "threshold_grid": {"start": 0.30, "end": 0.70, "step": 0.01},
        "threshold_metric": "accuracy",
        "best_threshold": best_threshold,
        "mean_cv_accuracy": trial_result["trial_row"]["mean_cv_accuracy"],
        "mean_cv_auc": trial_result["trial_row"]["mean_cv_auc"],
        "oof_accuracy": trial_result["trial_row"]["oof_accuracy"],
        "oof_auc": trial_result["trial_row"]["oof_auc"],
        "oof_f1": trial_result["trial_row"]["oof_f1"],
        "fold_best_iterations": trial_result["fold_best_iterations"],
        "cv_seconds": cv_seconds,
        "final_train_seconds": final_seconds,
        "final_train_summary": final_result["train_summary"],
        "final_train_run": final_result["train_run"],
        "managed_train_dir": final_result["managed_train_dir"],
        "submission_validation": submission_validation,
    }
    write_json(OUT_DIR / "catboost_project_cv_summary.json", summary)

    lines = [
        "# CatBoost Project 5-Fold CV Summary",
        "",
        "- Source: `origin/upload/project-kevinhe` full export.",
        "- Protocol: fixed report project-side CatBoost config with 5-fold OOF threshold search.",
        "- Config: `iterations=2000`, `learning_rate=0.03`, `depth=6`, `l2_leaf_reg=5.0`.",
        "- CV folds: `5`, random_state: `42`, early stopping rounds: `100`.",
        f"- Best OOF threshold: `{best_threshold:.2f}`.",
        f"- Mean CV accuracy: `{trial_result['trial_row']['mean_cv_accuracy']:.6f}`.",
        f"- OOF accuracy: `{trial_result['trial_row']['oof_accuracy']:.6f}`.",
        f"- OOF ROC-AUC: `{trial_result['trial_row']['oof_auc']:.6f}`.",
        f"- Final train accuracy: `{final_result['train_summary']['train_accuracy']:.6f}`.",
        f"- Submission: `{submission_path}`.",
        f"- Submission valid: `{submission_validation['id_order_match'] and submission_validation['boolean_like']}`.",
        "",
        "This supplements the existing strict 80/20 CatBoost rerun; it is the report/PPT project-workflow lane.",
    ]
    (OUT_DIR / "CATBOOST_PROJECT_CV_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "CATBOOST_PROJECT_CV_SUMMARY.md")


if __name__ == "__main__":
    main()
