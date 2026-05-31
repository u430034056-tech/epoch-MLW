from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


GROUP3 = Path(__file__).resolve().parents[1]
SOURCE_DIR = GROUP3 / "sources" / "github_feature_random_forest_full"
LOCAL_PACKAGE_DIR = GROUP3 / "sources" / "random_forest_local_package"
ORIGINAL_PACKAGE_SUBMISSION = (
    GROUP3 / "reproduction" / "outputs" / "random_forest_5fold_package" / "submission_rf_original_package.csv"
)
RAW_DATA_DIR = GROUP3 / "data" / "raw"
OUT_DIR = GROUP3 / "reproduction" / "outputs" / "random_forest_github_branch"
RUN_DIR = GROUP3 / "reproduction" / "work" / "random_forest_github_branch" / "source"
LOG_PATH = GROUP3 / "reproduction" / "logs" / "random_forest_github_branch_run.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def prepare_branch_data() -> None:
    data_dir = RUN_DIR / "spaceship-titanic"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        shutil.copy2(RAW_DATA_DIR / name, data_dir / name)


def prepare_branch_run_dir() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    shutil.copytree(
        SOURCE_DIR,
        RUN_DIR,
        ignore=shutil.ignore_patterns("processed", "spaceship-titanic", "submission_rf.csv", "__pycache__", ".DS_Store"),
    )


def compare_predictions(left: Path, right: Path) -> dict[str, Any]:
    left_df = pd.read_csv(left)
    right_df = pd.read_csv(right)
    aligned = left_df["PassengerId"].astype(str).tolist() == right_df["PassengerId"].astype(str).tolist()
    if not aligned:
        return {"id_order_match": False, "prediction_diff_count": None}
    diff_count = int((left_df["Transported"].astype(str) != right_df["Transported"].astype(str)).sum())
    return {"id_order_match": True, "prediction_diff_count": diff_count}


def main() -> None:
    reset_dir(OUT_DIR)
    prepare_branch_run_dir()
    prepare_branch_data()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "model_random_forest.py"]
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        subprocess.run(cmd, cwd=RUN_DIR, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    branch_submission = RUN_DIR / "submission_rf.csv"
    branch_tuning_summary = RUN_DIR / "outputs" / "models" / "random_forest_tuning_summary.json"
    copied_submission = OUT_DIR / "submission_rf_github_feature_branch.csv"
    copied_tuning = OUT_DIR / "random_forest_tuning_summary_github_feature_branch.json"
    shutil.copy2(branch_submission, copied_submission)
    shutil.copy2(branch_tuning_summary, copied_tuning)
    shutil.copy2(SOURCE_DIR / "model_random_forest.py", OUT_DIR / "model_random_forest_github_feature_branch.py")

    validation = validate_submission(copied_submission)
    package_submission = LOCAL_PACKAGE_DIR / "submission_rf.csv"
    prediction_compare = compare_predictions(copied_submission, package_submission) if package_submission.exists() else None
    original_package_compare = (
        compare_predictions(copied_submission, ORIGINAL_PACKAGE_SUBMISSION)
        if ORIGINAL_PACKAGE_SUBMISSION.exists()
        else None
    )
    source_hash_compare = {
        "github_feature_model_sha256": sha256(SOURCE_DIR / "model_random_forest.py"),
        "local_package_model_sha256": sha256(LOCAL_PACKAGE_DIR / "model_random_forest.py"),
        "model_files_identical": sha256(SOURCE_DIR / "model_random_forest.py")
        == sha256(LOCAL_PACKAGE_DIR / "model_random_forest.py"),
    }
    tuning_summary = json.loads(copied_tuning.read_text(encoding="utf-8"))

    summary = {
        "model": "Random Forest",
        "source_branch": "origin/feature/random_forest",
        "source_dir": str(SOURCE_DIR),
        "run_dir": str(RUN_DIR),
        "protocol": "GitHub feature branch fixed-params 5-fold CV run, tune=False",
        "validation": validation,
        "source_hash_compare": source_hash_compare,
        "prediction_compare_to_local_package_submission": prediction_compare,
        "prediction_compare_to_original_package_submission": original_package_compare,
        "archived_tuning_summary": tuning_summary,
        "log_path": str(LOG_PATH),
    }
    write_json(OUT_DIR / "random_forest_github_branch_summary.json", summary)

    lines = [
        "# Random Forest GitHub Feature Branch Summary",
        "",
        "- Source: `origin/feature/random_forest` full export.",
        "- Run command: `.venv/bin/python model_random_forest.py` with `tune=False`.",
        "- Protocol in code: fixed tuned RF params, 5-fold CV accuracy check, threshold `0.51`.",
        f"- Branch submission: `{copied_submission}`.",
        f"- Submission valid: `{validation['id_order_match'] and validation['boolean_like']}`.",
        f"- Model file identical to local RF package: `{source_hash_compare['model_files_identical']}`.",
    ]
    if prediction_compare is not None:
        lines.append(
            f"- Prediction diff versus current local package rerun submission: `{prediction_compare['prediction_diff_count']}` rows."
        )
    if original_package_compare is not None:
        lines.append(
            f"- Prediction diff versus original packaged submission copy: `{original_package_compare['prediction_diff_count']}` rows."
        )
    lines.extend(
        [
            f"- Archived best CV accuracy: `{tuning_summary.get('best_cv_accuracy')}`.",
            f"- Archived OOF threshold accuracy: `{tuning_summary.get('oof_accuracy_at_best_threshold')}`.",
        ]
    )
    (OUT_DIR / "RANDOM_FOREST_GITHUB_BRANCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "RANDOM_FOREST_GITHUB_BRANCH_SUMMARY.md")


if __name__ == "__main__":
    main()
