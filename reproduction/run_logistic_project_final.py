from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


GROUP3 = Path(__file__).resolve().parents[1]
MAIN_SOURCE = GROUP3 / "sources" / "model_code_selected" / "main"
HELPER_SOURCE = GROUP3 / "sources" / "github_upload_project_kevinhe_full"
RAW_DATA_DIR = GROUP3 / "data" / "raw"
OUT_DIR = GROUP3 / "reproduction" / "outputs" / "logistic_project_final"
WORK_DIR = GROUP3 / "reproduction" / "work" / "logistic_project_final"
SOURCE_DIR = WORK_DIR / "source"
PROJECT_ROOT = WORK_DIR / "project_root"
LOG_PATH = GROUP3 / "reproduction" / "logs" / "logistic_project_final_run.log"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_sources() -> None:
    reset_dir(SOURCE_DIR)
    for name in ["inference_utils.py", "run_utils.py", "selftrain_utils.py"]:
        shutil.copy2(HELPER_SOURCE / name, SOURCE_DIR / name)
    for name in ["model_logistic_regression.py", "preprocess.py"]:
        shutil.copy2(MAIN_SOURCE / name, SOURCE_DIR / name)


def prepare_project_root() -> None:
    reset_dir(PROJECT_ROOT)
    data_dir = PROJECT_ROOT / "spaceship-titanic"
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
    reset_dir(OUT_DIR)
    prepare_sources()
    prepare_project_root()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "model_logistic_regression.py",
        "--project-root",
        str(PROJECT_ROOT),
        "--artifacts-dir",
        str(OUT_DIR / "artifacts"),
        "--submissions-dir",
        str(OUT_DIR / "submissions"),
        "--mode",
        "final_train",
        "--threshold",
        "0.5",
    ]
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        subprocess.run(cmd, cwd=SOURCE_DIR, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    submission_path = OUT_DIR / "submissions" / "submission_logistic_regression.csv"
    validation = validate_submission(submission_path)
    train_summaries = sorted((OUT_DIR / "artifacts" / "train" / "logistic_regression").glob("*/train_summary.json"))
    if not train_summaries:
        raise FileNotFoundError("No managed Logistic Regression train_summary.json was produced.")
    train_summary_path = train_summaries[-1]
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    coefficient_path = OUT_DIR / "artifacts" / "logistic_regression" / "coefficients.csv"
    feature_count = int(pd.read_csv(coefficient_path).shape[0]) if coefficient_path.exists() else None

    summary = {
        "model": "Logistic Regression",
        "source_model_script": str(MAIN_SOURCE / "model_logistic_regression.py"),
        "source_preprocess_script": str(MAIN_SOURCE / "preprocess.py"),
        "helper_source": str(HELPER_SOURCE),
        "note": "main@8f61ec2 LR script requires helper modules absent from main, so helper APIs are supplied from upload/project-kevinhe.",
        "protocol": "project-side final_train, saga l2 C=1.0, threshold=0.5",
        "train_summary": train_summary,
        "feature_count": feature_count,
        "coefficient_path": str(coefficient_path) if coefficient_path.exists() else None,
        "train_summary_path": str(train_summary_path),
        "submission_validation": validation,
        "log_path": str(LOG_PATH),
    }
    write_json(OUT_DIR / "logistic_project_final_summary.json", summary)

    lines = [
        "# Logistic Regression Project Script Summary",
        "",
        "- Model script source: `sources/model_code_selected/main/model_logistic_regression.py`.",
        "- Preprocess source: `sources/model_code_selected/main/preprocess.py`.",
        "- Runtime helper source: `sources/github_upload_project_kevinhe_full/`.",
        "- Protocol: project-side `final_train`, `solver=saga`, `penalty=l2`, `C=1.0`, threshold `0.5`.",
        f"- Train accuracy: `{train_summary['train_accuracy']:.6f}`.",
        f"- Feature count: `{feature_count}`.",
        f"- Submission: `{submission_path}`.",
        f"- Submission valid: `{validation['id_order_match'] and validation['boolean_like']}`.",
        "",
        "This is separate from the strict 80/20 LR validation table and the PPT `lbfgs` baseline.",
    ]
    (OUT_DIR / "LOGISTIC_PROJECT_FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "LOGISTIC_PROJECT_FINAL_SUMMARY.md")


if __name__ == "__main__":
    main()
