from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


GROUP3 = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = GROUP3 / "sources" / "lightgbm_submission" / "model_lightgbm1_feature_ablation.py"
DATA_DIR = GROUP3 / "data" / "raw"
OUT_DIR = GROUP3 / "reproduction" / "outputs" / "lightgbm_top30"
LOG_DIR = GROUP3 / "reproduction" / "logs" / "lightgbm_top30"
FEATURE_COUNTS = (20, 30, 40)


def load_module() -> Any:
    sys.path.insert(0, str(SOURCE_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("group3_lightgbm_top_features", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_submission(path: Path) -> dict[str, Any]:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = pd.read_csv(path)
    return {
        "path": str(path),
        "shape": [int(submission.shape[0]), int(submission.shape[1])],
        "columns": submission.columns.tolist(),
        "id_order_match": bool(
            submission["PassengerId"].astype(str).tolist() == sample["PassengerId"].astype(str).tolist()
        ),
        "transported_boolean_like": bool(
            set(submission["Transported"].dropna().astype(str).unique()).issubset({"True", "False", "0", "1"})
        ),
        "true_count": int(submission["Transported"].astype(bool).sum()),
        "positive_rate": float(submission["Transported"].astype(bool).mean()),
    }


def run_variant(module: Any, feature_count: int, artifacts_dir: Path) -> dict[str, Any]:
    result = module.train_top_feature_model(
        min_features=feature_count,
        max_features=feature_count,
        selected_feature_count=feature_count,
    )
    model_path = module.save_model(result, artifacts_dir / f"lightgbm1_top{feature_count}_features_ensemble.joblib")
    submission_path = module.build_submission(
        result,
        output_path=artifacts_dir / f"submission_lightgbm1_top{feature_count}_features.csv",
    )
    report_path = module.save_feature_selection_report(
        result,
        artifacts_dir / f"lightgbm1_top{feature_count}_features_report.csv",
    )
    preview_path = artifacts_dir / f"lightgbm1_top{feature_count}_prediction_preview.csv"
    module.predict_test_set(result, return_proba=True).head(20).to_csv(preview_path, index=False)
    return {
        "model": f"LightGBM top-{feature_count} feature",
        "source_script": str(SOURCE_SCRIPT),
        "cv_accuracy_at_0_5": float(result["cv_accuracy_at_0_5"]),
        "tuned_cv_accuracy": float(result["cv_accuracy"]),
        "threshold": float(result["threshold"]),
        "final_n_estimators": int(result["final_n_estimators"]),
        "selected_feature_count": int(result["feature_count"]),
        "selected_features": [str(feature) for feature in result["feature_selection"]["selected_features"]],
        "cv_group_feature_mode": str(result["cv_group_feature_mode"]),
        "final_group_feature_mode": str(result["final_group_feature_mode"]),
        "extra_feature_mode": str(result["extra_feature_mode"]),
        "best_params": result["best_params"],
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "feature_report_path": str(report_path),
        "prediction_preview_path": str(preview_path),
        "submission_validation": validate_submission(Path(submission_path)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_dir = OUT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    module = load_module()
    variants = []
    for feature_count in FEATURE_COUNTS:
        print(f"[lightgbm-top] running top {feature_count}")
        variants.append(run_variant(module, feature_count, artifacts_dir))

    summary = {
        "model_family": "LightGBM top-feature",
        "source_script": str(SOURCE_SCRIPT),
        "variants": variants,
        "selected_report_variant": next(item for item in variants if item["selected_feature_count"] == 30),
    }
    (OUT_DIR / "lightgbm_top_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LightGBM Top-Feature Reproduction",
        "",
        f"Source script: `{SOURCE_SCRIPT}`",
        "",
        "| Variant | CV @ 0.5 | Tuned CV | Threshold | Final trees | Submission valid |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in variants:
        lines.append(
            f"| Top {variant['selected_feature_count']} | {variant['cv_accuracy_at_0_5']:.4f} | "
            f"{variant['tuned_cv_accuracy']:.4f} | {variant['threshold']:.3f} | "
            f"{variant['final_n_estimators']} | {variant['submission_validation']['id_order_match']} |"
        )
    lines.extend(
        [
            "",
            "## Submission Validation",
            "",
            json.dumps([variant["submission_validation"] for variant in variants], indent=2),
        ]
    )
    (OUT_DIR / "LIGHTGBM_TOP30_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_lines = [
        "LightGBM top-feature reproduction complete",
        json.dumps(summary, sort_keys=True),
    ]
    (LOG_DIR / "lightgbm_top30.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(OUT_DIR / "LIGHTGBM_TOP30_SUMMARY.md")


if __name__ == "__main__":
    main()
