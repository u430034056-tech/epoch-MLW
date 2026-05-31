"""Run the LightGBM feature-ablation workflow with exactly 30 selected features.

This script delegates to ``model_lightgbm1_feature_ablation.py`` so the training,
validation, prediction, and reporting logic stays identical to the main
feature-ablation workflow. The only intentional change is that feature
selection is fixed to 30 features.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


BASE_SCRIPT_PATH = Path(__file__).with_name("model_lightgbm1_feature_ablation.py")
TOP_FEATURE_COUNT = 30


def _load_base_module() -> Any:
    """Load the original feature-ablation module from its filename."""
    spec = importlib.util.spec_from_file_location("lightgbm1_feature_ablation_base", BASE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base LightGBM ablation script: {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base_module()


def __getattr__(name: str) -> Any:
    """Expose the original module's functions and constants for compatibility."""
    return getattr(_base, name)


def train_top_feature_model(**train_kwargs: Any) -> dict[str, Any]:
    """Train the same top-feature model, but force exactly 30 selected features."""
    train_kwargs.pop("min_features", None)
    train_kwargs.pop("max_features", None)
    train_kwargs.pop("selected_feature_count", None)
    return _base.train_top_feature_model(
        min_features=TOP_FEATURE_COUNT,
        max_features=TOP_FEATURE_COUNT,
        selected_feature_count=TOP_FEATURE_COUNT,
        **train_kwargs,
    )


if __name__ == "__main__":
    result = train_top_feature_model()

    print("Cross-validated accuracy @ 0.5:", result["cv_accuracy_at_0_5"])
    print("Cross-validated accuracy @ tuned threshold:", result["cv_accuracy"])
    print("Best threshold:", result["threshold"])
    print("Best params:", result["best_params"])
    print("Final n_estimators:", result["final_n_estimators"])
    print("CV group feature mode:", result["cv_group_feature_mode"])
    print("Final group feature mode:", result["final_group_feature_mode"])
    print("Extra feature mode:", result["extra_feature_mode"])
    print("Selected feature count:", result["feature_count"])
    print("Selected features:", result["feature_selection"]["selected_features"])

    model_path = _base.save_model(result, "artifacts/lightgbm1_top30_features_ensemble.joblib")
    submission_path = _base.build_submission(
        result,
        output_path="artifacts/submission_lightgbm1_top30_features.csv",
    )
    report_path = _base.save_feature_selection_report(
        result,
        "artifacts/lightgbm1_top30_features_report.csv",
    )

    print("Saved model to:", model_path)
    print("Saved submission to:", submission_path)
    print("Saved feature selection report to:", report_path)
    print("Prediction preview:")
    print(_base.predict_test_set(result, return_proba=True).head())
