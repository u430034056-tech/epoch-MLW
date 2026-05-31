from __future__ import annotations

from pathlib import Path
from typing import Any

from preprocess import get_project_paths, load_preprocessed_bundle, run_all_preprocessing


PREPROCESSING_MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "knn",
)


def load_or_run_preprocessing(project_root: str | Path | None = None, save_outputs: bool = True) -> dict[str, Any]:
    """Load saved preprocessing artifacts when present, otherwise run preprocessing."""
    paths = get_project_paths(project_root)
    expected = [paths["common_dir"] / "preprocessed_common.joblib"]
    expected.extend(
        paths[f"{model_name}_dir"] / f"preprocessed_{model_name}.joblib"
        for model_name in PREPROCESSING_MODEL_NAMES
    )

    if all(path.exists() for path in expected):
        print("[main] Existing preprocessing bundles detected. Loading from disk.")
        results: dict[str, Any] = {"common": load_preprocessed_bundle("common", paths["processed_root"])}
        for model_name in PREPROCESSING_MODEL_NAMES:
            results[model_name] = load_preprocessed_bundle(model_name, paths["processed_root"])
        return results

    print("[main] Preprocessing bundles not found. Running preprocessing now.")
    return run_all_preprocessing(project_root=project_root, save_outputs=save_outputs)


def main() -> dict[str, Any]:
    """Run preprocessing only and print the generated bundle keys."""
    results = load_or_run_preprocessing()
    print("[main] Top-level preprocessing result keys:", sorted(results.keys()))
    for model_name, bundle in results.items():
        save_path = bundle.get("save_path", "not-saved-in-memory")
        print(f"[main] {model_name}: save_path={save_path}")
        if model_name != "common":
            reloaded = load_preprocessed_bundle(model_name)
            print(f"[main] {model_name}: reloaded keys sample={sorted(reloaded.keys())[:8]}")
    return results


if __name__ == "__main__":
    main()
