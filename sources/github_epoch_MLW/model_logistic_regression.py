"""Training and inference utilities for the Logistic Regression branch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from inference_utils import (
    build_submission_frame,
    extract_bundle_test_features,
    load_joblib_with_pandas_compat,
    read_test_dataframe,
    resolve_submission_output_path,
    save_inference_artifacts,
    validate_bundle_test_ids,
    validate_test_csv_schema,
)
from preprocess import get_project_paths, load_preprocessed_bundle as load_saved_preprocessed_bundle, run_all_preprocessing
from run_utils import (
    MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR,
    allocate_infer_run,
    create_infer_run_dirs,
    current_timestamp,
    dedupe_preserve_order,
    ensure_infer_run_available,
    get_latest_train_run,
    get_legacy_artifact_dir,
    get_next_run_id,
    get_run_root,
    list_existing_runs,
    raise_run_error,
    reject_minimal_runtime_train_data_path,
    relative_to_project,
    resolve_legacy_model_artifact,
    resolve_train_run_dir,
    resolve_model_artifact_from_train_run,
    update_latest_run,
    update_run_registry,
    validate_train_run_meta,
    write_json,
)
from selftrain_utils import (
    DEFAULT_INFER_BUNDLE_MODE,
    EXPLICIT_TRAIN_DATA_BUNDLE_MODE,
    build_explicit_training_bundle,
)


MODEL_NAME = "logistic_regression"
LEGACY_MODEL_FILENAME = "model.joblib"
DEFAULT_CONFIG = {
    "solver": "saga",
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 5000,
    "tol": 1e-4,
    "random_state": 42,
    "class_weight": None,
}
REQUIRED_BUNDLE_KEYS = ("X_train", "X_test", "y_train", "train_ids", "test_ids")


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root is not None else Path.cwd()


def _resolve_processed_root(
    processed_root: str | Path | None = None, project_root: str | Path | None = None
) -> Path:
    if processed_root is not None:
        return Path(processed_root)
    return get_project_paths(project_root)["processed_root"]


def _resolve_artifact_dir(
    artifacts_dir: str | Path | None = None, project_root: str | Path | None = None
) -> Path:
    if artifacts_dir is not None:
        return Path(artifacts_dir) / MODEL_NAME
    return _resolve_project_root(project_root) / "artifacts" / MODEL_NAME


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _validate_non_infer_args(args: argparse.Namespace) -> None:
    invalid_flags: list[str] = []
    if getattr(args, "train_run", None):
        invalid_flags.append("--train-run")
    if getattr(args, "infer_run", None):
        invalid_flags.append("--infer-run")
    if invalid_flags:
        raise ValueError(
            f"[{MODEL_NAME}] {', '.join(invalid_flags)} is only supported when --mode infer is used."
        )


def _managed_model_candidate_names() -> list[str]:
    return dedupe_preserve_order(["model.joblib", "model.pkl", LEGACY_MODEL_FILENAME])


def _is_infer_compatible_train_meta(train_meta: dict[str, Any]) -> bool:
    return train_meta.get("infer_bundle_mode", DEFAULT_INFER_BUNDLE_MODE) == DEFAULT_INFER_BUNDLE_MODE


def _resolve_bundle_path(
    processed_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    processed_root = _resolve_processed_root(processed_root=processed_dir, project_root=project_root)
    return processed_root / MODEL_NAME / f"preprocessed_{MODEL_NAME}.joblib"


def _resolve_compatibility_submission_path(
    *,
    project_root: str | Path | None,
    submissions_dir: str | Path | None,
) -> Path:
    return resolve_submission_output_path(
        project_root=project_root,
        submissions_dir=submissions_dir,
        model_name=MODEL_NAME,
        output_name=None,
    )


def _resolve_optional_output_path(
    *,
    project_root: str | Path | None,
    submissions_dir: str | Path | None,
    output_name: str | None,
) -> Path | None:
    if output_name is None:
        return None
    return resolve_submission_output_path(
        project_root=project_root,
        submissions_dir=submissions_dir,
        model_name=MODEL_NAME,
        output_name=output_name,
    )


def _resolve_train_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_run:
        train_run_dir = resolve_train_run_dir(
            project_root=args.project_root,
            artifacts_dir=args.artifacts_dir,
            model_name=MODEL_NAME,
            train_run=args.train_run,
        )
        train_meta = validate_train_run_meta(train_run_dir, model_name=MODEL_NAME, train_run=args.train_run)
        model_path = resolve_model_artifact_from_train_run(
            train_run_dir,
            model_name=MODEL_NAME,
            train_run=args.train_run,
            candidate_names=_managed_model_candidate_names(),
        )
        if not _is_infer_compatible_train_meta(train_meta):
            raise_run_error(
                model_name=MODEL_NAME,
                stage="infer",
                run_id=args.train_run,
                message="This managed train run was created from explicit train data and is not compatible with default infer bundle loading.",
                attempted_paths=[train_run_dir / "run_meta.json"],
                fix_hint="Use a default processed-bundle train run for infer mode, or rely on the training-time submission for this explicit-train run.",
            )
        return {
            "source_mode": "managed_train_run",
            "train_run": args.train_run,
            "train_run_dir": train_run_dir,
            "train_meta": train_meta,
            "model_path": model_path,
        }

    train_root = get_run_root(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        stage="train",
        model_name=MODEL_NAME,
    )
    for latest_train_run in reversed(list_existing_runs(train_root, "train")):
        train_run_dir = resolve_train_run_dir(
            project_root=args.project_root,
            artifacts_dir=args.artifacts_dir,
            model_name=MODEL_NAME,
            train_run=latest_train_run,
        )
        train_meta = validate_train_run_meta(train_run_dir, model_name=MODEL_NAME, train_run=latest_train_run)
        if not _is_infer_compatible_train_meta(train_meta):
            continue
        model_path = resolve_model_artifact_from_train_run(
            train_run_dir,
            model_name=MODEL_NAME,
            train_run=latest_train_run,
            candidate_names=_managed_model_candidate_names(),
        )
        return {
            "source_mode": "managed_train_run",
            "train_run": latest_train_run,
            "train_run_dir": train_run_dir,
            "train_meta": train_meta,
            "model_path": model_path,
        }

    model_path = resolve_legacy_model_artifact(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        model_name=MODEL_NAME,
        legacy_model_filename=LEGACY_MODEL_FILENAME,
    )
    return {
        "source_mode": "legacy_artifact_fallback",
        "train_run": None,
        "train_run_dir": None,
        "train_meta": None,
        "model_path": model_path,
    }


def _check_explicit_infer_run(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.infer_run:
        return None
    artifact_dir, submission_dir, submission_path = ensure_infer_run_available(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        submissions_dir=args.submissions_dir,
        model_name=MODEL_NAME,
        infer_run=args.infer_run,
    )
    return {
        "infer_run": args.infer_run,
        "artifact_dir": artifact_dir,
        "submission_dir": submission_dir,
        "submission_path": submission_path,
    }


def _allocate_infer_context(args: argparse.Namespace) -> dict[str, Any]:
    if args.infer_run:
        checked = _check_explicit_infer_run(args)
        if checked is None:
            raise AssertionError("Explicit infer run should have been validated.")
        return checked

    infer_run, artifact_dir, submission_dir, submission_path = allocate_infer_run(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        submissions_dir=args.submissions_dir,
        model_name=MODEL_NAME,
    )
    return {
        "infer_run": infer_run,
        "artifact_dir": artifact_dir,
        "submission_dir": submission_dir,
        "submission_path": submission_path,
    }


def _build_source_train_payload(
    *,
    source_context: dict[str, Any],
    bundle_path: Path,
    project_root: str | Path | None,
) -> dict[str, Any]:
    payload = {
        "model_name": MODEL_NAME,
        "source_train_run": source_context["train_run"],
        "source_mode": source_context["source_mode"],
        "source_model_path": relative_to_project(source_context["model_path"], project_root),
        "source_bundle_path": relative_to_project(bundle_path, project_root),
    }
    return payload


def _persist_train_run_metadata(
    *,
    training_result: dict[str, Any],
    train_run: str,
    train_dir: Path,
    project_root: str | Path | None,
) -> None:
    metadata = training_result["metadata"]
    bundle_path = metadata.get("bundle_path")
    train_data_path = metadata.get("train_data_path")
    source_test_path = metadata.get("source_test_path")
    if bundle_path is None and train_data_path is None:
        bundle_path = _resolve_bundle_path(
            processed_dir=None,
            project_root=project_root,
        )
    write_json(train_dir / "train_config.json", training_result["config"])
    write_json(train_dir / "train_summary.json", training_result["train_summary"])
    bundle_ref_payload = {
        "bundle_path": relative_to_project(bundle_path, project_root) if bundle_path is not None else None,
        "bundle_role": "training_and_test_feature_source",
    }
    if train_data_path is not None:
        bundle_ref_payload["train_data_path"] = relative_to_project(train_data_path, project_root)
    if source_test_path is not None:
        bundle_ref_payload["source_test_path"] = relative_to_project(source_test_path, project_root)
    write_json(
        train_dir / "bundle_ref.json",
        bundle_ref_payload,
    )
    write_json(
        train_dir / "run_meta.json",
        {
            "run_id": train_run,
            "stage": "train",
            "model_name": MODEL_NAME,
            "created_at": current_timestamp(),
            "project_root": str(_resolve_project_root(project_root).resolve()),
            "artifact_dir": relative_to_project(train_dir, project_root),
            "bundle_path": relative_to_project(bundle_path, project_root) if bundle_path is not None else None,
            "train_data_path": relative_to_project(train_data_path, project_root) if train_data_path is not None else None,
            "source_test_path": relative_to_project(source_test_path, project_root)
            if source_test_path is not None
            else None,
            "infer_bundle_mode": metadata.get("infer_bundle_mode", DEFAULT_INFER_BUNDLE_MODE),
            "train_data_mode": metadata.get("train_data_mode"),
            "source_dataset_run": metadata.get("source_dataset_run"),
            "source_selftrain_run": metadata.get("source_selftrain_run"),
            "source_infer_run": metadata.get("source_infer_run"),
        },
    )


def _persist_infer_metadata(
    *,
    infer_context: dict[str, Any],
    source_context: dict[str, Any],
    prepared_test: dict[str, Any],
    compatibility_submission_path: Path,
    project_root: str | Path | None,
    threshold: float,
    save_proba: bool,
) -> None:
    source_payload = _build_source_train_payload(
        source_context=source_context,
        bundle_path=prepared_test["bundle_path"],
        project_root=project_root,
    )
    infer_config = {
        "mode": "infer",
        "threshold": float(threshold),
        "save_proba": bool(save_proba),
        "test_path": relative_to_project(prepared_test["raw_test_path"], project_root),
        "bundle_path": relative_to_project(prepared_test["bundle_path"], project_root),
        "source_train_run": source_context["train_run"],
        "source_mode": source_context["source_mode"],
        "model_path": relative_to_project(source_context["model_path"], project_root),
    }
    run_meta = {
        "run_id": infer_context["infer_run"],
        "stage": "infer",
        "model_name": MODEL_NAME,
        "created_at": current_timestamp(),
        "project_root": str(_resolve_project_root(project_root).resolve()),
        "source_train_run": source_context["train_run"],
        "source_mode": source_context["source_mode"],
        "artifact_dir": relative_to_project(infer_context["artifact_dir"], project_root),
        "submission_path": relative_to_project(infer_context["submission_path"], project_root),
        "compatibility_submission_path": relative_to_project(compatibility_submission_path, project_root),
        "test_path": relative_to_project(prepared_test["raw_test_path"], project_root),
        "threshold": float(threshold),
        "save_proba": bool(save_proba),
    }
    write_json(infer_context["artifact_dir"] / "infer_config.json", infer_config)
    write_json(infer_context["artifact_dir"] / "run_meta.json", run_meta)
    write_json(infer_context["artifact_dir"] / "source_train_run.json", source_payload)


def _coerce_binary_labels(y_train: Any) -> np.ndarray:
    y_series = pd.Series(y_train)
    if y_series.empty:
        raise ValueError("[model_logistic_regression] y_train is empty.")

    normalized_values: set[int] = set()
    for value in y_series.dropna().tolist():
        if isinstance(value, (bool, np.bool_)):
            normalized_values.add(int(bool(value)))
            continue
        if isinstance(value, (int, np.integer)):
            if int(value) in {0, 1}:
                normalized_values.add(int(value))
                continue
            raise ValueError("[model_logistic_regression] y_train must contain only binary labels.")
        if isinstance(value, (float, np.floating)):
            if float(value) in {0.0, 1.0}:
                normalized_values.add(int(value))
                continue
            raise ValueError("[model_logistic_regression] y_train must contain only binary labels.")
        raise ValueError("[model_logistic_regression] y_train must contain only binary labels.")

    if normalized_values - {0, 1}:
        raise ValueError("[model_logistic_regression] y_train must contain only binary labels.")

    return y_series.astype(int).to_numpy()


def _sample_submission_path(project_root: str | Path | None = None) -> Path:
    return get_project_paths(project_root)["data_dir"] / "sample_submission.csv"


def load_preprocessed_data(
    processed_root: str | Path | None = None,
    project_root: str | Path | None = None,
    save_outputs: bool = False,
) -> dict[str, Any]:
    """Load the saved preprocessing bundle or rebuild it in memory when needed."""
    resolved_processed_root = _resolve_processed_root(processed_root=processed_root, project_root=project_root)
    try:
        print(f"[{MODEL_NAME}] Loading preprocessed bundle from: {resolved_processed_root}")
        return load_saved_preprocessed_bundle(MODEL_NAME, processed_root=resolved_processed_root)
    except Exception as exc:
        print(
            f"[{MODEL_NAME}] Falling back to in-memory preprocessing because bundle loading failed: "
            f"{type(exc).__name__}: {exc}"
        )

    results = run_all_preprocessing(project_root=project_root, save_outputs=save_outputs)
    if MODEL_NAME not in results:
        raise KeyError(f"[{MODEL_NAME}] Preprocessing results did not include '{MODEL_NAME}'.")
    return results[MODEL_NAME]


def validate_lr_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate Logistic Regression bundle contents without changing preprocessing."""
    missing_keys = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing_keys:
        raise KeyError(f"[{MODEL_NAME}] Missing required bundle keys: {missing_keys}")

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = _coerce_binary_labels(bundle["y_train"])
    train_ids = bundle["train_ids"]
    test_ids = bundle["test_ids"]

    if not hasattr(X_train, "shape") or not hasattr(X_test, "shape"):
        raise TypeError(f"[{MODEL_NAME}] X_train and X_test must expose a shape.")

    if int(X_train.shape[0]) == 0 or int(X_test.shape[0]) == 0:
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test must be non-empty.")
    if int(X_train.shape[1]) == 0 or int(X_test.shape[1]) == 0:
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test must contain at least one feature.")
    if int(X_train.shape[0]) != int(len(y_train)):
        raise ValueError(f"[{MODEL_NAME}] X_train rows must match y_train length.")
    if int(X_train.shape[0]) != int(len(train_ids)):
        raise ValueError(f"[{MODEL_NAME}] X_train rows must match train_ids length.")
    if int(X_test.shape[0]) != int(len(test_ids)):
        raise ValueError(f"[{MODEL_NAME}] X_test rows must match test_ids length.")
    if int(X_train.shape[1]) != int(X_test.shape[1]):
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test must have the same feature count.")

    feature_names = bundle.get("feature_names")
    if feature_names is not None and int(len(feature_names)) != int(X_train.shape[1]):
        raise ValueError(f"[{MODEL_NAME}] feature_names length must match transformed feature count.")

    if sparse.issparse(X_train) != sparse.issparse(X_test):
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test must both be sparse or both be dense.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "feature_names": feature_names,
    }


def build_lr_model(config: dict[str, Any] | None = None) -> LogisticRegression:
    """Create a LogisticRegression instance for the preprocessed LR branch."""
    model_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model_config.update(config)
    if model_config.get("penalty") != "elasticnet":
        model_config.pop("l1_ratio", None)
    return LogisticRegression(**model_config)


def predict_proba(model: LogisticRegression, X: Any) -> np.ndarray:
    """Return positive-class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"[{MODEL_NAME}] The provided model does not support predict_proba.")
    probabilities = model.predict_proba(X)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"[{MODEL_NAME}] predict_proba must return a two-column probability matrix.")
    return probabilities[:, 1]


def predict(model: LogisticRegression, X: Any, threshold: float = 0.5) -> np.ndarray:
    """Return boolean predictions using a fixed threshold on positive-class probability."""
    return predict_proba(model, X) >= float(threshold)


def train_model(
    bundle: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    processed_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "final_train",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Train the Logistic Regression model from the existing LR bundle."""
    if mode == "strict_validation":
        raise RuntimeError(MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR)
    if mode != "final_train":
        raise ValueError(f"[{MODEL_NAME}] Unsupported mode: {mode}")

    if bundle is None:
        bundle = load_preprocessed_data(processed_root=processed_root, project_root=project_root, save_outputs=False)

    validated = validate_lr_bundle(bundle)
    model_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model_config.update(config)

    model = build_lr_model(model_config)
    print(f"[{MODEL_NAME}] Training LogisticRegression with config: {model_config}")
    model.fit(validated["X_train"], validated["y_train"])

    train_probabilities = predict_proba(model, validated["X_train"])
    train_predictions = train_probabilities >= float(threshold)
    y_train_bool = validated["y_train"].astype(bool)
    train_accuracy = float(np.mean(train_predictions == y_train_bool))

    train_summary = {
        "summary_type": "train_only",
        "mode": mode,
        "train_accuracy": train_accuracy,
        "train_positive_rate_observed": float(np.mean(y_train_bool)),
        "train_positive_rate_predicted": float(np.mean(train_predictions)),
        "n_train_samples": int(validated["X_train"].shape[0]),
        "n_test_samples": int(validated["X_test"].shape[0]),
    }
    metadata = {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": model_config.get("random_state", DEFAULT_CONFIG["random_state"]),
        "train_shape": [int(size) for size in validated["X_train"].shape],
        "test_shape": [int(size) for size in validated["X_test"].shape],
        "threshold": float(threshold),
        "bundle_path": bundle.get("save_path"),
        "bundle_source": bundle.get("save_path") or "in_memory_preprocessing",
        "feature_count": int(validated["X_train"].shape[1]),
        "config": model_config,
    }

    return {
        "model": model,
        "metadata": metadata,
        "train_summary": train_summary,
        "bundle": bundle,
        "config": model_config,
    }


def save_model_artifacts(
    training_result: dict[str, Any],
    project_root: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, str]:
    """Persist the trained Logistic Regression model and metadata outside processed/."""
    root = _resolve_project_root(project_root)
    output_dir = Path(artifact_dir) if artifact_dir is not None else root / "artifacts" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    config_path = output_dir / "config.json"
    metadata_path = output_dir / "metadata.json"
    coefficients_path = output_dir / "coefficients.csv"

    joblib.dump(training_result["model"], model_path)
    _write_json(config_path, training_result["config"])

    metadata_payload = dict(training_result["metadata"])
    metadata_payload["train_summary"] = training_result["train_summary"]
    _write_json(metadata_path, metadata_payload)

    bundle = training_result["bundle"]
    feature_names = bundle.get("feature_names")
    if feature_names is not None:
        coefficients = np.asarray(training_result["model"].coef_, dtype=float).reshape(-1)
        if int(len(feature_names)) != int(len(coefficients)):
            raise ValueError(f"[{MODEL_NAME}] feature_names length does not match coefficient count.")
        coefficients_df = pd.DataFrame(
            {
                "feature_name": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
                "odds_ratio": np.exp(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)
        coefficients_df.to_csv(coefficients_path, index=False)

    artifact_paths = {
        "artifact_dir": str(output_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
    }
    if coefficients_path.exists():
        artifact_paths["coefficients_path"] = str(coefficients_path)
    print(f"[{MODEL_NAME}] Saved artifacts to: {output_dir}")
    return artifact_paths


def generate_submission(
    model: LogisticRegression,
    bundle: dict[str, Any],
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
) -> Path:
    """Generate a Kaggle-style submission aligned to sample_submission.csv."""
    validated = validate_lr_bundle(bundle)
    sample_path = _sample_submission_path(project_root)
    sample_submission = pd.read_csv(sample_path)

    predictions = np.asarray(predict(model, validated["X_test"], threshold=threshold), dtype=bool)
    prediction_frame = pd.DataFrame(
        {
            "PassengerId": pd.Series(validated["test_ids"]).astype(str),
            "Transported": predictions,
        }
    )

    sample_ids = sample_submission["PassengerId"].astype(str)
    if sample_ids.tolist() == prediction_frame["PassengerId"].tolist():
        submission = sample_submission.copy()
        submission["Transported"] = predictions
    else:
        submission = sample_submission[["PassengerId"]].copy()
        submission["PassengerId"] = submission["PassengerId"].astype(str)
        submission = submission.merge(prediction_frame, on="PassengerId", how="left", validate="one_to_one")
        if submission["Transported"].isna().any():
            raise ValueError(f"[{MODEL_NAME}] Submission merge produced missing predictions.")
        submission["Transported"] = submission["Transported"].astype(bool)

    if submission.columns.tolist() != sample_submission.columns.tolist():
        raise ValueError(f"[{MODEL_NAME}] Submission columns do not match sample_submission.csv.")
    if int(len(submission)) != int(len(sample_submission)):
        raise ValueError(f"[{MODEL_NAME}] Submission row count does not match sample_submission.csv.")
    if submission["PassengerId"].astype(str).tolist() != sample_ids.tolist():
        raise ValueError(f"[{MODEL_NAME}] Submission PassengerId order does not match sample_submission.csv.")

    final_output_path = (
        Path(output_path)
        if output_path is not None
        else _resolve_project_root(project_root) / "submissions" / f"submission_{MODEL_NAME}.csv"
    )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(final_output_path, index=False)
    print(f"[{MODEL_NAME}] Saved submission to: {final_output_path}")
    return final_output_path


def load_model_artifact_from_path(model_path: str | Path) -> tuple[LogisticRegression, Path]:
    """Load a trained Logistic Regression artifact from an explicit path."""
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"[{MODEL_NAME}] Model artifact not found at '{resolved_model_path}'.")

    try:
        model = load_joblib_with_pandas_compat(resolved_model_path)
    except Exception as exc:
        raise RuntimeError(
            f"[{MODEL_NAME}] Failed to load model artifact from '{resolved_model_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(model, LogisticRegression):
        raise TypeError(f"[{MODEL_NAME}] Loaded artifact is not a LogisticRegression model.")

    # Older sklearn artifacts may omit this attribute; restoring the default avoids a false incompatibility.
    if not hasattr(model, "multi_class"):
        model.multi_class = "auto"

    return model, resolved_model_path


def load_model_artifact(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[LogisticRegression, Path]:
    """Load a trained Logistic Regression artifact without triggering retraining."""
    artifact_dir = _resolve_artifact_dir(artifacts_dir=artifacts_dir, project_root=project_root)
    model_path = artifact_dir / LEGACY_MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"[{MODEL_NAME}] Model artifact not found at '{model_path}'. "
            "Run training first or provide a valid --artifacts-dir."
        )

    return load_model_artifact_from_path(model_path)


def load_preprocessed_bundle(
    processed_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load the saved preprocessing bundle required for infer mode."""
    resolved_processed_root = _resolve_processed_root(processed_root=processed_dir, project_root=project_root)
    bundle_path = resolved_processed_root / MODEL_NAME / f"preprocessed_{MODEL_NAME}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"[{MODEL_NAME}] Preprocessed bundle not found at '{bundle_path}'. "
            "Infer mode requires an existing saved bundle."
        )

    try:
        bundle = load_joblib_with_pandas_compat(bundle_path)
    except Exception as exc:
        raise RuntimeError(
            f"[{MODEL_NAME}] Failed to load preprocessed bundle from '{bundle_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(bundle, dict):
        raise TypeError(f"[{MODEL_NAME}] Loaded bundle from '{bundle_path}' is not a dictionary.")

    return bundle, bundle_path


def validate_test_schema(raw_test_df: pd.DataFrame, strict_schema: bool = False) -> tuple[pd.Series, dict[str, Any]]:
    """Validate the raw infer CSV before any model artifact is used."""
    return validate_test_csv_schema(raw_test_df, model_name=MODEL_NAME, strict_schema=strict_schema)


def prepare_test_features(args: argparse.Namespace, source_context: dict[str, Any]) -> dict[str, Any]:
    """Read the raw test CSV, validate it, and align it with the saved bundle test split."""
    test_path = Path(args.test_path) if args.test_path is not None else Path("<missing-test-path>")
    try:
        raw_test_df, raw_test_path = read_test_dataframe(args.test_path, MODEL_NAME)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Failed to read infer test.csv: {type(exc).__name__}: {exc}",
            attempted_paths=[test_path],
            fix_hint="Provide a readable --test-path pointing to the target raw test.csv.",
        )

    try:
        passenger_ids, schema_report = validate_test_schema(raw_test_df, strict_schema=args.strict_schema)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Input test.csv schema validation failed: {type(exc).__name__}: {exc}",
            attempted_paths=[raw_test_path],
            fix_hint="Use the expected Spaceship Titanic test.csv schema and preserve PassengerId uniqueness/order.",
        )

    bundle_path = _resolve_bundle_path(processed_dir=args.processed_dir, project_root=args.project_root)
    try:
        bundle, bundle_path = load_preprocessed_bundle(processed_dir=args.processed_dir, project_root=args.project_root)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Failed to load infer bundle: {type(exc).__name__}: {exc}",
            attempted_paths=[bundle_path],
            fix_hint="Generate or restore the saved preprocessing bundle before running infer mode.",
        )

    try:
        x_test, bundle_test_ids = extract_bundle_test_features(bundle, MODEL_NAME)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Bundle is missing required infer content: {type(exc).__name__}: {exc}",
            attempted_paths=[bundle_path],
            fix_hint="Ensure the bundle contains X_test and test_ids, then retry infer mode.",
        )

    try:
        validate_bundle_test_ids(bundle_test_ids, passenger_ids, MODEL_NAME)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Bundle test_ids do not align with the provided test.csv: {type(exc).__name__}: {exc}",
            attempted_paths=[raw_test_path, bundle_path],
            fix_hint="Use the exact test.csv that matches the saved bundle, or rebuild the bundle for this test split.",
        )

    return {
        "raw_test_path": raw_test_path,
        "passenger_ids": passenger_ids,
        "schema_report": schema_report,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "X_test": x_test,
        "feature_preparation_mode": "bundle_test_features",
    }


def predict_test(
    model: LogisticRegression,
    prepared_test: dict[str, Any],
    source_context: dict[str, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict labels and positive-class probabilities for the saved X_test matrix."""
    try:
        probabilities = np.asarray(predict_proba(model, prepared_test["X_test"]), dtype=float)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Inference failed during predict_proba: {type(exc).__name__}: {exc}",
            attempted_paths=[source_context["model_path"], prepared_test["bundle_path"]],
            fix_hint="Verify the saved model artifact and bundle belong to the same feature space.",
        )

    predictions = np.asarray(probabilities >= float(threshold), dtype=bool)
    if int(len(predictions)) != int(len(prepared_test["passenger_ids"])):
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=(
                f"Prediction count ({int(len(predictions))}) does not match input row count "
                f"({int(len(prepared_test['passenger_ids']))})."
            ),
            attempted_paths=[source_context["model_path"], prepared_test["bundle_path"], prepared_test["raw_test_path"]],
            fix_hint="Ensure the model predicts one label per saved X_test row in the bundle.",
        )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }


def build_submission(passenger_ids: pd.Series, predictions: Any) -> pd.DataFrame:
    """Build the infer-mode submission with strict schema and order validation."""
    return build_submission_frame(passenger_ids, predictions, MODEL_NAME)


def save_inference_outputs(
    *,
    prepared_test: dict[str, Any],
    prediction_output: dict[str, Any],
    source_context: dict[str, Any],
    infer_context: dict[str, Any],
    project_root: str | Path | None,
    submissions_dir: str | Path | None,
    output_name: str | None,
    save_proba: bool,
    threshold: float,
) -> dict[str, str]:
    """Persist infer-mode outputs without affecting training artifacts or processed data."""
    try:
        submission = build_submission(prepared_test["passenger_ids"], prediction_output["predictions"])
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=infer_context["infer_run"],
            message=f"Submission construction failed: {type(exc).__name__}: {exc}",
            attempted_paths=[source_context["model_path"], prepared_test["raw_test_path"]],
            fix_hint="Ensure predictions are one-dimensional, non-empty, and safely convertible to Transported booleans.",
        )

    compatibility_submission_path = _resolve_compatibility_submission_path(
        project_root=project_root,
        submissions_dir=submissions_dir,
    )
    additional_submission_paths: list[Path] = []
    extra_output_path = _resolve_optional_output_path(
        project_root=project_root,
        submissions_dir=submissions_dir,
        output_name=output_name,
    )
    if extra_output_path is not None:
        additional_submission_paths.append(extra_output_path)

    try:
        create_infer_run_dirs(infer_context["artifact_dir"], infer_context["submission_dir"])
    except FileExistsError:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=infer_context["infer_run"],
            message="Infer output directories already exist.",
            attempted_paths=[infer_context["artifact_dir"], infer_context["submission_dir"]],
            fix_hint="Choose a different --infer-run or remove the conflicting directories first. Overwrite is not supported.",
        )

    return save_inference_artifacts(
        model_name=MODEL_NAME,
        submission=submission,
        submission_path=infer_context["submission_path"],
        compatibility_submission_path=compatibility_submission_path,
        additional_submission_paths=additional_submission_paths,
        artifact_dir=infer_context["artifact_dir"],
        raw_test_path=prepared_test["raw_test_path"],
        bundle_path=prepared_test["bundle_path"],
        model_path=source_context["model_path"],
        schema_report=prepared_test["schema_report"],
        threshold=threshold,
        probabilities=prediction_output["probabilities"],
        save_proba=save_proba,
        feature_preparation_mode=prepared_test["feature_preparation_mode"],
        extra_summary_fields={
            "infer_run": infer_context["infer_run"],
            "source_train_run": source_context["train_run"],
            "source_mode": source_context["source_mode"],
        },
    )


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Run strict bundle-based infer mode for the Logistic Regression branch."""
    reject_minimal_runtime_train_data_path(getattr(args, "train_data_path", None), model_name=MODEL_NAME)
    print(f"[{MODEL_NAME}] Running infer mode.")
    source_context = _resolve_train_source(args)
    explicit_infer_context = _check_explicit_infer_run(args)
    prepared_test = prepare_test_features(args, source_context)

    if args.dry_run:
        print(f"[{MODEL_NAME}] Dry run passed. Bundle test_ids match the input test.csv; no prediction files were written.")
        return {
            "mode": "infer",
            "dry_run": True,
            "model_path": str(source_context["model_path"]),
            "bundle_path": str(prepared_test["bundle_path"]),
            "input_test_path": str(prepared_test["raw_test_path"]),
            "n_test_samples": int(len(prepared_test["passenger_ids"])),
            "feature_preparation_mode": prepared_test["feature_preparation_mode"],
            "source_mode": source_context["source_mode"],
            "source_train_run": source_context["train_run"],
        }

    try:
        model, loaded_model_path = load_model_artifact_from_path(source_context["model_path"])
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Failed to load model artifact: {type(exc).__name__}: {exc}",
            attempted_paths=[source_context["model_path"]],
            fix_hint="Restore the saved model artifact or re-run training for this model.",
        )

    prediction_output = predict_test(model, prepared_test, source_context=source_context, threshold=args.threshold)
    infer_context = explicit_infer_context or _allocate_infer_context(args)
    saved_outputs = save_inference_outputs(
        prepared_test=prepared_test,
        prediction_output=prediction_output,
        source_context={**source_context, "model_path": loaded_model_path},
        infer_context=infer_context,
        project_root=args.project_root,
        submissions_dir=args.submissions_dir,
        output_name=args.output_name,
        save_proba=args.save_proba,
        threshold=args.threshold,
    )
    compatibility_submission_path = Path(saved_outputs["compatibility_submission_path"])
    _persist_infer_metadata(
        infer_context=infer_context,
        source_context={**source_context, "model_path": loaded_model_path},
        prepared_test=prepared_test,
        compatibility_submission_path=compatibility_submission_path,
        project_root=args.project_root,
        threshold=args.threshold,
        save_proba=args.save_proba,
    )
    update_run_registry(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        model_name=MODEL_NAME,
        stage="infer",
        run_id=infer_context["infer_run"],
    )
    update_latest_run(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        model_name=MODEL_NAME,
        stage="infer",
        run_id=infer_context["infer_run"],
    )
    print(f"[{MODEL_NAME}] Infer mode completed. Submission saved to: {saved_outputs['submission_path']}")
    return {
        "mode": "infer",
        "dry_run": False,
        "model_path": str(loaded_model_path),
        "bundle_path": str(prepared_test["bundle_path"]),
        "input_test_path": str(prepared_test["raw_test_path"]),
        "n_test_samples": int(len(prepared_test["passenger_ids"])),
        "feature_preparation_mode": prepared_test["feature_preparation_mode"],
        "source_mode": source_context["source_mode"],
        "source_train_run": source_context["train_run"],
        "infer_run": infer_context["infer_run"],
        **saved_outputs,
    }


def run_training_pipeline(
    project_root: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    submissions_dir: str | Path | None = None,
    processed_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "final_train",
    threshold: float = 0.5,
    train_data_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run final training, persist artifacts, and generate a submission file."""
    reject_minimal_runtime_train_data_path(train_data_path, model_name=MODEL_NAME)
    project_root_path = _resolve_project_root(project_root)
    training_bundle: dict[str, Any] | None = None
    explicit_train_context: dict[str, Any] | None = None
    if train_data_path is not None:
        training_bundle, explicit_train_context = build_explicit_training_bundle(
            model_name=MODEL_NAME,
            train_data_path=train_data_path,
            project_root=project_root_path,
        )
    training_result = train_model(
        bundle=training_bundle,
        project_root=project_root_path,
        processed_root=processed_root,
        config=config,
        mode=mode,
        threshold=threshold,
    )
    if explicit_train_context is None:
        training_result["metadata"]["infer_bundle_mode"] = DEFAULT_INFER_BUNDLE_MODE
        training_result["metadata"]["train_data_mode"] = "default_processed_bundle"
    else:
        training_result["metadata"]["infer_bundle_mode"] = EXPLICIT_TRAIN_DATA_BUNDLE_MODE
        training_result["metadata"]["train_data_mode"] = "explicit_train_data"
        training_result["metadata"]["train_data_path"] = str(explicit_train_context["train_data_path"])
        training_result["metadata"]["source_test_path"] = str(explicit_train_context["source_test_path"])
        training_result["metadata"]["bundle_source"] = "explicit_train_data_in_memory_bundle"
        dataset_context = explicit_train_context.get("dataset_context")
        if dataset_context is not None:
            training_result["metadata"]["source_dataset_run"] = dataset_context.get("dataset_run")
            training_result["metadata"]["source_selftrain_run"] = dataset_context.get("source_selftrain_run")
            training_result["metadata"]["source_infer_run"] = dataset_context.get("source_infer_run")
    train_root = get_run_root(
        project_root=project_root_path,
        artifacts_dir=artifacts_dir,
        stage="train",
        model_name=MODEL_NAME,
    )
    train_run = get_next_run_id(train_root, "train")
    managed_train_dir = train_root / train_run
    managed_train_dir.mkdir(parents=True, exist_ok=False)
    managed_artifact_paths = save_model_artifacts(
        training_result,
        project_root=project_root_path,
        artifact_dir=managed_train_dir,
    )
    _persist_train_run_metadata(
        training_result=training_result,
        train_run=train_run,
        train_dir=managed_train_dir,
        project_root=project_root_path,
    )
    legacy_artifact_paths = save_model_artifacts(
        training_result,
        project_root=project_root_path,
        artifact_dir=_resolve_artifact_dir(artifacts_dir=artifacts_dir, project_root=project_root_path),
    )
    update_run_registry(
        project_root=project_root_path,
        artifacts_dir=artifacts_dir,
        model_name=MODEL_NAME,
        stage="train",
        run_id=train_run,
    )
    update_latest_run(
        project_root=project_root_path,
        artifacts_dir=artifacts_dir,
        model_name=MODEL_NAME,
        stage="train",
        run_id=train_run,
    )
    result: dict[str, Any] = {
        **training_result,
        "artifact_paths": legacy_artifact_paths,
        "managed_artifact_paths": managed_artifact_paths,
        "train_run": train_run,
        "managed_train_dir": str(managed_train_dir),
    }
    submission_path = generate_submission(
        training_result["model"],
        training_result["bundle"],
        project_root=project_root_path,
        output_path=_resolve_compatibility_submission_path(
            project_root=project_root_path,
            submissions_dir=submissions_dir,
        ),
        threshold=threshold,
    )
    result["submission_path"] = str(submission_path)
    result["compatibility_submission_path"] = str(submission_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Logistic Regression model and generate a submission.")
    parser.add_argument("--project-root", default=None, help="Project root directory. Defaults to the current working directory.")
    parser.add_argument(
        "--processed-dir",
        "--processed-root",
        dest="processed_dir",
        default=None,
        help="Optional processed bundle directory override.",
    )
    parser.add_argument("--artifacts-dir", default=None, help="Optional artifacts root directory override.")
    parser.add_argument("--submissions-dir", default=None, help="Optional submissions directory override.")
    parser.add_argument("--train-run", default=None, help="Managed training run id to use in infer mode.")
    parser.add_argument("--infer-run", default=None, help="Optional managed infer run id to create in infer mode.")
    parser.add_argument("--test-path", default=None, help="Raw test CSV path used by infer mode.")
    parser.add_argument("--output-name", default=None, help="Optional infer submission file name (must end with .csv).")
    parser.add_argument("--save-proba", action="store_true", help="Save positive-class probabilities for infer mode.")
    parser.add_argument("--strict-schema", action="store_true", help="Require the raw test CSV columns to match exactly.")
    parser.add_argument("--dry-run", action="store_true", help="Validate infer inputs without running predict or writing files.")
    parser.add_argument("--train-data-path", default=None, help="Optional explicit train CSV path for training modes.")
    parser.add_argument("--mode", default="final_train", choices=["final_train", "strict_validation", "infer"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for positive predictions.")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    reject_minimal_runtime_train_data_path(args.train_data_path, model_name=MODEL_NAME)
    if args.mode == "infer":
        return run_inference(args)

    _validate_non_infer_args(args)
    result = run_training_pipeline(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        submissions_dir=args.submissions_dir,
        processed_root=args.processed_dir,
        mode=args.mode,
        threshold=args.threshold,
        train_data_path=args.train_data_path,
    )
    print(f"[{MODEL_NAME}] Train summary: {result['train_summary']}")
    return result


if __name__ == "__main__":
    main()
