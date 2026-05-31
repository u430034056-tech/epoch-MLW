"""Training and inference utilities for the CatBoost branch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
from preprocess import (
    build_fold_catboost_bundle,
    get_project_paths,
    load_preprocessed_bundle as load_saved_preprocessed_bundle,
    run_all_preprocessing,
)
from run_utils import (
    MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR,
    allocate_stage_run,
    allocate_infer_run,
    create_infer_run_dirs,
    current_timestamp,
    dedupe_preserve_order,
    ensure_infer_run_available,
    get_latest_train_run,
    get_next_run_id,
    get_run_root,
    list_existing_runs,
    raise_run_error,
    reject_minimal_runtime_train_data_path,
    relative_to_project,
    resolve_legacy_model_artifact,
    resolve_train_run_dir,
    resolve_model_artifact_from_train_run,
    update_stage_latest,
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


MODEL_NAME = "catboost"
LEGACY_MODEL_FILENAME = "model.cbm"
MISSING_CATEGORY_TOKEN = "__MISSING__"
DEFAULT_CONFIG = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": 100,
    "allow_writing_files": False,
    "thread_count": -1,
}
REQUIRED_BUNDLE_KEYS = ("X_train", "X_test", "y_train", "train_ids", "test_ids")
THRESHOLD_FALLBACK_METRIC = "accuracy"
THRESHOLD_RESULT_COLUMNS = ["threshold", "accuracy", "auc", "f1", "precision", "recall", "logloss"]
TUNING_RESULT_SORT_COLUMNS = ["mean_cv_accuracy", "mean_cv_auc"]


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
    return {
        "model_name": MODEL_NAME,
        "source_train_run": source_context["train_run"],
        "source_mode": source_context["source_mode"],
        "source_model_path": relative_to_project(source_context["model_path"], project_root),
        "source_bundle_path": relative_to_project(bundle_path, project_root),
    }


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
    write_json(
        infer_context["artifact_dir"] / "infer_config.json",
        {
            "mode": "infer",
            "threshold": float(threshold),
            "save_proba": bool(save_proba),
            "test_path": relative_to_project(prepared_test["raw_test_path"], project_root),
            "bundle_path": relative_to_project(prepared_test["bundle_path"], project_root),
            "source_train_run": source_context["train_run"],
            "source_mode": source_context["source_mode"],
            "model_path": relative_to_project(source_context["model_path"], project_root),
        },
    )
    write_json(
        infer_context["artifact_dir"] / "run_meta.json",
        {
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
        },
    )
    write_json(
        infer_context["artifact_dir"] / "source_train_run.json",
        _build_source_train_payload(
            source_context=source_context,
            bundle_path=prepared_test["bundle_path"],
            project_root=project_root,
        ),
    )


def _coerce_binary_labels(y_train: Any) -> np.ndarray:
    y_series = pd.Series(y_train)
    if y_series.empty:
        raise ValueError("[model_catboost] y_train is empty.")

    normalized_values: set[int] = set()
    for value in y_series.dropna().tolist():
        if isinstance(value, (bool, np.bool_)):
            normalized_values.add(int(bool(value)))
            continue
        if isinstance(value, (int, np.integer)):
            if int(value) in {0, 1}:
                normalized_values.add(int(value))
                continue
            raise ValueError("[model_catboost] y_train must contain only binary labels.")
        if isinstance(value, (float, np.floating)):
            if float(value) in {0.0, 1.0}:
                normalized_values.add(int(value))
                continue
            raise ValueError("[model_catboost] y_train must contain only binary labels.")
        raise ValueError("[model_catboost] y_train must contain only binary labels.")

    if normalized_values - {0, 1}:
        raise ValueError("[model_catboost] y_train must contain only binary labels.")

    return y_series.astype(int).to_numpy()


def _sample_submission_path(project_root: str | Path | None = None) -> Path:
    return get_project_paths(project_root)["data_dir"] / "sample_submission.csv"


def _get_catboost_components() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise ImportError(
            "[model_catboost] CatBoost is not available in this environment. "
            "Install a working catboost package before running CatBoost training."
        ) from exc
    return CatBoostClassifier, Pool


def _get_optuna_module() -> Any | None:
    try:
        import optuna
    except ImportError:
        return None
    return optuna


def _coerce_optional_int(value: Any, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None and coerced < minimum:
        return None
    return coerced


def _safe_auc_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return float("nan")


def _clip_probabilities(proba: Any) -> np.ndarray:
    return np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)


def _sanitize_catboost_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    model_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model_config.update(config)
    for key in [
        "early_stopping_rounds",
        "final_iteration_strategy",
        "fold_best_iterations",
        "final_iterations",
        "tuned_iterations_upper_bound",
        "final_train_used_early_stopping",
        "final_iteration_source",
        "valid_fold_best_iteration_count",
        "missing_fold_best_iteration_count",
    ]:
        model_config.pop(key, None)
    bootstrap_type = model_config.get("bootstrap_type")
    if bootstrap_type != "Bernoulli":
        model_config.pop("subsample", None)
    if bootstrap_type != "Bayesian":
        model_config.pop("bagging_temperature", None)
    return model_config


def _build_cv_model_config(params: dict[str, Any], random_state: int) -> dict[str, Any]:
    model_config = _sanitize_catboost_config(
        {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": int(random_state),
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            **params,
        }
    )
    return model_config


def compute_binary_classification_metrics(y_true: Any, proba: Any, threshold: float) -> dict[str, float]:
    """Compute standard binary classification metrics from probabilities at a fixed threshold."""
    y_true_array = np.asarray(y_true, dtype=int)
    proba_array = _clip_probabilities(proba)
    predictions = (proba_array >= float(threshold)).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "auc": _safe_auc_score(y_true_array, proba_array),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "logloss": float(log_loss(y_true_array, proba_array, labels=[0, 1])),
    }


def build_threshold_grid(start: float = 0.30, end: float = 0.70, step: float = 0.01) -> np.ndarray:
    """Build a stable inclusive threshold grid."""
    start_value = float(start)
    end_value = float(end)
    step_value = float(step)
    if step_value <= 0:
        raise ValueError(f"[{MODEL_NAME}] threshold step must be positive.")
    if end_value < start_value:
        raise ValueError(f"[{MODEL_NAME}] threshold end must be greater than or equal to threshold start.")
    thresholds = np.arange(start_value, end_value + (step_value / 2.0), step_value, dtype=float)
    thresholds = np.round(thresholds, 10)
    if thresholds.size == 0:
        raise ValueError(f"[{MODEL_NAME}] threshold grid is empty.")
    return thresholds


def search_best_threshold(
    y_true: Any,
    proba: Any,
    thresholds: Any | None = None,
    metric: str = THRESHOLD_FALLBACK_METRIC,
) -> dict[str, Any]:
    """Search a threshold grid on OOF probabilities and return the best threshold plus all scores."""
    metric_name = str(metric).lower()
    if metric_name not in {"accuracy", "f1", "precision", "recall"}:
        raise ValueError(f"[{MODEL_NAME}] Unsupported threshold metric '{metric_name}'.")

    threshold_values = np.asarray(thresholds if thresholds is not None else build_threshold_grid(), dtype=float)
    y_true_array = np.asarray(y_true, dtype=int)
    proba_array = _clip_probabilities(proba)
    threshold_results: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for threshold in threshold_values:
        metrics = compute_binary_classification_metrics(y_true_array, proba_array, float(threshold))
        row = {"threshold": float(threshold), **metrics}
        threshold_results.append(row)
        if best_row is None:
            best_row = row
            continue
        current_value = float(row[metric_name])
        best_value = float(best_row[metric_name])
        current_distance = abs(float(row["threshold"]) - 0.5)
        best_distance = abs(float(best_row["threshold"]) - 0.5)
        if (
            current_value > best_value
            or (
                current_value == best_value
                and (
                    current_distance < best_distance
                    or (current_distance == best_distance and float(row["threshold"]) < float(best_row["threshold"]))
                )
            )
        ):
            best_row = row

    if best_row is None:
        raise ValueError(f"[{MODEL_NAME}] Threshold search produced no results.")

    return {
        "metric": metric_name,
        "best_threshold": float(best_row["threshold"]),
        "best_metric_value": float(best_row[metric_name]),
        "best_metrics": {key: float(best_row[key]) for key in THRESHOLD_RESULT_COLUMNS if key != "threshold"},
        "threshold_results": threshold_results,
    }


def _normalize_fold_best_iterations(model: Any) -> int | None:
    """Normalize CatBoost early-stopping output to effective iteration-count semantics."""
    tree_count = _coerce_optional_int(getattr(model, "tree_count_", None), minimum=1)
    if tree_count is not None:
        return tree_count
    raw_best_iteration: int | None = None
    try:
        raw_best_iteration = _coerce_optional_int(model.get_best_iteration(), minimum=0)
    except Exception:
        raw_best_iteration = None
    if raw_best_iteration is not None:
        return raw_best_iteration + 1
    return None


def _mean_without_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _std_without_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def _float_or_none(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric_value):
        return None
    return numeric_value


def _derive_final_iteration_payload(
    fold_best_iterations: list[Any],
    tuned_params: dict[str, Any],
) -> dict[str, Any]:
    normalized_fold_best_iterations = [_coerce_optional_int(value, minimum=1) for value in fold_best_iterations]
    valid_fold_best_iterations = [value for value in normalized_fold_best_iterations if value is not None]
    tuned_iterations_upper_bound = _coerce_optional_int(tuned_params.get("iterations"), minimum=1)

    if valid_fold_best_iterations:
        derived_or_fallback_iterations = int(round(float(np.mean(valid_fold_best_iterations))))
        final_iteration_source = "valid_fold_mean"
    elif tuned_iterations_upper_bound is not None:
        derived_or_fallback_iterations = int(tuned_iterations_upper_bound)
        final_iteration_source = "tuned_iterations_fallback"
    else:
        derived_or_fallback_iterations = int(DEFAULT_CONFIG["iterations"])
        final_iteration_source = "default_config_fallback"

    candidate_iterations = int(derived_or_fallback_iterations)
    if tuned_iterations_upper_bound is not None:
        candidate_iterations = min(int(candidate_iterations), int(tuned_iterations_upper_bound))
    final_iterations = max(50, int(candidate_iterations))

    return {
        "final_iteration_strategy": "mean_best_iteration",
        "fold_best_iterations": normalized_fold_best_iterations,
        "valid_fold_best_iteration_count": int(len(valid_fold_best_iterations)),
        "missing_fold_best_iteration_count": int(len(normalized_fold_best_iterations) - len(valid_fold_best_iterations)),
        "tuned_iterations_upper_bound": tuned_iterations_upper_bound,
        "derived_or_fallback_iterations": int(derived_or_fallback_iterations),
        "final_iterations": int(final_iterations),
        "final_iteration_source": final_iteration_source,
        "final_train_used_early_stopping": False,
    }


def get_catboost_search_space() -> dict[str, Any]:
    """Return the CatBoost tuning search space used by random and optional Bayesian search."""
    return {
        "depth": [4, 5, 6, 7, 8, 9, 10],
        "learning_rate": {"low": 0.01, "high": 0.15, "scale": "log"},
        "iterations": {"low": 300, "high": 3000},
        "l2_leaf_reg": {"low": 1.0, "high": 15.0},
        "min_data_in_leaf": {"low": 1, "high": 64},
        "random_strength": {"low": 0.0, "high": 2.0},
        "bagging_temperature": {"low": 0.0, "high": 5.0},
        "rsm": {"low": 0.6, "high": 1.0},
        "border_count": [64, 128, 254],
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"],
        "subsample": [0.66, 0.8, 0.9, 1.0],
    }


def sample_catboost_random_config(rng: np.random.Generator) -> dict[str, Any]:
    """Sample a reproducible random CatBoost configuration."""
    search_space = get_catboost_search_space()
    bootstrap_type = str(rng.choice(search_space["bootstrap_type"]))
    sampled = {
        "depth": int(rng.choice(search_space["depth"])),
        "learning_rate": float(
            np.exp(
                rng.uniform(
                    np.log(search_space["learning_rate"]["low"]),
                    np.log(search_space["learning_rate"]["high"]),
                )
            )
        ),
        "iterations": int(rng.integers(search_space["iterations"]["low"], search_space["iterations"]["high"] + 1)),
        "l2_leaf_reg": float(rng.uniform(search_space["l2_leaf_reg"]["low"], search_space["l2_leaf_reg"]["high"])),
        "min_data_in_leaf": int(
            rng.integers(search_space["min_data_in_leaf"]["low"], search_space["min_data_in_leaf"]["high"] + 1)
        ),
        "random_strength": float(
            rng.uniform(search_space["random_strength"]["low"], search_space["random_strength"]["high"])
        ),
        "rsm": float(rng.uniform(search_space["rsm"]["low"], search_space["rsm"]["high"])),
        "border_count": int(rng.choice(search_space["border_count"])),
        "bootstrap_type": bootstrap_type,
    }
    if bootstrap_type == "Bayesian":
        sampled["bagging_temperature"] = float(
            rng.uniform(search_space["bagging_temperature"]["low"], search_space["bagging_temperature"]["high"])
        )
    if bootstrap_type == "Bernoulli":
        sampled["subsample"] = float(rng.choice(search_space["subsample"]))
    return sampled


def _sample_catboost_optuna_config(trial: Any) -> dict[str, Any]:
    """Sample a CatBoost configuration from an Optuna trial."""
    search_space = get_catboost_search_space()
    bootstrap_type = trial.suggest_categorical("bootstrap_type", search_space["bootstrap_type"])
    sampled = {
        "depth": int(trial.suggest_categorical("depth", search_space["depth"])),
        "learning_rate": float(
            trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["low"],
                search_space["learning_rate"]["high"],
                log=True,
            )
        ),
        "iterations": int(
            trial.suggest_int("iterations", search_space["iterations"]["low"], search_space["iterations"]["high"])
        ),
        "l2_leaf_reg": float(
            trial.suggest_float("l2_leaf_reg", search_space["l2_leaf_reg"]["low"], search_space["l2_leaf_reg"]["high"])
        ),
        "min_data_in_leaf": int(
            trial.suggest_int(
                "min_data_in_leaf",
                search_space["min_data_in_leaf"]["low"],
                search_space["min_data_in_leaf"]["high"],
            )
        ),
        "random_strength": float(
            trial.suggest_float(
                "random_strength",
                search_space["random_strength"]["low"],
                search_space["random_strength"]["high"],
            )
        ),
        "rsm": float(trial.suggest_float("rsm", search_space["rsm"]["low"], search_space["rsm"]["high"])),
        "border_count": int(trial.suggest_categorical("border_count", search_space["border_count"])),
        "bootstrap_type": str(bootstrap_type),
    }
    if bootstrap_type == "Bayesian":
        sampled["bagging_temperature"] = float(
            trial.suggest_float(
                "bagging_temperature",
                search_space["bagging_temperature"]["low"],
                search_space["bagging_temperature"]["high"],
            )
        )
    if bootstrap_type == "Bernoulli":
        sampled["subsample"] = float(trial.suggest_categorical("subsample", search_space["subsample"]))
    return sampled


def load_preprocessed_data(
    processed_root: str | Path | None = None,
    project_root: str | Path | None = None,
    save_outputs: bool = False,
    refresh_saved_bundle: bool = False,
) -> dict[str, Any]:
    """Load the saved preprocessing bundle or rebuild it in memory when needed."""
    if refresh_saved_bundle:
        print(f"[{MODEL_NAME}] Refreshing preprocessing bundles from source code before training.")
        results = run_all_preprocessing(project_root=project_root, save_outputs=True)
        if MODEL_NAME not in results:
            raise KeyError(f"[{MODEL_NAME}] Preprocessing results did not include '{MODEL_NAME}'.")
        return results[MODEL_NAME]

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


def validate_cat_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate CatBoost bundle contents and derive categorical feature metadata."""
    missing_keys = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if missing_keys:
        raise KeyError(f"[{MODEL_NAME}] Missing required bundle keys: {missing_keys}")

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"[{MODEL_NAME}] CatBoost expects pandas DataFrame inputs from preprocessing.")
    if X_train.empty or X_test.empty:
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test must be non-empty.")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError(f"[{MODEL_NAME}] X_train and X_test columns must match exactly.")

    y_train = _coerce_binary_labels(bundle["y_train"])
    train_ids = bundle["train_ids"]
    test_ids = bundle["test_ids"]
    if int(len(y_train)) != int(len(train_ids)) or int(len(y_train)) != int(X_train.shape[0]):
        raise ValueError(f"[{MODEL_NAME}] Training rows, train_ids, and y_train must align.")
    if int(len(test_ids)) != int(X_test.shape[0]):
        raise ValueError(f"[{MODEL_NAME}] Test rows and test_ids must align.")

    categorical_feature_names = bundle.get("categorical_feature_names")
    categorical_feature_indices = bundle.get("categorical_feature_indices")

    if categorical_feature_names is not None:
        categorical_feature_names = [str(name) for name in categorical_feature_names]
        missing_columns = [name for name in categorical_feature_names if name not in X_train.columns]
        if missing_columns:
            raise ValueError(f"[{MODEL_NAME}] Missing categorical columns in X_train: {missing_columns}")
        categorical_feature_indices = [int(X_train.columns.get_loc(name)) for name in categorical_feature_names]
        cat_feature_spec = categorical_feature_names
    elif categorical_feature_indices is not None:
        categorical_feature_indices = [int(index) for index in categorical_feature_indices]
        invalid_indices = [index for index in categorical_feature_indices if index < 0 or index >= X_train.shape[1]]
        if invalid_indices:
            raise ValueError(f"[{MODEL_NAME}] Invalid categorical feature indices: {invalid_indices}")
        categorical_feature_names = [str(X_train.columns[index]) for index in categorical_feature_indices]
        cat_feature_spec = categorical_feature_indices
    else:
        categorical_feature_names = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        categorical_feature_indices = [int(X_train.columns.get_loc(name)) for name in categorical_feature_names]
        cat_feature_spec = categorical_feature_names

    if "Surname" in X_train.columns and "Surname" not in categorical_feature_names:
        raise ValueError("[model_catboost] Surname must remain a categorical feature for CatBoost.")

    numeric_feature_names = [column for column in X_train.columns if column not in categorical_feature_names]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "feature_names": [str(column) for column in X_train.columns.tolist()],
        "categorical_feature_names": categorical_feature_names,
        "categorical_feature_indices": categorical_feature_indices,
        "cat_feature_spec": cat_feature_spec,
        "numeric_feature_names": numeric_feature_names,
    }


def prepare_catboost_inputs(bundle: dict[str, Any]) -> dict[str, Any]:
    """Prepare the CatBoost DataFrames without changing the preprocessing feature set."""
    validated = validate_cat_bundle(bundle)
    X_train = validated["X_train"].copy()
    X_test = validated["X_test"].copy()

    for column in validated["categorical_feature_names"]:
        X_train[column] = X_train[column].where(~X_train[column].isna(), MISSING_CATEGORY_TOKEN).astype(str)
        X_test[column] = X_test[column].where(~X_test[column].isna(), MISSING_CATEGORY_TOKEN).astype(str)

    for column in validated["numeric_feature_names"]:
        X_train[column] = pd.to_numeric(X_train[column], errors="raise")
        X_test[column] = pd.to_numeric(X_test[column], errors="raise")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": validated["y_train"],
        "train_ids": validated["train_ids"],
        "test_ids": validated["test_ids"],
        "feature_names": validated["feature_names"],
        "categorical_feature_names": validated["categorical_feature_names"],
        "categorical_feature_indices": validated["categorical_feature_indices"],
        "cat_feature_spec": validated["cat_feature_spec"],
    }


def build_catboost_model(config: dict[str, Any] | None = None) -> Any:
    """Create a CatBoostClassifier instance lazily."""
    CatBoostClassifier, _ = _get_catboost_components()
    model_config = DEFAULT_CONFIG.copy()
    if config is not None:
        model_config.update(config)
    return CatBoostClassifier(**model_config)


def predict_proba(model: Any, X: Any) -> np.ndarray:
    """Return positive-class probabilities."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"[{MODEL_NAME}] The provided model does not support predict_proba.")
    probabilities = model.predict_proba(X)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(f"[{MODEL_NAME}] predict_proba must return a two-column probability matrix.")
    return probabilities[:, 1]


def predict(model: Any, X: Any, threshold: float = 0.5) -> np.ndarray:
    """Return boolean predictions using a fixed threshold on positive-class probability."""
    return predict_proba(model, X) >= float(threshold)


def train_model(
    bundle: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    processed_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "final_train",
    threshold: float = 0.5,
    refresh_saved_bundle: bool = False,
) -> dict[str, Any]:
    """Train the CatBoost model from the existing CatBoost bundle."""
    if mode == "strict_validation":
        raise RuntimeError(MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR)
    if mode != "final_train":
        raise ValueError(f"[{MODEL_NAME}] Unsupported mode: {mode}")

    if bundle is None:
        bundle = load_preprocessed_data(
            processed_root=processed_root,
            project_root=project_root,
            save_outputs=False,
            refresh_saved_bundle=refresh_saved_bundle,
        )

    prepared_inputs = prepare_catboost_inputs(bundle)
    _, Pool = _get_catboost_components()

    model_config = _sanitize_catboost_config(config)

    model = build_catboost_model(model_config)
    print(f"[{MODEL_NAME}] Training CatBoostClassifier with config: {model_config}")

    train_pool = Pool(
        prepared_inputs["X_train"],
        prepared_inputs["y_train"],
        cat_features=prepared_inputs["cat_feature_spec"],
    )
    test_pool = Pool(
        prepared_inputs["X_test"],
        cat_features=prepared_inputs["cat_feature_spec"],
    )
    model.fit(train_pool)

    train_probabilities = predict_proba(model, train_pool)
    _ = predict_proba(model, test_pool)
    train_metrics = compute_binary_classification_metrics(prepared_inputs["y_train"], train_probabilities, threshold)
    train_predictions = train_probabilities >= float(threshold)
    y_train_bool = prepared_inputs["y_train"].astype(bool)
    feature_importance = np.asarray(model.get_feature_importance(train_pool), dtype=float)

    train_summary = {
        "summary_type": "train_only",
        "mode": mode,
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_auc": float(train_metrics["auc"]),
        "train_f1": float(train_metrics["f1"]),
        "train_precision": float(train_metrics["precision"]),
        "train_recall": float(train_metrics["recall"]),
        "train_logloss": float(train_metrics["logloss"]),
        "train_positive_rate_observed": float(np.mean(y_train_bool)),
        "train_positive_rate_predicted": float(np.mean(train_predictions)),
        "n_train_samples": int(prepared_inputs["X_train"].shape[0]),
        "n_test_samples": int(prepared_inputs["X_test"].shape[0]),
        "categorical_feature_count": int(len(prepared_inputs["categorical_feature_names"])),
    }
    metadata = {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": model_config.get("random_seed", DEFAULT_CONFIG["random_seed"]),
        "train_shape": [int(size) for size in prepared_inputs["X_train"].shape],
        "test_shape": [int(size) for size in prepared_inputs["X_test"].shape],
        "threshold": float(threshold),
        "bundle_path": bundle.get("save_path"),
        "bundle_source": bundle.get("save_path") or "in_memory_preprocessing",
        "feature_count": int(prepared_inputs["X_train"].shape[1]),
        "categorical_feature_names": prepared_inputs["categorical_feature_names"],
        "categorical_feature_count": int(len(prepared_inputs["categorical_feature_names"])),
        "config": model_config,
    }

    return {
        "model": model,
        "metadata": metadata,
        "train_summary": train_summary,
        "bundle": bundle,
        "config": model_config,
        "prepared_inputs": prepared_inputs,
        "feature_importance": feature_importance,
    }


def save_model_artifacts(
    training_result: dict[str, Any],
    project_root: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, str]:
    """Persist the trained CatBoost model and metadata outside processed/."""
    root = _resolve_project_root(project_root)
    output_dir = Path(artifact_dir) if artifact_dir is not None else root / "artifacts" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.cbm"
    config_path = output_dir / "config.json"
    metadata_path = output_dir / "metadata.json"
    importance_path = output_dir / "feature_importance.csv"

    training_result["model"].save_model(str(model_path))
    _write_json(config_path, training_result["config"])

    metadata_payload = dict(training_result["metadata"])
    metadata_payload["train_summary"] = training_result["train_summary"]
    _write_json(metadata_path, metadata_payload)

    prepared_inputs = training_result["prepared_inputs"]
    feature_importance = np.asarray(training_result["feature_importance"], dtype=float)
    if int(len(prepared_inputs["feature_names"])) != int(len(feature_importance)):
        raise ValueError(f"[{MODEL_NAME}] feature importance length does not match feature count.")
    importance_df = pd.DataFrame(
        {
            "feature_name": prepared_inputs["feature_names"],
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    artifact_paths = {
        "artifact_dir": str(output_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
        "feature_importance_path": str(importance_path),
    }
    print(f"[{MODEL_NAME}] Saved artifacts to: {output_dir}")
    return artifact_paths


def generate_submission(
    model: Any,
    bundle: dict[str, Any],
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
) -> Path:
    """Generate a Kaggle-style submission aligned to sample_submission.csv."""
    prepared_inputs = prepare_catboost_inputs(bundle)
    _, Pool = _get_catboost_components()

    test_pool = Pool(
        prepared_inputs["X_test"],
        cat_features=prepared_inputs["cat_feature_spec"],
    )
    predictions = np.asarray(predict(model, test_pool, threshold=threshold), dtype=bool)

    sample_path = _sample_submission_path(project_root)
    sample_submission = pd.read_csv(sample_path)
    prediction_frame = pd.DataFrame(
        {
            "PassengerId": pd.Series(prepared_inputs["test_ids"]).astype(str),
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


def load_model_artifact_from_path(model_path: str | Path) -> tuple[Any, Path]:
    """Load a trained CatBoost artifact from an explicit path."""
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"[{MODEL_NAME}] Model artifact not found at '{resolved_model_path}'.")

    try:
        CatBoostClassifier, _ = _get_catboost_components()
        model = CatBoostClassifier()
        model.load_model(str(resolved_model_path))
    except Exception as exc:
        raise RuntimeError(
            f"[{MODEL_NAME}] Failed to load model artifact from '{resolved_model_path}': {type(exc).__name__}: {exc}"
        ) from exc

    return model, resolved_model_path


def load_model_artifact(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[Any, Path]:
    """Load a trained CatBoost artifact without triggering retraining."""
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
    """Validate the raw infer CSV before any CatBoost prediction is attempted."""
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
        _, bundle_test_ids = extract_bundle_test_features(bundle, MODEL_NAME)
        prepared_inputs = prepare_catboost_inputs(bundle)
    except Exception as exc:
        raise_run_error(
            model_name=MODEL_NAME,
            stage="infer",
            run_id=source_context["train_run"],
            message=f"Bundle is missing required infer content: {type(exc).__name__}: {exc}",
            attempted_paths=[bundle_path],
            fix_hint="Ensure the bundle contains X_test, test_ids, and valid CatBoost-prepared features.",
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
        "bundle_path": bundle_path,
        "X_test": prepared_inputs["X_test"],
        "cat_feature_spec": prepared_inputs["cat_feature_spec"],
        "feature_preparation_mode": "bundle_test_features",
    }


def predict_test(
    model: Any,
    prepared_test: dict[str, Any],
    source_context: dict[str, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict labels and positive-class probabilities for the saved CatBoost test frame."""
    try:
        _, Pool = _get_catboost_components()
        test_pool = Pool(prepared_test["X_test"], cat_features=prepared_test["cat_feature_spec"])
        probabilities = np.asarray(predict_proba(model, test_pool), dtype=float)
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
    """Run strict bundle-based infer mode for the CatBoost branch."""
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


def _evaluate_cv_trial(
    raw_train_df: pd.DataFrame,
    *,
    trial_index: int,
    params: dict[str, Any],
    cv_folds: int,
    random_state: int,
    early_stopping_rounds: int,
    threshold_grid: np.ndarray,
    threshold_metric: str,
) -> dict[str, Any]:
    """Run one fold-local CatBoost CV trial and return all ranking artifacts."""
    _, Pool = _get_catboost_components()
    y_all = raw_train_df["Transported"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))
    oof_probabilities = np.zeros(len(raw_train_df), dtype=float)
    oof_folds = np.full(len(raw_train_df), -1, dtype=int)
    fold_records: list[dict[str, Any]] = []
    model_config = _build_cv_model_config(params, random_state=random_state)

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(raw_train_df, y_all), start=1):
        train_fold_df = raw_train_df.iloc[train_idx].reset_index(drop=True)
        valid_fold_df = raw_train_df.iloc[valid_idx].reset_index(drop=True)
        fold_bundle = build_fold_catboost_bundle(train_fold_df, valid_fold_df)
        if int(fold_bundle["X_valid"].shape[0]) != int(len(fold_bundle["y_valid"])):
            raise ValueError(f"[{MODEL_NAME}] Fold {fold_number} valid rows and labels are misaligned.")
        expected_valid_ids = raw_train_df.iloc[valid_idx]["PassengerId"].astype(str).tolist()
        actual_valid_ids = pd.Series(fold_bundle["valid_ids"]).astype(str).tolist()
        if expected_valid_ids != actual_valid_ids:
            raise ValueError(f"[{MODEL_NAME}] Fold {fold_number} valid PassengerId order is not preserved.")

        train_pool = Pool(
            fold_bundle["X_train"],
            fold_bundle["y_train"],
            cat_features=fold_bundle["cat_feature_spec"],
        )
        valid_pool = Pool(
            fold_bundle["X_valid"],
            fold_bundle["y_valid"],
            cat_features=fold_bundle["cat_feature_spec"],
        )

        model = build_catboost_model(model_config)
        fit_kwargs = {
            "eval_set": valid_pool,
            "use_best_model": True,
            "verbose": False,
        }
        if int(early_stopping_rounds) > 0:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        model.fit(train_pool, **fit_kwargs)

        valid_proba = predict_proba(model, valid_pool)
        normalized_fold_best_iterations = _normalize_fold_best_iterations(model)
        oof_probabilities[valid_idx] = valid_proba
        oof_folds[valid_idx] = fold_number
        fold_records.append(
            {
                "fold": int(fold_number),
                "valid_idx": valid_idx.copy(),
                "valid_ids": pd.Series(fold_bundle["valid_ids"]).astype(str).tolist(),
                "y_valid": np.asarray(fold_bundle["y_valid"], dtype=int),
                "valid_proba": valid_proba,
                "fold_best_iterations": normalized_fold_best_iterations,
            }
        )

    if (oof_folds < 0).any():
        raise ValueError(f"[{MODEL_NAME}] OOF predictions did not cover every training row.")

    threshold_search = search_best_threshold(
        y_true=y_all,
        proba=oof_probabilities,
        thresholds=threshold_grid,
        metric=threshold_metric,
    )
    best_threshold = float(threshold_search["best_threshold"])
    fold_metrics: list[dict[str, Any]] = []
    for fold_record in fold_records:
        metrics = compute_binary_classification_metrics(fold_record["y_valid"], fold_record["valid_proba"], best_threshold)
        fold_metrics.append(
            {
                "fold": int(fold_record["fold"]),
                "fold_best_iterations": fold_record["fold_best_iterations"],
                **metrics,
            }
        )

    aggregated_row: dict[str, Any] = {
        "trial": int(trial_index),
        "best_threshold": best_threshold,
        "threshold_metric": str(threshold_metric).lower(),
        "mean_fold_best_iterations": _mean_without_nan(
            [float(value) for value in [record["fold_best_iterations"] for record in fold_metrics] if value is not None]
        ),
        "std_fold_best_iterations": _std_without_nan(
            [float(value) for value in [record["fold_best_iterations"] for record in fold_metrics] if value is not None]
        ),
        "fold_best_iterations_json": json.dumps(
            [record["fold_best_iterations"] for record in fold_metrics], ensure_ascii=False
        ),
        "params_json": json.dumps(params, sort_keys=True),
    }
    for metric_name in ["accuracy", "auc", "f1", "precision", "recall", "logloss"]:
        metric_values = [_float_or_none(record[metric_name]) for record in fold_metrics]
        valid_metric_values = [value for value in metric_values if value is not None]
        aggregated_row[f"mean_cv_{metric_name}"] = _mean_without_nan(valid_metric_values)
        aggregated_row[f"std_cv_{metric_name}"] = _std_without_nan(valid_metric_values)

    oof_metrics = compute_binary_classification_metrics(y_all, oof_probabilities, best_threshold)
    for metric_name, metric_value in oof_metrics.items():
        aggregated_row[f"oof_{metric_name}"] = float(metric_value)

    return {
        "trial": int(trial_index),
        "params": dict(params),
        "model_config": dict(model_config),
        "trial_row": aggregated_row,
        "oof_probabilities": oof_probabilities,
        "oof_folds": oof_folds,
        "fold_metrics": fold_metrics,
        "fold_records": fold_records,
        "fold_best_iterations": [record["fold_best_iterations"] for record in fold_metrics],
        "threshold_search": threshold_search,
        "recommended_threshold": best_threshold,
        "oof_metrics": oof_metrics,
    }


def _cv_trial_sort_key(trial_result: dict[str, Any]) -> tuple[float, float, float]:
    row = trial_result["trial_row"]
    mean_cv_accuracy = _float_or_none(row.get("mean_cv_accuracy"))
    mean_cv_auc = _float_or_none(row.get("mean_cv_auc"))
    return (
        mean_cv_accuracy if mean_cv_accuracy is not None else float("-inf"),
        mean_cv_auc if mean_cv_auc is not None else float("-inf"),
        -float(trial_result["trial"]),
    )


def run_cv_tuning(args: argparse.Namespace) -> dict[str, Any]:
    """Run fold-local CatBoost CV tuning with OOF threshold search."""
    if not args.train_csv_path:
        raise ValueError(f"[{MODEL_NAME}] --train-csv-path is required for --mode cv_tune.")

    train_csv_path = Path(args.train_csv_path)
    raw_train_df = pd.read_csv(train_csv_path)
    if "Transported" not in raw_train_df.columns:
        raise ValueError(f"[{MODEL_NAME}] train CSV must contain Transported.")
    raw_train_df = raw_train_df.reset_index(drop=True)

    cv_folds = int(args.cv_folds)
    n_trials = int(args.n_trials)
    random_state = int(args.random_state)
    early_stopping_rounds = int(args.early_stopping_rounds)
    if cv_folds < 2:
        raise ValueError(f"[{MODEL_NAME}] --cv-folds must be at least 2.")
    if n_trials < 1:
        raise ValueError(f"[{MODEL_NAME}] --n-trials must be at least 1.")

    run_id, run_dir = allocate_stage_run(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        stage="tuning",
        prefix="tuning",
        model_name=MODEL_NAME,
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    requested_search_method = str(args.search_method).lower()
    effective_search_method = requested_search_method
    search_fallback_reason: str | None = None
    threshold_grid = build_threshold_grid(args.threshold_start, args.threshold_end, args.threshold_step)
    trial_results: list[dict[str, Any]] = []

    if requested_search_method == "bayes":
        optuna = _get_optuna_module()
        if optuna is None:
            effective_search_method = "random"
            search_fallback_reason = "optuna_not_installed"
        else:
            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            def objective(trial: Any) -> float:
                params = _sample_catboost_optuna_config(trial)
                result = _evaluate_cv_trial(
                    raw_train_df,
                    trial_index=int(trial.number) + 1,
                    params=params,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    early_stopping_rounds=early_stopping_rounds,
                    threshold_grid=threshold_grid,
                    threshold_metric=args.threshold_metric,
                )
                trial_results.append(result)
                metric_value = _float_or_none(result["trial_row"]["mean_cv_accuracy"])
                return metric_value if metric_value is not None else float("-inf")

            study.optimize(objective, n_trials=n_trials)

    if effective_search_method == "random":
        rng = np.random.default_rng(random_state)
        for trial_index in range(1, n_trials + 1):
            params = sample_catboost_random_config(rng)
            trial_results.append(
                _evaluate_cv_trial(
                    raw_train_df,
                    trial_index=trial_index,
                    params=params,
                    cv_folds=cv_folds,
                    random_state=random_state,
                    early_stopping_rounds=early_stopping_rounds,
                    threshold_grid=threshold_grid,
                    threshold_metric=args.threshold_metric,
                )
            )

    if not trial_results:
        raise RuntimeError(f"[{MODEL_NAME}] CV tuning produced no trial results.")

    best_trial = max(trial_results, key=_cv_trial_sort_key)
    best_threshold = float(best_trial["recommended_threshold"])
    iteration_payload = _derive_final_iteration_payload(best_trial["fold_best_iterations"], best_trial["params"])
    recommended_final_train_config = _sanitize_catboost_config(best_trial["model_config"])
    recommended_final_train_config["iterations"] = int(iteration_payload["final_iterations"])

    cv_rows = []
    for trial_result in sorted(trial_results, key=lambda item: int(item["trial"])):
        row = dict(trial_result["trial_row"])
        row["requested_search_method"] = requested_search_method
        row["effective_search_method"] = effective_search_method
        cv_rows.append(row)
    cv_results_df = pd.DataFrame(cv_rows)
    cv_results_path = run_dir / "cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)

    best_params_payload = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "requested_search_method": requested_search_method,
        "effective_search_method": effective_search_method,
        "search_fallback_reason": search_fallback_reason,
        "best_trial": int(best_trial["trial"]),
        "best_params": best_trial["params"],
        "fold_best_iterations": iteration_payload["fold_best_iterations"],
        "final_iteration_strategy": iteration_payload["final_iteration_strategy"],
        "final_iteration_payload": iteration_payload,
        "recommended_threshold": best_threshold,
        "recommended_final_train_config": recommended_final_train_config,
    }
    best_params_path = run_dir / "best_params.json"
    _write_json(best_params_path, best_params_payload)

    recommended_threshold_payload = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "threshold_metric": best_trial["threshold_search"]["metric"],
        "threshold_start": float(args.threshold_start),
        "threshold_end": float(args.threshold_end),
        "threshold_step": float(args.threshold_step),
        "best_threshold": best_threshold,
        "best_metric_value": float(best_trial["threshold_search"]["best_metric_value"]),
        "best_metrics": best_trial["threshold_search"]["best_metrics"],
        "threshold_results": best_trial["threshold_search"]["threshold_results"],
    }
    recommended_threshold_path = run_dir / "recommended_threshold.json"
    _write_json(recommended_threshold_path, recommended_threshold_payload)

    oof_predictions_path: str | None = None
    if args.save_oof:
        oof_df = pd.DataFrame(
            {
                "PassengerId": raw_train_df["PassengerId"].astype(str),
                "y_true": raw_train_df["Transported"].astype(int),
                "oof_proba": np.asarray(best_trial["oof_probabilities"], dtype=float),
                "recommended_pred": (
                    np.asarray(best_trial["oof_probabilities"], dtype=float) >= float(best_threshold)
                ).astype(int),
                "fold": np.asarray(best_trial["oof_folds"], dtype=int),
            }
        )
        oof_output_path = run_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_output_path, index=False)
        oof_predictions_path = str(oof_output_path)

    final_train_result: dict[str, Any] | None = None
    if args.final_train_after_tune:
        final_train_summary_updates = {
            **iteration_payload,
            "recommended_threshold": best_threshold,
            "tuning_run_id": run_id,
            "requested_search_method": requested_search_method,
            "effective_search_method": effective_search_method,
        }
        final_train_metadata_updates = {
            **iteration_payload,
            "recommended_threshold": best_threshold,
            "tuning_run_id": run_id,
            "tuning_summary_path": relative_to_project(run_dir / "tuning_summary.json", args.project_root),
            "requested_search_method": requested_search_method,
            "effective_search_method": effective_search_method,
        }
        final_train_result = run_training_pipeline(
            project_root=args.project_root,
            artifacts_dir=args.artifacts_dir,
            submissions_dir=args.submissions_dir,
            processed_root=args.processed_dir,
            config=recommended_final_train_config,
            mode="final_train",
            threshold=best_threshold,
            train_data_path=None,
            train_summary_updates=final_train_summary_updates,
            metadata_updates=final_train_metadata_updates,
        )

    tuning_summary_payload = {
        "model_name": MODEL_NAME,
        "run_id": run_id,
        "train_csv_path": str(train_csv_path),
        "n_train_samples": int(len(raw_train_df)),
        "cv_folds": cv_folds,
        "n_trials": n_trials,
        "requested_search_method": requested_search_method,
        "effective_search_method": effective_search_method,
        "search_fallback_reason": search_fallback_reason,
        "early_stopping_rounds": early_stopping_rounds,
        "threshold_metric": str(args.threshold_metric).lower(),
        "best_trial": int(best_trial["trial"]),
        "best_threshold": best_threshold,
        "best_trial_metrics": {
            "mean_cv_accuracy": best_trial["trial_row"]["mean_cv_accuracy"],
            "mean_cv_auc": best_trial["trial_row"]["mean_cv_auc"],
            "oof_accuracy": best_trial["trial_row"]["oof_accuracy"],
            "oof_auc": best_trial["trial_row"]["oof_auc"],
            "oof_logloss": best_trial["trial_row"]["oof_logloss"],
        },
        "best_params": best_trial["params"],
        "fold_best_iterations": iteration_payload["fold_best_iterations"],
        "final_iteration_payload": iteration_payload,
        "outputs": {
            "cv_results_csv": relative_to_project(cv_results_path, args.project_root),
            "best_params_json": relative_to_project(best_params_path, args.project_root),
            "recommended_threshold_json": relative_to_project(recommended_threshold_path, args.project_root),
            "oof_predictions_csv": relative_to_project(oof_predictions_path, args.project_root)
            if oof_predictions_path is not None
            else None,
        },
        "final_train": None,
    }
    if final_train_result is not None:
        tuning_summary_payload["final_train"] = {
            "train_run": final_train_result["train_run"],
            "managed_train_dir": relative_to_project(final_train_result["managed_train_dir"], args.project_root),
            "submission_path": relative_to_project(final_train_result["submission_path"], args.project_root),
            "compatibility_submission_path": relative_to_project(
                final_train_result["compatibility_submission_path"], args.project_root
            ),
        }

    tuning_summary_path = run_dir / "tuning_summary.json"
    _write_json(tuning_summary_path, tuning_summary_payload)
    update_stage_latest(
        project_root=args.project_root,
        artifacts_dir=args.artifacts_dir,
        stage="tuning",
        run_id=run_id,
        model_name=MODEL_NAME,
        extra_payload={"latest_tuning_run": run_id},
    )

    print(
        f"[{MODEL_NAME}] CV tuning completed. Best trial={best_trial['trial']} "
        f"mean_cv_accuracy={best_trial['trial_row']['mean_cv_accuracy']:.6f} "
        f"mean_cv_auc={best_trial['trial_row']['mean_cv_auc']:.6f} "
        f"recommended_threshold={best_threshold:.2f}"
    )
    return {
        "mode": "cv_tune",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "requested_search_method": requested_search_method,
        "effective_search_method": effective_search_method,
        "best_trial": int(best_trial["trial"]),
        "best_params": best_trial["params"],
        "recommended_threshold": best_threshold,
        "fold_best_iterations": iteration_payload["fold_best_iterations"],
        "recommended_final_train_config": recommended_final_train_config,
        "final_train_after_tune": bool(args.final_train_after_tune),
        "final_train_result": final_train_result,
        "tuning_summary_path": str(tuning_summary_path),
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
    train_summary_updates: dict[str, Any] | None = None,
    metadata_updates: dict[str, Any] | None = None,
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
        refresh_saved_bundle=explicit_train_context is None,
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
    if train_summary_updates:
        training_result["train_summary"].update(train_summary_updates)
    if metadata_updates:
        training_result["metadata"].update(metadata_updates)
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
    training_result["artifact_paths"] = legacy_artifact_paths
    training_result["managed_artifact_paths"] = managed_artifact_paths
    training_result["train_run"] = train_run
    training_result["managed_train_dir"] = str(managed_train_dir)
    training_result["submission_path"] = str(submission_path)
    training_result["compatibility_submission_path"] = str(submission_path)
    return training_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CatBoost model and generate a submission.")
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
    parser.add_argument("--train-csv-path", default=None, help="Raw training CSV path used by cv_tune mode.")
    parser.add_argument("--mode", default="final_train", choices=["final_train", "strict_validation", "infer", "cv_tune"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for positive predictions.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of StratifiedKFold splits for cv_tune.")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of hyperparameter trials for cv_tune.")
    parser.add_argument(
        "--search-method",
        default="random",
        choices=["random", "bayes"],
        help="Hyperparameter search strategy for cv_tune.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used for CV splits and sampling.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
        help="Early stopping rounds used inside each cv_tune fold.",
    )
    parser.add_argument(
        "--threshold-metric",
        default=THRESHOLD_FALLBACK_METRIC,
        choices=["accuracy", "f1", "precision", "recall"],
        help="Metric optimized during OOF threshold search.",
    )
    parser.add_argument("--threshold-start", type=float, default=0.30, help="Start of the OOF threshold grid.")
    parser.add_argument("--threshold-end", type=float, default=0.70, help="End of the OOF threshold grid.")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step size of the OOF threshold grid.")
    parser.add_argument("--final-train-after-tune", action="store_true", help="Run final full-data training after cv_tune.")
    parser.set_defaults(save_oof=True)
    parser.add_argument("--save-oof", dest="save_oof", action="store_true", help="Persist oof_predictions.csv in cv_tune mode.")
    parser.add_argument(
        "--no-save-oof",
        dest="save_oof",
        action="store_false",
        help="Skip writing oof_predictions.csv while still using OOF predictions for threshold search.",
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    reject_minimal_runtime_train_data_path(args.train_data_path, model_name=MODEL_NAME)
    if args.mode == "infer":
        return run_inference(args)
    if args.mode == "cv_tune":
        _validate_non_infer_args(args)
        result = run_cv_tuning(args)
        print(
            f"[{MODEL_NAME}] CV tune summary: "
            f"best_trial={result['best_trial']}, recommended_threshold={result['recommended_threshold']}"
        )
        return result

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
