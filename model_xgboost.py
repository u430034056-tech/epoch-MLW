"""Training and inference utilities for the XGBoost branch."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, log_loss
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
from preprocess import get_project_paths, load_preprocessed_bundle as load_saved_preprocessed_bundle, run_all_preprocessing
from run_utils import (
    MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR,
    allocate_infer_run,
    create_infer_run_dirs,
    current_timestamp,
    dedupe_preserve_order,
    ensure_infer_run_available,
    get_next_run_id,
    get_run_root,
    list_existing_runs,
    raise_run_error,
    reject_minimal_runtime_train_data_path,
    relative_to_project,
    resolve_legacy_model_artifact,
    resolve_model_artifact_from_train_run,
    resolve_train_run_dir,
    update_latest_run,
    update_run_registry,
    validate_train_run_meta,
    write_json,
)
from selftrain_utils import DEFAULT_INFER_BUNDLE_MODE


MODEL_NAME = "xgboost"
LEGACY_MODEL_FILENAME = "model.joblib"
REQUIRED_BUNDLE_KEYS = ("X_train", "X_test", "y_train", "train_ids", "test_ids")
DEFAULT_CONFIG = {
    "n_splits": 5,
    "split_random_state": 42,
    "num_boost_round": 6000,
    "early_stopping_rounds": 200,
    "member_groups": [
        {
            "name": "default",
            "seeds": [42],
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "eta": 0.3,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
        },
        {
            "name": "regularized",
            "seeds": [42, 123, 314],
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "eta": 0.05,
                "max_depth": 5,
                "min_child_weight": 2,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "deep",
            "seeds": [42, 123, 314],
            "params": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "eta": 0.1,
                "max_depth": 7,
                "min_child_weight": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "gamma": 0.0,
                "reg_alpha": 0.05,
                "reg_lambda": 1.5,
            },
        },
    ],
}


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root is not None else Path.cwd()


def _resolve_processed_root(
    processed_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    if processed_root is not None:
        return Path(processed_root)
    return get_project_paths(project_root)["processed_root"]


def _resolve_artifact_dir(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path | None = None,
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


def _get_xgboost_module() -> Any:
    try:
        import xgboost as xgb_module
    except ImportError as exc:
        raise ImportError(
            f"[{MODEL_NAME}] XGBoost is not available in this environment. "
            "Install a working xgboost package before running XGBoost train/infer."
        ) from exc
    return xgb_module


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
                fix_hint="Use a default processed-bundle train run for infer mode.",
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
    if bundle_path is None:
        bundle_path = _resolve_bundle_path(processed_dir=None, project_root=project_root)

    write_json(train_dir / "train_config.json", training_result["config"])
    write_json(train_dir / "train_summary.json", training_result["train_summary"])
    write_json(
        train_dir / "bundle_ref.json",
        {
            "bundle_path": relative_to_project(bundle_path, project_root),
            "bundle_role": "training_and_test_feature_source",
        },
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
            "bundle_path": relative_to_project(bundle_path, project_root),
            "train_data_path": None,
            "source_test_path": None,
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
        raise ValueError(f"[{MODEL_NAME}] y_train is empty.")

    normalized_values: set[int] = set()
    for value in y_series.dropna().tolist():
        if isinstance(value, (bool, np.bool_)):
            normalized_values.add(int(bool(value)))
            continue
        if isinstance(value, (int, np.integer)):
            if int(value) in {0, 1}:
                normalized_values.add(int(value))
                continue
            raise ValueError(f"[{MODEL_NAME}] y_train must contain only binary labels.")
        if isinstance(value, (float, np.floating)):
            if float(value) in {0.0, 1.0}:
                normalized_values.add(int(value))
                continue
            raise ValueError(f"[{MODEL_NAME}] y_train must contain only binary labels.")
        raise ValueError(f"[{MODEL_NAME}] y_train must contain only binary labels.")

    if normalized_values - {0, 1}:
        raise ValueError(f"[{MODEL_NAME}] y_train must contain only binary labels.")

    return y_series.astype(int).to_numpy()


def _sample_submission_path(project_root: str | Path | None = None) -> Path:
    return get_project_paths(project_root)["data_dir"] / "sample_submission.csv"


def _build_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    model_config = copy.deepcopy(DEFAULT_CONFIG)
    if config is not None:
        model_config.update(config)
    member_groups = model_config.get("member_groups")
    if not isinstance(member_groups, list) or not member_groups:
        raise ValueError(f"[{MODEL_NAME}] Config must provide a non-empty 'member_groups' list.")
    return model_config


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


def validate_xgb_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Validate XGBoost bundle contents without changing preprocessing."""
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
        "feature_names": [str(name) for name in feature_names] if feature_names is not None else None,
    }


def _resolve_feature_names(validated_bundle: dict[str, Any]) -> list[str]:
    feature_names = validated_bundle.get("feature_names")
    if feature_names is not None:
        return [str(name) for name in feature_names]
    return [f"f{i}" for i in range(int(validated_bundle["X_train"].shape[1]))]


def _best_iteration(booster: Any) -> int:
    best_iteration = getattr(booster, "best_iteration", None)
    if isinstance(best_iteration, (int, np.integer)) and int(best_iteration) >= 0:
        return int(best_iteration)
    num_rounds = getattr(booster, "num_boosted_rounds", None)
    if callable(num_rounds):
        return int(num_rounds()) - 1
    return 0


def _predict_booster_proba(booster: Any, dmatrix: Any) -> np.ndarray:
    best_iteration = _best_iteration(booster)
    return np.asarray(booster.predict(dmatrix, iteration_range=(0, best_iteration + 1)), dtype=float)


def _train_group(
    group_config: dict[str, Any],
    *,
    X_train: Any,
    y_train: np.ndarray,
    feature_names: list[str],
    model_config: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    group_name = str(group_config["name"])
    base_params = dict(group_config["params"])
    seeds = [int(seed) for seed in group_config["seeds"]]
    xgb_module = _get_xgboost_module()
    splitter = StratifiedKFold(
        n_splits=int(model_config["n_splits"]),
        shuffle=True,
        random_state=int(model_config["split_random_state"]),
    )

    seed_oof_predictions: list[np.ndarray] = []
    members: list[dict[str, Any]] = []
    for seed in seeds:
        seed_oof = np.zeros(len(y_train), dtype=float)
        best_iterations: list[int] = []
        params = dict(base_params)
        params["seed"] = seed
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_train, y_train), start=1):
            dtrain = xgb_module.DMatrix(X_train[train_idx], label=y_train[train_idx], feature_names=feature_names)
            dvalid = xgb_module.DMatrix(X_train[valid_idx], label=y_train[valid_idx], feature_names=feature_names)
            booster = xgb_module.train(
                params,
                dtrain,
                num_boost_round=int(model_config["num_boost_round"]),
                evals=[(dvalid, "valid")],
                early_stopping_rounds=int(model_config["early_stopping_rounds"]),
                verbose_eval=False,
            )
            seed_oof[valid_idx] = _predict_booster_proba(booster, dvalid)
            best_iteration = _best_iteration(booster)
            best_iterations.append(best_iteration)
            members.append(
                {
                    "group_name": group_name,
                    "seed": seed,
                    "fold": fold,
                    "best_iteration": best_iteration,
                    "booster": booster,
                }
            )

        seed_accuracy = float(accuracy_score(y_train, (seed_oof >= float(threshold)).astype(int)))
        print(
            f"[{MODEL_NAME}] Group '{group_name}' seed={seed}: "
            f"OOF accuracy={seed_accuracy:.6f}, mean_best_iteration={np.mean(best_iterations):.1f}"
        )
        seed_oof_predictions.append(seed_oof)

    group_oof = np.mean(np.vstack(seed_oof_predictions), axis=0)
    group_accuracy = float(accuracy_score(y_train, (group_oof >= float(threshold)).astype(int)))
    group_log_loss = float(log_loss(y_train, np.clip(group_oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
    return {
        "name": group_name,
        "params": base_params,
        "seeds": seeds,
        "oof_probabilities": group_oof,
        "oof_accuracy": group_accuracy,
        "oof_log_loss": group_log_loss,
        "members": members,
    }


def _select_group_blend(groups: list[dict[str, Any]], y_train: np.ndarray, threshold: float) -> list[str]:
    remaining = list(range(len(groups)))
    selected: list[int] = []
    best_global_accuracy = -1.0
    while remaining:
        best_candidate: int | None = None
        best_candidate_accuracy = -1.0
        for index in remaining:
            candidate = selected + [index]
            candidate_prob = np.mean(
                np.vstack([groups[group_index]["oof_probabilities"] for group_index in candidate]),
                axis=0,
            )
            candidate_accuracy = float(accuracy_score(y_train, (candidate_prob >= float(threshold)).astype(int)))
            if candidate_accuracy > best_candidate_accuracy:
                best_candidate_accuracy = candidate_accuracy
                best_candidate = index
        if best_candidate is None or best_candidate_accuracy <= best_global_accuracy:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_global_accuracy = best_candidate_accuracy
        print(
            f"[{MODEL_NAME}] Added blend group '{groups[best_candidate]['name']}' "
            f"with OOF accuracy={best_candidate_accuracy:.6f}"
        )

    if not selected:
        best_index = max(range(len(groups)), key=lambda idx: groups[idx]["oof_accuracy"])
        selected = [best_index]

    return [groups[index]["name"] for index in selected]


def _build_model_payload(
    groups: list[dict[str, Any]],
    *,
    feature_names: list[str],
    selected_group_names: list[str],
) -> dict[str, Any]:
    return {
        "model_type": "xgboost_group_blend_ensemble",
        "feature_names": feature_names,
        "selected_group_names": selected_group_names,
        "groups": groups,
    }


def predict_proba(model: dict[str, Any], X: Any) -> np.ndarray:
    """Return positive-class probabilities for the screened XGBoost ensemble."""
    feature_names = model.get("feature_names")
    groups = model.get("groups")
    selected_group_names = model.get("selected_group_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise TypeError(f"[{MODEL_NAME}] Loaded ensemble is missing feature name metadata.")
    if not isinstance(groups, list) or not groups:
        raise TypeError(f"[{MODEL_NAME}] Loaded ensemble does not contain any booster groups.")
    if not isinstance(selected_group_names, list) or not selected_group_names:
        raise TypeError(f"[{MODEL_NAME}] Loaded ensemble does not contain any selected blend groups.")

    xgb_module = _get_xgboost_module()
    dmatrix = xgb_module.DMatrix(X, feature_names=feature_names)
    group_predictions: list[np.ndarray] = []
    selected_names = set(str(name) for name in selected_group_names)
    for group in groups:
        if group.get("name") not in selected_names:
            continue
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise TypeError(f"[{MODEL_NAME}] Selected XGBoost group '{group.get('name')}' has no members.")
        member_probabilities = [_predict_booster_proba(member["booster"], dmatrix) for member in members]
        group_predictions.append(np.mean(np.vstack(member_probabilities), axis=0))

    if not group_predictions:
        raise TypeError(f"[{MODEL_NAME}] No selected blend group produced predictions.")
    return np.mean(np.vstack(group_predictions), axis=0)


def predict(model: dict[str, Any], X: Any, threshold: float = 0.5) -> np.ndarray:
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
    """Train the screened XGBoost ensemble from the existing bundle."""
    if mode == "strict_validation":
        raise RuntimeError(MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR)
    if mode != "final_train":
        raise ValueError(f"[{MODEL_NAME}] Unsupported mode: {mode}")

    if bundle is None:
        bundle = load_preprocessed_data(processed_root=processed_root, project_root=project_root, save_outputs=False)

    validated = validate_xgb_bundle(bundle)
    model_config = _build_model_config(config)
    feature_names = _resolve_feature_names(validated)
    groups = [
        _train_group(
            group_config,
            X_train=validated["X_train"],
            y_train=validated["y_train"],
            feature_names=feature_names,
            model_config=model_config,
            threshold=threshold,
        )
        for group_config in model_config["member_groups"]
    ]
    selected_group_names = _select_group_blend(groups, validated["y_train"], threshold)
    model_payload = _build_model_payload(groups, feature_names=feature_names, selected_group_names=selected_group_names)
    train_probabilities = predict_proba(model_payload, validated["X_train"])
    train_predictions = train_probabilities >= float(threshold)
    train_accuracy = float(accuracy_score(validated["y_train"], train_predictions.astype(int)))
    train_log_loss_value = float(log_loss(validated["y_train"], np.clip(train_probabilities, 1e-6, 1 - 1e-6), labels=[0, 1]))

    train_summary = {
        "summary_type": "cross_validated_group_blend",
        "mode": mode,
        "train_accuracy": train_accuracy,
        "train_log_loss": train_log_loss_value,
        "train_positive_rate_predicted": float(np.mean(train_predictions)),
        "train_positive_rate_observed": float(np.mean(validated["y_train"])),
        "n_train_samples": int(validated["X_train"].shape[0]),
        "n_test_samples": int(validated["X_test"].shape[0]),
        "group_count": int(len(groups)),
        "selected_group_names": selected_group_names,
        "booster_count": int(sum(len(group["members"]) for group in groups)),
    }
    metadata = {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(model_config["split_random_state"]),
        "train_shape": [int(size) for size in validated["X_train"].shape],
        "test_shape": [int(size) for size in validated["X_test"].shape],
        "threshold": float(threshold),
        "bundle_path": bundle.get("save_path"),
        "bundle_source": bundle.get("save_path") or "in_memory_preprocessing",
        "feature_count": int(validated["X_train"].shape[1]),
        "selected_group_names": selected_group_names,
        "group_count": int(len(groups)),
        "booster_count": int(sum(len(group["members"]) for group in groups)),
        "config": model_config,
    }

    return {
        "model": model_payload,
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
    """Persist the screened XGBoost ensemble and metadata outside processed/."""
    root = _resolve_project_root(project_root)
    output_dir = Path(artifact_dir) if artifact_dir is not None else root / "artifacts" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / LEGACY_MODEL_FILENAME
    config_path = output_dir / "config.json"
    metadata_path = output_dir / "metadata.json"
    importance_path = output_dir / "feature_importance.csv"

    selected_group_names = set(training_result["model"]["selected_group_names"])
    saved_groups: list[dict[str, Any]] = []
    selected_boosters: list[dict[str, Any]] = []
    for group in training_result["model"]["groups"]:
        saved_members: list[dict[str, Any]] = []
        for member in group["members"]:
            booster_file = f"xgb_{group['name']}_s{member['seed']}_f{member['fold']}.json"
            member["booster"].save_model(str(output_dir / booster_file))
            saved_member = {
                "group_name": group["name"],
                "seed": int(member["seed"]),
                "fold": int(member["fold"]),
                "best_iteration": int(member["best_iteration"]),
                "model_file": booster_file,
            }
            saved_members.append(saved_member)
            if group["name"] in selected_group_names:
                selected_boosters.append({"booster": member["booster"]})
        saved_groups.append(
            {
                "name": group["name"],
                "params": group["params"],
                "seeds": group["seeds"],
                "oof_accuracy": group["oof_accuracy"],
                "oof_log_loss": group["oof_log_loss"],
                "members": saved_members,
            }
        )

    payload = {
        "model_type": "xgboost_group_blend_ensemble",
        "feature_names": training_result["model"]["feature_names"],
        "selected_group_names": training_result["model"]["selected_group_names"],
        "groups": saved_groups,
    }
    joblib.dump(payload, model_path)
    _write_json(config_path, training_result["config"])

    metadata_payload = dict(training_result["metadata"])
    metadata_payload["train_summary"] = training_result["train_summary"]
    _write_json(metadata_path, metadata_payload)

    feature_names = training_result["model"]["feature_names"]
    if feature_names and selected_boosters:
        gain_totals = {feature_name: 0.0 for feature_name in feature_names}
        weight_totals = {feature_name: 0.0 for feature_name in feature_names}
        booster_count = float(len(selected_boosters))
        for member in selected_boosters:
            booster = member["booster"]
            for feature_name, value in booster.get_score(importance_type="gain").items():
                gain_totals[str(feature_name)] = gain_totals.get(str(feature_name), 0.0) + float(value)
            for feature_name, value in booster.get_score(importance_type="weight").items():
                weight_totals[str(feature_name)] = weight_totals.get(str(feature_name), 0.0) + float(value)
        pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance_gain_mean": [gain_totals.get(name, 0.0) / booster_count for name in feature_names],
                "importance_weight_mean": [weight_totals.get(name, 0.0) / booster_count for name in feature_names],
            }
        ).sort_values("importance_gain_mean", ascending=False).to_csv(importance_path, index=False)

    artifact_paths = {
        "artifact_dir": str(output_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
    }
    if importance_path.exists():
        artifact_paths["feature_importance_path"] = str(importance_path)
    print(f"[{MODEL_NAME}] Saved artifacts to: {output_dir}")
    return artifact_paths


def generate_submission(
    model: dict[str, Any],
    bundle: dict[str, Any],
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
) -> Path:
    """Generate a Kaggle-style submission aligned to sample_submission.csv."""
    validated = validate_xgb_bundle(bundle)
    predictions = np.asarray(predict(model, validated["X_test"], threshold=threshold), dtype=bool)
    submission = build_submission_frame(pd.Series(validated["test_ids"]).astype(str), predictions, MODEL_NAME)

    sample_submission = pd.read_csv(_sample_submission_path(project_root))
    sample_ids = sample_submission["PassengerId"].astype(str).tolist()
    if submission["PassengerId"].astype(str).tolist() != sample_ids:
        raise ValueError(f"[{MODEL_NAME}] Submission PassengerId order does not match sample_submission.csv.")
    if int(len(submission)) != int(len(sample_submission)):
        raise ValueError(f"[{MODEL_NAME}] Submission row count does not match sample_submission.csv.")

    final_output_path = (
        Path(output_path)
        if output_path is not None
        else _resolve_project_root(project_root) / "submissions" / f"submission_{MODEL_NAME}.csv"
    )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(final_output_path, index=False)
    print(f"[{MODEL_NAME}] Saved submission to: {final_output_path}")
    return final_output_path


def load_model_artifact_from_path(model_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a trained XGBoost ensemble payload from an explicit path."""
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"[{MODEL_NAME}] Model artifact not found at '{resolved_model_path}'.")

    try:
        payload = load_joblib_with_pandas_compat(resolved_model_path)
    except Exception as exc:
        raise RuntimeError(
            f"[{MODEL_NAME}] Failed to load model artifact from '{resolved_model_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise TypeError(f"[{MODEL_NAME}] Loaded artifact is not a dictionary payload.")
    if payload.get("model_type") != "xgboost_group_blend_ensemble":
        raise TypeError(f"[{MODEL_NAME}] Loaded artifact is not a screened XGBoost ensemble.")

    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise TypeError(f"[{MODEL_NAME}] Loaded ensemble does not contain any booster groups.")

    loaded_groups: list[dict[str, Any]] = []
    for group in groups:
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise TypeError(f"[{MODEL_NAME}] Loaded group '{group.get('name')}' does not contain any members.")
        loaded_members: list[dict[str, Any]] = []
        for member in members:
            model_file = member.get("model_file")
            if not isinstance(model_file, str) or not model_file:
                raise TypeError(f"[{MODEL_NAME}] Loaded member metadata is missing model_file.")
            booster_path = resolved_model_path.parent / model_file
            if not booster_path.exists():
                raise FileNotFoundError(f"[{MODEL_NAME}] Booster artifact not found at '{booster_path}'.")
            xgb_module = _get_xgboost_module()
            booster = xgb_module.Booster()
            booster.load_model(str(booster_path))
            loaded_member = dict(member)
            loaded_member["booster"] = booster
            loaded_members.append(loaded_member)
        loaded_group = dict(group)
        loaded_group["members"] = loaded_members
        loaded_groups.append(loaded_group)

    model = dict(payload)
    model["groups"] = loaded_groups
    return model, resolved_model_path


def load_model_artifact(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load a trained XGBoost artifact without triggering retraining."""
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
        "bundle_path": bundle_path,
        "passenger_ids": passenger_ids,
        "schema_report": schema_report,
        "X_test": x_test,
        "feature_preparation_mode": "bundle_test_features",
    }


def predict_test(
    model: dict[str, Any],
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
    """Run strict bundle-based infer mode for the XGBoost branch."""
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
    training_result = train_model(
        project_root=project_root_path,
        processed_root=processed_root,
        config=config,
        mode=mode,
        threshold=threshold,
    )
    training_result["metadata"]["infer_bundle_mode"] = DEFAULT_INFER_BUNDLE_MODE
    training_result["metadata"]["train_data_mode"] = "default_processed_bundle"
    training_result["metadata"]["source_dataset_run"] = None
    training_result["metadata"]["source_selftrain_run"] = None
    training_result["metadata"]["source_infer_run"] = None

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
    return {
        **training_result,
        "artifact_paths": legacy_artifact_paths,
        "managed_artifact_paths": managed_artifact_paths,
        "train_run": train_run,
        "managed_train_dir": str(managed_train_dir),
        "submission_path": str(submission_path),
        "compatibility_submission_path": str(submission_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost model and generate a submission.")
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
