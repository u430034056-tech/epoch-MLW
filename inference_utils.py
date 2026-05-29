"""Shared read-only helpers for model inference entrypoints."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ID_COLUMN = "PassengerId"
TARGET_COLUMN = "Transported"
DEFAULT_TEST_COLUMNS = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name",
]


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload with stable UTF-8 formatting."""
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


@contextmanager
def pandas_pickle_compat() -> Any:
    """Allow older pandas string/categorical arrays inside saved joblib bundles."""
    import pandas.core.arrays.categorical as categorical_arrays
    import pandas.core.arrays.string_ as string_arrays
    from pandas._libs.arrays import NDArrayBacked

    original_string_setstate = string_arrays.StringArray.__setstate__
    original_categorical_setstate = categorical_arrays.Categorical.__setstate__

    def _compat_string_setstate(self: Any, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], pd.StringDtype):
            values = np.asarray(state[1], dtype=object)
            self.__init__(values)
            return
        original_string_setstate(self, state)

    def _compat_categorical_setstate(self: Any, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], pd.CategoricalDtype):
            NDArrayBacked.__init__(self, np.asarray(state[1]), state[0])
            return
        original_categorical_setstate(self, state)

    string_arrays.StringArray.__setstate__ = _compat_string_setstate
    categorical_arrays.Categorical.__setstate__ = _compat_categorical_setstate
    try:
        yield
    finally:
        string_arrays.StringArray.__setstate__ = original_string_setstate
        categorical_arrays.Categorical.__setstate__ = original_categorical_setstate


def load_joblib_with_pandas_compat(path: str | Path) -> Any:
    """Load a joblib artifact with temporary pandas compatibility shims."""
    resolved_path = Path(path)
    with pandas_pickle_compat():
        return joblib.load(resolved_path)


def read_test_dataframe(test_path: str | Path | None, model_name: str) -> tuple[pd.DataFrame, Path]:
    """Read a raw test CSV from disk with clear path errors."""
    if test_path is None:
        raise ValueError(f"[{model_name}] --test-path is required in infer mode.")

    resolved_path = Path(test_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"[{model_name}] Test CSV not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"[{model_name}] Test path is not a file: {resolved_path}")

    try:
        test_df = pd.read_csv(resolved_path)
    except Exception as exc:
        raise RuntimeError(f"[{model_name}] Failed to read test CSV '{resolved_path}': {type(exc).__name__}: {exc}") from exc

    if test_df.empty:
        raise ValueError(f"[{model_name}] Test CSV '{resolved_path}' is empty.")

    return test_df, resolved_path.resolve()


def validate_test_csv_schema(
    raw_test_df: pd.DataFrame,
    model_name: str,
    expected_schema: list[str] | None = None,
    strict_schema: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """Validate the raw test schema and return normalized PassengerId values."""
    expected_columns = list(expected_schema or DEFAULT_TEST_COLUMNS)
    input_columns = [str(column) for column in raw_test_df.columns.tolist()]

    if TARGET_COLUMN in input_columns:
        raise ValueError(f"[{model_name}] Input test data must not contain '{TARGET_COLUMN}' in infer mode.")
    if ID_COLUMN not in input_columns:
        raise ValueError(f"[{model_name}] Input test data must contain '{ID_COLUMN}'.")

    missing_columns = [column for column in expected_columns if column not in input_columns]
    extra_columns = [column for column in input_columns if column not in expected_columns]
    column_order_matches = input_columns == expected_columns

    if missing_columns:
        raise ValueError(f"[{model_name}] Input test data is missing required columns: {missing_columns}")
    if strict_schema and (extra_columns or not column_order_matches):
        raise ValueError(
            f"[{model_name}] Input test schema does not match the expected column set/order exactly. "
            f"Extra={extra_columns}, order_matches={column_order_matches}"
        )

    passenger_ids = raw_test_df[ID_COLUMN].astype("string").str.strip()
    if passenger_ids.isna().any() or passenger_ids.eq("").any():
        raise ValueError(f"[{model_name}] '{ID_COLUMN}' must not contain missing or empty values.")
    if not passenger_ids.is_unique:
        duplicates = passenger_ids[passenger_ids.duplicated()].astype(str).unique().tolist()[:5]
        raise ValueError(f"[{model_name}] '{ID_COLUMN}' must be unique. Duplicate examples: {duplicates}")

    warnings: list[str] = []
    if extra_columns:
        warnings.append(f"Extra columns ignored for infer validation: {extra_columns}")
    if not column_order_matches:
        warnings.append("Input column order differs from the expected schema.")

    schema_report = {
        "row_count": int(len(raw_test_df)),
        "input_columns": input_columns,
        "expected_columns": expected_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "column_order_matches_expected": bool(column_order_matches),
        "strict_schema": bool(strict_schema),
        "warnings": warnings,
    }
    return passenger_ids.astype(str), schema_report


def extract_bundle_test_features(bundle: dict[str, Any], model_name: str) -> tuple[Any, pd.Series]:
    """Read X_test and test_ids from a saved inference bundle."""
    if "X_test" not in bundle:
        raise KeyError(f"[{model_name}] Bundle does not contain 'X_test'. Infer mode requires a saved test feature matrix.")
    if "test_ids" not in bundle:
        raise KeyError(f"[{model_name}] Bundle does not contain 'test_ids'. Infer mode requires saved test IDs.")

    x_test = bundle["X_test"]
    test_ids = pd.Series(bundle["test_ids"]).astype("string").str.strip()
    if test_ids.isna().any() or test_ids.eq("").any():
        raise ValueError(f"[{model_name}] Bundle 'test_ids' contains missing or empty values.")
    if not test_ids.is_unique:
        raise ValueError(f"[{model_name}] Bundle 'test_ids' must be unique.")
    if hasattr(x_test, "shape") and int(x_test.shape[0]) != int(len(test_ids)):
        raise ValueError(
            f"[{model_name}] Bundle X_test row count ({int(x_test.shape[0])}) does not match test_ids length ({int(len(test_ids))})."
        )

    return x_test, test_ids.astype(str)


def validate_bundle_test_ids(bundle_test_ids: pd.Series, input_passenger_ids: pd.Series, model_name: str) -> None:
    """Require bundle test IDs to match the input CSV exactly, without fallback."""
    bundle_ids = bundle_test_ids.astype(str).tolist()
    input_ids = input_passenger_ids.astype(str).tolist()
    if bundle_ids == input_ids:
        return

    mismatch_index = None
    max_shared = min(len(bundle_ids), len(input_ids))
    for index in range(max_shared):
        if bundle_ids[index] != input_ids[index]:
            mismatch_index = index
            break

    if mismatch_index is not None:
        raise ValueError(
            f"[{model_name}] Bundle test_ids do not match input PassengerId order. "
            f"First mismatch at row {mismatch_index + 1}: bundle='{bundle_ids[mismatch_index]}', input='{input_ids[mismatch_index]}'. "
            "Infer mode forbids automatic preprocessing, automatic retraining, or implicit fallback."
        )

    raise ValueError(
        f"[{model_name}] Bundle test_ids length ({len(bundle_ids)}) does not match input PassengerId length ({len(input_ids)}). "
        "Infer mode forbids automatic preprocessing, automatic retraining, or implicit fallback."
    )


def coerce_predictions_to_bool(predictions: Any, model_name: str) -> np.ndarray:
    """Normalize final submission labels into a strict boolean array."""
    prediction_array = np.asarray(predictions)
    if prediction_array.ndim != 1:
        raise ValueError(f"[{model_name}] Predictions must be a one-dimensional array.")
    if prediction_array.size == 0:
        raise ValueError(f"[{model_name}] Predictions must not be empty.")
    if pd.isna(prediction_array).any():
        raise ValueError(f"[{model_name}] Predictions contain missing values.")

    if prediction_array.dtype == bool:
        return prediction_array.astype(bool)

    if np.issubdtype(prediction_array.dtype, np.number):
        numeric_values = prediction_array.astype(float)
        valid_mask = np.isclose(numeric_values, 0.0) | np.isclose(numeric_values, 1.0)
        if not np.all(valid_mask):
            raise ValueError(f"[{model_name}] Numeric predictions must already be binary before submission conversion.")
        return numeric_values.astype(int).astype(bool)

    unique_values = {str(value) for value in prediction_array.tolist()}
    if unique_values.issubset({"True", "False"}):
        return np.asarray([str(value) == "True" for value in prediction_array.tolist()], dtype=bool)

    raise ValueError(f"[{model_name}] Predictions could not be converted into boolean submission labels.")


def build_submission_frame(passenger_ids: pd.Series, predictions: Any, model_name: str) -> pd.DataFrame:
    """Build and validate the final submission DataFrame."""
    normalized_ids = passenger_ids.astype(str).tolist()
    bool_predictions = coerce_predictions_to_bool(predictions, model_name)
    if int(len(normalized_ids)) != int(len(bool_predictions)):
        raise ValueError(
            f"[{model_name}] Prediction count ({int(len(bool_predictions))}) does not match input row count ({int(len(normalized_ids))})."
        )

    submission = pd.DataFrame(
        {
            ID_COLUMN: normalized_ids,
            TARGET_COLUMN: bool_predictions.astype(bool),
        }
    )
    validate_submission_frame(submission, normalized_ids, model_name)
    return submission


def validate_submission_frame(submission: pd.DataFrame, expected_ids: list[str], model_name: str) -> None:
    """Enforce the submission acceptance contract."""
    expected_columns = [ID_COLUMN, TARGET_COLUMN]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(f"[{model_name}] Submission columns must be exactly {expected_columns}.")
    if int(len(submission)) != int(len(expected_ids)):
        raise ValueError(f"[{model_name}] Submission row count does not match the input test CSV.")
    if submission[ID_COLUMN].astype(str).tolist() != list(expected_ids):
        raise ValueError(f"[{model_name}] Submission PassengerId order does not match the input test CSV.")
    if submission[ID_COLUMN].isna().any() or submission[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError(f"[{model_name}] Submission '{ID_COLUMN}' contains missing or empty values.")
    if not submission[ID_COLUMN].astype(str).is_unique:
        raise ValueError(f"[{model_name}] Submission '{ID_COLUMN}' contains duplicate values.")
    if submission[TARGET_COLUMN].isna().any():
        raise ValueError(f"[{model_name}] Submission '{TARGET_COLUMN}' contains missing values.")


def resolve_submission_output_path(
    project_root: str | Path | None,
    submissions_dir: str | Path | None,
    model_name: str,
    output_name: str | None,
) -> Path:
    """Resolve the final submission output path without allowing arbitrary directories."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    base_dir = Path(submissions_dir) if submissions_dir is not None else root / "submissions"

    if output_name is None:
        filename = f"submission_{model_name}.csv"
    else:
        candidate = Path(output_name)
        if candidate.is_absolute() or candidate.parent != Path("."):
            raise ValueError(f"[{model_name}] --output-name must be a file name only, not a path.")
        filename = candidate.name
        if not filename.lower().endswith(".csv"):
            raise ValueError(f"[{model_name}] --output-name must end with .csv.")

    return base_dir / filename


def save_inference_artifacts(
    *,
    model_name: str,
    submission: pd.DataFrame,
    submission_path: Path,
    compatibility_submission_path: Path | None = None,
    additional_submission_paths: list[Path] | None = None,
    artifact_dir: Path,
    raw_test_path: Path,
    bundle_path: Path,
    model_path: Path,
    schema_report: dict[str, Any],
    threshold: float,
    probabilities: np.ndarray | None,
    save_proba: bool,
    feature_preparation_mode: str = "bundle_test_features",
    extra_summary_fields: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist inference outputs inside artifacts/ and submissions/."""
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    submission.to_csv(submission_path, index=False)

    written_additional_paths: list[str] = []
    compatibility_written = False
    target_paths: list[tuple[Path, bool]] = []
    if compatibility_submission_path is not None:
        target_paths.append((compatibility_submission_path, True))
    if additional_submission_paths:
        target_paths.extend((path, False) for path in additional_submission_paths)

    seen_targets: set[str] = set()
    for extra_path, is_compatibility_path in target_paths:
        normalized = str(extra_path.resolve()) if extra_path.exists() else str(extra_path)
        if normalized in seen_targets or extra_path == submission_path:
            continue
        seen_targets.add(normalized)
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(extra_path, index=False)
        if is_compatibility_path:
            compatibility_written = True
        else:
            written_additional_paths.append(str(extra_path))

    schema_check_path = artifact_dir / "schema_check.json"
    write_json(schema_check_path, schema_report)

    proba_path: Path | None = None
    if save_proba:
        if probabilities is None:
            raise ValueError(f"[{model_name}] save_proba=True but no probability output was provided.")
        proba_array = np.asarray(probabilities, dtype=float)
        if proba_array.ndim != 1 or int(len(proba_array)) != int(len(submission)):
            raise ValueError(f"[{model_name}] Probability output shape does not match the submission row count.")
        proba_path = artifact_dir / "test_pred_proba.csv"
        proba_frame = pd.DataFrame(
            {
                ID_COLUMN: submission[ID_COLUMN].astype(str),
                "pred_proba": proba_array,
                "pred_label": submission[TARGET_COLUMN].astype(bool),
            }
        )
        proba_frame.to_csv(proba_path, index=False)

    summary_path = artifact_dir / "inference_summary.json"
    summary_payload = {
        "model_name": model_name,
        "mode": "infer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_test_path": str(raw_test_path),
        "submission_path": str(submission_path),
        "schema_check_path": str(schema_check_path),
        "probability_path": str(proba_path) if proba_path is not None else None,
        "n_test_samples": int(len(submission)),
        "used_bundle_path": str(bundle_path),
        "used_model_path": str(model_path),
        "feature_preparation_mode": feature_preparation_mode,
        "save_proba": bool(save_proba),
        "threshold": float(threshold),
        "compatibility_submission_path": str(compatibility_submission_path) if compatibility_written else None,
        "additional_submission_paths": written_additional_paths,
    }
    if extra_summary_fields:
        summary_payload.update(json_safe(extra_summary_fields))
    write_json(summary_path, summary_payload)

    return {
        "submission_path": str(submission_path),
        "compatibility_submission_path": str(compatibility_submission_path) if compatibility_submission_path is not None else "",
        "schema_check_path": str(schema_check_path),
        "inference_summary_path": str(summary_path),
        "probability_path": str(proba_path) if proba_path is not None else "",
    }
