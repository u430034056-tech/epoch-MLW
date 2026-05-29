"""Shared helpers for managed train/infer run handling."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


RUN_ID_RE = re.compile(r"^(?P<prefix>[a-z][a-z_]*)_(?P<index>\d{3})$")
MINIMAL_RUNTIME_STRICT_VALIDATION_ERROR = (
    "Strict validation requires fold-local preprocessing; minimal runtime mode does not support it."
)
MINIMAL_RUNTIME_TRAIN_DATA_PATH_ERROR = (
    "Explicit train-data / self-training pipeline has been removed in minimal runtime mode. "
    "Use default processed-bundle training only."
)


class RunManagementError(RuntimeError):
    """Raised when managed run resolution or validation fails."""


def json_safe(value: Any) -> Any:
    """Convert common path/numpy/pandas-like values into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON with stable UTF-8 formatting."""
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def reject_minimal_runtime_train_data_path(train_data_path: str | Path | None, *, model_name: str) -> None:
    """Reject explicit train-data entrypoints in minimal runtime mode."""
    if train_data_path in (None, ""):
        return
    raise ValueError(f"[{model_name}] {MINIMAL_RUNTIME_TRAIN_DATA_PATH_ERROR}")


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the project root for managed run handling."""
    return Path(project_root) if project_root is not None else Path.cwd()


def resolve_artifacts_root(
    artifacts_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the artifacts root directory."""
    root = resolve_project_root(project_root)
    return Path(artifacts_dir) if artifacts_dir is not None else root / "artifacts"


def resolve_submissions_root(
    submissions_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the submissions root directory."""
    root = resolve_project_root(project_root)
    return Path(submissions_dir) if submissions_dir is not None else root / "submissions"


def get_run_root(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    stage: str,
    model_name: str,
) -> Path:
    """Return the managed run root for a given stage/model."""
    return resolve_artifacts_root(artifacts_dir=artifacts_dir, project_root=project_root) / stage / model_name


def get_stage_root(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    stage: str,
    model_name: str | None = None,
) -> Path:
    """Return the artifact root for a generic stage with optional model scoping."""
    base_root = resolve_artifacts_root(artifacts_dir=artifacts_dir, project_root=project_root) / stage
    if model_name is None:
        return base_root
    return base_root / model_name


def get_legacy_artifact_dir(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
) -> Path:
    """Return the legacy artifact directory for a model."""
    return resolve_artifacts_root(artifacts_dir=artifacts_dir, project_root=project_root) / model_name


def get_submission_model_root(
    project_root: str | Path | None,
    submissions_dir: str | Path | None,
    model_name: str,
) -> Path:
    """Return the managed submissions root for a model."""
    return resolve_submissions_root(submissions_dir=submissions_dir, project_root=project_root) / model_name


def relative_to_project(path: str | Path, project_root: str | Path | None) -> str:
    """Return a project-relative path when possible, otherwise an absolute path."""
    resolved_path = Path(path).resolve()
    root = resolve_project_root(project_root).resolve()
    try:
        return str(resolved_path.relative_to(root))
    except ValueError:
        return str(resolved_path)


def current_timestamp() -> str:
    """Return an ISO 8601 timestamp using local timezone information."""
    return datetime.now().astimezone().isoformat()


def get_git_context(project_root: str | Path | None = None) -> dict[str, Any]:
    """Return the current git commit when available, otherwise a stable fallback payload."""
    project_root_path = resolve_project_root(project_root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root_path,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {
            "git_commit": None,
            "git_commit_status": f"git_unavailable:{type(exc).__name__}",
        }

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "not a git repository" in stderr:
            status = "not_a_git_repo"
        else:
            status = f"git_error:{stderr or result.returncode}"
        return {
            "git_commit": None,
            "git_commit_status": status,
        }

    commit = (result.stdout or "").strip()
    if not commit:
        return {
            "git_commit": None,
            "git_commit_status": "git_commit_empty",
        }
    return {
        "git_commit": commit,
        "git_commit_status": "ok",
    }


def format_run_error(
    *,
    model_name: str,
    stage: str,
    message: str,
    run_id: str | None = None,
    attempted_paths: Sequence[str | Path] | None = None,
    fix_hint: str | None = None,
) -> str:
    """Build a detailed run-management error message."""
    parts = [f"[{model_name}]", f"stage={stage}"]
    if run_id is not None:
        parts.append(f"run_id={run_id}")
    header = " ".join(parts)

    path_text = "none"
    if attempted_paths:
        normalized = [str(Path(path)) for path in attempted_paths]
        path_text = "; ".join(normalized)

    fix_text = fix_hint or "Inspect the paths and metadata for this run and retry."
    return f"{header} {message} Attempted path(s): {path_text}. Fix: {fix_text}"


def raise_run_error(
    *,
    model_name: str,
    stage: str,
    message: str,
    run_id: str | None = None,
    attempted_paths: Sequence[str | Path] | None = None,
    fix_hint: str | None = None,
) -> None:
    """Raise a RunManagementError with a normalized message."""
    raise RunManagementError(
        format_run_error(
            model_name=model_name,
            stage=stage,
            message=message,
            run_id=run_id,
            attempted_paths=attempted_paths,
            fix_hint=fix_hint,
        )
    )


def _extract_run_index(run_id: str, prefix: str) -> int | None:
    match = RUN_ID_RE.match(run_id)
    if match is None or match.group("prefix") != prefix:
        return None
    return int(match.group("index"))


def list_existing_runs(run_root: Path, prefix: str) -> list[str]:
    """Return sorted run ids for a managed run root."""
    if not run_root.exists():
        return []

    items: list[tuple[int, str]] = []
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        index = _extract_run_index(child.name, prefix)
        if index is None:
            continue
        items.append((index, child.name))
    return [name for _, name in sorted(items)]


def list_existing_runs_across_roots(run_roots: Iterable[Path], prefix: str) -> list[str]:
    """Return sorted unique run ids found across multiple roots."""
    seen: dict[str, int] = {}
    for root in run_roots:
        for run_id in list_existing_runs(root, prefix):
            index = _extract_run_index(run_id, prefix)
            if index is not None:
                seen[run_id] = index
    return [name for name, _ in sorted(seen.items(), key=lambda item: item[1])]


def get_next_run_id(run_root: Path, prefix: str) -> str:
    """Return the next run id for a single root."""
    return get_next_run_id_across_roots([run_root], prefix)


def get_next_run_id_across_roots(run_roots: Iterable[Path], prefix: str) -> str:
    """Return the next run id considering all provided roots."""
    existing = list_existing_runs_across_roots(run_roots, prefix)
    if not existing:
        return f"{prefix}_001"
    last_index = max(_extract_run_index(run_id, prefix) or 0 for run_id in existing)
    return f"{prefix}_{last_index + 1:03d}"


def get_latest_train_run(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
) -> str | None:
    """Return the latest managed train run id, if any."""
    run_root = get_run_root(project_root=project_root, artifacts_dir=artifacts_dir, stage="train", model_name=model_name)
    latest_path = run_root / "latest.json"
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            latest_train_run = payload.get("latest_train_run")
            if isinstance(latest_train_run, str) and (run_root / latest_train_run).is_dir():
                return latest_train_run
    existing = list_existing_runs(run_root, "train")
    return existing[-1] if existing else None


def get_latest_infer_run(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
) -> str | None:
    """Return the latest managed infer run id, if any."""
    run_root = get_run_root(project_root=project_root, artifacts_dir=artifacts_dir, stage="infer", model_name=model_name)
    latest_path = run_root / "latest.json"
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            latest_infer_run = payload.get("latest_infer_run")
            if isinstance(latest_infer_run, str) and (run_root / latest_infer_run).is_dir():
                return latest_infer_run
    existing = list_existing_runs(run_root, "infer")
    return existing[-1] if existing else None


def resolve_train_run_dir(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
    train_run: str,
) -> Path:
    """Resolve a managed train run directory and fail clearly if missing."""
    train_root = get_run_root(project_root=project_root, artifacts_dir=artifacts_dir, stage="train", model_name=model_name)
    train_run_dir = train_root / train_run
    if not train_run_dir.exists() or not train_run_dir.is_dir():
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=train_run,
            message="Managed train run directory does not exist.",
            attempted_paths=[train_run_dir],
            fix_hint="Use an existing train run under artifacts/train/<model>/ or omit --train-run to use latest managed run or legacy fallback.",
        )
    return train_run_dir


def read_json_file(
    path: Path,
    *,
    model_name: str,
    stage: str,
    run_id: str | None,
    fix_hint: str,
) -> dict[str, Any]:
    """Load and validate a JSON object from disk."""
    if not path.exists() or not path.is_file():
        raise_run_error(
            model_name=model_name,
            stage=stage,
            run_id=run_id,
            message="Run metadata file is missing or unreadable.",
            attempted_paths=[path],
            fix_hint=fix_hint,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise_run_error(
            model_name=model_name,
            stage=stage,
            run_id=run_id,
            message=f"Run metadata is corrupted: {type(exc).__name__}: {exc}",
            attempted_paths=[path],
            fix_hint=fix_hint,
        )
    if not isinstance(payload, dict):
        raise_run_error(
            model_name=model_name,
            stage=stage,
            run_id=run_id,
            message="Run metadata must be a JSON object.",
            attempted_paths=[path],
            fix_hint=fix_hint,
        )
    return payload


def validate_train_run_meta(
    train_run_dir: Path,
    *,
    model_name: str,
    train_run: str,
) -> dict[str, Any]:
    """Validate managed train run metadata."""
    meta_path = train_run_dir / "run_meta.json"
    fix_hint = "Re-run training to regenerate a valid managed train run, or choose another train run."
    payload = read_json_file(
        meta_path,
        model_name=model_name,
        stage="infer",
        run_id=train_run,
        fix_hint=fix_hint,
    )

    required_keys = {"run_id", "stage", "model_name"}
    missing = sorted(required_keys - set(payload))
    if missing:
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=train_run,
            message=f"Run metadata is missing required fields: {missing}.",
            attempted_paths=[meta_path],
            fix_hint=fix_hint,
        )
    if payload.get("stage") != "train":
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=train_run,
            message=f"Run metadata stage must be 'train', got '{payload.get('stage')}'.",
            attempted_paths=[meta_path],
            fix_hint=fix_hint,
        )
    if payload.get("run_id") != train_run:
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=train_run,
            message=f"Run metadata run_id '{payload.get('run_id')}' does not match directory '{train_run}'.",
            attempted_paths=[meta_path, train_run_dir],
            fix_hint=fix_hint,
        )
    if payload.get("model_name") != model_name:
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=train_run,
            message=(
                f"Managed train run metadata model_name '{payload.get('model_name')}' "
                f"does not match requested model '{model_name}'."
            ),
            attempted_paths=[meta_path, train_run_dir],
            fix_hint="Use a train run created by the same model script, or choose the correct model entrypoint.",
        )
    return payload


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return a de-duplicated list while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_model_artifact_from_train_run(
    train_run_dir: Path,
    *,
    model_name: str,
    train_run: str,
    candidate_names: Sequence[str],
) -> Path:
    """Resolve a model artifact inside a managed train run."""
    candidates = [train_run_dir / candidate for candidate in dedupe_preserve_order(candidate_names) if candidate]
    for candidate_path in candidates:
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path
    raise_run_error(
        model_name=model_name,
        stage="infer",
        run_id=train_run,
        message="No model artifact matched any managed train-run candidate filename.",
        attempted_paths=candidates,
        fix_hint="Ensure the managed train run contains a saved model artifact or re-run training for this model.",
    )


def resolve_legacy_model_artifact(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    *,
    model_name: str,
    legacy_model_filename: str,
) -> Path:
    """Resolve the legacy model artifact path."""
    legacy_dir = get_legacy_artifact_dir(project_root=project_root, artifacts_dir=artifacts_dir, model_name=model_name)
    model_path = legacy_dir / legacy_model_filename
    if not model_path.exists() or not model_path.is_file():
        raise_run_error(
            model_name=model_name,
            stage="infer",
            message="Legacy model artifact fallback could not be resolved.",
            attempted_paths=[model_path],
            fix_hint="Train the model first or create a managed train run before running infer.",
        )
    return model_path


def get_infer_run_paths(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    submissions_dir: str | Path | None,
    model_name: str,
    infer_run: str,
) -> tuple[Path, Path, Path]:
    """Return managed infer artifact dir, submission dir, and submission path."""
    artifact_dir = get_run_root(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage="infer",
        model_name=model_name,
    ) / infer_run
    submission_dir = get_submission_model_root(
        project_root=project_root,
        submissions_dir=submissions_dir,
        model_name=model_name,
    ) / infer_run
    return artifact_dir, submission_dir, submission_dir / "submission.csv"


def ensure_infer_run_available(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    submissions_dir: str | Path | None,
    model_name: str,
    infer_run: str,
) -> tuple[Path, Path, Path]:
    """Ensure an explicitly provided infer run does not collide with existing paths."""
    artifact_dir, submission_dir, submission_path = get_infer_run_paths(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        submissions_dir=submissions_dir,
        model_name=model_name,
        infer_run=infer_run,
    )
    conflicts = [path for path in (artifact_dir, submission_dir) if path.exists()]
    if conflicts:
        raise_run_error(
            model_name=model_name,
            stage="infer",
            run_id=infer_run,
            message="Explicit infer run collides with an existing output directory.",
            attempted_paths=conflicts,
            fix_hint="Choose a new --infer-run value. Overwrite is not supported in this implementation.",
        )
    return artifact_dir, submission_dir, submission_path


def allocate_infer_run(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    submissions_dir: str | Path | None,
    model_name: str,
) -> tuple[str, Path, Path, Path]:
    """Allocate the next available infer run id using both infer roots."""
    infer_artifact_root = get_run_root(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage="infer",
        model_name=model_name,
    )
    infer_submission_root = get_submission_model_root(
        project_root=project_root,
        submissions_dir=submissions_dir,
        model_name=model_name,
    )
    infer_run = get_next_run_id_across_roots([infer_artifact_root, infer_submission_root], "infer")
    artifact_dir, submission_dir, submission_path = get_infer_run_paths(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        submissions_dir=submissions_dir,
        model_name=model_name,
        infer_run=infer_run,
    )
    return infer_run, artifact_dir, submission_dir, submission_path


def allocate_stage_run(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    stage: str,
    prefix: str,
    model_name: str | None = None,
) -> tuple[str, Path]:
    """Allocate the next run id for a generic artifact stage."""
    stage_root = get_stage_root(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage=stage,
        model_name=model_name,
    )
    run_id = get_next_run_id(stage_root, prefix)
    return run_id, stage_root / run_id


def create_infer_run_dirs(artifact_dir: Path, submission_dir: Path) -> None:
    """Create managed infer output directories."""
    artifact_dir.mkdir(parents=True, exist_ok=False)
    submission_dir.mkdir(parents=True, exist_ok=False)


def update_run_registry(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
    stage: str,
    run_id: str,
) -> Path:
    """Update the run registry with a successful managed run."""
    registry_path = resolve_artifacts_root(artifacts_dir=artifacts_dir, project_root=project_root) / "run_registry.json"
    if registry_path.exists():
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    model_entry = payload.get(model_name)
    if not isinstance(model_entry, dict):
        model_entry = {"train_runs": [], "infer_runs": []}

    key = "train_runs" if stage == "train" else "infer_runs"
    runs = model_entry.get(key)
    if not isinstance(runs, list):
        runs = []
    if run_id not in runs:
        runs.append(run_id)
        runs = sorted(
            dedupe_preserve_order(str(item) for item in runs),
            key=lambda item: _extract_run_index(item, "train" if stage == "train" else "infer") or 0,
        )
    model_entry[key] = runs
    model_entry.setdefault("train_runs", [])
    model_entry.setdefault("infer_runs", [])
    payload[model_name] = model_entry

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(registry_path, payload)
    return registry_path


def update_latest_run(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    model_name: str,
    stage: str,
    run_id: str,
) -> Path:
    """Persist a stage-local latest.json marker."""
    latest_path = get_run_root(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage=stage,
        model_name=model_name,
    ) / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    key = "latest_train_run" if stage == "train" else "latest_infer_run"
    write_json(latest_path, {key: run_id})
    return latest_path


def update_stage_latest(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    stage: str,
    run_id: str,
    model_name: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    """Persist a latest marker for a generic stage."""
    payload = {f"latest_{stage}_run": run_id}
    if extra_payload:
        payload.update(extra_payload)
    return write_stage_latest_payload(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage=stage,
        payload=payload,
        model_name=model_name,
    )


def write_stage_latest_payload(
    project_root: str | Path | None,
    artifacts_dir: str | Path | None,
    stage: str,
    payload: dict[str, Any],
    model_name: str | None = None,
) -> Path:
    """Persist an arbitrary latest marker payload for a stage."""
    latest_path = get_stage_root(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        stage=stage,
        model_name=model_name,
    ) / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(latest_path, payload)
    return latest_path
