"""Minimal-runtime compatibility shim for removed self-training utilities."""

from __future__ import annotations

from typing import Any


DEFAULT_INFER_BUNDLE_MODE = "processed_default_compatible"
EXPLICIT_TRAIN_DATA_BUNDLE_MODE = "custom_train_data_only"

_REMOVED_PIPELINE_MESSAGE = (
    "Explicit train-data / self-training pipeline has been removed in minimal runtime mode. "
    "Use default processed-bundle training only."
)


def build_explicit_training_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject explicit train-data bundle creation in minimal runtime mode."""
    raise NotImplementedError(_REMOVED_PIPELINE_MESSAGE)
