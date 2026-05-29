"""Centralised configuration for the XGBoost pipeline.

All paths are absolute to avoid surprises when scripts are launched from
different working directories.  All hyper-parameter defaults live here, so
the ablation log can reference a single source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "processed"
COMMON_BUNDLE: Final[Path] = PROCESSED_DIR / "common" / "preprocessed_common.joblib"
XGB_BUNDLE: Final[Path] = PROCESSED_DIR / "xgboost" / "preprocessed_xgboost.joblib"

REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports" / "xgboost"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
LOGS_DIR: Final[Path] = REPORTS_DIR / "logs"
SUBMISSIONS_DIR: Final[Path] = REPORTS_DIR / "submissions"

for _d in (REPORTS_DIR, FIGURES_DIR, LOGS_DIR, SUBMISSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cross-validation & reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42
N_SPLITS: Final[int] = 5
# Extra seeds used for multi-seed bagging (A6).  Each seed produces one set of
# 5-fold OOF + test predictions; the final prediction averages probabilities.
SEED_POOL: Final[tuple[int, ...]] = (42, 2024, 7, 1337, 88)

# ---------------------------------------------------------------------------
# Feature schema (applies to the common DataFrame we derive features from)
# ---------------------------------------------------------------------------
ID_COLUMN: Final[str] = "PassengerId"
GROUP_COLUMN: Final[str] = "GroupID"
TARGET_COLUMN: Final[str] = "Transported"

# These columns are dropped from the model matrix: high cardinality text,
# raw cabin strings, or audit identifiers that would leak / overfit.
DROP_COLUMNS: Final[tuple[str, ...]] = (
    "PassengerId",
    "GroupID",
    "Cabin",
    "Name",
    "Surname",
)

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "AgeGroup",
    "CabinNumBin",
    "DeckSide",
    "HomePlanetDestination",
)

# Columns used as keys for fold-local target encoding.  High-cardinality
# combinations are safer here than one-hotting them.
TARGET_ENCODE_COLUMNS: Final[tuple[str, ...]] = (
    "HomePlanet",
    "Destination",
    "Deck",
    "Side",
    "DeckSide",
    "HomePlanetDestination",
    "AgeGroup",
    "CabinNumBin",
)

# Numerical spend columns that benefit from log1p.  The raw values stay too.
SPEND_COLUMNS: Final[tuple[str, ...]] = (
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
)

# ---------------------------------------------------------------------------
# Baseline hyper-parameters (matches tmp/docs/benchmark_models.py so A0 is a
# faithful reproduction of the current team baseline before we improve it).
# ---------------------------------------------------------------------------
BASELINE_PARAMS: Final[dict] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 400,
    "n_jobs": 4,
    "random_state": RANDOM_SEED,
}

# Reasonable starting point once we have early stopping + our own features.
STRONG_PARAMS: Final[dict] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.8,
    "gamma": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 1.5,
    "n_estimators": 3000,
    "n_jobs": 4,
    "random_state": RANDOM_SEED,
}

EARLY_STOPPING_ROUNDS: Final[int] = 150

# ---------------------------------------------------------------------------
# Optuna search space used by tune.py.
# ---------------------------------------------------------------------------
OPTUNA_TRIALS: Final[int] = 60
OPTUNA_TIMEOUT_S: Final[int | None] = None  # None = no wall-clock cap


def describe() -> str:
    """Readable summary used by logs."""
    return (
        f"XGBoost config: seed={RANDOM_SEED}, folds={N_SPLITS}, "
        f"cat={len(CATEGORICAL_COLUMNS)}, drop={len(DROP_COLUMNS)}, "
        f"early_stopping={EARLY_STOPPING_ROUNDS}"
    )
