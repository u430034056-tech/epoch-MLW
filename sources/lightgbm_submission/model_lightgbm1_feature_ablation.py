"""Leakage-aware LightGBM ensemble training and inference for Spaceship Titanic."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from preprocess import (
    _validate_common_frame,
    apply_common_statistics,
    apply_cryosleep_spend_rule,
    basic_cleaning,
    build_group_features_single_split,
    build_group_features_with_combined_ids,
    create_age_features,
    create_group_structure_features,
    create_interaction_categorical_features,
    create_missing_indicators,
    create_spend_features,
    create_spend_structure_features,
    enforce_dtypes,
    extract_name_features,
    fill_group_consistent_categories,
    finalize_common_frame,
    fit_common_statistics,
    get_feature_sets_for_lgbm,
    get_project_paths,
    infer_missing_cryosleep,
    load_preprocessed_bundle,
    load_raw_data,
    preprocess_for_lightgbm,
    split_cabin_features,
)


MODEL_NAME = "lightgbm"
DEFAULT_RANDOM_STATE = 42
DEFAULT_ENSEMBLE_SEEDS = (42, 52, 62)
DEFAULT_THRESHOLD_GRID = tuple(np.round(np.arange(0.44, 0.541, 0.005), 3))
DEFAULT_CV_GROUP_FEATURE_MODE = "split_local"
DEFAULT_FINAL_GROUP_FEATURE_MODE = "combined"
DEFAULT_EXTRA_FEATURE_MODE = "none"
DEFAULT_TOP_FEATURE_MIN = 20
DEFAULT_TOP_FEATURE_MAX = 40
DEFAULT_FEATURE_IMPORTANCE_TYPE = "gain"
LIGHTGBM_SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LIGHTGBM_RESEARCH_NUMERIC_FEATURES = [
    "SpendMissingCount",
    "AllSpendKnown",
    "KnownSpendTotal",
    "HasKnownPositiveSpend",
    "IsZeroSpendStrict",
    "CryoSleepSpendConflict",
    "CryoSleepNoSpending",
    "CryoSleepStrictZeroSpend",
    "CryoSleepSpendCount",
    "GroupSurnameSize",
]
LIGHTGBM_RESEARCH_CATEGORICAL_FEATURES = ["GroupSurname"]
LIGHTGBM_LOG_NUMERIC_FEATURES = [
    "LogRoomService",
    "LogFoodCourt",
    "LogShoppingMall",
    "LogSpa",
    "LogVRDeck",
    "LogTotalSpend",
    "LogLuxurySpend",
    "LogBasicSpend",
    "LogSpendPerActiveCategory",
    "CabinNumLog1p",
    "SpendPerGroupMember",
    "LuxuryMinusBasicSpend",
]
LIGHTGBM_SUMMARY_NUMERIC_FEATURES = [
    "GroupSpendMean",
    "GroupSpendMax",
    "GroupAgeMean",
    "GroupZeroSpendShare",
    "GroupSpendGap",
    "GroupAgeGap",
    "GroupSpendRankPct",
    "CabinOccupancy",
    "DeckSideOccupancy",
    "DeckCabinGap",
    "HomeDestinationSpendMean",
    "HomeDestinationSpendGap",
]
LIGHTGBM_EXTRA_NUMERIC_FEATURES = LIGHTGBM_LOG_NUMERIC_FEATURES + LIGHTGBM_SUMMARY_NUMERIC_FEATURES

BASE_MODEL_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_TUNED_MODEL_PARAMS: dict[str, Any] = {
    "learning_rate": 0.03,
    "num_leaves": 23,
    "min_child_samples": 40,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.2,
    "reg_lambda": 2.5,
    "max_depth": -1,
    "min_split_gain": 0.02,
    "subsample_freq": 1,
}

DEFAULT_PARAM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "more_regularized",
        **DEFAULT_TUNED_MODEL_PARAMS,
    },
    {
        "name": "balanced",
        "learning_rate": 0.03,
        "num_leaves": 28,
        "min_child_samples": 38,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "max_depth": -1,
        "min_split_gain": 0.01,
        "subsample_freq": 1,
    },
    {
        "name": "old_best",
        "learning_rate": 0.035,
        "num_leaves": 25,
        "min_child_samples": 33,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "max_depth": -1,
        "min_split_gain": 0.02,
        "subsample_freq": 2,
    },
    {
        "name": "shallow",
        "learning_rate": 0.04,
        "num_leaves": 27,
        "min_child_samples": 45,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "max_depth": 5,
        "min_split_gain": 0.02,
        "subsample_freq": 2,
    },
    {
        "name": "wide",
        "learning_rate": 0.03,
        "num_leaves": 35,
        "min_child_samples": 25,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
        "max_depth": -1,
        "min_split_gain": 0.01,
        "subsample_freq": 1,
    },
    {
        "name": "cat_light",
        "learning_rate": 0.03,
        "num_leaves": 23,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.5,
        "max_depth": -1,
        "min_split_gain": 0.02,
        "subsample_freq": 1,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "cat_smooth": 5.0,
        "cat_l2": 1.0,
    },
    {
        "name": "cat_medium",
        "learning_rate": 0.03,
        "num_leaves": 23,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.5,
        "max_depth": -1,
        "min_split_gain": 0.02,
        "subsample_freq": 1,
        "max_bin": 255,
        "min_data_in_bin": 3,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
    },
)


def _project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the project root."""
    return Path(project_root) if project_root is not None else Path(__file__).resolve().parent


def load_raw_project_data(project_root: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw Spaceship Titanic train/test CSV files."""
    paths = get_project_paths(_project_root(project_root))
    return load_raw_data(paths["data_dir"])


def load_saved_lightgbm_bundle(
    project_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a persisted LightGBM preprocessing bundle from disk."""
    if bundle_path is not None:
        resolved_path = Path(bundle_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Preprocessed LightGBM bundle not found: {resolved_path}")
        return joblib.load(resolved_path)

    paths = get_project_paths(_project_root(project_root))
    return load_preprocessed_bundle(MODEL_NAME, processed_root=paths["processed_root"])


def _run_quietly(enabled: bool, func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a callable with stdout suppressed when requested."""
    if not enabled:
        return func(*args, **kwargs)

    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _generate_random_param_candidates(
    iterations: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Generate extra LightGBM candidates for optional random search."""
    if iterations <= 0:
        return []

    rng = np.random.default_rng(random_state)
    candidates: list[dict[str, Any]] = []
    for index in range(iterations):
        candidates.append(
            {
                "name": f"random_{index + 1}",
                "learning_rate": float(rng.choice([0.02, 0.025, 0.03, 0.035, 0.04])),
                "num_leaves": int(rng.integers(20, 42)),
                "min_child_samples": int(rng.integers(20, 56)),
                "subsample": float(rng.choice([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])),
                "colsample_bytree": float(rng.choice([0.65, 0.7, 0.75, 0.8, 0.85, 0.9])),
                "reg_alpha": float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.4, 0.8])),
                "reg_lambda": float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])),
                "max_depth": int(rng.choice([-1, 4, 5, 6])),
                "min_split_gain": float(rng.choice([0.0, 0.01, 0.02, 0.05])),
                "subsample_freq": int(rng.choice([0, 1, 2])),
                "max_bin": int(rng.choice([127, 191, 255, 383])),
                "min_data_in_bin": int(rng.choice([1, 3, 5, 10])),
                "cat_smooth": float(rng.choice([0.0, 5.0, 10.0, 20.0, 40.0, 80.0])),
                "cat_l2": float(rng.choice([1.0, 5.0, 10.0, 20.0, 40.0])),
            }
        )
    return candidates


def _iter_thresholds(threshold_values: Sequence[float] | None = None) -> tuple[float, ...]:
    """Return the threshold grid."""
    if threshold_values is None:
        return DEFAULT_THRESHOLD_GRID
    return tuple(float(value) for value in threshold_values)


def _find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold_values: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Find the classification threshold that maximizes accuracy."""
    best_threshold = 0.5
    best_score = float(accuracy_score(y_true, probabilities >= 0.5))

    for threshold in _iter_thresholds(threshold_values):
        score = float(accuracy_score(y_true, probabilities >= threshold))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score

    return best_threshold, best_score


def _safe_log1p(series: pd.Series) -> pd.Series:
    """Apply a stable log1p transform to a non-negative numeric series."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return np.log1p(numeric.clip(lower=0.0))


def _spend_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric spend columns while preserving missingness for rule features."""
    return df[LIGHTGBM_SPEND_COLUMNS].apply(pd.to_numeric, errors="coerce")


def _add_raw_spend_rule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Capture strict zero-spend and CryoSleep/spend conflicts before imputation."""
    updated = df.copy()
    spend_values = _spend_values(updated)
    spend_missing_count = spend_values.isna().sum(axis=1)
    known_spend_total = spend_values.fillna(0.0).sum(axis=1)
    has_known_positive_spend = spend_values.gt(0).any(axis=1)
    all_spend_known = spend_missing_count.eq(0)
    cryo_true = updated["CryoSleep"].eq("True").fillna(False)

    updated["SpendMissingCount"] = spend_missing_count.astype(int)
    updated["AllSpendKnown"] = all_spend_known.astype(int)
    updated["KnownSpendTotal"] = known_spend_total.astype(float)
    updated["HasKnownPositiveSpend"] = has_known_positive_spend.astype(int)
    updated["IsZeroSpendStrict"] = (all_spend_known & known_spend_total.eq(0)).astype(int)
    updated["CryoSleepSpendConflict"] = (cryo_true & has_known_positive_spend).astype(int)
    return updated


def _add_cryosleep_spend_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add LightGBM numeric interactions from CryoSleep and spend structure."""
    updated = df.copy()
    cryo_true = updated["CryoSleep"].eq("True").fillna(False).astype(int)
    updated["CryoSleepNoSpending"] = (cryo_true & updated["IsZeroSpend"].astype(int)).astype(int)
    updated["CryoSleepStrictZeroSpend"] = (
        cryo_true & updated["IsZeroSpendStrict"].astype(int)
    ).astype(int)
    updated["CryoSleepSpendCount"] = (cryo_true * updated["SpendCount"].astype(int)).astype(int)
    return updated


def _add_group_surname_features_single_split(df: pd.DataFrame) -> pd.DataFrame:
    """Build surname-within-group features without using target information."""
    updated = df.copy()
    group_values = updated["GroupID"].astype("string").fillna("UnknownGroup")
    surname_values = updated["Surname"].astype("string").fillna("Unknown")
    updated["GroupSurname"] = (group_values + "_" + surname_values).astype("string")
    group_surname_sizes = updated["GroupSurname"].value_counts(dropna=False).to_dict()
    updated["GroupSurnameSize"] = updated["GroupSurname"].map(group_surname_sizes).fillna(1).astype(int)
    return updated


def _add_group_surname_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_feature_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build group/surname features with the same scope as the GroupSize strategy."""
    if group_feature_mode == "combined":
        combined = pd.concat(
            [
                train_df.assign(_split="train"),
                test_df.assign(_split="test"),
            ],
            axis=0,
            ignore_index=False,
        )
        combined = _add_group_surname_features_single_split(combined)
        train_extended = combined.loc[combined["_split"] == "train"].drop(columns=["_split"]).copy()
        test_extended = combined.loc[combined["_split"] == "test"].drop(columns=["_split"]).copy()
        return train_extended, test_extended
    if group_feature_mode == "split_local":
        return (
            _add_group_surname_features_single_split(train_df),
            _add_group_surname_features_single_split(test_df),
        )
    raise ValueError(f"Unsupported group_feature_mode: {group_feature_mode}")


def _add_split_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build LightGBM-only numeric summary features inside one split."""
    updated = df.copy()

    updated["GroupSpendMean"] = updated.groupby("GroupID", dropna=False)["TotalSpend"].transform("mean")
    updated["GroupSpendMax"] = updated.groupby("GroupID", dropna=False)["TotalSpend"].transform("max")
    updated["GroupAgeMean"] = updated.groupby("GroupID", dropna=False)["Age"].transform("mean")
    updated["GroupZeroSpendShare"] = updated.groupby("GroupID", dropna=False)["IsZeroSpend"].transform("mean")
    updated["GroupSpendRankPct"] = (
        updated.groupby("GroupID", dropna=False)["TotalSpend"].rank(method="average", pct=True).fillna(0.5)
    )

    updated["DeckSideOccupancy"] = (
        updated.groupby("DeckSide", dropna=False)["PassengerId"].transform("size").astype(float)
    )

    known_cabin_mask = updated["CabinMissing"].eq(0)
    updated["CabinOccupancy"] = 0.0
    if known_cabin_mask.any():
        updated.loc[known_cabin_mask, "CabinOccupancy"] = (
            updated.loc[known_cabin_mask].groupby("Cabin", dropna=False)["PassengerId"].transform("size").astype(float)
        )

    deck_cabin_median = updated.groupby("Deck", dropna=False)["CabinNum"].transform("median")
    updated["DeckCabinGap"] = updated["CabinNum"] - deck_cabin_median

    updated["HomeDestinationSpendMean"] = (
        updated.groupby("HomePlanetDestination", dropna=False)["TotalSpend"].transform("mean")
    )
    updated["HomeDestinationSpendGap"] = updated["TotalSpend"] - updated["HomeDestinationSpendMean"]
    updated["GroupSpendGap"] = updated["TotalSpend"] - updated["GroupSpendMean"]
    updated["GroupAgeGap"] = updated["Age"] - updated["GroupAgeMean"]

    return updated


def _add_lightgbm_only_features(
    common_train: pd.DataFrame,
    common_test: pd.DataFrame,
    summary_scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add LightGBM-specific numeric features on top of the shared preprocessing output."""
    train_extended = common_train.copy()
    test_extended = common_test.copy()

    for current_df in (train_extended, test_extended):
        current_df["LogRoomService"] = _safe_log1p(current_df["RoomService"])
        current_df["LogFoodCourt"] = _safe_log1p(current_df["FoodCourt"])
        current_df["LogShoppingMall"] = _safe_log1p(current_df["ShoppingMall"])
        current_df["LogSpa"] = _safe_log1p(current_df["Spa"])
        current_df["LogVRDeck"] = _safe_log1p(current_df["VRDeck"])
        current_df["LogTotalSpend"] = _safe_log1p(current_df["TotalSpend"])
        current_df["LogLuxurySpend"] = _safe_log1p(current_df["LuxurySpend"])
        current_df["LogBasicSpend"] = _safe_log1p(current_df["BasicSpend"])
        current_df["LogSpendPerActiveCategory"] = _safe_log1p(current_df["SpendPerActiveCategory"])
        current_df["CabinNumLog1p"] = _safe_log1p(current_df["CabinNum"])
        current_df["SpendPerGroupMember"] = (
            current_df["TotalSpend"] / current_df["GroupSize"].replace(0, np.nan)
        ).fillna(0.0)
        current_df["LuxuryMinusBasicSpend"] = current_df["LuxurySpend"] - current_df["BasicSpend"]

    if summary_scope == "combined":
        combined = pd.concat(
            [
                train_extended.assign(_split="train"),
                test_extended.assign(_split="test"),
            ],
            axis=0,
            ignore_index=False,
        )
        combined = _add_split_summary_features(combined)
        train_extended = combined.loc[combined["_split"] == "train"].drop(columns=["_split"]).copy()
        test_extended = combined.loc[combined["_split"] == "test"].drop(columns=["_split"]).copy()
    elif summary_scope == "split_local":
        train_extended = _add_split_summary_features(train_extended)
        test_extended = _add_split_summary_features(test_extended)
    else:
        raise ValueError(f"Unsupported summary_scope: {summary_scope}")

    for current_df in (train_extended, test_extended):
        for column in LIGHTGBM_EXTRA_NUMERIC_FEATURES:
            current_df[column] = pd.to_numeric(current_df[column], errors="coerce").fillna(0.0)

    return train_extended, test_extended


def _selected_extra_numeric_features(extra_feature_mode: str) -> list[str]:
    """Return the selected LightGBM-only feature subset."""
    if extra_feature_mode == "none":
        return []
    if extra_feature_mode == "logs_only":
        return LIGHTGBM_LOG_NUMERIC_FEATURES.copy()
    if extra_feature_mode == "summary_only":
        return LIGHTGBM_SUMMARY_NUMERIC_FEATURES.copy()
    if extra_feature_mode == "all":
        return LIGHTGBM_EXTRA_NUMERIC_FEATURES.copy()
    raise ValueError(f"Unsupported extra_feature_mode: {extra_feature_mode}")


def _build_lightgbm_feature_set(
    common_train: pd.DataFrame,
    common_test: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
    summary_scope: str,
    extra_feature_mode: str,
) -> dict[str, Any]:
    """Build the final LightGBM feature set including LightGBM-only features."""
    extended_train, extended_test = _add_lightgbm_only_features(
        common_train,
        common_test,
        summary_scope=summary_scope,
    )
    selected_extra_numeric_features = _selected_extra_numeric_features(extra_feature_mode)
    feature_set = get_feature_sets_for_lgbm(
        extended_train,
        extended_test,
        train_ids,
        test_ids,
    )
    feature_set["numeric_features"] = (
        feature_set["numeric_features"]
        + LIGHTGBM_RESEARCH_NUMERIC_FEATURES
        + selected_extra_numeric_features
    )
    feature_set["categorical_features"] = (
        feature_set["categorical_features"] + LIGHTGBM_RESEARCH_CATEGORICAL_FEATURES
    )
    selected_columns = feature_set["numeric_features"] + feature_set["categorical_features"]
    feature_set["train_df"] = extended_train[selected_columns].copy()
    feature_set["test_df"] = extended_test[selected_columns].copy()
    feature_set["extra_feature_mode"] = extra_feature_mode
    return feature_set


def _apply_group_feature_mode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_feature_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build group-size features using the selected strategy."""
    if group_feature_mode == "combined":
        return build_group_features_with_combined_ids(train_df, test_df)
    if group_feature_mode == "split_local":
        return build_group_features_single_split(train_df), build_group_features_single_split(test_df)
    raise ValueError(f"Unsupported group_feature_mode: {group_feature_mode}")


def _build_lightgbm_bundle_from_raw(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_feature_mode: str,
    extra_feature_mode: str,
    selected_features: Sequence[str] | None = None,
    quiet: bool = True,
) -> dict[str, Any]:
    """Preprocess raw data into a LightGBM-ready bundle."""

    def _build() -> dict[str, Any]:
        y_train = train_df["Transported"].astype(int).copy()
        train_ids = train_df["PassengerId"].astype("string").copy()
        test_ids = test_df["PassengerId"].astype("string").copy()

        train_working = enforce_dtypes(basic_cleaning(train_df.drop(columns=["Transported"])))
        test_source = test_df.drop(columns=["Transported"]) if "Transported" in test_df.columns else test_df
        test_working = enforce_dtypes(basic_cleaning(test_source))

        train_working, test_working = _apply_group_feature_mode(
            train_working,
            test_working,
            group_feature_mode=group_feature_mode,
        )

        train_working = split_cabin_features(train_working)
        test_working = split_cabin_features(test_working)
        train_working = extract_name_features(train_working)
        test_working = extract_name_features(test_working)
        train_working, test_working = _add_group_surname_features(
            train_working,
            test_working,
            group_feature_mode=group_feature_mode,
        )

        train_working = create_missing_indicators(train_working)
        test_working = create_missing_indicators(test_working)
        train_working = _add_raw_spend_rule_features(train_working)
        test_working = _add_raw_spend_rule_features(test_working)

        group_fill_columns = ["HomePlanet", "VIP", "Destination"]
        train_working, _ = fill_group_consistent_categories(train_working, group_fill_columns)
        test_working, _ = fill_group_consistent_categories(test_working, group_fill_columns)

        train_working, _ = infer_missing_cryosleep(train_working)
        test_working, _ = infer_missing_cryosleep(test_working)
        train_working = apply_cryosleep_spend_rule(train_working)
        test_working = apply_cryosleep_spend_rule(test_working)

        common_stats = fit_common_statistics(train_working)
        train_working = apply_common_statistics(train_working, common_stats)
        test_working = apply_common_statistics(test_working, common_stats)

        train_working = create_spend_features(train_working)
        test_working = create_spend_features(test_working)
        train_working = _add_cryosleep_spend_interaction_features(train_working)
        test_working = _add_cryosleep_spend_interaction_features(test_working)
        train_working = create_age_features(train_working)
        test_working = create_age_features(test_working)
        train_working = create_spend_structure_features(train_working)
        test_working = create_spend_structure_features(test_working)
        train_working = create_group_structure_features(train_working)
        test_working = create_group_structure_features(test_working)
        train_working = create_interaction_categorical_features(train_working)
        test_working = create_interaction_categorical_features(test_working)

        train_working = finalize_common_frame(train_working, "train")
        test_working = finalize_common_frame(test_working, "test")
        _validate_common_frame(train_working, "lightgbm_train_final")
        _validate_common_frame(test_working, "lightgbm_test_final")

        feature_set = _build_lightgbm_feature_set(
            train_working,
            test_working,
            train_ids,
            test_ids,
            summary_scope=group_feature_mode,
            extra_feature_mode=extra_feature_mode,
        )
        model_bundle = preprocess_for_lightgbm(feature_set, y_train)
        if selected_features is not None:
            model_bundle = _subset_lightgbm_bundle(model_bundle, selected_features)
        model_bundle["common_stats"] = common_stats
        model_bundle["group_feature_mode"] = group_feature_mode
        model_bundle["extra_feature_mode"] = extra_feature_mode
        return model_bundle

    return _run_quietly(quiet, _build)


def _build_seeded_params(model_params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build a deterministic parameter set for one ensemble member."""
    params = dict(BASE_MODEL_PARAMS)
    params.update(model_params)
    params.update(
        {
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        }
    )
    return params


def _cross_validate_from_raw(
    train_df: pd.DataFrame,
    model_params: dict[str, Any],
    ensemble_seeds: Sequence[int],
    n_splits: int,
    early_stopping_rounds: int,
    threshold_values: Sequence[float] | None = None,
    group_feature_mode: str = DEFAULT_CV_GROUP_FEATURE_MODE,
    extra_feature_mode: str = DEFAULT_EXTRA_FEATURE_MODE,
    selected_features: Sequence[str] | None = None,
    quiet_preprocessing: bool = True,
) -> dict[str, Any]:
    """Run leakage-aware cross-validation with fold-local preprocessing."""
    y_all = train_df["Transported"].astype(int).copy()
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )

    oof_probabilities = np.zeros(len(train_df), dtype=float)
    fold_summaries: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_index, (fit_index, valid_index) in enumerate(splitter.split(train_df, y_all), start=1):
        fit_df = train_df.iloc[fit_index].copy()
        valid_df = train_df.iloc[valid_index].copy()
        bundle = _build_lightgbm_bundle_from_raw(
            fit_df,
            valid_df,
            group_feature_mode=group_feature_mode,
            extra_feature_mode=extra_feature_mode,
            selected_features=selected_features,
            quiet=quiet_preprocessing,
        )

        X_fit = bundle["X_train_lgbm"]
        X_valid = bundle["X_test_lgbm"]
        y_fit = bundle["y_train_lgbm"]
        y_valid = valid_df["Transported"].astype(int).copy()
        categorical_features = bundle["categorical_feature_names_lgbm"]

        fold_probabilities = np.zeros(len(valid_df), dtype=float)
        seed_summaries: list[dict[str, Any]] = []

        for seed in ensemble_seeds:
            model = LGBMClassifier(**_build_seeded_params(model_params, seed))
            model.fit(
                X_fit,
                y_fit,
                categorical_feature=categorical_features,
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(0),
                ],
            )

            valid_probabilities = model.predict_proba(X_valid)[:, 1]
            fold_probabilities += valid_probabilities / len(ensemble_seeds)

            best_iteration = int(model.best_iteration_ or model.n_estimators)
            best_iterations.append(best_iteration)
            seed_summaries.append(
                {
                    "seed": int(seed),
                    "best_iteration": best_iteration,
                    "validation_accuracy_at_0_5": float(
                        accuracy_score(y_valid, valid_probabilities >= 0.5)
                    ),
                }
            )

        oof_probabilities[valid_index] = fold_probabilities
        fold_summaries.append(
            {
                "fold": fold_index,
                "group_feature_mode": group_feature_mode,
                "extra_feature_mode": extra_feature_mode,
                "validation_accuracy_at_0_5": float(
                    accuracy_score(y_valid, fold_probabilities >= 0.5)
                ),
                "seed_summaries": seed_summaries,
            }
        )

    best_threshold, tuned_accuracy = _find_best_threshold(
        y_all,
        oof_probabilities,
        threshold_values=threshold_values,
    )

    return {
        "oof_probabilities": oof_probabilities,
        "cv_accuracy_at_0_5": float(accuracy_score(y_all, oof_probabilities >= 0.5)),
        "cv_accuracy": float(tuned_accuracy),
        "best_threshold": float(best_threshold),
        "best_iteration_mean": float(np.mean(best_iterations)) if best_iterations else None,
        "fold_summaries": fold_summaries,
    }


def _cross_validate_from_preprocessed_bundle(
    bundle: dict[str, Any],
    model_params: dict[str, Any],
    ensemble_seeds: Sequence[int],
    n_splits: int,
    early_stopping_rounds: int,
    threshold_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run feature-space cross-validation directly from a saved LightGBM preprocessing bundle."""
    X_all, y_all, _, categorical_features, _, _ = _get_bundle_views(bundle)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=DEFAULT_RANDOM_STATE,
    )

    oof_probabilities = np.zeros(len(X_all), dtype=float)
    fold_summaries: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_index, (fit_index, valid_index) in enumerate(splitter.split(X_all, y_all), start=1):
        X_fit = X_all.iloc[fit_index].copy()
        X_valid = X_all.iloc[valid_index].copy()
        y_fit = y_all.iloc[fit_index].copy()
        y_valid = y_all.iloc[valid_index].copy()

        fold_probabilities = np.zeros(len(valid_index), dtype=float)
        seed_summaries: list[dict[str, Any]] = []

        for seed in ensemble_seeds:
            model = LGBMClassifier(**_build_seeded_params(model_params, seed))
            model.fit(
                X_fit,
                y_fit,
                categorical_feature=list(categorical_features),
                eval_set=[(X_valid, y_valid)],
                eval_metric="binary_logloss",
                callbacks=[
                    early_stopping(early_stopping_rounds, verbose=False),
                    log_evaluation(0),
                ],
            )

            valid_probabilities = model.predict_proba(X_valid)[:, 1]
            fold_probabilities += valid_probabilities / len(ensemble_seeds)

            best_iteration = int(model.best_iteration_ or model.n_estimators)
            best_iterations.append(best_iteration)
            seed_summaries.append(
                {
                    "seed": int(seed),
                    "best_iteration": best_iteration,
                    "validation_accuracy_at_0_5": float(
                        accuracy_score(y_valid, valid_probabilities >= 0.5)
                    ),
                }
            )

        oof_probabilities[valid_index] = fold_probabilities
        fold_summaries.append(
            {
                "fold": fold_index,
                "cv_source": "preprocessed_bundle",
                "validation_accuracy_at_0_5": float(
                    accuracy_score(y_valid, fold_probabilities >= 0.5)
                ),
                "seed_summaries": seed_summaries,
            }
        )

    best_threshold, tuned_accuracy = _find_best_threshold(
        y_all,
        oof_probabilities,
        threshold_values=threshold_values,
    )

    return {
        "oof_probabilities": oof_probabilities,
        "cv_accuracy_at_0_5": float(accuracy_score(y_all, oof_probabilities >= 0.5)),
        "cv_accuracy": float(tuned_accuracy),
        "best_threshold": float(best_threshold),
        "best_iteration_mean": float(np.mean(best_iterations)) if best_iterations else None,
        "fold_summaries": fold_summaries,
    }


def tune_model_params(
    project_root: str | Path | None = None,
    param_candidates: Sequence[dict[str, Any]] | None = None,
    random_search_iterations: int = 0,
    ensemble_seeds: Sequence[int] = DEFAULT_ENSEMBLE_SEEDS,
    n_splits: int = 5,
    early_stopping_rounds: int = 250,
    threshold_values: Sequence[float] | None = None,
    group_feature_mode: str = DEFAULT_CV_GROUP_FEATURE_MODE,
    extra_feature_mode: str = DEFAULT_EXTRA_FEATURE_MODE,
    selected_features: Sequence[str] | None = None,
    quiet_preprocessing: bool = True,
) -> dict[str, Any]:
    """Evaluate LightGBM candidates with fold-local preprocessing and return the best one."""
    train_df, _ = load_raw_project_data(project_root=project_root)
    candidate_pool = list(param_candidates or DEFAULT_PARAM_CANDIDATES)
    candidate_pool.extend(_generate_random_param_candidates(random_search_iterations))

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, raw_candidate in enumerate(candidate_pool, start=1):
        candidate = dict(raw_candidate)
        candidate_name = str(candidate.pop("name", f"candidate_{candidate_index}"))
        evaluation = _cross_validate_from_raw(
            train_df=train_df,
            model_params=candidate,
            ensemble_seeds=ensemble_seeds,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            threshold_values=threshold_values,
            group_feature_mode=group_feature_mode,
            extra_feature_mode=extra_feature_mode,
            selected_features=selected_features,
            quiet_preprocessing=quiet_preprocessing,
        )
        candidate_results.append(
            {
                "name": candidate_name,
                "params": candidate,
                "extra_feature_mode": extra_feature_mode,
                "cv_accuracy": evaluation["cv_accuracy"],
                "cv_accuracy_at_0_5": evaluation["cv_accuracy_at_0_5"],
                "best_threshold": evaluation["best_threshold"],
                "best_iteration_mean": evaluation["best_iteration_mean"],
            }
        )

    best_candidate = max(
        candidate_results,
        key=lambda item: (item["cv_accuracy"], item["cv_accuracy_at_0_5"]),
    )
    return {
        "best_name": best_candidate["name"],
        "best_params": best_candidate["params"],
        "best_threshold": best_candidate["best_threshold"],
        "candidate_results": candidate_results,
    }


def _tune_model_params_from_preprocessed_bundle(
    bundle: dict[str, Any],
    param_candidates: Sequence[dict[str, Any]] | None = None,
    random_search_iterations: int = 0,
    ensemble_seeds: Sequence[int] = DEFAULT_ENSEMBLE_SEEDS,
    n_splits: int = 5,
    early_stopping_rounds: int = 250,
    threshold_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate LightGBM candidates on a saved preprocessing bundle and return the best one."""
    candidate_pool = list(param_candidates or DEFAULT_PARAM_CANDIDATES)
    candidate_pool.extend(_generate_random_param_candidates(random_search_iterations))

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, raw_candidate in enumerate(candidate_pool, start=1):
        candidate = dict(raw_candidate)
        candidate_name = str(candidate.pop("name", f"candidate_{candidate_index}"))
        evaluation = _cross_validate_from_preprocessed_bundle(
            bundle=bundle,
            model_params=candidate,
            ensemble_seeds=ensemble_seeds,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            threshold_values=threshold_values,
        )
        candidate_results.append(
            {
                "name": candidate_name,
                "params": candidate,
                "cv_accuracy": evaluation["cv_accuracy"],
                "cv_accuracy_at_0_5": evaluation["cv_accuracy_at_0_5"],
                "best_threshold": evaluation["best_threshold"],
                "best_iteration_mean": evaluation["best_iteration_mean"],
            }
        )

    best_candidate = max(
        candidate_results,
        key=lambda item: (item["cv_accuracy"], item["cv_accuracy_at_0_5"]),
    )
    return {
        "best_name": best_candidate["name"],
        "best_params": best_candidate["params"],
        "best_threshold": best_candidate["best_threshold"],
        "candidate_results": candidate_results,
    }


def _fit_final_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: Sequence[str],
    model_params: dict[str, Any],
    ensemble_seeds: Sequence[int],
    final_n_estimators: int,
) -> tuple[list[LGBMClassifier], np.ndarray]:
    """Fit the final LightGBM ensemble on the full training data."""
    models: list[LGBMClassifier] = []
    test_probabilities = np.zeros(len(X_test), dtype=float)

    final_params = dict(model_params)
    final_params["n_estimators"] = int(final_n_estimators)

    for seed in ensemble_seeds:
        model = LGBMClassifier(**_build_seeded_params(final_params, seed))
        model.fit(
            X_train,
            y_train,
            categorical_feature=list(categorical_features),
        )
        models.append(model)
        test_probabilities += model.predict_proba(X_test)[:, 1] / len(ensemble_seeds)

    return models, test_probabilities


def _aggregate_feature_importance(
    models: Sequence[LGBMClassifier],
    feature_names: Sequence[str],
    importance_type: str = DEFAULT_FEATURE_IMPORTANCE_TYPE,
) -> pd.DataFrame:
    """Aggregate LightGBM feature importance across an ensemble."""
    if not models:
        raise ValueError("At least one trained model is required to rank features.")

    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        booster = getattr(model, "booster_", None)
        if booster is None:
            importance_values = np.asarray(model.feature_importances_, dtype=float)
        else:
            importance_values = np.asarray(
                booster.feature_importance(importance_type=importance_type),
                dtype=float,
            )
        if len(importance_values) != len(feature_names):
            raise ValueError(
                "Feature importance length does not match feature_names length: "
                f"{len(importance_values)} != {len(feature_names)}"
            )

        for feature_name, importance in zip(feature_names, importance_values):
            rows.append(
                {
                    "model_index": model_index,
                    "feature": str(feature_name),
                    "importance": float(importance),
                }
            )

    importance_frame = pd.DataFrame(rows)
    summary = (
        importance_frame.groupby("feature", as_index=False)["importance"]
        .agg(["mean", "std", "sum"])
        .reset_index()
        .rename(
            columns={
                "mean": "importance_mean",
                "std": "importance_std",
                "sum": "importance_sum",
            }
        )
    )
    summary["importance_std"] = summary["importance_std"].fillna(0.0)
    order_lookup = {str(feature_name): index for index, feature_name in enumerate(feature_names)}
    summary["original_order"] = summary["feature"].map(order_lookup).astype(int)
    return summary.sort_values(
        ["importance_mean", "importance_sum", "original_order"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _select_top_feature_names(
    importance_frame: pd.DataFrame,
    min_features: int = DEFAULT_TOP_FEATURE_MIN,
    max_features: int = DEFAULT_TOP_FEATURE_MAX,
    selected_feature_count: int | None = None,
) -> list[str]:
    """Select a bounded top-feature list from an importance summary."""
    if importance_frame.empty:
        raise ValueError("Cannot select top features from an empty importance frame.")
    if min_features <= 0 or max_features <= 0:
        raise ValueError("min_features and max_features must be positive.")
    if min_features > max_features:
        raise ValueError("min_features cannot be greater than max_features.")

    total_features = len(importance_frame)
    if selected_feature_count is None:
        nonzero_count = int(importance_frame["importance_mean"].gt(0).sum())
        target_count = nonzero_count if nonzero_count > 0 else total_features
        target_count = min(max_features, max(min_features, target_count))
    else:
        target_count = int(selected_feature_count)
        if target_count < min_features or target_count > max_features:
            raise ValueError(
                "selected_feature_count must be between "
                f"{min_features} and {max_features}, got {target_count}."
            )

    target_count = min(target_count, total_features)
    return importance_frame.head(target_count)["feature"].astype(str).tolist()


def _subset_lightgbm_bundle(bundle: dict[str, Any], selected_features: Sequence[str]) -> dict[str, Any]:
    """Return a LightGBM preprocessing bundle restricted to selected feature columns."""
    selected_feature_list = [str(feature) for feature in selected_features]
    if not selected_feature_list:
        raise ValueError("selected_features cannot be empty.")

    X_train, _, X_test, categorical_features, _, _ = _get_bundle_views(bundle)
    missing_features = [
        feature for feature in selected_feature_list if feature not in X_train.columns or feature not in X_test.columns
    ]
    if missing_features:
        raise ValueError(f"Selected features are missing from the LightGBM bundle: {missing_features}")

    restricted = dict(bundle)
    X_train_subset = X_train.loc[:, selected_feature_list].copy()
    X_test_subset = X_test.loc[:, selected_feature_list].copy()
    categorical_subset = [feature for feature in categorical_features if feature in selected_feature_list]

    restricted["X_train"] = X_train_subset
    restricted["X_test"] = X_test_subset
    restricted["X_train_lgbm"] = X_train_subset
    restricted["X_test_lgbm"] = X_test_subset
    restricted["feature_names"] = selected_feature_list
    restricted["feature_names_lgbm"] = selected_feature_list
    restricted["categorical_feature_names"] = categorical_subset
    restricted["categorical_feature_names_lgbm"] = categorical_subset
    restricted["selected_features"] = selected_feature_list
    restricted["selected_feature_count"] = len(selected_feature_list)
    return restricted


def _get_bundle_views(
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], pd.Series | None, pd.Series | None]:
    """Extract the standard LightGBM bundle views."""
    X_train = bundle.get("X_train_lgbm", bundle.get("X_train"))
    X_test = bundle.get("X_test_lgbm", bundle.get("X_test"))
    y_train = bundle.get("y_train_lgbm", bundle.get("y_train"))
    categorical_features = bundle.get(
        "categorical_feature_names_lgbm",
        bundle.get("categorical_feature_names", []),
    )
    train_ids = bundle.get("train_ids")
    test_ids = bundle.get("test_ids")

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("Expected X_train and X_test to be pandas DataFrames for LightGBM.")
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name="Transported")

    return X_train.copy(), y_train.copy(), X_test.copy(), list(categorical_features), train_ids, test_ids


def _collect_category_levels(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: Sequence[str],
) -> dict[str, list[Any]]:
    """Collect stable category levels from the final train/test bundle."""
    levels: dict[str, list[Any]] = {}
    for column in categorical_features:
        shared_categories = pd.Index(X_train[column].cat.categories).union(X_test[column].cat.categories)
        levels[column] = shared_categories.tolist()
    return levels


def _prepare_prediction_frame(model_artifact: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Align an input frame to the final ensemble feature space."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")

    feature_names = list(model_artifact["feature_names"])
    missing_columns = [column for column in feature_names if column not in X.columns]
    if missing_columns:
        raise ValueError(f"Prediction input is missing required columns: {missing_columns}")

    prepared = X.loc[:, feature_names].copy()
    for column, levels in model_artifact["category_levels"].items():
        prepared[column] = prepared[column].astype("category").cat.set_categories(levels)
    return prepared


def train_model(
    project_root: str | Path | None = None,
    model_params: dict[str, Any] | None = None,
    optimize_params: bool = False,
    param_candidates: Sequence[dict[str, Any]] | None = None,
    random_search_iterations: int = 0,
    ensemble_seeds: Sequence[int] = DEFAULT_ENSEMBLE_SEEDS,
    n_splits: int = 5,
    early_stopping_rounds: int = 250,
    threshold_values: Sequence[float] | None = None,
    cv_group_feature_mode: str = DEFAULT_CV_GROUP_FEATURE_MODE,
    final_group_feature_mode: str = DEFAULT_FINAL_GROUP_FEATURE_MODE,
    extra_feature_mode: str = DEFAULT_EXTRA_FEATURE_MODE,
    quiet_preprocessing: bool = True,
    use_preprocessed_bundle: bool = False,
    preprocessed_bundle_path: str | Path | None = None,
    selected_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Train a LightGBM ensemble.

    When ``use_preprocessed_bundle`` is False, cross-validation uses fold-local
    preprocessing to avoid optimistic scores and the final refit rebuilds the
    selected feature pipeline from raw CSV files.

    When ``use_preprocessed_bundle`` is True, the function loads
    ``processed/lightgbm/preprocessed_lightgbm.joblib`` (or an explicitly
    provided bundle path) and trains directly on that saved feature space.
    """
    using_saved_bundle = bool(use_preprocessed_bundle or preprocessed_bundle_path is not None)
    preprocessed_bundle: dict[str, Any] | None = None

    if using_saved_bundle:
        if extra_feature_mode != "none":
            raise ValueError(
                "Saved LightGBM preprocessing bundles currently support only extra_feature_mode='none'."
            )
        preprocessed_bundle = load_saved_lightgbm_bundle(
            project_root=project_root,
            bundle_path=preprocessed_bundle_path,
        )
        if not bool(preprocessed_bundle.get("input_validation_passed", True)):
            raise ValueError("Loaded preprocessed LightGBM bundle did not pass input validation.")
        if selected_features is not None:
            preprocessed_bundle = _subset_lightgbm_bundle(preprocessed_bundle, selected_features)
        train_df = None
        test_df = None
    else:
        train_df, test_df = load_raw_project_data(project_root=project_root)

    search_result: dict[str, Any] | None = None
    if optimize_params:
        if using_saved_bundle:
            assert preprocessed_bundle is not None
            search_result = _tune_model_params_from_preprocessed_bundle(
                bundle=preprocessed_bundle,
                param_candidates=param_candidates,
                random_search_iterations=random_search_iterations,
                ensemble_seeds=ensemble_seeds,
                n_splits=n_splits,
                early_stopping_rounds=early_stopping_rounds,
                threshold_values=threshold_values,
            )
        else:
            assert train_df is not None
            search_result = tune_model_params(
                project_root=project_root,
                param_candidates=param_candidates,
                random_search_iterations=random_search_iterations,
                ensemble_seeds=ensemble_seeds,
                n_splits=n_splits,
                early_stopping_rounds=early_stopping_rounds,
                threshold_values=threshold_values,
                group_feature_mode=cv_group_feature_mode,
                extra_feature_mode=extra_feature_mode,
                selected_features=selected_features,
                quiet_preprocessing=quiet_preprocessing,
            )
        selected_params = dict(search_result["best_params"])
    else:
        selected_params = dict(DEFAULT_TUNED_MODEL_PARAMS)

    if model_params is not None:
        selected_params.update(model_params)

    if using_saved_bundle:
        assert preprocessed_bundle is not None
        cv_result = _cross_validate_from_preprocessed_bundle(
            bundle=preprocessed_bundle,
            model_params=selected_params,
            ensemble_seeds=ensemble_seeds,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            threshold_values=threshold_values,
        )
        final_bundle = preprocessed_bundle
        cv_group_feature_mode_used = "preprocessed_bundle"
        final_group_feature_mode_used = "preprocessed_bundle"
        extra_feature_mode_used = final_bundle.get("extra_feature_mode", "none")
        training_mode = "preprocessed_bundle_cv_plus_full_refit"
    else:
        assert train_df is not None
        assert test_df is not None
        cv_result = _cross_validate_from_raw(
            train_df=train_df,
            model_params=selected_params,
            ensemble_seeds=ensemble_seeds,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            threshold_values=threshold_values,
            group_feature_mode=cv_group_feature_mode,
            extra_feature_mode=extra_feature_mode,
            selected_features=selected_features,
            quiet_preprocessing=quiet_preprocessing,
        )

        final_bundle = _build_lightgbm_bundle_from_raw(
            train_df=train_df,
            test_df=test_df,
            group_feature_mode=final_group_feature_mode,
            extra_feature_mode=extra_feature_mode,
            selected_features=selected_features,
            quiet=False,
        )
        cv_group_feature_mode_used = cv_group_feature_mode
        final_group_feature_mode_used = final_group_feature_mode
        extra_feature_mode_used = extra_feature_mode
        training_mode = "fold_local_cv_plus_full_refit"

    X_train, y_train, X_test, categorical_features, train_ids, test_ids = _get_bundle_views(final_bundle)
    category_levels = _collect_category_levels(X_train, X_test, categorical_features)

    final_n_estimators = max(int(round(cv_result["best_iteration_mean"] or 300)), 100)
    final_models, test_probabilities = _fit_final_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        categorical_features=categorical_features,
        model_params=selected_params,
        ensemble_seeds=ensemble_seeds,
        final_n_estimators=final_n_estimators,
    )

    return {
        "model_name": "lightgbm1_leakage_aware_ensemble",
        "training_mode": training_mode,
        "models": final_models,
        "best_params": selected_params,
        "cv_accuracy": cv_result["cv_accuracy"],
        "cv_accuracy_at_0_5": cv_result["cv_accuracy_at_0_5"],
        "threshold": cv_result["best_threshold"],
        "best_iteration_mean": cv_result["best_iteration_mean"],
        "final_n_estimators": final_n_estimators,
        "feature_names": X_train.columns.tolist(),
        "categorical_feature_names": list(categorical_features),
        "category_levels": category_levels,
        "ensemble_seeds": list(int(seed) for seed in ensemble_seeds),
        "n_splits": int(n_splits),
        "cv_group_feature_mode": cv_group_feature_mode_used,
        "final_group_feature_mode": final_group_feature_mode_used,
        "extra_feature_mode": extra_feature_mode_used,
        "bundle_source": "saved_preprocessed_bundle" if using_saved_bundle else "raw_rebuild",
        "preprocessed_bundle_path": str(final_bundle.get("save_path")) if using_saved_bundle else None,
        "selected_features": X_train.columns.tolist() if selected_features is not None else None,
        "feature_count": int(X_train.shape[1]),
        "oof_probabilities": cv_result["oof_probabilities"],
        "test_probabilities": test_probabilities,
        "X_test": X_test,
        "test_ids": test_ids,
        "train_ids": train_ids,
        "bundle": final_bundle,
        "fold_summaries": cv_result["fold_summaries"],
        "search_result": search_result,
    }


def train_top_feature_model(
    project_root: str | Path | None = None,
    preprocessed_bundle_path: str | Path | None = None,
    min_features: int = DEFAULT_TOP_FEATURE_MIN,
    max_features: int = DEFAULT_TOP_FEATURE_MAX,
    selected_feature_count: int | None = None,
    importance_type: str = DEFAULT_FEATURE_IMPORTANCE_TYPE,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """
    Rank features with the original LightGBM code, then retrain using the best 20-40 features.

    The first pass trains the full-feature model. Its ensemble feature
    importance defines the feature ranking. The second pass reruns CV and final
    refit on the selected features, using raw rebuild by default so new
    research features are included.
    """
    train_options = dict(train_kwargs)
    train_options.pop("selected_features", None)
    use_preprocessed_bundle = bool(train_options.pop("use_preprocessed_bundle", False) or preprocessed_bundle_path)
    train_options["project_root"] = project_root
    train_options["use_preprocessed_bundle"] = use_preprocessed_bundle
    train_options["preprocessed_bundle_path"] = preprocessed_bundle_path

    full_feature_result = train_model(**train_options)
    importance_frame = _aggregate_feature_importance(
        full_feature_result["models"],
        full_feature_result["feature_names"],
        importance_type=importance_type,
    )
    selected_feature_names = _select_top_feature_names(
        importance_frame,
        min_features=min_features,
        max_features=max_features,
        selected_feature_count=selected_feature_count,
    )

    retrain_options = dict(train_options)
    retrain_options["selected_features"] = selected_feature_names
    top_feature_result = train_model(**retrain_options)
    top_feature_result["model_name"] = "lightgbm1_top_feature_ensemble"
    top_feature_result["training_mode"] = "top_feature_selection_plus_retrain"
    top_feature_result["feature_selection"] = {
        "importance_type": importance_type,
        "min_features": int(min_features),
        "max_features": int(max_features),
        "selected_feature_count": int(len(selected_feature_names)),
        "selected_features": selected_feature_names,
        "importance_frame": importance_frame,
        "full_feature_cv_accuracy": full_feature_result["cv_accuracy"],
        "full_feature_cv_accuracy_at_0_5": full_feature_result["cv_accuracy_at_0_5"],
        "full_feature_count": int(full_feature_result["feature_count"]),
    }
    return top_feature_result


def predict(
    model_artifact: dict[str, Any],
    X: pd.DataFrame,
    threshold: float | None = None,
    return_proba: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    Predict labels from a preprocessed LightGBM feature frame.

    The input dataframe must already match the final LightGBM feature space.
    """
    prepared = _prepare_prediction_frame(model_artifact, X)
    probability_values = np.mean(
        [model.predict_proba(prepared)[:, 1] for model in model_artifact["models"]],
        axis=0,
    )

    positive_proba = pd.Series(
        probability_values,
        index=prepared.index,
        name="TransportedProbability",
    )
    effective_threshold = model_artifact.get("threshold", 0.5) if threshold is None else threshold
    predictions = pd.Series(
        positive_proba >= float(effective_threshold),
        index=prepared.index,
        name="Transported",
    )

    if return_proba:
        return predictions, positive_proba
    return predictions


def predict_test_set(
    model_artifact: dict[str, Any],
    threshold: float | None = None,
    return_proba: bool = False,
) -> pd.DataFrame:
    """Predict on the competition test set."""
    test_ids = model_artifact.get("test_ids")
    cached_probabilities = model_artifact.get("test_probabilities")

    if test_ids is None:
        raise ValueError("test_ids are missing from the model artifact.")
    if cached_probabilities is None:
        X_test = model_artifact.get("X_test")
        if not isinstance(X_test, pd.DataFrame):
            raise ValueError("X_test is missing from the model artifact.")
        _, probabilities = predict(
            model_artifact=model_artifact,
            X=X_test,
            threshold=threshold,
            return_proba=True,
        )
        probability_values = probabilities.to_numpy()
    else:
        probability_values = np.asarray(cached_probabilities, dtype=float)

    effective_threshold = model_artifact.get("threshold", 0.5) if threshold is None else threshold
    predictions = probability_values >= float(effective_threshold)
    result = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "Transported": predictions.astype(bool),
        }
    )

    if return_proba:
        result["TransportedProbability"] = probability_values

    return result


def save_model(model_artifact: dict[str, Any], output_path: str | Path) -> Path:
    """Save the LightGBM ensemble artifact."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, output_path)
    return output_path


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load a saved LightGBM ensemble artifact."""
    return joblib.load(Path(model_path))


def save_feature_selection_report(model_artifact: dict[str, Any], output_path: str | Path) -> Path:
    """Save the selected-feature ranking report for review."""
    feature_selection = model_artifact.get("feature_selection")
    if not isinstance(feature_selection, dict):
        raise ValueError("model_artifact does not contain feature_selection results.")

    importance_frame = feature_selection.get("importance_frame")
    if not isinstance(importance_frame, pd.DataFrame):
        raise ValueError("feature_selection does not contain an importance DataFrame.")

    selected_features = set(feature_selection.get("selected_features", []))
    report = importance_frame.copy()
    report["selected"] = report["feature"].isin(selected_features)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path


def build_submission(
    model_artifact: dict[str, Any],
    output_path: str | Path = "artifacts/submission_lightgbm1.csv",
    threshold: float | None = None,
) -> Path:
    """Build the Kaggle submission CSV."""
    submission_df = predict_test_set(
        model_artifact=model_artifact,
        threshold=threshold,
        return_proba=False,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    result = train_top_feature_model(
        min_features=DEFAULT_TOP_FEATURE_MIN,
        max_features=DEFAULT_TOP_FEATURE_MAX,
    )

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

    model_path = save_model(result, "artifacts/lightgbm1_top_features_ensemble.joblib")
    submission_path = build_submission(result, output_path="artifacts/submission_lightgbm1_top_features.csv")
    report_path = save_feature_selection_report(result, "artifacts/lightgbm1_top_features_report.csv")

    print("Saved model to:", model_path)
    print("Saved submission to:", submission_path)
    print("Saved feature selection report to:", report_path)
    print("Prediction preview:")
    print(predict_test_set(result, return_proba=True).head())
