from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORY_FILL_COLUMNS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Surname"]
MISSING_INDICATORS = [
    "AgeMissing",
    "HomePlanetMissing",
    "CryoSleepMissing",
    "CabinMissing",
    "DestinationMissing",
    "VIPMissing",
    "RoomServiceMissing",
    "FoodCourtMissing",
    "ShoppingMallMissing",
    "SpaMissing",
    "VRDeckMissing",
    "NameMissing",
]
REQUIRED_IDENTIFIER_COLUMNS = ["PassengerId", "GroupID"]
COMMON_RAW_AUDIT_COLUMNS = ["PassengerId", "GroupID", "Name", "Cabin", "Surname"]
AGE_GROUP_ALLOWED_VALUES = ["Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior", "Unknown"]
FULL_PREPROCESSING_GROUP_FEATURE_MODE = "combined_train_test_full_preprocessing"
COMMON_CATEGORICAL_COLUMNS = [
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
]
BASE_NUMERIC_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "SpendCount",
    "GroupMemberNo",
    "GroupSize",
    "CabinNum",
    "SurnameFreq",
    "LuxurySpend",
    "BasicSpend",
    "LuxuryShare",
    "SpendPerActiveCategory",
    "HasAnyLuxurySpend",
    "AgeWasOutOfRange",
    "IsChild",
    "IsSenior",
    "IsSolo",
    "IsMultiPassengerGroup",
    "GroupMemberIsLeader",
    *MISSING_INDICATORS,
    "IsZeroSpend",
]
KNN_NUMERIC_FEATURES = [
    "Age",
    "TotalSpend",
    "SpendCount",
    "LuxurySpend",
    "BasicSpend",
    "LuxuryShare",
    "SpendPerActiveCategory",
    "GroupMemberNo",
    "GroupSize",
    "CabinNum",
    "SurnameFreq",
    "IsZeroSpend",
    "HasAnyLuxurySpend",
    "AgeWasOutOfRange",
    "IsChild",
    "IsSenior",
    "IsSolo",
    "GroupMemberIsLeader",
    *MISSING_INDICATORS,
]
KNN_CATEGORICAL_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "AgeGroup",
    "CabinNumBin",
    "DeckSide",
]
LOG1P_FEATURE_COLUMNS = SPEND_COLUMNS + ["TotalSpend", "LuxurySpend", "BasicSpend", "SpendPerActiveCategory"]
MODEL_SUFFIXES = {
    "logistic_regression": "lr",
    "random_forest": "rf",
    "hist_gradient_boosting": "hgb",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "catboost": "cat",
    "knn": "knn",
}
LEGACY_MODEL_NAME_ALIASES = {
    "histgradientboosting": "hist_gradient_boosting",
}
SPEND_GROUP_LEVELS = [
    ["HomePlanet", "Destination", "VIP", "Deck"],
    ["HomePlanet", "Destination", "Deck"],
    ["HomePlanet", "Deck"],
]
MISSING_LOOKUP_TOKEN = "__MISSING__"
SHARED_ENGINEERED_FEATURES = [
    "GroupID",
    "GroupMemberNo",
    "GroupSize",
    "Deck",
    "CabinNum",
    "CabinNumBin",
    "Side",
    "Surname",
    "SurnameFreq",
    "TotalSpend",
    "IsZeroSpend",
    "SpendCount",
    "LuxurySpend",
    "BasicSpend",
    "LuxuryShare",
    "SpendPerActiveCategory",
    "HasAnyLuxurySpend",
    "AgeWasOutOfRange",
    "IsChild",
    "IsSenior",
    "AgeGroup",
    "IsSolo",
    "IsMultiPassengerGroup",
    "GroupMemberIsLeader",
    "DeckSide",
    "HomePlanetDestination",
    *MISSING_INDICATORS,
]


def get_project_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    """Return all important project paths used by the preprocessing scaffold."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    processed_root = root / "processed"
    # The repo may store Kaggle CSVs either under an extracted `spaceship-titanic/`
    # directory or under `data/raw/`. Prefer the extracted directory when present.
    default_kaggle_dir = root / "spaceship-titanic"
    fallback_raw_dir = root / "data" / "raw"
    data_dir = default_kaggle_dir if default_kaggle_dir.exists() else fallback_raw_dir
    paths = {
        "project_root": root,
        "data_dir": data_dir,
        "processed_root": processed_root,
        "common_dir": processed_root / "common",
    }
    for model_name in MODEL_SUFFIXES:
        paths[f"{model_name}_dir"] = processed_root / model_name
    return paths


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSV files from the Spaceship Titanic dataset directory."""
    data_path = Path(data_dir)
    print(f"[load_raw_data] Loading raw CSV files from: {data_path}")
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    print(f"[load_raw_data] train shape={train_df.shape}, test shape={test_df.shape}")
    return train_df, test_df


def inspect_raw_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Build a lightweight inspection report and log key raw-data facts."""
    report = {
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "train_columns": train_df.columns.tolist(),
        "test_columns": test_df.columns.tolist(),
        "target_present_in_train": "Transported" in train_df.columns,
        "target_present_in_test": "Transported" in test_df.columns,
    }
    print("[inspect_raw_data] Target in train only:", report["target_present_in_train"], report["target_present_in_test"])
    return report


def build_data_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable summary of raw train and test schema and missingness."""
    print("[build_data_summary] Building raw data summary metadata.")

    def describe_frame(df: pd.DataFrame) -> dict[str, Any]:
        categorical_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        return {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_counts": {col: int(count) for col, count in df.isna().sum().items()},
            "unique_counts": {col: int(count) for col, count in df.nunique(dropna=True).items()},
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
        }

    return {
        "train": describe_frame(train_df),
        "test": describe_frame(test_df),
        "shared_columns": sorted(set(train_df.columns).intersection(test_df.columns)),
        "train_only_columns": sorted(set(train_df.columns) - set(test_df.columns)),
        "test_only_columns": sorted(set(test_df.columns) - set(train_df.columns)),
        "target": {
            "name": "Transported",
            "present_in_train": "Transported" in train_df.columns,
            "present_in_test": "Transported" in test_df.columns,
            "description": "Binary target sourced from train.csv only and cast to int 0/1 in shared preprocessing.",
        },
    }


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light raw-data cleanup without performing model-specific transformations."""
    print("[basic_cleaning] Normalizing obvious empty strings and trimming whitespace.")
    cleaned_df = df.copy()
    object_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in object_columns:
        cleaned_df[column] = cleaned_df[column].replace(r"^\s*$", np.nan, regex=True)
        cleaned_df[column] = cleaned_df[column].apply(lambda value: value.strip() if isinstance(value, str) else value)
    return cleaned_df


def _normalize_boolean_like(series: pd.Series) -> pd.Series:
    """Normalize mixed boolean-like values into pandas string True/False values."""

    def normalize_value(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, (bool, np.bool_)):
            numeric_value = float(value)
            if numeric_value == 1.0:
                return "True"
            if numeric_value == 0.0:
                return "False"
            return pd.NA

        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return "True"
        if normalized in {"false", "0", "no"}:
            return "False"
        return pd.NA

    return series.apply(normalize_value).astype("string")


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw fields into stable pandas dtypes used by the shared preprocessing pipeline."""
    print("[enforce_dtypes] Enforcing numeric and categorical-friendly dtypes.")
    typed_df = df.copy()
    for column in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        if column in typed_df.columns:
            typed_df[column] = pd.to_numeric(typed_df[column], errors="coerce")
    for column in ["PassengerId", "Cabin", "Name", "HomePlanet", "Destination"]:
        if column in typed_df.columns:
            typed_df[column] = typed_df[column].astype("string")
    for column in ["CryoSleep", "VIP"]:
        if column in typed_df.columns:
            typed_df[column] = _normalize_boolean_like(typed_df[column])
    if "Transported" in typed_df.columns:
        typed_df["Transported"] = typed_df["Transported"].astype(int)
    return typed_df


def split_passenger_id_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split PassengerId into group and member-level components."""
    print("[split_passenger_id_features] Splitting PassengerId into group features.")
    split_df = df.copy()
    passenger_parts = split_df["PassengerId"].astype("string").str.split("_", n=1, expand=True)
    split_df["GroupID"] = passenger_parts[0].astype("string")
    split_df["GroupMemberNo"] = pd.to_numeric(passenger_parts[1], errors="coerce")
    return split_df


def build_group_features_with_combined_ids(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build PassengerId-derived group features using combined train and test IDs."""
    print("[build_group_features_with_combined_ids] Building GroupSize from combined train/test PassengerId structure.")
    train_grouped = split_passenger_id_features(train_df)
    test_grouped = split_passenger_id_features(test_df)
    combined_groups = pd.concat(
        [train_grouped[["PassengerId", "GroupID"]], test_grouped[["PassengerId", "GroupID"]]],
        axis=0,
        ignore_index=True,
    )
    group_sizes = combined_groups["GroupID"].value_counts(dropna=False).to_dict()
    # This combined-ID computation is acceptable for the current full preprocessing stage.
    # If future cross-validation is added, GroupSize must be recomputed inside each fold.
    for current_df in (train_grouped, test_grouped):
        current_df["GroupSize"] = current_df["GroupID"].map(group_sizes).fillna(1).astype(int)
    return train_grouped, test_grouped


def build_group_features_single_split(df: pd.DataFrame) -> pd.DataFrame:
    """Build PassengerId-derived group features using only a single split."""
    print("[build_group_features_single_split] Building split-local group features without cross-split peeking.")
    grouped_df = split_passenger_id_features(df)
    group_sizes = grouped_df["GroupID"].value_counts(dropna=False).to_dict()
    grouped_df["GroupSize"] = grouped_df["GroupID"].map(group_sizes).fillna(1).astype(int)
    return grouped_df


def split_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin into Deck, CabinNum, and Side columns."""
    print("[split_cabin_features] Splitting Cabin into deck/number/side.")
    split_df = df.copy()
    cabin_parts = split_df["Cabin"].astype("string").str.split("/", n=2, expand=True)
    split_df["Deck"] = cabin_parts[0].astype("string")
    split_df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    split_df["Side"] = cabin_parts[2].astype("string")
    return split_df


def extract_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract surname-level features from the Name column."""
    print("[extract_name_features] Extracting surname tokens from Name.")
    name_df = df.copy()
    name_df["Surname"] = name_df["Name"].astype("string").str.split().str[-1]
    return name_df


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing-value indicator columns before any imputation is applied."""
    print("[create_missing_indicators] Creating missing-value indicator columns.")
    indicator_df = df.copy()
    mapping = {
        "AgeMissing": "Age",
        "HomePlanetMissing": "HomePlanet",
        "CryoSleepMissing": "CryoSleep",
        "CabinMissing": "Cabin",
        "DestinationMissing": "Destination",
        "VIPMissing": "VIP",
        "RoomServiceMissing": "RoomService",
        "FoodCourtMissing": "FoodCourt",
        "ShoppingMallMissing": "ShoppingMall",
        "SpaMissing": "Spa",
        "VRDeckMissing": "VRDeck",
        "NameMissing": "Name",
    }
    for indicator_name, source_column in mapping.items():
        indicator_df[indicator_name] = indicator_df[source_column].isna().astype(int)
    return indicator_df


def fill_group_consistent_categories(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill missing category values from groups that have one observed non-null value."""
    print(f"[fill_group_consistent_categories] Filling group-consistent columns: {columns}")
    filled_df = df.copy()
    fill_counts = {column: 0 for column in columns}

    for column in columns:
        if column not in filled_df.columns:
            continue

        observed = filled_df.loc[filled_df[column].notna(), ["GroupID", column]].copy()
        if observed.empty:
            continue

        unique_counts = observed.groupby("GroupID")[column].nunique(dropna=True)
        valid_group_ids = unique_counts[unique_counts == 1].index
        if len(valid_group_ids) == 0:
            continue

        group_value_map = observed[observed["GroupID"].isin(valid_group_ids)].groupby("GroupID")[column].first().to_dict()
        candidate_values = filled_df["GroupID"].map(group_value_map)
        fill_mask = filled_df[column].isna() & candidate_values.notna()
        fill_counts[column] = int(fill_mask.sum())
        if fill_counts[column] > 0:
            filled_df.loc[fill_mask, column] = candidate_values.loc[fill_mask]

    return filled_df, fill_counts


def infer_missing_cryosleep(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Infer missing CryoSleep values using strict spend-based rules."""
    print("[infer_missing_cryosleep] Applying rule-based CryoSleep inference.")
    inferred_df = df.copy()
    summary = {
        "positive_spend_to_false": 0,
        "adult_zero_spend_to_true": 0,
    }

    spend_df = inferred_df[SPEND_COLUMNS]
    positive_spend_mask = spend_df.gt(0).any(axis=1)
    rule_false_mask = inferred_df["CryoSleep"].isna() & positive_spend_mask
    summary["positive_spend_to_false"] = int(rule_false_mask.sum())
    inferred_df.loc[rule_false_mask, "CryoSleep"] = "False"

    updated_spend_df = inferred_df[SPEND_COLUMNS]
    all_spend_known = updated_spend_df.notna().all(axis=1)
    zero_total_spend = updated_spend_df.sum(axis=1) == 0
    adult_zero_mask = (
        inferred_df["CryoSleep"].isna()
        & inferred_df["Age"].notna()
        & (inferred_df["Age"] >= 18)
        & all_spend_known
        & zero_total_spend
    )
    summary["adult_zero_spend_to_true"] = int(adult_zero_mask.sum())
    inferred_df.loc[adult_zero_mask, "CryoSleep"] = "True"

    return inferred_df, summary


def apply_cryosleep_spend_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the business rule that CryoSleep passengers should have zero spend."""
    print("[apply_cryosleep_spend_rule] Forcing spend columns to zero when CryoSleep is True.")
    rule_df = df.copy()
    cryo_mask = rule_df["CryoSleep"].eq("True").fillna(False)
    for column in SPEND_COLUMNS:
        rule_df.loc[cryo_mask, column] = 0.0
    return rule_df


def _encode_lookup_key(values: list[Any]) -> str:
    """Encode lookup values into a stable JSON-safe key."""
    encoded_values = [MISSING_LOOKUP_TOKEN if pd.isna(value) else str(value) for value in values]
    return json.dumps(encoded_values, ensure_ascii=False)


def create_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared spend-based engineered features after spend columns are cleaned."""
    print("[create_spend_features] Building TotalSpend, IsZeroSpend, and SpendCount.")
    spend_df = df.copy()
    spend_df["TotalSpend"] = spend_df[SPEND_COLUMNS].sum(axis=1)
    spend_df["IsZeroSpend"] = (spend_df["TotalSpend"] == 0).astype(int)
    spend_df["SpendCount"] = (spend_df[SPEND_COLUMNS] > 0).sum(axis=1).astype(int)
    return spend_df


def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared age-derived features using fixed bins."""
    print("[create_age_features] Building age segment features.")
    age_df = df.copy()
    age_df["IsChild"] = (age_df["Age"] < 18).astype(int)
    age_df["IsSenior"] = (age_df["Age"] >= 60).astype(int)
    age_bins = [-0.001, 13, 18, 30, 45, 60, np.inf]
    age_df["AgeGroup"] = pd.cut(
        age_df["Age"],
        bins=age_bins,
        labels=AGE_GROUP_ALLOWED_VALUES[:-1],
        right=False,
        include_lowest=True,
    )
    age_df["AgeGroup"] = age_df["AgeGroup"].astype("string").fillna("Unknown")
    return age_df


def create_spend_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared spend-structure features after total spend is available."""
    print("[create_spend_structure_features] Building spend structure features.")
    spend_df = df.copy()
    spend_df["LuxurySpend"] = spend_df["Spa"] + spend_df["VRDeck"]
    spend_df["BasicSpend"] = spend_df["RoomService"] + spend_df["FoodCourt"] + spend_df["ShoppingMall"]
    spend_df["LuxuryShare"] = spend_df["LuxurySpend"] / (spend_df["TotalSpend"] + 1.0)
    active_count = spend_df["SpendCount"].replace(0, np.nan)
    spend_df["SpendPerActiveCategory"] = (spend_df["TotalSpend"] / active_count).fillna(0.0)
    spend_df["HasAnyLuxurySpend"] = (spend_df["LuxurySpend"] > 0).astype(int)
    spend_df["LuxuryShare"] = np.nan_to_num(spend_df["LuxuryShare"], nan=0.0, posinf=0.0, neginf=0.0)
    spend_df["SpendPerActiveCategory"] = np.nan_to_num(
        spend_df["SpendPerActiveCategory"], nan=0.0, posinf=0.0, neginf=0.0
    )
    return spend_df


def create_group_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shared group-structure features from passenger grouping columns."""
    print("[create_group_structure_features] Building group structure features.")
    group_df = df.copy()
    group_df["IsSolo"] = (group_df["GroupSize"] == 1).astype(int)
    group_df["IsMultiPassengerGroup"] = (group_df["GroupSize"] > 1).astype(int)
    group_df["GroupMemberIsLeader"] = (group_df["GroupMemberNo"] == 1).astype(int)
    return group_df


def create_interaction_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable categorical interaction features after category filling."""
    print("[create_interaction_categorical_features] Building interaction categorical features.")
    interaction_df = df.copy()
    deck_values = interaction_df["Deck"].astype("string").fillna("Unknown")
    side_values = interaction_df["Side"].astype("string").fillna("Unknown")
    home_values = interaction_df["HomePlanet"].astype("string").fillna("Unknown")
    destination_values = interaction_df["Destination"].astype("string").fillna("Unknown")
    interaction_df["DeckSide"] = (deck_values + "_" + side_values).astype("string")
    interaction_df["HomePlanetDestination"] = (home_values + "_" + destination_values).astype("string")
    return interaction_df


def _fit_spend_group_medians(train_df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Fit train-only hierarchical median maps for each spend column."""
    spend_group_median_maps: dict[str, dict[str, dict[str, float]]] = {}

    for spend_column in SPEND_COLUMNS:
        spend_group_median_maps[spend_column] = {}
        for level_columns in SPEND_GROUP_LEVELS:
            level_name = "|".join(level_columns)
            observed = train_df.loc[train_df[spend_column].notna(), level_columns + [spend_column]].copy()
            if observed.empty:
                spend_group_median_maps[spend_column][level_name] = {}
                continue

            lookup_keys = observed[level_columns].apply(lambda row: _encode_lookup_key(row.tolist()), axis=1)
            medians = observed.groupby(lookup_keys, sort=False)[spend_column].median()
            spend_group_median_maps[spend_column][level_name] = {
                str(key): float(value) for key, value in medians.items()
            }

    return spend_group_median_maps


def _fit_cabin_num_bin_edges(train_df: pd.DataFrame, target_bins: int = 5) -> list[float]:
    """Fit train-only CabinNum bin edges using non-missing train values only."""
    observed = pd.to_numeric(train_df["CabinNum"], errors="coerce").dropna()
    if observed.empty:
        return [float("-inf"), float("inf")]

    unique_count = int(observed.nunique())
    if unique_count <= 1:
        return [float("-inf"), float("inf")]

    max_bins = min(target_bins, unique_count)
    observed_values = observed.to_numpy(dtype=float)

    for bin_count in range(max_bins, 1, -1):
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        edges = np.quantile(observed_values, quantiles)
        unique_edges = np.unique(edges.astype(float))
        if len(unique_edges) < 2:
            continue
        cut_edges = unique_edges.tolist()
        cut_edges[0] = float("-inf")
        cut_edges[-1] = float("inf")
        if len(cut_edges) >= 2:
            return [float(edge) for edge in cut_edges]

    return [float("-inf"), float("inf")]


def _normalize_age_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Coerce age to numeric, flag out-of-range values, and mask them as missing."""
    numeric_age = pd.to_numeric(series, errors="coerce")
    out_of_range_mask = numeric_age.lt(0) | numeric_age.gt(100)
    normalized_age = numeric_age.mask(out_of_range_mask)
    return normalized_age, out_of_range_mask.fillna(False).astype(int)


def fit_common_statistics(train_df: pd.DataFrame) -> dict[str, Any]:
    """Fit train-only statistics shared across model-specific preprocessing branches."""
    print("[fit_common_statistics] Fitting train-only statistics for Age, SurnameFreq, spend, and CabinNumBin.")
    normalized_age, _ = _normalize_age_series(train_df["Age"])
    surname_counts = train_df["Surname"].dropna().value_counts().to_dict()
    spend_global_median_map: dict[str, float] = {}
    spend_positive_median_map: dict[str, float] = {}
    for spend_column in SPEND_COLUMNS:
        observed_values = train_df[spend_column].dropna()
        global_median = float(observed_values.median()) if not observed_values.empty else 0.0
        positive_values = train_df.loc[train_df[spend_column].gt(0).fillna(False), spend_column].dropna()
        positive_median = float(positive_values.median()) if not positive_values.empty else global_median
        spend_global_median_map[spend_column] = global_median
        spend_positive_median_map[spend_column] = positive_median

    train_fit_notes = [
        "SurnameFreq is fit on train surnames only and must be recomputed inside each CV fold.",
        "CabinNumBin edges are fit on non-missing train CabinNum values only and must be recomputed inside each CV fold.",
        "Spend medians and hierarchical spend group medians are train-fit statistics and must be recomputed inside each CV fold.",
        "Age out-of-range handling and age median are train-fit preprocessing assumptions for future fold-local reuse.",
        "GroupSize currently uses combined_train_test_full_preprocessing and must switch to split-local or fold-local helpers in future CV.",
    ]

    return {
        "age_median": float(normalized_age.median(skipna=True)),
        "surname_freq_map": {str(key): int(value) for key, value in surname_counts.items()},
        "spend_global_median_map": spend_global_median_map,
        "spend_positive_median_map": spend_positive_median_map,
        "spend_group_levels": [level.copy() for level in SPEND_GROUP_LEVELS],
        "spend_group_median_maps": _fit_spend_group_medians(train_df),
        "cabin_num_bin_edges": _fit_cabin_num_bin_edges(train_df),
        "train_fit_notes": train_fit_notes,
        "config": {
            "cabin_num_bin_target_bins": 5,
            "group_feature_mode": FULL_PREPROCESSING_GROUP_FEATURE_MODE,
            "future_cv_recompute_items": [
                "SurnameFreq",
                "CabinNumBin edges",
                "spend medians",
                "hierarchical spend medians",
                "age median after out-of-range filtering",
                "combined-ID GroupSize",
            ],
        },
    }


def _apply_cabin_num_bins(
    df: pd.DataFrame, cabin_num_bin_edges: list[float], original_cabin_missing_mask: pd.Series
) -> pd.Series:
    """Apply train-fit CabinNum bin edges while keeping missing values in a separate bucket."""
    non_missing_mask = ~original_cabin_missing_mask
    bin_labels = [f"CabinBin_{index}" for index in range(1, max(len(cabin_num_bin_edges), 2))]
    binned = pd.Series(index=df.index, dtype="string")

    if non_missing_mask.any():
        non_missing_values = pd.to_numeric(df.loc[non_missing_mask, "CabinNum"], errors="coerce")
        cut_values = pd.cut(
            non_missing_values,
            bins=cabin_num_bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )
        binned.loc[non_missing_mask] = cut_values.astype("string")

    binned.loc[original_cabin_missing_mask] = "CabinBin_Missing"
    binned = binned.astype("string")

    if binned.isna().any():
        raise ValueError("[apply_common_statistics] CabinNumBin generation left missing values.")

    return binned


def _apply_spend_imputation(df: pd.DataFrame, stats: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Impute spend columns using hierarchical train-fit medians and fallbacks."""
    print("[_apply_spend_imputation] Filling spend columns with hierarchical medians and fallbacks.")
    filled_df = df.copy()
    original_df = df.copy()
    spend_fill_summary = {
        spend_column: {"group_median": 0, "positive_median": 0, "global_median": 0}
        for spend_column in SPEND_COLUMNS
    }

    for spend_column in SPEND_COLUMNS:
        cryo_mask = filled_df["CryoSleep"].eq("True").fillna(False)
        not_cryo_true_mask = filled_df["CryoSleep"].ne("True").fillna(True)
        filled_df.loc[cryo_mask, spend_column] = 0.0

        remaining_mask = filled_df[spend_column].isna() & not_cryo_true_mask
        for level_columns in stats["spend_group_levels"]:
            if not remaining_mask.any():
                break

            level_name = "|".join(level_columns)
            lookup_map = stats["spend_group_median_maps"][spend_column].get(level_name, {})
            if not lookup_map:
                continue

            lookup_keys = filled_df.loc[remaining_mask, level_columns].apply(
                lambda row: _encode_lookup_key(row.tolist()),
                axis=1,
            )
            mapped_values = lookup_keys.map(lookup_map)
            fillable_index = mapped_values[mapped_values.notna()].index
            if len(fillable_index) > 0:
                filled_df.loc[fillable_index, spend_column] = mapped_values.loc[fillable_index].astype(float)
                spend_fill_summary[spend_column]["group_median"] += int(len(fillable_index))
                not_cryo_true_mask = filled_df["CryoSleep"].ne("True").fillna(True)
                remaining_mask = filled_df[spend_column].isna() & not_cryo_true_mask

        if remaining_mask.any():
            other_spend_columns = [column for column in SPEND_COLUMNS if column != spend_column]
            observed_positive_other = (
                original_df.loc[remaining_mask, other_spend_columns].gt(0)
                & original_df.loc[remaining_mask, other_spend_columns].notna()
            ).any(axis=1)
            positive_index = observed_positive_other[observed_positive_other].index
            if len(positive_index) > 0:
                filled_df.loc[positive_index, spend_column] = float(stats["spend_positive_median_map"][spend_column])
                spend_fill_summary[spend_column]["positive_median"] += int(len(positive_index))
                not_cryo_true_mask = filled_df["CryoSleep"].ne("True").fillna(True)
                remaining_mask = filled_df[spend_column].isna() & not_cryo_true_mask

        if remaining_mask.any():
            global_index = remaining_mask[remaining_mask].index
            filled_df.loc[global_index, spend_column] = float(stats["spend_global_median_map"][spend_column])
            spend_fill_summary[spend_column]["global_median"] += int(len(global_index))

        filled_df[spend_column] = pd.to_numeric(filled_df[spend_column], errors="coerce")
        if filled_df[spend_column].isna().any():
            raise ValueError(
                f"[apply_common_statistics] Spend imputation failed for column '{spend_column}'; missing values remain."
            )

    return filled_df, spend_fill_summary


def apply_common_statistics(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    """Apply train-fit statistics and deterministic fills to a shared feature frame."""
    print("[apply_common_statistics] Applying train-fit statistics and deterministic fills.")
    filled_df = df.copy()
    original_cabin_missing_mask = filled_df["CabinNum"].isna()
    normalized_age, age_out_of_range_mask = _normalize_age_series(filled_df["Age"])
    filled_df["AgeWasOutOfRange"] = age_out_of_range_mask.astype(int)
    filled_df["Age"] = normalized_age.fillna(stats["age_median"])
    filled_df["GroupMemberNo"] = filled_df["GroupMemberNo"].fillna(-1).astype(int)
    filled_df["GroupSize"] = filled_df["GroupSize"].fillna(1).astype(int)
    filled_df, spend_fill_summary = _apply_spend_imputation(filled_df, stats)
    filled_df["SurnameFreq"] = filled_df["Surname"].map(stats["surname_freq_map"]).fillna(0).astype(int)
    filled_df["CabinNumBin"] = _apply_cabin_num_bins(filled_df, stats["cabin_num_bin_edges"], original_cabin_missing_mask)
    filled_df["CabinNum"] = filled_df["CabinNum"].fillna(-1)

    for column in CATEGORY_FILL_COLUMNS:
        filled_df[column] = filled_df[column].astype("string")
        filled_df[column] = filled_df[column].fillna("Unknown")

    for column in COMMON_CATEGORICAL_COLUMNS:
        if column in filled_df.columns:
            filled_df[column] = filled_df[column].astype("string")

    for column in MISSING_INDICATORS:
        if column in filled_df.columns:
            filled_df[column] = filled_df[column].astype(int)

    numeric_columns = [
        "Age",
        "AgeWasOutOfRange",
        "CabinNum",
        "GroupMemberNo",
        "GroupSize",
        "SurnameFreq",
        *SPEND_COLUMNS,
        *MISSING_INDICATORS,
    ]
    for column in numeric_columns:
        if filled_df[column].isna().any():
            raise ValueError(f"[apply_common_statistics] Numeric column '{column}' still contains missing values.")

    filled_df.attrs["spend_fill_summary"] = spend_fill_summary
    return filled_df


def finalize_common_frame(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Finalize the shared common frame so raw audit columns remain stable and complete."""
    print(f"[finalize_common_frame] Finalizing common frame for {split_name}.")
    finalized_df = df.copy()

    for column in REQUIRED_IDENTIFIER_COLUMNS:
        finalized_df[column] = finalized_df[column].astype("string")
        if finalized_df[column].isna().any():
            raise ValueError(f"[finalize_common_frame:{split_name}] Identifier column '{column}' contains missing values.")

    for column in ["Name", "Cabin", "Surname"]:
        finalized_df[column] = finalized_df[column].astype("string")
    finalized_df["Name"] = finalized_df["Name"].fillna("Unknown")
    finalized_df["Cabin"] = finalized_df["Cabin"].fillna("Unknown")
    finalized_df["Surname"] = finalized_df["Surname"].fillna("Unknown")

    for column in COMMON_CATEGORICAL_COLUMNS + COMMON_RAW_AUDIT_COLUMNS:
        if column in finalized_df.columns:
            finalized_df[column] = finalized_df[column].astype("string")

    object_columns = finalized_df.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        finalized_df[column] = finalized_df[column].astype("string")

    return finalized_df


def _validate_common_frame(df: pd.DataFrame, stage_name: str) -> None:
    """Validate the final shared feature frame before model-specific preprocessing begins."""
    for column in ["Age", "CabinNum", *SPEND_COLUMNS]:
        if df[column].isna().any():
            raise ValueError(f"[{stage_name}] Required numeric column '{column}' still contains missing values.")

    for column in BASE_NUMERIC_FEATURES:
        if column not in df.columns:
            raise ValueError(f"[{stage_name}] Required BASE_NUMERIC_FEATURES column '{column}' is missing.")
        if df[column].isna().any():
            raise ValueError(f"[{stage_name}] BASE_NUMERIC_FEATURES column '{column}' still contains missing values.")

    for column in COMMON_CATEGORICAL_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"[{stage_name}] Required categorical column '{column}' is missing.")
        if df[column].isna().any():
            raise ValueError(f"[{stage_name}] Categorical column '{column}' still contains missing values.")

    invalid_age_groups = sorted(set(df["AgeGroup"].astype(str)) - set(AGE_GROUP_ALLOWED_VALUES))
    if invalid_age_groups:
        raise ValueError(f"[{stage_name}] AgeGroup contains unsupported values: {invalid_age_groups}")

    for column in COMMON_RAW_AUDIT_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"[{stage_name}] Required audit column '{column}' is missing.")
        if df[column].isna().any():
            raise ValueError(f"[{stage_name}] Audit column '{column}' still contains missing values.")

    if not df["PassengerId"].is_unique:
        raise ValueError(f"[{stage_name}] PassengerId values must be unique within each split.")

    cryo_true_rows = df["CryoSleep"].eq("True").fillna(False)
    if cryo_true_rows.any():
        invalid_spend_rows = df.loc[cryo_true_rows, SPEND_COLUMNS].ne(0).any(axis=1)
        if invalid_spend_rows.any():
            raise ValueError(f"[{stage_name}] CryoSleep == 'True' rows must have all spend columns equal to 0.")

    if df.isna().any().any():
        missing_columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"[{stage_name}] Final common frame still contains missing values: {missing_columns}")


def _assert_ids_aligned(expected_ids: pd.Series, frame: pd.DataFrame, split_name: str) -> str:
    """Assert that extracted IDs remain aligned with the finalized common frame order."""
    expected_values = expected_ids.astype("string").astype(str).tolist()
    frame_values = frame["PassengerId"].astype("string").astype(str).tolist()
    if expected_values != frame_values:
        raise ValueError(f"[build_common_features] {split_name} ids are not aligned with finalized common frame rows.")
    return "pass"


def _summarize_missing_counts(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize missing counts without embedding row-level data."""
    missing_counts = df.isna().sum()
    non_zero_missing = {str(column): int(count) for column, count in missing_counts.items() if int(count) > 0}
    return {
        "total_missing_values": int(missing_counts.sum()),
        "columns_with_missing": int((missing_counts > 0).sum()),
        "missing_counts_by_column": non_zero_missing,
    }


def _make_json_serializable(value: Any) -> Any:
    """Convert pandas/numpy-heavy structures into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(key): _make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, pd.Series):
        return [_make_json_serializable(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [_make_json_serializable(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_make_json_serializable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def _collect_categorical_cardinality_summary(
    common_train: pd.DataFrame, common_test: pd.DataFrame
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Compute train/test/combined cardinality summaries for common categorical columns."""
    combined_unique_counts: dict[str, int] = {}
    cardinality_summary: dict[str, dict[str, int]] = {}

    for column in COMMON_CATEGORICAL_COLUMNS:
        train_values = common_train[column].dropna().astype("string").astype(str)
        test_values = common_test[column].dropna().astype("string").astype(str)
        train_unique = {str(value) for value in train_values.tolist()}
        test_unique = {str(value) for value in test_values.tolist()}
        combined_unique = train_unique.union(test_unique)
        combined_unique_counts[column] = int(len(combined_unique))
        cardinality_summary[column] = {
            "train_unique_count": int(len(train_unique)),
            "test_unique_count": int(len(test_unique)),
            "combined_unique_count": int(len(combined_unique)),
        }

    return combined_unique_counts, cardinality_summary


def build_preprocessing_quality_report(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    common_train: pd.DataFrame,
    common_test: pd.DataFrame,
    stats: dict[str, Any],
    rule_summary: dict[str, Any],
    train_ids_alignment_check: str,
    test_ids_alignment_check: str,
) -> dict[str, Any]:
    """Build a lightweight, JSON-safe preprocessing quality report."""
    print("[build_preprocessing_quality_report] Building preprocessing quality report.")
    overlap_count = int(len(set(common_train["PassengerId"].astype(str)).intersection(set(common_test["PassengerId"].astype(str)))))
    categorical_unique_counts, categorical_cardinality_summary = _collect_categorical_cardinality_summary(
        common_train, common_test
    )
    age_quality_summary = {
        "age_missing_after_fill": int(common_train["Age"].isna().sum() + common_test["Age"].isna().sum()),
        "age_out_of_range_flag_count": int(
            common_train["AgeWasOutOfRange"].sum() + common_test["AgeWasOutOfRange"].sum()
        ),
        "age_group_unknown_count_train": int(common_train["AgeGroup"].eq("Unknown").sum()),
        "age_group_unknown_count_test": int(common_test["AgeGroup"].eq("Unknown").sum()),
    }
    cabin_bin_summary = {
        "cabin_num_bin_count": int(categorical_cardinality_summary["CabinNumBin"]["combined_unique_count"]),
        "cabin_num_missing_bucket_count_train": int(common_train["CabinNumBin"].eq("CabinBin_Missing").sum()),
        "cabin_num_missing_bucket_count_test": int(common_test["CabinNumBin"].eq("CabinBin_Missing").sum()),
    }

    report = {
        "raw_missing_summary": {
            "train": _summarize_missing_counts(raw_train_df),
            "test": _summarize_missing_counts(raw_test_df),
        },
        "final_common_missing_summary": {
            "train": _summarize_missing_counts(common_train),
            "test": _summarize_missing_counts(common_test),
        },
        "identifier_checks": {
            "train_passenger_id_unique": bool(common_train["PassengerId"].is_unique),
            "test_passenger_id_unique": bool(common_test["PassengerId"].is_unique),
            "cross_split_passenger_id_overlap_count": overlap_count,
            "train_ids_alignment_check": train_ids_alignment_check,
            "test_ids_alignment_check": test_ids_alignment_check,
        },
        "common_shape_summary": {
            "common_train_shape": [int(common_train.shape[0]), int(common_train.shape[1])],
            "common_test_shape": [int(common_test.shape[0]), int(common_test.shape[1])],
        },
        "common_categorical_unique_counts": categorical_unique_counts,
        "common_categorical_cardinality_summary": categorical_cardinality_summary,
        "model_input_overview": {},
        "age_quality_summary": age_quality_summary,
        "cabin_bin_summary": cabin_bin_summary,
        "group_feature_mode": FULL_PREPROCESSING_GROUP_FEATURE_MODE,
        "model_branch_notes": {
            "lightgbm_category_levels_aligned": True,
        },
        "engineered_features": SHARED_ENGINEERED_FEATURES,
        "train_fit_statistics": {
            "stat_keys": sorted(stats.keys()),
            "surname_freq_entry_count": int(len(stats["surname_freq_map"])),
            "spend_group_levels": ["|".join(level) for level in stats["spend_group_levels"]],
            "cabin_num_bin_count": int(max(len(stats["cabin_num_bin_edges"]) - 1, 1)),
            "train_fit_notes": stats["train_fit_notes"],
        },
        "future_cv_recompute_items": stats["config"]["future_cv_recompute_items"],
        "rule_summary": rule_summary,
    }
    return _make_json_serializable(report)


def build_common_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Build the shared cleaned and engineered feature tables used by all models."""
    print("[build_common_features] Starting shared preprocessing pipeline.")

    raw_summary = build_data_summary(train_df, test_df)
    y_train = train_df["Transported"].astype(int).copy()
    train_ids = train_df["PassengerId"].astype("string").copy()
    test_ids = test_df["PassengerId"].astype("string").copy()

    train_working = enforce_dtypes(basic_cleaning(train_df.drop(columns=["Transported"])))
    test_working = enforce_dtypes(basic_cleaning(test_df))

    train_working, test_working = build_group_features_with_combined_ids(train_working, test_working)
    train_working = split_cabin_features(train_working)
    test_working = split_cabin_features(test_working)
    train_working = extract_name_features(train_working)
    test_working = extract_name_features(test_working)

    train_working = create_missing_indicators(train_working)
    test_working = create_missing_indicators(test_working)

    group_fill_columns = ["HomePlanet", "VIP", "Destination"]
    train_working, train_group_fill = fill_group_consistent_categories(train_working, group_fill_columns)
    test_working, test_group_fill = fill_group_consistent_categories(test_working, group_fill_columns)

    train_working, train_cryosleep_summary = infer_missing_cryosleep(train_working)
    test_working, test_cryosleep_summary = infer_missing_cryosleep(test_working)
    train_working = apply_cryosleep_spend_rule(train_working)
    test_working = apply_cryosleep_spend_rule(test_working)

    common_stats = fit_common_statistics(train_working)
    train_working = apply_common_statistics(train_working, common_stats)
    test_working = apply_common_statistics(test_working, common_stats)
    train_spend_fill_summary = train_working.attrs.get("spend_fill_summary", {})
    test_spend_fill_summary = test_working.attrs.get("spend_fill_summary", {})

    train_working = create_spend_features(train_working)
    test_working = create_spend_features(test_working)
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

    _validate_common_frame(train_working, "common_train_final")
    _validate_common_frame(test_working, "common_test_final")

    overlap_count = int(len(set(train_working["PassengerId"].astype(str)).intersection(set(test_working["PassengerId"].astype(str)))))
    if overlap_count > 0:
        raise ValueError("[build_common_features] Train and test PassengerId values must not overlap.")

    train_ids_alignment_check = _assert_ids_aligned(train_ids, train_working, "train")
    test_ids_alignment_check = _assert_ids_aligned(test_ids, test_working, "test")

    rule_summary = {
        "train": {
            "group_fill": train_group_fill,
            "cryosleep_inference": train_cryosleep_summary,
            "spend_fill": train_spend_fill_summary,
        },
        "test": {
            "group_fill": test_group_fill,
            "cryosleep_inference": test_cryosleep_summary,
            "spend_fill": test_spend_fill_summary,
        },
    }

    quality_report = build_preprocessing_quality_report(
        raw_train_df=train_df,
        raw_test_df=test_df,
        common_train=train_working,
        common_test=test_working,
        stats=common_stats,
        rule_summary=rule_summary,
        train_ids_alignment_check=train_ids_alignment_check,
        test_ids_alignment_check=test_ids_alignment_check,
    )

    common_bundle = {
        "common_train": train_working,
        "common_test": test_working,
        "y_train": y_train,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "stats": common_stats,
        "data_summary": raw_summary,
        "notes": [
            "Transported is sourced from train.csv only and cast to int 0/1 in shared preprocessing.",
            "SurnameFreq is a train-fit frequency statistic. Unseen test surnames map to 0.",
            "GroupSize currently uses combined_train_test_full_preprocessing for full preprocessing only.",
            "If cross-validation is added later, SurnameFreq, CabinNumBin edges, age median after out-of-range filtering, spend medians, and GroupSize must be recomputed inside each fold.",
            "CryoSleep and VIP use robust boolean normalization instead of direct bool-only mapping.",
            "HomePlanet, VIP, and Destination use within-split group-consistent filling when a GroupID has exactly one observed value.",
            "Missing CryoSleep values are inferred only by strict spend-based rules before spend imputation.",
            "Spend columns use train-fit hierarchical median imputation, positive-spend fallback, and global median fallback.",
            "CabinNumBin is fit on non-missing train CabinNum values and keeps raw missing CabinNum values in CabinBin_Missing.",
            "Age is coerced to numeric before range checks; out-of-range ages set AgeWasOutOfRange=1, are reset to missing, and then filled with the train-fit age median.",
            "AgeGroup uses fixed bins plus an explicit Unknown bucket; Adult is not used as an exception fallback.",
            "Common feature tables retain raw audit columns for traceability, but model feature sets continue to exclude them by default.",
            "A lightweight quality_report summarizes missingness, identifier checks, engineered features, and train-fit preprocessing assumptions.",
        ],
        "rule_summary": rule_summary,
        "quality_report": quality_report,
    }
    print("[build_common_features] Shared preprocessing pipeline completed.")
    return common_bundle


def _build_feature_set(
    common_train: pd.DataFrame,
    common_test: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    dropped_features: list[str],
) -> dict[str, Any]:
    """Assemble a structured model feature selection result."""
    selected_columns = numeric_features + categorical_features
    return {
        "model_name": model_name,
        "train_df": common_train[selected_columns].copy(),
        "test_df": common_test[selected_columns].copy(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_features": dropped_features,
        "train_ids": train_ids.copy(),
        "test_ids": test_ids.copy(),
    }


def get_feature_sets_for_lr(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select Logistic Regression input columns from the shared feature tables."""
    print("[get_feature_sets_for_lr] Selecting Logistic Regression feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "logistic_regression",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS.copy(),
        COMMON_RAW_AUDIT_COLUMNS.copy(),
    )


def get_feature_sets_for_rf(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select Random Forest input columns from the shared feature tables."""
    print("[get_feature_sets_for_rf] Selecting Random Forest feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "random_forest",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS.copy(),
        COMMON_RAW_AUDIT_COLUMNS.copy(),
    )


def get_feature_sets_for_hgb(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select HistGradientBoosting input columns from the shared feature tables."""
    print("[get_feature_sets_for_hgb] Selecting HistGradientBoosting feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "hist_gradient_boosting",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS.copy(),
        COMMON_RAW_AUDIT_COLUMNS.copy(),
    )


def get_feature_sets_for_hist_gradient_boosting(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select HistGradientBoosting input columns using the canonical public branch name."""
    return get_feature_sets_for_hgb(common_train, common_test, train_ids, test_ids)


def get_feature_sets_for_xgb(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select XGBoost input columns from the shared feature tables."""
    print("[get_feature_sets_for_xgb] Selecting XGBoost feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "xgboost",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS.copy(),
        COMMON_RAW_AUDIT_COLUMNS.copy(),
    )


def get_feature_sets_for_lgbm(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select LightGBM input columns from the shared feature tables."""
    print("[get_feature_sets_for_lgbm] Selecting LightGBM feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "lightgbm",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS.copy(),
        COMMON_RAW_AUDIT_COLUMNS.copy(),
    )


def get_feature_sets_for_cat(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select CatBoost input columns from the shared feature tables."""
    print("[get_feature_sets_for_cat] Selecting CatBoost feature columns.")
    dropped_features = [column for column in COMMON_RAW_AUDIT_COLUMNS if column != "Surname"]
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "catboost",
        BASE_NUMERIC_FEATURES.copy(),
        COMMON_CATEGORICAL_COLUMNS + ["Surname"],
        dropped_features,
    )


def get_feature_sets_for_knn(
    common_train: pd.DataFrame, common_test: pd.DataFrame, train_ids: pd.Series, test_ids: pd.Series
) -> dict[str, Any]:
    """Select a compact KNN feature subset from the shared feature tables."""
    print("[get_feature_sets_for_knn] Selecting compact KNN feature columns.")
    return _build_feature_set(
        common_train,
        common_test,
        train_ids,
        test_ids,
        "knn",
        KNN_NUMERIC_FEATURES.copy(),
        KNN_CATEGORICAL_FEATURES.copy(),
        COMMON_RAW_AUDIT_COLUMNS + ["HomePlanetDestination"],
    )


def _clone_with_log1p(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy with log1p applied to selected non-negative numeric columns."""
    updated_df = df.copy()
    for column in columns:
        if column in updated_df.columns:
            updated_df[column] = np.log1p(updated_df[column])
    return updated_df


def _build_model_bundle(
    model_name: str,
    X_train: Any,
    X_test: Any,
    y_train: pd.Series,
    train_ids: pd.Series,
    test_ids: pd.Series,
    feature_names: list[str] | None = None,
    preprocessor: Any | None = None,
    categorical_feature_names: list[str] | None = None,
    categorical_feature_indices: list[int] | None = None,
    dropped_features: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Create a model bundle with both generic and model-suffixed keys."""
    suffix = MODEL_SUFFIXES[model_name]
    bundle = {
        "model_name": model_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "train_ids": train_ids,
        "test_ids": test_ids,
        f"X_train_{suffix}": X_train,
        f"X_test_{suffix}": X_test,
        f"y_train_{suffix}": y_train,
        "dropped_features": dropped_features or [],
        "notes": notes or [],
    }
    if feature_names is not None:
        bundle["feature_names"] = feature_names
        bundle[f"feature_names_{suffix}"] = feature_names
    if preprocessor is not None:
        bundle["preprocessor"] = preprocessor
        bundle[f"preprocessor_{suffix}"] = preprocessor
    if categorical_feature_names is not None:
        bundle["categorical_feature_names"] = categorical_feature_names
        bundle[f"categorical_feature_names_{suffix}"] = categorical_feature_names
    if categorical_feature_indices is not None:
        bundle["categorical_feature_indices"] = categorical_feature_indices
        bundle[f"categorical_feature_indices_{suffix}"] = categorical_feature_indices
    return bundle


def preprocess_for_logistic_regression(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Transform shared features into a Logistic Regression-ready design matrix."""
    print("[preprocess_for_logistic_regression] Applying log1p, scaling, and one-hot encoding.")
    train_df = _clone_with_log1p(feature_set["train_df"], LOG1P_FEATURE_COLUMNS)
    test_df = _clone_with_log1p(feature_set["test_df"], LOG1P_FEATURE_COLUMNS)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), feature_set["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), feature_set["categorical_features"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    return _build_model_bundle(
        model_name="logistic_regression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=preprocessor.get_feature_names_out().tolist(),
        preprocessor=preprocessor,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "Logistic Regression uses standardized numeric inputs plus one-hot encoded categorical columns.",
            "Spend-related numeric columns use log1p before scaling to reduce long-tail effects.",
            "Raw audit columns remain in the common table for traceability, but are excluded from model inputs.",
        ],
    )


def preprocess_for_random_forest(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Transform shared features into a Random Forest-ready design matrix."""
    print("[preprocess_for_random_forest] Applying one-hot encoding without scaling.")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_set["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), feature_set["categorical_features"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_train = preprocessor.fit_transform(feature_set["train_df"])
    X_test = preprocessor.transform(feature_set["test_df"])
    return _build_model_bundle(
        model_name="random_forest",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=preprocessor.get_feature_names_out().tolist(),
        preprocessor=preprocessor,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "Random Forest uses cleaned numeric features and one-hot categorical features.",
            "No scaling is applied because split-based tree models are insensitive to monotonic rescaling.",
            "Raw audit columns remain in the common table for traceability, but are excluded from model inputs.",
        ],
    )


def preprocess_for_histgradientboosting(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Transform shared features into a dense HistGradientBoosting-ready table."""
    print("[preprocess_for_histgradientboosting] Applying ordinal encoding for dense gradient boosting input.")
    train_df = _clone_with_log1p(feature_set["train_df"], LOG1P_FEATURE_COLUMNS)
    test_df = _clone_with_log1p(feature_set["test_df"], LOG1P_FEATURE_COLUMNS)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded_train = encoder.fit_transform(train_df[feature_set["categorical_features"]])
    encoded_test = encoder.transform(test_df[feature_set["categorical_features"]])
    encoded_train_df = pd.DataFrame(encoded_train, columns=feature_set["categorical_features"], index=train_df.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=feature_set["categorical_features"], index=test_df.index)
    X_train = pd.concat([train_df[feature_set["numeric_features"]], encoded_train_df], axis=1)
    X_test = pd.concat([test_df[feature_set["numeric_features"]], encoded_test_df], axis=1)
    categorical_indices = [X_train.columns.get_loc(column) for column in feature_set["categorical_features"]]
    return _build_model_bundle(
        model_name="hist_gradient_boosting",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=X_train.columns.tolist(),
        preprocessor=encoder,
        categorical_feature_names=feature_set["categorical_features"],
        categorical_feature_indices=categorical_indices,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "HistGradientBoosting uses dense numeric input with ordinal-encoded categorical columns.",
            "Spend-related numeric columns use log1p before ordinal encoding and dense table assembly.",
            "Raw audit columns remain in the common table for traceability, but are excluded from model inputs.",
        ],
    )


def preprocess_for_hist_gradient_boosting(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Prepare the HistGradientBoosting preprocessing bundle using the canonical public branch name."""
    return preprocess_for_histgradientboosting(feature_set, y_train)


def preprocess_for_xgboost(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Transform shared features into an XGBoost-ready design matrix."""
    print("[preprocess_for_xgboost] Applying log1p and one-hot encoding without scaling.")
    train_df = _clone_with_log1p(feature_set["train_df"], LOG1P_FEATURE_COLUMNS)
    test_df = _clone_with_log1p(feature_set["test_df"], LOG1P_FEATURE_COLUMNS)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_set["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), feature_set["categorical_features"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    return _build_model_bundle(
        model_name="xgboost",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=preprocessor.get_feature_names_out().tolist(),
        preprocessor=preprocessor,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "XGBoost uses cleaned numeric features, log1p spend transforms, and one-hot categorical features.",
            "No scaling is applied because tree boosting is scale-invariant for split decisions.",
            "Raw audit columns remain in the common table for traceability, but are excluded from model inputs.",
        ],
    )


def preprocess_for_lightgbm(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Prepare a pandas DataFrame suitable for later LightGBM training."""
    print("[preprocess_for_lightgbm] Preserving a native tabular DataFrame with category dtypes.")
    X_train = feature_set["train_df"].copy()
    X_test = feature_set["test_df"].copy()
    for column in feature_set["categorical_features"]:
        shared_categories = pd.Index(
            pd.concat([X_train[column], X_test[column]], axis=0).dropna().astype("string").astype(str).unique()
        ).tolist()
        X_train[column] = pd.Categorical(X_train[column], categories=shared_categories)
        X_test[column] = pd.Categorical(X_test[column], categories=shared_categories)
    return _build_model_bundle(
        model_name="lightgbm",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=X_train.columns.tolist(),
        categorical_feature_names=feature_set["categorical_features"],
        dropped_features=feature_set["dropped_features"],
        notes=[
            "LightGBM keeps native categorical columns as pandas category dtype.",
            "LightGBM branch now enforces shared train/test category levels for categorical features.",
            "Raw audit columns remain in the common table for traceability, but are excluded from model inputs.",
        ],
    )


def preprocess_for_catboost(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Prepare a pandas DataFrame suitable for later CatBoost training."""
    print("[preprocess_for_catboost] Preserving native string categorical features for CatBoost.")
    X_train = feature_set["train_df"].copy()
    X_test = feature_set["test_df"].copy()
    for column in feature_set["categorical_features"]:
        X_train[column] = X_train[column].astype(str)
        X_test[column] = X_test[column].astype(str)
    categorical_indices = [X_train.columns.get_loc(column) for column in feature_set["categorical_features"]]
    return _build_model_bundle(
        model_name="catboost",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=X_train.columns.tolist(),
        categorical_feature_names=feature_set["categorical_features"],
        categorical_feature_indices=categorical_indices,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "CatBoost preserves categorical features as strings, including Surname.",
            "Raw audit columns remain in the common table for traceability, but only Surname is retained as a CatBoost-specific categorical input.",
        ],
    )


def preprocess_for_knn(feature_set: dict[str, Any], y_train: pd.Series) -> dict[str, Any]:
    """Transform shared features into a dense KNN-ready numeric matrix."""
    print("[preprocess_for_knn] Applying log1p, scaling, and dense one-hot encoding.")
    train_df = _clone_with_log1p(feature_set["train_df"], LOG1P_FEATURE_COLUMNS)
    test_df = _clone_with_log1p(feature_set["test_df"], LOG1P_FEATURE_COLUMNS)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), feature_set["numeric_features"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_set["categorical_features"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if np.isnan(X_train).any():
        raise ValueError("[preprocess_for_knn] X_train contains missing values after preprocessing.")
    if np.isnan(X_test).any():
        raise ValueError("[preprocess_for_knn] X_test contains missing values after preprocessing.")

    return _build_model_bundle(
        model_name="knn",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        train_ids=feature_set["train_ids"],
        test_ids=feature_set["test_ids"],
        feature_names=preprocessor.get_feature_names_out().tolist(),
        preprocessor=preprocessor,
        dropped_features=feature_set["dropped_features"],
        notes=[
            "KNN uses standardized numeric features and one-hot encoded categorical features.",
            "KNN is a distance-based model, so the final output is a dense numeric matrix.",
            "Spend-related numeric columns and TotalSpend use log1p before scaling to reduce long-tail effects.",
            "KNN uses a compact feature subset and explicitly excludes high-cardinality audit columns and HomePlanetDestination to avoid one-hot dimension inflation.",
        ],
    )


def _shape_as_list(value: Any) -> list[int] | None:
    """Return a JSON-friendly shape description when the object exposes shape."""
    if hasattr(value, "shape"):
        return [int(size) for size in value.shape]
    return None


def _feature_name_mentions_column(feature_name: str, column_name: str) -> bool:
    """Return True when a transformed feature name still references a disallowed source column."""
    if feature_name == column_name:
        return True
    if feature_name in {f"num__{column_name}", f"cat__{column_name}", f"remainder__{column_name}"}:
        return True
    if feature_name.startswith(f"cat__{column_name}_"):
        return True
    if feature_name.startswith(f"remainder__{column_name}_"):
        return True
    return False


def _assert_model_feature_isolation(model_name: str, feature_set: dict[str, Any], bundle: dict[str, Any]) -> None:
    """Ensure raw audit columns never leak into default model inputs."""
    disallowed_columns = set(COMMON_RAW_AUDIT_COLUMNS)
    if model_name == "catboost":
        disallowed_columns.discard("Surname")
    if model_name == "knn":
        disallowed_columns.add("HomePlanetDestination")

    selected_columns = set(feature_set["train_df"].columns).union(set(feature_set["test_df"].columns))
    leaked_selected_columns = sorted(disallowed_columns.intersection(selected_columns))
    if leaked_selected_columns:
        raise ValueError(
            f"[run_all_preprocessing] Model '{model_name}' selected disallowed audit/high-cardinality columns: "
            f"{leaked_selected_columns}"
        )

    output_feature_names: list[str] = []
    if "feature_names" in bundle:
        output_feature_names = [str(name) for name in bundle["feature_names"]]
    elif isinstance(bundle.get("X_train"), pd.DataFrame):
        output_feature_names = [str(column) for column in bundle["X_train"].columns]

    leaked_output_columns = sorted(
        {
            column
            for column in disallowed_columns
            if any(_feature_name_mentions_column(feature_name, column) for feature_name in output_feature_names)
        }
    )
    if leaked_output_columns:
        raise ValueError(
            f"[run_all_preprocessing] Model '{model_name}' output still references disallowed audit/high-cardinality "
            f"columns: {leaked_output_columns}"
        )


def _bundle_contains_nan(values: Any) -> bool:
    """Return True when a model input object still contains missing values."""
    if isinstance(values, pd.DataFrame):
        return bool(values.isna().any().any())
    if hasattr(values, "toarray") and not isinstance(values, np.ndarray):
        dense_values = np.asarray(values.toarray(), dtype=float)
        return bool(np.isnan(dense_values).any())
    if hasattr(values, "data") and not isinstance(values, np.ndarray):
        return bool(np.isnan(np.asarray(values.data, dtype=float)).any())
    return bool(np.isnan(np.asarray(values, dtype=float)).any())


def validate_model_input_bundle(model_name: str, feature_set: dict[str, Any], bundle: dict[str, Any]) -> None:
    """Validate row alignment, missingness, and feature dimensions for a model bundle."""
    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    train_ids = bundle["train_ids"]
    test_ids = bundle["test_ids"]

    if int(len(train_ids)) != int(X_train.shape[0]):
        raise ValueError(f"[validate_model_input_bundle] {model_name} train_ids length does not match X_train rows.")
    if int(len(test_ids)) != int(X_test.shape[0]):
        raise ValueError(f"[validate_model_input_bundle] {model_name} test_ids length does not match X_test rows.")
    if int(len(y_train)) != int(X_train.shape[0]):
        raise ValueError(f"[validate_model_input_bundle] {model_name} y_train length does not match X_train rows.")

    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if X_train.isna().any().any() or X_test.isna().any().any():
            raise ValueError(f"[validate_model_input_bundle] {model_name} DataFrame inputs contain missing values.")
        if X_train.columns.tolist() != X_test.columns.tolist():
            raise ValueError(f"[validate_model_input_bundle] {model_name} train/test DataFrame columns are misaligned.")
        if "feature_names" in bundle and [str(name) for name in bundle["feature_names"]] != [str(column) for column in X_train.columns]:
            raise ValueError(f"[validate_model_input_bundle] {model_name} feature_names do not match DataFrame columns.")
    else:
        if int(X_train.shape[1]) != int(X_test.shape[1]):
            raise ValueError(f"[validate_model_input_bundle] {model_name} train/test feature counts differ.")
        if "feature_names" in bundle and int(len(bundle["feature_names"])) != int(X_train.shape[1]):
            raise ValueError(
                f"[validate_model_input_bundle] {model_name} feature_names length does not match transformed columns."
            )
        if _bundle_contains_nan(X_train) or _bundle_contains_nan(X_test):
            raise ValueError(f"[validate_model_input_bundle] {model_name} transformed inputs contain missing values.")

    bundle["input_validation_passed"] = True


def save_preprocessed_bundle(bundle: dict[str, Any], save_dir: str | Path, model_name: str) -> Path:
    """Save a preprocessing bundle to joblib plus a JSON metadata sidecar."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    bundle_path = save_path / f"preprocessed_{model_name}.joblib"
    metadata_path = save_path / f"metadata_{model_name}.json"
    print(f"[save_preprocessed_bundle] Saving {model_name} bundle to: {bundle_path}")
    bundle["save_path"] = str(bundle_path)
    suffix = MODEL_SUFFIXES.get(model_name)
    if suffix is not None:
        bundle[f"save_path_{suffix}"] = str(bundle_path)

    joblib.dump(bundle, bundle_path)

    metadata = {
        "model_name": model_name,
        "save_path": str(bundle_path),
        "bundle_keys": sorted(bundle.keys()),
        "X_train_shape": _shape_as_list(bundle.get("X_train")),
        "X_test_shape": _shape_as_list(bundle.get("X_test")),
        "y_train_shape": _shape_as_list(bundle.get("y_train")),
        "feature_name_count": int(len(bundle.get("feature_names", []))) if "feature_names" in bundle else None,
        "categorical_feature_names": bundle.get("categorical_feature_names"),
        "categorical_feature_indices": bundle.get("categorical_feature_indices"),
        "dropped_features": bundle.get("dropped_features", []),
        "notes": bundle.get("notes", []),
    }
    if "input_validation_passed" in bundle:
        metadata["input_validation_passed"] = bool(bundle["input_validation_passed"])
    if model_name == "common" and "rule_summary" in bundle:
        metadata["rule_summary"] = bundle["rule_summary"]
    if model_name == "common" and "quality_report" in bundle:
        metadata["quality_report"] = bundle["quality_report"]

    metadata_path.write_text(json.dumps(_make_json_serializable(metadata), indent=2), encoding="utf-8")

    if model_name == "common":
        bundle["common_train"].head(100).to_csv(save_path / "common_train_preview.csv", index=False)
        bundle["common_test"].head(100).to_csv(save_path / "common_test_preview.csv", index=False)
        (save_path / "data_summary.json").write_text(
            json.dumps(_make_json_serializable(bundle["data_summary"]), indent=2),
            encoding="utf-8",
        )
        (save_path / "quality_report.json").write_text(
            json.dumps(_make_json_serializable(bundle["quality_report"]), indent=2),
            encoding="utf-8",
        )
    elif isinstance(bundle.get("X_train"), pd.DataFrame):
        bundle["X_train"].head(100).to_csv(save_path / f"{model_name}_train_preview.csv", index=False)
        bundle["X_test"].head(100).to_csv(save_path / f"{model_name}_test_preview.csv", index=False)

    return bundle_path


def load_preprocessed_bundle(model_name: str, processed_root: str | Path | None = None) -> dict[str, Any]:
    """Load a saved preprocessing bundle from the processed directory."""
    base_processed = Path(processed_root) if processed_root is not None else get_project_paths()["processed_root"]
    resolved_model_name = LEGACY_MODEL_NAME_ALIASES.get(model_name, model_name)
    bundle_path = base_processed / resolved_model_name / f"preprocessed_{resolved_model_name}.joblib"
    print(f"[load_preprocessed_bundle] Loading bundle from: {bundle_path}")
    return joblib.load(bundle_path)


def run_all_preprocessing(project_root: str | Path | None = None, save_outputs: bool = True) -> dict[str, dict[str, Any]]:
    """Run the complete preprocessing workflow and optionally persist all bundles."""
    print("[run_all_preprocessing] Running full preprocessing workflow.")
    paths = get_project_paths(project_root)
    train_df, test_df = load_raw_data(paths["data_dir"])
    inspection = inspect_raw_data(train_df, test_df)
    common_bundle = build_common_features(train_df, test_df)
    common_bundle["inspection"] = inspection

    results: dict[str, dict[str, Any]] = {"common": common_bundle}

    feature_builders = {
        "logistic_regression": get_feature_sets_for_lr,
        "random_forest": get_feature_sets_for_rf,
        "hist_gradient_boosting": get_feature_sets_for_hist_gradient_boosting,
        "xgboost": get_feature_sets_for_xgb,
        "lightgbm": get_feature_sets_for_lgbm,
        "catboost": get_feature_sets_for_cat,
        "knn": get_feature_sets_for_knn,
    }
    preprocessors = {
        "logistic_regression": preprocess_for_logistic_regression,
        "random_forest": preprocess_for_random_forest,
        "hist_gradient_boosting": preprocess_for_hist_gradient_boosting,
        "xgboost": preprocess_for_xgboost,
        "lightgbm": preprocess_for_lightgbm,
        "catboost": preprocess_for_catboost,
        "knn": preprocess_for_knn,
    }

    for model_name in feature_builders:
        feature_set = feature_builders[model_name](
            common_bundle["common_train"],
            common_bundle["common_test"],
            common_bundle["train_ids"],
            common_bundle["test_ids"],
        )
        model_bundle = preprocessors[model_name](feature_set, common_bundle["y_train"])
        _assert_model_feature_isolation(model_name, feature_set, model_bundle)
        validate_model_input_bundle(model_name, feature_set, model_bundle)
        if save_outputs:
            bundle_path = save_preprocessed_bundle(model_bundle, paths[f"{model_name}_dir"], model_name)
            model_bundle["save_path"] = str(bundle_path)
            model_bundle[f"save_path_{MODEL_SUFFIXES[model_name]}"] = str(bundle_path)
        results[model_name] = model_bundle

    common_bundle["quality_report"]["model_input_overview"] = {
        model_name: {
            "X_train_shape": _shape_as_list(results[model_name]["X_train"]),
            "X_test_shape": _shape_as_list(results[model_name]["X_test"]),
        }
        for model_name in feature_builders
    }
    common_bundle["quality_report"] = _make_json_serializable(common_bundle["quality_report"])

    if save_outputs:
        save_preprocessed_bundle(common_bundle, paths["common_dir"], "common")

    print("[run_all_preprocessing] Completed preprocessing for all configured model branches.")
    return results


if __name__ == "__main__":
    run_all_preprocessing()
