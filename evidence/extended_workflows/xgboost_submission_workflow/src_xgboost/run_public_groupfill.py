"""Raw-CSV XGBoost branch with group-consistent fills and group/cabin features.

This is a second "from scratch" branch separate from ``run_public_style``.
It keeps the public-notebook spirit but injects ideas seen repeatedly in the
Kaggle discussion area:

- fill missing categorical values from travel-group consensus when possible
- infer ``CryoSleep`` from zero / positive spend
- add ``GroupSize`` / ``IsSolo`` / ``CabinNumber``

The goal is not to beat the main V2 pipeline on honest OOF, but to produce a
materially different XGBoost submission family that can still be audited.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from . import config


DEFAULT_RAW_DATA_DIR = Path("/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw")
DEFAULT_OUT_DIR = config.REPORTS_DIR / "submission_candidates"

NUMERIC_COLUMNS = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Expenses",
    "Age",
    "GroupSize",
    "CabinNumber",
]
CATEGORICAL_COLUMNS = [
    "CryoSleep",
    "Deck",
    "Side",
    "VIP",
    "HomePlanet",
    "Destination",
    "IsSolo",
]
GROUPFILL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "n_estimators": 900,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.3,
    "reg_lambda": 2.5,
    "gamma": 0.0,
    "n_jobs": 4,
}


@dataclass
class GroupfillOutputs:
    submission_path: str
    metadata_path: str
    oof_proba_path: str
    test_proba_path: str
    honest_oof_acc_at_050: float
    honest_oof_acc_at_threshold: float
    threshold: float
    threshold_scan: dict[str, float]
    positive_rate: float
    changed_vs_050: int
    raw_data_dir: str


def _split_cabin(series: pd.Series) -> pd.DataFrame:
    parts = series.astype("string").str.split("/", expand=True)
    parts = parts.reindex(columns=[0, 1, 2])
    parts.columns = ["Deck", "CabinNumber", "Side"]
    return parts


def _group_mode(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode()
    return modes.iloc[0] if not modes.empty else np.nan


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train_df.drop(columns=["Transported"]), test_df],
        axis=0,
        ignore_index=True,
    )
    combined["Group"] = combined["PassengerId"].astype("string").str.split("_").str[0]
    combined["Expenses"] = combined[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    ].fillna(0).sum(axis=1)

    for column in ["HomePlanet", "Destination", "VIP", "CryoSleep"]:
        group_modes = combined.groupby("Group")[column].agg(_group_mode)
        combined[column] = combined[column].fillna(combined["Group"].map(group_modes))

    combined.loc[combined["Expenses"].eq(0) & combined["CryoSleep"].isna(), "CryoSleep"] = True
    combined.loc[combined["Expenses"].gt(0) & combined["CryoSleep"].isna(), "CryoSleep"] = False

    combined["GroupSize"] = combined.groupby("Group")["PassengerId"].transform("count")
    combined["IsSolo"] = combined["GroupSize"].eq(1)

    cabin_parts = _split_cabin(combined["Cabin"])
    combined["Deck"] = cabin_parts["Deck"]
    combined["CabinNumber"] = pd.to_numeric(cabin_parts["CabinNumber"], errors="coerce")
    combined["Side"] = cabin_parts["Side"]

    features = combined[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("object")
        features.loc[features[column].isna(), column] = np.nan
    train_features = features.iloc[: len(train_df)].reset_index(drop=True)
    test_features = features.iloc[len(train_df) :].reset_index(drop=True)
    return train_features, test_features


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), NUMERIC_COLUMNS),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def _fit_transform(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = _build_preprocessor()
    train_matrix = preprocessor.fit_transform(train_features)
    test_matrix = preprocessor.transform(test_features)
    feature_names = list(preprocessor.get_feature_names_out())
    return (
        pd.DataFrame(train_matrix, columns=feature_names),
        pd.DataFrame(test_matrix, columns=feature_names),
    )


def _build_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(**GROUPFILL_PARAMS, random_state=random_state)


def _compute_honest_oof(
    train_features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> tuple[np.ndarray, dict[str, float]]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_features), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_features, y, groups)):
        x_train = train_features.iloc[train_idx].reset_index(drop=True)
        x_valid = train_features.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        ready_train, ready_valid = _fit_transform(x_train, x_valid)
        model = _build_model(random_state=100 + fold_idx)
        model.fit(ready_train, y_train)
        oof[valid_idx] = model.predict_proba(ready_valid)[:, 1]
    scan = {}
    for threshold in (0.48, 0.49, 0.50, 0.51, 0.52):
        scan[f"{threshold:.2f}"] = float(accuracy_score(y, oof >= threshold))
    return oof, scan


def _write_outputs(
    train_df_ids: pd.Series,
    passenger_ids: pd.Series,
    test_proba: np.ndarray,
    threshold: float,
    out_dir: Path,
    tag: str,
    oof_scan: dict[str, float],
    oof_proba: np.ndarray,
    y: pd.Series,
    raw_data_dir: Path,
) -> GroupfillOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"submission_{tag}"
    submission_path = out_dir / f"{stem}.csv"
    metadata_path = out_dir / f"{stem}.json"
    oof_proba_path = config.LOGS_DIR / f"{tag}_oof_proba.csv"
    test_proba_path = config.LOGS_DIR / f"{tag}_test_proba.csv"
    preds = test_proba >= threshold
    pd.DataFrame(
        {
            "PassengerId": passenger_ids.astype(str),
            "Transported": preds.astype(bool),
        }
    ).to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "PassengerId": train_df_ids.astype(str),
            "y_proba": np.asarray(oof_proba),
        }
    ).to_csv(oof_proba_path, index=False)
    pd.DataFrame(
        {
            "PassengerId": passenger_ids.astype(str),
            "y_proba": np.asarray(test_proba),
        }
    ).to_csv(test_proba_path, index=False)
    base_preds = test_proba >= 0.50
    metadata = GroupfillOutputs(
        submission_path=str(submission_path),
        metadata_path=str(metadata_path),
        oof_proba_path=str(oof_proba_path),
        test_proba_path=str(test_proba_path),
        honest_oof_acc_at_050=float(accuracy_score(y, oof_proba >= 0.50)),
        honest_oof_acc_at_threshold=float(accuracy_score(y, oof_proba >= threshold)),
        threshold=float(threshold),
        threshold_scan=oof_scan,
        positive_rate=float(preds.mean()),
        changed_vs_050=int(np.sum(preds != base_preds)),
        raw_data_dir=str(raw_data_dir),
    )
    metadata_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n")
    return metadata


def run_public_groupfill(
    data_dir: Path = DEFAULT_RAW_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    threshold: float = 0.52,
    tag: str = "public_groupfill_t052",
) -> GroupfillOutputs:
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_features, test_features = _prepare_features(train_df, test_df)
    y = train_df["Transported"].astype(int)
    groups = train_df["PassengerId"].astype("string").str.split("_").str[0]
    oof_proba, oof_scan = _compute_honest_oof(train_features, y, groups)

    full_train_ready, full_test_ready = _fit_transform(train_features, test_features)
    model = _build_model(random_state=2026)
    model.fit(full_train_ready, y)
    test_proba = model.predict_proba(full_test_ready)[:, 1]

    return _write_outputs(
        train_df_ids=train_df["PassengerId"],
        passenger_ids=test_df["PassengerId"],
        test_proba=test_proba,
        threshold=threshold,
        out_dir=out_dir,
        tag=tag,
        oof_scan=oof_scan,
        oof_proba=oof_proba,
        y=y,
        raw_data_dir=data_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--tag", type=str, default="public_groupfill_t052")
    args = parser.parse_args()
    outputs = run_public_groupfill(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        threshold=args.threshold,
        tag=args.tag,
    )
    print(json.dumps(asdict(outputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
