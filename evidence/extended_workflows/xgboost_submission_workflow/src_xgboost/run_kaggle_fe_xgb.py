"""Kaggle-style raw-CSV XGBoost branch for leaderboard chasing.

This branch intentionally leaves the shared/common preprocessing untouched. It
builds a separate XGB-only feature table from the raw Spaceship Titanic CSVs,
using the feature-engineering patterns that repeatedly appear in higher-scoring
Kaggle notebooks:

- travel group size and within-group passenger index;
- surname/family-size signals;
- cabin deck/side/number/bin combinations;
- spend totals, log spend, no-spend flags, and CryoSleep-spend rules;
- group/surname/feature-bucket imputation before one-hot encoding;
- multi-seed CV-model averaging and public-LB positive-rate anchors.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config


RAW_DATA_CANDIDATES = (
    Path("/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw"),
    config.PROJECT_ROOT / "data" / "raw",
    config.PROJECT_ROOT / "archived_github_main" / "epoch-MLW-main" / "data" / "raw",
)
OUT_DIR = config.REPORTS_DIR / "submission_candidates"
LOG_PREFIX = "kaggle_fe_xgb"

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
BASE_NUMERIC = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
    "LogTotalSpend",
    "LuxurySpend",
    "ServiceSpend",
    "SpendNonZeroCount",
    "SpendMissingCount",
    "SpendPerAge",
    "GroupSize",
    "PassengerNo",
    "CabinNumber",
    "CabinNumberBinCode",
    "SurnameSize",
]
BASE_CATEGORICAL = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "DeckSide",
    "CabinNumberBin",
    "AgeGroup",
    "IsSolo",
    "NoSpend",
    "HasCabin",
    "HasName",
    "HomeDestination",
    "PlanetDeck",
    "CryoNoSpend",
]

PARAM_SETS = {
    "regularized": {
        "learning_rate": 0.030,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.86,
        "colsample_bytree": 0.82,
        "gamma": 0.10,
        "reg_alpha": 0.30,
        "reg_lambda": 4.0,
        "n_estimators": 900,
    },
    "public": {
        "learning_rate": 0.06672065863100594,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.9527591724824661,
        "colsample_bytree": 0.9241969052729379,
        "gamma": 0.0,
        "reg_alpha": 4.581902571574289,
        "reg_lambda": 3.0610042624477543,
        "n_estimators": 730,
    },
    "shallow": {
        "learning_rate": 0.040,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "gamma": 0.05,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0,
        "n_estimators": 950,
    },
}

TARGET_POSITIVE_RATES = {
    "rate526": 0.5260,
    "rate532": 0.5324,
    "rate536": 0.5366,
}


@dataclass
class ModelResult:
    name: str
    cv_kind: str
    seeds: tuple[int, ...]
    params: dict[str, float | int]
    oof_accuracy_050: float
    oof_best_threshold: float
    oof_best_accuracy: float
    oof_logloss: float
    oof_auc: float
    test_positive_rate_050: float
    oof_proba_path: str
    test_proba_path: str


@dataclass
class SubmissionResult:
    name: str
    file: str
    metadata_path: str
    source_model: str
    threshold: float
    target_rate_name: str
    positive_rate: float
    oof_accuracy: float
    changed_vs_v2_best: int | None
    changed_vs_anchor_a7: int | None
    changed_vs_hs10: int | None


def _resolve_raw_data_dir(preferred: Path | None = None) -> Path:
    candidates = [preferred] if preferred is not None else []
    candidates.extend(RAW_DATA_CANDIDATES)
    for data_dir in candidates:
        if data_dir is None:
            continue
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        if not train_path.exists() or not test_path.exists():
            continue
        train = pd.read_csv(train_path, nrows=3)
        test = pd.read_csv(test_path, nrows=3)
        if "Transported" in train.columns and "Transported" not in test.columns:
            return data_dir
    raise FileNotFoundError("Could not find raw Kaggle train.csv/test.csv with unlabeled test.csv")


def _split_cabin(series: pd.Series) -> pd.DataFrame:
    parts = series.astype("string").str.split("/", expand=True)
    parts = parts.reindex(columns=[0, 1, 2])
    parts.columns = ["Deck", "CabinNumber", "Side"]
    return parts


def _mode_or_nan(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode()
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _fill_by_group_mode(df: pd.DataFrame, column: str, key: str) -> None:
    modes = df.groupby(key)[column].agg(_mode_or_nan)
    df[column] = df[column].fillna(df[key].map(modes))


def _fill_numeric_by_keys(df: pd.DataFrame, column: str, keys: list[str]) -> None:
    df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
    for key in keys:
        medians = df.groupby(key, dropna=True)[column].median()
        df[column] = df[column].fillna(df[key].map(medians))
    df[column] = df[column].fillna(df[column].median())


def _prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    combined = pd.concat(
        [train_df.drop(columns=["Transported"]).assign(_is_train=True), test_df.assign(_is_train=False)],
        axis=0,
        ignore_index=True,
    )

    combined["PassengerId"] = combined["PassengerId"].astype("string")
    combined["GroupID"] = combined["PassengerId"].str.split("_").str[0]
    combined["PassengerNo"] = pd.to_numeric(combined["PassengerId"].str.split("_").str[1], errors="coerce")
    combined["GroupSize"] = combined.groupby("GroupID")["PassengerId"].transform("count")
    combined["IsSolo"] = combined["GroupSize"].eq(1)

    cabin_parts = _split_cabin(combined["Cabin"])
    combined["Deck"] = cabin_parts["Deck"]
    combined["CabinNumber"] = pd.to_numeric(cabin_parts["CabinNumber"], errors="coerce")
    combined["Side"] = cabin_parts["Side"]
    combined["HasCabin"] = combined["Cabin"].notna()
    _fill_by_group_mode(combined, "Deck", "GroupID")
    _fill_by_group_mode(combined, "Side", "GroupID")
    _fill_numeric_by_keys(combined, "CabinNumber", ["GroupID", "Deck"])

    combined["Surname"] = combined["Name"].astype("string").str.split().str[-1]
    combined["HasName"] = combined["Name"].notna()
    _fill_by_group_mode(combined, "Surname", "GroupID")
    combined["Surname"] = combined["Surname"].fillna("Unknown")
    combined["SurnameSize"] = combined.groupby("Surname")["PassengerId"].transform("count")

    for col in SPEND_COLUMNS:
        combined[f"{col}_missing"] = combined[col].isna()
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    observed_spend = combined[SPEND_COLUMNS].sum(axis=1, min_count=1)
    combined.loc[combined["CryoSleep"].eq(True), SPEND_COLUMNS] = combined.loc[
        combined["CryoSleep"].eq(True), SPEND_COLUMNS
    ].fillna(0)
    for col in SPEND_COLUMNS:
        _fill_numeric_by_keys(combined, col, ["CryoSleep", "HomePlanet", "Destination"])
    combined["TotalSpend"] = combined[SPEND_COLUMNS].sum(axis=1)
    combined["LogTotalSpend"] = np.log1p(combined["TotalSpend"])
    combined["LuxurySpend"] = combined[["FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    combined["ServiceSpend"] = combined[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    combined["SpendNonZeroCount"] = combined[SPEND_COLUMNS].gt(0).sum(axis=1)
    combined["SpendMissingCount"] = combined[[f"{col}_missing" for col in SPEND_COLUMNS]].sum(axis=1)
    combined["NoSpend"] = combined["TotalSpend"].eq(0)
    combined["SpendPerAge"] = combined["TotalSpend"] / (combined["Age"].fillna(combined["Age"].median()) + 1)

    combined.loc[combined["TotalSpend"].eq(0) & combined["CryoSleep"].isna(), "CryoSleep"] = True
    combined.loc[combined["TotalSpend"].gt(0) & combined["CryoSleep"].isna(), "CryoSleep"] = False

    for col in ["HomePlanet", "Destination", "VIP", "CryoSleep"]:
        _fill_by_group_mode(combined, col, "GroupID")
        _fill_by_group_mode(combined, col, "Surname")

    deck_planet = {"A": "Europa", "B": "Europa", "C": "Europa", "T": "Europa", "G": "Earth"}
    combined["HomePlanet"] = combined["HomePlanet"].fillna(combined["Deck"].map(deck_planet))
    for col in ["HomePlanet", "Destination", "VIP", "CryoSleep", "Deck", "Side"]:
        combined[col] = combined[col].fillna(_mode_or_nan(combined[col]))

    _fill_numeric_by_keys(combined, "Age", ["HomePlanet", "CryoSleep", "Destination", "GroupSize"])
    combined["AgeGroup"] = pd.cut(
        combined["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 100],
        labels=["child", "teen", "young", "adult", "mature", "senior"],
    ).astype("object")
    combined["CabinNumberBin"] = pd.cut(
        combined["CabinNumber"],
        bins=[-1, 300, 600, 900, 1200, 1500, 2000],
        labels=["0000_0300", "0301_0600", "0601_0900", "0901_1200", "1201_1500", "1501_2000"],
    ).astype("object")
    combined["CabinNumberBinCode"] = pd.Categorical(combined["CabinNumberBin"]).codes
    combined["DeckSide"] = combined["Deck"].astype(str) + "_" + combined["Side"].astype(str)
    combined["HomeDestination"] = combined["HomePlanet"].astype(str) + "_" + combined["Destination"].astype(str)
    combined["PlanetDeck"] = combined["HomePlanet"].astype(str) + "_" + combined["Deck"].astype(str)
    combined["CryoNoSpend"] = combined["CryoSleep"].astype(str) + "_" + combined["NoSpend"].astype(str)

    for col in BASE_CATEGORICAL:
        combined[col] = combined[col].astype("object")
    for col in BASE_NUMERIC:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    features = combined[BASE_NUMERIC + BASE_CATEGORICAL].copy()
    train_features = features.iloc[: len(train_df)].reset_index(drop=True)
    test_features = features.iloc[len(train_df) :].reset_index(drop=True)
    return train_features, test_features, combined.loc[: len(train_df) - 1, "GroupID"].astype("string")


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                BASE_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                BASE_CATEGORICAL,
            ),
        ]
    )


def _build_model(params: dict[str, float | int], random_state: int) -> XGBClassifier:
    merged = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": random_state,
        **params,
    }
    return XGBClassifier(**merged)


def _scan_best_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    accs = np.array([accuracy_score(y, proba >= threshold) for threshold in thresholds])
    idx = int(accs.argmax())
    return float(thresholds[idx]), float(accs[idx])


def _threshold_for_positive_rate(score: np.ndarray, target_rate: float) -> float:
    return float(np.quantile(score, 1.0 - target_rate, method="lower"))


def _fit_cv_average(
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    *,
    params: dict[str, float | int],
    seeds: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    seeds = tuple(seeds)
    oof_sum = np.zeros(len(x_train), dtype=float)
    oof_count = np.zeros(len(x_train), dtype=float)
    test_preds = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x_train, y)):
            preprocessor = _build_preprocessor()
            fit_train = x_train.iloc[train_idx].reset_index(drop=True)
            fit_valid = x_train.iloc[valid_idx].reset_index(drop=True)
            ready_train = preprocessor.fit_transform(fit_train)
            ready_valid = preprocessor.transform(fit_valid)
            ready_test = preprocessor.transform(x_test)
            model = _build_model(params, random_state=seed * 100 + fold)
            model.fit(ready_train, y[train_idx])
            oof_sum[valid_idx] += model.predict_proba(ready_valid)[:, 1]
            oof_count[valid_idx] += 1
            test_preds.append(model.predict_proba(ready_test)[:, 1])

    return oof_sum / np.maximum(oof_count, 1), np.mean(test_preds, axis=0)


def _load_reference_predictions() -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    candidates = {
        "v2_best": config.SUBMISSIONS_DIR / "submission_v2_best.csv",
        "anchor_a7": OUT_DIR / "submission_anchor_a7_hi52_lo45.csv",
        "hs10": OUT_DIR
        / "submission_kaggle_hs_10_prob_public536_groupfill10_legacy_public05_public15_stepwise_balanced35_stepwise_safe_te05_v230.csv",
    }
    for name, path in candidates.items():
        if path.exists():
            refs[name] = pd.read_csv(path)["Transported"].astype(bool).to_numpy()
    return refs


def _write_model_probs(
    tag: str,
    train_ids: pd.Series,
    test_ids: pd.Series,
    oof: np.ndarray,
    test: np.ndarray,
) -> tuple[Path, Path]:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = config.LOGS_DIR / f"{tag}_oof_proba.csv"
    test_path = config.LOGS_DIR / f"{tag}_test_proba.csv"
    pd.DataFrame({"PassengerId": train_ids.astype(str), "y_proba": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({"PassengerId": test_ids.astype(str), "y_proba": test}).to_csv(test_path, index=False)
    return oof_path, test_path


def _write_submission(
    *,
    name: str,
    passenger_ids: pd.Series,
    pred: np.ndarray,
    source_model: str,
    threshold: float,
    target_rate_name: str,
    oof_accuracy: float,
    refs: dict[str, np.ndarray],
) -> SubmissionResult:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{name}.csv"
    metadata_path = OUT_DIR / f"{name}.json"
    pred_bool = np.asarray(pred).astype(bool)
    pd.DataFrame({"PassengerId": passenger_ids.astype(str), "Transported": pred_bool}).to_csv(file_path, index=False)
    result = SubmissionResult(
        name=name,
        file=str(file_path),
        metadata_path=str(metadata_path),
        source_model=source_model,
        threshold=float(threshold),
        target_rate_name=target_rate_name,
        positive_rate=float(pred_bool.mean()),
        oof_accuracy=float(oof_accuracy),
        changed_vs_v2_best=int(np.sum(pred_bool != refs["v2_best"])) if "v2_best" in refs else None,
        changed_vs_anchor_a7=int(np.sum(pred_bool != refs["anchor_a7"])) if "anchor_a7" in refs else None,
        changed_vs_hs10=int(np.sum(pred_bool != refs["hs10"])) if "hs10" in refs else None,
    )
    metadata_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
    return result


def run_kaggle_fe_xgb(
    data_dir: Path | None = None,
    seeds: tuple[int, ...] = (42, 2024, 7),
) -> tuple[list[ModelResult], list[SubmissionResult]]:
    raw_dir = _resolve_raw_data_dir(data_dir)
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")
    x_train, x_test, _groups = _prepare_features(train_df, test_df)
    y = train_df["Transported"].astype(int).to_numpy()
    refs = _load_reference_predictions()

    model_results: list[ModelResult] = []
    submission_results: list[SubmissionResult] = []
    per_model_scores: dict[str, tuple[np.ndarray, np.ndarray, ModelResult]] = {}

    for param_name, params in PARAM_SETS.items():
        tag = f"{LOG_PREFIX}_{param_name}"
        oof, test = _fit_cv_average(x_train, y, x_test, params=params, seeds=seeds)
        oof_path, test_path = _write_model_probs(tag, train_df["PassengerId"], test_df["PassengerId"], oof, test)
        best_t, best_acc = _scan_best_threshold(y, oof)
        result = ModelResult(
            name=tag,
            cv_kind="StratifiedKFold multi-seed fold-model average",
            seeds=tuple(seeds),
            params=params,
            oof_accuracy_050=float(accuracy_score(y, oof >= 0.5)),
            oof_best_threshold=best_t,
            oof_best_accuracy=best_acc,
            oof_logloss=float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
            oof_auc=float(roc_auc_score(y, oof)),
            test_positive_rate_050=float((test >= 0.5).mean()),
            oof_proba_path=str(oof_path),
            test_proba_path=str(test_path),
        )
        model_results.append(result)
        per_model_scores[tag] = (oof, test, result)

    # Ensemble the new Kaggle-FE XGB family. Keep it XGB-only.
    sorted_models = sorted(model_results, key=lambda item: item.oof_best_accuracy, reverse=True)
    top_names = [item.name for item in sorted_models[:3]]
    ens_oof = np.mean([per_model_scores[name][0] for name in top_names], axis=0)
    ens_test = np.mean([per_model_scores[name][1] for name in top_names], axis=0)
    ens_t, ens_acc = _scan_best_threshold(y, ens_oof)
    ens_tag = f"{LOG_PREFIX}_ensemble"
    ens_oof_path, ens_test_path = _write_model_probs(
        ens_tag, train_df["PassengerId"], test_df["PassengerId"], ens_oof, ens_test
    )
    ens_result = ModelResult(
        name=ens_tag,
        cv_kind="mean of top Kaggle-FE XGB parameter families",
        seeds=tuple(seeds),
        params={"members": ",".join(top_names)},
        oof_accuracy_050=float(accuracy_score(y, ens_oof >= 0.5)),
        oof_best_threshold=ens_t,
        oof_best_accuracy=ens_acc,
        oof_logloss=float(log_loss(y, np.clip(ens_oof, 1e-6, 1 - 1e-6))),
        oof_auc=float(roc_auc_score(y, ens_oof)),
        test_positive_rate_050=float((ens_test >= 0.5).mean()),
        oof_proba_path=str(ens_oof_path),
        test_proba_path=str(ens_test_path),
    )
    model_results.append(ens_result)
    per_model_scores[ens_tag] = (ens_oof, ens_test, ens_result)

    chosen = [ens_result, *sorted_models[:2]]
    used_names: set[str] = set()
    for model in chosen:
        if model.name in used_names:
            continue
        used_names.add(model.name)
        oof, test, _ = per_model_scores[model.name]
        thresholds = {
            "t050": 0.5,
            "best_oof": model.oof_best_threshold,
        }
        thresholds.update({rate_name: _threshold_for_positive_rate(test, rate) for rate_name, rate in TARGET_POSITIVE_RATES.items()})
        for target_rate_name, threshold in thresholds.items():
            pred = test >= threshold
            oof_acc = accuracy_score(y, oof >= threshold)
            safe_name = model.name.replace(LOG_PREFIX + "_", "")
            submission_results.append(
                _write_submission(
                    name=f"submission_{LOG_PREFIX}_{safe_name}_{target_rate_name}",
                    passenger_ids=test_df["PassengerId"],
                    pred=pred,
                    source_model=model.name,
                    threshold=float(threshold),
                    target_rate_name=target_rate_name,
                    oof_accuracy=float(oof_acc),
                    refs=refs,
                )
            )

    summary_path = config.LOGS_DIR / f"{LOG_PREFIX}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "raw_data_dir": str(raw_dir),
                "feature_count": len(BASE_NUMERIC) + len(BASE_CATEGORICAL),
                "model_results": [asdict(item) for item in model_results],
                "submission_results": [asdict(item) for item in submission_results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return model_results, submission_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7])
    args = parser.parse_args()
    models, submissions = run_kaggle_fe_xgb(data_dir=args.data_dir, seeds=tuple(args.seeds))
    print(
        json.dumps(
            {
                "models": [asdict(item) for item in models],
                "submissions": [asdict(item) for item in submissions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
