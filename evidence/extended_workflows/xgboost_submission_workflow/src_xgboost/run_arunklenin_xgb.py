"""XGB-only reproduction branch inspired by arunklenin's high-score notebook.

This is a Kaggle leaderboard-chasing side branch. It intentionally does not
touch the team's common preprocessing bundle. The source notebook's public
0.82066 submission is not pure XGBoost: its final cell ORs several external
submission files. This script keeps only the reproducible raw-CSV feature
engineering and XGB parts, then writes several submit-ready CSVs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer, StandardScaler
from xgboost import XGBClassifier

from . import config


RAW_DATA_CANDIDATES = (
    Path("/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw"),
    config.PROJECT_ROOT / "data" / "raw",
    config.PROJECT_ROOT / "archived_github_main" / "epoch-MLW-main" / "data" / "raw",
)
OUT_DIR = config.REPORTS_DIR / "submission_candidates"
TAG = "arunklenin_xgb_only"

EXPENSE_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
ENCODE_CATEGORICALS = ["HomePlanet", "cabin_deck", "Destination", "cabin_side"]
TARGET_RATES = {
    "rate520": 0.5200,
    "rate532": 0.5324,
    "rate535": 0.5350,
    "rate536": 0.5366,
}
XGB_PARAMS = {
    "colsample_bytree": 0.8498791800104656,
    "learning_rate": 0.020233442882782587,
    "max_depth": 4,
    "n_estimators": 469,
    "subsample": 0.746529796772373,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 4,
}


@dataclass
class ArunkleninModelResult:
    raw_data_dir: str
    feature_count: int
    seeds: tuple[int, ...]
    n_splits: int
    oof_accuracy_050: float
    oof_best_threshold: float
    oof_best_accuracy: float
    oof_logloss: float
    oof_auc: float
    test_positive_rate_050: float
    oof_proba_path: str
    test_proba_path: str


@dataclass
class ArunkleninSubmissionResult:
    name: str
    file: str
    metadata_path: str
    threshold: float
    target_rate_name: str
    positive_rate: float
    oof_accuracy: float
    changed_vs_umanglodaya_smote: int | None
    changed_vs_umanglodaya_seed0: int | None


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
    raise FileNotFoundError("Could not find raw Spaceship Titanic train/test CSVs")


def _bool_to_num(value: object) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return np.nan


def _last_name(value: object) -> str:
    text = str(value).strip().lower()
    parts = text.split()
    return parts[-1] if len(parts) > 1 else text


def _acc_cutoff(y_valid: np.ndarray, y_pred_valid: np.ndarray) -> float:
    fpr, _tpr, thresholds = roc_curve(y_valid, y_pred_valid)
    if len(fpr) == 0:
        return 0.5
    labels = (y_pred_valid[None, :] > thresholds[:, None]).astype(int)
    scores = (labels == y_valid[None, :]).mean(axis=1)
    return float(thresholds[int(scores.argmax())])


def _safe_power_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    temp_cols: list[str] = [column]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[[column]])
    test_scaled = scaler.transform(test[[column]])

    train[f"log_{column}"] = np.log1p(train_scaled)
    test[f"log_{column}"] = np.log1p(test_scaled)
    temp_cols.append(f"log_{column}")

    train[f"sqrt_{column}"] = np.sqrt(np.clip(train_scaled, 0, None))
    test[f"sqrt_{column}"] = np.sqrt(np.clip(test_scaled, 0, None))
    temp_cols.append(f"sqrt_{column}")

    for prefix, transformer, train_input, test_input in (
        ("bx_cx_", PowerTransformer(method="box-cox"), train_scaled + 1.0, test_scaled + 1.0),
        ("y_J_", PowerTransformer(method="yeo-johnson"), train[[column]].to_numpy(), test[[column]].to_numpy()),
    ):
        new_col = f"{prefix}{column}"
        try:
            train[new_col] = transformer.fit_transform(train_input)
            test[new_col] = transformer.transform(test_input)
        except Exception:
            train[new_col] = 0.0
            test[new_col] = 0.0
        temp_cols.append(new_col)

    for name, power in (("pow_", 0.25), ("pow2_", 0.10)):
        new_col = f"{name}{column}"
        transformer = FunctionTransformer(lambda x, p=power: np.power(np.clip(x, 0, None), p))
        train[new_col] = transformer.fit_transform(train_scaled)
        test[new_col] = transformer.transform(test_scaled)
        temp_cols.append(new_col)

    train[f"log_pow2{column}"] = np.log1p(train[f"pow2_{column}"])
    test[f"log_pow2{column}"] = np.log1p(test[f"pow2_{column}"])
    temp_cols.append(f"log_pow2{column}")

    train[temp_cols] = train[temp_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test[temp_cols] = test[temp_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pca_col = f"{column}_pca_comb"
    svd = TruncatedSVD(n_components=1, random_state=random_state)
    train[pca_col] = svd.fit_transform(train[temp_cols])
    test[pca_col] = svd.transform(test[temp_cols])
    temp_cols.append(pca_col)
    return train, test, temp_cols


def _select_best_univariate(train: pd.DataFrame, temp_cols: list[str], y: np.ndarray) -> tuple[str, list[str]]:
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores: list[tuple[str, float]] = []
    for feature in temp_cols:
        x = train[[feature]].to_numpy()
        accs = []
        for train_idx, valid_idx in kf.split(x, y):
            model = LogisticRegression(max_iter=500)
            model.fit(x[train_idx], y[train_idx])
            valid_proba = model.predict_proba(x[valid_idx])[:, 1]
            cutoff = _acc_cutoff(y[valid_idx], valid_proba)
            accs.append(accuracy_score(y[valid_idx], valid_proba > cutoff))
        scores.append((feature, float(np.mean(accs))))
    best_feature = max(scores, key=lambda item: item[1])[0]
    drop_features = [feature for feature in temp_cols if feature != best_feature]
    return best_feature, drop_features


def _add_numeric_transform_winner_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    *,
    fast: bool,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_features = [
        col
        for col in train.columns
        if col not in {"Transported", "PassengerId"} and train[col].dtype != "object" and train[col].nunique() > 10
    ]
    drop_features: list[str] = []
    for column in numeric_features:
        train, test, temp_cols = _safe_power_features(train, test, column, random_state=random_state)
        if fast:
            # Keep all variants in fast mode; it is usually more robust than doing a
            # noisy one-feature logistic selection with a shortened run.
            continue
        _best, to_drop = _select_best_univariate(train, temp_cols, y)
        drop_features.extend(to_drop)
    if drop_features:
        existing = [col for col in set(drop_features) if col in train.columns]
        train = train.drop(columns=existing)
        test = test.drop(columns=existing)
    return train, test


def _add_last_name_tfidf(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(max_features=1000)
    train_vectors = vectorizer.fit_transform(train["Last_Name"].astype(str))
    test_vectors = vectorizer.transform(test["Last_Name"].astype(str))
    n_components = min(5, max(1, train_vectors.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_tfidf = pd.DataFrame(svd.fit_transform(train_vectors), columns=[f"Last_Name_tfidf_{i}" for i in range(n_components)])
    test_tfidf = pd.DataFrame(svd.transform(test_vectors), columns=[f"Last_Name_tfidf_{i}" for i in range(n_components)])
    train = pd.concat([train.reset_index(drop=True), train_tfidf], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_tfidf], axis=1)
    return train.drop(columns=["Name", "Last_Name"]), test.drop(columns=["Name", "Last_Name"])


def _add_category_encodings(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for feature in ENCODE_CATEGORICALS:
        target_order = train.groupby(feature)["Transported"].mean().sort_values().index
        target_map = {key: idx for idx, key in enumerate(target_order)}
        train[f"{feature}_target"] = train[feature].map(target_map)
        test[f"{feature}_target"] = test[feature].map(target_map)

        counts = train[feature].value_counts().to_dict()
        train[f"{feature}_count"] = np.log1p(train[feature].map(counts))
        test[f"{feature}_count"] = np.log1p(test[feature].map(counts))

        count_labels = dict(zip(counts.keys(), np.arange(len(counts), 0, -1)))
        train[f"{feature}_count_label"] = train[feature].map(count_labels)
        test[f"{feature}_count_label"] = test[feature].map(count_labels)

        positives = train.groupby(feature)["Transported"].sum()
        totals = train.groupby(feature)["Transported"].count()
        woe = np.log1p(positives / (totals - positives).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        train[f"{feature}_WOE"] = train[feature].map(woe)
        test[f"{feature}_WOE"] = test[feature].map(woe)

        encoded_cols = [
            f"{feature}_target",
            f"{feature}_count",
            f"{feature}_count_label",
            f"{feature}_WOE",
        ]
        imputer = KNNImputer(n_neighbors=5)
        train[encoded_cols] = imputer.fit_transform(train[encoded_cols])
        test[encoded_cols] = imputer.transform(test[encoded_cols])

    return train.drop(columns=ENCODE_CATEGORICALS), test.drop(columns=ENCODE_CATEGORICALS)


def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    fast: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    train["Transported"] = train["Transported"].astype(int)

    for frame in (train, test):
        frame["group"] = frame["PassengerId"].astype(str).str[:-3].astype(int)
        cabin_parts = frame["Cabin"].astype("string").str.split("/", expand=True).reindex(columns=[0, 1, 2])
        frame["cabin_deck"] = cabin_parts[0].fillna("Missing_Deck").astype(str)
        frame["cabin_num"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        frame["cabin_side"] = cabin_parts[2].fillna("Missing_Side").astype(str)
        frame.drop(columns=["Cabin"], inplace=True)

        frame["Name"] = frame["Name"].fillna("No_Name")
        frame["Last_Name"] = frame["Name"].apply(_last_name)
        frame["VIP"] = frame["VIP"].apply(_bool_to_num)
        frame["CryoSleep"] = frame["CryoSleep"].apply(_bool_to_num)

    for frame in (train, test):
        for column in frame.select_dtypes(include="object").columns:
            if column not in {"PassengerId"}:
                frame[column] = frame[column].fillna(f"missing_{column}")

    for frame in (train, test):
        expenditure = frame[EXPENSE_COLUMNS].sum(axis=1)
        frame["CryoSleep"] = np.where(expenditure == 0, 1.0, 0.0)
        frame["VIP"] = np.where(frame["CryoSleep"] == 0, 1.0, 0.0)
        for column in EXPENSE_COLUMNS:
            frame[column] = np.where(frame["CryoSleep"] == 1, 0.0, frame[column])

    miss_cont = [
        col
        for col in train.columns
        if col not in {"Transported", "PassengerId"}
        and train[col].dtype != "object"
        and (train[col].isna().any() or test[col].isna().any())
    ]
    if miss_cont:
        imputer = KNNImputer(n_neighbors=5)
        train[miss_cont] = imputer.fit_transform(train[miss_cont])
        test[miss_cont] = imputer.transform(test[miss_cont])

    train["expenditure"] = train["VRDeck"] + train["Spa"] + train["RoomService"]
    test["expenditure"] = test["VRDeck"] + test["Spa"] + test["RoomService"]
    y = train["Transported"].to_numpy(dtype=int)

    train, test = _add_numeric_transform_winner_features(train, test, y, fast=fast, random_state=42)
    train, test = _add_last_name_tfidf(train, test)
    train, test = _add_category_encodings(train, test)

    train_ids = train["PassengerId"].copy()
    test_ids = test["PassengerId"].copy()
    train = train.drop(columns=["PassengerId"])
    test = test.drop(columns=["PassengerId"])

    feature_cols = [col for col in train.columns if col != "Transported"]
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    x_test = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)
    x_train.insert(0, "_PassengerId", train_ids.astype(str).values)
    x_test.insert(0, "_PassengerId", test_ids.astype(str).values)
    return x_train, y, x_test


def _build_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS, random_state=random_state)


def _fit_cv_average(
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    *,
    seeds: Iterable[int],
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in x_train.columns if col != "_PassengerId"]
    x = x_train[feature_cols]
    test = x_test[feature_cols]
    oof_sum = np.zeros(len(x), dtype=float)
    oof_count = np.zeros(len(x), dtype=float)
    test_preds: list[np.ndarray] = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
            model = _build_model(random_state=seed * 100 + fold)
            model.fit(x.iloc[train_idx], y[train_idx])
            oof_sum[valid_idx] += model.predict_proba(x.iloc[valid_idx])[:, 1]
            oof_count[valid_idx] += 1
            test_preds.append(model.predict_proba(test)[:, 1])
    return oof_sum / np.maximum(oof_count, 1), np.mean(test_preds, axis=0)


def _scan_best_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    scores = np.array([accuracy_score(y, proba >= threshold) for threshold in thresholds])
    idx = int(scores.argmax())
    return float(thresholds[idx]), float(scores[idx])


def _threshold_for_positive_rate(score: np.ndarray, target_rate: float) -> float:
    return float(np.quantile(score, 1.0 - target_rate, method="lower"))


def _load_reference(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return pd.read_csv(path)["Transported"].astype(bool).to_numpy()


def _write_submission(
    *,
    name: str,
    passenger_ids: pd.Series,
    pred: np.ndarray,
    threshold: float,
    target_rate_name: str,
    oof_accuracy: float,
    ref_smote: np.ndarray | None,
    ref_seed0: np.ndarray | None,
) -> ArunkleninSubmissionResult:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{name}.csv"
    metadata_path = OUT_DIR / f"{name}.json"
    pred_bool = np.asarray(pred).astype(bool)
    pd.DataFrame({"PassengerId": passenger_ids.astype(str), "Transported": pred_bool}).to_csv(file_path, index=False)
    result = ArunkleninSubmissionResult(
        name=name,
        file=str(file_path),
        metadata_path=str(metadata_path),
        threshold=float(threshold),
        target_rate_name=target_rate_name,
        positive_rate=float(pred_bool.mean()),
        oof_accuracy=float(oof_accuracy),
        changed_vs_umanglodaya_smote=int(np.sum(pred_bool != ref_smote)) if ref_smote is not None else None,
        changed_vs_umanglodaya_seed0=int(np.sum(pred_bool != ref_seed0)) if ref_seed0 is not None else None,
    )
    metadata_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
    return result


def run_arunklenin_xgb(
    data_dir: Path | None = None,
    *,
    seeds: tuple[int, ...] = (2140,),
    n_splits: int = 10,
    fast: bool = False,
) -> tuple[ArunkleninModelResult, list[ArunkleninSubmissionResult]]:
    raw_dir = _resolve_raw_data_dir(data_dir)
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")
    x_train_with_ids, y, x_test_with_ids = _prepare_features(train_df, test_df, fast=fast)
    test_ids = x_test_with_ids["_PassengerId"].copy()
    feature_count = x_train_with_ids.shape[1] - 1

    oof, test_proba = _fit_cv_average(x_train_with_ids, y, x_test_with_ids, seeds=seeds, n_splits=n_splits)
    best_t, best_acc = _scan_best_threshold(y, oof)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = config.LOGS_DIR / f"{TAG}_oof_proba.csv"
    test_path = config.LOGS_DIR / f"{TAG}_test_proba.csv"
    pd.DataFrame({"PassengerId": x_train_with_ids["_PassengerId"].astype(str), "y_proba": oof}).to_csv(
        oof_path, index=False
    )
    pd.DataFrame({"PassengerId": test_ids.astype(str), "y_proba": test_proba}).to_csv(test_path, index=False)

    model_result = ArunkleninModelResult(
        raw_data_dir=str(raw_dir),
        feature_count=int(feature_count),
        seeds=tuple(seeds),
        n_splits=int(n_splits),
        oof_accuracy_050=float(accuracy_score(y, oof >= 0.5)),
        oof_best_threshold=best_t,
        oof_best_accuracy=best_acc,
        oof_logloss=float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        oof_auc=float(roc_auc_score(y, oof)),
        test_positive_rate_050=float((test_proba >= 0.5).mean()),
        oof_proba_path=str(oof_path),
        test_proba_path=str(test_path),
    )

    ref_smote = _load_reference(OUT_DIR / "submission_umanglodaya_xgb_smote.csv")
    ref_seed0 = _load_reference(OUT_DIR / "submission_umanglodaya_xgb_smote_seed0.csv")
    thresholds = {
        "t050": 0.5,
        "best_oof": best_t,
    }
    thresholds.update({name: _threshold_for_positive_rate(test_proba, rate) for name, rate in TARGET_RATES.items()})

    submissions: list[ArunkleninSubmissionResult] = []
    for target_rate_name, threshold in thresholds.items():
        pred = test_proba >= threshold
        submissions.append(
            _write_submission(
                name=f"submission_{TAG}_{target_rate_name}",
                passenger_ids=test_ids,
                pred=pred,
                threshold=threshold,
                target_rate_name=target_rate_name,
                oof_accuracy=accuracy_score(y, oof >= threshold),
                ref_smote=ref_smote,
                ref_seed0=ref_seed0,
            )
        )

    summary_path = config.LOGS_DIR / f"{TAG}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": asdict(model_result),
                "submissions": [asdict(item) for item in submissions],
                "source": "XGB-only branch from arunklenin/space-titanic-eda-advanced-feature-engineering",
                "excluded": "final public notebook OR blend with external submissions",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return model_result, submissions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2140])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--fast", action="store_true", help="Keep all numeric transform variants instead of selecting winners.")
    args = parser.parse_args()
    model, submissions = run_arunklenin_xgb(
        data_dir=args.data_dir,
        seeds=tuple(args.seeds),
        n_splits=args.n_splits,
        fast=args.fast,
    )
    print(json.dumps({"model": asdict(model), "submissions": [asdict(item) for item in submissions]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
