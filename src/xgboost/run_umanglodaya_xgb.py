"""Exact-style rebuild of umanglodaya's public Spaceship Titanic XGBoost.

This is a leaderboard-chasing side branch. It does not touch the team's common
preprocessing bundle. The implementation follows the public Kaggle notebook's
actual final path more closely than ``run_public_style.py``:

- raw train/test concatenation;
- CryoSleep spend zeroing and Expenses feature;
- PassengerId group/room guide fills for Cabin/VIP/HomePlanet/Destination;
- Cabin deck/side split;
- mean/mode imputation + one-hot encoding;
- permutation-importance drop list from the notebook;
- SMOTE balancing before fitting the final XGB model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from . import config

try:
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:
    class SMOTE:  # type: ignore[no-redef]
        """Minimal local SMOTE fallback for rerunning this archived branch.

        The original high-score run used ``imblearn.SMOTE``.  This fallback
        keeps the same call surface when imblearn is unavailable; exact
        synthetic rows may differ slightly from imblearn's implementation.
        """

        def __init__(
            self,
            sampling_strategy: float = 1,
            random_state: int | None = None,
            k_neighbors: int = 5,
        ) -> None:
            self.sampling_strategy = sampling_strategy
            self.random_state = random_state
            self.k_neighbors = k_neighbors

        def fit_resample(self, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
            rng = np.random.RandomState(self.random_state)
            y_arr = np.asarray(y, dtype="int8")
            labels, counts = np.unique(y_arr, return_counts=True)
            if len(labels) != 2:
                return x.copy(), y.copy()

            minority_label = labels[int(np.argmin(counts))]
            majority_count = int(counts.max())
            target_minority_count = int(round(majority_count * self.sampling_strategy))
            minority_idx = np.flatnonzero(y_arr == minority_label)
            n_new = target_minority_count - len(minority_idx)
            if n_new <= 0 or len(minority_idx) < 2:
                return x.copy(), y.copy()

            x_np = x.to_numpy(dtype="float64", copy=True)
            minority_x = x_np[minority_idx]
            n_neighbors = min(self.k_neighbors + 1, len(minority_idx))
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(minority_x)
            neighbor_indices = nn.kneighbors(minority_x, return_distance=False)[:, 1:]

            flat_indices = rng.randint(low=0, high=neighbor_indices.size, size=n_new)
            base_rows = np.floor_divide(flat_indices, neighbor_indices.shape[1])
            neighbor_cols = np.mod(flat_indices, neighbor_indices.shape[1])
            neighbor_rows = neighbor_indices[base_rows, neighbor_cols]
            gaps = rng.uniform(size=n_new)[:, np.newaxis]
            synthetic = minority_x[base_rows] + gaps * (minority_x[neighbor_rows] - minority_x[base_rows])

            x_resampled = pd.DataFrame(np.vstack([x_np, synthetic]), columns=x.columns)
            y_resampled = pd.Series(np.concatenate([y_arr, np.full(n_new, minority_label, dtype="int8")]))
            return x_resampled, y_resampled


RAW_DATA_CANDIDATES = (
    Path("/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw"),
    config.PROJECT_ROOT / "data" / "raw",
    config.PROJECT_ROOT / "00_GitHub主线_已提交" / "epoch-MLW-main" / "data" / "raw",
)
OUT_DIR = config.REPORTS_DIR / "submission_candidates"
TAG = "umanglodaya_xgb_smote"

EXPENSE_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM_COLS = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
CAT_COLS = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
DROP_LIST = [
    "ShoppingMall",
    "Age",
    "CryoSleep_True",
    "HomePlanet_Earth",
    "HomePlanet_Europa",
    "VIP_True",
    "HomePlanet_Mars",
    "Destination_PSO J318.5-22",
    "VIP_False",
    "Destination_55 Cancri e",
    "FoodCourt",
    "Destination_TRAPPIST-1e",
]
PARAMS_XGB_BEST = {
    "reg_lambda": 3.0610042624477543,
    "reg_alpha": 4.581902571574289,
    "colsample_bytree": 0.9241969052729379,
    "subsample": 0.9527591724824661,
    "learning_rate": 0.06672065863100594,
    "n_estimators": 730,
    "max_depth": 5,
    "min_child_weight": 1,
    "num_parallel_tree": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 4,
}


@dataclass
class UmanglodayaOutputs:
    submission_path: str
    metadata_path: str
    oof_proba_path: str
    test_proba_path: str
    oof_accuracy_050: float
    oof_accuracy_best: float
    oof_best_threshold: float
    positive_rate_050: float
    changed_vs_public_no_smote: int | None
    changed_vs_hs10: int | None
    raw_data_dir: str


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
        train = pd.read_csv(train_path, nrows=2)
        test = pd.read_csv(test_path, nrows=2)
        if "Transported" in train.columns and "Transported" not in test.columns:
            return data_dir
    raise FileNotFoundError("Could not find raw Spaceship Titanic train/test CSVs")


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prepare_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_test = pd.concat([train_df, test_df.assign(Transported=np.nan)], axis=0, ignore_index=True)

    train_test.loc[train_test["CryoSleep"].eq(True), EXPENSE_COLUMNS] = 0
    train_test["Expenses"] = train_test[EXPENSE_COLUMNS].sum(axis=1)
    train_test.loc[train_test["Expenses"].eq(0) & train_test["CryoSleep"].isna(), "CryoSleep"] = True

    train_test["Room"] = train_test["PassengerId"].astype(str).str[:4]
    for column in ["Cabin", "VIP", "HomePlanet", "Destination"]:
        guide = train_test[["Room", column]].dropna().drop_duplicates("Room")
        train_test = train_test.merge(guide, how="left", on="Room", suffixes=("", "_y"))
        train_test[column] = train_test[column].fillna(train_test[f"{column}_y"])
        train_test = train_test.drop(columns=[f"{column}_y"])

    cabin_parts = train_test["Cabin"].astype("string").str.split("/", expand=True)
    cabin_parts = cabin_parts.reindex(columns=[0, 1, 2])
    train_test["Cabin_1"] = cabin_parts[0]
    train_test["Cabin_2"] = cabin_parts[1]
    train_test["Cabin_3"] = cabin_parts[2]

    work = train_test[NUM_COLS + CAT_COLS + ["Transported"]].copy()
    work[NUM_COLS] = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(work[NUM_COLS]), columns=NUM_COLS)
    for column in CAT_COLS:
        work[column] = work[column].astype("object").where(work[column].notna(), np.nan)
    work[CAT_COLS] = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(work[CAT_COLS]), columns=CAT_COLS)

    encoder = _one_hot_encoder()
    encoded = pd.DataFrame(encoder.fit_transform(work[CAT_COLS]), columns=encoder.get_feature_names_out(CAT_COLS))
    work = pd.concat([work.drop(columns=CAT_COLS), encoded], axis=1)

    train_ready = work[work["Transported"].notna()].copy()
    y = train_ready["Transported"].astype(int)
    x = train_ready.drop(columns=["Transported"])
    x_test = work[work["Transported"].isna()].drop(columns=["Transported"]).reset_index(drop=True)
    drop_existing = [column for column in DROP_LIST if column in x.columns]
    x = x.drop(columns=drop_existing)
    x_test = x_test.drop(columns=drop_existing)
    return x.reset_index(drop=True), y.reset_index(drop=True), x_test


def _fit_predict_smote(x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, random_state: int) -> np.ndarray:
    x_fit, y_fit = shuffle(x, y, random_state=random_state)
    smote = SMOTE(sampling_strategy=1, random_state=random_state)
    x_sm, y_sm = smote.fit_resample(x_fit, y_fit)
    model = XGBClassifier(**PARAMS_XGB_BEST, random_state=random_state)
    model.fit(x_sm, y_sm)
    return model.predict_proba(x_test)[:, 1]


def _compute_oof(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    oof = np.zeros(len(x), dtype=float)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        smote = SMOTE(sampling_strategy=1, random_state=42 + fold)
        x_sm, y_sm = smote.fit_resample(x_train, y_train)
        model = XGBClassifier(**PARAMS_XGB_BEST, random_state=42 + fold)
        model.fit(x_sm, y_sm)
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
    return oof


def _scan_threshold(y: pd.Series, oof: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    accs = np.array([accuracy_score(y, oof >= threshold) for threshold in thresholds])
    idx = int(accs.argmax())
    return float(thresholds[idx]), float(accs[idx])


def _load_reference(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return pd.read_csv(path)["Transported"].astype(bool).to_numpy()


def run_umanglodaya_xgb(data_dir: Path | None = None, seeds: tuple[int, ...] = (1, 42, 2024)) -> UmanglodayaOutputs:
    raw_dir = _resolve_raw_data_dir(data_dir)
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")
    x, y, x_test = _prepare_matrix(train_df, test_df)

    oof = _compute_oof(x, y)
    seed_probas = {seed: _fit_predict_smote(x, y, x_test, seed) for seed in seeds}
    test_proba = np.mean(list(seed_probas.values()), axis=0)
    best_t, best_acc = _scan_threshold(y, oof)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUT_DIR / f"submission_{TAG}.csv"
    metadata_path = OUT_DIR / f"submission_{TAG}.json"
    oof_proba_path = config.LOGS_DIR / f"{TAG}_oof_proba.csv"
    test_proba_path = config.LOGS_DIR / f"{TAG}_test_proba.csv"

    pred = test_proba >= 0.5
    pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(str), "Transported": pred.astype(bool)}).to_csv(
        submission_path, index=False
    )
    pd.DataFrame({"PassengerId": train_df["PassengerId"].astype(str), "y_proba": oof}).to_csv(
        oof_proba_path, index=False
    )
    pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(str), "y_proba": test_proba}).to_csv(
        test_proba_path, index=False
    )

    public_no_smote = _load_reference(OUT_DIR / "submission_public81599_style.csv")
    hs10 = _load_reference(
        OUT_DIR
        / "submission_kaggle_hs_10_prob_public536_groupfill10_legacy_public05_public15_stepwise_balanced35_stepwise_safe_te05_v230.csv"
    )
    anchor = _load_reference(OUT_DIR / "submission_anchor_a7_hi52_lo45.csv")
    for seed, seed_proba in seed_probas.items():
        seed_pred = seed_proba >= 0.5
        seed_name = f"submission_{TAG}_seed{seed}"
        seed_submission_path = OUT_DIR / f"{seed_name}.csv"
        seed_metadata_path = OUT_DIR / f"{seed_name}.json"
        seed_test_proba_path = config.LOGS_DIR / f"{TAG}_seed{seed}_test_proba.csv"
        pd.DataFrame(
            {"PassengerId": test_df["PassengerId"].astype(str), "Transported": seed_pred.astype(bool)}
        ).to_csv(seed_submission_path, index=False)
        pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(str), "y_proba": seed_proba}).to_csv(
            seed_test_proba_path, index=False
        )
        seed_metadata = {
            "file": str(seed_submission_path),
            "test_proba_path": str(seed_test_proba_path),
            "source": "umanglodaya exact-style XGB + SMOTE single seed",
            "seed": seed,
            "positive_rate": float(seed_pred.mean()),
            "changed_vs_public_no_smote": int(np.sum(seed_pred != public_no_smote))
            if public_no_smote is not None
            else None,
            "changed_vs_hs10": int(np.sum(seed_pred != hs10)) if hs10 is not None else None,
            "changed_vs_anchor_a7_hi52_lo45": int(np.sum(seed_pred != anchor)) if anchor is not None else None,
        }
        seed_metadata_path.write_text(json.dumps(seed_metadata, ensure_ascii=False, indent=2) + "\n")
    outputs = UmanglodayaOutputs(
        submission_path=str(submission_path),
        metadata_path=str(metadata_path),
        oof_proba_path=str(oof_proba_path),
        test_proba_path=str(test_proba_path),
        oof_accuracy_050=float(accuracy_score(y, oof >= 0.5)),
        oof_accuracy_best=best_acc,
        oof_best_threshold=best_t,
        positive_rate_050=float(pred.mean()),
        changed_vs_public_no_smote=int(np.sum(pred != public_no_smote)) if public_no_smote is not None else None,
        changed_vs_hs10=int(np.sum(pred != hs10)) if hs10 is not None else None,
        raw_data_dir=str(raw_dir),
    )
    metadata_path.write_text(json.dumps(asdict(outputs), ensure_ascii=False, indent=2) + "\n")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 2024])
    args = parser.parse_args()
    outputs = run_umanglodaya_xgb(data_dir=args.data_dir, seeds=tuple(args.seeds))
    print(json.dumps(asdict(outputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
