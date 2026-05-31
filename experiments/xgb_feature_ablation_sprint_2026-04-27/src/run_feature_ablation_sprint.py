from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


SPRINT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = LOCAL_ROOT.parent
MAIN_REPO = PROJECT_ROOT / "00_GitHub主线_已提交" / "epoch-MLW-main"

COMMON_BUNDLE = LOCAL_ROOT / "processed" / "common" / "preprocessed_common.joblib"
XGB_BUNDLE = LOCAL_ROOT / "processed" / "xgboost" / "preprocessed_xgboost.joblib"
OLD_LOGS = LOCAL_ROOT / "reports" / "xgboost" / "logs"

REPORTS_DIR = SPRINT_ROOT / "reports"
SUBMISSIONS_DIR = SPRINT_ROOT / "submissions"
ARCHIVE_INPUTS_DIR = SPRINT_ROOT / "archive_inputs"

SPEND_COLUMNS = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
CATEGORICAL_COLUMNS = (
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
AUDIT_DROP_COLUMNS = ("PassengerId", "GroupID", "Cabin", "Name", "Surname")
TARGET_RATES = {
    "rate5127_a7": 0.5127425765723638,
    "rate517_a6": 0.5167173252279635,
    "rate5324_anchor": 0.5323825111059154,
    "rate5352_umang": 0.5351882160392799,
    "rate5366_public": 0.53659,
}
SEEDS = (2024, 17, 7, 42, 88)

BASE_PARAMS = {
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
    "n_estimators": 2500,
    "n_jobs": 4,
    "random_state": 42,
    "enable_categorical": True,
    "verbosity": 0,
}
KAGGLE_SMOTE_PARAMS = {
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
    "verbosity": 0,
}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    mode: str
    groups: tuple[str, ...] = ()
    description: str = ""


@dataclass
class EvalResult:
    name: str
    description: str
    n_features: int
    oof_acc_050: float
    oof_acc_best: float
    oof_best_threshold: float
    oof_logloss: float
    oof_auc: float
    mean_best_iter: float
    positive_rate_050: float
    test_positive_rate_best: float
    elapsed_seconds: float
    oof: np.ndarray
    test: np.ndarray
    y: np.ndarray
    passenger_ids: list[str]
    feature_names: list[str]
    importance: pd.DataFrame
    source: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _load_best_params() -> dict[str, Any]:
    params = dict(BASE_PARAMS)
    best_params_path = OLD_LOGS / "best_params.json"
    if best_params_path.exists():
        payload = json.loads(best_params_path.read_text(encoding="utf-8"))
        params.update(payload.get("best_params", {}))
    return params


def _write_run_metadata() -> None:
    ARCHIVE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    git_head = ""
    git_status = ""
    head_path = ARCHIVE_INPUTS_DIR / "github_main_head.txt"
    status_path = ARCHIVE_INPUTS_DIR / "github_main_status_before.txt"
    if head_path.exists():
        git_head = head_path.read_text(encoding="utf-8").strip()
    if status_path.exists():
        git_status = status_path.read_text(encoding="utf-8").strip()
    meta = {
        "sprint_root": SPRINT_ROOT,
        "team_preprocessing_only": True,
        "common_bundle": COMMON_BUNDLE,
        "xgb_bundle": XGB_BUNDLE,
        "old_logs_used_for_blend": {
            "A6_oof": OLD_LOGS / "A6_oof.csv",
            "A6_test": OLD_LOGS / "A6_test_proba.csv",
            "A7_oof": OLD_LOGS / "A7_oof.csv",
            "A7_test": OLD_LOGS / "A7_test_proba.csv",
        },
        "github_main_head": git_head,
        "github_main_status_before": git_status,
        "forbidden_training_sources": [
            "raw Kaggle-template preprocessing branch",
            "archived umanglodaya raw CSV submissions as model input",
        ],
    }
    (ARCHIVE_INPUTS_DIR / "source_audit.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_common() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    bundle = joblib.load(COMMON_BUNDLE)
    train = bundle["common_train"].reset_index(drop=True).copy()
    test = bundle["common_test"].reset_index(drop=True).copy()
    y = pd.Series(bundle["y_train"]).reset_index(drop=True).astype("int8")
    groups = train["GroupID"].reset_index(drop=True).astype("string")
    passenger_ids = test["PassengerId"].astype("string").tolist()
    return train, test, y, groups, passenger_ids


def _add_model_side_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SPEND_COLUMNS:
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(out[col].astype("float64").fillna(0.0))
    if "TotalSpend" in out.columns:
        out["log1p_TotalSpend"] = np.log1p(out["TotalSpend"].astype("float64").fillna(0.0))
    if "LuxurySpend" in out.columns:
        out["log1p_LuxurySpend"] = np.log1p(out["LuxurySpend"].astype("float64").fillna(0.0))
    if "BasicSpend" in out.columns:
        out["log1p_BasicSpend"] = np.log1p(out["BasicSpend"].astype("float64").fillna(0.0))
    if "SpendPerActiveCategory" in out.columns:
        out["log1p_SpendPerActiveCategory"] = np.log1p(
            out["SpendPerActiveCategory"].astype("float64").fillna(0.0)
        )
    if {"TotalSpend", "Age"} <= set(out.columns):
        out["SpendPerAge"] = out["TotalSpend"].astype("float64") / (out["Age"].astype("float64") + 1.0)
    if {"TotalSpend", "GroupSize"} <= set(out.columns):
        out["SpendPerGroupMember"] = out["TotalSpend"].astype("float64") / out["GroupSize"].astype("float64").clip(lower=1)
    if {"LuxurySpend", "BasicSpend"} <= set(out.columns):
        out["LuxuryMinusBasic"] = out["LuxurySpend"].astype("float64") - out["BasicSpend"].astype("float64")
    if {"CryoSleep", "TotalSpend"} <= set(out.columns):
        cryo_true = out["CryoSleep"].astype("string").eq("True")
        out["CryoSleepSpendAnomaly"] = (cryo_true & (out["TotalSpend"].astype("float64") > 0)).astype("int8")
    miss_cols = [col for col in out.columns if col.endswith("Missing")]
    if miss_cols:
        out["MissingCount"] = out[miss_cols].sum(axis=1).astype("int16")
    return out


def _align_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    te = test.copy()
    for col in tr.columns:
        if col not in te.columns:
            continue
        if col in CATEGORICAL_COLUMNS or tr[col].dtype.name in {"string", "object"}:
            left = tr[col].astype("string").fillna("__MISSING__")
            right = te[col].astype("string").fillna("__MISSING__")
            levels = sorted(set(left.astype(str)).union(set(right.astype(str))))
            tr[col] = pd.Categorical(left.astype(str), categories=levels)
            te[col] = pd.Categorical(right.astype(str), categories=levels)
        elif str(tr[col].dtype).startswith("Int"):
            tr[col] = tr[col].astype("int64")
            te[col] = te[col].astype("int64")
        elif tr[col].dtype == bool:
            tr[col] = tr[col].astype("int8")
            te[col] = te[col].astype("int8")
        else:
            tr[col] = pd.to_numeric(tr[col], errors="coerce").astype("float64")
            te[col] = pd.to_numeric(te[col], errors="coerce").astype("float64")
    return tr, te


def _build_common_native_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    train, test, y, groups, passenger_ids = _load_common()
    train = _add_model_side_features(train)
    test = _add_model_side_features(test)
    drop = [col for col in AUDIT_DROP_COLUMNS if col in train.columns]
    train = train.drop(columns=drop)
    test = test.drop(columns=[col for col in drop if col in test.columns])
    common_cols = [col for col in train.columns if col in test.columns]
    train, test = train[common_cols], test[common_cols]
    train, test = _align_categories(train, test)
    return train, test, y, groups, passenger_ids


def _group_columns(columns: list[str], group: str) -> list[str]:
    cols = set(columns)
    exact: set[str] = set()
    if group == "spend_core":
        exact = set(SPEND_COLUMNS) | {"TotalSpend", "IsZeroSpend", "SpendCount", "log1p_TotalSpend"}
        exact |= {f"log1p_{col}" for col in SPEND_COLUMNS}
    elif group == "spend_structure":
        exact = {
            "LuxurySpend",
            "BasicSpend",
            "LuxuryShare",
            "SpendPerActiveCategory",
            "HasAnyLuxurySpend",
            "log1p_LuxurySpend",
            "log1p_BasicSpend",
            "log1p_SpendPerActiveCategory",
            "LuxuryMinusBasic",
        }
    elif group == "cabin":
        exact = {"Deck", "Side", "DeckSide", "CabinNum", "CabinNumBin"}
    elif group == "group":
        exact = {"GroupSize", "GroupMemberNo", "IsSolo", "IsMultiPassengerGroup", "GroupMemberIsLeader", "SpendPerGroupMember"}
    elif group == "age":
        exact = {"Age", "AgeGroup", "IsChild", "IsSenior", "AgeWasOutOfRange", "SpendPerAge"}
    elif group == "broad_categorical":
        exact = {"HomePlanet", "Destination", "VIP", "HomePlanetDestination"}
    elif group == "surname":
        exact = {"SurnameFreq"}
    elif group == "missing_flags":
        return [col for col in columns if col.endswith("Missing") or col == "MissingCount"]
    elif group == "cryo":
        exact = {"CryoSleep", "CryoSleepSpendAnomaly"}
    return [col for col in columns if col in exact and col in cols]


def _compact_keep(columns: list[str], variant: str) -> list[str]:
    keep: set[str] = set()
    keep.update(_group_columns(columns, "cryo"))
    keep.update({"RoomService", "Spa", "VRDeck", "TotalSpend", "IsZeroSpend", "SpendCount"})
    keep.update({"log1p_RoomService", "log1p_Spa", "log1p_VRDeck", "log1p_TotalSpend"})
    keep.update({"Deck", "Side", "DeckSide", "CabinNumBin"})
    if variant in {"compact_plus_group", "compact_plus_group_age"}:
        keep.update({"GroupSize", "IsSolo", "IsMultiPassengerGroup", "SpendPerGroupMember"})
    if variant in {"compact_plus_age", "compact_plus_group_age"}:
        keep.update({"AgeGroup", "IsChild", "IsSenior"})
    if variant == "compact_spend_luxury":
        keep.update({"LuxurySpend", "BasicSpend", "LuxuryShare", "log1p_LuxurySpend", "log1p_BasicSpend"})
    return [col for col in columns if col in keep]


def _view_specs() -> list[ViewSpec]:
    return [
        ViewSpec("full_common_native", "drop", (), "Full team common feature view with model-side log/interactions"),
        ViewSpec("drop_missing_flags", "drop", ("missing_flags",), "Drop all missing indicators and MissingCount"),
        ViewSpec("drop_surname", "drop", ("surname",), "Drop SurnameFreq"),
        ViewSpec("drop_cabinnum", "drop_exact", ("CabinNum",), "Drop raw CabinNum only"),
        ViewSpec("drop_vip", "drop_exact", ("VIP",), "Drop VIP only"),
        ViewSpec("drop_homeplanetdestination", "drop_exact", ("HomePlanetDestination",), "Drop HomePlanetDestination interaction"),
        ViewSpec("drop_age", "drop", ("age",), "Drop age family"),
        ViewSpec("drop_groupmemberno", "drop_exact", ("GroupMemberNo",), "Drop GroupMemberNo only"),
        ViewSpec("drop_spend_structure", "drop", ("spend_structure",), "Drop spend structure features"),
        ViewSpec("drop_broad_categorical", "drop", ("broad_categorical",), "Drop HomePlanet/Destination/VIP/HomePlanetDestination"),
        ViewSpec("drop_group", "drop", ("group",), "Drop passenger group structure"),
        ViewSpec("drop_cabin", "drop", ("cabin",), "Drop cabin deck/side/bin family"),
        ViewSpec("drop_spend_core", "drop", ("spend_core",), "Drop original/core spend features"),
        ViewSpec("compact_highscore", "keep", ("compact_highscore",), "High-score inspired compact view from team common output"),
        ViewSpec("compact_plus_group", "keep", ("compact_plus_group",), "Compact view plus group signals"),
        ViewSpec("compact_plus_age", "keep", ("compact_plus_age",), "Compact view plus age signals"),
        ViewSpec("compact_plus_group_age", "keep", ("compact_plus_group_age",), "Compact view plus group and age signals"),
        ViewSpec("compact_spend_luxury", "keep", ("compact_spend_luxury",), "Compact view plus luxury/basic spend structure"),
    ]


def _apply_view(X: pd.DataFrame, X_test: pd.DataFrame, spec: ViewSpec) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cols = list(X.columns)
    if spec.mode == "drop":
        drops: list[str] = []
        for group in spec.groups:
            drops.extend(_group_columns(cols, group))
        drops = sorted(set(drops))
        keep = [col for col in cols if col not in set(drops)]
    elif spec.mode == "drop_exact":
        drops = [col for col in spec.groups if col in cols]
        keep = [col for col in cols if col not in set(drops)]
    elif spec.mode == "keep":
        keep = _compact_keep(cols, spec.groups[0])
        drops = [col for col in cols if col not in set(keep)]
    else:
        raise ValueError(f"Unknown view mode: {spec.mode}")
    if not keep:
        raise ValueError(f"Feature view {spec.name} has no features")
    return X[keep].copy(), X_test[keep].copy(), drops


def _scan_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    accs = np.array([accuracy_score(y, proba >= t) for t in thresholds])
    idx = int(accs.argmax())
    return float(thresholds[idx]), float(accs[idx])


def _fit_native_cv(
    name: str,
    description: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    passenger_ids: list[str],
    params: dict[str, Any],
) -> EvalResult:
    t0 = time.perf_counter()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype="float32")
    test_sum = np.zeros(len(X_test), dtype="float32")
    best_iterations: list[int] = []
    importance_rows: list[pd.DataFrame] = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(np.zeros(len(y)), y, groups)):
        model = XGBClassifier(**params)
        model.set_params(callbacks=[EarlyStopping(rounds=150, save_best=True)])
        Xt = X.iloc[train_idx].reset_index(drop=True)
        Xv = X.iloc[valid_idx].reset_index(drop=True)
        yt = y.iloc[train_idx].reset_index(drop=True)
        yv = y.iloc[valid_idx].reset_index(drop=True)
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        oof[valid_idx] = model.predict_proba(Xv)[:, 1]
        test_sum += model.predict_proba(X_test[Xt.columns])[:, 1].astype("float32")
        best_iterations.append(int(getattr(model, "best_iteration", params.get("n_estimators", 0))))
        booster = model.get_booster()
        feature_names = list(booster.feature_names or Xt.columns)
        raw_gain = booster.get_score(importance_type="gain")
        raw_weight = booster.get_score(importance_type="weight")
        raw_cover = booster.get_score(importance_type="cover")
        importance_rows.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "gain": [raw_gain.get(col, 0.0) for col in feature_names],
                    "weight": [raw_weight.get(col, 0.0) for col in feature_names],
                    "cover": [raw_cover.get(col, 0.0) for col in feature_names],
                }
            )
        )
    test = test_sum / 5.0
    y_arr = y.to_numpy(dtype="int8")
    best_threshold, best_acc = _scan_threshold(y_arr, oof)
    importance = (
        pd.concat(importance_rows)
        .groupby("feature", as_index=False)[["gain", "weight", "cover"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    return EvalResult(
        name=name,
        description=description,
        n_features=int(X.shape[1]),
        oof_acc_050=float(accuracy_score(y_arr, oof >= 0.5)),
        oof_acc_best=best_acc,
        oof_best_threshold=best_threshold,
        oof_logloss=float(log_loss(y_arr, np.clip(oof, 1e-6, 1 - 1e-6))),
        oof_auc=float(roc_auc_score(y_arr, oof)),
        mean_best_iter=float(np.mean(best_iterations)),
        positive_rate_050=float(np.mean(test >= 0.5)),
        test_positive_rate_best=float(np.mean(test >= best_threshold)),
        elapsed_seconds=float(time.perf_counter() - t0),
        oof=oof,
        test=test.astype("float32"),
        y=y_arr,
        passenger_ids=passenger_ids,
        feature_names=list(X.columns),
        importance=importance,
        source="team common bundle + native categorical XGBoost",
    )


def _one_hot_numeric(X: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X, X_test], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined, dummy_na=False, dtype=np.float32)
    train_encoded = encoded.iloc[: len(X)].reset_index(drop=True)
    test_encoded = encoded.iloc[len(X):].reset_index(drop=True)
    return train_encoded, test_encoded


def _smote_resample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    k_neighbors: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y, dtype="int8")
    labels, counts = np.unique(y_arr, return_counts=True)
    minority_label = labels[int(np.argmin(counts))]
    majority_count = int(counts.max())
    minority_idx = np.flatnonzero(y_arr == minority_label)
    n_new = majority_count - len(minority_idx)
    if n_new <= 0 or len(minority_idx) < 2:
        return X.copy(), y.copy()
    x_np = X.to_numpy(dtype="float32", copy=True)
    minority_x = x_np[minority_idx]
    n_neighbors = min(k_neighbors + 1, len(minority_idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(minority_x)
    neighbor_indices = nn.kneighbors(minority_x, return_distance=False)[:, 1:]
    flat_indices = rng.randint(low=0, high=neighbor_indices.size, size=n_new)
    base_rows = np.floor_divide(flat_indices, neighbor_indices.shape[1])
    neighbor_cols = np.mod(flat_indices, neighbor_indices.shape[1])
    neighbor_rows = neighbor_indices[base_rows, neighbor_cols]
    gaps = rng.uniform(size=n_new)[:, np.newaxis].astype("float32")
    synthetic = minority_x[base_rows] + gaps * (minority_x[neighbor_rows] - minority_x[base_rows])
    x_resampled = pd.DataFrame(np.vstack([x_np, synthetic]), columns=X.columns)
    y_resampled = pd.Series(np.concatenate([y_arr, np.full(n_new, minority_label, dtype="int8")]))
    return x_resampled, y_resampled


def _fit_smote_probability_set(
    base: EvalResult,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    k_neighbors: int,
) -> EvalResult:
    X_num, X_test_num = _one_hot_numeric(X, X_test)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_by_seed: list[np.ndarray] = []
    test_by_seed: list[np.ndarray] = []
    t0 = time.perf_counter()
    y_arr = y.to_numpy(dtype="int8")
    for seed in SEEDS:
        oof = np.zeros(len(y), dtype="float32")
        test_sum = np.zeros(len(X_test_num), dtype="float32")
        for fold, (train_idx, valid_idx) in enumerate(folds.split(np.zeros(len(y)), y, groups)):
            Xt = X_num.iloc[train_idx].reset_index(drop=True)
            Xv = X_num.iloc[valid_idx].reset_index(drop=True)
            yt = y.iloc[train_idx].reset_index(drop=True)
            smote_seed = int(seed + fold)
            X_sm, y_sm = _smote_resample(Xt, yt, random_state=smote_seed, k_neighbors=k_neighbors)
            model = XGBClassifier(**KAGGLE_SMOTE_PARAMS, random_state=smote_seed)
            model.fit(X_sm, y_sm, verbose=False)
            oof[valid_idx] = model.predict_proba(Xv)[:, 1]
            test_sum += model.predict_proba(X_test_num)[:, 1].astype("float32")
        X_fit, y_fit = shuffle(X_num, y, random_state=seed)
        X_sm_full, y_sm_full = _smote_resample(X_fit, y_fit, random_state=seed, k_neighbors=k_neighbors)
        final_model = XGBClassifier(**KAGGLE_SMOTE_PARAMS, random_state=int(seed))
        final_model.fit(X_sm_full, y_sm_full, verbose=False)
        oof_by_seed.append(oof)
        test_by_seed.append(final_model.predict_proba(X_test_num)[:, 1].astype("float32"))
    oof_mean = np.mean(np.stack(oof_by_seed), axis=0).astype("float32")
    test_mean = np.mean(np.stack(test_by_seed), axis=0).astype("float32")
    threshold, best_acc = _scan_threshold(y_arr, oof_mean)
    return EvalResult(
        name=f"{base.name}_smote_k{k_neighbors}_multi{len(SEEDS)}",
        description=f"{base.description} + local SMOTE k={k_neighbors} seeds={list(SEEDS)}",
        n_features=int(X_num.shape[1]),
        oof_acc_050=float(accuracy_score(y_arr, oof_mean >= 0.5)),
        oof_acc_best=best_acc,
        oof_best_threshold=threshold,
        oof_logloss=float(log_loss(y_arr, np.clip(oof_mean, 1e-6, 1 - 1e-6))),
        oof_auc=float(roc_auc_score(y_arr, oof_mean)),
        mean_best_iter=float(KAGGLE_SMOTE_PARAMS["n_estimators"]),
        positive_rate_050=float(np.mean(test_mean >= 0.5)),
        test_positive_rate_best=float(np.mean(test_mean >= threshold)),
        elapsed_seconds=float(time.perf_counter() - t0),
        oof=oof_mean,
        test=test_mean,
        y=y_arr,
        passenger_ids=base.passenger_ids,
        feature_names=list(X_num.columns),
        importance=pd.DataFrame(columns=["feature", "gain", "weight", "cover"]),
        source=f"{base.name} one-hot + team preprocessing + local SMOTE",
    )


def _read_existing_pset(name: str, oof_path: Path, test_path: Path) -> EvalResult | None:
    if not oof_path.exists() or not test_path.exists():
        return None
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    oof_col = "y_proba_blend" if "y_proba_blend" in oof_df.columns else "y_proba"
    test_col = "y_proba_blend" if "y_proba_blend" in test_df.columns else "y_proba"
    y_col = "y_true" if "y_true" in oof_df.columns else None
    if y_col is None:
        return None
    y = oof_df[y_col].to_numpy(dtype="int8")
    oof = oof_df[oof_col].to_numpy(dtype="float32")
    test = test_df[test_col].to_numpy(dtype="float32")
    threshold, best_acc = _scan_threshold(y, oof)
    return EvalResult(
        name=name,
        description=f"Existing team probability artifact: {oof_path.name} + {test_path.name}",
        n_features=-1,
        oof_acc_050=float(accuracy_score(y, oof >= 0.5)),
        oof_acc_best=best_acc,
        oof_best_threshold=threshold,
        oof_logloss=float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
        oof_auc=float(roc_auc_score(y, oof)),
        mean_best_iter=float("nan"),
        positive_rate_050=float(np.mean(test >= 0.5)),
        test_positive_rate_best=float(np.mean(test >= threshold)),
        elapsed_seconds=0.0,
        oof=oof,
        test=test,
        y=y,
        passenger_ids=test_df["PassengerId"].astype("string").tolist(),
        feature_names=[],
        importance=pd.DataFrame(columns=["feature", "gain", "weight", "cover"]),
        source=str(oof_path),
    )


def _blend(a: EvalResult, b: EvalResult) -> EvalResult:
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for w in np.arange(0.15, 0.86, 0.05):
        oof = (w * a.oof + (1.0 - w) * b.oof).astype("float32")
        threshold, acc = _scan_threshold(a.y, oof)
        ll = log_loss(a.y, np.clip(oof, 1e-6, 1 - 1e-6))
        score = (acc, -ll)
        if best is None or score > best[:2]:
            test = (w * a.test + (1.0 - w) * b.test).astype("float32")
            best = (acc, -ll, threshold, oof, test)
            best_w = float(w)
    assert best is not None
    acc, neg_ll, threshold, oof_best, test_best = best
    name = f"blend_{a.name}_w{int(round(best_w * 100)):02d}_{b.name}_w{int(round((1-best_w) * 100)):02d}"
    return EvalResult(
        name=name,
        description=f"OOF-selected probability blend: {a.name} x {best_w:.2f} + {b.name} x {1-best_w:.2f}",
        n_features=-1,
        oof_acc_050=float(accuracy_score(a.y, oof_best >= 0.5)),
        oof_acc_best=float(acc),
        oof_best_threshold=float(threshold),
        oof_logloss=float(-neg_ll),
        oof_auc=float(roc_auc_score(a.y, oof_best)),
        mean_best_iter=float("nan"),
        positive_rate_050=float(np.mean(test_best >= 0.5)),
        test_positive_rate_best=float(np.mean(test_best >= threshold)),
        elapsed_seconds=0.0,
        oof=oof_best,
        test=test_best,
        y=a.y,
        passenger_ids=a.passenger_ids,
        feature_names=[],
        importance=pd.DataFrame(columns=["feature", "gain", "weight", "cover"]),
        source=f"blend({a.name}, {b.name})",
    )


def _submission_at_rate(proba: np.ndarray, rate: float) -> tuple[np.ndarray, float]:
    k = int(round(len(proba) * rate))
    order = np.argsort(-proba)
    pred = np.zeros(len(proba), dtype=bool)
    pred[order[:k]] = True
    if 0 < k < len(proba):
        threshold = float((proba[order[k - 1]] + proba[order[k]]) / 2.0)
    else:
        threshold = 0.5
    return pred, threshold


def _write_submission(result: EvalResult, suffix: str, pred: np.ndarray, threshold: float, extra: dict[str, Any]) -> dict[str, Any]:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"submission_{result.name}_{suffix}".replace(".", "p")
    csv_path = SUBMISSIONS_DIR / f"{safe_name}.csv"
    meta_path = SUBMISSIONS_DIR / f"{safe_name}.json"
    frame = pd.DataFrame({"PassengerId": result.passenger_ids, "Transported": pred.astype(bool)})
    frame.to_csv(csv_path, index=False)
    meta = {
        "name": safe_name,
        "file": csv_path,
        "team_preprocessing_only": True,
        "probability_set": result.name,
        "source": result.source,
        "threshold": threshold,
        "oof_acc_best": result.oof_acc_best,
        "oof_acc_050": result.oof_acc_050,
        "oof_logloss": result.oof_logloss,
        "oof_auc": result.oof_auc,
        "positive_rate": float(pred.mean()),
        "true_count": int(pred.sum()),
        **extra,
    }
    meta_path.write_text(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return _json_safe(meta)


def _generate_submissions(results: list[EvalResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ranked = sorted(results, key=lambda r: (r.oof_acc_best, -r.oof_logloss), reverse=True)
    used_names: set[str] = set()
    for result in ranked[:8]:
        if result.name in used_names:
            continue
        used_names.add(result.name)
        pred_best = result.test >= result.oof_best_threshold
        rows.append(
            _write_submission(
                result,
                f"oofbest_t{result.oof_best_threshold:.3f}",
                pred_best,
                result.oof_best_threshold,
                {"candidate_kind": "oof_best_threshold"},
            )
        )
        for rate_name, rate in TARGET_RATES.items():
            pred, threshold = _submission_at_rate(result.test, rate)
            rows.append(
                _write_submission(
                    result,
                    rate_name,
                    pred,
                    threshold,
                    {"candidate_kind": "positive_rate_anchor", "target_positive_rate": rate},
                )
            )
    manifest = pd.DataFrame(rows).sort_values(["oof_acc_best", "positive_rate"], ascending=[False, False])
    top = manifest.head(8).copy()
    keep_files = set(top["file"].astype(str))
    for path in SUBMISSIONS_DIR.glob("submission_*.csv"):
        if str(path) not in keep_files:
            archive_dir = SUBMISSIONS_DIR / "99_extra_candidates_not_top8"
            archive_dir.mkdir(exist_ok=True)
            shutil.move(str(path), archive_dir / path.name)
            meta = path.with_suffix(".json")
            if meta.exists():
                shutil.move(str(meta), archive_dir / meta.name)
    top.to_csv(SUBMISSIONS_DIR / "submission_manifest_top8.csv", index=False)
    return top


def _summary_row(result: EvalResult, baseline_acc: float | None = None) -> dict[str, Any]:
    delta = None if baseline_acc is None else result.oof_acc_best - baseline_acc
    return {
        "name": result.name,
        "description": result.description,
        "n_features": result.n_features,
        "oof_acc_050": result.oof_acc_050,
        "oof_acc_best": result.oof_acc_best,
        "delta_vs_full_common": delta,
        "oof_best_threshold": result.oof_best_threshold,
        "oof_logloss": result.oof_logloss,
        "oof_auc": result.oof_auc,
        "mean_best_iter": result.mean_best_iter,
        "positive_rate_050": result.positive_rate_050,
        "test_positive_rate_best": result.test_positive_rate_best,
        "elapsed_seconds": result.elapsed_seconds,
        "source": result.source,
    }


def _write_feature_report(rows: pd.DataFrame, top_submissions: pd.DataFrame) -> None:
    baseline = rows.loc[rows["name"] == "full_common_native"]
    baseline_acc = float(baseline["oof_acc_best"].iloc[0]) if not baseline.empty else float("nan")
    direct_drop_names = {spec.name for spec in _view_specs() if spec.name.startswith("drop_")}
    must_keep: list[str] = []
    removable: list[str] = []
    pending: list[str] = []
    drop_rows = rows.loc[rows["name"].isin(direct_drop_names)].sort_values("delta_vs_full_common")
    for _, row in drop_rows.iterrows():
        name = str(row["name"])
        delta = float(row["delta_vs_full_common"])
        feature_label = name.removeprefix("drop_")
        if delta <= -0.0015:
            must_keep.append(f"{feature_label}: deletion lowered OOF by {delta:+.4f}")
        elif delta >= 0.0:
            removable.append(f"{feature_label}: deletion did not hurt OOF ({delta:+.4f})")
        else:
            pending.append(f"{feature_label}: small negative delta ({delta:+.4f}), needs public-LB check")
    if not must_keep:
        must_keep.append("No drop test crossed the strict -0.0015 OOF threshold.")
    if not removable:
        removable.append("No feature group clearly improved OOF when removed.")
    if not pending:
        pending.append("No borderline drop group in this run.")

    lines = [
        "# XGB Feature Ablation Sprint Report - 2026-04-27",
        "",
        "Scope: team preprocessing only. This run reads `processed/common` and `processed/xgboost`; raw Kaggle-template preprocessing is not used as model input.",
        "",
        "## Baseline",
        "",
        f"- `full_common_native` OOF best accuracy: `{baseline_acc:.6f}`",
        f"- Experiment rows: `{len(rows)}`",
        f"- Direct drop tests used for feature verdict: `{len(drop_rows)}`",
        "- SMOTE and blend rows are excluded from feature-redundancy classification.",
        "",
        "## 必须保留",
        "",
        *[f"- {item}" for item in must_keep],
        "",
        "## 可删 / 疑似冗余",
        "",
        *[f"- {item}" for item in removable],
        "",
        "## 待验证",
        "",
        *[f"- {item}" for item in pending],
        "",
        "## Top Candidate CSV",
        "",
    ]
    for i, (_, row) in enumerate(top_submissions.iterrows(), start=1):
        lines.append(
            f"{i}. `{Path(row['file']).name}` | OOF={row['oof_acc_best']:.6f} | "
            f"positive_rate={row['positive_rate']:.6f} | true={int(row['true_count'])}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `reports/feature_ablation_matrix.csv`",
            "- `reports/feature_importance_stability.csv`",
            "- `submissions/submission_manifest_top8.csv`",
            "- `archive_inputs/source_audit.json`",
            "",
            "Interpretation rule: feature groups are judged by OOF change first, then by candidate positive-rate risk. Public leaderboard confirmation is still required before claiming final superiority.",
        ]
    )
    (REPORTS_DIR / "feature_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_submissions() -> pd.DataFrame:
    _, _, _, _, expected_ids = _load_common()
    expected_ids = [str(item) for item in expected_ids]
    rows = []
    for path in sorted(SUBMISSIONS_DIR.glob("submission_*.csv")):
        if path.parent.name.startswith("99_"):
            continue
        if path.name == "submission_manifest_top8.csv":
            continue
        df = pd.read_csv(path)
        expected_columns = list(df.columns) == ["PassengerId", "Transported"]
        row_count_ok = len(df) == len(expected_ids)
        passenger_id_aligned = row_count_ok and df["PassengerId"].astype(str).tolist() == expected_ids if expected_columns else False
        transported_bool = pd.api.types.is_bool_dtype(df["Transported"]) if "Transported" in df else False
        ok = expected_columns and row_count_ok and passenger_id_aligned and transported_bool
        rows.append(
            {
                "file": str(path),
                "rows": len(df),
                "columns": ",".join(df.columns),
                "true_count": int(df["Transported"].astype(bool).sum()) if "Transported" in df else None,
                "passenger_id_aligned": bool(passenger_id_aligned),
                "transported_bool": bool(transported_bool),
                "valid": bool(ok),
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(REPORTS_DIR / "submission_validation.csv", index=False)
    if not validation.empty and not bool(validation["valid"].all()):
        raise RuntimeError("At least one generated submission failed validation")
    return validation


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_run_metadata()

    params = _load_best_params()
    X, X_test, y, groups, passenger_ids = _build_common_native_matrix()
    all_results: list[EvalResult] = []
    matrices_by_name: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for spec in _view_specs():
        X_view, X_test_view, dropped = _apply_view(X, X_test, spec)
        result = _fit_native_cv(spec.name, spec.description, X_view, X_test_view, y, groups, passenger_ids, params)
        result.source += f"; dropped={dropped}"
        all_results.append(result)
        matrices_by_name[result.name] = (X_view, X_test_view)
        result.importance.assign(view=result.name).to_csv(REPORTS_DIR / f"importance_{result.name}.csv", index=False)
        print(f"[view] {result.name}: OOF={result.oof_acc_best:.6f} features={result.n_features}")

    baseline_acc = next(r.oof_acc_best for r in all_results if r.name == "full_common_native")
    rows = pd.DataFrame([_summary_row(r, baseline_acc=baseline_acc) for r in all_results])

    compact_or_strong = rows.sort_values(["oof_acc_best", "oof_logloss"], ascending=[False, True]).head(4)["name"].tolist()
    smote_results: list[EvalResult] = []
    for name in compact_or_strong:
        base = next(r for r in all_results if r.name == name)
        X_view, X_test_view = matrices_by_name[name]
        for k in (3, 5):
            smote_result = _fit_smote_probability_set(base, X_view, X_test_view, y, groups, k_neighbors=k)
            smote_results.append(smote_result)
            print(f"[smote] {smote_result.name}: OOF={smote_result.oof_acc_best:.6f}")

    old_sets = [
        _read_existing_pset("team_A7_blend", OLD_LOGS / "A7_oof.csv", OLD_LOGS / "A7_test_proba.csv"),
        _read_existing_pset("team_A6_ensemble", OLD_LOGS / "A6_oof.csv", OLD_LOGS / "A6_test_proba.csv"),
    ]
    old_sets = [r for r in old_sets if r is not None]
    blend_results: list[EvalResult] = []
    for candidate in sorted(smote_results + all_results, key=lambda r: r.oof_acc_best, reverse=True)[:5]:
        for old in old_sets:
            blend_results.append(_blend(candidate, old))

    all_probability_sets = all_results + smote_results + old_sets + blend_results
    all_rows = pd.DataFrame([_summary_row(r, baseline_acc=baseline_acc) for r in all_probability_sets])
    all_rows.sort_values(["oof_acc_best", "oof_logloss"], ascending=[False, True]).to_csv(
        REPORTS_DIR / "feature_ablation_matrix.csv",
        index=False,
    )

    importance_files = sorted(REPORTS_DIR.glob("importance_*.csv"))
    if importance_files:
        imp = pd.concat([pd.read_csv(path) for path in importance_files], ignore_index=True)
        imp.groupby("feature", as_index=False)[["gain", "weight", "cover"]].agg(["mean", "std"]).to_csv(
            REPORTS_DIR / "feature_importance_stability.csv"
        )

    top_submissions = _generate_submissions(all_probability_sets)
    _write_feature_report(all_rows, top_submissions)
    validation = _validate_submissions()

    summary = {
        "feature_ablation_matrix": REPORTS_DIR / "feature_ablation_matrix.csv",
        "feature_verdict": REPORTS_DIR / "feature_verdict.md",
        "top_submissions": SUBMISSIONS_DIR / "submission_manifest_top8.csv",
        "submission_validation": REPORTS_DIR / "submission_validation.csv",
        "valid_submission_count": int(validation["valid"].sum()) if not validation.empty else 0,
        "best_oof": float(all_rows["oof_acc_best"].max()),
    }
    (REPORTS_DIR / "run_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
