from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from xgboost import XGBClassifier


SPRINT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = LOCAL_ROOT.parent

COMMON_BUNDLE = LOCAL_ROOT / "processed" / "common" / "preprocessed_common.joblib"
XGB_BUNDLE = LOCAL_ROOT / "processed" / "xgboost" / "preprocessed_xgboost.joblib"
OLD_LOGS = LOCAL_ROOT / "reports" / "xgboost" / "logs"

REPORTS_DIR = SPRINT_ROOT / "reports"
SUBMISSIONS_DIR = SPRINT_ROOT / "submissions"
ARCHIVE_INPUTS_DIR = SPRINT_ROOT / "archive_inputs"
HANDOFF_DIR = LOCAL_ROOT / "00_下一轮Kaggle提交_看这里"

EXPENSE_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
RAW_NUMERIC = ["ShoppingMall", "FoodCourt", "RoomService", "Spa", "VRDeck", "Expenses", "Age"]
RAW_CATEGORICAL = ["CryoSleep", "Cabin_1", "Cabin_3", "VIP", "HomePlanet", "Destination"]
RAW_DROP_LIST = {
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
}
SEEDS = (2024, 17, 7, 42, 88)
TARGET_RATES = {
    "rate53495_seed17": 0.5349544072948328,
    "rate53519_seed2024": 0.5351882160392799,
    "rate53566_seed7": 0.5356558335281739,
    "rate5366_public": 0.53659,
    "rate5324_anchor": 0.5323825111059154,
    "rate5116_safe": 0.5115735328501286,
}
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
    "verbosity": 0,
}


@dataclass
class ProbabilitySet:
    name: str
    view: str
    smote_k: int
    seed_label: str
    oof: np.ndarray
    test: np.ndarray
    y: np.ndarray
    passenger_ids: list[str]
    source: str
    elapsed_seconds: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_audit() -> None:
    ARCHIVE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "team_preprocessing_only": True,
        "purpose": "Recover public leaderboard score after drop_cabinnum public failure by using raw-highscore-like compact views derived from team common bundle.",
        "common_bundle": COMMON_BUNDLE,
        "xgb_bundle_reference_only": XGB_BUNDLE,
        "old_probability_logs_reference_only": OLD_LOGS,
        "forbidden_training_sources": [
            "raw Kaggle train/test CSV",
            "archived umanglodaya raw-template submission CSV as model input",
            "raw-template test probabilities as blend input",
        ],
        "known_public_feedback": {
            "submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5366_public.csv": 0.80430,
            "lesson": "Do not prioritize drop_cabinnum OOF gains; keep CabinNum-derived information and compact public-style views separate from A7-heavy blends.",
        },
    }
    (ARCHIVE_INPUTS_DIR / "source_audit.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_common() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    bundle = joblib.load(COMMON_BUNDLE)
    train = bundle["common_train"].reset_index(drop=True).copy()
    test = bundle["common_test"].reset_index(drop=True).copy()
    y = pd.Series(bundle["y_train"]).reset_index(drop=True).astype("int8")
    passenger_ids = test["PassengerId"].astype("string").tolist()
    return train, test, y, passenger_ids


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _raw_like_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in EXPENSE_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["Expenses"] = pd.to_numeric(df["TotalSpend"], errors="coerce")
    out["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["CryoSleep"] = df["CryoSleep"].astype("object")
    out["Cabin_1"] = df["Deck"].astype("object")
    out["Cabin_3"] = df["Side"].astype("object")
    out["VIP"] = df["VIP"].astype("object")
    out["HomePlanet"] = df["HomePlanet"].astype("object")
    out["Destination"] = df["Destination"].astype("object")
    return out


def _build_raw_highscore_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    view: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_raw = _raw_like_frame(train)
    test_raw = _raw_like_frame(test)
    combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)

    numeric_cols = list(RAW_NUMERIC)
    categorical_cols = list(RAW_CATEGORICAL)
    if view in {"raw_plus_group", "raw_plus_group_cabin"}:
        combined["GroupSize"] = pd.concat([train["GroupSize"], test["GroupSize"]], ignore_index=True)
        combined["IsSolo"] = pd.concat([train["IsSolo"], test["IsSolo"]], ignore_index=True)
        numeric_cols.extend(["GroupSize", "IsSolo"])
    if view == "raw_plus_group_cabin":
        combined["CabinNum"] = pd.concat([train["CabinNum"], test["CabinNum"]], ignore_index=True)
        combined["CabinNumBin"] = pd.concat([train["CabinNumBin"], test["CabinNumBin"]], ignore_index=True).astype("object")
        numeric_cols.append("CabinNum")
        categorical_cols.append("CabinNumBin")

    numeric = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(combined[numeric_cols]),
        columns=numeric_cols,
    )
    categorical_input = combined[categorical_cols].astype("object").where(combined[categorical_cols].notna(), np.nan)
    categorical = pd.DataFrame(
        SimpleImputer(strategy="most_frequent").fit_transform(categorical_input),
        columns=categorical_cols,
    )
    encoder = _one_hot_encoder()
    encoded = pd.DataFrame(
        encoder.fit_transform(categorical),
        columns=encoder.get_feature_names_out(categorical_cols),
    )
    matrix = pd.concat([numeric, encoded], axis=1)
    if view != "raw_no_drop":
        drop_cols = [col for col in matrix.columns if col in RAW_DROP_LIST]
        matrix = matrix.drop(columns=drop_cols)
    else:
        drop_cols = []

    n_train = len(train)
    x = matrix.iloc[:n_train].reset_index(drop=True).astype("float32")
    x_test = matrix.iloc[n_train:].reset_index(drop=True).astype("float32")
    return x, x_test, drop_cols


def _smote_resample(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    k_neighbors: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype="int8")
    labels, counts = np.unique(y_arr, return_counts=True)
    if len(labels) != 2:
        return x.copy(), y.copy()
    minority_label = labels[int(np.argmin(counts))]
    majority_count = int(counts.max())
    target_minority_count = majority_count
    minority_idx = np.flatnonzero(y_arr == minority_label)
    n_new = target_minority_count - len(minority_idx)
    if n_new <= 0 or len(minority_idx) < 2:
        return x.copy(), y.copy()

    x_np = x.to_numpy(dtype="float32", copy=True)
    minority_x = x_np[minority_idx]
    n_neighbors = min(k_neighbors + 1, len(minority_idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(minority_x)
    neighbor_indices = nn.kneighbors(minority_x, return_distance=False)

    base_rows = rng.integers(0, len(minority_idx), size=n_new)
    neighbor_choices = rng.integers(1, n_neighbors, size=n_new) if n_neighbors > 1 else np.zeros(n_new, dtype=int)
    neighbor_rows = neighbor_indices[base_rows, neighbor_choices]
    gaps = rng.random((n_new, 1), dtype=np.float32)
    synthetic = minority_x[base_rows] + gaps * (minority_x[neighbor_rows] - minority_x[base_rows])

    x_resampled = pd.DataFrame(np.vstack([x_np, synthetic]), columns=x.columns)
    y_resampled = pd.Series(np.concatenate([y_arr, np.full(n_new, minority_label, dtype="int8")]))
    return x_resampled, y_resampled


def _fit_seed(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    *,
    seed: int,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(x), dtype="float32")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
        smote_seed = seed + fold
        x_train, y_train = x.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        x_valid = x.iloc[valid_idx].reset_index(drop=True)
        x_sm, y_sm = _smote_resample(x_train, y_train, random_state=smote_seed, k_neighbors=k_neighbors)
        model = XGBClassifier(**PARAMS_XGB_BEST, random_state=smote_seed)
        model.fit(x_sm, y_sm)
        oof[valid_idx] = model.predict_proba(x_valid)[:, 1]

    x_fit, y_fit = shuffle(x, y, random_state=seed)
    x_sm, y_sm = _smote_resample(x_fit, y_fit, random_state=seed, k_neighbors=k_neighbors)
    model = XGBClassifier(**PARAMS_XGB_BEST, random_state=seed)
    model.fit(x_sm, y_sm)
    test = model.predict_proba(x_test)[:, 1].astype("float32")
    return oof, test


def _scan_threshold(y: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    accs = np.array([accuracy_score(y, proba >= t) for t in thresholds])
    idx = int(accs.argmax())
    return float(thresholds[idx]), float(accs[idx])


def _pred_at_rate(proba: np.ndarray, rate: float) -> tuple[np.ndarray, float]:
    k = int(round(len(proba) * rate))
    order = np.argsort(-proba)
    pred = np.zeros(len(proba), dtype=bool)
    pred[order[:k]] = True
    if 0 < k < len(proba):
        threshold = float((proba[order[k - 1]] + proba[order[k]]) / 2.0)
    else:
        threshold = 0.5
    return pred, threshold


def _write_proba(pset: ProbabilitySet) -> None:
    pd.DataFrame({"y_true": pset.y, "y_proba": pset.oof}).to_csv(
        REPORTS_DIR / f"{pset.name}_oof_proba.csv",
        index=False,
    )
    pd.DataFrame({"PassengerId": pset.passenger_ids, "y_proba": pset.test}).to_csv(
        REPORTS_DIR / f"{pset.name}_test_proba.csv",
        index=False,
    )


def _build_probability_sets(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    passenger_ids: list[str],
    *,
    view: str,
    smote_k: int,
    source: str,
) -> list[ProbabilitySet]:
    started = time.perf_counter()
    seed_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for seed in SEEDS:
        seed_outputs[seed] = _fit_seed(x, y, x_test, seed=seed, k_neighbors=smote_k)
        print(f"[seed] view={view} k={smote_k} seed={seed}")

    results: list[ProbabilitySet] = []
    for seed in (2024, 17, 7):
        oof, test = seed_outputs[seed]
        results.append(
            ProbabilitySet(
                name=f"team_common_rawlike_{view}_k{smote_k}_seed{seed}",
                view=view,
                smote_k=smote_k,
                seed_label=str(seed),
                oof=oof,
                test=test,
                y=y.to_numpy(dtype="int8"),
                passenger_ids=passenger_ids,
                source=f"{source}; single_seed={seed}",
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    for label, seeds in {"multi3_public": (2024, 17, 7), "multi5": SEEDS}.items():
        oof = np.mean(np.stack([seed_outputs[seed][0] for seed in seeds]), axis=0).astype("float32")
        test = np.mean(np.stack([seed_outputs[seed][1] for seed in seeds]), axis=0).astype("float32")
        results.append(
            ProbabilitySet(
                name=f"team_common_rawlike_{view}_k{smote_k}_{label}",
                view=view,
                smote_k=smote_k,
                seed_label=label,
                oof=oof,
                test=test,
                y=y.to_numpy(dtype="int8"),
                passenger_ids=passenger_ids,
                source=f"{source}; averaged_seeds={list(seeds)}",
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return results


def _write_submission(pset: ProbabilitySet, suffix: str, pred: np.ndarray, threshold: float, meta: dict[str, Any]) -> dict[str, Any]:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS_DIR / f"submission_{pset.name}_{suffix}.csv"
    meta_path = path.with_suffix(".json")
    pd.DataFrame({"PassengerId": pset.passenger_ids, "Transported": pred.astype(bool)}).to_csv(path, index=False)
    row = {
        "name": path.stem,
        "file": str(path),
        "team_preprocessing_only": True,
        "probability_set": pset.name,
        "feature_view": pset.view,
        "smote_k": pset.smote_k,
        "seed_label": pset.seed_label,
        "source": pset.source,
        "threshold": threshold,
        "positive_rate": float(pred.mean()),
        "true_count": int(pred.sum()),
        **meta,
    }
    meta_path.write_text(json.dumps(_json_safe(row), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return row


def _candidate_rows(pset: ProbabilitySet) -> list[dict[str, Any]]:
    y = pset.y
    best_t, best_acc = _scan_threshold(y, pset.oof)
    base_meta = {
        "oof_best_threshold": best_t,
        "oof_best_accuracy": best_acc,
        "oof_acc_050": float(accuracy_score(y, pset.oof >= 0.5)),
        "oof_logloss": float(log_loss(y, pset.oof.clip(1e-6, 1 - 1e-6))),
        "oof_auc": float(roc_auc_score(y, pset.oof)),
    }
    rows = []
    for suffix, threshold in {"t050": 0.5, f"oofbest_t{best_t:.3f}".replace(".", "p"): best_t}.items():
        rows.append(
            _write_submission(
                pset,
                suffix,
                pset.test >= threshold,
                threshold,
                {**base_meta, "candidate_kind": "fixed_threshold"},
            )
        )
    for suffix, rate in TARGET_RATES.items():
        pred, threshold = _pred_at_rate(pset.test, rate)
        rows.append(
            _write_submission(
                pset,
                suffix,
                pred,
                threshold,
                {**base_meta, "candidate_kind": "positive_rate_anchor", "target_positive_rate": rate},
            )
        )
    return rows


def _validate_submissions(expected_ids: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(SUBMISSIONS_DIR.glob("submission_*.csv")):
        if path.name == "submission_manifest_public_recovery.csv":
            continue
        df = pd.read_csv(path)
        expected_columns = list(df.columns) == ["PassengerId", "Transported"]
        row_count_ok = len(df) == len(expected_ids)
        passenger_id_aligned = expected_columns and row_count_ok and df["PassengerId"].astype(str).tolist() == expected_ids
        transported_bool = "Transported" in df and pd.api.types.is_bool_dtype(df["Transported"])
        valid = expected_columns and row_count_ok and passenger_id_aligned and transported_bool
        rows.append(
            {
                "file": str(path),
                "rows": len(df),
                "true_count": int(df["Transported"].astype(bool).sum()) if "Transported" in df else None,
                "passenger_id_aligned": bool(passenger_id_aligned),
                "transported_bool": bool(transported_bool),
                "valid": bool(valid),
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(REPORTS_DIR / "submission_validation.csv", index=False)
    if not validation.empty and not bool(validation["valid"].all()):
        raise RuntimeError("Invalid recovery submission generated")
    return validation


def _promote_top(manifest: pd.DataFrame) -> pd.DataFrame:
    public_band = manifest.loc[
        (manifest["positive_rate"].between(0.532, 0.537))
        & manifest["name"].str.contains("raw_core|raw_plus_group", regex=True)
        & ~manifest["name"].str.contains("raw_plus_group_cabin", regex=False)
    ].copy()
    if public_band.empty:
        public_band = manifest.copy()
    public_band["single_seed_priority"] = public_band["seed_label"].map({"2024": 0, "17": 1, "7": 2}).fillna(3)
    top = public_band.sort_values(
        ["single_seed_priority", "oof_best_accuracy", "oof_logloss", "positive_rate"],
        ascending=[True, False, True, False],
    ).head(8)
    for path in SUBMISSIONS_DIR.glob("submission_*.csv"):
        if path.name == "submission_manifest_public_recovery.csv":
            continue
        if str(path) not in set(top["file"]):
            archive_dir = SUBMISSIONS_DIR / "99_extra_candidates_not_top8"
            archive_dir.mkdir(exist_ok=True)
            shutil.move(str(path), archive_dir / path.name)
            meta = path.with_suffix(".json")
            if meta.exists():
                shutil.move(str(meta), archive_dir / meta.name)
    top.to_csv(SUBMISSIONS_DIR / "submission_manifest_public_recovery.csv", index=False)
    return top


def _write_report(manifest: pd.DataFrame, top: pd.DataFrame, validation: pd.DataFrame) -> None:
    lines = [
        "# XGB Public Recovery Sprint - 2026-04-27",
        "",
        "Scope: team preprocessing only. The feature matrix is derived from `processed/common/preprocessed_common.joblib`; raw Kaggle CSVs and archived raw-template probabilities are not used as model input.",
        "",
        "## Why This Sprint Exists",
        "",
        "- The just-submitted `drop_cabinnum + A7` candidate scored `0.80430` public, so that OOF gain is treated as leaderboard overfit.",
        "- This sprint goes back toward the public-proven high-score structure: compact spend/CryoSleep/cabin/home/destination features, SMOTE, and high-score XGB parameters.",
        "",
        "## Top 8 Upload Order",
        "",
    ]
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"{i}. `{Path(row['file']).name}` | view={row['feature_view']} | seed={row['seed_label']} | "
            f"k={int(row['smote_k'])} | pos={row['positive_rate']:.6f} | OOF={row['oof_best_accuracy']:.6f}"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Active CSV count: `{len(validation)}`",
            f"- Invalid CSV count: `{int((~validation['valid']).sum()) if not validation.empty else 0}`",
            "",
            "## Files",
            "",
            "- `submissions/submission_manifest_public_recovery.csv`",
            "- `reports/public_recovery_manifest_all.csv`",
            "- `reports/submission_validation.csv`",
            "- `archive_inputs/source_audit.json`",
        ]
    )
    (REPORTS_DIR / "public_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_audit()

    train, test, y, passenger_ids = _load_common()
    views = ("raw_core", "raw_plus_group", "raw_plus_group_cabin")
    all_sets: list[ProbabilitySet] = []
    schema_rows = []
    for view in views:
        x, x_test, drops = _build_raw_highscore_matrix(train, test, view)
        schema_rows.append(
            {
                "view": view,
                "n_features": x.shape[1],
                "dropped_raw_template_columns": json.dumps(drops, ensure_ascii=False),
                "columns": json.dumps(list(x.columns), ensure_ascii=False),
            }
        )
        source = f"{COMMON_BUNDLE}; view={view}; n_features={x.shape[1]}; dropped={drops}"
        for smote_k in (3, 5):
            all_sets.extend(
                _build_probability_sets(
                    x,
                    y,
                    x_test,
                    passenger_ids,
                    view=view,
                    smote_k=smote_k,
                    source=source,
                )
            )
    pd.DataFrame(schema_rows).to_csv(REPORTS_DIR / "feature_schema_public_recovery.csv", index=False)

    candidate_rows: list[dict[str, Any]] = []
    for pset in all_sets:
        _write_proba(pset)
        candidate_rows.extend(_candidate_rows(pset))

    manifest = pd.DataFrame(candidate_rows).sort_values(
        ["oof_best_accuracy", "positive_rate"],
        ascending=[False, False],
    )
    manifest.to_csv(REPORTS_DIR / "public_recovery_manifest_all.csv", index=False)
    top = _promote_top(manifest)
    validation = _validate_submissions(passenger_ids)
    _write_report(manifest, top, validation)
    print(
        json.dumps(
            {
                "manifest": str(REPORTS_DIR / "public_recovery_manifest_all.csv"),
                "top_manifest": str(SUBMISSIONS_DIR / "submission_manifest_public_recovery.csv"),
                "report": str(REPORTS_DIR / "public_recovery_report.md"),
                "valid_submission_count": int(validation["valid"].sum()),
                "top": top[["name", "feature_view", "smote_k", "seed_label", "positive_rate", "oof_best_accuracy"]].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
