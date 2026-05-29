"""Team-preprocessing-only XGBoost sprint candidates.

This script is intentionally strict: every model probability used here comes
from the team's saved preprocessing outputs under ``processed/``.  Kaggle
high-score notebooks are only used for model-side ideas: SMOTE, seed sweeps,
positive-rate anchoring, and probability blending.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from xgboost import XGBClassifier

from . import config


OUT_DIR = config.REPORTS_DIR / "submission_candidates"
HANDOFF_DIR = config.PROJECT_ROOT / "00_下一轮Kaggle提交_看这里"
TAG = "team_xgb107_smote"

EXPLOIT_SEEDS = (2024, 17, 7, 42, 88)
TARGET_RATES = {
    "rate5127_a7": 0.5127425765723638,
    "rate517_a6": 0.5167173252279635,
    "rate5324_anchor": 0.5323825111059154,
    "rate5352_umang": 0.5351882160392799,
    "rate5366_public": 0.53659,
}

KAGGLE_XGB_SMOTE_PARAMS = {
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

ARUNKLENIN_XGB_ONLY_PARAMS = {
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

PARAM_SETS = {
    "umang81575": KAGGLE_XGB_SMOTE_PARAMS,
    "arun_xgb_only": ARUNKLENIN_XGB_ONLY_PARAMS,
}

SMOTE_VARIANTS = {
    "k5": {"sampling_strategy": 1, "k_neighbors": 5},
    "k3": {"sampling_strategy": 1, "k_neighbors": 3},
}

# Translated from Umang Lodaya's permutation-importance drop list onto the
# team's 107-feature XGBoost bundle. This is model-side pruning only: the
# saved team preprocessing bundle remains untouched.
KAGGLE_LOW_IMPORTANCE_DROPS = (
    "num__ShoppingMall",
    "num__Age",
    "cat__CryoSleep_True",
    "cat__HomePlanet_Earth",
    "cat__HomePlanet_Europa",
    "cat__VIP_True",
    "cat__HomePlanet_Mars",
    "cat__Destination_PSO J318.5-22",
    "cat__VIP_False",
    "cat__Destination_55 Cancri e",
    "num__FoodCourt",
    "cat__Destination_TRAPPIST-1e",
)

FEATURE_VIEWS = {
    "full107": (),
    "drop_umang_low95": KAGGLE_LOW_IMPORTANCE_DROPS,
}


@dataclass(frozen=True)
class SprintConfig:
    name: str
    feature_view: str
    param_set: str
    smote_variant: str


SPRINT_CONFIGS = (
    SprintConfig("team_xgb107_umang_k5", "full107", "umang81575", "k5"),
    SprintConfig("team_xgb095_umang_k5", "drop_umang_low95", "umang81575", "k5"),
    SprintConfig("team_xgb095_umang_k3", "drop_umang_low95", "umang81575", "k3"),
    SprintConfig("team_xgb095_arun_k5", "drop_umang_low95", "arun_xgb_only", "k5"),
)


@dataclass
class ProbabilitySet:
    name: str
    oof: np.ndarray
    test: np.ndarray
    y: np.ndarray
    passenger_ids: list[str]
    source: str


def _load_team_xgb_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    bundle = joblib.load(config.XGB_BUNDLE)
    feature_names = bundle["feature_names"]
    x = pd.DataFrame(np.asarray(bundle["X_train"], dtype="float32"), columns=feature_names)
    x_test = pd.DataFrame(np.asarray(bundle["X_test"], dtype="float32"), columns=feature_names)
    y = pd.Series(bundle["y_train"]).reset_index(drop=True).astype("int8")
    passenger_ids = pd.Series(bundle["test_ids"]).astype("string").tolist()
    return x, x_test, y, passenger_ids


def _apply_feature_view(
    x: pd.DataFrame,
    x_test: pd.DataFrame,
    feature_view: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    drops = [column for column in FEATURE_VIEWS[feature_view] if column in x.columns]
    if not drops:
        return x.copy(), x_test.copy(), []
    return x.drop(columns=drops).copy(), x_test.drop(columns=drops).copy(), drops


def _smote_resample(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    sampling_strategy: float,
    k_neighbors: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Small local SMOTE implementation to avoid adding a runtime dependency."""
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype="int8")
    labels, counts = np.unique(y_arr, return_counts=True)
    if len(labels) != 2:
        return x.copy(), y.copy()

    minority_label = labels[int(np.argmin(counts))]
    majority_count = int(counts.max())
    target_minority_count = int(round(majority_count * sampling_strategy))
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
    if n_neighbors > 1:
        neighbor_choices = rng.integers(1, n_neighbors, size=n_new)
    else:
        neighbor_choices = np.zeros(n_new, dtype=int)
    neighbor_rows = neighbor_indices[base_rows, neighbor_choices]
    gaps = rng.random((n_new, 1), dtype=np.float32)
    synthetic = minority_x[base_rows] + gaps * (minority_x[neighbor_rows] - minority_x[base_rows])

    x_resampled = pd.DataFrame(np.vstack([x_np, synthetic]), columns=x.columns)
    y_resampled = pd.Series(np.concatenate([y_arr, np.full(n_new, minority_label, dtype="int8")]))
    return x_resampled, y_resampled


def _fit_smote_predict(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    seed: int,
    params: dict,
    smote_kwargs: dict,
) -> np.ndarray:
    x_fit, y_fit = shuffle(x, y, random_state=seed)
    x_sm, y_sm = _smote_resample(x_fit, y_fit, random_state=seed, **smote_kwargs)
    model = XGBClassifier(**params, random_state=seed)
    model.fit(x_sm, y_sm)
    return model.predict_proba(x_test)[:, 1].astype("float32")


def _oof_smote_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    seeds: tuple[int, ...],
    run_config: SprintConfig,
    dropped_columns: list[str],
) -> ProbabilitySet:
    oof_by_seed = []
    test_by_seed = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    params = PARAM_SETS[run_config.param_set]
    smote_kwargs = SMOTE_VARIANTS[run_config.smote_variant]

    for seed in seeds:
        oof = np.zeros(len(x), dtype="float32")
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
            x_train = x.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            x_valid = x.iloc[valid_idx].reset_index(drop=True)
            smote_seed = seed + fold
            x_sm, y_sm = _smote_resample(x_train, y_train, random_state=smote_seed, **smote_kwargs)
            model = XGBClassifier(**params, random_state=smote_seed)
            model.fit(x_sm, y_sm)
            oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
        oof_by_seed.append(oof)
        test_by_seed.append(_fit_smote_predict(x, y, x_test, seed, params, smote_kwargs))

    return ProbabilitySet(
        name=f"{run_config.name}_multi{len(seeds)}",
        oof=np.mean(np.stack(oof_by_seed), axis=0).astype("float32"),
        test=np.mean(np.stack(test_by_seed), axis=0).astype("float32"),
        y=y.to_numpy(dtype="int8"),
        passenger_ids=[],
        source=(
            f"{config.XGB_BUNDLE} + feature_view={run_config.feature_view} "
            f"dropped={dropped_columns} + params={run_config.param_set} "
            f"+ SMOTE={run_config.smote_variant}{smote_kwargs} seeds={list(seeds)}"
        ),
    )


def _read_team_probability_set(name: str, oof_path: Path, test_path: Path) -> ProbabilitySet | None:
    if not oof_path.exists() or not test_path.exists():
        return None
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    if "y_proba_blend" in oof_df.columns:
        oof = oof_df["y_proba_blend"].to_numpy("float32")
    else:
        oof = oof_df["y_proba"].to_numpy("float32")
    if "y_proba_blend" in test_df.columns:
        test = test_df["y_proba_blend"].to_numpy("float32")
    else:
        test = test_df["y_proba"].to_numpy("float32")
    y_col = "y_true" if "y_true" in oof_df.columns else None
    if y_col is None:
        return None
    return ProbabilitySet(
        name=name,
        oof=oof,
        test=test,
        y=oof_df[y_col].to_numpy("int8"),
        passenger_ids=test_df["PassengerId"].astype("string").tolist(),
        source=f"{oof_path} + {test_path}",
    )


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


def _write_submission(
    name: str,
    passenger_ids: list[str],
    pred: np.ndarray,
    meta: dict,
) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    meta_path = OUT_DIR / f"{name}.json"
    pd.DataFrame({"PassengerId": passenger_ids, "Transported": pred.astype(bool)}).to_csv(path, index=False)
    full_meta = {
        "file": str(path),
        "team_preprocessing_only": True,
        "source_bundles": {
            "xgboost": str(config.XGB_BUNDLE),
            "common": str(config.COMMON_BUNDLE),
        },
        **meta,
        "positive_rate": float(pred.mean()),
        "true_count": int(pred.sum()),
    }
    meta_path.write_text(json.dumps(full_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "path": str(path), **full_meta}


def _candidate_rows(pset: ProbabilitySet, passenger_ids: list[str]) -> list[dict]:
    rows = []
    best_t, best_acc = _scan_threshold(pset.y, pset.oof)
    threshold_specs = {
        "t050": 0.5,
        f"oofbest_t{best_t:.3f}".replace(".", "p"): best_t,
    }
    for suffix, threshold in threshold_specs.items():
        pred = pset.test >= threshold
        rows.append(
            _write_submission(
                f"submission_{pset.name}_{suffix}",
                passenger_ids,
                pred,
                {
                    "probability_set": pset.name,
                    "source": pset.source,
                    "threshold": float(threshold),
                    "oof_best_threshold": best_t,
                    "oof_best_accuracy": best_acc,
                    "oof_accuracy_at_threshold": float(accuracy_score(pset.y, pset.oof >= threshold)),
                    "oof_logloss": float(log_loss(pset.y, pset.oof.clip(1e-6, 1 - 1e-6))),
                    "oof_auc": float(roc_auc_score(pset.y, pset.oof)),
                },
            )
        )
    for suffix, rate in TARGET_RATES.items():
        pred, threshold = _pred_at_rate(pset.test, rate)
        rows.append(
            _write_submission(
                f"submission_{pset.name}_{suffix}",
                passenger_ids,
                pred,
                {
                    "probability_set": pset.name,
                    "source": pset.source,
                    "threshold": threshold,
                    "target_positive_rate": rate,
                    "oof_best_threshold": best_t,
                    "oof_best_accuracy": best_acc,
                    "oof_logloss": float(log_loss(pset.y, pset.oof.clip(1e-6, 1 - 1e-6))),
                    "oof_auc": float(roc_auc_score(pset.y, pset.oof)),
                },
            )
        )
    return rows


def _blend_probability_sets(a: ProbabilitySet, b: ProbabilitySet, passenger_ids: list[str]) -> tuple[ProbabilitySet, dict]:
    best = None
    for w in np.arange(0.15, 0.86, 0.05):
        oof = w * a.oof + (1.0 - w) * b.oof
        threshold, acc = _scan_threshold(a.y, oof)
        ll = log_loss(a.y, oof.clip(1e-6, 1 - 1e-6))
        auc = roc_auc_score(a.y, oof)
        score = (acc, -ll, auc)
        if best is None or score > best[0]:
            test = w * a.test + (1.0 - w) * b.test
            best = (score, w, threshold, acc, ll, auc, oof.astype("float32"), test.astype("float32"))
    _, w, threshold, acc, ll, auc, oof, test = best
    name = f"blend_{a.name}_w{int(round(w * 100)):02d}_{b.name}_w{int(round((1-w) * 100)):02d}"
    pset = ProbabilitySet(
        name=name,
        oof=oof,
        test=test,
        y=a.y,
        passenger_ids=passenger_ids,
        source=f"blend({a.source}, {b.source})",
    )
    meta = {
        "blend_weight_first": float(w),
        "blend_weight_second": float(1.0 - w),
        "oof_best_threshold": float(threshold),
        "oof_best_accuracy": float(acc),
        "oof_logloss": float(ll),
        "oof_auc": float(auc),
    }
    return pset, meta


def main() -> None:
    x, x_test, y, passenger_ids = _load_team_xgb_matrix()

    rows = []
    smote_sets: list[ProbabilitySet] = []
    for run_config in SPRINT_CONFIGS:
        x_view, x_test_view, dropped_columns = _apply_feature_view(x, x_test, run_config.feature_view)
        pset = _oof_smote_ensemble(
            x_view,
            y,
            x_test_view,
            EXPLOIT_SEEDS,
            run_config,
            dropped_columns,
        )
        pset.passenger_ids = passenger_ids
        smote_sets.append(pset)
        rows.extend(_candidate_rows(pset, passenger_ids))

    logs = config.LOGS_DIR
    team_sets = [
        _read_team_probability_set("team_A7_blend", logs / "A7_oof.csv", logs / "A7_test_proba.csv"),
        _read_team_probability_set("team_A6_ensemble", logs / "A6_oof.csv", logs / "A6_test_proba.csv"),
    ]
    for pset in [p for p in team_sets if p is not None]:
        rows.extend(_candidate_rows(pset, passenger_ids))
        for smote_set in smote_sets:
            blended, blend_meta = _blend_probability_sets(smote_set, pset, passenger_ids)
            rows.extend(_candidate_rows(blended, passenger_ids))
            (OUT_DIR / f"submission_{blended.name}_blend_meta.json").write_text(
                json.dumps(blend_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["oof_best_accuracy", "positive_rate"],
        ascending=[False, False],
    )
    manifest_path = OUT_DIR / "team_xgb_sprint_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    top = manifest.head(8)
    for _, row in top.iterrows():
        src = Path(row["file"])
        dst = HANDOFF_DIR / src.name
        dst.write_bytes(src.read_bytes())

    readme = [
        "# 下一轮 Kaggle 提交候选（团队预处理限定）",
        "",
        "硬约束：这里的 CSV 全部基于团队 `processed/common` 或 `processed/xgboost` 输出。",
        "Kaggle 高分模板只迁移了 XGBoost 参数、SMOTE、seed、阈值和正例率锚定思路。",
        "",
        "优先级按本地 OOF 与 public-LB 正例率经验综合排序：",
    ]
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        readme.append(
            f"{i}. `{Path(row['file']).name}` | "
            f"OOF={row.get('oof_best_accuracy', float('nan')):.6f} | "
            f"positive_rate={row['positive_rate']:.6f} | true={int(row['true_count'])}"
        )
    readme.append("")
    readme.append("注意：目录里可能保留上一轮团队候选；本轮提交顺序按上方 1-8 执行。")
    readme.append("")
    readme.append(f"完整 manifest: `{manifest_path}`")
    (HANDOFF_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps({
        "manifest": str(manifest_path),
        "handoff_dir": str(HANDOFF_DIR),
        "top": top[["name", "positive_rate", "true_count", "oof_best_accuracy", "file"]].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
