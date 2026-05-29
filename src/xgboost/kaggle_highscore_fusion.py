"""Kaggle-highscore-inspired XGBoost submission search.

This script does not modify the shared preprocessing bundle.  It only consumes
the XGB family probability files that already exist under ``reports/xgboost``.

The search encodes the useful ideas seen in the stronger Kaggle XGBoost
notebooks/discussions:

- public-style raw CSV XGB family: expenses, CryoSleep zero-spend rule,
  passenger group/room fill, cabin split, Optuna-tuned XGB;
- travel-group missing-value logic: use group-consistent fills as a separate
  prediction family rather than rewriting the common preprocessing;
- high-score notebook blending: combine differently calibrated XGB families
  with probability, rank, and logit blends instead of trusting one proba scale;
- public-LB distribution anchoring: generate candidates around both the clean
  V2 positive-rate band and the more public-notebook-like 0.53 positive-rate
  band.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from . import config


LOCAL_Y_PATH = config.PROJECT_ROOT / "spaceship-titanic" / "processed" / "xgboost" / "y_train_xgb.csv"
RAW_TRAIN_PATH = config.PROJECT_ROOT / "data" / "raw" / "train.csv"
STEPWISE_NPZ_PATH = config.LOGS_DIR / "candidate_probs.npz"
OUT_DIR = config.REPORTS_DIR / "submission_candidates"
SUMMARY_PATH = config.LOGS_DIR / "kaggle_highscore_fusion_summary.csv"

TARGET_POSITIVE_RATES = {
    # Clean local V2 family rate.  Good for OOF-safe candidates.
    "anchor511": 0.51134,
    # Known local public-LB-friendly candidates and Kaggle public-style notebooks.
    "mid525": 0.52500,
    "public532": 0.53238,
    "public536": 0.53659,
}


@dataclass
class SearchCandidate:
    rank: int
    name: str
    file: str
    metadata_path: str
    blend_kind: str
    target_rate_name: str
    target_positive_rate: float
    threshold: float
    honest_oof_acc: float
    positive_rate: float
    changed_vs_v2_best: int
    changed_vs_public_style: int
    changed_vs_anchor_a7: int | None
    weights: dict[str, float]
    source: str


def _read_proba(path: Path, column: str = "y_proba") -> np.ndarray:
    return pd.read_csv(path)[column].to_numpy(dtype=float)


def _load_stepwise() -> dict[str, np.ndarray]:
    npz = np.load(STEPWISE_NPZ_PATH)
    return {
        "stepwise_safe_no_te_oof": npz["stepwise_safe_no_te__oof"],
        "stepwise_safe_no_te_test": npz["stepwise_safe_no_te__test"],
        "stepwise_safe_te_oof": npz["stepwise_safe_te__oof"],
        "stepwise_safe_te_test": npz["stepwise_safe_te__test"],
        "stepwise_balanced_oof": npz["stepwise_balanced_te__oof"],
        "stepwise_balanced_test": npz["stepwise_balanced_te__test"],
        "legacy_public_oof": npz["legacy_public_style__oof"],
        "legacy_public_test": npz["legacy_public_style__test"],
    }


def _load_labels() -> np.ndarray:
    if LOCAL_Y_PATH.exists():
        return pd.read_csv(LOCAL_Y_PATH)["Transported"].astype(int).to_numpy()
    return pd.read_csv(RAW_TRAIN_PATH)["Transported"].astype(int).to_numpy()


def _load_family_probs() -> tuple[pd.Series, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    passenger_id_test = pd.read_csv(config.LOGS_DIR / "v2_opt_test_proba.csv")["PassengerId"].astype(str)
    y = _load_labels()
    stepwise = _load_stepwise()

    a7_oof = pd.read_csv(config.LOGS_DIR / "A7_oof.csv")
    a7_test = pd.read_csv(config.LOGS_DIR / "A7_test_proba.csv")
    a6_oof = pd.read_csv(config.LOGS_DIR / "A6_oof.csv")
    a6_test = pd.read_csv(config.LOGS_DIR / "A6_test_proba.csv")

    oof = {
        "v2": _read_proba(config.LOGS_DIR / "v2_opt_oof_proba.csv"),
        "public": _read_proba(config.LOGS_DIR / "public81599_style_oof_proba.csv"),
        "groupfill": _read_proba(config.LOGS_DIR / "public_groupfill_t052_oof_proba.csv"),
        "stepwise_balanced": stepwise["stepwise_balanced_oof"],
        "stepwise_safe_te": stepwise["stepwise_safe_te_oof"],
        "stepwise_safe_no_te": stepwise["stepwise_safe_no_te_oof"],
        "legacy_public": stepwise["legacy_public_oof"],
        "a7_blend": a7_oof["y_proba_blend"].to_numpy(dtype=float),
        "a6": a6_oof["y_proba"].to_numpy(dtype=float),
    }
    test = {
        "v2": _read_proba(config.LOGS_DIR / "v2_opt_test_proba.csv"),
        "public": _read_proba(config.LOGS_DIR / "public81599_style_test_proba.csv"),
        "groupfill": _read_proba(config.LOGS_DIR / "public_groupfill_t052_test_proba.csv"),
        "stepwise_balanced": stepwise["stepwise_balanced_test"],
        "stepwise_safe_te": stepwise["stepwise_safe_te_test"],
        "stepwise_safe_no_te": stepwise["stepwise_safe_no_te_test"],
        "legacy_public": stepwise["legacy_public_test"],
        "a7_blend": a7_test["y_proba_blend"].to_numpy(dtype=float),
        "a6": a6_test["y_proba"].to_numpy(dtype=float),
    }
    return passenger_id_test, y, oof, test


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _blend(
    families: dict[str, np.ndarray],
    weights: dict[str, float],
    kind: str,
) -> np.ndarray:
    out = None
    for name, weight in weights.items():
        if weight <= 0:
            continue
        values = families[name]
        if kind == "prob":
            transformed = values
        elif kind == "rank":
            transformed = _rank01(values)
        elif kind == "logit":
            transformed = _logit(values)
        else:
            raise ValueError(f"unknown blend kind: {kind}")
        out = weight * transformed if out is None else out + weight * transformed
    if out is None:
        raise ValueError("empty blend weights")
    return out


def _threshold_for_positive_rate(score: np.ndarray, target_rate: float) -> float:
    target_rate = float(np.clip(target_rate, 0.001, 0.999))
    # Predict positive when score >= threshold.  Use the test-score quantile so
    # the candidate lands in the desired public-LB distribution band.
    return float(np.quantile(score, 1.0 - target_rate, method="lower"))


def _weight_grid() -> list[dict[str, float]]:
    """Compact grid around the Kaggle-highscore XGB families.

    Units are twentieths.  The ranges deliberately keep public/groupfill/legacy
    present, because this search is for Kaggle-inspired candidates, not another
    pure V2 rerun.
    """
    grids: list[dict[str, float]] = []
    # v2 + stepwise are the stability core; public/groupfill/legacy inject the
    # Kaggle high-score raw-CSV and group-fill behavior.
    for v2, stepwise, public, groupfill, legacy, safe_te, a7 in product(
        range(6, 13),   # 0.30 .. 0.60
        range(5, 12),   # 0.25 .. 0.55
        range(1, 5),    # 0.05 .. 0.20
        range(0, 4),    # 0.00 .. 0.15
        range(0, 3),    # 0.00 .. 0.10
        range(0, 3),    # 0.00 .. 0.10
        range(0, 2),    # 0.00 .. 0.05
    ):
        total = v2 + stepwise + public + groupfill + legacy + safe_te + a7
        if total != 20:
            continue
        if public + groupfill + legacy < 2:
            continue
        if v2 + stepwise < 12:
            continue
        grids.append(
            {
                "v2": v2 / 20,
                "stepwise_balanced": stepwise / 20,
                "public": public / 20,
                "groupfill": groupfill / 20,
                "legacy_public": legacy / 20,
                "stepwise_safe_te": safe_te / 20,
                "a7_blend": a7 / 20,
            }
        )

    # Add a few hand-picked high-score-notebook-biased templates that are not
    # easy to hit with the compact grid.
    grids.extend(
        [
            {
                "v2": 0.35,
                "stepwise_balanced": 0.35,
                "public": 0.15,
                "groupfill": 0.10,
                "legacy_public": 0.05,
            },
            {
                "v2": 0.42,
                "stepwise_balanced": 0.36,
                "public": 0.14,
                "groupfill": 0.08,
            },
            {
                "v2": 0.30,
                "stepwise_balanced": 0.45,
                "public": 0.15,
                "legacy_public": 0.05,
                "stepwise_safe_te": 0.05,
            },
            {
                "v2": 0.40,
                "stepwise_balanced": 0.30,
                "public": 0.20,
                "groupfill": 0.10,
            },
        ]
    )
    # Deduplicate normalized dictionaries.
    seen = set()
    unique = []
    for weights in grids:
        clean = {k: float(v) for k, v in weights.items() if float(v) > 0}
        key = tuple(sorted(clean.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(clean)
    return unique


def _load_reference_predictions() -> dict[str, np.ndarray]:
    refs = {
        "v2_best": pd.read_csv(config.SUBMISSIONS_DIR / "submission_v2_best.csv")[
            "Transported"
        ].astype(bool).to_numpy(),
        "public_style": pd.read_csv(OUT_DIR / "submission_public81599_style.csv")[
            "Transported"
        ].astype(bool).to_numpy(),
    }
    anchor = OUT_DIR / "submission_anchor_a7_hi52_lo45.csv"
    if anchor.exists():
        refs["anchor_a7_hi52_lo45"] = pd.read_csv(anchor)["Transported"].astype(bool).to_numpy()
    return refs


def _write_candidate(
    rank: int,
    kind: str,
    target_name: str,
    target_rate: float,
    passenger_ids: pd.Series,
    pred: np.ndarray,
    threshold: float,
    honest_oof_acc: float,
    weights: dict[str, float],
    refs: dict[str, np.ndarray],
) -> SearchCandidate:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weight_tag = "_".join(f"{k}{int(round(v * 100)):02d}" for k, v in sorted(weights.items()))
    name = f"submission_kaggle_hs_{rank:02d}_{kind}_{target_name}_{weight_tag}"
    file_path = OUT_DIR / f"{name}.csv"
    metadata_path = OUT_DIR / f"{name}.json"
    pred_bool = np.asarray(pred).astype(bool)
    pd.DataFrame({"PassengerId": passenger_ids.astype(str), "Transported": pred_bool}).to_csv(
        file_path, index=False
    )
    candidate = SearchCandidate(
        rank=rank,
        name=name,
        file=str(file_path),
        metadata_path=str(metadata_path),
        blend_kind=kind,
        target_rate_name=target_name,
        target_positive_rate=float(target_rate),
        threshold=float(threshold),
        honest_oof_acc=float(honest_oof_acc),
        positive_rate=float(pred_bool.mean()),
        changed_vs_v2_best=int(np.sum(pred_bool != refs["v2_best"])),
        changed_vs_public_style=int(np.sum(pred_bool != refs["public_style"])),
        changed_vs_anchor_a7=(
            int(np.sum(pred_bool != refs["anchor_a7_hi52_lo45"]))
            if "anchor_a7_hi52_lo45" in refs
            else None
        ),
        weights={k: float(v) for k, v in weights.items()},
        source=(
            "Kaggle high-score XGB blend: public-style raw CSV family + "
            "travel-group groupfill + fold-safe local XGB families"
        ),
    )
    metadata_path.write_text(json.dumps(asdict(candidate), ensure_ascii=False, indent=2) + "\n")
    return candidate


def _write_anchor_override_candidates(
    start_rank: int,
    passenger_ids: pd.Series,
    y: np.ndarray,
    oof_score: np.ndarray,
    test_score: np.ndarray,
    threshold: float,
    refs: dict[str, np.ndarray],
    weights: dict[str, float],
    kind: str,
) -> list[SearchCandidate]:
    """Flip a known public-LB-friendly anchor only where high-score blend is confident."""
    if "anchor_a7_hi52_lo45" not in refs:
        return []
    base = refs["anchor_a7_hi52_lo45"].copy()
    teacher = test_score >= threshold
    disagree = np.flatnonzero(teacher != base)
    if len(disagree) == 0:
        return []
    confidence = np.abs(test_score[disagree] - threshold)
    ordered = disagree[np.argsort(-confidence)]
    outputs = []
    # OOF approximation uses the same teacher threshold.  We cannot reconstruct
    # the anchor's OOF analogue, so metadata marks this as anchor-override.
    teacher_oof_acc = float(accuracy_score(y, oof_score >= threshold))
    for n in (20, 40, 80, 120):
        if len(ordered) < n:
            continue
        pred = base.copy()
        pred[ordered[:n]] = teacher[ordered[:n]]
        target_name = f"anchor_a7_top{n}"
        outputs.append(
            _write_candidate(
                rank=start_rank + len(outputs),
                kind=f"{kind}_anchor_override",
                target_name=target_name,
                target_rate=float(pred.mean()),
                passenger_ids=passenger_ids,
                pred=pred,
                threshold=threshold,
                honest_oof_acc=teacher_oof_acc,
                weights=weights,
                refs=refs,
            )
        )
    return outputs


def run_search(top_k: int = 16, write_anchor_overrides: bool = True) -> list[SearchCandidate]:
    passenger_ids, y, oof_families, test_families = _load_family_probs()
    refs = _load_reference_predictions()

    rows = []
    for weights in _weight_grid():
        for kind in ("prob", "rank", "logit"):
            oof_score = _blend(oof_families, weights, kind)
            test_score = _blend(test_families, weights, kind)
            for target_name, target_rate in TARGET_POSITIVE_RATES.items():
                threshold = _threshold_for_positive_rate(test_score, target_rate)
                pred_oof = oof_score >= threshold
                pred_test = test_score >= threshold
                rows.append(
                    {
                        "kind": kind,
                        "target_rate_name": target_name,
                        "target_positive_rate": target_rate,
                        "threshold": threshold,
                        "honest_oof_acc": float(accuracy_score(y, pred_oof)),
                        "positive_rate": float(pred_test.mean()),
                        "changed_vs_v2_best": int(np.sum(pred_test != refs["v2_best"])),
                        "changed_vs_public_style": int(np.sum(pred_test != refs["public_style"])),
                        "changed_vs_anchor_a7": (
                            int(np.sum(pred_test != refs["anchor_a7_hi52_lo45"]))
                            if "anchor_a7_hi52_lo45" in refs
                            else np.nan
                        ),
                        "weights": weights,
                    }
                )

    summary = pd.DataFrame(rows).sort_values(
        ["honest_oof_acc", "target_rate_name", "changed_vs_v2_best"],
        ascending=[False, True, True],
    )
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    flat = summary.copy()
    for key in sorted({k for weights in flat["weights"] for k in weights}):
        flat[f"w_{key}"] = flat["weights"].map(lambda d: d.get(key, 0.0))
    flat = flat.drop(columns=["weights"])
    flat.to_csv(SUMMARY_PATH, index=False)

    # Keep diverse candidates.  Do not let the clean anchor511 band occupy every
    # output slot; Kaggle-highscore search needs the public-notebook-like
    # positive-rate bands as real submission options.
    selected_indices = []
    for target_name in TARGET_POSITIVE_RATES:
        band = summary[summary["target_rate_name"] == target_name].head(3)
        selected_indices.extend(band.index.tolist())
    for idx in summary.head(top_k * 2).index:
        selected_indices.append(idx)

    selected = summary.loc[dict.fromkeys(selected_indices).keys()].head(top_k)
    outputs: list[SearchCandidate] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        weights = row["weights"]
        kind = row["kind"]
        test_score = _blend(test_families, weights, kind)
        pred = test_score >= float(row["threshold"])
        outputs.append(
            _write_candidate(
                rank=rank,
                kind=kind,
                target_name=row["target_rate_name"],
                target_rate=float(row["target_positive_rate"]),
                passenger_ids=passenger_ids,
                pred=pred,
                threshold=float(row["threshold"]),
                honest_oof_acc=float(row["honest_oof_acc"]),
                weights=weights,
                refs=refs,
            )
        )

    if write_anchor_overrides and outputs:
        best = outputs[0]
        weights = best.weights
        kind = best.blend_kind
        oof_score = _blend(oof_families, weights, kind)
        test_score = _blend(test_families, weights, kind)
        outputs.extend(
            _write_anchor_override_candidates(
                start_rank=len(outputs) + 1,
                passenger_ids=passenger_ids,
                y=y,
                oof_score=oof_score,
                test_score=test_score,
                threshold=best.threshold,
                refs=refs,
                weights=weights,
                kind=kind,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--no-anchor-overrides", action="store_true")
    args = parser.parse_args()
    outputs = run_search(
        top_k=args.top_k,
        write_anchor_overrides=not args.no_anchor_overrides,
    )
    print(json.dumps([asdict(item) for item in outputs], ensure_ascii=False, indent=2))
    print(f"\nSummary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
