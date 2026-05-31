"""Small public-style refinements around the current best Umanglodaya SMOTE XGB.

Current Kaggle feedback shows:

- ``submission_public81599_style.csv`` is strong but weaker;
- ``submission_umanglodaya_xgb_smote.csv`` is stronger;
- the two files differ on only a small set of test rows.

This script keeps the SMOTE submission as the anchor and flips only the rows
where the no-SMOTE public-style model disagrees and has a stronger confidence
margin. The rule is selected with the analogous OOF probabilities.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from . import config


OUT_DIR = config.REPORTS_DIR / "submission_candidates"
PUBLIC_SUB = OUT_DIR / "submission_public81599_style.csv"
SMOTE_SUB = OUT_DIR / "submission_umanglodaya_xgb_smote.csv"
PUBLIC_OOF = config.LOGS_DIR / "public81599_style_oof_proba.csv"
PUBLIC_TEST = config.LOGS_DIR / "public81599_style_test_proba.csv"
SMOTE_OOF = config.LOGS_DIR / "umanglodaya_xgb_smote_oof_proba.csv"
SMOTE_TEST = config.LOGS_DIR / "umanglodaya_xgb_smote_test_proba.csv"
RAW_TRAIN = Path("/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw/train.csv")


@dataclass
class Refinement:
    name: str
    file: str
    metadata_path: str
    strategy: str
    n_flipped: int
    positive_rate: float
    oof_accuracy: float
    oof_delta_vs_smote: float
    changed_vs_smote: int
    changed_vs_public: int


def _load_bool(path: Path) -> np.ndarray:
    return pd.read_csv(path)["Transported"].astype(bool).to_numpy()


def _write_candidate(
    *,
    name: str,
    pred: np.ndarray,
    strategy: str,
    n_flipped: int,
    oof_accuracy: float,
    smote_oof_accuracy: float,
    smote_test_pred: np.ndarray,
    public_test_pred: np.ndarray,
    passenger_ids: pd.Series,
) -> Refinement:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUT_DIR / f"{name}.csv"
    metadata_path = OUT_DIR / f"{name}.json"
    pred_bool = np.asarray(pred).astype(bool)
    pd.DataFrame({"PassengerId": passenger_ids.astype(str), "Transported": pred_bool}).to_csv(file_path, index=False)
    result = Refinement(
        name=name,
        file=str(file_path),
        metadata_path=str(metadata_path),
        strategy=strategy,
        n_flipped=int(n_flipped),
        positive_rate=float(pred_bool.mean()),
        oof_accuracy=float(oof_accuracy),
        oof_delta_vs_smote=float(oof_accuracy - smote_oof_accuracy),
        changed_vs_smote=int(np.sum(pred_bool != smote_test_pred)),
        changed_vs_public=int(np.sum(pred_bool != public_test_pred)),
    )
    metadata_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
    return result


def _orders(public_score: np.ndarray, smote_score: np.ndarray, public_pred: np.ndarray, smote_pred: np.ndarray):
    disagree = np.flatnonzero(public_pred != smote_pred)
    public_margin = np.abs(public_score[disagree] - 0.5)
    smote_margin = np.abs(smote_score[disagree] - 0.5)
    return {
        "public_conf": disagree[np.argsort(-public_margin)],
        "public_minus_smote_conf": disagree[np.argsort(-(public_margin - smote_margin))],
        "smote_uncertain": disagree[np.argsort(smote_margin)],
    }


def run_refinement() -> list[Refinement]:
    y = pd.read_csv(RAW_TRAIN)["Transported"].astype(int).to_numpy()
    public_oof = pd.read_csv(PUBLIC_OOF)["y_proba"].to_numpy(float)
    public_test = pd.read_csv(PUBLIC_TEST)["y_proba"].to_numpy(float)
    smote_oof = pd.read_csv(SMOTE_OOF)["y_proba"].to_numpy(float)
    smote_test = pd.read_csv(SMOTE_TEST)["y_proba"].to_numpy(float)
    passenger_ids = pd.read_csv(SMOTE_SUB)["PassengerId"]

    public_oof_pred = public_oof >= 0.5
    public_test_pred = public_test >= 0.5
    smote_oof_pred = smote_oof >= 0.5
    smote_test_pred = smote_test >= 0.5
    smote_oof_accuracy = float(accuracy_score(y, smote_oof_pred))

    oof_orders = _orders(public_oof, smote_oof, public_oof_pred, smote_oof_pred)
    test_orders = _orders(public_test, smote_test, public_test_pred, smote_test_pred)

    outputs: list[Refinement] = []
    recipes = [
        ("public_conf", 10),
        ("public_conf", 13),
        ("public_conf", 15),
        ("public_conf", 20),
        ("public_minus_smote_conf", 10),
        ("public_minus_smote_conf", 13),
        ("public_minus_smote_conf", 15),
        ("public_minus_smote_conf", 25),
        ("smote_uncertain", 5),
    ]
    for strategy, n in recipes:
        if len(test_orders[strategy]) < n or len(oof_orders[strategy]) < n:
            continue
        pred_oof = smote_oof_pred.copy()
        pred_oof[oof_orders[strategy][:n]] = public_oof_pred[oof_orders[strategy][:n]]
        pred_test = smote_test_pred.copy()
        pred_test[test_orders[strategy][:n]] = public_test_pred[test_orders[strategy][:n]]
        outputs.append(
            _write_candidate(
                name=f"submission_umanglodaya_refine_{strategy}_{n:02d}",
                pred=pred_test,
                strategy=strategy,
                n_flipped=n,
                oof_accuracy=accuracy_score(y, pred_oof),
                smote_oof_accuracy=smote_oof_accuracy,
                smote_test_pred=smote_test_pred,
                public_test_pred=public_test_pred,
                passenger_ids=passenger_ids,
            )
        )

    summary_path = config.LOGS_DIR / "umanglodaya_public_refine_summary.csv"
    pd.DataFrame([asdict(item) for item in outputs]).sort_values(
        ["oof_accuracy", "n_flipped"], ascending=[False, True]
    ).to_csv(summary_path, index=False)
    return outputs


def main() -> None:
    outputs = run_refinement()
    print(json.dumps([asdict(item) for item in outputs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
