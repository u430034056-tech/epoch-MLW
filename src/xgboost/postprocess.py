"""Post-processing helpers: threshold scan + rule-based corrections.

Kaggle's Spaceship Titanic metric is *accuracy*, so picking a threshold other
than 0.5 is legitimate and usually worth 0.2–0.5% on the public LB.  In
addition, several hard physical constraints (CryoSleep passengers cannot spend
money) can be used as tie-breakers when the model probability is close to 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .data import CommonData


@dataclass
class ThresholdResult:
    best_threshold: float
    best_accuracy: float
    scan: pd.DataFrame


def scan_threshold(y_true: np.ndarray, y_proba: np.ndarray, step: float = 0.01) -> ThresholdResult:
    """Brute-force search the optimal probability threshold on OOF data.

    Returns a DataFrame of (threshold, accuracy) rows in addition to the best
    pair so we can plot the sensitivity.
    """
    thresholds = np.arange(0.3, 0.7 + 1e-9, step)
    accs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        accs.append(accuracy_score(y_true, preds))
    scan = pd.DataFrame({"threshold": thresholds, "accuracy": accs})
    idx = int(scan["accuracy"].idxmax())
    return ThresholdResult(
        best_threshold=float(scan.loc[idx, "threshold"]),
        best_accuracy=float(scan.loc[idx, "accuracy"]),
        scan=scan,
    )


def apply_cryosleep_rule(
    proba: np.ndarray,
    raw: pd.DataFrame,
    boost: float = 0.0,
    penalty: float = 0.0,
) -> np.ndarray:
    """Optionally bias probabilities using the CryoSleep consistency signal.

    Default ``boost=0, penalty=0`` is a no-op because the diagnostic on our OOF
    showed that naive additive nudges hurt accuracy by about 0.002 — the model
    already captures the CryoSleep signal through the ``CryoSleep`` category
    and the ``IsZeroSpend`` feature.  The knob is kept for ablation purposes
    and for downstream teammates who may want to experiment.

    * CryoSleep=True + TotalSpend==0 → ``proba += boost``
    * CryoSleep=True + TotalSpend>0  → ``proba -= penalty``

    Values are clipped to ``[1e-6, 1-1e-6]`` to stay log-loss-safe.
    """
    if "CryoSleep" not in raw.columns or (boost == 0.0 and penalty == 0.0):
        return proba.astype("float32").copy()
    out = proba.astype("float32").copy()

    cryo_true = raw["CryoSleep"].astype("string").eq("True").values
    total_spend = raw.get("TotalSpend")
    if total_spend is None:
        return out
    spend = total_spend.astype("float64").fillna(0.0).values

    boost_mask = cryo_true & (spend == 0.0)
    penalty_mask = cryo_true & (spend > 0.0)
    out[boost_mask] = np.clip(out[boost_mask] + boost, 1e-6, 1 - 1e-6)
    out[penalty_mask] = np.clip(out[penalty_mask] - penalty, 1e-6, 1 - 1e-6)
    return out


def diagnose_rule(
    proba: np.ndarray,
    y_true: np.ndarray,
    raw: pd.DataFrame,
    boost: float = 0.05,
    penalty: float = 0.10,
) -> dict:
    """Return how many predictions a CryoSleep rule would flip and whether
    the flips are correct.  Lets us audit the rule before deploying it.
    """
    base_pred = (proba >= 0.5).astype(int)
    rule_proba = apply_cryosleep_rule(proba, raw, boost=boost, penalty=penalty)
    rule_pred = (rule_proba >= 0.5).astype(int)
    flipped = base_pred != rule_pred
    correct = int(((rule_pred == y_true) & flipped).sum())
    wrong = int(flipped.sum() - correct)
    base_acc = float(accuracy_score(y_true, base_pred))
    rule_acc = float(accuracy_score(y_true, rule_pred))
    return dict(
        boost=boost,
        penalty=penalty,
        flipped=int(flipped.sum()),
        correct_flip=correct,
        wrong_flip=wrong,
        delta_correct=correct - wrong,
        base_acc=base_acc,
        rule_acc=rule_acc,
        delta_acc=rule_acc - base_acc,
    )


def make_submission(
    proba: np.ndarray,
    passenger_ids: Iterable[str],
    threshold: float,
) -> pd.DataFrame:
    """Build the Kaggle-ready submission DataFrame.

    The competition expects columns ``PassengerId`` and ``Transported`` with
    boolean-like values.
    """
    ids = list(passenger_ids)
    preds = (np.asarray(proba) >= threshold).astype(bool)
    return pd.DataFrame({"PassengerId": ids, "Transported": preds})
