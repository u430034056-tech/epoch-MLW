"""Diagnose why A5 (rule+threshold) and A6 (multi-seed) did not help.

Checks
------
1. For A4's OOF:
   - How many rows would the CryoSleep rule flip?  (boost-mask + penalty-mask)
   - Among flipped rows, how many flips match the true label?
   - What's the optimal threshold scan for pure A4 (no rule)?
2. For A6's members:
   - Per-seed OOF accuracy.
   - Pairwise Pearson correlation between seed OOF probabilities.
   - Averaged OOF accuracy at best threshold.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.xgboost import config
from src.xgboost.data import load_common


def main() -> None:
    # -----------------------------------------
    # A4 OOF
    # -----------------------------------------
    a4_dir = config.LOGS_DIR / "A4"
    a4_oof = pd.read_csv(a4_dir / "A4_oof.csv")
    common = load_common()
    train = common.train.reset_index(drop=True)

    cryo = train["CryoSleep"].astype("string")
    spend = train["TotalSpend"].astype("float64").fillna(0.0)
    boost_mask = (cryo == "True") & (spend == 0.0)
    penalty_mask = (cryo == "True") & (spend > 0.0)
    print(f"Rows in boost mask (CryoSleep=True & spend=0): {boost_mask.sum()}")
    print(f"  target mean there: {a4_oof.loc[boost_mask, 'y_true'].mean():.4f}")
    print(f"  model mean there : {a4_oof.loc[boost_mask, 'y_proba'].mean():.4f}")
    print(f"Rows in penalty mask (CryoSleep=True & spend>0): {penalty_mask.sum()}")
    if penalty_mask.sum():
        print(f"  target mean there: {a4_oof.loc[penalty_mask, 'y_true'].mean():.4f}")
        print(f"  model mean there : {a4_oof.loc[penalty_mask, 'y_proba'].mean():.4f}")

    base_pred = (a4_oof["y_proba"].values >= 0.5).astype(int)
    boosted = a4_oof["y_proba"].values.copy()
    boosted[boost_mask.values] += 0.05
    boosted[penalty_mask.values] -= 0.10
    rule_pred = (boosted >= 0.5).astype(int)
    flipped = base_pred != rule_pred
    print(f"\nPredictions flipped by rule: {flipped.sum()}")
    if flipped.sum():
        y_true = a4_oof["y_true"].values
        correct_flip = (rule_pred[flipped] == y_true[flipped]).sum()
        wrong_flip = flipped.sum() - correct_flip
        print(f"  correct flips: {correct_flip}, wrong flips: {wrong_flip}")
        delta = correct_flip - wrong_flip
        print(f"  net delta on acc: {delta}/{len(y_true)} = {delta/len(y_true):+.4f}")

    print("\n--- pure threshold scan for A4 ---")
    y_true = a4_oof["y_true"].values
    y_proba = a4_oof["y_proba"].values
    scan = []
    for t in np.arange(0.30, 0.70 + 1e-9, 0.005):
        scan.append((t, accuracy_score(y_true, (y_proba >= t).astype(int))))
    scan_df = pd.DataFrame(scan, columns=["threshold", "accuracy"])
    best_row = scan_df.loc[scan_df["accuracy"].idxmax()]
    print(f"best pure threshold: t={best_row['threshold']:.3f}, acc={best_row['accuracy']:.4f}")
    print(f"acc at 0.50       : {accuracy_score(y_true, (y_proba >= 0.50).astype(int)):.4f}")

    # -----------------------------------------
    # A6 multi-seed inspection
    # -----------------------------------------
    print("\n--- A6 multi-seed diagnostics ---")
    a6_oof_path = config.LOGS_DIR / "A6_oof.csv"
    if a6_oof_path.exists():
        a6 = pd.read_csv(a6_oof_path)
        print(f"A6 OOF at 0.5 (raw) : {accuracy_score(a6['y_true'], (a6['y_proba_raw'] >= 0.5).astype(int)):.4f}")
        print(f"A6 OOF at 0.5 (rule): {accuracy_score(a6['y_true'], (a6['y_proba_rule'] >= 0.5).astype(int)):.4f}")
        # pure threshold on raw a6
        accs = [(t, accuracy_score(a6["y_true"], (a6["y_proba_raw"] >= t).astype(int)))
                for t in np.arange(0.30, 0.70 + 1e-9, 0.005)]
        accs_df = pd.DataFrame(accs, columns=["t", "acc"])
        best = accs_df.loc[accs_df["acc"].idxmax()]
        print(f"A6 pure threshold best: t={best['t']:.3f}, acc={best['acc']:.4f}")


if __name__ == "__main__":
    main()
