"""V2 pipeline: push this XGBoost model to the leaderboard limit.

Key decisions (all informed by the diagnosis in reports/xgboost/logs/v2_*):

1. **No target encoding at all.**  Plain TE leaked OOF (self-leak) without
   helping the LB (0.80406 < 0.80804 old baseline).  LOO TE actually
   amplifies the leak because ``sum_y - y_self`` makes the encoding a direct
   function of the target within a fold.  OOF TE gave zero OOF lift over
   "none" (both 0.8162) so it adds risk with no reward.
2. **Keep fold-aware SurnameFreq + CabinNumBin refit.**  These are pure
   statistics, do not see y, and add ~0.002 honestly.
3. **Native categorical dtype** (not the legacy 107-column one-hot).  XGBoost
   handles categorical splits natively, preserves the gain profile we saw in
   A0's importance (Deck_E / Side_P / DeckSide_*) as a single gain budget on
   each categorical feature.
4. **STRONG_PARAMS recipe** (lr=0.03, depth=6, min_child_weight=3, gamma=0.5,
   reg_alpha=0.5, reg_lambda=1.5, colsample_bylevel=0.8).  This was the best
   non-Optuna recipe and sits between the Optuna-overfitted V2_strong(lr=.044,
   d=4) and the vanilla baseline.  Depth=6 matches the legacy LB=0.80804
   config.
5. **15-seed bagging** averaged over probabilities.  More seeds => more
   variance reduction, especially important now that we are not masking it
   with TE-induced bias.
6. **Pseudo-labeling (optional).**  Take test rows whose first-pass proba
   >= 0.98 or <= 0.02, treat them as labeled, concatenate with train, retrain
   with the same 15-seed procedure.  Small training-set boost with negligible
   risk at those high confidence thresholds.
7. **Threshold = 0.5 hardcoded.**  No OOF tuning of the decision threshold
   (that was one of the OOF-overfitting vectors).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config, data
from .ensemble import EnsembleReport, run_multi_seed
from .train import StageConfig, run_cv


logger = logging.getLogger("xgb.v2")

V2_SEEDS = (42, 2024, 7, 1337, 88, 13, 99, 2025, 314, 271, 1618, 2718, 5, 50, 500)


@dataclass
class V2Output:
    oof_acc: float
    oof_logloss: float
    oof_auc: float
    seeds_used: list[int]
    submission_path: Path
    oof_proba_path: Path
    test_proba_path: Path


def build_stage(params: dict | None = None, name: str = "V2") -> StageConfig:
    """Single source of truth for the V2 pipeline configuration.

    No target encoding, no group aggregates, no surname rate (that needs an
    OOF-based implementation before it is safe to add).  Just the clean
    fold-aware feature set.
    """
    p = dict(config.STRONG_PARAMS) if params is None else dict(params)
    return StageConfig(
        name=name,
        group_aware_cv=True,
        fold_aware_surname=True,
        fold_aware_cabin_bin=True,
        target_encode_cols=(),
        target_encode_mode="none",
        use_surname_rate=False,
        use_oof_target_encoding=False,
        use_group_aggregates=False,
        use_common_features=True,
        params_override=p,
        use_early_stopping=True,
        description="V2: no-TE + fold-aware surname/cabin + STRONG_PARAMS",
    )


def _save_submission(
    passenger_ids: Iterable[str],
    proba: np.ndarray,
    threshold: float,
    path: Path,
) -> Path:
    preds = (np.asarray(proba) >= threshold)
    df = pd.DataFrame({
        "PassengerId": list(passenger_ids),
        "Transported": preds.astype(bool),
    })
    df.to_csv(path, index=False)
    return path


def _save_proba(
    passenger_ids: Iterable[str],
    proba: np.ndarray,
    path: Path,
) -> Path:
    pd.DataFrame({"PassengerId": list(passenger_ids), "y_proba": np.asarray(proba)}).to_csv(
        path, index=False
    )
    return path


def run_v2(
    seeds: Iterable[int] = V2_SEEDS,
    threshold: float = 0.5,
    params: dict | None = None,
    out_tag: str = "v2",
) -> V2Output:
    """Run the V2 pipeline and write ``submission_{out_tag}.csv``."""
    stage = build_stage(params=params, name=f"V2_{out_tag}")
    ensemble = run_multi_seed(stage, seeds=seeds, verbose=False)

    # OOF metrics on the averaged probability
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    oof_acc = float(accuracy_score(ensemble.oof_true, (ensemble.oof_proba >= threshold).astype(int)))
    oof_logloss = float(log_loss(ensemble.oof_true, ensemble.oof_proba))
    oof_auc = float(roc_auc_score(ensemble.oof_true, ensemble.oof_proba))

    logger.info(
        "[V2/%s] 15-seed bagged OOF acc=%.4f  logloss=%.4f  auc=%.4f  (threshold=%.2f)",
        out_tag, oof_acc, oof_logloss, oof_auc, threshold,
    )
    # Per-seed OOF for sanity-checking bagging stability
    per_seed = []
    for m in ensemble.members:
        sacc = float(accuracy_score(m.oof_true, (m.oof_proba >= threshold).astype(int)))
        per_seed.append(sacc)
    logger.info("[V2/%s] per-seed OOF acc: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                out_tag, float(np.mean(per_seed)), float(np.std(per_seed)),
                float(np.min(per_seed)), float(np.max(per_seed)))

    sub_path = _save_submission(
        ensemble.passenger_id_test,
        ensemble.test_proba,
        threshold,
        config.SUBMISSIONS_DIR / f"submission_{out_tag}.csv",
    )
    oof_path = _save_proba(
        list(data.load_common().passenger_id_train),
        ensemble.oof_proba,
        config.LOGS_DIR / f"{out_tag}_oof_proba.csv",
    )
    test_path = _save_proba(
        ensemble.passenger_id_test,
        ensemble.test_proba,
        config.LOGS_DIR / f"{out_tag}_test_proba.csv",
    )
    logger.info("[V2/%s] wrote %s", out_tag, sub_path)
    return V2Output(
        oof_acc=oof_acc,
        oof_logloss=oof_logloss,
        oof_auc=oof_auc,
        seeds_used=ensemble.seeds_used,
        submission_path=sub_path,
        oof_proba_path=oof_path,
        test_proba_path=test_path,
    )


def run_v2_with_pseudo_labeling(
    seeds: Iterable[int] = V2_SEEDS,
    threshold: float = 0.5,
    pos_cutoff: float = 0.98,
    neg_cutoff: float = 0.02,
    params: dict | None = None,
) -> V2Output:
    """Run V2, take high-confidence test rows as labeled, retrain with them.

    The pseudo-labels come from the 15-seed averaged probability.  We only
    include rows in the extreme tails (default: top 2% and bottom 2% by
    confidence) so the contamination risk is tiny.  This is a well-known
    Kaggle trick for Spaceship Titanic where many test rows are easy.
    """
    base = run_v2(seeds=seeds, threshold=threshold, params=params, out_tag="v2_pass1")

    # Build pseudo-labeled frames
    common = data.load_common()
    probas = pd.read_csv(base.test_proba_path)
    high_mask = (probas["y_proba"] >= pos_cutoff) | (probas["y_proba"] <= neg_cutoff)
    logger.info(
        "[V2/pseudo] high-confidence test rows: %d / %d  (pos>=%.2f | neg<=%.2f)",
        int(high_mask.sum()), len(probas), pos_cutoff, neg_cutoff,
    )
    if int(high_mask.sum()) < 100:
        logger.warning("[V2/pseudo] too few pseudo-labels, skipping retrain")
        return base

    pseudo_ids = probas.loc[high_mask, "PassengerId"].tolist()
    pseudo_labels_int = (probas.loc[high_mask, "y_proba"] >= 0.5).astype(int).tolist()

    # Build the augmented train frame: original train + high-confidence test rows
    test_df = common.test.copy()
    pseudo_rows = test_df[test_df[config.ID_COLUMN].isin(pseudo_ids)].reset_index(drop=True)
    pseudo_rows[config.TARGET_COLUMN] = pseudo_rows[config.ID_COLUMN].map(
        dict(zip(pseudo_ids, pseudo_labels_int))
    ).astype(bool)

    augmented_train = pd.concat([common.train, pseudo_rows], axis=0, ignore_index=True)
    augmented_y = pd.concat([
        common.y.astype("int8"),
        pseudo_rows[config.TARGET_COLUMN].astype(bool).astype("int8"),
    ], axis=0, ignore_index=True).reset_index(drop=True)
    # Use the original GroupID from each pseudo row (test groups do not overlap
    # with train groups, so StratifiedGroupKFold still treats them as fresh)
    augmented_groups = augmented_train[config.GROUP_COLUMN].astype("string").reset_index(drop=True)

    logger.info(
        "[V2/pseudo] augmented train: %d rows (orig %d + pseudo %d)",
        len(augmented_train), len(common.train), len(pseudo_rows),
    )

    # Monkey-patch data.load_common so downstream modules see the augmented set
    _orig_load_common = data.load_common
    augmented_common = data.CommonData(
        train=augmented_train.reset_index(drop=True),
        test=common.test.reset_index(drop=True),
        y=augmented_y,
        groups=augmented_groups,
        passenger_id_train=augmented_train[config.ID_COLUMN].copy(),
        passenger_id_test=common.test[config.ID_COLUMN].copy(),
        stats=common.stats,
    )

    def _patched_load_common():
        return augmented_common

    data.load_common = _patched_load_common  # type: ignore[assignment]
    try:
        pass2 = run_v2(seeds=seeds, threshold=threshold, params=params, out_tag="v2_pseudo")
    finally:
        data.load_common = _orig_load_common  # type: ignore[assignment]

    return pass2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pseudo", action="store_true",
                        help="Also run a pseudo-labeling pass using V2 pass-1 as teacher")
    parser.add_argument("--tag", type=str, default="v2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    seeds = args.seeds if args.seeds else V2_SEEDS
    logger.info("V2 seeds: %s  (N=%d)", list(seeds), len(list(seeds)))
    logger.info("Threshold=%.2f  pseudo=%s", args.threshold, args.pseudo)

    if args.pseudo:
        out = run_v2_with_pseudo_labeling(seeds=seeds, threshold=args.threshold)
    else:
        out = run_v2(seeds=seeds, threshold=args.threshold, out_tag=args.tag)

    logger.info(
        "=== DONE ===  OOF acc=%.4f logloss=%.4f auc=%.4f  submission=%s",
        out.oof_acc, out.oof_logloss, out.oof_auc, out.submission_path,
    )


if __name__ == "__main__":
    main()
