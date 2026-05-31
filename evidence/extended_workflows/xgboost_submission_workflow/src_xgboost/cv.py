"""Cross-validation scaffolding for the Spaceship Titanic XGBoost model.

Key design choices
------------------
* ``StratifiedGroupKFold``: passengers in the same ``GroupID`` (often family /
  travel companions) share strong features.  Splitting them between train and
  valid causes the inflated 0.8132 CV we currently observe, so we force the
  split to respect ``GroupID``.
* ``shuffle=True`` with a controllable seed so we can do multi-seed bagging.
* The iterator returns *both* the raw ``common`` DataFrame slice (so fold-aware
  features can refit) and the already-engineered model matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


@dataclass
class FoldSlice:
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


def make_folds(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> list[FoldSlice]:
    """Generate ``n_splits`` StratifiedGroupKFold splits respecting ``groups``.

    The returned list is materialised so iteration is deterministic and cheap.
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: list[FoldSlice] = []
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups.astype("string"))
    for i, (train_idx, valid_idx) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr, g_arr)):
        out.append(FoldSlice(fold=i, train_idx=train_idx, valid_idx=valid_idx))
    return out


def fold_health_report(folds: Sequence[FoldSlice], y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Summarise class balance and group overlap for each fold."""
    rows = []
    y_arr = np.asarray(y)
    g_arr = np.asarray(groups.astype("string"))
    for f in folds:
        g_train = set(g_arr[f.train_idx])
        g_valid = set(g_arr[f.valid_idx])
        overlap = len(g_train & g_valid)
        rows.append(
            dict(
                fold=f.fold,
                n_train=len(f.train_idx),
                n_valid=len(f.valid_idx),
                train_pos_rate=float(y_arr[f.train_idx].mean()),
                valid_pos_rate=float(y_arr[f.valid_idx].mean()),
                group_overlap=overlap,
            )
        )
    return pd.DataFrame(rows)
