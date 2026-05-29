"""Load the shared `common` preprocessed DataFrames and produce XGBoost-ready inputs.

The legacy `processed/xgboost/preprocessed_xgboost.joblib` bundle is *not* used
here.  It stores the design matrix as an `object`-dtype NumPy array with 107
one-hot columns, which is actively harmful to XGBoost's `hist` splitting.

Instead we start from `processed/common/preprocessed_common.joblib`, which is a
clean DataFrame with 51 engineered features and correct dtypes (Int64, Float64,
StringDtype).  We drop the high-cardinality audit columns, keep the engineered
binary / numeric features, and convert categorical columns to pandas
``category`` so that XGBoost 2.x can consume them natively via
``enable_categorical=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Raw common DataFrame access
# ---------------------------------------------------------------------------


@dataclass
class CommonData:
    """Aggregate of the shared pre-processed data used by XGBoost."""

    train: pd.DataFrame
    test: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    passenger_id_train: pd.Series
    passenger_id_test: pd.Series
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert len(self.train) == len(self.y) == len(self.groups), "len mismatch"


def load_common() -> CommonData:
    """Read `preprocessed_common.joblib` and return a :class:`CommonData`.

    The underlying file is pickled with scikit-learn 1.6; we only read DataFrame
    payloads so the sklearn version mismatch warning is a non-issue.
    """
    if not config.COMMON_BUNDLE.exists():
        raise FileNotFoundError(
            f"Missing common bundle at {config.COMMON_BUNDLE}.  Please run the shared "
            "preprocessing pipeline first."
        )
    b = joblib.load(config.COMMON_BUNDLE)
    train: pd.DataFrame = b["common_train"].reset_index(drop=True).copy()
    test: pd.DataFrame = b["common_test"].reset_index(drop=True).copy()
    y: pd.Series = pd.Series(b["y_train"]).reset_index(drop=True).astype("int8")
    groups = train[config.GROUP_COLUMN].reset_index(drop=True).astype("string")

    return CommonData(
        train=train,
        test=test,
        y=y,
        groups=groups,
        passenger_id_train=train[config.ID_COLUMN].copy(),
        passenger_id_test=test[config.ID_COLUMN].copy(),
        stats=b.get("stats", {}),
    )


# ---------------------------------------------------------------------------
# Shared (non fold-aware) feature preparation
# ---------------------------------------------------------------------------


def _coerce_categoricals(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """Cast the configured categorical columns to pandas ``category`` dtype.

    XGBoost 2.x accepts ``category`` columns natively when ``enable_categorical``
    is ``True``.  That avoids the 107-dim one-hot blow-up that the legacy bundle
    produced and lets XGBoost pick the optimal subset split per level.
    """
    d = df.copy()
    for col in categories:
        if col not in d.columns:
            continue
        d[col] = d[col].astype("string").fillna("__MISSING__").astype("category")
    return d


def _align_category_levels(train: pd.DataFrame, test: pd.DataFrame, categories: Sequence[str]) -> None:
    """Ensure train and test share the same category levels in-place.

    Without this, a value seen only in test (or in a CV valid fold) would be
    treated as an *unknown* by XGBoost.  Taking the union keeps the learned
    split tables valid.
    """
    for col in categories:
        if col not in train.columns or col not in test.columns:
            continue
        combined = pd.api.types.union_categoricals(
            [train[col].astype("category"), test[col].astype("category")],
            sort_categories=True,
        )
        levels = combined.categories
        train[col] = pd.Categorical(train[col], categories=levels)
        test[col] = pd.Categorical(test[col], categories=levels)


def _log1p_spend(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[f"log1p_{c}"] = np.log1p(d[c].astype("float64").fillna(0.0))
    if "TotalSpend" in d.columns:
        d["log1p_TotalSpend"] = np.log1p(d["TotalSpend"].astype("float64").fillna(0.0))
    return d


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap non-linear interactions that boosted trees still enjoy."""
    d = df.copy()
    if {"TotalSpend", "Age"} <= set(d.columns):
        d["SpendPerAge"] = d["TotalSpend"].astype("float64") / (d["Age"].astype("float64") + 1.0)
    if {"TotalSpend", "GroupSize"} <= set(d.columns):
        d["SpendPerGroupMember"] = d["TotalSpend"].astype("float64") / d["GroupSize"].astype("float64").clip(lower=1)
    if {"LuxurySpend", "BasicSpend"} <= set(d.columns):
        d["LuxuryMinusBasic"] = d["LuxurySpend"].astype("float64") - d["BasicSpend"].astype("float64")
    # Rule-based CryoSleep anomaly flag:  if CryoSleep=True but spend > 0, row is
    # inconsistent.  The flag itself is almost always 0 after cleaning, but the
    # model can still learn a useful interaction when it is non-zero.
    if {"CryoSleep", "TotalSpend"} <= set(d.columns):
        cryo_true = d["CryoSleep"].astype("string").eq("True")
        d["CryoSleepSpendAnomaly"] = (cryo_true & (d["TotalSpend"].astype("float64") > 0)).astype("int8")
    # Combined missingness count — often captures "chaotic" rows.
    miss_cols = [c for c in d.columns if c.endswith("Missing")]
    if miss_cols:
        d["MissingCount"] = d[miss_cols].sum(axis=1).astype("int16")
    return d


def build_xgb_features(common: CommonData) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Turn the raw common DataFrames into model-ready inputs.

    Returns
    -------
    X_train, X_test, y, groups
        All aligned by row index.  Categorical columns are ``category`` dtype
        and share the same level set between train and test.
    """
    train = common.train.copy()
    test = common.test.copy()

    train = _log1p_spend(train, config.SPEND_COLUMNS)
    test = _log1p_spend(test, config.SPEND_COLUMNS)
    train = _add_interaction_features(train)
    test = _add_interaction_features(test)

    drop = [c for c in config.DROP_COLUMNS if c in train.columns]
    train = train.drop(columns=drop)
    test = test.drop(columns=[c for c in drop if c in test.columns])

    train = _coerce_categoricals(train, config.CATEGORICAL_COLUMNS)
    test = _coerce_categoricals(test, config.CATEGORICAL_COLUMNS)
    _align_category_levels(train, test, config.CATEGORICAL_COLUMNS)

    # All remaining Int64 / string columns that are NOT in CATEGORICAL_COLUMNS
    # must still be numeric.  Coerce any StringDtype leftovers to numeric where
    # possible; otherwise fall back to category to avoid XGBoost errors.
    for col in train.columns:
        if col in config.CATEGORICAL_COLUMNS:
            continue
        if train[col].dtype.name == "string" or train[col].dtype == object:
            # Unknown string column – safest is to treat it as category.
            train[col] = train[col].astype("string").fillna("__MISSING__").astype("category")
            if col in test.columns:
                test[col] = test[col].astype("string").fillna("__MISSING__").astype("category")
        elif str(train[col].dtype).startswith("Int"):
            train[col] = train[col].astype("int64")
            if col in test.columns:
                test[col] = test[col].astype("int64")

    # Align column order
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    return train, test, common.y, common.groups


def summarise_schema(X: pd.DataFrame) -> dict:
    """Return a compact description of the XGBoost input schema (for logs)."""
    dt_counts: dict[str, int] = {}
    for col in X.columns:
        dt_counts[str(X[col].dtype)] = dt_counts.get(str(X[col].dtype), 0) + 1
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    return {
        "n_rows": int(len(X)),
        "n_cols": int(X.shape[1]),
        "dtype_counts": dt_counts,
        "n_categorical": len(cats),
        "categorical_columns": cats,
    }
