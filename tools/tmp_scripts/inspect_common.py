"""Deep inspection of common bundle: columns, dtypes, sample values, GroupID coverage."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/shenyijie/Desktop/MLWP project")


def main() -> None:
    b = joblib.load(ROOT / "processed" / "common" / "preprocessed_common.joblib")
    train: pd.DataFrame = b["common_train"]
    test: pd.DataFrame = b["common_test"]
    y: pd.Series = b["y_train"]
    stats: dict = b["stats"]

    print(f"train shape: {train.shape}, test shape: {test.shape}, y shape: {y.shape}")
    print(f"y dist: {y.value_counts().to_dict()}")
    print()
    print("=" * 100)
    print("ALL COLUMNS WITH DTYPE + SAMPLE + MISSING:")
    print("=" * 100)
    for col in train.columns:
        dt = train[col].dtype
        nmiss_tr = int(train[col].isna().sum())
        nmiss_te = int(test[col].isna().sum()) if col in test.columns else -1
        nuniq = int(train[col].nunique(dropna=True))
        sample = train[col].dropna().head(3).tolist()
        print(f"  {col:30s} dt={str(dt):22s} uniq={nuniq:5d} miss_tr={nmiss_tr:4d} miss_te={nmiss_te:4d} sample={sample}")

    print()
    print("=" * 100)
    print("GroupID coverage:")
    print("=" * 100)
    if "GroupID" in train.columns:
        g_tr = train["GroupID"]
        g_te = test["GroupID"]
        print(f"train unique groups: {g_tr.nunique()}")
        print(f"test unique groups : {g_te.nunique()}")
        print(f"overlap: {len(set(g_tr.unique()) & set(g_te.unique()))}")
        # group size distribution in train
        print(f"group size in train: {g_tr.value_counts().value_counts().sort_index().to_dict()}")

    print()
    print("=" * 100)
    print("stats keys + summary:")
    print("=" * 100)
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}: dict(len={len(v)}) example={dict(list(v.items())[:3])}")
        elif isinstance(v, (list, tuple, np.ndarray)):
            print(f"  {k}: {type(v).__name__}(len={len(v)}) first={v[0] if len(v) else None}")
        else:
            print(f"  {k}: {type(v).__name__}={v}")


if __name__ == "__main__":
    main()
