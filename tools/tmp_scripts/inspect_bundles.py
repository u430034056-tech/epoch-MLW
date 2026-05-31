"""Inspect processed joblib bundles to know what we actually have."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path("/Users/shenyijie/Desktop/MLWP project")


def describe(name: str, obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame{obj.shape} dtypes={dict(obj.dtypes.value_counts())}"
    if isinstance(obj, pd.Series):
        return f"Series(len={len(obj)}, dtype={obj.dtype})"
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape} dtype={obj.dtype}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)}) first={obj[0] if obj else None}"
    if isinstance(obj, dict):
        return f"dict(len={len(obj)}) keys={list(obj.keys())[:12]}"
    if isinstance(obj, (int, float, str, bool)):
        return f"{type(obj).__name__}={obj}"
    return f"{type(obj).__name__}={str(obj)[:80]}"


def main() -> None:
    xgb_bundle = joblib.load(ROOT / "processed" / "xgboost" / "preprocessed_xgboost.joblib")
    print("=" * 80)
    print("XGBoost bundle keys:")
    for k in sorted(xgb_bundle.keys()):
        print(f"  {k:40s} -> {describe(k, xgb_bundle[k])}")

    common_bundle = joblib.load(ROOT / "processed" / "common" / "preprocessed_common.joblib")
    print("=" * 80)
    print("Common bundle keys:")
    for k in sorted(common_bundle.keys()):
        print(f"  {k:40s} -> {describe(k, common_bundle[k])}")

    print("=" * 80)
    if "X_train" in xgb_bundle:
        X = xgb_bundle["X_train"]
        print(f"X_train type: {type(X)}")
        if isinstance(X, pd.DataFrame):
            print(f"  columns[:20] = {list(X.columns[:20])}")
            print(f"  columns[-20:] = {list(X.columns[-20:])}")
            print(f"  any NaN: {X.isna().any().any()}")
        elif isinstance(X, np.ndarray):
            print(f"  shape = {X.shape}, dtype = {X.dtype}, any NaN = {np.isnan(X).any()}")

    if "feature_names" in xgb_bundle:
        fn = xgb_bundle["feature_names"]
        print(f"feature_names (n={len(fn)}): first 10 = {list(fn)[:10]}, last 10 = {list(fn)[-10:]}")

    print("=" * 80)
    if "common_train" in common_bundle:
        ct = common_bundle["common_train"]
        print(f"common_train type: {type(ct)}, shape: {ct.shape if hasattr(ct, 'shape') else '?'}")
        if isinstance(ct, pd.DataFrame):
            print(f"  columns = {list(ct.columns)}")
            print(f"  dtypes summary: {ct.dtypes.value_counts().to_dict()}")

    print("=" * 80)
    if "y_train" in xgb_bundle:
        y = xgb_bundle["y_train"]
        print(f"y_train type: {type(y)}")
        if hasattr(y, "value_counts"):
            print(f"  dist: {y.value_counts().to_dict()}")
        elif isinstance(y, np.ndarray):
            uniq, cnts = np.unique(y, return_counts=True)
            print(f"  dist: {dict(zip(uniq.tolist(), cnts.tolist()))}")

    if "train_ids" in xgb_bundle:
        ti = xgb_bundle["train_ids"]
        print(f"train_ids sample: {list(ti[:5]) if hasattr(ti, '__iter__') else ti}")


if __name__ == "__main__":
    main()
