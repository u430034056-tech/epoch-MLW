"""Fold-aware feature engineering utilities.

These helpers are invoked *inside* each CV fold so that statistics estimated
from the fold's train partition never leak into its validation partition.

The shared ``common`` preprocessing already produced a clean 51-column table,
but some of its statistics (``SurnameFreq``, ``CabinNumBin`` edges, and the
hierarchical spend medians) were fit on the union of train+test.  That is fine
for generating a submission-time feature table, but for honest CV we need to
recompute them per fold.  This module implements that fold-local recomputation
plus a modern fold-safe target encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fold-local SurnameFreq
# ---------------------------------------------------------------------------


def refit_surname_freq(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame | None,
    test_raw: pd.DataFrame | None = None,
    surname_col: str = "Surname",
) -> tuple[pd.Series, pd.Series | None, pd.Series | None]:
    """Recompute ``SurnameFreq`` inside a fold.

    Parameters
    ----------
    train_raw, valid_raw, test_raw
        DataFrames that *must* still contain the raw ``Surname`` column.  Pass
        ``None`` for valid/test if those slices are not available.

    Returns
    -------
    freq_train, freq_valid, freq_test
        Integer series aligned to each input.  Surnames unseen in train map to
        0 in valid/test – mirroring the original train-fit behaviour.
    """
    if surname_col not in train_raw.columns:
        raise KeyError(f"`{surname_col}` missing from train frame")

    surnames = train_raw[surname_col].astype("string")
    vc = surnames.value_counts(dropna=True)
    freq_map = vc.to_dict()

    def _map(s: pd.Series) -> pd.Series:
        return s.astype("string").map(freq_map).fillna(0).astype("int32")

    freq_train = _map(surnames)
    freq_valid = _map(valid_raw[surname_col]) if valid_raw is not None and surname_col in valid_raw.columns else None
    freq_test = _map(test_raw[surname_col]) if test_raw is not None and surname_col in test_raw.columns else None
    return freq_train, freq_valid, freq_test


# ---------------------------------------------------------------------------
# Fold-local Surname Transported rate (LOO for train, plain for valid/test)
# ---------------------------------------------------------------------------


def refit_surname_loo_rate(
    train_raw: pd.DataFrame,
    y_train: pd.Series,
    valid_raw: pd.DataFrame | None,
    test_raw: pd.DataFrame | None = None,
    surname_col: str = "Surname",
    smoothing: float = 10.0,
) -> tuple[pd.Series, pd.Series | None, pd.Series | None]:
    """Compute a fold-safe Surname Transported rate.

    * Train rows receive a *leave-one-out* Bayesian-smoothed aggregate so a row
      never sees its own y in the computation.
    * Valid/test rows receive a plain Bayesian-smoothed mapping computed from
      the whole fold-train surname aggregate.  They are never in the fit set so
      there is no self-leak.

    Spaceship Titanic's train/test share ~1500 surnames, covering 90% of test
    rows.  Exploiting that overlap via transported-rate is a well-known strong
    signal, but it is only fold-safe when the row-level aggregate excludes the
    row itself.
    """
    if surname_col not in train_raw.columns:
        empty = pd.Series(np.nan, index=train_raw.index, dtype="float64")
        return empty, (None if valid_raw is None else pd.Series(np.nan, index=valid_raw.index, dtype="float64")), (None if test_raw is None else pd.Series(np.nan, index=test_raw.index, dtype="float64"))

    global_mean = float(pd.Series(y_train).mean())
    names = train_raw[surname_col].astype("string").fillna("__MISSING__")
    y_arr = np.asarray(y_train, dtype="float64")

    df = pd.DataFrame({"sn": names.values, "y": y_arr})
    grp = df.groupby("sn")["y"]
    sum_y = grp.transform("sum").values
    count = grp.transform("count").values
    loo_num = sum_y - y_arr + global_mean * smoothing
    loo_den = (count - 1) + smoothing
    train_rate = np.where(loo_den > 0, loo_num / loo_den, global_mean).astype("float64")

    stats = df.groupby("sn")["y"].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    m = smoothed.to_dict()

    def _map_plain(s: pd.Series) -> pd.Series:
        mapped = s.astype("string").fillna("__MISSING__").map(m).astype("float64")
        return mapped.fillna(global_mean)

    r_train = pd.Series(train_rate, index=train_raw.index, name="SurnameRate")
    r_valid = _map_plain(valid_raw[surname_col]) if valid_raw is not None and surname_col in valid_raw.columns else None
    r_test = _map_plain(test_raw[surname_col]) if test_raw is not None and surname_col in test_raw.columns else None
    return r_train, r_valid, r_test


# ---------------------------------------------------------------------------
# Fold-local CabinNumBin edges
# ---------------------------------------------------------------------------


def refit_cabin_num_bin(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame | None,
    test_raw: pd.DataFrame | None,
    cabin_num_col: str = "CabinNum",
    n_bins: int = 5,
) -> tuple[pd.Series, pd.Series | None, pd.Series | None, list[float]]:
    """Quantile-bin ``CabinNum`` using only the fold-train non-missing values.

    Missing cabin numbers get their own level (``CabinBin_Missing``).
    """
    if cabin_num_col not in train_raw.columns:
        raise KeyError(f"`{cabin_num_col}` missing from train frame")

    train_nums = pd.to_numeric(train_raw[cabin_num_col], errors="coerce")
    observed = train_nums.dropna()
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    inner_edges = np.quantile(observed.values, qs) if len(observed) else np.array([])
    edges = [-np.inf] + sorted(set(float(e) for e in inner_edges)) + [np.inf]

    def _bin(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        labels = [f"CabinBin_{i+1}" for i in range(len(edges) - 1)]
        out = pd.cut(s, bins=edges, labels=labels, include_lowest=True)
        out = out.astype("string").fillna("CabinBin_Missing")
        return out

    return (
        _bin(train_raw[cabin_num_col]),
        _bin(valid_raw[cabin_num_col]) if valid_raw is not None else None,
        _bin(test_raw[cabin_num_col]) if test_raw is not None else None,
        edges,
    )


# ---------------------------------------------------------------------------
# Fold-safe target encoder with smoothing & optional out-of-fold variant
# ---------------------------------------------------------------------------


@dataclass
class TargetEncoder:
    """Target encoder with Bayesian smoothing.

    * ``fit`` computes per-level Transported means from the fold-train partition.
    * ``transform`` maps each level to that smoothed mean; unseen levels fall
      back to the global mean.

    When the training partition needs its own encoded values (for model fitting)
    we use :func:`oof_encode` instead, which runs an inner KFold on the train
    partition to avoid a row seeing its own target.  The plain ``transform`` is
    reserved for valid / test sets, which are never seen during fitting and
    therefore do not suffer from target leakage.
    """

    cols: Sequence[str]
    smoothing: float = 20.0
    _maps: dict[str, dict[str, float]] = None
    _global_mean: float = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self._maps = {}
        self._global_mean = float(y.mean())
        for col in self.cols:
            if col not in X.columns:
                continue
            s = X[col].astype("string").fillna("__MISSING__")
            df = pd.DataFrame({"k": s.values, "y": np.asarray(y)})
            stats = df.groupby("k")["y"].agg(["mean", "count"])
            smoothed = (
                stats["mean"] * stats["count"] + self._global_mean * self.smoothing
            ) / (stats["count"] + self.smoothing)
            self._maps[col] = smoothed.to_dict()
        return self

    def loo_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Leave-one-out target encoding for the fit rows.

        Removes the row's own y from its group aggregate, so a train row never
        sees its own target (the critical difference from plain ``transform``).
        Only use this for the *same* rows that were fit.  Valid/test rows should
        use :meth:`transform`, which does not suffer from self-leak because they
        were never in the fit set.
        """
        if self._maps is None:
            raise RuntimeError("TargetEncoder has not been fit yet")
        d = X.copy()
        y_arr = np.asarray(y, dtype="float64")
        for col in self.cols:
            if col not in d.columns:
                continue
            s = d[col].astype("string").fillna("__MISSING__")
            df = pd.DataFrame({"k": s.values, "y": y_arr})
            grp = df.groupby("k")["y"]
            sum_y = grp.transform("sum").values
            count = grp.transform("count").values
            loo_num = sum_y - y_arr + self._global_mean * self.smoothing
            loo_den = (count - 1) + self.smoothing
            loo = np.where(loo_den > 0, loo_num / loo_den, self._global_mean)
            d[f"te_{col}"] = loo.astype("float64")
        return d

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._maps is None:
            raise RuntimeError("TargetEncoder has not been fit yet")
        d = X.copy()
        for col in self.cols:
            if col not in d.columns:
                continue
            s = d[col].astype("string").fillna("__MISSING__")
            mapped = s.map(self._maps[col]).astype("float64").fillna(self._global_mean)
            d[f"te_{col}"] = mapped
        return d

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def oof_target_encode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cols: Sequence[str],
    smoothing: float = 20.0,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold target encoding for the *training* partition only.

    Each row is assigned the encoded value computed on the other ``n_splits-1``
    partitions, so no row ever sees its own target.  This is the canonical
    technique from Kaggle's old tutorials (Owen Zhang's playbook) and is the
    right companion to the fold-aware CV we already do at the outer level.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = pd.DataFrame(index=X_train.index)
    for col in cols:
        out[f"te_{col}"] = np.nan
    if X_train.empty:
        return out

    global_mean = float(y_train.mean())
    for tr_idx, va_idx in kf.split(X_train):
        enc = TargetEncoder(cols=list(cols), smoothing=smoothing)
        enc.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        for col in cols:
            if col not in X_train.columns:
                continue
            s = X_train.iloc[va_idx][col].astype("string").fillna("__MISSING__")
            mapped = s.map(enc._maps[col]).astype("float64").fillna(global_mean)
            out.iloc[va_idx, out.columns.get_loc(f"te_{col}")] = mapped.values
    return out


# ---------------------------------------------------------------------------
# Group-aware aggregations (fold-safe because GroupID is never split)
# ---------------------------------------------------------------------------


def add_group_aggregates(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame | None,
    test_raw: pd.DataFrame | None,
    group_col: str = "GroupID",
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Build per-group aggregates using *only* the supplied frame.

    Since our outer CV splits by ``GroupID``, a given group appears entirely in
    one side of the split.  That lets us safely compute group-level stats on
    the union of (train, valid, test): no y is involved, so no leakage.

    Returned frames share the same index as the inputs and add columns:
        group_mean_age, group_max_spend, group_has_cryosleep, group_size_bucket
    """
    frames = [("train", train_raw)]
    if valid_raw is not None:
        frames.append(("valid", valid_raw))
    if test_raw is not None:
        frames.append(("test", test_raw))
    combined = pd.concat([f.assign(_origin=name) for name, f in frames], axis=0, ignore_index=False)

    if group_col not in combined.columns:
        return train_raw, valid_raw, test_raw

    g = combined.groupby(group_col, dropna=False)
    agg_age = g["Age"].transform("mean") if "Age" in combined.columns else np.nan
    agg_spend_max = (
        g["TotalSpend"].transform("max") if "TotalSpend" in combined.columns else np.nan
    )
    agg_spend_mean = (
        g["TotalSpend"].transform("mean") if "TotalSpend" in combined.columns else np.nan
    )
    cryo_true = combined["CryoSleep"].astype("string").eq("True").astype("int8")
    agg_has_cryo = cryo_true.groupby(combined[group_col]).transform("max")

    combined["group_mean_age"] = agg_age
    combined["group_max_spend"] = agg_spend_max
    combined["group_mean_spend"] = agg_spend_mean
    combined["group_has_cryosleep"] = agg_has_cryo.astype("int8")
    combined["group_size_from_ids"] = g.size().reindex(combined[group_col]).values

    out_train = combined.loc[combined["_origin"] == "train"].drop(columns=["_origin"])
    out_valid = (
        combined.loc[combined["_origin"] == "valid"].drop(columns=["_origin"])
        if valid_raw is not None else None
    )
    out_test = (
        combined.loc[combined["_origin"] == "test"].drop(columns=["_origin"])
        if test_raw is not None else None
    )
    return out_train, out_valid, out_test


def apply_fold_features(
    X_train_fold: pd.DataFrame,
    X_valid_fold: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train_fold: pd.Series,
    raw_train_fold: pd.DataFrame,
    raw_valid_fold: pd.DataFrame,
    raw_test_full: pd.DataFrame,
    target_encode_cols: Sequence[str] | None = None,
    target_encode_mode: str = "plain",  # one of: plain / loo / oof / none
    te_smoothing: float = 20.0,
    use_surname_refit: bool = True,
    use_surname_rate: bool = False,
    surname_rate_smoothing: float = 10.0,
    use_cabin_bin_refit: bool = True,
    use_oof_target_encoding: bool = False,  # legacy switch, kept for backward compat
    use_group_aggregates: bool = False,
    oof_te_splits: int = 5,
    oof_te_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply fold-aware feature engineering and return three updated frames.

    ``target_encode_mode`` controls how :func:`TargetEncoder` attaches encoded
    columns to the training partition:

    * ``"plain"`` – default, mean + smoothing, **train rows see their own y**
      (has subtle self-leak; kept for backward compat / ablation A3).
    * ``"loo"`` – leave-one-out mean + smoothing for train; plain mapping for
      valid/test.  Safer.
    * ``"oof"`` – inner K-fold OOF encoding for train; plain mapping for
      valid/test.  Strongest leak protection.
    * ``"none"`` – do not attach ``te_*`` columns at all.

    The legacy ``use_oof_target_encoding=True`` maps to ``target_encode_mode``
    ``"oof"`` to preserve earlier ablation semantics.
    """
    Xt, Xv, Xte = X_train_fold.copy(), X_valid_fold.copy(), X_test_full.copy()

    if use_surname_refit and "Surname" in raw_train_fold.columns:
        ft, fv, ffte = refit_surname_freq(raw_train_fold, raw_valid_fold, raw_test_full)
        Xt["SurnameFreq"] = ft.values
        Xv["SurnameFreq"] = fv.values
        Xte["SurnameFreq"] = ffte.values

    if use_surname_rate and "Surname" in raw_train_fold.columns:
        rt, rv, rte = refit_surname_loo_rate(
            raw_train_fold.reset_index(drop=True),
            y_train_fold.reset_index(drop=True),
            raw_valid_fold.reset_index(drop=True) if raw_valid_fold is not None else None,
            raw_test_full.reset_index(drop=True),
            smoothing=surname_rate_smoothing,
        )
        Xt["SurnameRate"] = rt.values
        if rv is not None:
            Xv["SurnameRate"] = rv.values
        if rte is not None:
            Xte["SurnameRate"] = rte.values

    if use_cabin_bin_refit and "CabinNum" in raw_train_fold.columns:
        bt, bv, bte, _edges = refit_cabin_num_bin(raw_train_fold, raw_valid_fold, raw_test_full)
        levels = sorted(set(bt.unique()) | set(bv.unique()) | set(bte.unique()))
        Xt["CabinNumBin"] = pd.Categorical(bt.values, categories=levels)
        Xv["CabinNumBin"] = pd.Categorical(bv.values, categories=levels)
        Xte["CabinNumBin"] = pd.Categorical(bte.values, categories=levels)

    # Resolve the target-encoding mode (legacy switch has priority if True).
    te_mode = target_encode_mode
    if use_oof_target_encoding:
        te_mode = "oof"

    if target_encode_cols and te_mode != "none":
        enc = TargetEncoder(cols=list(target_encode_cols), smoothing=float(te_smoothing))
        raw_tr_fresh = raw_train_fold.reset_index(drop=True)
        raw_va_fresh = raw_valid_fold.reset_index(drop=True)
        raw_te_fresh = raw_test_full.reset_index(drop=True)
        y_tr_fresh = y_train_fold.reset_index(drop=True)
        enc.fit(raw_tr_fresh, y_tr_fresh)
        te_valid = enc.transform(raw_va_fresh).filter(like="te_")
        te_test = enc.transform(raw_te_fresh).filter(like="te_")

        if te_mode == "oof":
            te_train = oof_target_encode(
                raw_tr_fresh,
                y_tr_fresh,
                cols=list(target_encode_cols),
                smoothing=float(te_smoothing),
                n_splits=oof_te_splits,
                seed=oof_te_seed,
            )
        elif te_mode == "loo":
            te_train = enc.loo_transform(raw_tr_fresh, y_tr_fresh).filter(like="te_")
        else:  # "plain"
            te_train = enc.transform(raw_tr_fresh).filter(like="te_")

        Xt = pd.concat([te_train.reset_index(drop=True), Xt.reset_index(drop=True)], axis=1)
        Xv = pd.concat([te_valid.reset_index(drop=True), Xv.reset_index(drop=True)], axis=1)
        Xte = pd.concat([te_test.reset_index(drop=True), Xte.reset_index(drop=True)], axis=1)

    if use_group_aggregates:
        raw_tr2, raw_va2, raw_te2 = add_group_aggregates(
            raw_train_fold.reset_index(drop=True),
            raw_valid_fold.reset_index(drop=True),
            raw_test_full.reset_index(drop=True),
        )
        agg_cols = [
            "group_mean_age",
            "group_max_spend",
            "group_mean_spend",
            "group_has_cryosleep",
            "group_size_from_ids",
        ]
        for col in agg_cols:
            if col in raw_tr2.columns:
                Xt[col] = pd.to_numeric(raw_tr2[col].values, errors="coerce").astype("float64")
                Xv[col] = pd.to_numeric(raw_va2[col].values, errors="coerce").astype("float64")
                Xte[col] = pd.to_numeric(raw_te2[col].values, errors="coerce").astype("float64")

    return Xt, Xv, Xte
