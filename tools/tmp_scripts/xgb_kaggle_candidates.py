"""Build Kaggle submission candidates by blending current and legacy XGB assets.

This script intentionally mixes two generations of *XGBoost-only* work:

1. The older public-LB-friendly snapshot under
   ``/Users/shenyijie/Desktop/20260319_xgboost 2``.
2. The newer fold-aware pipeline in the current MLWP workspace.

The goal is pragmatic: recover the old public strength (0.80804) and then make
small, evidence-backed edits instead of replacing the whole submission with a
clean-but-weaker model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CURRENT_ROOT = Path("/Users/shenyijie/Desktop/MLWP project")
LEGACY_ROOT = Path("/Users/shenyijie/Desktop/20260319_xgboost 2")
OUT_DIR = CURRENT_ROOT / "reports" / "xgboost" / "submission_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Candidate:
    name: str
    passenger_ids: pd.Series
    proba: np.ndarray
    threshold: float
    notes: dict

    def submission(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "PassengerId": self.passenger_ids.astype("string"),
                "Transported": (self.proba >= self.threshold).astype(bool),
            }
        )


def load_current() -> dict[str, object]:
    common = joblib.load(CURRENT_ROOT / "processed" / "common" / "preprocessed_common.joblib")
    a7 = pd.read_csv(CURRENT_ROOT / "reports" / "xgboost" / "logs" / "A7_test_proba.csv")
    v2 = pd.read_csv(CURRENT_ROOT / "reports" / "xgboost" / "logs" / "v2_opt_test_proba.csv")
    return {
        "common_test": common["common_test"].reset_index(drop=True),
        "passenger_ids": common["common_test"]["PassengerId"].reset_index(drop=True).astype("string"),
        "a7blend_test": a7["y_proba_blend"].to_numpy(dtype="float64"),
        "v2opt_test": v2["y_proba"].to_numpy(dtype="float64"),
    }


def load_legacy() -> dict[str, object]:
    npz = np.load(LEGACY_ROOT / "xgb_stepwise_lab" / "artifacts" / "reports" / "candidate_probs.npz")
    final_sel = json.loads(
        (LEGACY_ROOT / "xgb_stepwise_lab" / "artifacts" / "reports" / "final_selection.json").read_text(
            encoding="utf-8"
        )
    )
    anchor = pd.read_csv(LEGACY_ROOT / "submission_xgb_conservative.csv")
    return {
        "candidate_probs": npz,
        "final_selection": final_sel,
        "anchor_submission": anchor.reset_index(drop=True),
    }


def apply_group_mean_replace(
    proba: np.ndarray,
    groups: pd.Series,
    *,
    size_min: int,
    std_max: float,
) -> tuple[np.ndarray, int]:
    out = np.asarray(proba, dtype="float64").copy()
    changed_rows = 0
    for _, idx in groups.groupby(groups).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        if len(idx_arr) < size_min:
            continue
        vals = out[idx_arr]
        if float(vals.std()) <= std_max:
            mean_val = float(vals.mean())
            if not np.allclose(vals, mean_val):
                changed_rows += int(len(idx_arr))
            out[idx_arr] = mean_val
    return out, changed_rows


def build_candidates() -> list[Candidate]:
    cur = load_current()
    leg = load_legacy()
    pids = cur["passenger_ids"]
    common_test = cur["common_test"]
    candidate_probs = leg["candidate_probs"]
    anchor_df = leg["anchor_submission"]

    legacy_public = candidate_probs["legacy_public_style__test"].astype("float64")
    stepwise_balanced = candidate_probs["stepwise_balanced_te__test"].astype("float64")
    a7blend = cur["a7blend_test"]

    pair_w_step, pair_w_legacy = 0.35, 0.65
    pair_proba = pair_w_step * stepwise_balanced + pair_w_legacy * legacy_public

    pair = Candidate(
        name="submission_legacy_pair_revival.csv",
        passenger_ids=pids,
        proba=pair_proba,
        threshold=0.485,
        notes={
            "source": "legacy final_selection revival",
            "weights": {"stepwise_balanced_te": pair_w_step, "legacy_public_style": pair_w_legacy},
            "threshold": 0.485,
            "anchor_submission": str(LEGACY_ROOT / "submission_xgb_conservative.csv"),
            "changed_vs_anchor": int(
                (
                    ((pair_proba >= 0.485).astype(int))
                    != anchor_df["Transported"].astype(bool).astype(int).to_numpy()
                ).sum()
            ),
        },
    )

    pair_groupfix_proba, changed_rows = apply_group_mean_replace(
        pair_proba,
        common_test["GroupID"].astype("string"),
        size_min=4,
        std_max=0.12,
    )
    pair_groupfix = Candidate(
        name="submission_legacy_pair_groupfix.csv",
        passenger_ids=pids,
        proba=pair_groupfix_proba,
        threshold=0.485,
        notes={
            "source": "legacy pair + group mean replace",
            "base_candidate": pair.name,
            "rule": {"group_key": "GroupID", "size_min": 4, "std_max": 0.12},
            "changed_rows_by_rule": changed_rows,
            "changed_vs_anchor": int(
                (
                    ((pair_groupfix_proba >= 0.485).astype(int))
                    != anchor_df["Transported"].astype(bool).astype(int).to_numpy()
                ).sum()
            ),
        },
    )

    anchor_pred = anchor_df["Transported"].astype(bool).astype(int).to_numpy().copy()
    to_one = (anchor_pred == 0) & (a7blend >= 0.55)
    to_zero = (anchor_pred == 1) & (a7blend <= 0.45)
    anchor_override_pred = anchor_pred.copy()
    anchor_override_pred[to_one] = 1
    anchor_override_pred[to_zero] = 0
    anchor_override = Candidate(
        name="submission_anchor_a7_override.csv",
        passenger_ids=pids,
        proba=anchor_override_pred.astype("float64"),
        threshold=0.5,
        notes={
            "source": "anchor conservative + A7 high-confidence override",
            "rule": {"flip_to_one_if_a7_ge": 0.55, "flip_to_zero_if_a7_le": 0.45},
            "changed_vs_anchor": int((anchor_override_pred != anchor_pred).sum()),
            "n_to_one": int(to_one.sum()),
            "n_to_zero": int(to_zero.sum()),
        },
    )

    asym_candidates: list[Candidate] = []
    for hi, lo in [(0.52, 0.46), (0.52, 0.45), (0.52, 0.44)]:
        asym_pred = anchor_pred.copy()
        to_one = (asym_pred == 0) & (a7blend >= hi)
        to_zero = (asym_pred == 1) & (a7blend <= lo)
        asym_pred[to_one] = 1
        asym_pred[to_zero] = 0
        asym_candidates.append(
            Candidate(
                name=f"submission_anchor_a7_hi{int(hi * 100)}_lo{int(lo * 100)}.csv",
                passenger_ids=pids,
                proba=asym_pred.astype("float64"),
                threshold=0.5,
                notes={
                    "source": "anchor conservative + A7 asymmetric override",
                    "rule": {"flip_to_one_if_a7_ge": hi, "flip_to_zero_if_a7_le": lo},
                    "changed_vs_anchor": int((asym_pred != anchor_pred).sum()),
                    "n_to_one": int(to_one.sum()),
                    "n_to_zero": int(to_zero.sum()),
                },
            )
        )

    return [pair, pair_groupfix, anchor_override, *asym_candidates]


def main() -> None:
    manifest: list[dict] = []
    for candidate in build_candidates():
        sub = candidate.submission()
        out_path = OUT_DIR / candidate.name
        sub.to_csv(out_path, index=False)
        meta_path = out_path.with_suffix(".json")
        meta = {
            "file": str(out_path),
            "threshold": candidate.threshold,
            "positive_rate": float(sub["Transported"].astype(int).mean()),
            **candidate.notes,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append(meta)
        print(f"[ok] {out_path}")
        print(json.dumps(meta, indent=2))

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
