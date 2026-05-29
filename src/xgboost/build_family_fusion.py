"""Build XGBoost-only fusion candidates from independent family probabilities.

This script intentionally stays within the XGBoost lane: every input
probability file comes from an XGB-based branch.  The point is to exploit
complementarity between:

- the clean high-OOF ``v2_opt`` family
- the public-notebook-style raw-CSV family
- the group-fill raw-CSV family

It writes ready-to-submit CSV files plus compact JSON metadata under
``reports/xgboost/submission_candidates``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from . import config


RAW_TRAIN_PATH = config.PROJECT_ROOT / "data" / "raw" / "train.csv"
STEPWISE_NPZ_PATH = config.LOGS_DIR / "candidate_probs.npz"
SUBMISSION_CANDIDATES_DIR = config.REPORTS_DIR / "submission_candidates"


@dataclass
class FusionCandidate:
    file: str
    metadata_path: str
    source: str
    honest_oof_acc: float
    threshold: float
    positive_rate: float
    changed_vs_v2_best: int
    changed_vs_public_style: int
    changed_vs_groupfill: int
    details: dict


def _load_oof_frames() -> pd.DataFrame:
    y = pd.read_csv(RAW_TRAIN_PATH)[["PassengerId", "Transported"]]
    v2 = pd.read_csv(config.LOGS_DIR / "v2_opt_oof_proba.csv").rename(columns={"y_proba": "v2"})
    public = pd.read_csv(config.LOGS_DIR / "public81599_style_oof_proba.csv").rename(
        columns={"y_proba": "public"}
    )
    groupfill = pd.read_csv(config.LOGS_DIR / "public_groupfill_t052_oof_proba.csv").rename(
        columns={"y_proba": "groupfill"}
    )
    a6 = pd.read_csv(config.LOGS_DIR / "A6_oof.csv").rename(columns={"y_proba": "a6"})
    a7 = pd.read_csv(config.LOGS_DIR / "A7_oof.csv")[["y_proba_blend"]].rename(
        columns={"y_proba_blend": "a7_blend"}
    )
    return (
        y.merge(v2, on="PassengerId")
        .merge(public, on="PassengerId")
        .merge(groupfill, on="PassengerId")
        .join(a6[["a6"]])
        .join(a7[["a7_blend"]])
    )


def _load_test_frames() -> pd.DataFrame:
    v2 = pd.read_csv(config.LOGS_DIR / "v2_opt_test_proba.csv").rename(columns={"y_proba": "v2"})
    public = pd.read_csv(config.LOGS_DIR / "public81599_style_test_proba.csv").rename(
        columns={"y_proba": "public"}
    )
    groupfill = pd.read_csv(config.LOGS_DIR / "public_groupfill_t052_test_proba.csv").rename(
        columns={"y_proba": "groupfill"}
    )
    a6 = pd.read_csv(config.LOGS_DIR / "A6_test_proba.csv").rename(columns={"y_proba": "a6"})
    a7 = pd.read_csv(config.LOGS_DIR / "A7_test_proba.csv")[["PassengerId", "y_proba_blend"]].rename(
        columns={"y_proba_blend": "a7_blend"}
    )
    return v2.merge(public, on="PassengerId").merge(groupfill, on="PassengerId").merge(a6, on="PassengerId").merge(a7, on="PassengerId")


def _load_stepwise_probs() -> dict[str, np.ndarray]:
    npz = np.load(STEPWISE_NPZ_PATH)
    return {
        "stepwise_safe_te_oof": npz["stepwise_safe_te__oof"],
        "stepwise_safe_te_test": npz["stepwise_safe_te__test"],
        "stepwise_balanced_te_oof": npz["stepwise_balanced_te__oof"],
        "stepwise_balanced_te_test": npz["stepwise_balanced_te__test"],
        "legacy_public_style_oof": npz["legacy_public_style__oof"],
        "legacy_public_style_test": npz["legacy_public_style__test"],
    }


def _load_reference_labels() -> dict[str, np.ndarray]:
    return {
        "v2_best": pd.read_csv(config.SUBMISSIONS_DIR / "submission_v2_best.csv")["Transported"]
        .astype(bool)
        .to_numpy(),
        "public_style": pd.read_csv(
            SUBMISSION_CANDIDATES_DIR / "submission_public81599_style.csv"
        )["Transported"]
        .astype(bool)
        .to_numpy(),
        "groupfill": pd.read_csv(
            SUBMISSION_CANDIDATES_DIR / "submission_public_groupfill_t052.csv"
        )["Transported"]
        .astype(bool)
        .to_numpy(),
    }


def _write_candidate(
    name: str,
    source: str,
    passenger_ids: pd.Series,
    pred: np.ndarray,
    honest_oof_acc: float,
    threshold: float,
    refs: dict[str, np.ndarray],
    details: dict,
) -> FusionCandidate:
    SUBMISSION_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SUBMISSION_CANDIDATES_DIR / f"{name}.csv"
    metadata_path = SUBMISSION_CANDIDATES_DIR / f"{name}.json"
    pred_bool = np.asarray(pred).astype(bool)
    pd.DataFrame({"PassengerId": passenger_ids.astype(str), "Transported": pred_bool}).to_csv(
        file_path, index=False
    )
    metadata = FusionCandidate(
        file=str(file_path),
        metadata_path=str(metadata_path),
        source=source,
        honest_oof_acc=float(honest_oof_acc),
        threshold=float(threshold),
        positive_rate=float(pred_bool.mean()),
        changed_vs_v2_best=int(np.sum(pred_bool != refs["v2_best"])),
        changed_vs_public_style=int(np.sum(pred_bool != refs["public_style"])),
        changed_vs_groupfill=int(np.sum(pred_bool != refs["groupfill"])),
        details=details,
    )
    metadata_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n")
    return metadata


def build_candidates() -> list[FusionCandidate]:
    oof = _load_oof_frames()
    test = _load_test_frames()
    refs = _load_reference_labels()
    stepwise = _load_stepwise_probs()
    y = oof["Transported"].astype(int).to_numpy()

    oof_v2 = oof["v2"].to_numpy()
    oof_public = oof["public"].to_numpy()
    oof_groupfill = oof["groupfill"].to_numpy()
    test_v2 = test["v2"].to_numpy()
    test_public = test["public"].to_numpy()
    test_groupfill = test["groupfill"].to_numpy()
    test_a6 = test["a6"].to_numpy()
    test_a7_blend = test["a7_blend"].to_numpy()
    stepwise_safe_te_oof = stepwise["stepwise_safe_te_oof"]
    stepwise_safe_te_test = stepwise["stepwise_safe_te_test"]
    stepwise_balanced_oof = stepwise["stepwise_balanced_te_oof"]
    stepwise_balanced_test = stepwise["stepwise_balanced_te_test"]
    legacy_public_oof = stepwise["legacy_public_style_oof"]
    legacy_public_test = stepwise["legacy_public_style_test"]
    oof_a6 = oof["a6"].to_numpy()
    oof_a7_blend = oof["a7_blend"].to_numpy()

    outputs: list[FusionCandidate] = []

    # Candidate 1: honest-best linear blend found so far.
    blend_oof = 0.95 * oof_v2 + 0.05 * oof_public
    blend_test = 0.95 * test_v2 + 0.05 * test_public
    outputs.append(
        _write_candidate(
            name="submission_xgb_family_blend_v2p95_publicp05_t0505",
            source="v2_opt 95% + public_style 5% linear blend",
            passenger_ids=test["PassengerId"],
            pred=blend_test >= 0.505,
            honest_oof_acc=accuracy_score(y, blend_oof >= 0.505),
            threshold=0.505,
            refs=refs,
            details={
                "weights": {"v2_opt": 0.95, "public_style": 0.05, "groupfill": 0.0},
            },
        )
    )

    # Candidate 1b: same blend, but lower threshold to match the historical
    # public-LB-friendly positive rate (~0.533).
    outputs.append(
        _write_candidate(
            name="submission_xgb_family_blend_v2p95_publicp05_t0475",
            source="v2_opt 95% + public_style 5% linear blend, lowered threshold for public-LB chase",
            passenger_ids=test["PassengerId"],
            pred=blend_test >= 0.475,
            honest_oof_acc=accuracy_score(y, blend_oof >= 0.475),
            threshold=0.475,
            refs=refs,
            details={
                "weights": {"v2_opt": 0.95, "public_style": 0.05, "groupfill": 0.0},
                "rationale": "match historical strong-XGB public positive rate near 0.533",
            },
        )
    )

    # Candidate 2: honest-best nonlinear override found so far.
    uncertain_oof = (oof_v2 >= 0.45) & (oof_v2 <= 0.56)
    uncertain_test = (test_v2 >= 0.45) & (test_v2 <= 0.56)
    force1_oof = uncertain_oof & (oof_public >= 0.65) & (oof_groupfill >= 0.65)
    force0_oof = uncertain_oof & (oof_public <= 0.42) & (oof_groupfill <= 0.42)
    force1_test = uncertain_test & (test_public >= 0.65) & (test_groupfill >= 0.65)
    force0_test = uncertain_test & (test_public <= 0.42) & (test_groupfill <= 0.42)
    override_oof = (oof_v2 >= 0.5).astype(int)
    override_test = (test_v2 >= 0.5).astype(int)
    override_oof[force1_oof] = 1
    override_oof[force0_oof] = 0
    override_test[force1_test] = 1
    override_test[force0_test] = 0
    outputs.append(
        _write_candidate(
            name="submission_xgb_family_override_v2_public_groupfill",
            source="v2_opt base with public/groupfill consensus override",
            passenger_ids=test["PassengerId"],
            pred=override_test.astype(bool),
            honest_oof_acc=accuracy_score(y, override_oof),
            threshold=0.5,
            refs=refs,
            details={
                "base": "v2_opt >= 0.5",
                "uncertain_band": [0.45, 0.56],
                "force_one_if": {"public_style_min": 0.65, "groupfill_min": 0.65},
                "force_zero_if": {"public_style_max": 0.42, "groupfill_max": 0.42},
                "changed_rows_oof": int(force1_oof.sum() + force0_oof.sum()),
                "changed_rows_test": int(force1_test.sum() + force0_test.sum()),
            },
        )
    )

    # Candidate 3: public-family blend, useful as a more radical Kaggle-facing option.
    public_pair_oof = 0.5 * oof_public + 0.5 * oof_groupfill
    public_pair_test = 0.5 * test_public + 0.5 * test_groupfill
    outputs.append(
        _write_candidate(
            name="submission_xgb_public_family_pair_t051",
            source="public_style 50% + groupfill 50% linear blend",
            passenger_ids=test["PassengerId"],
            pred=public_pair_test >= 0.51,
            honest_oof_acc=accuracy_score(y, public_pair_oof >= 0.51),
            threshold=0.51,
            refs=refs,
            details={
                "weights": {"v2_opt": 0.0, "public_style": 0.5, "groupfill": 0.5},
            },
        )
    )

    # Candidate 4: bridge the current clean V2 family with the older but
    # explicitly fold-safe stepwise balanced-TE family.
    bridge_oof = 0.55 * oof_v2 + 0.45 * stepwise_balanced_oof
    bridge_test = 0.55 * test_v2 + 0.45 * stepwise_balanced_test
    outputs.append(
        _write_candidate(
            name="submission_xgb_bridge_v2p55_stepwiseBalanced45_t0485",
            source="v2_opt 55% + fold-safe stepwise_balanced_te 45%",
            passenger_ids=test["PassengerId"],
            pred=bridge_test >= 0.485,
            honest_oof_acc=accuracy_score(y, bridge_oof >= 0.485),
            threshold=0.485,
            refs=refs,
            details={
                "weights": {"v2_opt": 0.55, "stepwise_balanced_te": 0.45},
            },
        )
    )

    # Candidate 5: same bridge with a lower threshold to recover the
    # historical public-LB-style positive rate band near 0.53.
    bridge_public_oof = 0.60 * oof_v2 + 0.40 * stepwise_balanced_oof
    bridge_public_test = 0.60 * test_v2 + 0.40 * stepwise_balanced_test
    outputs.append(
        _write_candidate(
            name="submission_xgb_bridge_v2p60_stepwiseBalanced40_t0475",
            source="v2_opt 60% + fold-safe stepwise_balanced_te 40%, lowered threshold",
            passenger_ids=test["PassengerId"],
            pred=bridge_public_test >= 0.475,
            honest_oof_acc=accuracy_score(y, bridge_public_oof >= 0.475),
            threshold=0.475,
            refs=refs,
            details={
                "weights": {"v2_opt": 0.60, "stepwise_balanced_te": 0.40},
                "rationale": "keep stronger OOF while moving test positive rate back near 0.53",
            },
        )
    )

    # Candidate 6: four-way XGB blend.  This is the current strongest honest
    # OOF point we found while still keeping the test positive rate near the
    # historical strong-public-LB band.
    four_way_oof = (
        0.40 * oof_v2
        + 0.40 * stepwise_balanced_oof
        + 0.10 * oof_public
        + 0.10 * oof_groupfill
    )
    four_way_test = (
        0.40 * test_v2
        + 0.40 * stepwise_balanced_test
        + 0.10 * test_public
        + 0.10 * test_groupfill
    )
    outputs.append(
        _write_candidate(
            name="submission_xgb_fourway_v2p40_stepwise40_public10_group10_t0477",
            source="v2_opt 40% + fold-safe stepwise_balanced_te 40% + public_style 10% + groupfill 10%",
            passenger_ids=test["PassengerId"],
            pred=four_way_test >= 0.477,
            honest_oof_acc=accuracy_score(y, four_way_oof >= 0.477),
            threshold=0.477,
            refs=refs,
            details={
                "weights": {
                    "v2_opt": 0.40,
                    "stepwise_balanced_te": 0.40,
                    "public_style": 0.10,
                    "groupfill": 0.10,
                },
                "rationale": "current strongest honest OOF while keeping test positive rate around 0.534",
            },
        )
    )

    # Candidate 7: locally refined version of the four-way blend.
    four_way_refined_oof = (
        0.40 * oof_v2
        + 0.42 * stepwise_balanced_oof
        + 0.10 * oof_public
        + 0.08 * oof_groupfill
    )
    four_way_refined_test = (
        0.40 * test_v2
        + 0.42 * stepwise_balanced_test
        + 0.10 * test_public
        + 0.08 * test_groupfill
    )
    outputs.append(
        _write_candidate(
            name="submission_xgb_fourway_refined_v2p40_stepwise42_public10_group8_t0477",
            source="refined four-way blend around the strongest coarse XGB point",
            passenger_ids=test["PassengerId"],
            pred=four_way_refined_test >= 0.477,
            honest_oof_acc=accuracy_score(y, four_way_refined_oof >= 0.477),
            threshold=0.477,
            refs=refs,
            details={
                "weights": {
                    "v2_opt": 0.40,
                    "stepwise_balanced_te": 0.42,
                    "public_style": 0.10,
                    "groupfill": 0.08,
                },
                "rationale": "local refinement of the strongest four-way blend",
            },
        )
    )

    # Candidate 8: anchor the blend to the actual test positive rate of the
    # historical best submission while improving honest OOF.
    anchor4_oof = (
        0.39 * oof_v2
        + 0.48 * stepwise_balanced_oof
        + 0.11 * oof_public
        + 0.02 * legacy_public_oof
    )
    anchor4_test = (
        0.39 * test_v2
        + 0.48 * stepwise_balanced_test
        + 0.11 * test_public
        + 0.02 * legacy_public_test
    )
    outputs.append(
        _write_candidate(
            name="submission_xgb_anchor511_v2p39_stepwise48_public11_legacy2_t0507",
            source="anchor-511 blend: v2_opt 39% + stepwise_balanced_te 48% + public_style 11% + legacy_public 2%",
            passenger_ids=test["PassengerId"],
            pred=anchor4_test >= 0.507,
            honest_oof_acc=accuracy_score(y, anchor4_oof >= 0.507),
            threshold=0.507,
            refs=refs,
            details={
                "weights": {
                    "v2_opt": 0.39,
                    "stepwise_balanced_te": 0.48,
                    "public_style": 0.11,
                    "legacy_public_style": 0.02,
                },
                "rationale": "match the historical best submission positive rate (~0.51134) instead of chasing a 0.53x prior",
            },
        )
    )

    # Candidate 9: locally refined anchor-511 variant, current strongest
    # XGB-only candidate after re-anchoring to the historical v2_best
    # submission distribution.
    anchor5_oof = (
        0.38613861386138615 * oof_v2
        + 0.45544554455445546 * stepwise_balanced_oof
        + 0.1089108910891089 * oof_public
        + 0.0297029702970297 * legacy_public_oof
        + 0.019801980198019802 * stepwise_safe_te_oof
    )
    anchor5_test = (
        0.38613861386138615 * test_v2
        + 0.45544554455445546 * stepwise_balanced_test
        + 0.1089108910891089 * test_public
        + 0.0297029702970297 * legacy_public_test
        + 0.019801980198019802 * stepwise_safe_te_test
    )
    outputs.append(
        _write_candidate(
            name="submission_xgb_anchor511_v2p39_stepwise46_public11_legacy3_safeTe2_t0507",
            source="anchor-511 refined blend around the strongest v2_best-aligned point",
            passenger_ids=test["PassengerId"],
            pred=anchor5_test >= 0.507,
            honest_oof_acc=accuracy_score(y, anchor5_oof >= 0.507),
            threshold=0.507,
            refs=refs,
            details={
                "weights": {
                    "v2_opt": 0.38613861386138615,
                    "stepwise_balanced_te": 0.45544554455445546,
                    "public_style": 0.1089108910891089,
                    "legacy_public_style": 0.0297029702970297,
                    "stepwise_safe_te": 0.019801980198019802,
                },
                "rationale": "current best honest OOF while matching submission_v2_best test positive rate exactly",
            },
        )
    )

    # Candidate 10: tiny extra weight from the A7 / groupfill families.
    anchor_plus_oof = (
        0.38490445605177737 * oof_v2
        + 0.46093627311420354 * stepwise_balanced_oof
        + 0.11026447231061248 * oof_public
        + 0.02311992005896517 * legacy_public_oof
        + 0.013140661866260665 * stepwise_safe_te_oof
        + 0.005755940836541332 * oof_a7_blend
        + 0.001878275761639495 * oof_groupfill
    )
    anchor_plus_test = (
        0.38490445605177737 * test_v2
        + 0.46093627311420354 * stepwise_balanced_test
        + 0.11026447231061248 * test_public
        + 0.02311992005896517 * legacy_public_test
        + 0.013140661866260665 * stepwise_safe_te_test
        + 0.005755940836541332 * test_a7_blend
        + 0.001878275761639495 * test_groupfill
    )
    outputs.append(
        _write_candidate(
            name="submission_xgb_anchor511_plus_a7blend_groupfill_t0507",
            source="anchor-511 blend with tiny extra A7/groupfill weights",
            passenger_ids=test["PassengerId"],
            pred=anchor_plus_test >= 0.507,
            honest_oof_acc=accuracy_score(y, anchor_plus_oof >= 0.507),
            threshold=0.507,
            refs=refs,
            details={
                "weights": {
                    "v2_opt": 0.38490445605177737,
                    "stepwise_balanced_te": 0.46093627311420354,
                    "public_style": 0.11026447231061248,
                    "legacy_public_style": 0.02311992005896517,
                    "stepwise_safe_te": 0.013140661866260665,
                    "A7_blend": 0.005755940836541332,
                    "groupfill": 0.001878275761639495,
                    "A6": 0.0,
                },
                "rationale": "best local multi-family blend around the anchor-511 solution",
            },
        )
    )

    # Candidate 11: strongest current XGB-only candidate.  Start from the
    # refined anchor-plus blend and only override the uncertainty band using
    # A7/groupfill consensus.
    anchor_plus_override_oof = (anchor_plus_oof >= 0.507).astype(int)
    anchor_plus_override_test = (anchor_plus_test >= 0.507).astype(int)
    uncertain_oof = (anchor_plus_oof >= 0.479) & (anchor_plus_oof <= 0.507)
    uncertain_test = (anchor_plus_test >= 0.479) & (anchor_plus_test <= 0.507)
    force1_oof = uncertain_oof & (oof_a7_blend >= 0.62) & (oof_groupfill >= 0.525)
    force0_oof = uncertain_oof & (oof_a7_blend <= 0.375) & (oof_groupfill <= 0.405)
    force1_test = uncertain_test & (test_a7_blend >= 0.62) & (test_groupfill >= 0.525)
    force0_test = uncertain_test & (test_a7_blend <= 0.375) & (test_groupfill <= 0.405)
    anchor_plus_override_oof[force1_oof] = 1
    anchor_plus_override_oof[force0_oof] = 0
    anchor_plus_override_test[force1_test] = 1
    anchor_plus_override_test[force0_test] = 0
    outputs.append(
        _write_candidate(
            name="submission_xgb_anchor511_plus_a7groupfill_override_t0507",
            source="anchor-plus base with A7/groupfill consensus override on the uncertainty band",
            passenger_ids=test["PassengerId"],
            pred=anchor_plus_override_test.astype(bool),
            honest_oof_acc=accuracy_score(y, anchor_plus_override_oof),
            threshold=0.507,
            refs=refs,
            details={
                "base_weights": {
                    "v2_opt": 0.38490445605177737,
                    "stepwise_balanced_te": 0.46093627311420354,
                    "public_style": 0.11026447231061248,
                    "legacy_public_style": 0.02311992005896517,
                    "stepwise_safe_te": 0.013140661866260665,
                    "A7_blend": 0.005755940836541332,
                    "groupfill": 0.001878275761639495,
                },
                "override": {
                    "uncertain_band": [0.479, 0.507],
                    "force_one_if": {"A7_blend_min": 0.62, "groupfill_min": 0.525},
                    "force_zero_if": {"A7_blend_max": 0.375, "groupfill_max": 0.405},
                    "changed_rows_oof": int(force1_oof.sum() + force0_oof.sum()),
                    "changed_rows_test": int(force1_test.sum() + force0_test.sum()),
                },
                "rationale": "current strongest XGB-only candidate after local uncertainty-band search",
            },
        )
    )

    return outputs


def main() -> None:
    results = [asdict(item) for item in build_candidates()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
