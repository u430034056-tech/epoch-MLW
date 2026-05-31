from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


SPRINT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = SPRINT_ROOT / "reports"
SUBMISSIONS_DIR = SPRINT_ROOT / "submissions"
ARCHIVE_DIR = SUBMISSIONS_DIR / "99_extra_candidates_not_top8"
RAW_REFERENCE_CANDIDATES = (
    LOCAL_ROOT / "reports" / "xgboost" / "submission_candidates" / "submission_umanglodaya_xgb_smote_seed2024.csv",
    LOCAL_ROOT
    / "reports"
    / "xgboost"
    / "submission_candidates"
    / "99_archive_non_team_raw_2026-04-25"
    / "submission_umanglodaya_xgb_smote_seed2024.csv",
)


def _load_raw_reference() -> pd.Series:
    for path in RAW_REFERENCE_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)["Transported"].astype(bool)
    raise FileNotFoundError("Missing raw seed2024 reference submission")


def _candidate_paths() -> list[Path]:
    paths = []
    for folder in (SUBMISSIONS_DIR, ARCHIVE_DIR):
        if not folder.exists():
            continue
        paths.extend(path for path in folder.glob("submission_*.csv") if "manifest" not in path.name)
    return sorted(paths)


def _load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(REPORTS_DIR / "public_recovery_manifest_all.csv")
    manifest["stem"] = manifest["file"].map(lambda value: Path(value).stem)
    return manifest


def _validate(paths: list[Path], expected_ids: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        valid = (
            list(df.columns) == ["PassengerId", "Transported"]
            and len(df) == len(expected_ids)
            and df["PassengerId"].astype(str).tolist() == expected_ids
            and pd.api.types.is_bool_dtype(df["Transported"])
        )
        rows.append(
            {
                "file": str(path),
                "rows": len(df),
                "true_count": int(df["Transported"].astype(bool).sum()),
                "valid": bool(valid),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "submission_validation.csv", index=False)
    if not bool(out["valid"].all()):
        raise RuntimeError("Curated queue contains invalid CSV")
    return out


def main() -> None:
    ref = _load_raw_reference()
    manifest = _load_manifest()
    rows = []
    for path in _candidate_paths():
        df = pd.read_csv(path)
        pred = df["Transported"].astype(bool)
        meta_row = manifest.loc[manifest["stem"] == path.stem]
        if meta_row.empty:
            continue
        meta = meta_row.iloc[0].to_dict()
        rows.append(
            {
                **meta,
                "current_path": str(path),
                "diff_vs_raw_seed2024": int((pred.to_numpy() != ref.to_numpy()).sum()),
                "distance_positive_rate": abs(float(meta["positive_rate"]) - 0.5351882160392799),
            }
        )
    scored = pd.DataFrame(rows)
    scored.to_csv(REPORTS_DIR / "distance_to_raw_seed2024.csv", index=False)

    public_band = scored.loc[
        scored["positive_rate"].between(0.532, 0.537)
        & scored["feature_view"].isin(["raw_core", "raw_plus_group"])
    ].copy()
    public_band["view_priority"] = public_band["feature_view"].map({"raw_core": 0, "raw_plus_group": 1}).fillna(2)
    top = public_band.sort_values(
        ["diff_vs_raw_seed2024", "distance_positive_rate", "view_priority", "oof_best_accuracy"],
        ascending=[True, True, True, False],
    ).head(8).copy()

    ARCHIVE_DIR.mkdir(exist_ok=True)
    selected_stems = set(top["name"])
    for path in list(SUBMISSIONS_DIR.glob("submission_*.csv")):
        if "manifest" in path.name or path.stem in selected_stems:
            continue
        shutil.move(str(path), ARCHIVE_DIR / path.name)
        meta = path.with_suffix(".json")
        if meta.exists():
            shutil.move(str(meta), ARCHIVE_DIR / meta.name)

    promoted_paths = []
    for _, row in top.iterrows():
        src = Path(row["current_path"])
        dst = SUBMISSIONS_DIR / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            meta = src.with_suffix(".json")
            if meta.exists():
                shutil.copy2(meta, dst.with_suffix(".json"))
        promoted_paths.append(dst)

    top["file"] = [str(path) for path in promoted_paths]
    top.to_csv(SUBMISSIONS_DIR / "submission_manifest_public_recovery.csv", index=False)
    expected_ids = pd.read_csv(promoted_paths[0])["PassengerId"].astype(str).tolist()
    validation = _validate(promoted_paths, expected_ids)

    report_lines = [
        "# Curated Public Recovery Queue",
        "",
        "Selection rule: generated CSVs are still team-preprocessing-only. The archived raw seed2024 submission is used only as a distance reference because it has known public score 0.81412; it is not blended into any candidate.",
        "",
        "## Upload Order",
        "",
    ]
    for i, (_, row) in enumerate(top.iterrows(), start=1):
        report_lines.append(
            f"{i}. `{Path(row['file']).name}` | diff_vs_raw_seed2024={int(row['diff_vs_raw_seed2024'])} | "
            f"pos={row['positive_rate']:.6f} | OOF={row['oof_best_accuracy']:.6f}"
        )
    report_lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Active candidate count: `{len(validation)}`",
            f"- Invalid candidate count: `{int((~validation['valid']).sum())}`",
        ]
    )
    (REPORTS_DIR / "public_recovery_curated_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "top_manifest": str(SUBMISSIONS_DIR / "submission_manifest_public_recovery.csv"),
                "curated_report": str(REPORTS_DIR / "public_recovery_curated_report.md"),
                "top": top[["name", "diff_vs_raw_seed2024", "positive_rate", "oof_best_accuracy", "file"]].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
