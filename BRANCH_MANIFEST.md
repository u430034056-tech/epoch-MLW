# Branch Manifest

## Source Workspace

- Local source root: `/Users/shenyijie/Desktop/MLWP project/01_本地副本_实验`
- GitHub branch name: `shenyijie-xgb`
- Purpose: archive the local XGBoost-only experiment workflow separately from the final group submission package on `main`.

## Copied Paths

- `src/xgboost/` from local `src/xgboost/`
- `data/raw/` from local `data/raw/`
- `reports/xgboost/` from local `reports/xgboost/`
- `experiments/xgb_feature_ablation_sprint_2026-04-27/` from local `xgb_feature_ablation_sprint_2026-04-27/`
- `experiments/xgb_public_recovery_sprint_2026-04-27/` from local `xgb_public_recovery_sprint_2026-04-27/`
- `submissions/final_submission_2026-05-13/` from local `final_submission_2026-05-13/`
- `submissions/next_kaggle_candidates/` from local Kaggle candidate queue; non-English local directory names were normalized to English for GitHub.
- `tools/tmp_scripts/` from local `tmp/*.py`

## Excluded Generated Artifacts

- Python caches: `__pycache__/`, `.ipynb_checkpoints/`
- Local environments: `.venv/`
- Rebuildable preprocessing caches: `processed/`
- Heavy model binaries: `*.joblib`, `*.cbm`, `*.pkl`, `*.pickle`
- Archive files: `*.zip`
