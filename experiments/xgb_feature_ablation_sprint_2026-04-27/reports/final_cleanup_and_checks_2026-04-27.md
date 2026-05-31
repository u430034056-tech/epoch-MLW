# XGB Sprint Final Cleanup And Checks - 2026-04-27

## Scope

- Sprint root: `01_本地副本_实验/xgb_feature_ablation_sprint_2026-04-27/`
- Training input: team preprocessing only.
- Common bundle: `01_本地副本_实验/processed/common/preprocessed_common.joblib`
- XGB bundle: `01_本地副本_实验/processed/xgboost/preprocessed_xgboost.joblib`
- Forbidden source not used as training input: raw Kaggle-template / archived `umanglodaya` raw submissions.

## Top-Level Layout After Cleanup

```text
.
./.venv
./00_GitHub主线_已提交
./01_本地副本_实验
./02_快捷入口索引
./99_本地缓存_可删_未删除
./README_文件夹整理说明.md
```

## Sprint Outputs

- Script: `src/run_feature_ablation_sprint.py`
- Feature matrix: `reports/feature_ablation_matrix.csv`
- Verdict: `reports/feature_verdict.md`
- Submission manifest: `submissions/submission_manifest_top8.csv`
- CSV validation: `reports/submission_validation.csv`
- Source audit: `archive_inputs/source_audit.json`

## Feature Verdict

- 必须保留: `spend_core`, `cabin`, `age`
- 可删 / 疑似冗余: `homeplanetdestination`, `vip`, `surname`, `cabinnum`
- 待验证: `spend_structure`, `broad_categorical`, `missing_flags`, `group`, `groupmemberno`

## Candidate CSVs

Top sprint candidate:
`submissions/submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5366_public.csv`

The sprint keeps 8 active CSV candidates under `submissions/`.
Every active candidate has:

- 4277 rows
- Columns exactly `PassengerId,Transported`
- `PassengerId` order aligned with team common `test_ids`
- Boolean `Transported`

## Safety Checks

- GitHub main status: clean.
- Top-level symlinks: none.
- Active Kaggle queue top-level CSV count: 16.
- Active Kaggle queue top-level `umanglodaya` raw-template CSV count: 0.
- Non-cache `__pycache__` count after cleanup: 0.
