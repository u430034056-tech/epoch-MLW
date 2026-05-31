# XGB Feature Ablation Sprint - 2026-04-27

本目录是隔离实验副本，只用于判断团队 XGBoost 特征中哪些必须保留、哪些疑似冗余，并生成少量 Kaggle 候选 CSV。

硬约束：

- 训练输入只来自团队统一预处理产物：
  - `../processed/common/preprocessed_common.joblib`
  - `../processed/xgboost/preprocessed_xgboost.joblib`
- 不修改 `../processed/`。
- 不修改 `../../00_GitHub主线_已提交/epoch-MLW-main/`。
- raw Kaggle-template 高分分支只能作为参数、SMOTE、seed、正例率锚定的启发，不能作为本目录训练输入。

运行入口：

```bash
.venv/bin/python 01_本地副本_实验/xgb_feature_ablation_sprint_2026-04-27/src/run_feature_ablation_sprint.py
```

主要输出：

- `reports/feature_ablation_matrix.csv`
- `reports/feature_verdict.md`
- `reports/feature_importance_stability.csv`
- `reports/submission_validation.csv`
- `submissions/submission_manifest_top8.csv`
- `archive_inputs/source_audit.json`
