# Extended Workflow Reproduction Summary

This file records project-side tuning and submission-workflow evidence used by the report.

## What Was Added

- LR / RF / CatBoost: reran 3-fold candidate CV evidence using fold-local preprocessing.
- RF project-side package: added the separate 5-fold RandomizedSearchCV / OOF-threshold local package from `rf_local_package_2026-05-29 (2).zip`.
- LightGBM top-feature: links to the already rerun top-20/top-30/top-40 workflow; top-30 reproduces the report/PPT CV value.
- XGBoost submission workflow: copied source, ablation/Optuna/fusion logs, final submission artifacts, and validates the archived final CSV.

## Report Tuning / Workflow Comparison

| Model/workflow | Reported CV | Local/archived value | Status |
|---|---:|---:|---|
| Logistic Regression | 0.7866 | 0.7899 | same_protocol_rerun_not_exact_random_search_match |
| Random Forest | 0.8069 | 0.8094 | same_protocol_rerun_not_exact_random_search_match |
| Random Forest 5-fold package | 0.8156 | 0.8151 current rerun / 0.8157 package OOF threshold | separate_project_side_workflow |
| XGBoost | 0.8223 | 0.8223 | rounded_match |
| LightGBM top-feature | 0.8177 | 0.8177 | rounded_match |
| CatBoost | 0.8138 | 0.8156 | same_protocol_rerun_not_exact_random_search_match |

## XGBoost Final Submission Evidence

- Final CSV rows: `4277`.
- PassengerId order matches sample submission: `True`.
- True count: `2291`.
- Kaggle public leaderboard scores are preserved as historical external evidence.

## Boundary

The files here cover the report's tuning, OOF/CV, ablation, and submission-selection evidence.
