# group3 Package Validation

## Data Checks

- train.csv: shape=(8693, 14), has Transported=True
- test.csv: shape=(4277, 13), has Transported=False
- sample_submission.csv: shape=(4277, 2), columns=['PassengerId', 'Transported']

## Submission Checks

- reproduction/outputs/catboost_project_cv/submissions/submission_catboost.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top20_features.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top30_features.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top40_features.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/logistic_project_final/submissions/submission_logistic_regression.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/random_forest_5fold_package/submission_rf_original_package.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/random_forest_5fold_package/submission_rf_rerun_current_env.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- reproduction/outputs/random_forest_github_branch/submission_rf_github_feature_branch.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/catboost_project_cv/submissions/submission_catboost.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/logistic_project_final/submissions/submission_logistic_regression.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/random_forest_5fold_package/local_rerun/submission_rf_rerun_current_env.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/random_forest_5fold_package/submission_rf.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/random_forest_github_branch/submission_rf_github_feature_branch.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True
- evidence/extended_workflows/xgboost_submission_workflow/final_submission_2026-05-13/final_submission_public_0p81412.csv: shape=(4277, 2), columns=['PassengerId', 'Transported'], id_order_match=True, boolean_like=True

## Required Folders

- `sources` exists: True
- `data` exists: True
- `reproduction` exists: True
- `evidence` exists: True
- `report` exists: True
- `ppt` exists: True
