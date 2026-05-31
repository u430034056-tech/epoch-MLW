# group3 Manifest

## Purpose

This repository root stores the public, GitHub-friendly contents of the final `group3` package: final report, presentation deck, source-code snapshots, raw data, and supporting evidence for the Spaceship Titanic ML Workshop project.

The submitted course archive was `course_project_materials_group3.zip`. This GitHub mirror intentionally excludes heavyweight generated caches and trained-model binaries such as `processed/`, `*.joblib`, and `*.cbm`.

## Source Code

- Main GitHub snapshot: `sources/github_epoch_MLW`, remote `https://github.com/u430034056-tech/epoch-MLW.git`, commit `8f61ec248a4d4b082ef232ed3d8309b45fb27025`.
- Upload branch export: `sources/github_upload_project_kevinhe_full`, commit `73de2c325f46786b5aeef070f43c116a476b7d5d`.
- Random Forest branch export: `sources/github_feature_random_forest_full`, commit `47b0ea7250f5f4bb1bf1fd83d54d839a1880a2a2`.
- Selected model-code copies are stored under `sources/model_code_selected/`.
- LightGBM top-feature source is stored under `sources/lightgbm_submission/`.
- Random Forest local package source is stored under `sources/random_forest_local_package/`.
- XGBoost submission-workflow source and evidence are stored under `evidence/extended_workflows/xgboost_submission_workflow/`.

## Original Data

- `train.csv`: shape `(8693, 14)`, includes `Transported`.
- `test.csv`: shape `(4277, 13)`, does not include `Transported`.
- `sample_submission.csv`: shape `(4277, 2)`, columns `PassengerId, Transported`.
- Data is present in `data/raw/`, `data/spaceship-titanic/`, and `sources/data/`.

## Final Report And PPT

- Final report PDF: `report/main(1).pdf`.
- Final report LaTeX source: `report/main.tex`.
- Presentation deck: `ppt/ml ppt.pptx`.
- Source-group copies are also stored under `sources/report_original/` and `sources/ppt_original/`.

## Main Report Results

- LightGBM has the highest local validation accuracy in the report table: `0.8171`.
- CatBoost has the highest local validation ROC-AUC in the report table: `0.9125`.
- XGBoost has the strongest retained Kaggle public leaderboard evidence: `0.81412`.
- CatBoost, Random Forest, and Logistic Regression public-score notes are not used as package evidence because their original leaderboard screenshots or submission records are not retained in this package.
- The final XGBoost evidence file is `evidence/xgboost_public_lb/final_submission_public_0p81412.csv`.

## Reproduction Commands

Run from `reproduction/` after installing `requirements.txt`:

```bash
../.venv/bin/python -u run_lightgbm_top30.py
../.venv/bin/python -u run_catboost_project_cv.py
../.venv/bin/python -u run_logistic_project_final.py
../.venv/bin/python -u run_random_forest_github_branch.py
```

The source snapshots also contain the original model scripts used for Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.

## Supporting Evidence

- `evidence/lightgbm_top30/`: LightGBM top-feature rerun summary and artifacts.
- `evidence/extended_workflows/`: project-side CV, tuning, and submission-workflow evidence.
- `evidence/xgboost_public_lb/`: final XGBoost public leaderboard score evidence and final CSV.
- `PACKAGE_VALIDATION.md`: data and submission shape checks.

## Submission Validation Summary

- Final public XGBoost CSV: `4277` rows, columns `PassengerId, Transported`, PassengerId order matches `sample_submission.csv`, boolean target values, true count `2291`.
- All generated Kaggle-style submission files in the package use the expected two-column format and the official test-set row count.

## Notes

- Kaggle public leaderboard scores are treated as public submission feedback, not private-leaderboard guarantees.
- The report, PPT, code snapshots, raw data, and final public-score screenshot are included here.
- The full submitted ZIP is the archival copy for heavyweight generated artifacts that are not appropriate for GitHub `main`.
