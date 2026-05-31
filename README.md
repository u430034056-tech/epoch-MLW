# group3 Submission Package

This repository mirrors the public, GitHub-friendly contents of the final `group3` submission package for the Spaceship Titanic ML Workshop project.

The course ZIP submitted through iSpace was `course_project_materials_group3.zip`. This `main` branch keeps the report, presentation slides, runnable source code, raw data, validation summaries, and key evidence, while excluding heavyweight generated caches and trained-model binaries such as `processed/`, `*.joblib`, and `*.cbm`.

## Main Files

- `report/main(1).pdf`: final report PDF.
- `report/main.tex`: LaTeX source of the report.
- `ppt/ml ppt.pptx`: presentation slides.
- `data/raw/`: original Kaggle `train.csv`, `test.csv`, and `sample_submission.csv`.
- `sources/`: source-code snapshots from GitHub branches and local model packages.
- `reproduction/`: selected scripts used to rerun project-side workflows.
- `evidence/`: logs, metrics, Kaggle public-score evidence, and validated submissions.
- `MANIFEST.md`: source commits, commands, and result summaries.
- `PACKAGE_VALIDATION.md`: data and submission validation checks.

## GitHub Source

The main GitHub repository snapshot is included in `sources/github_epoch_MLW/`.

Remote: `https://github.com/u430034056-tech/epoch-MLW.git`

Main source snapshot used in the submitted package: `8f61ec248a4d4b082ef232ed3d8309b45fb27025`

Additional source snapshots are included for the upload branch, Random Forest branch, LightGBM submission code, and local Random Forest 5-fold package. See `MANIFEST.md` for details.

## Environment

Create a Python environment and install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The reproduction environment used:

- `pandas`
- `numpy`
- `scipy`
- `joblib`
- `scikit-learn==1.6.1`
- `xgboost`
- `lightgbm`
- `catboost`

## Reproduction

Run from `reproduction/`:

```bash
../.venv/bin/python -u run_lightgbm_top30.py
../.venv/bin/python -u run_catboost_project_cv.py
../.venv/bin/python -u run_logistic_project_final.py
../.venv/bin/python -u run_random_forest_github_branch.py
```

Detailed commands and output locations are listed in `MANIFEST.md`.

## Notes

- Kaggle public leaderboard scores are preserved as historical evidence only.
- The package validates generated submissions for row count, columns, PassengerId order, and boolean target values.
- Heavy generated artifacts were intentionally left out of this GitHub branch; the submitted course ZIP remains the full archival package.
