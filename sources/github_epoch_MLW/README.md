# epoch-MLW

Spaceship Titanic machine-learning project for AI3023 Machine Learning Workshop.

## Submission Status

Current final Kaggle submission:

`outputs/submissions/final_submission_public_0p81412.csv`

Verified public leaderboard score:

`0.81412`

Submission evidence and metadata:

- `outputs/submissions/kaggle_public_0p81412_evidence.png`
- `outputs/submissions/final_submission_public_0p81412.json`

The final CSV has `4277` prediction rows, exactly the columns `PassengerId,Transported`, boolean predictions, and PassengerId order aligned with `data/raw/sample_submission.csv`.

## Project Boundary

This repository version is cleaned for submission around the final Kaggle file above. The checked-in code path focuses on reproducible preprocessing and submission artifacts. Some exploratory local XGBoost sprint work and public-feedback search logs live outside this GitHub mainline copy.

## Current Entrypoint

Set up dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run preprocessing with:

```bash
python main.py
```

`main.py` builds reusable preprocessing bundles under `processed/` for:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost
- KNN

The preprocessing code reads Kaggle CSVs from `data/raw/`. If the older `spaceship-titanic/` dataset path exists, it is still supported for compatibility.

Generated `processed/` bundles are local artifacts and are ignored by Git.

## Data

Expected files:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

The test file must be the 4277-row Kaggle test set and must not contain `Transported`.

## Repository Contents

- `preprocess.py`: shared preprocessing and model-specific bundle construction.
- `main.py`: preprocessing-only entrypoint.
- `data/raw/`: Kaggle train/test/sample submission CSV files.
- `outputs/submissions/`: final validated Kaggle submission package.
- `README_preprocessing.md`: detailed preprocessing design notes.
- `SUBMISSION_READY_STATUS_2026-05-13.md`: final submission and validation status.

## Reporting Notes

The `0.81412` score is a Kaggle public leaderboard result. It should be reported as the final validated public submission score, not as a guarantee of private leaderboard performance.

The final submission came from a constrained public-feedback recovery step after earlier OOF-strong XGBoost candidates overfit the public leaderboard. In the report, describe this as final submission selection and public-feedback recovery, not as proof that the last small perturbation is a broadly superior modeling method.
