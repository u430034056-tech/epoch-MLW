# Submission Ready Status - 2026-05-13

## Final State

The repository is ready for course/GitHub submission around the final Kaggle public score:

`0.81412`

Final submission file:

`outputs/submissions/final_submission_public_0p81412.csv`

## Verification

Final CSV:

- Rows: `4277`
- Columns: `PassengerId,Transported`
- PassengerId order aligned with `data/raw/sample_submission.csv`
- `Transported` values are boolean
- True predictions: `2291`
- Positive rate: `0.5356558335281739`

Preprocessing entrypoint:

```bash
python main.py
```

Verified locally with the project virtual environment:

```bash
../../.venv/bin/python main.py
```

The command successfully loads or builds preprocessing bundles for Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM, CatBoost, and KNN.

## Cleanup Completed

- Corrected `data/raw/test.csv` to the real 4277-row Kaggle test file.
- Removed fragile imports from `main.py`; preprocessing no longer depends on incomplete training-helper modules.
- Added `data/raw` fallback path handling in `preprocess.py`.
- Ignored generated `processed/` artifacts in Git.
- Added final submission CSV, metadata, and public-score screenshot evidence under `outputs/submissions/`.
- Cleaned README wording so the final submission file and the score boundary are explicit.

## Remaining Boundary

The final `0.81412` score is a public leaderboard score. It is suitable as the validated public submission result, but it is not a private leaderboard guarantee.
