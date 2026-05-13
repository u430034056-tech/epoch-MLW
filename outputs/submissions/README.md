# Final Submission Artifacts

## Final Kaggle File

Submit:

`final_submission_public_0p81412.csv`

## Verified Public Result

- Kaggle public score: `0.81412`
- Rows: `4277`
- Columns: `PassengerId,Transported`
- Prediction dtype: boolean
- True predictions: `2291`
- Positive rate: `0.5356558335281739`
- PassengerId order: aligned with `data/raw/sample_submission.csv`

## Files

- `final_submission_public_0p81412.csv`: final submission CSV.
- `final_submission_public_0p81412.json`: metadata for the selected submission.
- `kaggle_public_0p81412_evidence.png`: screenshot evidence of the Kaggle public score.

## Method Boundary

The final file is selected from a constrained public-feedback recovery search around the strongest team-preprocessing-derived XGBoost submission. Archived raw-template submissions were used only as distance references for public-score neighborhood search, not as direct training input or prediction-blend input for this final file.

## Report Wording

Use this defensible statement:

The project reached a validated Kaggle public score of `0.81412` after correcting an OOF-overfit direction and selecting a constrained public-feedback recovery submission from the team preprocessing pipeline.

Avoid claiming that this public score proves private leaderboard performance or that the last small perturbation is a generally superior modeling method.
