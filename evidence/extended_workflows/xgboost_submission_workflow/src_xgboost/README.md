# XGBoost Submission Workflow

This folder contains the archived XGBoost-only workflow used for the Spaceship Titanic project.

The final public-score artifact used by the report is:

`../final_submission_2026-05-13/final_submission_public_0p81412.csv`

It corresponds to a Kaggle public leaderboard score of `0.81412`. The score is treated only as public leaderboard feedback, not as private leaderboard evidence.

## Main Components

- `run_v2.py`: clean XGBoost V2 workflow without unsafe target encoding.
- `run_public_style.py`: raw-CSV public-style XGBoost candidate workflow.
- `run_public_groupfill.py`: raw-CSV group-consistent fill variant.
- `run_team_xgb_sprint.py`: team-preprocessing candidate generation workflow.
- `features.py`: fold-aware feature construction helpers.
- `cv.py`: grouped cross-validation utilities.
- `tune.py`: Optuna-based parameter search utilities.
- `ensemble.py`: multi-seed and family-fusion helpers.
- `postprocess.py`: threshold and submission post-processing helpers.

## Validation Notes

The workflow separates local OOF validation from Kaggle public feedback. Target-encoding variants that inflated OOF results were not used for the final public-score artifact. The final report uses the public `0.81412` score as external submission evidence for the selected XGBoost submission branch.
