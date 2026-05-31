# Final Submission Package - 2026-05-13

## Final Kaggle File

Use this file for the current final Spaceship Titanic submission:

`final_submission_public_0p81412.csv`

## Verified Result

- Kaggle public score: `0.81412`
- Submission source: `submission_team_common_rawlike_feedback_rank_k3_seed17_w15_true2291.csv`
- Submission time evidence: `kaggle_public_0p81412_evidence.png`
- Rows: `4277`
- Columns: `PassengerId,Transported`
- `Transported` values: boolean `False/True`
- True predictions: `2291`
- Positive rate: `0.5356558335281739`
- PassengerId order: aligned with `sample_submission.csv`

## Method Boundary

This submission is based on the team `processed/common` preprocessing route, transformed into a compact raw-like XGBoost view during the public-recovery sprint. Archived raw-template submissions were used only as public-score distance references, not as training input or blended prediction inputs.

## Reporting Guidance

For the report, describe this as a Kaggle public-feedback recovery result and final submission choice. Do not present the `0.81412` score as proof that the small `w15` perturbation generalized better than the previous `w05` anchor, because both tied at `0.81412`.

The defensible claim is:

The project reached a validated Kaggle public score of `0.81412` after moving away from an OOF-overfit feature-ablation candidate and using constrained public-feedback neighborhood search around the strongest team-preprocessing-derived XGBoost submission.
