# Shenyijie XGBoost Workflow

This branch archives Shenyijie's local XGBoost experiment workflow for the Spaceship Titanic ML Workshop project.

It is intentionally separate from `main`. The `main` branch mirrors the final group3 submission package; this branch focuses on the XGBoost-only research, validation, tuning, public-feedback recovery, and final submission path.

## Contents

- `src/xgboost/`: core XGBoost workflow code, including data loading, feature engineering, cross-validation, tuning, training, ensembling, postprocessing, and runnable entrypoints.
- `data/raw/`: Kaggle raw `train.csv`, `test.csv`, and `sample_submission.csv`.
- `reports/xgboost/`: experiment logs, OOF/test probabilities, SHAP/importance figures, ablation records, Optuna results, and candidate submissions.
- `experiments/xgb_feature_ablation_sprint_2026-04-27/`: feature-ablation sprint source, reports, and submissions.
- `experiments/xgb_public_recovery_sprint_2026-04-27/`: public-feedback recovery sprint source, reports, and submissions.
- `submissions/final_submission_2026-05-13/`: retained final public-leaderboard evidence and final CSV for the `0.81412` public score.
- `submissions/next_kaggle_candidates/`: active and archived Kaggle candidate queue from the local workspace.
- `tools/tmp_scripts/`: temporary scripts used during XGBoost probing, diagnostics, Optuna, blending, and plotting.

Generated caches and heavy binary artifacts are excluded from this branch: `__pycache__/`, `.venv/`, `processed/`, `*.joblib`, `*.cbm`, `*.pkl`, `*.pickle`, and `*.zip`.

## Main Entry Points

Run from the repository root after creating a Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m src.xgboost.run_v2 --data-dir data/raw
```

One-off high-score reproduction command used for the retained XGBoost path:

```bash
PYTHONPATH=. .venv/bin/python -m src.xgboost.run_umanglodaya_xgb --data-dir data/raw --seeds 2024
```

## Evidence Boundary

The `0.81412` value is retained as Kaggle public leaderboard evidence, not as a private-leaderboard guarantee. Local validation, OOF experiments, public-feedback recovery, and the final submission artifact are preserved as separate evidence lanes.
