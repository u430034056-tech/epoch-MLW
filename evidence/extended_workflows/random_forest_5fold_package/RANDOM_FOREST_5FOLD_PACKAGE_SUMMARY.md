# Random Forest 5-Fold Package Summary

This folder preserves the local Random Forest package received as `rf_local_package_2026-05-29 (2).zip` and keeps it separate from the main report's lightweight Random Forest baseline.

## What This RF Package Uses

- Validation strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Tuning method: `RandomizedSearchCV`, accuracy scoring, default `n_iter=90`.
- Post-tuning threshold search: OOF probability sweep, best threshold `0.51`.
- Reported package tuning result: best CV accuracy `0.8155994179425367`; OOF accuracy at threshold `0.8157137927067756`.
- Best parameters: `n_estimators=2600`, `max_depth=16`, `max_features=0.5`, `criterion=log_loss`, `class_weight=balanced`, `bootstrap=False`, `max_leaf_nodes=128`, `min_samples_split=5`, `min_samples_leaf=5`, `ccp_alpha=3e-05`.

## Local Rerun Check

Command run from `group3/sources/random_forest_local_package/`:

```bash
../../reproduction/.venv/bin/python model_random_forest.py > ../../reproduction/logs/random_forest_5fold_package_run.log 2>&1
```

The script completed and wrote `submission_rf.csv`.

- Current-environment fixed-params 5-fold CV accuracy printed by the script: `0.81514`.
- Generated submission rows: `4277`.
- PassengerId order matches `sample_submission.csv`: `True`.
- Generated True count: `2172`.
- Original package True count: `2167`.
- Prediction difference between original package submission and current rerun submission: `7 / 4277`.

## Boundary

This is the RF project-side 5-fold tuning workflow. It differs from the report's lightweight tuning table, which summarized a smaller 3-fold random-search snapshot.

The rerun emits scikit-learn version warnings because the bundled preprocessing joblib was created with scikit-learn `1.4.2` and the group3 environment uses scikit-learn `1.6.1`. The package is runnable, but the current rerun should not be described as bit-for-bit identical to the original package output.
