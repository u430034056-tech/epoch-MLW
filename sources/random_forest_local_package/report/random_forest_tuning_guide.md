# Random Forest Tuning Notes

## 1. Objective

- Task: Spaceship Titanic binary classification (`Transported`).
- Metric: accuracy.
- Validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

## 2. Tuning Strategy

- Method: `RandomizedSearchCV`.
- Search budget: default `n_iter=90`, adjustable through `run(tune=True, n_iter=...)`.
- The search space is split by the `bootstrap` setting to avoid invalid parameter combinations.
- `max_samples` is searched only when `bootstrap=True`.
- No `max_samples` value is set when `bootstrap=False`.
- After tuning, OOF probabilities are scanned to select the threshold with the best accuracy.

## 3. Best Recorded Result

- Best CV accuracy: `0.81560`.
- Best OOF threshold: `0.51`.

Best parameters:

```python
{
    "n_estimators": 900,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "max_samples": 0.8,
    "max_leaf_nodes": null,
    "max_features": "sqrt",
    "max_depth": null,
    "class_weight": null,
    "bootstrap": true,
}
```

## 4. Implementation Notes

- The selected parameters are stored in `model_random_forest.py` as `BEST_RF_PARAMS`.
- The inference threshold is stored as `BEST_RF_THRESHOLD=0.51`.
- `python model_random_forest.py` uses the fixed best parameters by default.
- To rerun tuning, use `python -c "import model_random_forest as m; m.run(tune=True, n_iter=90)"`.

## 5. Model Artifact Note

The current script no longer exports a Random Forest `joblib` model by default. It writes the submission CSV and the tuning summary when tuning is enabled.
