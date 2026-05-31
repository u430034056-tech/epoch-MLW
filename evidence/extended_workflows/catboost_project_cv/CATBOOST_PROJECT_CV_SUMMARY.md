# CatBoost Project 5-Fold CV Summary

- Source: `origin/upload/project-kevinhe` full export.
- Protocol: fixed report project-side CatBoost config with 5-fold OOF threshold search.
- Config: `iterations=2000`, `learning_rate=0.03`, `depth=6`, `l2_leaf_reg=5.0`.
- CV folds: `5`, random_state: `42`, early stopping rounds: `100`.
- Best OOF threshold: `0.49`.
- Mean CV accuracy: `0.818935`.
- OOF accuracy: `0.818935`.
- OOF ROC-AUC: `0.906204`.
- Final train accuracy: `0.885195`.
- Submission: `/Users/shenyijie/Desktop/MLWP project/group3/reproduction/outputs/catboost_project_cv/submissions/submission_catboost.csv`.
- Submission valid: `True`.

This supplements the existing strict 80/20 CatBoost rerun; it is the report/PPT project-workflow lane.
