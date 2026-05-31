"""XGBoost pipeline for the Spaceship Titanic Kaggle competition.

Modules
-------
config       : paths, seeds, column schemas, search spaces
data         : load the shared `common` DataFrames and slice XGBoost inputs
cv           : StratifiedGroupKFold + fold-aware preprocessing
features     : fold-local feature engineering (SurnameFreq, bin edges, target encoding)
model        : build_xgb_model, fit_with_early_stopping
tune         : Optuna Bayesian hyper-parameter search
train        : top-level CV trainer producing OOF + submissions
evaluate     : ROC / PR / confusion matrix / feature importance / SHAP / learning curves
postprocess  : threshold scan + rule-based corrections
ensemble     : multi-seed / multi-config averaging
"""
