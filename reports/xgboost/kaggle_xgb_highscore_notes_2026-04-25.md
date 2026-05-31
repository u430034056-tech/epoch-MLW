# Kaggle XGB High-Score Notes - 2026-04-25

Scope: all local optimization below keeps the team preprocessing/feature-engineering outputs as the source of truth. Raw Kaggle-template preprocessing is used only as research evidence, not as the active submission base.

## Sources Checked

- Umang Lodaya, "Spaceship Titanic | XGBoost": public score shown on Kaggle page as 0.81575. Relevant transferable ideas: XGBoost parameter set, permutation-importance drop list, SMOTE before final fit, seed sensitivity.
- Kaggle discussion "SHAP values- making sense out of Feature Importance": XGBoost/SHAP discussion highlights CryoSleep and spend features as dominant drivers.
- Local Kaggle-template reproductions already present in this workspace:
  - `src/xgboost/run_umanglodaya_xgb.py`
  - `src/xgboost/run_public_style.py`
  - `src/xgboost/run_arunklenin_xgb.py`
  - `src/xgboost/run_kaggle_fe_xgb.py`

## Transferable Ideas Kept

- Keep team bundle as model input: `processed/xgboost/preprocessed_xgboost.joblib`.
- Translate Umang's low permutation-importance drop list onto team feature names:
  - `num__ShoppingMall`, `num__Age`, `cat__CryoSleep_True`, all one-hot `HomePlanet`, one-hot `VIP`, one-hot `Destination`, and `num__FoodCourt`.
- Preserve the high-score XGB regularization parameters:
  - `reg_lambda=3.0610042624477543`
  - `reg_alpha=4.581902571574289`
  - `colsample_bytree=0.9241969052729379`
  - `subsample=0.9527591724824661`
  - `learning_rate=0.06672065863100594`
  - `n_estimators=730`
  - `max_depth=5`
  - `min_child_weight=1`
- Test SMOTE as a model-side step only:
  - local SMOTE interpolation implementation
  - `sampling_strategy=1`
  - `k_neighbors=5` and `k_neighbors=3`
  - seeds `2024, 17, 7, 42, 88`
- Continue blending with team probability artifacts:
  - `reports/xgboost/logs/A7_oof.csv`
  - `reports/xgboost/logs/A7_test_proba.csv`
  - `reports/xgboost/logs/A6_oof.csv`
  - `reports/xgboost/logs/A6_test_proba.csv`
- Keep public-LB positive-rate anchors as thresholding candidates:
  - `0.5127425766`, `0.5167173252`, `0.5323825111`, `0.5351882160`, `0.53659`

## Local Result

Best current local OOF candidate:

- File: `submission_blend_team_xgb095_umang_k3_multi5_w25_team_A7_blend_w75_oofbest_t0p474.csv`
- Feature source: team 107-feature XGB bundle after dropping 12 Kaggle low-importance columns.
- XGB side: Umang parameter set.
- SMOTE side: local SMOTE, `sampling_strategy=1`, `k_neighbors=3`, five seeds.
- Blend: 25% new XGB probability + 75% team A7 probability.
- OOF best accuracy: `0.8216956172`.
- Test positive rate: `0.5391629647`.
- True count: `2306 / 4277`.

Leaderboard-risk ordering for next submissions:

1. `submission_blend_team_xgb095_umang_k3_multi5_w25_team_A7_blend_w75_rate5366_public.csv`
2. `submission_blend_team_xgb095_umang_k3_multi5_w25_team_A7_blend_w75_rate5352_umang.csv`
3. `submission_blend_team_xgb095_umang_k3_multi5_w25_team_A7_blend_w75_oofbest_t0p474.csv`
4. `submission_blend_team_xgb095_umang_k3_multi5_w25_team_A7_blend_w75_rate5324_anchor.csv`

Reasoning: the OOF-best threshold has the strongest local score but the positive rate is higher than the best recent public-LB band. The `rate5366_public` and `rate5352_umang` files stay closer to proven public-score positive-rate regions.
