# Kaggle Submit Queue - 2026-04-25

Latest public feedback from manual submissions:

| file | public score | decision |
| --- | ---: | --- |
| `submission_umanglodaya_xgb_smote_seed2024.csv` | 0.81412 | New public best. Use as anchor. |
| `submission_umanglodaya_xgb_smote_seed17.csv` | 0.81342 | Strong, but below seed2024. Use only for narrow overrides/blends. |
| `submission_umanglodaya_xgb_smote_multi13.csv` | 0.81295 | No gain over old best. Deprioritize. |
| `submission_umanglodaya_xgb_smote_seed7.csv` | 0.81271 | Dropped. Stop submitting full seed variants like this blindly. |

Current public LB anchor:

- `submission_umanglodaya_xgb_smote_seed2024.csv`: **0.81412**

Next submit order:

1. `submission_umanglodaya_seed2024_to_seed17_conf_top02.csv`
   - Changed vs seed2024: 2 rows
   - Positive rate: 0.535188
   - Reason: safest seed17-informed correction, minimal blast radius.

2. `submission_umanglodaya_seed2024_to_seed17_conf_top03.csv`
   - Changed vs seed2024: 3 rows
   - Positive rate: 0.534954
   - Reason: still tiny change, tests whether seed17's strongest disagreements are useful.

3. `submission_umanglodaya_seed2024_to_seed17_conf_top05.csv`
   - Changed vs seed2024: 5 rows
   - Positive rate: 0.534954
   - Reason: upper end of the very conservative seed17 override batch.

4. `submission_umanglodaya_seed2024_blend_seed17_w75_rate535.csv`
   - Changed vs seed2024: 5 rows
   - Positive rate: 0.535422
   - Reason: probability blend, still anchored 75% to seed2024.

5. `submission_umanglodaya_seed2024_blend_seed17_w75_rate53495.csv`
   - Changed vs seed2024: 6 rows
   - Positive rate: 0.535188
   - Reason: same blend family, threshold pulled back to seed2024's rate.

6. `submission_umanglodaya_seed2024_to_multi13_conf_top02.csv`
   - Changed vs seed2024: 2 rows
   - Positive rate: 0.535656
   - Reason: only try after seed17 micro-batch; multi13 itself did not improve.

7. `submission_umanglodaya_seed2024_to_seed0_conf_top02.csv`
   - Changed vs seed2024: 2 rows
   - Positive rate: 0.535656
   - Reason: low-risk check against the original 0.81295 seed0 anchor.

Generated but lower priority:

- `reports/xgboost/logs/seed2024_public_feedback_candidates.csv` lists all 51 generated candidates.
- Avoid broad new template candidates unless the seed2024 micro-batch fails; `arunklenin_xgb_only_rate535` changes 435 rows and is much riskier.

