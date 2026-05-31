# XGB Feature Ablation Sprint Report - 2026-04-27

Scope: team preprocessing only. This run reads `processed/common` and `processed/xgboost`; raw Kaggle-template preprocessing is not used as model input.

## Baseline

- `full_common_native` OOF best accuracy: `0.814448`
- Experiment rows: `38`
- Direct drop tests used for feature verdict: `12`
- SMOTE and blend rows are excluded from feature-redundancy classification.

## 必须保留

- spend_core: deletion lowered OOF by -0.0219
- cabin: deletion lowered OOF by -0.0171
- age: deletion lowered OOF by -0.0022

## 可删 / 疑似冗余

- homeplanetdestination: deletion did not hurt OOF (+0.0002)
- vip: deletion did not hurt OOF (+0.0009)
- surname: deletion did not hurt OOF (+0.0012)
- cabinnum: deletion did not hurt OOF (+0.0020)

## 待验证

- spend_structure: small negative delta (-0.0012), needs public-LB check
- broad_categorical: small negative delta (-0.0012), needs public-LB check
- missing_flags: small negative delta (-0.0009), needs public-LB check
- group: small negative delta (-0.0007), needs public-LB check
- groupmemberno: small negative delta (-0.0001), needs public-LB check

## Top Candidate CSV

1. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5366_public.csv` | OOF=0.820430 | positive_rate=0.536591 | true=2295
2. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5352_umang.csv` | OOF=0.820430 | positive_rate=0.535188 | true=2289
3. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5324_anchor.csv` | OOF=0.820430 | positive_rate=0.532383 | true=2277
4. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate517_a6.csv` | OOF=0.820430 | positive_rate=0.516717 | true=2210
5. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_oofbest_t0p510.csv` | OOF=0.820430 | positive_rate=0.514379 | true=2200
6. `submission_blend_drop_cabinnum_w40_team_A7_blend_w60_rate5127_a7.csv` | OOF=0.820430 | positive_rate=0.512743 | true=2193
7. `submission_team_A7_blend_rate5366_public.csv` | OOF=0.819970 | positive_rate=0.536591 | true=2295
8. `submission_team_A7_blend_rate5352_umang.csv` | OOF=0.819970 | positive_rate=0.535188 | true=2289

## Files

- `reports/feature_ablation_matrix.csv`
- `reports/feature_importance_stability.csv`
- `submissions/submission_manifest_top8.csv`
- `archive_inputs/source_audit.json`

Interpretation rule: feature groups are judged by OOF change first, then by candidate positive-rate risk. Public leaderboard confirmation is still required before claiming final superiority.
