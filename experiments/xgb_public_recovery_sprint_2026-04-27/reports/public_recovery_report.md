# XGB Public Recovery Sprint - 2026-04-27

Scope: team preprocessing only. The feature matrix is derived from `processed/common/preprocessed_common.joblib`; raw Kaggle CSVs and archived raw-template probabilities are not used as model input.

## Why This Sprint Exists

- The just-submitted `drop_cabinnum + A7` candidate scored `0.80430` public, so that OOF gain is treated as leaderboard overfit.
- This sprint goes back toward the public-proven high-score structure: compact spend/CryoSleep/cabin/home/destination features, SMOTE, and high-score XGB parameters.

## Top 8 Upload Order

1. `submission_team_common_rawlike_raw_core_k5_seed2024_rate5366_public.csv` | view=raw_core | seed=2024 | k=5 | pos=0.536591 | OOF=0.808812
2. `submission_team_common_rawlike_raw_core_k5_seed2024_rate53566_seed7.csv` | view=raw_core | seed=2024 | k=5 | pos=0.535656 | OOF=0.808812
3. `submission_team_common_rawlike_raw_core_k5_seed2024_rate53519_seed2024.csv` | view=raw_core | seed=2024 | k=5 | pos=0.535188 | OOF=0.808812
4. `submission_team_common_rawlike_raw_core_k5_seed2024_t050.csv` | view=raw_core | seed=2024 | k=5 | pos=0.534954 | OOF=0.808812
5. `submission_team_common_rawlike_raw_core_k5_seed2024_rate53495_seed17.csv` | view=raw_core | seed=2024 | k=5 | pos=0.534954 | OOF=0.808812
6. `submission_team_common_rawlike_raw_core_k5_seed2024_rate5324_anchor.csv` | view=raw_core | seed=2024 | k=5 | pos=0.532383 | OOF=0.808812
7. `submission_team_common_rawlike_raw_plus_group_k3_seed2024_rate5366_public.csv` | view=raw_plus_group | seed=2024 | k=3 | pos=0.536591 | OOF=0.808467
8. `submission_team_common_rawlike_raw_plus_group_k3_seed2024_rate53566_seed7.csv` | view=raw_plus_group | seed=2024 | k=3 | pos=0.535656 | OOF=0.808467

## Validation

- Active CSV count: `8`
- Invalid CSV count: `0`

## Files

- `submissions/submission_manifest_public_recovery.csv`
- `reports/public_recovery_manifest_all.csv`
- `reports/submission_validation.csv`
- `archive_inputs/source_audit.json`
