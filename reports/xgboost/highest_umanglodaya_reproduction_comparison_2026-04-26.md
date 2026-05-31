# Highest XGB Reproduction and Preprocessing Comparison - 2026-04-26

## 1. Reproduced Version

Highest public-score file:

- `submission_umanglodaya_xgb_smote_seed2024.csv`
- Public score recorded in the submit queue: `0.81412`
- Archived exact file:
  - `reports/xgboost/submission_candidates/99_archive_non_team_raw_2026-04-25/submission_umanglodaya_xgb_smote_seed2024.csv`

Rerun command:

```bash
PYTHONPATH=01_本地副本_实验 .venv/bin/python -m src.xgboost.run_umanglodaya_xgb --seeds 2024
```

Rerun output:

- New file:
  - `reports/xgboost/submission_candidates/submission_umanglodaya_xgb_smote_seed2024.csv`
- Raw data directory:
  - `/Users/shenyijie/Desktop/20260319_xgboost 2/data/raw`
- Submission rows: `4277`
- Columns: `PassengerId, Transported`
- True count: `2289`
- Positive rate: `0.5351882160392799`
- Prediction difference versus archived 0.81412 file: `0`

Conclusion: this rerun exactly reproduces the archived highest-score submission.

## 2. What This Highest Version Does

Source file:

- `src/xgboost/run_umanglodaya_xgb.py`

This is not based on the team unified preprocessing output. It starts from raw Kaggle CSV files and rebuilds a separate Kaggle-notebook-style XGB table.

### Data Preprocessing

1. Concatenate raw train and test.

   - Train keeps `Transported`.
   - Test is appended with `Transported = NaN`.
   - All later imputation and one-hot encoding are fit on this combined table.

2. Apply CryoSleep spend rule.

   - If `CryoSleep == True`, set all five spending columns to `0`.
   - Create `Expenses = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`.
   - If `Expenses == 0` and `CryoSleep` is missing, infer `CryoSleep = True`.

3. Use passenger group as a fill guide.

   - `Room = PassengerId[:4]`.
   - For each of `Cabin`, `VIP`, `HomePlanet`, `Destination`, build a per-room guide from non-missing rows.
   - Fill missing values from rows with the same `Room`.

4. Split cabin.

   - `Cabin_1`: deck.
   - `Cabin_2`: cabin number, created but not used in the final model.
   - `Cabin_3`: side.

5. Simple imputation.

   - Numeric columns use global mean imputation on combined train+test.
   - Categorical columns use global most-frequent imputation on combined train+test.

6. One-hot encode categorical columns.

   - One-hot is fit on combined train+test.

### Feature Engineering and Final Feature Set

Initial numeric columns:

- `ShoppingMall`
- `FoodCourt`
- `RoomService`
- `Spa`
- `VRDeck`
- `Expenses`
- `Age`

Initial categorical columns:

- `CryoSleep`
- `Cabin_1`
- `Cabin_3`
- `VIP`
- `HomePlanet`
- `Destination`

Permutation-importance drop list:

- `ShoppingMall`
- `Age`
- `CryoSleep_True`
- `HomePlanet_Earth`
- `HomePlanet_Europa`
- `VIP_True`
- `HomePlanet_Mars`
- `Destination_PSO J318.5-22`
- `VIP_False`
- `Destination_55 Cancri e`
- `FoodCourt`
- `Destination_TRAPPIST-1e`

Final model matrix:

- Train shape: `8693 x 15`
- Test shape: `4277 x 15`

Final 15 features:

- `RoomService`
- `Spa`
- `VRDeck`
- `Expenses`
- `CryoSleep_False`
- `Cabin_1_A`
- `Cabin_1_B`
- `Cabin_1_C`
- `Cabin_1_D`
- `Cabin_1_E`
- `Cabin_1_F`
- `Cabin_1_G`
- `Cabin_1_T`
- `Cabin_3_P`
- `Cabin_3_S`

### Model Side

XGBoost parameters:

- `reg_lambda = 3.0610042624477543`
- `reg_alpha = 4.581902571574289`
- `colsample_bytree = 0.9241969052729379`
- `subsample = 0.9527591724824661`
- `learning_rate = 0.06672065863100594`
- `n_estimators = 730`
- `max_depth = 5`
- `min_child_weight = 1`
- `num_parallel_tree = 1`
- `objective = binary:logistic`
- `eval_metric = logloss`
- `tree_method = hist`

SMOTE:

- `sampling_strategy = 1`
- `random_state = 2024`
- single-seed final submission threshold: `0.5`

## 3. What The Team Unified Preprocessing Does

Team source:

- `00_GitHub主线_已提交/epoch-MLW-main/preprocess.py`
- `00_GitHub主线_已提交/epoch-MLW-main/README_preprocessing.md`
- Output used by team branches:
  - `processed/common/preprocessed_common.joblib`
  - `processed/xgboost/preprocessed_xgboost.joblib`

Team processed shapes:

- Common train: `8693 x 51`
- Common test: `4277 x 51`
- XGBoost train: `8693 x 107`
- XGBoost test: `4277 x 107`

Team common features include:

- Raw audit columns:
  - `PassengerId`, `Cabin`, `Name`, `GroupID`, `Surname`
- Group/cabin structure:
  - `GroupMemberNo`, `GroupSize`, `Deck`, `CabinNum`, `Side`
- Missing indicators:
  - `AgeMissing`, `HomePlanetMissing`, `CryoSleepMissing`, `CabinMissing`, `DestinationMissing`, `VIPMissing`, spend-column missing flags, `NameMissing`
- Train-fit features:
  - `SurnameFreq`
  - `CabinNumBin`
- Spend features:
  - `TotalSpend`
  - `IsZeroSpend`
  - `SpendCount`
  - `LuxurySpend`
  - `BasicSpend`
  - `LuxuryShare`
  - `SpendPerActiveCategory`
  - `HasAnyLuxurySpend`
- Age features:
  - `AgeWasOutOfRange`
  - `IsChild`
  - `IsSenior`
  - `AgeGroup`
- Group structure:
  - `IsSolo`
  - `IsMultiPassengerGroup`
  - `GroupMemberIsLeader`
- Interactions:
  - `DeckSide`
  - `HomePlanetDestination`

Important team preprocessing rules:

- `Transported` is sourced only from `train.csv`.
- `CryoSleep` and `VIP` are normalized robustly to string-like True/False values.
- `HomePlanet`, `VIP`, and `Destination` use within-split group-consistent filling when a `GroupID` has exactly one observed value.
- Missing `CryoSleep` is inferred by stricter spend-based rules before spend imputation.
- Spend columns use train-fit hierarchical median imputation.
- `SurnameFreq` is fit on train surnames only; unseen test surnames map to `0`.
- `CabinNumBin` is fit on non-missing train `CabinNum` values only.
- `Age` out-of-range values are flagged, reset to missing, then filled with train-fit median.
- XGBoost branch one-hot encodes the team common feature set into 107 columns.

## 4. Direct Comparison

| Area | Highest 0.81412 version | Team unified preprocessing |
| --- | --- | --- |
| Data source | Raw `train.csv` + `test.csv` directly | Saved `processed/common` and `processed/xgboost` bundles |
| Train/test handling | Concatenates train and test before imputation and one-hot | Builds common shared outputs and model-specific bundles |
| Final XGB feature count | 15 | 107 |
| Missing numeric fill | Simple global mean on combined train+test | Train-fit hierarchical median and global fallback |
| Missing categorical fill | Simple global mode on combined train+test, plus room guide | Group-consistent fill within split, then robust defaults |
| Passenger group logic | `Room = PassengerId[:4]`; fills `Cabin`, `VIP`, `HomePlanet`, `Destination` | `GroupID`, `GroupMemberNo`, `GroupSize`; fills selected categoricals but keeps more audit structure |
| Cabin logic | Uses only deck and side after splitting; cabin number unused | Uses `Deck`, `CabinNum`, `Side`, `CabinNumBin`, `DeckSide` |
| Name/surname | Does not use surname | Extracts `Surname`, builds `SurnameFreq`, keeps raw surname for audit/CatBoost |
| Spend features | Only original spend subset plus `Expenses`; drops `FoodCourt` and `ShoppingMall` later | Rich spend structure: `TotalSpend`, count, luxury/basic split, ratios, active-category spend |
| Age features | Starts with `Age`, then drops it | Keeps `Age`, missing flag, out-of-range flag, child/senior flags, age group |
| Categorical features | Keeps only `CryoSleep_False`, cabin deck, cabin side after drop list | Keeps HomePlanet, Destination, VIP, Deck, Side, AgeGroup, CabinNumBin, DeckSide, HomePlanetDestination |
| SMOTE | Yes, final model uses SMOTE with seed `2024` | Team preprocessing itself does not do SMOTE; SMOTE only appears in later XGB experiments |
| Leaderboard behavior | Highest observed public score `0.81412` | Team-only translated attempts in the latest screenshot scored around `0.80804-0.80827` |

## 5. Why The Team-Based Translation Did Not Reach 0.81412

The latest team-based candidates copied model-side ideas from this high-score line: XGB parameters, drop-list spirit, SMOTE, seeds, and A7 blending. But they did not copy the raw-template preprocessing itself.

The public-score gap likely comes from the preprocessing differences, especially:

1. The 0.81412 version uses a very compact 15-feature table dominated by spend, CryoSleep, deck, and side.
2. It fills values after concatenating train and test, which changes imputation and one-hot category behavior.
3. It fills `Cabin` from same passenger group, while the team common preprocessing keeps a more conservative and auditable cabin handling path.
4. It drops many broad categorical one-hot signals that the team XGB 107-feature bundle still contains.
5. It is highly seed-sensitive: `seed2024` scored `0.81412`, while other nearby seeds scored lower.

## 6. Practical Interpretation

For Kaggle scoring, the 0.81412 file is the best public-LB artifact we currently have.

For project/report consistency, it should be described as:

> A separate Kaggle-style XGBoost reproduction branch based on raw CSV preprocessing, not the team unified preprocessing output.

For team-rule-compliant future work, the useful takeaway is not to submit this raw branch as if it were team preprocessing. The useful takeaway is:

- keep the team common output as the base;
- build an XGB-specific derived view that mimics the compact 15-feature idea;
- test cabin/group fill adjustments only as small, documented XGB-local changes;
- keep every derived bundle separate from the original team bundle.
