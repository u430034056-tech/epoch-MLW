# Spaceship Titanic Preprocessing

This stage only builds preprocessing artifacts and project structure. It reads raw data, summarizes schema, performs shared cleaning and feature engineering, prepares seven model-specific preprocessing outputs, and saves reusable bundles under `processed/`.

This stage intentionally does not train models, run cross-validation, tune hyperparameters, evaluate metrics, save trained models, generate predictions, or write a submission file.

## Main Responsibilities

`preprocess.py` contains the shared preprocessing pipeline and the seven model branches.

Key functions:
- `get_project_paths`: resolve dataset and output paths.
- `load_raw_data`: load `train.csv` and `test.csv`.
- `inspect_raw_data` and `build_data_summary`: summarize columns, dtypes, missing counts, and train/test differences.
- `basic_cleaning` and `enforce_dtypes`: normalize raw inputs.
- `split_passenger_id_features`, `build_group_features_with_combined_ids`: derive `GroupID`, `GroupMemberNo`, and `GroupSize`.
- `split_cabin_features`: derive `Deck`, `CabinNum`, and `Side`.
- `extract_name_features`: derive `Surname`.
- `create_missing_indicators`: create missing-value flags before filling.
- `fit_common_statistics`: fit train-only statistics such as age median, spend medians, `SurnameFreq`, and `CabinNumBin` edges.
- `apply_common_statistics`: apply train-fit statistics to train and test.
- `create_spend_features`, `create_age_features`, `create_spend_structure_features`, `create_group_structure_features`, `create_interaction_categorical_features`: build shared engineered features.
- `finalize_common_frame`: close remaining raw-audit-column missing values and normalize final common dtypes.
- `build_preprocessing_quality_report`: produce a lightweight summary of missingness, identifier checks, cardinality, and train-fit assumptions.
- `get_feature_sets_for_*`: separate shared feature construction from model-specific feature selection.
- `preprocess_for_*`: build model-specific preprocessing outputs.
- `save_preprocessed_bundle` and `load_preprocessed_bundle`: persist and reload artifacts.
- `run_all_preprocessing`: run the entire preprocessing-only workflow.

## Shared Feature Rules

- `Transported` is sourced from `train.csv` only and cast to integer `0/1`.
- `SurnameFreq` is a frequency-style global statistic, but the default implementation strictly fits it on train surnames only, then applies that mapping to both train and test. Unseen test surnames map to `0`.
- `CabinNumBin` is fit on non-missing train `CabinNum` values only. Raw missing `CabinNum` values are always assigned to `CabinBin_Missing`.
- `Age` is coerced to numeric before any age-quality checks. Values outside `[0, 100]` are flagged by `AgeWasOutOfRange`, reset to missing, and then filled with the train-fit age median.
- `AgeGroup` uses fixed semantic bins plus an explicit `Unknown` bucket. It no longer uses `Adult` as an exception fallback for abnormal values.
- `GroupSize` currently uses `combined_train_test_full_preprocessing` as an engineering choice for full preprocessing.
- If future cross-validation is added, `SurnameFreq`, `CabinNumBin` edges, age median after out-of-range filtering, spend medians, hierarchical spend medians, and `GroupSize` must all be recomputed inside each fold.
- Future CV or strict validation should replace the combined GroupSize builder with a split-local or fold-local helper such as `build_group_features_single_split(...)`.

## Shared Engineered Features

The shared common table now includes:
- Structural features: `GroupID`, `GroupMemberNo`, `GroupSize`, `Deck`, `CabinNum`, `Side`, `Surname`
- Missing indicators: all columns from `MISSING_INDICATORS`
- Age quality indicator: `AgeWasOutOfRange`
- Spend totals: `TotalSpend`, `IsZeroSpend`, `SpendCount`
- Age features: `IsChild`, `IsSenior`, `AgeGroup`
- Spend structure features: `LuxurySpend`, `BasicSpend`, `LuxuryShare`, `SpendPerActiveCategory`, `HasAnyLuxurySpend`
- Group structure features: `IsSolo`, `IsMultiPassengerGroup`, `GroupMemberIsLeader`
- Interaction categorical features: `DeckSide`, `HomePlanetDestination`
- Train-fit features: `SurnameFreq`, `CabinNumBin`

The common layer also keeps raw audit columns such as `PassengerId`, `GroupID`, `Name`, `Cabin`, and `Surname`. These columns are retained for traceability and inspection, but they are not automatically fed into default model feature sets. The main exception is CatBoost, which still keeps `Surname` as an explicit categorical input.

## Model-Specific Outputs

- Logistic Regression: standardized numeric features, `log1p` spend-related features, and one-hot categorical features.
- Random Forest: cleaned numeric features and one-hot categorical features without scaling.
- HistGradientBoosting (`hist_gradient_boosting`): dense numeric table with ordinal-encoded categorical features.
- XGBoost: cleaned numeric features, `log1p` spend-related features, and one-hot categorical features without scaling.
- LightGBM: pandas `DataFrame` with native categorical columns kept as `category`, with train/test category levels explicitly aligned at preprocessing time.
- CatBoost: pandas `DataFrame` with categorical columns preserved as strings, including `Surname`.
- KNN: standardized numeric features, one-hot categorical features, dense numeric matrix output, and a compact feature subset designed for distance-based modeling.

`PassengerId`, `GroupID`, `Name`, and `Cabin` are not sent directly into default model branches because they are audit or high-cardinality fields. `Surname` is also excluded from every default branch except CatBoost. KNN additionally excludes the higher-cardinality interaction column `HomePlanetDestination` to avoid unnecessary one-hot dimensional growth.

## Why `get_feature_sets_for_*` Exists

The seven `get_feature_sets_for_*` functions decouple shared feature construction from model-specific column selection. This keeps the common pipeline reusable, makes later model scripts simpler, and allows feature-set changes without breaking the shared preprocessing flow.

## Why Bundles Keep Generic And Suffixed Keys

Each model bundle keeps generic keys like `X_train`, `X_test`, and `y_train` for consistent downstream code, and model-suffixed aliases like `X_train_lr`, `X_train_knn`, or `X_train_xgb` to make saved artifacts explicit and easier to inspect.

## Processed Outputs

The `processed/` directory contains:
- `processed/common/`: shared feature tables, raw data summary JSON, `quality_report.json`, metadata, and previews.
- `processed/logistic_regression/`
- `processed/random_forest/`
- `processed/hist_gradient_boosting/`
- `processed/xgboost/`
- `processed/lightgbm/`
- `processed/catboost/`
- `processed/knn/`

Each model directory stores a `preprocessed_<model>.joblib` bundle plus a metadata JSON sidecar. DataFrame-based outputs also include preview CSV files.

## Quality Report

`processed/common/quality_report.json` remains lightweight and JSON-safe. It now includes:
- missingness summaries for raw and finalized common tables
- identifier uniqueness and ID-alignment checks
- `common_categorical_unique_counts`, where each count is based on train/test combined deduplicated values rather than a simple train-count plus test-count sum
- `common_categorical_cardinality_summary`, which breaks each shared categorical column into `train_unique_count`, `test_unique_count`, and `combined_unique_count`
- `common_shape_summary`, `model_input_overview`, `age_quality_summary`, `cabin_bin_summary`, and the current `group_feature_mode`

`model_input_overview` is filled after all seven model bundles are built, so the saved `quality_report.json` and `metadata_common.json` contain final `X_train` and `X_test` shapes for every branch.

## `main.py` Usage

`main.py` only orchestrates preprocessing. It can:
- import `preprocess.py`
- run preprocessing when saved bundles do not exist
- reload existing bundles
- print available result keys and bundle paths for all seven model branches

It does not train any model.

Dataset path resolution:

- Preferred current path: `data/raw/train.csv` and `data/raw/test.csv`
- Legacy compatible path: `spaceship-titanic/train.csv` and `spaceship-titanic/test.csv`

The loader uses the legacy path only when `spaceship-titanic/train.csv` exists; otherwise it falls back to `data/raw/`.

## Model Training Boundary

This cleaned GitHub submission is centered on the preprocessing pipeline and final Kaggle submission artifact. `main.py` does not import or execute the model-training scripts; it only builds and reloads preprocessing bundles.

If future training code is integrated into this mainline, it should:

- call `load_preprocessed_data(...)`
- consume the bundle for the relevant model
- keep helper modules such as inference, run orchestration, and self-training utilities in the repository instead of relying on local-only files

If cross-validation is introduced later, recompute all train-fit preprocessing statistics inside each fold instead of reusing full-train artifacts.
