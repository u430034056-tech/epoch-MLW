## EDA notes (one sentence per figure; based on `total_data = train + test`)

- `age_distribution_combined.png`: Age histogram (total_data): shows concentration, tails/outliers, and informs binning or scaling.
- `age_vs_target_box_train_subset.png`: Age vs Transported boxplot (train subset): compares age distributions across classes to assess separability.
- `cryosleep_distribution_combined.png`: CryoSleep frequency distribution (total_data): checks imbalance and missing/rare categories.
- `cryosleep_vs_target_train_subset.png`: CryoSleep vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
- `deck_distribution_combined.png`: Deck frequency distribution (total_data): checks imbalance and missing/rare categories.
- `deck_vs_target_train_subset.png`: Deck vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
- `destination_distribution_combined.png`: Destination frequency distribution (total_data): checks imbalance and missing/rare categories.
- `destination_vs_target_train_subset.png`: Destination vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
- `homeplanet_distribution_combined.png`: HomePlanet frequency distribution (total_data): checks imbalance and missing/rare categories.
- `homeplanet_vs_target_train_subset.png`: HomePlanet vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
- `logspend_vs_target_box_train_subset.png`: log1p(TotalSpend) vs Transported boxplot (train subset): checks whether overall spend correlates with the target.
- `missingness_rate_combined.png`: Missing-rate bar chart by feature: quickly identifies the most-missing fields to guide imputation/encoding.
- `side_distribution_combined.png`: Side frequency distribution (total_data): checks imbalance and missing/rare categories.
- `side_vs_target_train_subset.png`: Side vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
- `spend_distributions_logcount_combined.png`: Faceted histograms for the 5 spend features (total_data; log-scaled counts): reveals zero-inflation and long tails for feature engineering (e.g., log1p, zero-spend flags).
- `target_distribution_total_data.png`: Transported counts on total_data (including test as Missing): verifies the train+test merge and overall label mix.
- `vip_distribution_combined.png`: VIP frequency distribution (total_data): checks imbalance and missing/rare categories.
- `vip_vs_target_train_subset.png`: VIP vs Transported row-normalized stacked bars (train subset): compares the Transported=True proportion across categories.
