# LightGBM Top-Feature Reproduction

Source script: `/Users/shenyijie/Desktop/MLWP project/group3/sources/lightgbm_submission/model_lightgbm1_feature_ablation.py`

| Variant | CV @ 0.5 | Tuned CV | Threshold | Final trees | Submission valid |
|---|---:|---:|---:|---:|---|
| Top 20 | 0.8170 | 0.8187 | 0.460 | 391 | True |
| Top 30 | 0.8177 | 0.8177 | 0.500 | 350 | True |
| Top 40 | 0.8176 | 0.8182 | 0.485 | 392 | True |

## Submission Validation

[
  {
    "path": "/Users/shenyijie/Desktop/MLWP project/group3/reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top20_features.csv",
    "shape": [
      4277,
      2
    ],
    "columns": [
      "PassengerId",
      "Transported"
    ],
    "id_order_match": true,
    "transported_boolean_like": true,
    "true_count": 2311,
    "positive_rate": 0.5403320084171148
  },
  {
    "path": "/Users/shenyijie/Desktop/MLWP project/group3/reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top30_features.csv",
    "shape": [
      4277,
      2
    ],
    "columns": [
      "PassengerId",
      "Transported"
    ],
    "id_order_match": true,
    "transported_boolean_like": true,
    "true_count": 2170,
    "positive_rate": 0.5073649754500819
  },
  {
    "path": "/Users/shenyijie/Desktop/MLWP project/group3/reproduction/outputs/lightgbm_top30/artifacts/submission_lightgbm1_top40_features.csv",
    "shape": [
      4277,
      2
    ],
    "columns": [
      "PassengerId",
      "Transported"
    ],
    "id_order_match": true,
    "transported_boolean_like": true,
    "true_count": 2236,
    "positive_rate": 0.5227963525835866
  }
]
