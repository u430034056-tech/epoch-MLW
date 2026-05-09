# LightGBM1 根据研究建议的修改记录

生成日期：2026-05-06

## 修改范围

本次修改依据 `docs/spaceship_titanic_lightgbm_research.md` 中的建议执行。2026-05-09 追加了多模型概率融合版本。

涉及文件：

- `model_lightgbm1.py`
- `model_lightgbm1 特征消融.py`
- `model_ensemble.py`

原文件副本已保存到：

- `backups/lightgbm1_research_20260506_142603/model_lightgbm1.py`
- `backups/lightgbm1_research_20260506_142603/model_lightgbm1 特征消融.py`

## 核心修改

### 1. 增加严格消费缺失与零消费特征

新增 LightGBM 研究特征列表 `LIGHTGBM_RESEARCH_NUMERIC_FEATURES`，并加入模型输入。

新增特征包括：

- `SpendMissingCount`：五个消费字段中缺失字段数量。
- `AllSpendKnown`：五个消费字段是否全部非缺失。
- `KnownSpendTotal`：仅基于已知消费字段计算的消费总额，缺失按 0 参与求和。
- `HasKnownPositiveSpend`：是否存在任一已知正消费。
- `IsZeroSpendStrict`：只有在所有消费字段均已知且总消费为 0 时才标记为 1。

修改目的：

- 避免 `pandas.sum(skipna=True)` 将缺失消费误当作真实 0 消费。
- 区分“严格 0 消费”和“缺失导致的近似 0 消费”。

### 2. 增加 CryoSleep 与消费规则特征

新增函数 `_add_raw_spend_rule_features`，在消费字段被 `apply_cryosleep_spend_rule` 清零之前捕捉原始规则信息。

新增特征：

- `CryoSleepSpendConflict`：`CryoSleep=True` 且已有任一已知正消费。
- `CryoSleepNoSpending`：`CryoSleep=True` 且填充后的 `IsZeroSpend=1`。
- `CryoSleepStrictZeroSpend`：`CryoSleep=True` 且严格 0 消费。
- `CryoSleepSpendCount`：`CryoSleep=True` 与 `SpendCount` 的交互。

修改目的：

- 显式强化 SpaceShip Titanic 中最关键的业务规则：冷冻睡眠乘客理论上不应消费。
- 保留规则冲突行，避免在清零后丢失异常信号。

### 3. 增加 group / family 特征

新增函数：

- `_add_group_surname_features_single_split`
- `_add_group_surname_features`

新增特征：

- `GroupSurname`：`GroupID + "_" + Surname` 的组合类别特征。
- `GroupSurnameSize`：同一 `GroupSurname` 的人数。

修改目的：

- 增强同行组和家庭信息。
- 与 `GroupSize` 的作用域保持一致：
  - CV 阶段默认使用 `split_local`，避免交叉验证中跨验证折使用结构信息。
  - 最终训练默认使用 `combined`，利用公开 test 中的 `PassengerId` 结构信息。

### 4. 扩展 LightGBM 输入列

在 `_build_lightgbm_feature_set` 中，将新增研究特征加入 LightGBM 输入：

- 数值特征加入 `feature_set["numeric_features"]`
- `GroupSurname` 加入 `feature_set["categorical_features"]`

LightGBM 仍使用原生 categorical 处理方式，没有改为 one-hot，也没有加入模型融合。

### 5. 修改普通 LightGBM 脚本入口

`model_lightgbm1.py` 的命令行入口从：

```python
result = train_model(use_preprocessed_bundle=True)
```

改为：

```python
result = train_model(use_preprocessed_bundle=False)
```

修改目的：

- 默认从 raw CSV 重新构造 fold-local 预处理流程。
- 避免继续读取旧的 `processed/lightgbm/preprocessed_lightgbm.joblib`，导致新增特征不生效。

### 6. 修改特征消融脚本的训练方式

`model_lightgbm1 特征消融.py` 原先的 `train_top_feature_model` 会强制使用 saved bundle。

本次修改后：

- 默认使用 raw rebuild。
- 第一轮训练使用完整新特征空间计算 feature importance。
- 第二轮 top-feature 重训支持在 raw rebuild 下传入 `selected_features`。
- 仍保留 `use_preprocessed_bundle=True` 和 `preprocessed_bundle_path` 作为兼容入口。

相关修改包括：

- `_build_lightgbm_bundle_from_raw` 增加 `selected_features` 参数。
- `_cross_validate_from_raw` 增加 `selected_features` 参数。
- `tune_model_params` 增加 `selected_features` 参数。
- `train_model` 允许 `selected_features` 在 raw rebuild 模式下使用。
- `train_top_feature_model` 不再默认强制 `use_preprocessed_bundle=True`。

## 验证结果

已执行静态编译检查：

```powershell
python -m py_compile "E:\study\ML workshop\PROJECT-kevinhe\model_lightgbm1.py"
python -m py_compile "E:\study\ML workshop\PROJECT-kevinhe\model_lightgbm1 特征消融.py"
```

结果：两份脚本均通过。

已执行普通 LightGBM 小参数冒烟训练：

- `bundle_source=raw_rebuild`
- 新增研究特征全部进入 LightGBM 特征空间
- `missing_research_features=[]`

已执行特征消融二段流程小参数冒烟训练：

- `training_mode=top_feature_selection_plus_retrain`
- `bundle_source=raw_rebuild`
- top-feature 选择和重训流程可正常运行

说明：

- 以上验证是小参数冒烟测试，用于确认代码路径和新特征可用。
- 尚未执行完整默认参数的正式 5 折训练跑分。

## 追加：多模型 Ensemble

### 新增模型

新增独立脚本 `model_ensemble.py`，在保留原 LightGBM 主模型的基础上，加入 `ExtraTreesClassifier` 做概率融合。

说明：

- 当前环境未安装 `catboost` 和 `xgboost`，因此没有把 CatBoost / XGBoost 纳入本次可运行融合。
- `HistGradientBoostingClassifier` 和 `LogisticRegression` 曾参与候选评估，但正式 5 折权重搜索中权重为 0，已从默认融合脚本中移除。
- LightGBM 继续使用原生 categorical。
- sklearn 模型使用内部 encoder 转换 LightGBM 特征表。
- `GroupSurname` 高基数类别只保留给 LightGBM 使用；sklearn 模型会排除该类别列，但保留 `GroupSurnameSize` 数值特征。

### 融合方式

`model_ensemble.py` 使用 5 折 `StratifiedKFold` 生成各基模型 OOF probability。

融合权重通过 OOF 搜索得到：

- 权重非负。
- 权重和为 1。
- 默认搜索步长为 `0.05`。
- 搜索目标优先最大化 OOF tuned accuracy，平局时参考 logloss。

最终使用全量 train 重新训练各基模型，并对 test probability 做加权平均。

### 正式 5 折 Evaluation

运行命令：

```powershell
python "E:\study\ML workshop\PROJECT-kevinhe\model_ensemble.py" --weight-step 0.05 --output-model artifacts/ensemble_model.joblib --output-submission artifacts/submission_ensemble.csv --output-report artifacts/ensemble_report.json
```

输出文件：

- `artifacts/ensemble_model.joblib`
- `artifacts/submission_ensemble.csv`
- `artifacts/ensemble_report.json`

5 折 OOF 结果：

| 模型 | Accuracy @ 0.5 | Tuned Accuracy | Best Threshold | Logloss |
|---|---:|---:|---:|---:|
| LightGBM | 0.816634 | 0.817899 | 0.470 | 0.375606 |
| ExtraTrees | 0.799264 | 0.802255 | 0.450 | 0.414777 |
| Ensemble | 0.817439 | 0.819510 | 0.460 | 0.376052 |

最佳融合权重：

```python
{
    "lightgbm": 0.85,
    "extra_trees": 0.15,
}
```

相对当前 LightGBM 基线：

- Tuned Accuracy：`0.817899 -> 0.819510`
- 绝对提升：`+0.001611`
- 约等于 8693 条训练样本 OOF 中多分对约 14 条样本。
- Logloss：`0.375606 -> 0.376052`，略有变差，说明该融合主要提升阈值后 accuracy，不是提升概率校准。

## 未做事项

根据本次要求，以下内容暂未修改：

- 未加入 CatBoost / XGBoost，因为当前环境未安装对应依赖。
- 未做 LightGBM + CatBoost + XGBoost 融合。
- 未引入 target encoding。
- 未对 `preprocess.py` 的共享全模型预处理流程做全局改造。
- 未重新生成 `processed/` 目录中的 saved bundle。
