# SpaceShip Titanic 高分 LightGBM 方案分析与优化建议

生成日期：2026-05-06

> 说明：本文整理自公开可访问的 Kaggle 竞赛信息、公开博客/Notebook 复盘和公开方案说明。Kaggle 部分 Notebook 页面可能受登录、反爬或页面权限限制影响，无法稳定直接读取完整代码；因此本文只记录已实际可公开访问到的信息，不把未确认内容当作事实。

## 1. 结论摘要

- SpaceShip Titanic 的高分方案并不只靠模型参数，真正的分数来源主要是 `Cabin`、`PassengerId`、消费字段和 `CryoSleep` 规则这几类强特征。
- `Cabin` 拆分为 `Deck`、`Cabin_num`、`Side` 几乎是必做项，因为它捕捉了飞船区域、舱位编号和左右舷结构信息。
- `PassengerId` 拆分出的同行组 `Group`、`GroupSize`、是否独行 `Solo` 通常很有价值，因为同组乘客的目标变量存在相关性。
- 消费字段建议同时保留原始消费、`TotalSpent`、`NoSpending`、`SpendCount`、`LuxuryShare` 等聚合特征。
- `CryoSleep` 与消费逻辑是关键规则：处于冷冻睡眠的人理论上消费应为 0，因此它既可用于缺失填充，也可用于构造强交互特征。
- 公开方案中常见稳健验证方式是 `StratifiedKFold` 或带交叉验证的参数搜索，不建议只看一次 train/validation split 或 public leaderboard。
- LightGBM 单模可以达到不错水平，但公开方案常把 LightGBM 与 CatBoost、XGBoost 或其他树模型做 averaging / voting 来进一步稳定分数。
- 对你当前项目，最优先的改进方向应是：固定一套无泄漏预处理流程、做 5 折 OOF 验证、强化 `CryoSleep` 消费规则、再考虑 LightGBM + CatBoost/XGBoost 融合。

## 2. 高分 LightGBM 方案对比表

| 方案 | 来源 | 模型 | 核心特征工程 | 验证策略 | 亮点 | 风险 |
|---|---|---|---|---|---|---|
| Daniel J Smith SpaceShip Titanic | [blog.danieljsmith.org](https://blog.danieljsmith.org/posts/23_11_23_SpaceshipTitanic/index.html) | LightGBM | `TotalExpenditure`、`AgeGroup`、`GroupSize`、`Solo`、`CabinSide`、`CryoSleep` 消费规则 | `train_test_split` + `RandomizedSearchCV(cv=5)` | 特征工程清晰，LightGBM 参数完整 | 单次 holdout 仍可能有波动 |
| Fernandao Lacerda Dantas 0.8066 Solution | [Medium](https://medium.com/%40fernandao.lacerda.dantas/space-titanic-kaggle-competition-0-8066-score-solution-7a9c401281c6) | LightGBM / XGBoost 对比 | `Cabin` 拆分、KNNImputer、TargetEncoder、类别/数值管道 | `StratifiedKFold(5)` + `GridSearchCV` | 验证更稳健，预处理管道规范 | Target encoding 如果做法不严谨会有泄漏风险 |
| Maria Aguilera Spaceship Titanic Project | [maria-aguilera.github.io](https://maria-aguilera.github.io/projects/spaceship-titanic.html) | 多模型，包括 LightGBM | 常规清洗、OHE、Scaler、多模型对比 | stratified split + CV / GridSearch | 适合参考模型比较和融合思路 | 不是纯 LightGBM 方案，细节需结合代码确认 |
| CSDN LightGBM + Optuna | [CSDN](https://blog.csdn.net/2302_79308082/article/details/144576896) | LightGBM + Optuna | 消费 `log1p`、`PassengerId` team、`total_fee`、`Cabin` 拆分、OHE | train/validation split + Optuna | 调参方向具体，参数搜索空间可复用 | 只用单次切分时验证稳定性不足 |

## 3. 高分方案详细分析

### 方案 1：Daniel J Smith SpaceShip Titanic

来源链接：[https://blog.danieljsmith.org/posts/23_11_23_SpaceshipTitanic/index.html](https://blog.danieljsmith.org/posts/23_11_23_SpaceshipTitanic/index.html)

#### 基本信息

- 模型：`LGBMClassifier`
- 大致 public score：页面显示约 `0.801`
- 是否主要使用 LightGBM：是
- 是否使用 ensemble：该页面重点是 LightGBM 单模

#### 数据预处理

- 对 train/test 做一致的特征构造。
- 从 `PassengerId` 提取组信息。
- 对 `Cabin` 做拆分。
- 对消费字段构造聚合特征。
- 对缺失值和类别变量做统一处理。

#### 特征工程

- `TotalExpenditure`：将 `RoomService`、`FoodCourt`、`ShoppingMall`、`Spa`、`VRDeck` 聚合。
- `GroupSize`：基于 `PassengerId` 的组号统计同行人数。
- `Solo`：判断乘客是否独自旅行。
- `CabinSide`：从 `Cabin` 中拆分 side 信息。
- `AgeGroup`：年龄分箱。
- `CryoSleep` 与消费：利用冷冻睡眠乘客消费应为 0 的业务逻辑。

#### LightGBM 参数配置

```python
params = {
    "colsample_bytree": 0.9470125857910588,
    "learning_rate": 0.08878983735014609,
    "max_depth": 4,
    "min_child_samples": 25,
    "n_estimators": 255,
    "num_leaves": 27,
    "random_state": 594,
    "reg_alpha": 0.6589313013800681,
    "reg_lambda": 0.3417673497709479,
    "subsample": 0.7259411789298494,
    "verbose": -1,
}
```

#### 验证方式

- 使用 `train_test_split` 构造验证集。
- 使用 `RandomizedSearchCV` 和 `cv=5` 做参数搜索。
- 这种方式比只看 public leaderboard 更稳，但如果最终验证只依赖一次 holdout，仍可能有随机波动。

#### 可借鉴点

- 特征工程优先于盲目调参。
- `GroupSize`、`Solo` 和 `CryoSleep` 消费逻辑值得直接加入你当前流程。
- 参数整体偏中等复杂度，适合作为你的 LightGBM 基线之一。

#### 不建议直接复制的地方

- 不建议只用一次 train/validation split 决定最终模型。
- 不建议把所有缺失填充逻辑手写散落在模型脚本里，最好进入统一 preprocess pipeline。

### 方案 2：Fernandao Lacerda Dantas 0.8066 Solution

来源链接：[https://medium.com/%40fernandao.lacerda.dantas/space-titanic-kaggle-competition-0-8066-score-solution-7a9c401281c6](https://medium.com/%40fernandao.lacerda.dantas/space-titanic-kaggle-competition-0-8066-score-solution-7a9c401281c6)

#### 基本信息

- 模型：LightGBM 与 XGBoost 对比。
- 大致 public score：文章标题显示约 `0.8066`。
- 是否主要使用 LightGBM：包含 LightGBM。
- 是否使用 ensemble：文章重点更偏模型比较与调参，不是必须依赖 ensemble。

#### 数据预处理

- 使用 `KNNImputer` 处理数值缺失。
- 使用 `SimpleImputer` 与 `TargetEncoder` 处理类别变量。
- 对 train/test 做一致转换。
- 使用 pipeline 思路组织特征处理。

#### 特征工程

- `Cabin` 拆分为类似 `cabin_code`、`id_cabin`、`cabin_sector` 的结构。
- 保留并处理类别特征，如 `HomePlanet`、`Destination`、`VIP`。
- 使用更系统的缺失值处理，而不是简单全局众数/中位数。

#### LightGBM 参数配置

文章展示了 GridSearch 的搜索范围：

```python
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "num_leaves": [20, 50, 80, 100],
}
```

#### 验证方式

- 使用 `StratifiedKFold(5)`。
- 使用 `GridSearchCV`。
- 这比单次切分更适合 SpaceShip Titanic，因为目标变量接近平衡但仍需要保持折间分布一致。

#### 可借鉴点

- 你的 evaluation 应优先采用 5 折 `StratifiedKFold` 的 OOF 分数。
- 如果要使用 target encoding，必须保证在每个 fold 内 fit，避免把验证集标签信息泄漏进编码。
- KNNImputer 可以作为进阶实验，但需要和简单填充做对照。

#### 不建议直接复制的地方

- Target encoding 需要谨慎，尤其不能在完整 train 上 fit 后再 CV。
- GridSearch 搜索空间相对粗，建议后续用 Optuna 或 RandomizedSearch 做更细调参。

### 方案 3：Maria Aguilera Spaceship Titanic Project

来源链接：[https://maria-aguilera.github.io/projects/spaceship-titanic.html](https://maria-aguilera.github.io/projects/spaceship-titanic.html)

#### 基本信息

- 模型：多个模型对比，包括 LightGBM、CatBoost、XGBoost 等。
- 大致分数：公开项目中展示了约 `0.8066` 水平的结果。
- 是否主要使用 LightGBM：不是纯 LightGBM，但 LightGBM 是候选模型之一。
- 是否使用 ensemble：该类项目更适合参考模型比较和融合方向。

#### 数据预处理

- 典型流程包括缺失处理、类别编码、数值处理和模型比较。
- 倾向使用更规范的 pipeline 管理不同模型。

#### 特征工程

- 重点可参考其多模型统一处理流程。
- 对 LightGBM、CatBoost、XGBoost 使用相同数据口径做对比，有助于判断问题来自模型还是特征。

#### LightGBM 参数配置

公开信息中可参考的一组 LightGBM 参数方向：

```python
params = {
    "learning_rate": 0.05,
    "n_estimators": 500,
    "reg_lambda": 1,
}
```

#### 验证方式

- 使用 stratified split 和交叉验证/搜索策略。
- 适合作为你后续“模型家族对比”的参考。

#### 可借鉴点

- 在你当前 LightGBM 稳定后，可以用相同预处理结果训练 CatBoost / XGBoost，对比 OOF。
- 如果多个模型错误模式不完全一致，简单平均概率可能比单模更稳。

#### 不建议直接复制的地方

- 不应在特征还不稳定时过早堆很多模型，否则很难定位提升来自哪里。
- 融合前必须保证每个基础模型都有可信 OOF 分数。

### 方案 4：CSDN LightGBM + Optuna

来源链接：[https://blog.csdn.net/2302_79308082/article/details/144576896](https://blog.csdn.net/2302_79308082/article/details/144576896)

#### 基本信息

- 模型：`lgb.LGBMClassifier`
- 调参：Optuna 50 trials
- 是否主要使用 LightGBM：是
- 是否使用 ensemble：公开内容重点是 LightGBM 单模调参

#### 数据预处理

- 对消费字段做 `log1p`。
- 从 `PassengerId` 提取 team/group 信息。
- 从 `Cabin` 拆分 `deck`、`num`、`side`。
- 使用 OneHotEncoder 处理类别变量。

#### 特征工程

- `total_fee`：消费总额。
- `PassengerId` team：同行组特征。
- `Cabin` split：空间结构特征。
- 消费字段 `log1p`：缓解长尾分布。

#### LightGBM 参数配置

```python
params = {
    "max_depth": 5,
    "num_leaves": 64,
    "min_child_samples": 49,
    "min_child_weight": 2.4205967592730935,
    "subsample": 0.8284617968332849,
    "colsample_bytree": 0.815004121704074,
    "learning_rate": 0.052056911035826305,
    "reg_lambda": 0.002421309962401076,
    "reg_alpha": 9.088728193209626,
}
```

#### 验证方式

- 使用 train/validation split。
- 使用 Optuna 优化参数。
- 若要迁移到你的项目，建议把目标函数改成 5 折 `StratifiedKFold` OOF accuracy，而不是单次 split accuracy。

#### 可借鉴点

- 你的 LightGBM 可以加入 Optuna 调参，但先固定特征和 CV。
- 搜索空间可以参考上面的 `max_depth`、`num_leaves`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda`。

#### 不建议直接复制的地方

- 不建议在单次切分上过度调参，容易对该验证集过拟合。
- `num_leaves=64` 配 `max_depth=5` 需要观察是否过拟合，不能只看 public score。

## 4. SpaceShip Titanic 中最重要的特征工程总结

### 1. `Cabin` 拆分

`Cabin` 原始格式通常类似 `Deck/Num/Side`，直接当一个字符串使用会造成高基数和稀疏问题。拆成 `Deck`、`CabinNum`、`Side` 后，模型可以分别学习飞船区域、舱号位置和左右舷差异。

建议实现：

- `Deck = Cabin.split("/")[0]`
- `CabinNum = Cabin.split("/")[1]`
- `Side = Cabin.split("/")[2]`
- `CabinNumBin` 用分箱捕捉位置段
- `DeckSide = Deck + "_" + Side` 捕捉组合区域

### 2. `PassengerId` group 特征

`PassengerId` 形如 `0001_01`，前半部分表示同行组。同行组成员往往共享 `HomePlanet`、`Destination`、`Cabin` 或目标倾向。

建议实现：

- `Group = PassengerId.split("_")[0]`
- `GroupMemberNo = PassengerId.split("_")[1]`
- `GroupSize = Group` 在 train+test 中的计数
- `IsSolo = GroupSize == 1`

注意：`GroupSize` 可以用 train+test 共同统计，因为 test 的 `PassengerId` 本身是公开结构特征，不包含目标标签；但如果用目标均值做 group encoding，就必须严格 fold 内计算。

### 3. 消费总额 `TotalSpent`

五个消费字段是强信号，单独字段和总额都应该保留。

建议实现：

- `TotalSpent = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`
- `SpendCount = 五个消费字段中大于 0 的数量`
- `NoSpending = TotalSpent == 0`
- `LuxurySpend = FoodCourt + Spa + VRDeck`
- `BasicSpend = RoomService + ShoppingMall`
- `LuxuryShare = LuxurySpend / (TotalSpent + 1)`

注意：如果原始消费字段有缺失，`pandas.sum(skipna=True)` 会把部分缺失行算成偏低的总额，甚至误判为 0 消费。严格做法是保留缺失指示，或在预处理后再构造最终模型特征。

### 4. `CryoSleep` 与消费逻辑

业务逻辑上，`CryoSleep=True` 的乘客不应该产生消费。这是 SpaceShip Titanic 的核心规则之一。

建议实现：

- `CryoSleep=True` 且消费缺失时，可考虑将消费填 0。
- `TotalSpent=0` 且多个消费字段均为 0 时，可辅助推断 `CryoSleep=True`。
- 增加交互特征：`CryoSleep_x_NoSpending`、`CryoSleep_x_SpendCount`。
- 对违反规则的行保留异常标记：`CryoSleepSpendConflict`。

注意：不要在不知道原始缺失原因时，把所有含缺失的消费行都直接当作 0 消费。

### 5. 家庭/同行组信息

如果从 `Name` 提取 surname，可以捕捉家庭成员信息。公开方案常尝试 surname frequency 或 group-surname 组合。

建议实现：

- `Surname = Name.split(" ")[-1]`
- `SurnameFreq = Surname` 在 train+test 中的频数
- `GroupSurname = Group + "_" + Surname`

注意：`Name` 本身通常不直接喂给模型，因为高基数且泛化差；surname 频数比原始姓名更稳。

### 6. 缺失值填充策略

缺失不是纯噪声，它本身可能带有模式。建议保留缺失指示。

建议实现：

- 数值列：中位数填充 + `is_missing` 指示。
- 类别列：众数或 `"Unknown"` 填充。
- 布尔列：结合规则填充，如 `CryoSleep` 与消费逻辑。
- 进阶实验：KNNImputer 或 IterativeImputer，但必须用 CV 对比。

### 7. 类别变量编码

LightGBM 可以吃 one-hot，也可以使用 categorical feature。对于低基数类别，one-hot 很稳；对于中高基数特征，原生 categorical 或 target encoding 可能更有效。

建议实现：

- 低基数：`HomePlanet`、`Destination`、`VIP`、`Deck`、`Side` 可以 one-hot。
- 中等基数：`CabinNumBin`、`DeckSide` 可 one-hot 或 categorical。
- 高基数：原始 `Cabin`、`Name` 不建议直接 one-hot。
- Target encoding 只能 fold 内 fit，避免泄漏。

## 5. LightGBM 参数优化建议

### 5.1 稳健基础参数

这组参数适合在你已有预处理和特征工程上作为稳定基线：

```python
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbosity": -1,
}
```

建议配合：

```python
callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100),
]
```

### 5.2 进阶调参方向

优先搜索这些参数：

```python
search_space = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 47, 63],
    "max_depth": [-1, 3, 4, 5, 6],
    "min_child_samples": [10, 20, 30, 50, 80],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0, 5.0],
    "reg_lambda": [0.0, 0.1, 0.5, 1.0, 5.0],
}
```

调参顺序建议：

- 先固定特征和 5 折 `StratifiedKFold`。
- 先调 `num_leaves`、`max_depth`、`min_child_samples` 控制复杂度。
- 再调 `learning_rate` 与 `n_estimators`。
- 最后调 `subsample`、`colsample_bytree`、`reg_alpha`、`reg_lambda` 稳定泛化。
- 每次只比较 OOF accuracy、OOF logloss、test 正类比例和预测相关性，不要只看 public leaderboard。

## 6. 针对你当前 LightGBM 项目的优先级建议

### P0：统一评估口径

修改目标：所有模型都输出同一套 OOF 分数、test 预测、阈值、正类比例和可视化。

为什么要改：如果 baseline 和特征消融模型的评估口径不同，就无法判断哪个真的更好。

具体做法：

- 使用 5 折 `StratifiedKFold`。
- 保存每折 accuracy、OOF accuracy、OOF logloss。
- 保存 threshold sweep 的最佳阈值。
- 保存 test prediction agreement 和 probability correlation。

预期提升：不一定直接涨分，但能避免错误选择模型。

风险：CV 分数和 public leaderboard 仍可能不完全一致，这是 Kaggle 小数据集常见现象。

### P1：强化 `CryoSleep` 消费规则

修改目标：明确区分“真实 0 消费”和“缺失导致的 0 消费”。

为什么要改：`TotalSpent=0` 是强信号，但如果由 `skipna=True` 造成，会污染 `NoSpending`。

具体做法：

- 构造 `SpendMissingCount`。
- 构造 `AllSpendKnown`。
- 构造 `IsZeroSpendStrict = AllSpendKnown & (TotalSpent == 0)`。
- 构造 `CryoSleepSpendConflict`。

预期提升：提升模型对边界样本的判断稳定性。

风险：特征过多时要结合特征重要性和 CV 判断是否保留。

### P2：增加 group / family 特征

修改目标：增强同行组和家庭信息。

为什么要改：公开高分方案普遍重视 `PassengerId` group 和 surname。

具体做法：

- `GroupSize`
- `IsSolo`
- `SurnameFreq`
- `GroupSurnameSize`
- `HomePlanet_by_Group` 或组内众数填充

预期提升：通常能带来稳定小幅提升。

风险：不要用目标变量做 group target mean，除非严格 fold 内计算。

### P3：尝试 LightGBM 原生 categorical

修改目标：对 `Deck`、`Side`、`HomePlanet`、`Destination`、`CabinNumBin` 等类别列，比较 one-hot 与 LightGBM 原生 categorical。

为什么要改：LightGBM 对类别特征有原生处理能力，有时比 one-hot 更适合树模型。

具体做法：

- 保留类别列为 `category` dtype。
- 训练时传入 `categorical_feature`。
- 与当前 one-hot 版本在相同 CV 下比较。

预期提升：可能提升，也可能持平；主要价值是降低 one-hot 稀疏风险。

风险：如果 pipeline 当前已固定为 one-hot，切换成本较高，需要单独做实验分支。

### P4：模型融合

修改目标：在 LightGBM 稳定后，加入 CatBoost / XGBoost 做概率平均。

为什么要改：公开方案中多模型融合常能降低方差。

具体做法：

- 用同一份预处理数据训练 LightGBM、CatBoost、XGBoost。
- 保存每个模型 OOF probability。
- 简单平均：`0.5 * lgb + 0.3 * cat + 0.2 * xgb`。
- 用 OOF 搜索最佳权重。

预期提升：如果模型错误模式不同，public/private 稳定性会更好。

风险：融合不是越多越好；弱模型会拖累结果。

## 7. 推荐下一步

建议你按这个顺序推进：

1. 先用当前 `model_lightgbm1.py` 和 `model_lightgbm1 特征消融.py` 的 evaluation 结果确认 baseline 与 top-feature 模型差距。
2. 在 preprocess 中加入更严格的 `IsZeroSpendStrict`、`SpendMissingCount`、`CryoSleepSpendConflict`。
3. 用 5 折 OOF 重新比较 baseline 与 top-feature ablation。
4. 再做 Optuna 或 RandomizedSearch 调参。
5. 最后尝试 LightGBM + CatBoost + XGBoost averaging。

如果只选一个最值得马上做的动作：优先修正 `TotalSpent` / `NoSpending` 的缺失口径，并用 5 折 OOF 重新评估。这个动作最符合 SpaceShip Titanic 的数据规则，也最能减少“看起来有提升但实际泛化不稳”的风险。

## 8. 参考来源

- Kaggle Competition：[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- Daniel J Smith：[Spaceship Titanic](https://blog.danieljsmith.org/posts/23_11_23_SpaceshipTitanic/index.html)
- Fernandao Lacerda Dantas：[Space Titanic Kaggle Competition 0.8066 Score Solution](https://medium.com/%40fernandao.lacerda.dantas/space-titanic-kaggle-competition-0-8066-score-solution-7a9c401281c6)
- Maria Aguilera：[Spaceship Titanic Project](https://maria-aguilera.github.io/projects/spaceship-titanic.html)
- CSDN：[LightGBM + Optuna SpaceShip Titanic 公开方案](https://blog.csdn.net/2302_79308082/article/details/144576896)
