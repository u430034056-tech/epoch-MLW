# XGBoost 分支 — Spaceship Titanic

> 2026-05-13 submission note: 本 README 记录的是 XGBoost 分支的历史实验路线。当前最终可提交版本已经固定为仓库内的 `outputs/submissions/final_submission_public_0p81412.csv`，Kaggle public score 为 `0.81412`。下文中 `submission_v2_best.csv` 的“最终推荐”说法只保留为当时的干净方法主线记录，不再代表当前最终提交文件。

本目录实现了 Spaceship Titanic 项目中 **XGBoost** 一路的完整实验管线，从问题诊断 → 特征/验证重构 → 贝叶斯调参 → 后处理审计 → 集成融合 → **LB 导向修正**，全部可一键复现。

**最终推荐提交**：`reports/xgboost/submissions/submission_v2_best.csv`
- V2 管线：无 Target Encoding + fold-aware 预处理 + Optuna 窄搜索 + 15-seed bagging
- 诚实 OOF acc = **0.8171**（15-seed bagged, `StratifiedGroupKFold`, threshold=0.500）
- 推测 LB：**0.810 ~ 0.813**（以 OOF-LB gap ≈ 0.005 推算）

> **为什么不是 A7 的 OOF 0.8200**？因为 A7 的 OOF 高分来自 `te_*` 特征的 self-leak（训练行看到自己的 y），导致 **OOF 虚高但 LB 下降** —— 提交后 A7 LB=0.80383，比原基线 0.80804 还差 0.004。详见 Section 0。

---

## 0. V2 迭代 — LB 导向修正 (2026-04-24)

### 0.1 问题发现

初版 A0→A7 消融基于**诚实 OOF** 做评判，最优 A7 达到 OOF 0.8200。但实际提交 Kaggle 后 LB 反而低于旧基线：

| 提交 | OOF acc | LB acc | gap |
|---|---|---|---|
| 旧基线 (A0 再现) | 0.8132 | 0.80804 | 0.005（正常） |
| A4 (Optuna tuned) | 0.8184 | 0.80406 | **0.014**（异常） |
| A7 (blend) | 0.8200 | 0.80383 | **0.016**（异常） |

**gap 从 0.005 飙到 0.016** 说明我们对 OOF 过拟合了。

### 0.2 根因诊断

三条证据链锁定 **plain Target Encoding self-leak**：

1. **特征重要性**：A4 的 top-20 gain 中有 5 个 `te_*` 特征（`te_HomePlanet`, `te_HomePlanetDestination`, `te_Side`, `te_CabinNumBin`, `te_Deck`），合计占约 **20% 的 gain**。
2. **`TargetEncoder.fit_transform` 的泄漏机制**：plain smoothing 公式 `(mean × count + global × m) / (count + m)` 的 `mean` 里**包含了该行自己的 y**。Smoothing=20 只稀释不消除，模型依然可以学到"te_* ≈ y"的极强伪信号。
3. **A/B 实验**（`tmp/smoke_v2.py`）：在 V2 干净管线上分别装 6 种 te_mode，OOF 结果：

| Stage | te_mode | surname_rate | OOF acc |
|---|---|---|---|
| V2a | none | False | 0.8162 |
| V2b | **loo** | False | **0.5065** ← 反而放大 leak（`sum_y − y_self` 二值分离） |
| V2c | oof | False | 0.8162 |
| V2d | none | True | 0.8138 ← surname rate 自己有 leak |
| V2e | loo | True | 0.5065 |
| V2f | oof | True | 0.8119 |

**结论**：
- `plain TE` 带来的 OOF 提升 (+0.0009) 主要是 self-leak 伪信号
- `loo TE` 因 `sum_y − y_self` 的数学性质反而把 leak 放大成 trivially 可分
- `OOF TE` 没 leak 但也**没带来任何 OOF 增益**（0.8162 vs 0.8162）
- 最干净的做法：**完全扔掉 TE 特征**，让 XGBoost 原生 category 去处理这些 level

### 0.3 V2 最终方案

| 组件 | V1（A4/A7）| **V2**（推荐）|
|---|---|---|
| Target Encoding | plain + OOF blend | **无**（扔掉全部 `te_*`）|
| 特征 | 55 原始 + 8 个 `te_*` + 2 log1p | 55 原始 + 2 log1p（57 总列） |
| fold-aware SurnameFreq | ✅ | ✅ |
| fold-aware CabinNumBin | ✅ | ✅ |
| 调参策略 | 宽 Optuna 60 trials on **leaky** data | **窄** Optuna 50 trials on clean data |
| Bagging | 5 seeds | **15 seeds** |
| Post-processing | threshold scan + CryoSleep rule + w×blend | **threshold=0.500 hardcoded**（零 OOF 微调） |

### 0.4 V2 消融结果（同一套 `StratifiedGroupKFold`, seed=42）

| Run | 参数 | OOF（单 seed）| OOF（15-seed bag） | Honest OOF |
|:-:|---|:-:|:-:|:-:|
| V2_A0like | 旧 `BASELINE_PARAMS` | 0.8132 | — | — |
| V2_STRONG | `STRONG_PARAMS` | 0.8162 | 0.8153 | 0.8153 |
| **V2_opt** | **窄 Optuna best** | **0.8189** | **0.8171** | **0.8171** |
| V2_pseudo | V2_opt + 98%/2% 置信度 pseudo-label 重训 | — | 0.8240（含 pseudo 行虚高） | **0.8120**（acc 反而降了 0.005）|

**最终选择 `V2_opt`（submission_v2_best.csv）**：OOF 最高 + 无 pseudo 噪声 + 方差已通过 15-seed 压平。

Optuna 找到的最佳参数（见 `reports/xgboost/logs/v2_best_params.json`）：

```python
learning_rate=0.0341   max_depth=6     min_child_weight=4
subsample=0.803        colsample_bytree=0.710   colsample_bylevel=0.697
gamma=0.801            reg_alpha=0.438          reg_lambda=0.631
n_estimators=3000  # 早停决定实际轮数
```

这组参数比 A4 的 Optuna 解（`lr=0.0446, depth=4, gamma=0.86`）更保守 —— 没有被 TE leak 牵向"浅树+强正则"的 overfit 小洞。

### 0.5 V2 复现命令

```bash
# 15-seed bagging（默认，最稳，生成 submission_v2.csv）
PYTHONPATH=. python -m src.xgboost.run_v2

# 自定义 seed（例如快测 5-seed smoke）
PYTHONPATH=. python -m src.xgboost.run_v2 --seeds 42 2024 7 1337 88 --tag v2_smoke

# 加 pseudo-labeling 对照（生成 submission_v2_pseudo.csv）
PYTHONPATH=. python -m src.xgboost.run_v2 --pseudo

# 用 Optuna 搜出的 best_params 重跑（手动组装 params）
PYTHONPATH=. python -c "
import json
from src.xgboost import config
from src.xgboost.run_v2 import run_v2, V2_SEEDS
best = json.load(open('reports/xgboost/logs/v2_best_params.json'))
p = dict(config.STRONG_PARAMS); p.update(best['best_params']); p['n_estimators']=3000
run_v2(seeds=V2_SEEDS, threshold=0.5, params=p, out_tag='v2_opt')
"
```

### 0.6 独立副本：public 0.81599 风格 raw-CSV 重建

除了 `V2` 主线，这个分支现在还保留了一条**完全不依赖 `processed/common` bundle** 的副本路线：

```bash
# 从原始 Kaggle train.csv/test.csv 直接重建 notebook 风格 XGB
PYTHONPATH=. python -m src.xgboost.run_public_style \
  --data-dir data/raw \
  --tag public81599_style \
  --threshold 0.50
```

这条路线的目标不是“诚实 OOF 最优”，而是验证一个**公开 notebook 已证实能上 public LB 0.81599** 的 XGB 家族。它会同时产出：

- `reports/xgboost/submission_candidates/submission_public81599_style.csv`
- `reports/xgboost/submission_candidates/submission_public81599_style.json`

当前脚本版的 honest `StratifiedGroupKFold` OOF 约为 **0.8070**，明显低于 `V2` 主线，但它和 `submission_v2_best.csv` 仍有 **446 行差异**，所以它是一个值得单独占用 Kaggle 提交位的**独立候选族**，不是微调噪声。

进一步基于讨论区里的 travel-group missing-value 思路，新加了一条 `run_public_groupfill.py` 副本：

```bash
PYTHONPATH=. python -m src.xgboost.run_public_groupfill \
  --data-dir data/raw \
  --tag public_groupfill_t052 \
  --threshold 0.52
```

它会做 group-consistent categorical fill，并额外加入 `GroupSize / IsSolo / CabinNumber`。当前 honest `StratifiedGroupKFold` OOF 在 `t=0.52` 约为 **0.8079**，虽然仍低于 `V2`，但它与 `submission_public81599_style.csv` 之间仍有 **545 行差异**，说明这又是另一条值得保留的独立 XGB 家族。

在这三条 XGB 家族概率都可复用之后，还可以直接生成纯 XGB 融合候选：

```bash
PYTHONPATH=. python -m src.xgboost.build_family_fusion
```

当前脚本会产出多类纯 XGB 融合候选，当前最值得关注的是：

- `submission_xgb_anchor511_plus_a7groupfill_override_t0507.csv`
  - 以 `anchor511_plus` 为底座，只在最不确定的窄区间 `[0.479, 0.507]` 内让 `A7_blend + groupfill` 的高置信一致样本改判
  - 当前 honest OOF 约 **0.82319**，测试集正例率约 **0.51227**
  - 这是目前本地最强的 XGB-only 候选
- `submission_xgb_anchor511_plus_a7blend_groupfill_t0507.csv`
  - 在 `anchor511` 上只加入极小的 `A7_blend / groupfill` 权重：`0.58% + 0.19%`
  - 当前 honest OOF 约 **0.82227**，测试集正例率约 **0.51134**
- `submission_xgb_anchor511_v2p39_stepwise46_public11_legacy3_safeTe2_t0507.csv`
  - 以 `submission_v2_best.csv` 的真实测试集正例率 `0.51134` 为锚点重做融合：`v2_opt 38.61% + stepwise_balanced_te 45.54% + public_style 10.89% + legacy_public 2.97% + stepwise_safe_te 1.98%`
  - 当前 honest OOF 约 **0.82216**，测试集正例率约 **0.51134**
- `submission_xgb_anchor511_v2p39_stepwise48_public11_legacy2_t0507.csv`
  - 更简洁的 anchor-511 版本：`v2_opt 39% + stepwise_balanced_te 48% + public_style 11% + legacy_public 2%`
  - 当前 honest OOF 约 **0.82204**，测试集正例率约 **0.51134**
- `submission_xgb_family_override_v2_public_groupfill.csv`
  - 以 `v2_opt` 为底座，只在 `v2` 犹豫区间内，让 `public_style + groupfill` 的高置信一致样本强行改判
  - 当前 honest OOF 约 **0.81836**
- `submission_xgb_bridge_v2p55_stepwiseBalanced45_t0485.csv`
  - `v2_opt 55% + fold-safe stepwise_balanced_te 45%`
  - 当前 honest OOF 约 **0.82032**
- `submission_xgb_bridge_v2p60_stepwiseBalanced40_t0475.csv`
  - `v2_opt 60% + fold-safe stepwise_balanced_te 40%`
  - 当前 honest OOF 约 **0.81962**，测试集正例率约 **0.53051**
- `submission_xgb_fourway_v2p40_stepwise40_public10_group10_t0477.csv`
  - `v2_opt 40% + fold-safe stepwise_balanced_te 40% + public_style 10% + groupfill 10%`
  - 当前 honest OOF 约 **0.82135**，测试集正例率约 **0.53379**
- `submission_xgb_fourway_refined_v2p40_stepwise42_public10_group8_t0477.csv`
  - `v2_opt 40% + fold-safe stepwise_balanced_te 42% + public_style 10% + groupfill 8%`
  - 当前 honest OOF 约 **0.82147**，测试集正例率约 **0.53402**
- `submission_xgb_family_blend_v2p95_publicp05_t0505.csv`
  - `v2_opt 95% + public_style 5%`
  - 当前 honest OOF 约 **0.81755**
- `submission_xgb_family_blend_v2p95_publicp05_t0475.csv`
  - 同样是 `v2_opt 95% + public_style 5%`，但把阈值下调到 `0.475`
  - 当前 honest OOF 约 **0.81445**，测试集正例率约 **0.53285**，更接近历史强 public-LB XGB 提交的分布
- `submission_xgb_public_family_pair_t051.csv`
  - `public_style 50% + groupfill 50%`
  - 当前 honest OOF 约 **0.81215**，但比 `v2` 家族更激进，适合作为“更像 Kaggle 公榜口味”的备选

### 0.7 Kaggle 高分 XGB 风格融合搜索

如果目标是直接冲 Kaggle 分数，而不是只做一条方法最干净的主线，可以运行：

```bash
PYTHONPATH=. python -m src.xgboost.kaggle_highscore_fusion
```

这一步**不改共享数据预处理**，只读取现有 XGB family 的 OOF/test 概率文件，结合 Kaggle 高分 XGB notebook 里反复出现的三类有效模式：

- public-style raw CSV XGB：`Expenses`、`CryoSleep` 零消费规则、Passenger group/room fill、Cabin split、Optuna XGB
- travel-group missing-value strategy：把 group-consistent fill 作为独立 family，而不是重写 common bundle
- 高分 notebook 常见融合：probability / rank / logit 三种 blend，并分别锚定 `0.511 / 0.525 / 0.532 / 0.536` 测试集正例率区间

输出：

- `reports/xgboost/logs/kaggle_highscore_fusion_summary.csv`
- `reports/xgboost/submission_candidates/submission_kaggle_hs_*.csv`
- 对应的 `submission_kaggle_hs_*.json`

这些候选是为了占用 Kaggle 提交位冲分；报告里仍建议优先讲 `V2_opt` 作为干净主线，把这一步表述为 XGB-only Kaggle submission search。

### 0.8 Umang Lodaya public XGB notebook exact-style branch

`run_public_style.py` 只复刻了 raw-CSV public-style 特征和参数；Kaggle notebook 的最终提交路径还包含 **SMOTE balancing**。如果要贴近该 public XGB notebook 的真实提交逻辑，运行：

```bash
PYTHONPATH=. python -m src.xgboost.run_umanglodaya_xgb --seeds 0 1 42 2024
```

输出：

- `reports/xgboost/submission_candidates/submission_umanglodaya_xgb_smote.csv`
- `reports/xgboost/submission_candidates/submission_umanglodaya_xgb_smote_seed*.csv`
- `reports/xgboost/logs/umanglodaya_xgb_smote*_test_proba.csv`

这条分支是 Kaggle LB 冲分用的独立 XGB 分支：它不改 common preprocessing，但会在 XGBoost 模块内从 raw CSV 重新构造 notebook-style 特征并做 SMOTE。

基于 2026-04-25 的 Kaggle 反馈，`submission_umanglodaya_refine_public_conf_13.csv`
只有 **0.81225**，低于 `submission_umanglodaya_xgb_smote.csv` / `seed0` 的
**0.81295**。因此不要继续优先提交 `refine_umanglodaya_public.py` 生成的回退
public-style 候选；冲分应优先试 `submission_umanglodaya_xgb_smote_seed*.csv`
这类 SMOTE 单 seed 变体，尤其是与 `seed0` 只差几十行的候选。

如果需要复现已经证明掉分的 public-style refinement，可运行：

```bash
PYTHONPATH=. python -m src.xgboost.refine_umanglodaya_public
```

它以 SMOTE 版为底座，只在 SMOTE 与 public-style 分歧、且 public-style 置信度明显更强的少数样本上回退 public-style 预测。输出文件名为：

- `reports/xgboost/submission_candidates/submission_umanglodaya_refine_*.csv`
- `reports/xgboost/logs/umanglodaya_public_refine_summary.csv`

### 0.9 Arunklenin high-score template XGB-only branch

Kaggle notebook `arunklenin/space-titanic-eda-advanced-feature-engineering`
页面显示 public score **0.82066**，但最终提交不是纯 XGB：它最后读取并 OR 合并了
多个外部 submission 文件。为了保留可解释、可复现的 XGB 分支，本地实现只复刻其
raw-CSV feature engineering + XGB 部分：

```bash
PYTHONPATH=. python -m src.xgboost.run_arunklenin_xgb --seeds 2140 --n-splits 10
```

输出：

- `reports/xgboost/submission_candidates/submission_arunklenin_xgb_only_*.csv`
- `reports/xgboost/logs/arunklenin_xgb_only_*`

当前本地慢版结果：OOF best accuracy 约 **0.81019**，但与当前 best `seed0`
相差约 **426-499** 行，属于高风险提交候选。优先级应排在
`umanglodaya_xgb_smote_seed*.csv` 之后。

---

## 1. 目录结构

```
src/xgboost/
├── config.py          # 路径、随机种子、特征 schema、搜索空间
├── data.py            # 从 common bundle 派生 XGBoost 友好特征矩阵
├── features.py        # fold-aware: SurnameFreq / CabinNumBin 重拟合 + Target Encoding
├── cv.py              # StratifiedGroupKFold + 健康度报告
├── model.py           # XGBClassifier 工厂 + 每折训练+早停
├── tune.py            # Optuna TPE 贝叶斯搜参
├── postprocess.py     # 阈值扫描 + CryoSleep 规则审计
├── ensemble.py        # 多 seed bagging
├── evaluate.py        # ROC / PR / CM / Importance / SHAP / Learning curve
├── run_all.py         # ★ 入口：一次跑完 A0→A7 + 产出所有图/CSV/提交文件
├── requirements.txt   # 依赖
└── README.md          # 本文件
```

所有产物写入 `reports/xgboost/`：
```
reports/xgboost/
├── figures/           # ROC/PR/CM/Importance/SHAP/ablation 等 PNG
├── logs/
│   ├── A0..A4/        # 每个 stage 的 OOF csv / 特征重要性 / fold 分数
│   ├── A5_rule_audit.json
│   ├── A5_threshold_scan.csv
│   ├── A6_per_seed_oof.json, A6_oof.csv, A6_test_proba.csv
│   ├── A7_oof.csv, A7_test_proba.csv
│   ├── ablation.csv   # ★ 消融总表（OOF + Δ vs A0/A1）
│   ├── best_params.json, optuna_trials.csv, optuna_study.db
│   └── feature_schema.json, full_run.log
└── submissions/
    ├── submission_best.csv   # ★ A7 blend (OOF 0.8200)
    ├── submission_A5_A4.csv  # A4 单模型 + 阈值
    └── submission_A6_ensemble.csv
```

---

## 2. 问题诊断（一次搞清楚"为什么要重做"）

在 pick 起之前已有的 XGBoost 管线中，本地一共找出了 **10 处问题**（代码里都有消融证据，不是纸上谈兵）：

### 致命层面（直接影响分数）

| # | 问题 | 证据 | 影响 |
|---|------|------|------|
| 1 | `processed/xgboost/preprocessed_xgboost.joblib` 里的 `X_train` 是 **`object` dtype 的 ndarray (8693, 107)**，XGBoost `hist` 模式对 object 数组会退化为"每个值独立 bin" | `tmp/inspect_bundles.py` 输出 `dtype=object` | 训练变慢且分裂质量差 |
| 2 | `SurnameFreq` 用 train 全集拟合后 test 也共享 → **CV 内相当于 train-valid 共用字典 → 泄漏** | `processed/common/metadata_common.json` 的 `future_cv_recompute_items` 字段自认 | 水分约 +0.002–0.003 acc |
| 3 | `CabinNumBin` 的分位边界同样是 train+test **联合拟合**（`spend_group_levels.combined_train_test_full_preprocessing`） | 同上 | 同类泄漏 |
| 4 | 5-fold CV 是 `StratifiedKFold`，**未使用 `GroupID` 分组**。train 里 6217 个 group 被平均拆散到 train/valid → 家人或同舱乘客的强关联被"共享" | `benchmark_models.py` | A0→A1 暴露 −0.0025 的水分 |
| 5 | `n_estimators=400` 写死、**无早停** | `benchmark_models.py` | 欠调或过拟合无提示 |

### 工程层面（影响可交付性）

| # | 问题 | 证据 | 影响 |
|---|------|------|------|
| 6 | 把干净的 51 列 `common_train`（含正确 dtype）**one-hot 回 107 列 object** | `metadata_xgboost.json` | 浪费信息 + 训练慢 |
| 7 | 特征名变 `num__Age` 这种 ColumnTransformer 前缀 | bundle 的 `feature_names` | 可解释性差 |
| 8 | 无超参数搜索、无调参日志 | `benchmark_models.py` | 不能上强 |
| 9 | 无 ROC / PR / 混淆矩阵 / SHAP / Learning curve | 仓库缺失 | 报告写不出 |
| 10 | 无 submission 的可追溯命名（0.8132 对应哪组参数？查不到） | `submissions/` 缺 metadata | 贡献度说不清 |

---

## 3. 解决方案 — 消融 A0 → A7

每个阶段只在前一阶段之上 **加一个改动**，并在同一套 `StratifiedGroupKFold` 上报 OOF。

### 运行结果（`reports/xgboost/logs/ablation.csv`）

| Stage | OOF Acc | Δ vs A1 | Δ vs A0 | 说明 |
|:-:|:-:|:-:|:-:|:--|
| A0 | 0.8127 | +0.0025 | 0.0000 | 旧 bundle + `StratifiedKFold` + 固定 400 棵（复现现状，对应 LB 0.80804） |
| **A1** | **0.8102** | **0.0000** | **−0.0025** | 同特征，但切到 `StratifiedGroupKFold` + 早停 → **暴露 CV 水分** |
| A2 | 0.8143 | +0.0041 | +0.0016 | 改用 `common_train` 51 列 + XGBoost 原生 category dtype + 更合理默认参数 |
| A3 | 0.8171 | +0.0069 | +0.0044 | fold 内重算 `SurnameFreq` / `CabinNumBin`，叠加 8 列 fold-safe Target Encoding |
| A3b | 0.8150 | +0.0048 | +0.0023 | 消融：OOF-TE + group-level 聚合 → **反而退步**（`TotalSpend`、`GroupSize` 已覆盖该信息） |
| A4 | 0.8184 | +0.0082 | +0.0056 | 60 次 Optuna TPE 调参（见 `best_params.json`） |
| A5 | 0.8184 | +0.0082 | +0.0056 | 阈值扫描（最优 t=0.500，**无额外收益**）+ CryoSleep 规则审计：flipped=165 / correct=74 / wrong=91 → Δacc=−0.0020 → **规则禁用** |
| A6 | 0.8184 | +0.0082 | +0.0056 | 5-seed bagging：per-seed OOF ∈ [0.8143, 0.8184]，平均后 = 0.8184（调参解稳定，方差已挤干） |
| **A7** | **0.8200** | **+0.0098** | **+0.0072** | **plain-TE × 0.45 + OOF-TE × 0.55 的概率平均 + t=0.510**（两路 target encoding 方案互补） |

> 真正"涨分"的改动（按实际增量，从大到小）：
> **A1→A2 特征重构 + native category +0.0041**  >  **A2→A3 fold-aware 预处理 + Target Encoding +0.0028**  >  **A4→A7 plain-TE × OOF-TE Blend +0.0016**  >  **A3→A4 Optuna 调参 +0.0013**。
> 其余（阈值扫描 / CryoSleep 规则 / 多 seed bagging / group-level 聚合）均为 **中性或负收益**，消融里如实记录。
> 另外 A0→A1 的 **−0.0025** 不是"退步"，是把原先"表面 0.8132"的 CV 水分挤出来 —— 这是重做的先决条件。

> 注：A3b 是刻意保留的 negative result，证明"不是所有能加的特征都值得加"，避免在报告里吹嘘 group-level 聚合。

---

## 4. 关键设计决策

### 4.1 验证集：`StratifiedGroupKFold`
- `GroupID` 由 `PassengerId` 的前 4 位派生，代表同行家庭/伙伴。
- 统计：train 6217 组，test 3063 组，**train/test 无交集**；train 组大小 ∈ [1, 8]。
- 用普通 KFold 会把一家人拆到 train/valid 两边，造成 leakage；`StratifiedGroupKFold` 保证同组不拆分，同时尽量平衡类分布。
- 所有 OOF / 调参 / 消融都基于同一套 fold（`seed=42`），方便跨阶段直接比较 Δacc。

### 4.2 特征重构：绕开旧 bundle
- 从 `processed/common/preprocessed_common.joblib` 的 `common_train` (51 列, 正确 dtype) 起步。
- 数值列保留原始值 + `log1p(*)` for 5 项花销 + `TotalSpend`；额外加 `SpendPerAge`、`SpendPerGroupMember`、`LuxuryMinusBasic`、`CryoSleepSpendAnomaly`、`MissingCount`。
- **类别列全部转成 `pandas.Categorical` + `enable_categorical=True`**，XGBoost 2.x 会自动找最优子集分裂，省去 107 维 one-hot 的空间和统计噪声。

### 4.3 fold-aware 预处理（堵泄漏）
每折内单独做：
- `refit_surname_freq`：只用 fold-train 的 Surname 频次字典给 valid/test 映射，未见姓氏映 0。
- `refit_cabin_num_bin`：fold-train 非缺失 CabinNum 的 5 分位作为 bin 边界，缺失归 `CabinBin_Missing`。
- `TargetEncoder`（Bayesian smoothing, m=20）：对 8 列（`HomePlanet`, `Destination`, `Deck`, `Side`, `DeckSide`, `HomePlanetDestination`, `AgeGroup`, `CabinNumBin`）做 fold-train 全局目标均值；valid/test 直接映射，未见 level 落回全局均值。
- A7 的 "OOF" 变体进一步用 inner 5-KFold 把 **fold-train 的 encoding** 也变成 out-of-fold，避免 train-row 看到自己的 y。

### 4.4 Optuna 调参（A4）
- 采样器：`TPESampler(seed=42, multivariate=True)`。
- 搜索 10 个参数：`learning_rate`（对数 0.01–0.1）、`max_depth` 3–10、`min_child_weight` 1–20、`subsample/colsample_bytree/colsample_bylevel` 0.5–1、`gamma` 0–5、`reg_alpha/reg_lambda`（对数 1e-3–10）、`max_delta_step` 0–7。
- 每 trial 内部仍然跑完整 5-fold group-aware CV + fold-aware 预处理 + 早停（150 轮）。
- 60 trials ≈ 4 分钟（M2 Mac），best 出现在 trial #N，OOF=0.8184：
  ```
  learning_rate=0.0446  max_depth=4     min_child_weight=2
  subsample=0.707       colsample_bytree=0.756   colsample_bylevel=0.848
  gamma=0.862           reg_alpha=0.0494         reg_lambda=0.0113
  max_delta_step=0
  ```
  参数倾向"浅树 + 强子采样 + 高 γ"，和 Spaceship Titanic 数据量较小、噪声不低的特征分布一致。

### 4.5 A5 后处理审计（负结果也要写进报告）
- **阈值扫描**：在 [0.30, 0.70] 以 0.005 步长遍历。最优 t=0.500，`acc=0.8184`，**与默认完全相同**。结论：Optuna 已经把概率推到了 calibration-optimal，不需要手动调阈值。
- **CryoSleep 规则**：naive 加减规则 (boost=+0.05 / penalty=−0.10) 会翻转 165 个预测，但其中 correct=74、wrong=91，净 −17 → **Δacc = −0.0020**。保留代码但默认关闭，消融中明确写出"规则 DISABLED"。

### 4.6 A6 多 seed bagging
- 5 个 seed ∈ {42, 2024, 7, 1337, 88}，每个 seed 复制一遍完整流水线，最后 `p̂ = mean(p̂_seeds)`。
- per-seed OOF acc ∈ {0.8184, 0.8174, 0.8157, 0.8150, 0.8143}，平均后仍 = 0.8184。
- 结论：Optuna 找到的解对 `random_state` 不敏感，方差已低；这条更多是"稳健性背书"而非性能提升。

### 4.7 A7 Stacking-style blend（当前冠军）
- 直觉：plain-TE 和 OOF-TE 产生 **互补的概率分布**（plain 有更高 accuracy, OOF 有更低 logloss + 更高 AUC）。
- 做法：`p̂ = w·p̂_plain + (1-w)·p̂_ooft`，在 w ∈ {0.20, 0.25, …, 0.80} 上做 OOF-accuracy 网格搜索，同步扫 threshold。
- 最优：`w=0.45`, `threshold=0.510`, **OOF acc = 0.8200** (logloss 0.3785, AUC 0.9057)。

---

## 5. 如何复现

```bash
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/xgboost/requirements.txt

# 全流程 (A0 → A7 + Optuna 60 trials + SHAP + 所有图)
PYTHONPATH=. python -m src.xgboost.run_all

# 使用已有 Optuna best_params 快速重跑（~40 秒）
PYTHONPATH=. python -m src.xgboost.run_all --load-best-params

# 只跑某几个阶段
PYTHONPATH=. python -m src.xgboost.run_all --stages A3 A4 A7 --load-best-params

# 跳过调参 / 集成
PYTHONPATH=. python -m src.xgboost.run_all --skip-tuning --skip-ensemble
```

所有路径自动解析，不依赖 CWD；所有随机性由 `config.RANDOM_SEED=42` 控制。

---

## 6. 提交文件

### ★ V2 系列（LB 导向，推荐）

| 文件 | 说明 | OOF (honest) |
|---|---|---|
| `submission_v2_best.csv` | **V2_opt = V2 + Optuna best + 15-seed bagging** | **0.8171** |
| `submission_v2.csv` | V2 + STRONG_PARAMS + 15-seed bagging | 0.8153 |
| `submission_v2_opt.csv` | 与 `v2_best` 相同内容（源文件） | 0.8171 |
| `submission_v2_pseudo.csv` | V2_opt + 98%/2% pseudo-labeling | 0.8120 |
| `submission_v2_smoke.csv` | V2 + STRONG_PARAMS + 5-seed（快测） | 0.8157 |

**实际 Kaggle 提交**：`submission_v2_best.csv`。推测 LB 0.810–0.813（以 OOF-LB gap ≈ 0.005 推算）。

### V1 系列（A0→A7，已证伪但保留作为消融依据）

| 文件 | 说明 | OOF | **实测 LB** |
|---|---|---|---|
| `submission_best.csv` | A7 blend (w=0.45, threshold=0.510) | 0.8200 | 0.80383 ✗ |
| `submission_A5_A4.csv` | A4 单模型，threshold=0.500 | 0.8184 | 0.80406 ✗ |
| `submission_A6_ensemble.csv` | 5-seed bagging，threshold=0.500 | 0.8184 | ~0.80406 ✗ |

**V1 系列的 LB 实测低于旧基线 0.80804，已由 V2 取代**。这些文件保留作为 "te_* self-leak → LB 退步" 的消融证据。

---

## 7. 为报告准备的材料清单

已生成、可直接引用：

- `figures/ablation.png` — 每 stage OOF 柱状图（数字标注）
- `figures/A4_roc.png`, `A4_pr.png`, `A4_confusion_matrix.png` — 基础评价图
- `figures/A4_importance_gain.png`, `A4_importance_weight.png` — 特征重要性
- `figures/A4_shap_summary.png`, `A4_shap_bar.png` — SHAP 全局解释
- `figures/A4_fold_curve.png` — 每 fold accuracy 与 best_iter 双轴图
- `figures/A5_threshold_scan.png` — 阈值-accuracy 曲线
- `logs/ablation.csv` — 消融总表（直接粘进报告）
- `logs/A5_rule_audit.json` — CryoSleep 规则审计数字
- `logs/A6_per_seed_oof.json` — 多 seed 方差表
- `logs/best_params.json`, `optuna_trials.csv` — 调参日志
- `logs/A7_oof.csv`, `A7_test_proba.csv` — 最终 blend 的 OOF + 概率明细（便于队友做外层 stacking）

---

## 8. 后续可做（不影响交付，但时间允许时值得）

1. **真正的 LB 验证**：V2_best 推测 LB 0.810-0.813，必须在 Kaggle 上实际提交确认。如果符合预期，说明诊断闭环正确；如果仍然低，再启动二次诊断。
2. **OOF-TE 的再评估**：Section 0.2 的 A/B 实验显示 OOF TE 没带来增益，但也许这只是"当前特征集已经覆盖了 TE 信号"。如果将来加入 more categorical features（e.g. Name tokens），可以再尝试 OOF TE。
3. **Optuna 扩展到 150 trials**（窄空间 + warm start 自 `v2_best_params.json`），可能再挤 +0.001-0.002。
4. **Pseudo-labeling v2**：本次在 acc 上持平或下降 0.005，但 logloss/AUC 改善。若切换到 **AUC 目标**或 **更严格置信度阈值**（99.5%），可能净正。
5. **Cross-model stacking**（跨 XGBoost/LGBM/CatBoost）：把 `v2_oof_proba.csv` 交给队友做 level-2 LR 融合。
6. **Deck × CabinNum × Side 三维交互**的更细分桶（目前只到 CabinNumBin 5 桶）。

## 9. 教训

- **诚实 OOF ≠ LB 友好**。Target Encoding 在 binary classification + small data 上极易带来"OOF 涨 +0.002, LB 跌 −0.004" 的净亏损。
- **消融时必须同时观察 OOF-LB gap**，只盯 OOF 分数会被 leak 类特征骗。本次 A0→A7 期间我们没有 early LB 反馈，纯靠 OOF 迭代，是问题产生的土壤。
- **LOO Target Encoding 不是 plain 的安全版本**。`sum_y − y_self` 的数学性质使它 **退化为 y 的强信号**，XGBoost 几次 split 就完全拟合 train，valid 崩。这是一个反直觉但非常容易踩的坑。
- **Bagging 饱和**：从 5 seed 到 15 seed，OOF std 从 0.0014 降到 0.0011，收益极小。15 seed 更多是 **稳定概率估计**（降 logloss）而非 accuracy。
- **Post-processing 在 OOF 上 tune 是 hidden leak**。A5 的阈值扫描、A7 的 blend weight 搜索都可能把 decision boundary 过拟合到当前 fold 分布。V2 全部硬码 threshold=0.5 避免。
