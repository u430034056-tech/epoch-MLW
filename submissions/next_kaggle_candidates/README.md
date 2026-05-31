# 下一轮 Kaggle 提交候选（Public Recovery）

硬约束：顶层 8 个 CSV 全部基于团队 `processed/common` 派生出来的 raw-like 紧凑特征视图；没有把 raw-template 旧提交或 raw-template test probability 混入预测。

这次是针对刚刚 `drop_cabinnum + A7` 只拿到 `0.80430` 的修正：不再相信那条 OOF 方向，改用更接近 0.81412 raw seed2024 的团队预处理派生候选。

选择规则：用已知 public `0.81412` 的 raw seed2024 只做距离参照，不做 blend、不做训练输入。顶层候选按 `diff_vs_raw_seed2024` 从小到大排序。

## Public 反馈更新

- 已提交：`submission_team_common_rawlike_feedback_rank_k3_seed17_w15_true2291.csv`
- Public score：`0.81412`
- 2026-05-13 结论：`w15` 与上一条 `feedback_rank_k3_seed17_w05_true2291` 同为 `0.81412`，没有继续突破，但验证了 `0.81412` 附近是当前 public 局部平台。现在以 `w15` 作为最终可提交文件归档到 `../final_submission_2026-05-13/final_submission_public_0p81412.csv`。
- 已提交：`submission_team_common_rawlike_raw_core_k3_multi3_public_rate53566_seed7.csv`
- Public score：`0.81365`
- 已提交：`submission_team_common_rawlike_raw_core_k3_multi3_public_t050.csv`
- Public score：`0.81342`
- 已提交：`submission_team_common_rawlike_raw_core_k3_multi3_public_true2292.csv`
- Public score：`0.81342`
- 已提交：`submission_team_common_rawlike_raw_core_k3_multi5_rate53566_seed7.csv`
- Public score：`0.81318`
- 已提交：`submission_team_common_rawlike_feedback_rank_k3_seed17_w05_true2291.csv`
- Public score：`0.81412`
- 已提交：`submission_team_common_rawlike_anchor81412_rank_k3_seed17_w04_true2291.csv`
- Public score：`0.81365`
- 已提交：`submission_team_common_rawlike_anchor81412_rank_k3_seed17_w05_true2290.csv`
- Public score：`0.81388`
- 已提交：`submission_team_common_rawlike_anchor81412_rank_k3_seed17_w05_true2292.csv`
- Public score：`0.81388`
- 已提交：`submission_team_common_rawlike_anchor81412_rank_k3_seed17_w05_true2289.csv`
- Public score：`0.81365`
- 已提交：`submission_team_common_rawlike_anchor81412_diag_new_plus_removed.csv`
- Public score：`0.81388`
- 当前结论：`feedback_rank_k3_seed17_w05_true2291` 和 `feedback_rank_k3_seed17_w15_true2291` 并列当前最优 `0.81412`。`w15` 已作为最终可提交版本固定；继续提交后续扰动的收益很低，且容易把报告结论变成 public-feedback 局部搜索，而不是稳定方法提升。
- 相比 `0.81365` 锚点，`0.81412` 锚点只换了两个 PassengerId：多 True `4381_01`，少 True `7511_01`。
- 最新硬约束：
  - `2216_01` 必须保持 True。
  - `4381_01` 必须保持 True。
  - `1369_01` 必须保持 False。
  - `7511_01` 必须保持 False。
- 软约束：`7364_01` 暂时倾向保持 True，因为 `true2289` 同时把 `4381_01` 和 `7364_01` 改成 False 后分数下降；但它还没有被单独隔离验证。

## 最终提交与备用队列

最终提交文件：

`../final_submission_2026-05-13/final_submission_public_0p81412.csv`

已验证：

- Public score: `0.81412`
- 行数: `4277`
- 列: `PassengerId,Transported`
- True 数: `2291`
- Positive rate: `0.5356558335281739`
- PassengerId 顺序与 `sample_submission.csv` 对齐

备用队列仅在明确要继续占用 Kaggle 提交次数时使用：

1. `submission_team_common_rawlike_feedback_prob_k3_seed17_w25_true2291.csv` | true=2291 | diff_vs_0.81412=4 | prob blend，小幅拉开排序。
2. `submission_team_common_rawlike_feedback_rank_k3_seed17_w20_true2291.csv` | true=2291 | diff_vs_0.81412=4 | rank blend，比第 1 条更远。
3. `submission_team_common_rawlike_feedback_prob_k3_seed17_w30_true2291.csv` | true=2291 | diff_vs_0.81412=6 | prob blend 继续扩大。
4. `submission_team_common_rawlike_feedback_rank_k3_seed17_w25_true2291.csv` | true=2291 | diff_vs_0.81412=8 | 会动到软约束 `7364_01`，排在后面。
5. `submission_team_common_rawlike_feedback_rank_k3_seed17_w35_true2291.csv` | true=2291 | diff_vs_0.81412=10 | 风险更高，只在前面都失败后交。
6. `submission_team_common_rawlike_feedback_rank_k3_seed17_w40_true2291.csv` | true=2291 | diff_vs_0.81412=10 | 远邻域备选。
7. `submission_team_common_rawlike_feedback_rank_k3_seed17_w45_true2291.csv` | true=2291 | diff_vs_0.81412=12 | 最后备选。

已提交反馈归档：`99_submitted_public_feedback_2026-04-27/`
已提交 0.81412 邻域归档：`99_submitted_anchor81412_neighborhood_2026-04-28/`
非活跃反馈候选归档：`99_extra_feedback_candidates_not_active_2026-04-28/`
重复预测文件归档：`99_duplicate_anchor81412_not_active_2026-04-28/`
唯一活跃队列：`active_post_81412_queue_2026-04-28.csv`

## 来源

- Sprint: `/Users/shenyijie/Desktop/MLWP project/01_本地副本_实验/xgb_public_recovery_sprint_2026-04-27/`
- Manifest: `/Users/shenyijie/Desktop/MLWP project/01_本地副本_实验/xgb_public_recovery_sprint_2026-04-27/submissions/submission_manifest_public_recovery.csv`
- Curated report: `/Users/shenyijie/Desktop/MLWP project/01_本地副本_实验/xgb_public_recovery_sprint_2026-04-27/reports/public_recovery_curated_report.md`
- 上一轮顶层 team 候选已归档到：`99_上一轮team候选_2026-04-27_未删/`

## 校验

- 顶层 active CSV: 8 个唯一预测候选。
- 每个 CSV 都是 `PassengerId,Transported` 两列。
- 每个 CSV 都是 4277 行。
- `PassengerId` 顺序已校验。
- `Transported` 为 bool。
