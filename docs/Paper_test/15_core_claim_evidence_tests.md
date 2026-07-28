# 核心论点补充测试：history alignment、tracking–effort 与候选排序

> 完成日期：2026-07-28  
> 原始输出：`outputs/paper_claim_evidence_v1`  
> 范围：MuJoCo-only；未进行实机实验，也未修改论文正文。

## 1. 结论摘要

本轮完成 99 个新增 ABB 闭环 rollout、54 个 activation snapshot 的
counterfactual candidate test，以及 ABB/UR5e 既有 rollout 的峰值控制作用后处理。
35 个相关单元测试全部通过，99 个新增闭环 case 中没有 planner failure、joint-limit、
command-velocity 或 command-acceleration violation。

| 问题 | 证据结论 | 对论文 claim 的影响 |
| --- | --- | --- |
| 完整 GRU history 对齐是否有独立收益？ | **当前 nominal 测试不支持。** Stale-history 相对 Full 的 TCP RMSE 差为 −0.67 mm，95% CI [−2.02, 0.45] mm。 | 不应继续把 complete-history alignment 写成已被单独证实的性能来源。 |
| 整体 future alignment 是否有效？ | **支持。** No-alignment 相对 Full 增加 2.89 mm，95% CI [0.84, 4.97] mm。 | 保留“activation-time state/reference alignment 有效”，但降低对 history 子机制的归因。 |
| tracking gain 是否只是无限制增大 effort？ | 存在清楚的 accuracy–effort trade-off；2× effort regularization 比默认 1× 同时降低平均 TCP RMSE 和 torque RMS，但所有 MPC 点仍比 Projected IK 使用更高 command acceleration。 | Discussion 应主动报告 trade-off；当前 1× 配置不是该扫描中的经验 Pareto 最优点。 |
| learned model 是否能支持 CEM 候选排序？ | **强支持，但仅限 projection-active snapshots。** 组内 Spearman 0.976，pairwise concordance 0.988，平均 realized regret 0.192%。 | 可以增加“projection-consistent candidate-ranking diagnostic”，但不能声称已验证 projection-inactive 条件。 |
| 峰值 actuator effort 是否可接受？ | UR5e 的最大 XML force utilization 为 23.7%；ABB XML 未配置有限 `forcerange`，不能做同类限值声明。 | ABB 只能报告 torque/jerk/power 数值，不应声称低于 XML 或真实机器 torque rating。 |

## 2. 冻结配置与复现信息

正式输入沿用 schema-v6 frozen manifest：
`outputs/paper_delay_aware_two_stage_v2/manifests/paper.json`。

| 项目 | 值 |
| --- | --- |
| 当前代码 base commit | `b4e6f807a9c511e41826c788a3d852135fac1bf2` |
| 本轮代码 patch SHA-256 | `df98d7e8e58f1eac4fe965d6c9cf2031c3c9e303910b8af8af0c2f4566772efa` |
| frozen manifest 原始 commit | `77505a528459c6d21c44822d1bc6f9ed4cdb8da0` |
| checkpoint SHA-256 | `9b7f65384f892bde1b4465e7d0795d1dafc806cf7bcb4bf7fb0ff965165684e5` |
| normalizer SHA-256 | `5738f1e85dd34820a7652aec34cd0eb29195601321b766644fe4c0f59892f7f4` |
| ABB XML SHA-256 | `fea13013e537f1b8f199ff2096b1718150b28bfd3117307edb6d0f87d212dc92` |
| reference hashes | circle `c15a1206…`, figure-eight `fc025a1f…`, fast ellipse `a84ffca0…` |
| MPC | GRU history 16，H=20，128 samples，2 CEM iterations，two-stage compiled projection |
| 时间语义 | 10 ms control step，20 Hz virtual replanning，固定 D=6 |
| 软件/硬件 | Python 3.10.20，PyTorch 2.11.0+cu130，MuJoCo 3.8.1，RTX 4060 Laptop GPU |

本轮是在 clean base commit 上加入实验代码后执行，因此 case fingerprint 继承 frozen
manifest 的输入身份，新增实现另以 patch hash 和逐文件 hash 冻结。正式投稿仓库若提交这些
代码，应在提交后重新记录最终 commit，而不是只保留 patch hash。

## 3. 完整 recurrent history 独立消融

### 3.1 语义和矩阵

测试使用 circle、figure-eight、fast ellipse，每条轨迹 5 个 paired CEM seeds：

| 变体 | activation state | recurrent history | reference window |
| --- | --- | --- | --- |
| FullAlignment | future | 完整 forecast 到 activation | future |
| StaleHistory | future | launch-time tokens；仅最后 state token 替换为预测 activation state | future |
| NoAlignment | launch | launch | launch |

三种变体都保留 execution-time residual reanchoring 和 feedback。总计
3 variants × 3 trajectories × 5 seeds = 45 cases。

### 3.2 结果

| 变体 | TCP RMSE [mm] | TCP P95 [mm] | joint RMSE [rad] | torque RMS [Nm] |
| --- | ---: | ---: | ---: | ---: |
| FullAlignment | 30.01 | 59.79 | 0.01237 | 18.66 |
| StaleHistory | 29.35 | 57.93 | 0.01208 | 15.64 |
| NoAlignment | 32.90 | 64.19 | 0.01442 | 20.33 |

配对、trajectory→seed 两级 bootstrap：

| 比较 | TCP RMSE 差 [mm] | 95% CI [mm] | 三条轨迹的均值方向 |
| --- | ---: | ---: | --- |
| StaleHistory − Full | −0.67 | [−2.02, 0.45] | −1.29 / −0.29 / −0.42 |
| NoAlignment − Full | +2.89 | [+0.84, +4.97] | +1.60 / +4.48 / +2.59 |

NoAlignment 显著退化，支持整体 activation-time alignment。StaleHistory 没有退化，
且三条轨迹的均值方向都略优，但主置信区间跨零；这不能证明 stale history 更好，也不能
支持“完整 forecast history 独立提高 nominal tracking”。更稳妥的论文表述是：

> The complete protocol advances the activation state, recurrent context, and
> reference window consistently. A targeted nominal ablation confirmed the
> benefit of the overall future-alignment module, but did not isolate an
> independent tracking gain from advancing the recurrent history alone.

这项负结果不等于 recurrent history 在扰动、长延迟或分布外条件下永远无用，只说明当前
三条 nominal 轨迹和 D=6 下没有观察到独立收益。若标题或摘要继续把 complete history
作为主要新颖性，仍需要更有针对性的 history-sensitive 扰动实验；否则应降低其权重。

## 4. Tracking–effort Pareto

### 4.1 扫描定义

保持 tracking、residual magnitude、模型和约束不变，只将
`w_servo`、`w_residual_velocity`、`w_residual_acceleration`、`w_first`
共同乘以预注册 scale `{0.25, 0.5, 1, 2, 4, 8}`。每个点包含三条轨迹和 3 个
paired seeds，共 54 cases。该扫描保持原论文的 black-box cost 信息假设，没有向
控制器暴露 MuJoCo torque。

| scale | TCP RMSE [mm] | 95% CI [mm] | cmd accel. RMS [rad/s²] | torque RMS [Nm] | torque P95 [Nm] | power P95 proxy [W] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25× | 39.17 | [30.94, 48.45] | 7.29 | 29.81 | 74.51 | 30.30 |
| 0.5× | 35.82 | [30.25, 40.89] | 7.11 | 26.03 | 62.66 | 26.85 |
| 1× | 30.70 | [27.01, 34.16] | 6.78 | 19.31 | 45.80 | 19.98 |
| 2× | 30.30 | [27.85, 33.25] | 6.56 | 13.40 | 31.43 | 12.92 |
| 4× | 33.49 | [30.85, 36.54] | 6.32 | 11.89 | 27.52 | 11.92 |
| 8× | 36.74 | [34.30, 40.07] | 5.60 | 8.37 | 19.86 | 8.32 |

相同三轨迹的 Projected IK 基线为 40.68 mm、0.57 rad/s²、5.90 Nm。因此：

- 六个 MPC 点的平均 TCP RMSE 都优于该 Projected IK 基线；
- 没有一个 MPC 点达到 Projected IK 的 command-acceleration 水平；
- 8× 将 torque RMS 降至 Projected IK 的约 1.42 倍，但 command acceleration
  仍约为 9.8 倍；
- 2× 相对默认 1× 同时略降 TCP RMSE、command acceleration 和 torque RMS，
  因而默认 1× 在这组 virtual tests 中被 2× 经验支配。

这张图适合支撑“accuracy–effort trade-off”，但不能直接把 2× 替换为论文最终方法。
如果决定采用 2×，至少需要重跑 ABB main/ablation/threaded、重新检查延迟标定，并在
UR5e 上复现；否则论文应继续报告 frozen 1× 结果，并把 2× 作为 post-freeze sensitivity。

产物：

- `summaries/pareto_command_acceleration.pdf`
- `summaries/pareto_torque.pdf`
- `summaries/effort_pareto_metrics.csv`
- `summaries/effort_pareto_aggregate.csv`

## 5. Candidate-ranking 与 realized regret

### 5.1 设计

每条轨迹、每个 seed 在稳定 task segment 的 10%–90% 位置预注册 6 个 activation
snapshots，共 54 snapshots。每个 snapshot 保存 selected、zero-residual baseline 和
alternative elite；若两种角色对应同一可执行序列则合并，最终得到 149 个唯一候选。

Primary diagnostic 使用 activation-projected branch：从真实 activation state 和真实
15-token context 出发，learned model 与独立 MuJoCo branch 消耗完全相同的 20-step
可执行 command sequence。每个 branch 使用新的 MuJoCo 实例，父环境状态在 observer
前后逐元素相同。

### 5.2 结果

| 指标 | 结果 | hierarchical bootstrap 95% CI |
| --- | ---: | ---: |
| within-snapshot Spearman | 0.976 | [0.917, 1.000] |
| all-pair concordance | 0.988 | [0.957, 1.000] |
| selected vs distinct zero baseline accuracy | 0.976（41 组） | [0.884, 1.000] |
| selected vs alternative elite accuracy | 1.000（54 组） | [1.000, 1.000] |
| selected relative realized regret | 0.192% | [0, 0.481%] |

13 个 snapshot 中 selected 与 zero baseline 是同一个 projection-consistent sequence，
因此从 distinct pair accuracy 中排除，而不是按“排序正确”计入。所有 54 个 snapshots
都有 projection activity，所以不能生成 projection-inactive 对照。

| replay endpoint | joint prediction RMSE |
| ---: | ---: |
| 1 step | 0.111 mrad |
| 5 steps | 0.459 mrad |
| 10 steps | 0.593 mrad |
| 20 steps | 0.889 mrad |

结果说明 frozen learned model 在这些实际 activation snapshots 上能可靠排序被保留的
CEM 候选，而不仅是报告低 open-loop state RMSE。证据边界是：每组最多三个角色候选，
不是全部 128-candidate population；并且所有 primary branches 都经过 activation
projection。

产物：

- `summaries/candidate_ranking.json`
- `summaries/candidate_ranking_groups.csv`
- `summaries/candidate_ranking.pdf`
- `candidate_ranking/{trajectory}_seed{0,1,2}.npz`

## 6. 峰值 effort 与 actuator-limit 后处理

下表复用 hash-compatible 的 ABB/UR5e nominal rollout。P95/max 对所有时间和关节的
绝对值统计；mechanical power proxy 为每 tick 的
`sum_j |tau_j * dq_j|`，避免正负功率抵消。

| Robot / method | accel. RMS | jerk RMS | torque RMS | torque P95 / max | power P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABB Projected IK | 0.58 | 51.7 | 5.60 | 13.67 / 26.97 Nm | 4.77 W |
| ABB FullVirtual | 6.69 | 703.5 | 17.17 | 40.67 / 123.24 Nm | 17.22 W |
| ABB ThreadedAsync | 6.74 | 688.9 | 16.25 | 38.76 / 113.35 Nm | 15.51 W |
| UR5e Projected IK | 0.79 | 78.6 | 2.07 | 5.72 / 10.95 Nm | 4.67 W |
| UR5e FullVirtual | 7.96 | 857.9 | 3.88 | 9.14 / 21.49 Nm | 7.63 W |
| UR5e ThreadedAsync | 8.03 | 785.3 | 3.93 | 9.12 / 21.44 Nm | 7.43 W |

单位依次为 rad/s²、rad/s³、Nm、Nm 和 W。UR5e 的 force-utilization P95/max 为：

| 方法 | utilization P95 | utilization max |
| --- | ---: | ---: |
| Projected IK | 4.54% | 7.36% |
| FullVirtual | 9.50% | 23.65% |
| ThreadedAsync | 9.55% | 23.73% |

UR5e 结果低于该 MuJoCo XML 的 `forcerange`，但这不是实机额定能力验证。ABB XML
position actuators 只定义 `kp`、`dampratio` 和 `ctrlrange`，没有有限
`forcerange`；因此 ABB utilization 为 unavailable，不能从这份 XML 推导 torque
headroom。

详细数据：

- `summaries/effort_peak_metrics.csv`
- `summaries/effort_peak_aggregate.csv`

## 7. 测试、审计与复现命令

相关单元测试为 35/35 passed，日志位于
`outputs/paper_claim_evidence_v1/unit_tests.log`。正式新增闭环 index 分别包含
45 和 54 条，candidate group CSV 包含 54 条 snapshot rows。

```bash
conda run -n pendulum-rl python -m scripts.paper_experiments.workflow \
  --output-root outputs/paper_claim_evidence_v1 run \
  --manifest outputs/paper_delay_aware_two_stage_v2/manifests/paper.json \
  --suite history_alignment --resume

conda run -n pendulum-rl python -m scripts.paper_experiments.workflow \
  --output-root outputs/paper_claim_evidence_v1 run \
  --manifest outputs/paper_delay_aware_two_stage_v2/manifests/paper.json \
  --suite effort_pareto --resume

conda run -n pendulum-rl python -m scripts.paper_experiments.claim_evidence \
  --output-root outputs/paper_claim_evidence_v1 candidate-ranking \
  --manifest outputs/paper_delay_aware_two_stage_v2/manifests/paper.json --resume

conda run -n pendulum-rl python -m scripts.paper_experiments.claim_evidence \
  --output-root outputs/paper_claim_evidence_v1 effort-postprocess \
  --abb-index outputs/paper_delay_aware_two_stage_v2/runs/indexes/main.json \
  --ur5e-index outputs/paper_ur5e_v2/runs/indexes/main.json
```

## 8. 建议的论文处理

1. **必须修改 claim：** 保留整体 future-alignment 贡献，但不要声称 complete GRU
   history 已被独立证明能改善 nominal tracking。
2. **建议加入 Pareto 结果：** 至少在补充材料报告 1×、2×、8× 和 Projected IK；
   Discussion 明确所有 MPC 点仍显著增加 command acceleration。
3. **可加入 candidate-ranking：** 这是比继续扩展普通 open-loop RMSE 更直接的
   learned-model 决策证据，但 caption/正文要写清只有 retained candidates 且全部
   projection-active。
4. **不要宣称 ABB torque headroom：** ABB XML 无 finite force limit；真实执行器能力
   仍需厂家数据或实机验证。
5. **是否需要继续实验：**
   - 若保持 frozen 1× 方法投稿：本轮已足以收紧 claim，不必再增加普通随机扰动。
   - 若改用 2× 作为最终方法：需要重跑 ABB main/ablation/threaded 和 UR5e matched
     replication。
   - 若仍把 complete recurrent history 放在标题级创新位置：需要专门设计
     history-sensitive 扰动或更长 delay 实验；当前数据不支持这一强 claim。

