# Direct IK 三变体鲁棒性实验结果

## 1. 实验状态与可复现输入

本实验已完成 **540/540** 个 rollout，结果根目录为：

```text
outputs/robustness/paper_three_ik_l036_s5_v1/
```

实验矩阵为：

```text
3 个 IK 变体 × 4 条论文轨迹 × 9 个单因素条件 × 5 个 seed = 540
```

四条固定轨迹为 `circle`、`figure8`、`fast_ellipse`、`rounded_square`。reference
由 `outputs/paper_delay_aware_two_stage_v1/references/manifest.json` 冻结并在
`benchmark.json` 中记录 SHA-256。

| 变体 | 执行命令语义 | Preview |
|---|---|---:|
| RawIK (`raw`) | 直接发送下一步 IK command | 0 |
| PhysicalIK (`physical`) | 使用 physical command projection | 0 |
| PreviewIK (`preview`) | 使用 physical command projection | `P_cal=7` steps（70 ms） |

Direct IK 不加载 learned dynamics、checkpoint、CEM 或不确定性感知模块；它是固定
task-space IK reference 的同步 100 Hz 基线。五个 seed 对同一轨迹和条件配对，统计
单位是 trajectory-seed case，因此每个“变体 × 条件”表项有 `n=20`。

## 2. 扰动设置与验收

所有扰动均为单因素：一个 condition 中只有一种扰动非零。L0 是共享 nominal。

| 条件 | L3 | L6 |
|---|---:|---:|
| Payload | 4 kg | 12 kg |
| Actuator gain | Kp=0.70，Kd=0.837 | Kp=0.30，Kd=0.548 |
| Force pulse | 200 N，+Y，0.10 s | 500 N，+Y，0.10 s |
| Observation noise | q=0.001 rad，dq=0.010 rad/s | q=0.005 rad，dq=0.050 rad/s |

完整性与安全检查：

- summary 共 540 行，和 manifest 的 `run_count=540` 一致；
- failure、joint-limit、command velocity、command acceleration violation 全部为 0；
- 单个 rollout 的最大 control-compute P99 为 0.91 ms；
- 配对 bootstrap 使用 10,000 次重采样。

## 3. 汇总性能

下表为四条轨迹、五个 seed 的 pooled mean。RawIK 与 PhysicalIK 在所有记录指标上
数值一致，因此只列一列；两者逐 case 的 TCP RMSE 差恒为 0。

| 条件 | Raw/Physical TCP RMSE (mm) | Preview TCP RMSE (mm) | Preview − Physical (mm) | 相对变化 | Preview 姿态 RMSE (°) |
|---|---:|---:|---:|---:|---:|
| Nominal | 39.26 | 32.64 | -6.63 | -16.88% | 2.04 |
| Payload L3 | 57.20 | 52.73 | -4.48 | -7.82% | 3.67 |
| Payload L6 | 126.61 | 124.57 | -2.05 | -1.62% | 9.23 |
| Gain L3 | 45.20 | 38.95 | -6.25 | -13.82% | 2.29 |
| Gain L6 | 61.13 | 55.92 | -5.21 | -8.53% | 3.02 |
| Force L3 | 38.87 | 32.27 | -6.60 | -16.98% | 2.02 |
| Force L6 | 40.27 | 34.10 | -6.17 | -15.33% | 2.10 |
| Observation noise L3 | 39.26 | 32.64 | -6.63 | -16.88% | 2.04 |
| Observation noise L6 | 39.26 | 32.64 | -6.63 | -16.88% | 2.04 |

Nominal 条件下的其他指标：

| 变体 | TCP P95 (mm) | Joint RMSE (rad) | Orientation RMSE (°) | Torque RMS (Nm) |
|---|---:|---:|---:|---:|
| RawIK | 89.63 | 0.0148 | 2.01 | 5.60 |
| PhysicalIK | 89.63 | 0.0148 | 2.01 | 5.60 |
| PreviewIK | 74.75 | 0.0120 | 2.04 | 5.60 |

## 4. 轨迹差异

Nominal TCP RMSE 的 pooled mean 如下。PreviewIK 对每条正式轨迹均降低误差。

| 轨迹 | Raw/Physical (mm) | PreviewIK (mm) | 改善 (mm) |
|---|---:|---:|---:|
| Circle | 40.11 | 33.14 | -6.97 |
| Figure-8 | 38.02 | 31.70 | -6.32 |
| Fast ellipse | 43.93 | 36.74 | -7.18 |
| Rounded square | 35.00 | 28.97 | -6.03 |

Fast ellipse 是 nominal Direct IK 下误差最大的轨迹，rounded square 最小；这个差异同时
反映轨迹速度、几何、IK conditioning 与执行动力学，不能仅归因于轨迹形状。

## 5. 主要结论

1. **PreviewIK 在全部九个 condition 中均优于 PhysicalIK。** 平均 TCP RMSE 改善
   1.62–16.98%；nominal 下改善 6.63 mm，且所有四条轨迹均改善。
2. **Payload 是最主要的退化来源。** 12 kg payload 时 Raw/Physical TCP RMSE 从
   39.26 mm 升至 126.61 mm（+87.35 mm，+222%）；PreviewIK 也升至 124.57 mm。
   仅靠固定 preview 无法补偿显著的重力和惯量失配。
3. **低 actuator gain 次之。** L6 时 Raw/Physical 误差为 61.13 mm；PreviewIK 为
   55.92 mm。扭矩 RMS 降低不表示性能提升，而是 actuator authority 变弱的结果。
4. **Force pulse 主要增加瞬态误差与扭矩。** L6 使 Raw/Physical 的 TCP RMSE 增加
   1.01 mm、torque RMS 从 5.60 增至 11.34 Nm；PreviewIK 保持较低的总体 RMSE。
   L3 的轻微误差下降不应解释为外力有益，而是有限轨迹集合中的动力学抵消。
5. **Observation noise 对三种 Direct IK 都没有影响。** 两个 noise level 与 nominal
   完全相同，因为此基线根据固定 IK reference 发命令，不将观测用于在线反馈。这一结论
   不适用于后续使用状态反馈的 MPC。
6. **RawIK 与 PhysicalIK 在该 benchmark 上无性能差异。** 所有 paired TCP 差均为 0，
   表明这些冻结 reference 已满足当前位置、速度和加速度执行限制。physical projector
   仍是安全机制，但本数据集不能证明其改善 tracking；若要研究其作用，需要更激进的
   reference 或更严格的 command limits。

## 6. 原始结果与图件索引

- [实验 manifest](../../outputs/robustness/paper_three_ik_l036_s5_v1/experiment_manifest.json)
- [逐 rollout 指标](../../outputs/robustness/paper_three_ik_l036_s5_v1/direct_ik_robustness_summary.csv)
- [聚合统计](../../outputs/robustness/paper_three_ik_l036_s5_v1/direct_ik_robustness_aggregate.csv)
- [配对 bootstrap](../../outputs/robustness/paper_three_ik_l036_s5_v1/paired_bootstrap.json)
- [TCP RMSE 扰动曲线](../../outputs/robustness/paper_three_ik_l036_s5_v1/plots/tcp_rmse_vs_perturbation.png)
- [轨迹 × L6 对比](../../outputs/robustness/paper_three_ik_l036_s5_v1/plots/trajectory_type_level6_tcp_rmse.png)
- [外力恢复图](../../outputs/robustness/paper_three_ik_l036_s5_v1/plots/force_response_summary.png)

该实验仅应作为 Direct IK baseline 的鲁棒性证据。MPC 架构比较应使用相同 benchmark、
扰动矩阵和 seeds，但单独运行并报告其 planner latency、late packet、fallback 与
反馈/残差饱和等 MPC 特有指标。
