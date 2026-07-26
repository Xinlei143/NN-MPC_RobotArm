# Preview IK Nominal MPC：circle/figure8 鲁棒性结果

## 结论摘要

本次实验完成了 144/144 条 rollout。将已独立标定的 Preview IK
`P=7` 作为 residual MPC nominal 后，完整的 delay-aware MPC
（`VirtualDelayAware` 与 `ThreadedAsync`）在无扰动和八个单因素扰动条件下，
均显著优于不做 future alignment 的 `NaiveDelayed`；同时其 TCP 误差接近
零延迟 oracle (`IdealZeroDelay`)。

- 无扰动时，Virtual / Threaded 的 TCP RMSE 分别为 **23.63 / 24.84 mm**；
  Ideal 为 22.94 mm，Naive 为 62.94 mm。
- 所有扰动条件合并后，Virtual / Threaded 为 **38.33 / 38.65 mm**，
  Ideal 为 37.25 mm，Naive 为 72.41 mm。
- 144 条运行均没有 failure、关节限位、速度或加速度违例；Threaded 的
  fallback 占控制 tick 的 2.01%，但未导致 failure。
- 本实验只能说明 Preview-nominal 下四种 MPC 协议的相对表现；它**不能**单独证明
  MPC 优于 Preview IK baseline。该问题必须与已冻结的 Preview IK rollout 做配对比较。

测试日期：**2026-07-26**（Asia/Shanghai）。

## 冻结配置

| 项目 | 设置 |
|---|---|
| 输出目录 | `outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/` |
| 轨迹 | `circle`、`figure8` |
| CEM seeds | 0、1 |
| 条件 | nominal，以及 payload / actuator gain / force pulse / observation noise 的 L3、L6 |
| 总 rollout | 4 protocols × 2 trajectories × 2 seeds × 9 conditions = **144** |
| MPC protocols | `IdealZeroDelay`、`NaiveDelayed`、`VirtualDelayAware`、`ThreadedAsync` |
| 计算延迟 | `D=6` steps（从 `calibration/delay.json` 读取；Ideal 固定 `D=0`） |
| Preview nominal | `--mpc_preview_nominal_steps 7`，即 `q_nom[k]=q_des[t+k+1+7]` |
| 任务 tracking target | 未前移，仍为 `q_des[t+k+1]`；因此 P 不与 D 重复补偿 |
| 不确定性感知 | `--uncertainty_mode off` |
| learned dynamics | GRU、history 16，`gru_20260717_182930/best_model.pt` |
| CEM | horizon 20、128 candidates、2 iterations、每 5 tick replan |
| physical projection | on / compiled / two-stage |

完整可复现配置及文件 hash 位于
[`experiment_manifest.json`](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/experiment_manifest.json)。

## 控制语义

Preview 只改变 MPC 的 nominal actuator command：

```text
q_nom[k]    = q_des[t + k + 1 + P]       P = 7
q_ref[k]    = project(q_nom[k] + residual[k])
q_target[k] = q_des[t + k + 1]           tracking cost 不前移
```

`D=6` 只用于计划激活时刻和 future-state anchor。Threaded packet 保存的是
相对 nominal 的 residual；在每个真实执行 tick，使用当前 Preview nominal
重新构造命令，而不是重放过期的 absolute Preview command。

## 总体结果

下表是每个 protocol 的所有 36 条运行（2 trajectories × 2 seeds × 9 conditions）
平均值。TCP 和 P95 单位为 mm；orientation 单位为 degree。

| Protocol | TCP RMSE ↓ | TCP P95 ↓ | Orientation RMSE ↓ | Residual RMS (mrad) | Torque RMS (Nm) | Fallback tick rate | Failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | 35.66 | 68.06 | 1.974 | 26.47 | 225.32 | 0.00% | 0/36 |
| VirtualDelayAware | 36.70 | 68.47 | 2.128 | 23.28 | 225.16 | 0.33% | 0/36 |
| ThreadedAsync | 37.11 | 70.73 | 2.136 | 22.34 | 225.11 | 2.01% | 0/36 |
| NaiveDelayed | 71.36 | 147.29 | 11.990 | 92.81 | 222.31 | 0.33% | 0/36 |

Naive 的命令总变差较小（11.68 rad，其他三者约 22--23 rad），但这是明显
滞后、不能正确追踪轨迹的结果，并非更优的平滑性。

## 无扰动条件

无扰动条件共有每个方法 4 条配对运行（2 trajectories × 2 seeds）。

| Protocol | TCP RMSE (mm) ↓ | TCP P95 (mm) ↓ | Orientation RMSE (deg) ↓ | Residual RMS (mrad) | Fallback |
|---|---:|---:|---:|---:|---:|
| IdealZeroDelay | 22.94 | 47.37 | 1.271 | 18.49 | 0.00% |
| VirtualDelayAware | 23.63 | 46.06 | 1.319 | 18.11 | 0.33% |
| ThreadedAsync | 24.84 | 50.25 | 1.395 | 17.25 | 6.55% |
| NaiveDelayed | 62.94 | 129.00 | 11.549 | 88.63 | 0.33% |

相对 Ideal 的 paired bootstrap（n=4）显示：

- Virtual 的 TCP RMSE 增量为 **+0.70 mm**，95% CI `[+0.23, +1.10] mm`；
- Threaded 的 TCP RMSE 增量为 **+1.90 mm**，95% CI `[-0.17, +4.19] mm`；
- Naive 的 TCP RMSE 增量为 **+40.01 mm**，95% CI `[+35.02, +44.99] mm`；
- Virtual 相对 Naive 改善 **39.31 mm**，95% CI `[-44.49, -34.13] mm`。

因此，P=7 的 Preview nominal 已经消除了主要固定相位滞后后，future-aligned
MPC 与 zero-delay oracle 基本同量级；Naive delayed 的绝对命令重放仍然产生
了很大的位置和姿态误差。

## 扰动鲁棒性

下表为每种条件跨 circle/figure8 和两个 seed 的平均 TCP RMSE（mm，n=4）。

| 条件 | Ideal | Virtual | Threaded | Naive |
|---|---:|---:|---:|---:|
| nominal | 22.94 | 23.63 | 24.84 | 62.94 |
| actuator gain L3 | 26.05 | 27.24 | 26.01 | 67.52 |
| actuator gain L6 | 39.75 | 42.57 | 44.18 | 78.37 |
| force pulse L3 | 21.69 | 23.07 | 22.48 | 65.31 |
| force pulse L6 | 25.54 | 24.74 | 28.45 | 69.07 |
| observation noise L3 | 24.89 | 23.01 | 29.44 | 61.97 |
| observation noise L6 | 37.85 | 37.13 | 30.56 | 72.10 |
| payload L3 | 37.47 | 40.20 | 37.29 | 67.42 |
| payload L6 | 84.76 | 88.67 | 90.75 | 97.54 |

扰动条件（不含 nominal）合并后的结果：

| Protocol | TCP RMSE (mm) ↓ | Orientation RMSE (deg) ↓ | Torque RMS (Nm) | Command variation (rad) |
|---|---:|---:|---:|---:|
| IdealZeroDelay | 37.25 | 2.062 | 226.79 | 24.04 |
| VirtualDelayAware | 38.33 | 2.229 | 226.62 | 22.91 |
| ThreadedAsync | 38.65 | 2.229 | 226.55 | 22.48 |
| NaiveDelayed | 72.41 | 12.045 | 223.65 | 11.72 |

观察：

- Payload L6 是所有方法最困难的条件。Virtual、Threaded 的误差分别为
  88.67 和 90.75 mm，仍明显低于 Naive 的 97.54 mm，但与 Ideal 的差距扩大。
  这说明 Preview 只能处理近似固定的执行滞后，无法完全覆盖大的负载模型失配。
- Observation noise L6 下 Threaded 为 30.56 mm，优于 Virtual 的 37.13 mm；
  paired delta（Threaded − Virtual）为 **−6.57 mm**，95% CI
  `[-9.78, -1.66] mm`。该结果只有 n=4，应作为现象而非强结论。
- Force L6 下 Virtual 为 24.74 mm，优于 Threaded 的 28.45 mm；后者相对
  Virtual 的 paired delta 为 **+3.71 mm**，95% CI `[+0.71, +7.33] mm`。

## Threaded 实时性与安全性

ThreadedAsync 的 36 条平均统计：

| 指标 | 值 |
|---|---:|
| Planner E2E p50 / p95 / p99 | 45.18 / 53.09 / 58.49 ms |
| Planner update rate | 24.85 Hz |
| Control-period p99 | 10.16 ms |
| Wake-up lateness p99 | 0.35 ms |
| Late packet rate | 4.07% |
| Fallback tick rate | 2.01% |
| Packet expiration events | 73（跨全部 36 条运行） |
| Planner failures | 0 |
| Joint / velocity / acceleration violations | 0 / 0 / 0 |

Threaded 的误差相对 Virtual 略高，但仍远低于 Naive；其开销来自真实 wall-clock
异步求解、late packet 与 fallback，而不是 D/P 被重复计算。

## 可视化与原始数据

- [TCP RMSE vs perturbation](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/plots/tcp_rmse_vs_perturbation.png)
- [Orientation RMSE vs perturbation](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/plots/orientation_rmse_vs_perturbation.png)
- [Paired method TCP delta](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/plots/paired_method_tcp_delta.png)
- [Reliability and timing](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/plots/reliability_and_timing.png)
- [逐 rollout 汇总 CSV](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/delay_aware_mpc_robustness_summary.csv)
- [聚合统计 CSV](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/delay_aware_mpc_robustness_aggregate.csv)
- [paired bootstrap JSON](../../outputs/robustness/paper_preview_nominal_p7_circle_figure8_s2_v1/paired_bootstrap.json)

## 报告边界与复现实验注意事项

1. 本次输出使用同一份 immutable reference。由于该文件可供 `D=0, H=20, P=7`
   执行 1807 tick，而 `D=6, H=20, P=7` 只能安全执行 1802 tick，IdealZeroDelay
   比三种 delayed protocol 多运行最后 5 tick（0.28% episode）。这是为防止
   out-of-range Preview command 而做的尾部截断；没有修改任何参考样本。
2. 对论文中的严格配对主表，建议以 `--max_execution_steps 1802` 单独重跑
   IdealZeroDelay 的 36 条，令四种协议使用完全一致的评估窗口。现有趋势和所有
   delayed-protocol 的相对比较不受此问题影响，但 Ideal 对比应标注该边界。
3. 不要将“接近 Preview nominal 的误差”写成“MPC 优于 Preview IK”。需要把
   冻结的 Preview IK 结果以同一 circle/figure8、相同扰动和 seed 配对，才能回答
   residual MPC 是否带来额外收益。
