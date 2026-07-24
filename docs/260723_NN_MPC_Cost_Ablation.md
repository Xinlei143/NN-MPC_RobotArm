# NN-MPC 机械臂 Cost Function 消融实验报告

**GRU 动力学模型与 Residual CEM-MPC**

| 项目 | 配置 |
|---|---|
| 项目名称 | NN-MPC Robot Arm |
| 核心模型 | GRU learned dynamics |
| 控制器 | Residual CEM-MPC with virtual ASAP |
| 实验种子 | 0, 1, 2 |
| 日期 | 2026 年 7 月 |

本报告总结 MPC cost function 的主要消融结果。核心比较包括完整 cost、去除 smoothness 项和去除 residual magnitude anchor 三组多随机种子实验，并补充单随机种子的 servo proxy、velocity tracking 和 checkpoint 对比。

## 摘要

本实验固定 GRU checkpoint、normalizer、三圈圆形参考轨迹、MPC horizon、CEM 参数和 ASAP 时序设置，仅改变指定 cost 项，并在随机种子 0、1、2 下重复运行。

结果表明：

- 完整 cost 的重复性最好；
- 去除 smoothness 项可降低平均跟踪误差，但会增加执行器力矩和跨 seed 波动；
- 去除 residual magnitude anchor 后，跟踪误差、最大误差和力矩均明显恶化，说明该项对限制激进 residual 修正至关重要；
- Servo proxy 和 velocity tracking 在单 seed 下影响较小。

由于不同运行中的 late-plan drop 数量存在差异，本文将结果解释为具有工程参考价值的趋势，而非严格统计显著性结论。

## 1. 控制框架与实验目的

NN-MPC 使用神经网络预测机械臂未来状态：

$$
\hat{x}_{k+1}=f_{\theta}(\hat{x}_k,u_k),
$$

其中：

- $x=[q,\dot q]$ 为关节角与关节速度；
- $u$ 为关节位置型执行器参考指令；
- $f_\theta$ 为参数为 $\theta$ 的 GRU learned dynamics model。

CEM 在有限预测时域内采样候选 residual control sequence，并选择累计代价最小的控制序列：

$$
U^*=\arg\min_U J(U).
$$

本实验的目标是判断不同 cost 项分别承担什么作用，以及删除某一项后，对以下指标产生什么影响：

- 关节与末端执行器跟踪误差；
- 执行器力矩与控制激进程度；
- 不同随机种子下的结果稳定性；
- 规划延迟、late-plan drop 与实验可解释性。

## 2. Cost Function 与实验配置

### 2.1 总代价结构

当前 residual MPC cost 可概括为：

$$
\begin{aligned}
J={}&w_qJ_q+w_{\dot q}J_{\dot q}+w_rJ_r+w_sJ_s \\
&+w_{\Delta r}J_{\Delta r}+w_{\Delta^2r}J_{\Delta^2r}+w_{\mathrm{first}}J_{\mathrm{first}} \\
&+w_TJ_T+w_{q,\mathrm{lim}}J_{q,\mathrm{lim}}+w_{\dot q,\mathrm{lim}}J_{\dot q,\mathrm{lim}}.
\end{aligned}
$$

各项含义如下：

- $J_q$：关节位置跟踪误差；
- $J_{\dot q}$：关节速度跟踪或速度抑制；
- $J_r$：residual magnitude anchor，限制 MPC 偏离 nominal/IK command；
- $J_s$：servo proxy，限制参考指令与预测关节状态之间的偏差；
- $J_{\Delta r}$、$J_{\Delta^2r}$、$J_{\mathrm{first}}$：控制平滑性与首步连续性；
- $J_T$：预测末端执行器误差；
- $J_{q,\mathrm{lim}}$、$J_{\dot q,\mathrm{lim}}$：关节位置和速度安全项。

### 2.2 固定配置

| 项目 | 设置 |
|---|---|
| Checkpoint | `gru_20260717_182930/best_model.pt` |
| Normalizer | 同目录下 `normalizer.pt` |
| 参考轨迹 | 三圈圆形轨迹 `circle_3laps/reference.npz` |
| 模型 | GRU，history length = 16 |
| 预测时域 | horizon = 20 |
| 重规划间隔 | 5 control steps |
| 时序模式 | `virtual_asap` |
| Anticipation delay | 10 steps |
| CEM | 128 samples，2 iterations |
| 随机种子 | 0，1，2 |
| 运行长度 | 2097 steps |
| 硬件 | NVIDIA GeForce RTX 3060，6 GB |

### 2.3 实验组

| 实验组 | 相对 Full baseline 的唯一修改 |
|---|---|
| Full | 不修改，使用完整 cost |
| No Smoothness | $w_{\Delta r}=0$，$w_{\Delta^2r}=0$，$w_{\mathrm{first}}=0$ |
| No Residual Anchor | $w_r=0$ |
| No Servo Proxy | $w_s=0$，仅 seed 0 |
| No Velocity Tracking | $w_{\dot q}=0$，仅 seed 0 |
| Best Rollout Checkpoint | 仅将 checkpoint 替换为 `best_rollout_model.pt` |

## 3. 三随机种子核心结果

### 3.1 汇总结果

以下结果均报告为 **mean ± sample SD，$n=3$**。

| 配置 | TCP RMSE / mm | Joint RMSE / rad | Orientation RMSE / deg | TCP max / mm | Torque RMS / Nm |
|---|---:|---:|---:|---:|---:|
| Full | 39.06 ± 1.06 | 0.01462 ± 0.00072 | 2.103 ± 0.087 | 82.49 ± 2.86 | 12.13 ± 3.31 |
| No Smoothness | 34.16 ± 5.06 | 0.01299 ± 0.00259 | 1.871 ± 0.307 | 76.86 ± 13.00 | 15.55 ± 4.90 |
| No Residual Anchor | 40.96 ± 9.36 | 0.01729 ± 0.00517 | 2.490 ± 0.722 | 111.75 ± 39.41 | 23.87 ± 9.65 |

### 3.2 Full：最稳定的 baseline

Full 的 TCP RMSE 在三个 seed 下分别为 37.95、39.17 和 40.06 mm，标准差仅为 1.06 mm。

它不是平均精度最高的配置，但跨 seed 波动最小。这说明完整 cost 对 CEM 随机采样最不敏感，能够提供更稳定、更可预测的闭环表现。因此，Full cost 适合作为当前论文的 baseline。

### 3.3 No Smoothness：平均更准，但力矩和方差增大

相对 Full，去除 smoothness 项后：

- TCP RMSE 平均降低约 12.5%；
- Joint RMSE 平均降低约 11.2%；
- Orientation RMSE 平均降低约 11.1%；
- Torque RMS 平均增加约 28.2%；
- TCP RMSE 标准差由 1.06 mm 增至 5.06 mm。

这说明默认 smoothness 权重限制了 residual MPC 的修正能力。完全关闭后，控制器平均跟踪更准，但执行器负担和结果波动同时增加。

因此，不建议直接永久删除 smoothness 项。更合理的后续方案是降低相关权重，并寻找跟踪精度、执行器负担和重复性之间的折中点。

### 3.4 No Residual Anchor：性能不稳定且力矩显著增大

相对 Full，去除 residual magnitude anchor 后：

- TCP RMSE 平均恶化约 4.9%；
- Joint RMSE 平均恶化约 18.2%；
- TCP 最大误差平均增加约 35.5%；
- Torque RMS 平均增加约 96.8%；
- TCP RMSE 标准差由 1.06 mm 增至 9.36 mm。

Seed 0 和 seed 1 均出现高力矩和明显误差，seed 2 则表现较好。这种差异说明没有 anchor 时，CEM 容易产生过于激进、偶然有效但缺乏重复性的 residual 修正。

因此，$w_r$ 应当保留。Residual magnitude anchor 是当前 cost function 中最关键的正则化项。

## 4. 单随机种子的补充结果

以下比较均使用 seed 0，属于探索性结果。

| 配置 | TCP RMSE / mm | Joint RMSE / rad | Torque RMS / Nm |
|---|---:|---:|---:|
| Full，seed 0 | 37.95 | 0.01399 | 10.28 |
| No Servo Proxy，seed 0 | 36.23 | 0.01363 | 11.80 |
| No Velocity Tracking，seed 0 | 37.35 | 0.01388 | 10.78 |
| Best Rollout Checkpoint，seed 0 | 38.68 | 0.01424 | 12.52 |

删除 servo proxy 只带来约 4.5% 的 TCP RMSE 改善，却使 Torque RMS 增加约 14.7%。这说明该项主要承担温和的执行器正则化作用，而不是主要的跟踪性能来源。

删除 velocity tracking 的影响很小，说明该项在当前圆形轨迹任务中不是主要性能来源。它可以保持较小权重，并在更复杂或速度变化更明显的任务中继续验证。

使用 `best_rollout_model.pt` 后，主要指标未优于 `best_model.pt`。因此，后续主实验继续采用 `best_model.pt` 是合理的。

## 5. 实验局限与有效性

所有正式运行均完成 2097 steps，且没有 planner failure 或关节限位违规。实验组之间使用相同 checkpoint、参考轨迹、CEM 配置和硬件，因此具备基本公平性。

但是，不同运行中的 late-plan drop 数量差异明显。部分运行由于规划时间超过 anticipation delay，计算完成的 residual 未能及时执行。这会使 cost 差异与时序差异同时影响最终结果，构成潜在混杂变量。

因此，应按以下方式解释当前结果：

- 当前结果具有明确的工程参考意义；
- 规划时间和实时性结论只代表当前硬件；
- 正式论文定稿前，建议在更快硬件或近乎零 late-plan drop 的统一时序条件下复核核心三组；
- 论文中应同时报告 planning mean、planning p95 和 late-plan drop 数量。

## 6. 最终结论

综合当前实验，可以得到以下结论：

1. **Residual magnitude anchor 是最关键的 cost 项。** 删除后平均力矩近乎翻倍，最大误差和跨 seed 方差显著增加，应当保留。
2. **当前 smoothness 权重可能过于保守。** 删除后平均跟踪更准，但力矩和波动增大。下一步应降低权重，而不是直接永久删除。
3. **Servo proxy 有一定执行器保护作用。** 它不是主要跟踪性能来源，但有助于限制控制激进程度。
4. **Velocity tracking 在当前任务中影响较小。** 可保持较小权重，或在更多任务中进一步验证。
5. **Full cost 的重复性最好。** 它适合作为当前论文 baseline。

当前各项重要性可概括为：

$$
\text{Residual Anchor}
>
\text{Smoothness}
>
\text{Servo Proxy}
>
\text{Velocity Tracking}.
$$

## 7. 论文与 GitHub 呈现建议

论文主表建议只包含具有三个 seed 的以下三组：

- Full；
- No Smoothness；
- No Residual Anchor。

主表应报告 mean ± sample SD。No Servo Proxy、No Velocity Tracking 和 checkpoint 对比可放在附录或补充材料中。

GitHub PR 不应直接提交 checkpoint、完整 NPZ 文件或大量重复图片。更合适的目录结构如下：

```text
configs/
  cost_ablation.yaml
scripts/
  run_cost_ablation.ps1
  summarize_cost_ablation.py
docs/
  cost_ablation.md
results/
  cost_ablation_all_runs.csv
  cost_ablation_three_seed_summary.csv
```

PR 的核心价值应是：能够用统一脚本自动运行不同 cost profile 和随机 seed，并自动汇总结果，而不是仅上传手动运行截图。

## 建议的论文表述

> Across three CEM random seeds, removing the residual smoothness penalties reduced the mean TCP tracking error but increased actuator effort and run-to-run variability. Removing the residual-magnitude anchor nearly doubled the mean torque demand and substantially increased the variance of tracking performance. These results indicate that the residual anchor is essential for bounding MPC corrections, while the current smoothness weights may be overly conservative and should be retuned rather than removed.
