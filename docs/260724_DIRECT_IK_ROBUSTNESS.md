# DirectIK 鲁棒性实验

该工作流只运行同步 DirectIK，不加载 learned dynamics、checkpoint 或 CEM。它复用
`outputs/robustness/benchmark.json` 中的固定 task-space references，分别比较：

- `raw`：直接发送下一步 IK `q_des`；
- `physical`：发送经过共享速度/加速度投影的 IK 命令；
- 两者均固定 `ik_preview_steps=0`。

## 中等规模正式实验

默认配置使用 circle、figure-8、ellipse、square 各 3 个 case，独立测试 payload、
actuator gain、force pulse 和 observation noise 的 level 3/6，并包含共享 nominal。
总计 `12 cases × 9 conditions × 2 projections = 216` 个 rollout。

```bash
conda run -n pendulum-rl python scripts/robustness/evaluate_direct_ik.py \
  --manifest outputs/robustness/benchmark.json \
  --save_dir outputs/robustness/direct_ik_medium
```

中断后使用完全相同的命令并添加 `--resume`。fingerprint 会核对 reference、plant XML、
扰动、命令语义及全部运行参数；配置不一致时不会复用旧结果。

## 2026-07-24：正式结果与分析

结果目录为
[`outputs/robustness/direct_ik_medium`](../outputs/robustness/direct_ik_medium)。
完整性检查确认：

- 216/216 个 rollout，raw/physical 各 108；
- 12 个 case × 9 个 condition × 2 种 command projection；
- 每个 rollout 均为 500 steps；
- failure 和 joint/velocity/acceleration violation 均为 0。

### Raw 与 physical 完全一致

两个 projection 的 pooled mean 相同：

| Projection | TCP RMSE | Joint RMSE | Orientation RMSE | Command accel RMS | Torque RMS |
|---|---:|---:|---:|---:|---:|
| raw | 73.07 mm | 0.03115 rad | 4.161° | 1.045 rad/s² | 19.45 Nm |
| physical | 73.07 mm | 0.03115 rad | 4.161° | 1.045 rad/s² | 19.45 Nm |

所有 condition 的 paired TCP delta 都为 0；`projection_activation_rate`、
`projection_discrepancy_rms_rad` 和 `safety_projection_offset_rms_rad` 也全部为 0。
这不是 physical projector 没有执行，而是这些冻结的 IK reference 本身已经满足
position、velocity、acceleration 和 braking 约束，所以 projector 是 identity。

因此本数据集上不能证明 physical projection 能改善 Direct IK tracking；它的价值是
在 reference 更激进或异常时提供执行安全。后续 Direct IK 对照只需报告 raw，除非
reference 或 command limits 发生变化。

### 扰动敏感性

raw 和 physical 相同，所以下表只列一份结果：

| Condition | TCP RMSE | 相对 nominal | Joint RMSE | Orientation RMSE | Torque RMS |
|---|---:|---:|---:|---:|---:|
| nominal | 62.58 mm | — | 0.02586 rad | 3.120° | 11.14 Nm |
| actuator gain L3 | 70.75 mm | +8.17 mm (+13.1%) | 0.02961 rad | 3.593° | 9.34 Nm |
| actuator gain L6 | 92.09 mm | +29.51 mm (+47.2%) | 0.04018 rad | 4.983° | 5.87 Nm |
| force pulse L3 | 63.85 mm | +1.27 mm (+2.0%) | 0.02632 rad | 3.190° | 13.91 Nm |
| force pulse L6 | 69.97 mm | +7.39 mm (+11.8%) | 0.02851 rad | 3.534° | 23.53 Nm |
| observation noise L3 | 62.58 mm | 0 | 0.02586 rad | 3.120° | 11.14 Nm |
| observation noise L6 | 62.58 mm | 0 | 0.02586 rad | 3.120° | 11.14 Nm |
| payload L3 | 61.49 mm | -1.08 mm (-1.7%) | 0.02927 rad | 3.869° | 24.03 Nm |
| payload L6 | 111.73 mm | +49.15 mm (+78.5%) | 0.04891 rad | 8.913° | 64.91 Nm |

主要结论：

- **Payload L6 是最严重的扰动。** Direct IK 没有 dynamics prediction 或
  error feedback，无法补偿 12 kg payload 带来的重力和惯量变化；TCP RMSE 增加
  78.5%，torque RMS 增加到 64.91 Nm。
- **低 actuator gain 是第二大退化来源。** L6 下 TCP 增加 47.2%。虽然 torque
  RMS 下降，但这是 actuator authority 变弱的结果，不代表控制更高效。
- **Force pulse 主要增加瞬态误差和扭矩。** L6 的 pooled TCP 增加 11.8%，torque
  从 11.14 Nm 增至 23.53 Nm，但没有造成命令约束违规。
- **Observation noise 对 Direct IK 完全无影响。** Direct IK 按固定 IK reference
  发命令，不读取 noisy observation 做 feedback，因此两级 noise 与 nominal 完全相同。
  这不代表 plant state estimation 对其他闭环 MPC 方法不重要。
- Payload L3 的 TCP 轻微下降不能解释为 payload 有益；其 joint/orientation error
  和 torque 已经上升，TCP 的 -1.7% 是有限轨迹集合下的动力学偶然抵消。

paired bootstrap 进一步支持强扰动结论：payload L6 − nominal 的 TCP mean delta
为 +49.15 mm，actuator gain L6 − nominal 为 +29.51 mm，force L6 − nominal 为
+7.39 mm；对应 raw/physical 的 paired delta 均为 0。详细区间见
[`paired_bootstrap.json`](../outputs/robustness/direct_ik_medium/paired_bootstrap.json)。

### 轨迹差异与计算开销

Nominal 结果按轨迹类型：

| Trajectory | TCP RMSE | Joint RMSE | Orientation RMSE |
|---|---:|---:|---:|
| circle | 67.33 mm | 0.02907 rad | 3.279° |
| figure-8 | 62.31 mm | 0.02708 rad | 3.040° |
| ellipse | 61.89 mm | 0.02480 rad | 3.124° |
| square | 58.79 mm | 0.02248 rad | 3.038° |

Circle 在当前 IK reference 中最难，square 最低。这个排序同时包含轨迹几何、速度
profile 和 IK conditioning 的影响，不能只归因于形状名称。

Direct IK 的 nominal control-compute P99 约为 **0.22 ms**，远低于 learned CEM-MPC
约 30 ms 级的 solve latency。它适合作为低计算量 baseline，但在 payload/gain
model mismatch 下没有主动补偿能力。

Force recovery 指标只应在 `force_pulse` condition 中解释；其他 condition 中的 NaN
表示“不适用”，不能当作零恢复时间。

## 冒烟测试

使用独立目录，避免与正式实验 manifest 混用：

```bash
conda run -n pendulum-rl python scripts/robustness/evaluate_direct_ik.py \
  --manifest outputs/robustness/benchmark.json \
  --case_ids circle_00 \
  --levels 0,6 \
  --perturbations force_pulse \
  --ik_command_projections raw,physical \
  --max_execution_steps 20 \
  --bootstrap_samples 100 \
  --save_dir /tmp/direct_ik_robustness_smoke
```

## 输出

每个 `<projection>/<condition>/<case_id>/` 目录包含 `rollout.csv`、`rollout.npz`、
`run_summary.json`、`task_tracking_summary.json` 和完整跟踪/诊断图片。实验根目录包含：

- `experiment_manifest.json`：冻结的实验矩阵与输入文件身份；
- `direct_ik_robustness_summary.csv`：逐 rollout 指标；
- `direct_ik_robustness_aggregate.csv`：按条件和轨迹类型汇总的均值、标准差、中位数和 P95；
- `paired_bootstrap.json`：相对 nominal 及 physical-vs-raw 的配对差值和 95% CI；
- `plots/`：TCP、姿态、安全、平滑性、外力恢复和轨迹类型对比图。

默认只生成单因素条件。继承自底层 runner 的 `--payload_level` 等单项参数会被拒绝，
以免误将组合扰动混入同一实验。
