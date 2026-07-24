# Planner Projection 在 Delay-Aware MPC 中的作用与性能影响

## 1. 文档目的

本文说明本项目中 `planner_projection` 的准确含义、它与执行层安全投影的区别，以及
各类 MPC 控制器如何接入该开关。本文还解释为什么打开 planner projection 可能减少
planner--execution mismatch，却同时增加求解延迟，甚至使 threaded MPC 的跟踪误差
变差。

当前默认配置（2026-07-23 本地工作区）为：

```text
planner_projection = on
planner_projection_backend = compiled
planner_projection_strategy = two_stage
nominal_command_semantics = raw_ik
execution safety projection = always on
```

方案 B 在第一阶段对全部 CEM candidates 使用 cheap projection，在第二阶段只对
final elites、mean、best 和 baseline 做 exact physical projection 并重新计算
GRU rollout/cost。完整的历史、消融结果和默认值决策见
[MPC 架构与默认配置演进记录](260724_260724_MPC_ARCHITECTURE_EVOLUTION.md)。

无论 planner projection 选择 `off`、`full` 还是 `two_stage`，100 Hz 执行层都会
对最终命令做物理投影。因此，“关闭 planner projection”不等于关闭安全约束。

---

## 2. 两类 projection

### 2.1 Planner projection

Residual CEM 的候选动作为归一化 residual：

\[
r^{(i)}_{0:H-1}\in[-1,1]^{H\times n_q}.
\]

它先乘以每个关节的 residual 上限，再与 IK nominal 相加：

\[
q^{req,(i)}_t=q^{IK}_t+r^{(i)}_t.
\]

当 `planner_projection=on` 时，planner 从上一条实际命令
\((q^{cmd}_{-1},\dot q^{cmd}_{-1})\) 开始，沿 horizon 递推：

\[
q^{plan,(i)}_{0:H-1}
=\Pi\left(
q^{req,(i)}_{0:H-1};
q^{cmd}_{-1},\dot q^{cmd}_{-1}
\right).
\]

\(\Pi\) 同时考虑：

- joint bounds 与安全 margin；
- command velocity limit；
- command acceleration limit；
- 接近 joint limit 时的离散 braking distance。

投影后的序列随后作为 GRU 的 `future_q_ref`，用于 dynamics rollout 和 cost 计算：

```text
candidate residual
    -> raw IK nominal + residual
    -> horizon command projection
    -> GRU rollout
    -> tracking / residual / servo / limit cost
```

当 `planner_projection=off` 时，planner 仅做 residual bound 和 joint-limit clip：

```text
candidate residual
    -> raw IK nominal + residual
    -> joint-limit clip
    -> GRU rollout
    -> cost
```

对应实现位于：

- `mpc/planner_rollout.py::construct_residual_q_ref_sequence`
- `mpc/constraints.py::project_position_command_sequence`
- `mpc/planner_rollout.py::LearnedDynamicsPlanner.evaluate`

### 2.2 Execution projection

执行层投影发生在每一个 100 Hz control tick：

```text
当前 IK nominal
    + packet MPC residual
    + bounded feedback
    -> requested absolute command
    -> physical command projector
    -> actuator q_ref
```

它使用刚刚实际发送的 command 和 command velocity 作为递推初值，因此负责最终的
位置、速度、加速度和 braking 安全。它覆盖：

- 正常 active packet；
- 尚无首包的启动阶段；
- packet expiration；
- planner failure；
- late packet drop；
- zero residual；
- residual 与 feedback 相互抵消。

Execution projection 与 `planner_projection` 开关独立，正式配置下始终开启。

---

## 3. 不同控制器如何接入 planner projection

四种 MPC 共享同一个 `LearnedDynamicsPlanner` 和同一个
`PlannerRolloutConfig.project_residual_kinematics`。它们的差异主要在 plan 的锚点、
激活时刻、缓存语义和反馈，而不是使用不同的 projection 算法。

| 控制器 | Planner projection 接入位置 | Projection 对时序的影响 |
|---|---|---|
| IdealZeroDelay | `delay_aware_runner` 中同步调用 CEM；D=0 | 只用于逻辑零延迟上界。Projection 会改变候选 rollout 和 cost，但真实计算耗时不作为同 tick 可实现性的证明 |
| NaiveDelayed | `delay_aware_runner` 中同步调用 CEM；固定 D，`naive_delayed` protocol | 对候选绝对命令投影；packet 到达后按缓存绝对命令重放，没有 future alignment、re-anchor 和 fast feedback |
| FullVirtual | `delay_aware_runner` 中同步调用 CEM；固定逻辑 D | 对 future-aligned 候选序列投影；packet 激活时 residual 围绕当前 nominal 重新锚定，并叠加 fast feedback |
| ThreadedASAP | `asap_planner_worker` 后台 CUDA worker 中调用 CEM | Projection 的墙钟开销会直接降低 worker rate、增加 snapshot-to-publication E2E，并可能触发 late drop |
| Synchronous MPC | 主 runner 的同步 CEM 路径 | Projection 直接增加每次同步 control update 的计算时间 |
| DirectIK | 不经过 CEM planner，因此不读取 `planner_projection` | 可通过 `ik_command_projection=physical` 使用执行层 physical projector，但这不是 planner projection |

协议开关的具体差异为：

| Protocol | Future state/reference | Residual re-anchor | Fast feedback |
|---|---:|---:|---:|
| `full` | 是 | 是 | 是 |
| `naive_delayed` | 否 | 否，重放缓存绝对命令 | 否 |
| `no_future_alignment` | 否 | 是 | 是 |
| `no_reanchor` | 是 | 否 | 是 |
| `no_feedback` | 是 | 是 | 否 |

这些协议都可以在 virtual planner 中使用同一个 projection on/off 开关；
`threaded_asap` 只允许正式的 `full` protocol。

---

## 4. 为什么 projection 会影响误差

### 4.1 可能降低 prediction--execution mismatch

关闭 planner projection 时，GRU 看到的是 joint-clipped request，但执行层看到的最终
命令还会经过速度、加速度和 braking 投影。于是：

\[
q^{GRU}_t \neq q^{exec}_t.
\]

模型可能预测了一个实际不会被执行的快速动作，CEM 也可能据此低估 tracking cost。
打开 projection 后，GRU rollout 与 cost 基于更接近执行命令的序列，理论上能够降低：

- `projection_discrepancy`；
- `planner_execution_qref_error`；
- prediction--execution dynamics mismatch；
- 因执行层饱和造成的 residual 失真。

在当前 projection-off 三轨迹正式实验中，threaded 的 pooled mismatch 为：

```text
projection discrepancy RMS/P95/P99/max
= 13.30 / 18.90 / 55.84 / 191.47 mrad

planner-execution qref error RMS/P95/P99/max
= 12.65 / 17.07 / 54.13 / 191.35 mrad
```

这些数值说明 planner projection 并非没有理论价值；关闭它是在实时性和
planner--execution consistency 之间做取舍。

### 4.2 Projection 会改变 nominal 和 residual 的含义

当 nominal 为 raw IK 时，zero residual 原本对应：

\[
q^{req}_t=q^{IK}_t.
\]

打开 planner projection 后，zero residual 对应的 GRU 输入变成：

\[
q^{plan}_t=\Pi(q^{IK}_t),
\]

它可能滞后于 raw IK。此时：

- zero residual 不再是 raw Direct IK 的精确 planner baseline；
- `projected command - nominal` 不一定等于 requested residual；
- residual cost 若使用 requested residual，而 GRU 使用 projected offset，二者语义会分离；
- 原先针对 projection-off landscape 调整的 CEM std、cost weight 和 delay 不再等价。

因此，打开 projection 不保证 tracking RMSE 必然下降。它可能使单次计划更可执行，
也可能使 nominal 变慢、改变 elite 排序，或者让 CEM residual 用于补偿 projection
自身引入的滞后。

### 4.3 Threaded 中存在额外的“计划新鲜度”效应

对于真实 threaded 控制，tracking error 不只由单个 plan 的质量决定，还由 plan
发布频率和新鲜度决定：

\[
\text{closed-loop quality}
=f(\text{plan quality},\text{model error},\text{plan age},\text{feedback}).
\]

Projection 即使改善了单次 plan 的可执行性，只要它让 worker 明显变慢，就可能导致：

- worker update rate 下降；
- 使用更旧的 snapshot 和更旧的 CEM warm start；
- packet 激活时 residual 与当前状态偏差增大；
- 更接近 publication deadline；
- late drop 或 packet gap；
- fast feedback 需要修正更大的状态误差。

所以 threaded 的最终 RMSE 可能反而变差。Virtual 固定逻辑延迟不会完整复现这种
墙钟竞争，因此 projection on/off 必须同时报告 virtual 和 threaded。

---

## 5. 为什么 projection 会增加延迟

`project_position_command_sequence` 的 batch 维度是向量化的：128 个 candidate
并不是由 Python 逐条循环处理。但是 horizon 存在递推依赖：

\[
(q_t,\dot q_t)=g(q^{req}_t,q_{t-1},\dot q_{t-1}),
\]

所以 \(t+1\) 必须等待 \(t\) 的结果。代码对 H 个 horizon step 逐步执行，每一步包含
clip、加速度限制、速度限制、平方根 braking bound 和状态更新等多个小型 Torch
运算。

在历史 full-eager 配置：

```text
128 candidates × H20 × 2 CEM iterations
```

候选 batch 虽然并行，但 `H20 × 2 iterations` 的递推会产生大量顺序的小型 CUDA
kernel。每个 kernel 处理的数据量只有约 `128 × 6`，launch overhead 和同步依赖相对
计算量很大，GPU 难以获得大矩阵运算的吞吐优势。

已有配对诊断观察到：

```text
planner projection on:
    threaded E2E p95 ≈ 63.6 ms
    worker rate       ≈ 18.8 Hz

planner projection off:
    threaded E2E p95 ≈ 48.4--48.7 ms
    worker rate       ≈ 25.8--25.9 Hz
```

即 full population projection 增加约 14--16 ms E2E，并使 worker rate 从约 26 Hz
下降到约 19 Hz。该诊断用于解释机制，不应与不同 checkpoint、不同 semantics version
的正式结果直接合并。

当前 projection-off 三轨迹正式实验的 pooled threaded E2E 为：

```text
p50/p95/p99/max = 44.28 / 48.93 / 51.19 / 63.23 ms
worker rate      = 24.93--26.31 Hz
late drop        = 0
packet expiration= 0
```

---

## 6. 优化前的历史设计选择

在 H20 projection 优化完成前，论文实验曾选择：

```text
projection-free residual planning
+ always-on constrained execution
```

理由是：

1. execution projector 已保证最终命令的物理安全；
2. projection-off 恢复约 25--26 Hz worker rate；
3. 三条轨迹、三个 seed 中，ThreadedASAP 相对 FullVirtual 的总体 lap-RMSE 差为
   `-1.24 mm`，worst seed 为 `+2.24 mm`；
4. ThreadedASAP 相对 DirectIK 的总体 lap RMSE 改善为 `7.36 mm`；
5. 当时 full population projection 的实时开销大于已观察到的部署收益。

这项历史决策后来被第 8 节的 compiled/two-stage 优化结果取代。当前默认值为
`planner_projection=on`、`planner_projection_backend=compiled`、
`planner_projection_strategy=two_stage`。

---

## 7. 实验与论文报告要求

任何 projection on/off 比较必须冻结：

- checkpoint 与 normalizer；
- reference 文件及哈希；
- seeds；
- H、candidate 数和 CEM iterations；
- nominal、residual cost、packet residual 和 feasibility semantics；
- anticipation delay 与 planner guard；
- execution projector；
- controller protocol。

每次至少报告：

- full/lap/per-lap TCP RMSE；
- virtual planning time；
- threaded E2E p50/p95/p99/max 和 worker rate；
- late drop、packet expiration、planner failure、control deadline miss；
- command position/velocity/acceleration violation；
- projection discrepancy；
- planner--execution qref error；
- command acceleration max 和 torque RMS；
- paired seed difference、bootstrap CI 与 worst seed。

描述上述历史 projection-off 实验时，应使用如下表述：

> Delay-aware residual MPC with projection-free planning and
> braking-aware constrained execution.

不应写成 “projection-free execution” 或暗示 planner 直接优化了完全可执行的命令。

---

## 8. H20 projection 优化

当前实现额外提供两个显式实验路径：

- `planner_projection=on, planner_projection_strategy=full,
  planner_projection_backend=compiled`：把完全相同的 braking-aware 递推交给
  `torch.compile(fullgraph=True)` 融合。参数数值检查在进入 compiled core 前完成，
  执行层投影不变。
- `planner_projection=on, planner_projection_strategy=two_stage,
  planner_projection_backend=compiled`：两轮 population 使用 cheap joint-clipped
  rollout，随后将 final elites、跨轮 best、mean 和 baseline 去重并组成固定大小
  exact pool，重新执行 exact projection、GRU rollout 和 cost，只从 exact pool 选择。

H20、两个轨迹、三个 seed 的真实 Threaded E2E 标定结果为：

```text
off                 p95=49.88 ms -> D6
full eager exact    p95=66.30 ms -> D8
full compiled exact p95=51.61 ms -> D6
two-stage compiled  p95=45.81 ms -> D6
```

完整数据位于 `outputs/planner_projection_h20_optimization`。Two-stage 会改变 CEM
distribution update 的语义，因此必须作为独立策略报告；full compiled 则是
full-eager exact projection 的数学等价性能实现。

综合 E2E 标定和 tracking 结果，当前默认方案为 two-stage compiled，而
projection-off 和 full projection 保留为消融配置。
