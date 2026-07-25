# ASAP-MPC Projection、异步性能差距与 Seed 方差问题记录

更新日期：2026-07-23

## 0. 文档目的与问题范围

本文档记录当前 delay-aware ASAP-MPC 实验中尚未完全解决的一组相互关联的问题：

1. 历史 threaded-ASAP pilot 可以达到约35 mm TCP RMSE，但新实验一度退化到
   42--61 mm；
2. virtual-ASAP 通常优于 threaded-ASAP，但两者究竟是算法差异、wall-clock
   scheduling、packet gap，还是 command projection 导致，一开始没有被严格隔离；
3. 修复执行层 spike 后，一组结果看起来像 threaded 追上了 virtual，进一步检查却
   发现其中一部分来自 virtual 自身变差；
4. planner projection 理论上能够改善 prediction--execution consistency，但当前
   实现会明显增加 CEM latency；
5. 关闭 planner projection 可以恢复 threaded worker rate，却使 execution safety
   filter 更频繁地修改 planner 请求；
6. 相同配置的不同 CEM seed 之间出现约7--10 mm full/lap RMSE 差异，说明当前闭环
   仍存在较强的 stochastic optimizer path dependence；
7. 因此，当前还不能仅依据一两个 seed 决定论文最终采用 planner projection `on`
   还是 `off`。

本文档要回答四个问题：

```text
我们观察到了什么？
哪些原因已经由代码和日志确认？
哪些仍然只是待验证假设？
在什么条件满足后才能冻结论文配置？
```

本文档只讨论 learned GRU residual CEM-MPC 的 delay-aware virtual/threaded 部署。
Model C、payload robustness 和其他 discussion-only 实验不在本文主线中。

## 0.1 问题演化过程

### 阶段 A：历史 threaded pilot 表现较好

历史 `D_aligned_history_tick_start_snapshot` 运行使用旧 GRU、circle、seed 0、H20、
128 candidates、2 iterations 和 D6，得到约35.38 mm TCP RMSE，worker rate 约
28.9 Hz，且没有 control deadline miss 或 late packet。

这个结果最初让我们认为 threaded-ASAP 的异步 packet 切换实现已经基本稳定。
但该运行与后续正式 workflow 并非完全同条件：reference 长度、D、执行步数、代码
语义、projection 位置和统计区间都发生过变化，因此不能把35.38 mm与新实验的
full-trajectory RMSE直接比较。

### 阶段 B：统一旧模型、正式 reference 和 D7 后发现 threaded 退化

在同一旧模型、当前代码、正式 circle reference、D7 和 seeds 0--4 下，早期结果为：

```text
virtual full RMSE:  35.95 ± 1.10 mm
threaded full RMSE: 51.46 ± 6.87 mm

virtual lap RMSE:   33.17 ± 1.40 mm
threaded lap RMSE:  49.47 ± 4.14 mm
```

threaded 还出现4次 late packet，但没有 control deadline miss。这个结果说明问题不在
100 Hz控制线程本身，而更可能位于后台规划、packet replacement、fallback 或
prediction--execution command semantics。

### 阶段 C：发现 packet gap 的 zero-correction bypass

日志中观察到约1000--1500 rad/s²的命令加速度 spike，并与 packet age 从 `H-1`
变为 `-1` 的时刻重合。代码检查确认，旧 `project_executable_command_np()` 对零
correction 使用特殊分支，直接返回 nominal，不执行 velocity/acceleration projection。

其确定性故障链为：

```text
planner failure / late drop / replacement不及时
    -> 旧packet继续执行直至horizon耗尽
    -> active packet被清空
    -> requested correction变为0
    -> zero-correction特殊分支直接返回raw IK nominal
    -> command从旧residual command跳回nominal
    -> 产生极大离散velocity/acceleration spike
```

planner failure 本身不会立即发布一个零 residual packet。它只是使 replacement packet
缺失；真正的跳变发生在旧 packet 随后耗尽时。启动阶段、恰好为零的 residual，或
residual 与 feedback 相互抵消时也可能触发相同旁路，因此这不是单纯的 expiration
特殊情况，而是执行投影定义不完整。

### 阶段 D：修复 spike 后发现 virtual/threaded 比较仍被 projection 语义混杂

删除零 correction bypass 后，执行命令满足12.5 rad/s²加速度上限。但开启 planner
projection 后，threaded 仍比 virtual 差。进一步检查发现两个并行问题：

1. planner-side horizon projection 增加约14--16 ms E2E latency，将 worker rate 从
   约25.9 Hz降至约18.8--19.1 Hz；
2. planner projection、residual cost、packet residual 与 execution reanchoring 曾经
   使用不同的 residual 定义，使比较不再只是单个 projection 开关的消融。

因此代码进一步拆分：

```text
requested_mpc_residual
requested_feedback_correction
requested_total_correction
safety_projection_offset
command_nominal_offset
planner_execution_qref_error
```

并让 packet 同时保存 CEM requested residual 和 planner rollout 实际使用的 q-ref。

### 阶段 E：关闭 planner projection 后差距缩小，但不能立即宣布问题解决

关闭 planner projection、保留 execution projection 后，两个配对 seed 中 threaded 与
virtual 的差距缩小到约±3 mm。然而 seed 1 的 virtual 本身从 projection-on 的约
36.49 mm变为 projection-off 的44.04 mm，所以“差距变小”不能自动解释为 threaded
完全恢复。

另一方面，seed 0 中 virtual 基本不变（36.62 -> 37.07 mm），而 threaded 明显改善
（41.15 -> 37.61 mm），说明 worker latency 的改善确实给 threaded 带来了真实收益。

最终结论不是“off 已经获胜”，而是：

> 当前 planner projection `on` 受实时开销限制；`off` 能改善 threaded 部署，但会
> 增加 planner--execution action mismatch。两者都仍有需要解决的问题。

## 1. 当前结论

当前代码已经修复了 packet gap 时零 correction 绕过执行约束的问题，并统一了
virtual 与 threaded 执行层的 braking-aware position/velocity/acceleration
projection。修复后没有再观察到原来的 1000--1500 rad/s² 命令加速度尖峰。

但是，正式论文配置仍有一个尚未闭合的设计选择：

```text
planner_projection = on 还是 off
```

现有证据支持以下判断：

1. 100 Hz execution projection 必须始终开启，这是安全约束，不能作为性能开关关闭；
2. planner projection 开启时，预测、cost 与可执行命令的语义更一致；
3. 当前 Torch planner projection 实现明显增加计算时间，使 threaded worker 更新率下降并更接近 packet deadline；
4. planner projection 关闭时，实时性能改善，但 safety filter 经常修改 planner 请求，预测动作与实际动作仍存在不可忽略的偏差；
5. 当前只有两个完整配对 seed 的诊断，且 seed 间方差较大，因此不能据此冻结论文最终配置。

因此，当前代码默认的 `planner_projection=off` 应视为一个**暂定的部署候选**，
而不是已经证明优于 `on` 的最终方法定义。

## 2. 必须区分的两种 Projection

### 2.1 Planner projection

planner projection 在 CEM population rollout 内对每条候选命令序列施加关节位置、
速度和加速度约束：

```text
candidate residual
    -> nominal + residual
    -> planner-side physical projection
    -> GRU rollout
    -> tracking and actuator cost
```

开启它的主要优点是 GRU rollout 与 cost 使用更接近实际可执行的命令。

当前缺点是 projection 沿 horizon 递推，每一步依赖前一步 command 和 velocity。
在 `128 samples × H20 × 2 CEM iterations` 下，Python 循环会触发大量较小的
CUDA kernel，使真实 E2E latency 增大。

### 2.2 Execution projection

execution projection 位于 100 Hz 控制层：

```text
requested_mpc_residual
    + bounded_feedback
    -> requested_absolute_command
    -> shared braking-aware physical projection
    -> actual actuator command
```

该 projection 在 planner `on` 和 `off` 两种配置下都必须执行，也必须覆盖：

- 正常 active packet；
- 启动阶段尚无 packet；
- packet expiration；
- planner failure 后的 fallback；
- late drop 后的 fallback；
- residual 恰好为零的控制步。

因此：

> `planner_projection=off` 不等于关闭安全约束；它只表示安全投影没有进入 CEM
> population rollout，而是在 100 Hz execution layer 统一执行。

## 3. 已修复的确定性执行层缺陷

旧实现对零 correction 使用特殊分支，直接返回 IK nominal，绕过速度和加速度限制。
当 packet 从 age `H-1` 变成 `-1` 时，命令可能从 residual command 突然跳回
nominal，从而产生上千 rad/s² 的离散加速度。

当前修复包括：

- 删除 zero-correction bypass；
- 零 correction 与非零 correction 使用相同的物理投影；
- NumPy 与 Torch projection 使用相同的 acceleration、velocity、joint-limit 和
  braking-envelope 定义；
- packet gap 使用 constraint-projected Direct-IK fallback；
- 增加 packet expiration、fallback 和 planner result 的边沿事件；
- 普通 planner failure 与 worker fatal error 分开记录；
- 每个异步 planner result 使用唯一 `result_id`。

修复后的诊断运行中：

```text
maximum command acceleration = 12.5003 rad/s²
command acceleration violation count = 0
control deadline miss count = 0
packet expiration count = 0
```

## 4. Planner Projection On/Off 的现有结果

诊断条件固定为：

```text
model                  gru_20260717_152930
reference              outputs/references/circle_3laps/reference.npz
control rate           100 Hz
execution steps        1807
horizon                20
CEM                    128 samples, 2 iterations
anticipation delay     D = 7
feedback               Kq = 0.3, Kdq = 0.015
packet semantics       requested residual
residual cost          projected_offset（仅本轮 projection 定位）
nominal semantics      raw IK
execution projection   shared_physical_v2, always on
```

这一点非常重要：下表中的 `on/off` 使用相同的 `projected_offset` cost 以定位
planner projection 的计算与执行影响，并不是正式 `requested` residual-cost 配置的
最终比较。`off` 时 planner 侧只有 joint-limit clip；本轮轨迹上
`planned_q_ref - nominal - requested_mpc_residual` 的最大绝对误差仅
`2.98e-8 rad`，所以 `projected_offset` 与 `requested` 数值等价。`on` 时两者不再
等价，正式实验仍需在 held-out calibration 上重新冻结 requested-residual cost。

已有结果位于：

```text
outputs/residual_semantics_diagnostic/
```

### 4.1 Tracking 与实时性

| Planner projection | Seed | Virtual full/lap | Threaded full/lap | Threaded E2E p95 | Threaded rate |
|---|---:|---:|---:|---:|---:|
| on | 0 | 36.62 / 39.91 mm | 41.15 / 43.99 mm | 62.50 ms | 19.08 Hz |
| on | 1 | 36.49 / 39.53 mm | 44.95 / 49.78 mm | 63.58 ms | 18.77 Hz |
| off | 0 | 37.07 / 37.49 mm | 37.61 / 39.83 mm | 48.72 ms | 25.82 Hz |
| off | 1 | 44.04 / 49.03 mm | 41.92 / 45.85 mm | 48.36 ms | 25.94 Hz |

在两个 `off` 配对 seed 中，threaded-minus-virtual 为：

```text
seed 0: full +0.54 mm, lap +2.35 mm
seed 1: full -2.13 mm, lap -3.17 mm
```

两 seed 配对平均差为：

```text
full trajectory: -0.79 mm
lap only:        -0.41 mm
```

这些结果说明关闭 planner projection 后 threaded 与 virtual 的平均差距变小。
但它们不能证明 `off` 在算法上更优，原因包括：

- 只有两个 seed；
- `on` 和 `off` 改变了 CEM 实际评估的命令与 cost landscape；
- seed 1 的 virtual `off` 本身明显退化；
- threaded 和 virtual 的 solve schedule 不同，不能逐次复用完全相同的候选集合。

### 4.2 关闭 Planner Projection 后仍存在较大的执行修改

`off` 配置中的 execution safety projection 激活率为：

```text
virtual:  84.1% -- 87.5%
threaded: 85.8% -- 87.0%
```

全轨迹 safety projection offset RMS 为：

```text
virtual seed 0:  15.68 mrad
virtual seed 1:  23.15 mrad
threaded seed 0:  9.14 mrad
threaded seed 1: 15.95 mrad
```

因此不能声称 execution filter “几乎没有修改 planner 动作”。论文若使用 `off`，
必须同时报告：

- `safety_projection_offset`；
- `planner_execution_qref_error`；
- projection activation rate；
- requested 与 executed command 的约束统计。

## 5. 为什么两个 Seed 差异明显

MuJoCo plant 和 reference 在这些运行中是确定性的。seed 主要控制 CEM sampling，
所以 seed 差异表示 optimizer stochasticity，而不是环境随机性。

### 5.1 Per-lap tracking

| Mode | Seed | Lap 0 | Lap 1 | Lap 2 |
|---|---:|---:|---:|---:|
| Virtual off | 0 | 36.70 | 37.21 | 38.53 mm |
| Virtual off | 1 | 38.97 | 52.23 | 54.44 mm |
| Threaded off | 0 | 39.32 | 39.53 | 40.63 mm |
| Threaded off | 1 | 40.19 | 55.16 | 40.62 mm |

seed 1 主要在第二圈进入了更差的闭环轨迹，而不是在启动阶段整体平移。

### 5.2 Projection discrepancy 同时增大

Lap 1 的 safety projection offset RMS：

| Mode | Seed 0 | Seed 1 |
|---|---:|---:|
| Virtual off | 9.35 mrad | 33.31 mrad |
| Threaded off | 8.91 mrad | 27.04 mrad |

对应 torque RMS 也明显增大：

```text
virtual:  17.4 -> 24.1 Nm
threaded: 11.0 -> 17.3 Nm
```

与此同时，requested residual RMS 变化较小，也没有 residual saturation。因此问题
不是 seed 1 简单地产生了更大的 residual，而是不同 residual 时序经过有状态的
velocity/acceleration projection 后产生了不同的实际命令，并通过闭环状态逐步放大。

### 5.3 CEM 的路径依赖

当前 CEM 配置只有128个 samples 和2次 iterations，并且：

```text
reset_std_each_step = false
warm-start mean      = enabled
```

一次不同的 elite selection 会改变下一次 solve 的 mean 和 std；状态发生偏离后，
后续 CEM 又会在不同状态与 history 上继续优化。因此固定 seed 可以复现一次运行，
但不同 seed 之间可能进入不同的局部闭环轨迹。

此外，同一 seed 也不能让 virtual 和 threaded 使用逐次相同的随机候选：

- virtual 每5个控制步固定规划一次，约362次 solve；
- threaded 严格 ASAP，当前约467--469次 solve；
- 两者的 solve state、history、warm-start shift 和随机数消费次数均不同。

所以论文中应称为 `five paired CEM seeds`，不能将其解释为五个独立环境 trial，
也不能把 paired seed 理解为逐 action 的 common-random-number 实验。

## 6. 当前尚未解决的问题

### P0：正式配置尚未冻结

不能只根据当前两个 seed 决定 `planner_projection=on/off`。至少需要完成 seeds 0--4
的完整配对，并报告 mean、sample std、paired difference、bootstrap CI 和 worst seed。

### P0：Planner projection 的实时开销

当前 `on` 将 threaded E2E p95 推高到约63 ms，而 D7、5 ms guard 的 publish
deadline 约为65 ms，余量过小。需要定位并优化 horizon 递推中的 CUDA kernel
launch 开销，或在冻结 D 与 CEM budget 前重新标定。

### P0：Off 配置的 prediction--execution mismatch

`off` 的 projection activation rate 超过84%，且 discrepancy 不是数值噪声。
如果无法降低 mismatch，就不能将 `off` 描述为预测与执行一致的 MPC。

### P1：Seed variance

当前 seed 1 在特定 lap 明显退化。需要判断方差主要来自：

- CEM sample 数不足；
- 两次 iteration 不足；
- warm-start mean/std 路径依赖；
- std collapse；
- uniform exploration 比例；
- cost 对被 projection 修改的动作缺少辨识；
- fast feedback 与 execution filter 的相互作用。

### P1：Planner-on 后的 cost 需要重新标定

修正 residual 语义后，residual magnitude/velocity/acceleration cost 应作用于
`requested_mpc_residual`。不能因为旧权重在 projected-offset cost 下效果较好，
就把 safety projection lag 重新命名成 MPC residual。若 `on` 的 tracking 退化，
应该在独立 calibration trajectory 上重新标定 cost weights。

## 7. 建议的下一轮实验

第一阶段只做定位，不进入论文正式结果：

```text
projection: on, off
mode:       virtual, threaded
seeds:      0, 1, 2
trajectory: circle
D:          7
```

每个 case 必须报告：

```text
full/lap TCP RMSE
per-lap TCP RMSE
planner solve and E2E p50/p95/p99
actual planner rate
planner failure reason counts
late drop and packet expiration
requested MPC residual RMS
safety projection offset RMS/p95/max
planner-execution qref error RMS/p95/max
projection activation rate
command acceleration max and violation count
torque RMS
```

第二阶段针对 seed variance，在固定 projection 配置下比较：

```text
current warm start
reset_std_each_step = true
higher minimum std
higher uniform sample ratio
quasi-random/Sobol population
```

所有超参数只能在 held-out calibration trajectory 上选择，不能根据正式 circle、
figure-eight、fast ellipse 或 rounded-square 的结果反复调整。

## 8. 正式验收标准

### 安全硬标准

任何 seed、failure injection 或 packet gap 下都必须满足：

```text
no NaN/Inf command
no worker fatal error
zero joint/velocity/acceleration command violations
no acceleration spike at packet expiration
NumPy/Torch execution projection parity
correct planner-result and fallback event counts
```

### Threaded 可用性目标

正式 D7 seeds 0--4：

```text
control deadline miss = 0
packet expiration after first activation = 0
all_costs_invalid = 0
late drop rate near 0
planner E2E p95 has explicit guard margin
```

### 性能与一致性目标

```text
paired mean(threaded lap RMSE - virtual lap RMSE) <= 5 mm
worst-seed difference reported
95% paired bootstrap CI reported
projection discrepancy reported, not hidden
```

若使用 `planner_projection=off`，还需要预注册一个可接受的
`planner_execution_qref_error` 阈值；否则仅满足 tracking RMSE 不能证明 learned
rollout 与真实执行动作具有足够一致性。

## 9. 在论文中的暂定表述

在正式配置冻结前，不应写：

> Planner projection off is superior.

当前只能写：

> The 100 Hz execution layer always applies a shared braking-aware physical
> projection. Planner-side population projection improves prediction--action
> consistency but currently increases end-to-end planning latency. We therefore
> evaluate planner-side projection as an implementation trade-off and report
> the resulting planner--execution action discrepancy explicitly.

最终论文是否采用 `on` 或 `off`，应由完整五-seed结果、实时余量和动作一致性指标
共同决定，而不是由单个 seed 的 tracking RMSE 决定。
