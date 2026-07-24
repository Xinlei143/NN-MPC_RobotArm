# MPC 架构与默认配置演进记录

## 1. 文档范围与证据规则

本文按时间顺序记录 `scripts/run_cem_mpc.py` 所代表的 MPC 架构如何从基础
learned CEM-MPC 演化为当前的 H20、delay-aware、threaded asynchronous residual
MPC，并说明新增参数、现存测试结果和最终默认值的确定依据。

远端历史以 2026-07-23 检查到的
[`origin/main@2607892`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/26078927a9cea2857d6d5659ed9d57c09f96b4d8)
为基线。该 SHA 与远端 `refs/heads/main` 一致。本地部分尚未提交，因此没有可靠的
commit 时间；本文按照 2026-07-23 生成的 experiment manifest、结果文件时间和实现
依赖关系排列，单独标为“本地工作区演进”。

证据分为两类：

- **量化证据**：仓库中仍存在 CSV、JSON、manifest 或实验 README，可以复核本文数值。
- **代码证据**：提交 diff 能证明参数和行为发生了变化，但仓库中没有找到对应的独立
  消融结果。此类变化只解释设计目的，不声称性能提升。

---

## 2. 远端提交历史

### 2.1 2026-06-07：基础 learned CEM-MPC

提交：
[`542af41`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/542af41190f25803ccf0507024e67d0e5e06a0c9)
`add top-level learned CEM-MPC pipeline`

这是第一个完整的顶层闭环入口。控制器加载 MLP、GRU 或 Transformer dynamics model，
在每个控制周期内用 CEM 采样未来关节参考序列，通过 learned model rollout 后按 cost
选择动作。

初始主要参数为：

| 参数 | 初始默认值 | 作用 |
|---|---:|---|
| `horizon` | 20 | dynamics rollout 和控制优化长度 |
| `num_samples` | 1024 | 每轮 CEM 候选数 |
| `cem_iters` | 4 | CEM 更新轮数 |
| `elite_ratio` | 0.08 | elite 比例 |
| `init_std` | 0.12 | 初始采样标准差 |
| `rollout_batch_size` | 256 | model rollout batch |
| `ref_mode` | `delta` | 动作表示为参考增量 |
| `delta_base` | `previous_q_ref` | delta 的锚点 |
| `q_ref_rate_limit` | 0.08 | 参考命令变化限制 |

同时提供 `w_q`、`w_dq`、`w_u`、`w_du`、`w_terminal` 和
`w_joint_limit` 等 cost 权重。

**证据状态：仅有代码证据。** 当前仓库没有保留这组 1024 samples、4 iterations
配置与后续配置之间的独立配对实验，因此不能从提交本身得出速度或精度提升结论。

### 2.2 2026-07-04 至 07-07：输出规范和 cost 重构

提交：

- [`a16ad2d`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/a16ad2dfde0f9f11a062d7f4f7cfd34833a76b68)：
  规范 MPC 输出路径。
- [`96db182`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/96db182a6f3ff86a358f7f5388eefc4de6ff764e)：
  重构 tracking cost。

cost 开始区分状态速度、命令偏移、参考速度和参考加速度，新增
`w_u_offset`、`w_dqref`、`w_ddqref`、`velocity_cost_mode` 以及对应归一化尺度。
terminal 和 joint-limit 默认权重也被调整。

这一阶段的意义是把“跟踪目标”和“命令平滑/可执行性”拆开计价，为之后 residual
MPC 的 cost decomposition 奠定基础。

**证据状态：仅有代码证据。** 没有发现能够逐项归因到这些权重的保留消融结果。

### 2.3 2026-07-12：task-space reference、DLS IK 与 Direct IK

提交：
[`0911dbc`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/0911dbc8d9400ef19007a749c7e5a33b8d0ad3d1)
`Add task-space reference generation and DLS inverse kinematics`

入口开始支持：

- `reference_mode=task` 和固定 `reference_file`；
- 通过 `ee_site_name` 计算末端位姿；
- Damped Least Squares IK 生成 joint-space nominal；
- task position、orientation 和 IK 诊断输出。

这一步把架构从关节正弦跟踪扩展为“task-space trajectory → IK nominal → MPC
correction”，也建立了不经过 learned model 和 CEM 的 Direct IK 对照基线。

**证据状态：该提交本身只有代码证据。** Direct IK 的量化结果来自后续本地鲁棒性
工作流，见第 3.1 和 3.6 节。

### 2.4 2026-07-15：anchored residual CEM-MPC

提交：
[`7f7e49b`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/7f7e49ba6039af7c74dc44304125c99ddcaf91d2)
`Add anchored residual CEM control`

这是第一次关键架构重构。默认 action 不再是无锚点的 acceleration sequence，而是
围绕 IK nominal 的 bounded residual：

```text
task reference
  → DLS IK nominal
  → normalized residual candidates
  → nominal + bounded residual
  → learned dynamics rollout
  → CEM selection
```

新增或发生关键变化的参数包括：

| 参数组 | 主要参数 | 语义 |
|---|---|---|
| 控制器 | `controller_mode={mpc,ik_direct}` | MPC 与 Direct IK 共用入口 |
| 动作策略 | `mpc_policy={residual,legacy_acceleration}` | residual 成为默认策略 |
| residual | `residual_max` | 每关节 residual 幅值上限 |
| 命令约束 | `q_ref_velocity_limit`, `q_ref_acceleration_limit` | 规划命令的速度和加速度限制 |
| 物理限制 | `command_velocity_physical_limit`, `command_acceleration_physical_limit` | MuJoCo 执行层上限 |
| CEM | `num_samples=128`, `cem_iters=3` | 从基础版本的 1024/4 降低计算量 |
| 选择 | `cem_execute=lowest_cost` | 在 mean、best 和 residual baseline 间按 cost 选择 |
| recovery | `recovery_error_ratio`, `recovery_residual_fraction`, `recovery_*_steps` | planner failure 或持续饱和时退回 nominal |
| residual cost | `w_residual`, `w_residual_velocity`, `w_residual_acceleration`, `w_first` | residual 幅值、变化率和首步连续性 |
| cost profile | `blackbox`, `actuator_aware` | learned black-box 与 actuator-aware 代价配置 |

当时 `horizon` 暂时从 20 改为 10。控制器还强制加入 zero-residual baseline
candidate，使 CEM 至少可以选择 IK nominal，而不是被迫执行一个更差的随机候选。

**证据状态：只有架构和参数的代码证据。** 当前仓库没有保存 H10 anchored residual
与旧 acceleration MPC 的严格配对结果。

### 2.5 2026-07-19：delay-aware virtual ASAP

提交：
[`9848de2`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/9848de2a0c62826d068fc2c7644c25bf54ca0a33)
`Add delay-aware ASAP residual CEM control`

架构开始显式建模 planner latency，并恢复 `horizon=20`。新增
`multirate_mode=virtual_asap`，把 100 Hz executor 与较慢的 planner 逻辑分离：

```text
当前状态与历史
  → 预测到 packet 激活时刻
  → 对 future-aligned reference 规划 residual sequence
  → 延迟 D 步后激活 packet
  → 围绕激活时刻 IK nominal 重新锚定 residual
  → 叠加 100 Hz bounded state feedback
  → 执行命令
```

核心参数为：

| 参数 | 当时默认值 | 作用 |
|---|---:|---|
| `horizon` | 20 | future rollout 长度 |
| `replan_interval_steps` | 5 | virtual/synchronous 每次计划使用的步数 |
| `multirate_mode` | `virtual_asap` | 固定逻辑延迟的可复现实验模式 |
| `anticipation_delay_steps` | 6 | planner-to-activation 逻辑延迟 |
| `feedback_kq` | 0.30 | position feedback gain |
| `feedback_kdq` | 0.015 s | velocity feedback gain |
| `feedback_max` | 0.015 rad | feedback correction 上限 |
| `cem_iters` | 2 | 降低 planner latency |
| `mpc_warmup_plans` | 1 | 正式控制前执行 CUDA warm-up |

**证据状态：该提交本身没有独立消融结果。** 后续四协议实验验证了 future
alignment、re-anchor 和 fast feedback 合并后的效果，见第 3.5 节。

### 2.6 2026-07-19：真实 threaded ASAP 与时序对齐

提交：

- [`e192a9b`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/e192a9b3d3d487889eb3ce14dbdd4322152024ec)：
  引入真实后台 planner thread。
- [`280f704`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/280f7045b457af59a32c836054ebbf7167fb99b9)：
  修正 history、snapshot 和实时指标的相位对齐。

`threaded_asap` 不再用逻辑时间模拟 planner，而是在后台 CUDA worker 中持续求解，
executor 维持 100 Hz，并根据 activation deadline 接收或丢弃 packet。

新增参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `planner_guard_ms` | 5 ms | packet 必须早于激活时刻的安全余量 |
| `planner_min_interval_ms` | 0 ms | 相邻 planner launch 的最小间隔；0 表示 strict ASAP |
| `asap_history_mode` | `aligned` | 使用与训练数据一致的 `[x_t,u_t]` history |
| `asap_snapshot_mode` | `tick_start` | 在 control tick 开始时发布物理状态 snapshot |

同时开始记录 E2E latency、planner rate、late drop、control period、wakeup lateness 和
start jitter。只有真实 threaded 结果中的这些 wall-clock 指标具有部署含义。

**证据状态：时序修复有测试覆盖，但没有保留“修复前/修复后”完整配对结果。**

### 2.7 2026-07-21 至 07-22：backend、默认 threaded 和鲁棒性扰动

提交：

- [`8d3f8d9`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/8d3f8d93dde9a71e7622bbebdec6122d64ef2b2d)：
  新增 `dynamics_backend={learned,mujoco_oracle}` 和 Model-C 工作流。
- [`acd7e95`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/acd7e955269bc26a0ef2ff0455915be9d31a48a8)：
  将 `multirate_mode` 默认值改为 `threaded_asap`。
- [`1261604`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/126160491d240d8d58f41d82b0076954b0c12de1)：
  新增 Model-A 鲁棒性矩阵。

鲁棒性入口新增 `payload_level`、`actuator_gain_level`、`force_pulse_level` 和
`observation_noise_level`，范围均为 0–6。`mujoco_oracle` 是离线 virtual-ASAP
upper bound，不与带扰动的 learned MPC 混用。

**证据状态：代码和测试工作流证据。** 当前文档重点是架构与默认配置，不把不同
plant perturbation 的结果归因到单个控制参数。

### 2.8 2026-07-22：四种 delay protocol 和 Direct IK preview

提交：
[`e173e28`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/e173e281fb58ffd07808853056e41e61ccdc4c35)
`add reproducible delay-aware MPC paper experiments`

新增参数：

| 参数 | 选项 | 默认值 |
|---|---|---|
| `delay_protocol` | `full`, `naive_delayed`, `no_future_alignment`, `no_reanchor`, `no_feedback` | `full` |
| `ik_preview_steps` | 非负整数，仅限 Direct IK | 0 |
| `max_execution_steps` | 可选执行长度上限 | `None` |

四种正式比较结构由运行模式、协议和 D 组合而成：

| 方法 | 运行模式 | 协议 | D | 主要语义 |
|---|---|---|---:|---|
| IdealZeroDelay | virtual | full | 0 | logical zero-delay upper bound |
| NaiveDelayed | virtual | naive delayed | 标定值 | 重放过期 absolute commands |
| VirtualDelayAware | virtual | full | 标定值 | future alignment + re-anchor + feedback |
| ThreadedAsync | threaded | full | 标定值 | 真实异步 planner 与 100 Hz executor |

### 2.9 2026-07-23：统一 planner 与 execution projection 语义

提交：
[`eb0be65`](https://github.com/Xinlei143/NN-MPC_RobotArm/commit/eb0be65940ce441378b6631237f6ca7665c0f773)
`unify command projection and packet fallback semantics`

该提交把两个容易混淆的 projection 明确分开：

- **Planner projection**：决定 CEM/GRU rollout 是否基于完整 physical command
  projection 后的候选。
- **Execution projection**：100 Hz executor 对最终命令执行 joint、velocity、
  acceleration 和 braking projection；它始终开启，不受 planner 开关影响。

新增语义参数：

| 参数 | 远端默认值 | 作用 |
|---|---|---|
| `planner_projection` | `off` | 是否在 planner population 中做完整递归 projection |
| `ik_command_projection` | `raw` | Direct IK 使用 raw 或 physical command |
| `residual_cost_semantics` | `requested` | residual cost 使用 requested residual |
| `packet_residual_semantics` | `requested` | delayed packet 传输 requested residual |
| `residual_feasibility_semantics` | `finite` | 候选只要求有限 cost，而非限制 projected offset |
| `nominal_command_semantics` | `raw_ik` | nominal 保持 raw IK |

这些默认语义保留了 zero residual 与 raw Direct IK nominal 的对应关系，并确保
fallback、首包前、packet expiration 和 planner failure 都经过相同的执行层安全投影。

---

## 3. 本地工作区演进与量化结果

以下修改和实验尚未提交。所有测试使用
`dynamics_modeling/outputs/checkpoints/gru_20260717_182930`。

### 3.1 Direct IK 与四协议鲁棒性工作流

本地新增两个可恢复、带 fingerprint 的评估入口：

- `scripts/robustness/evaluate_direct_ik.py`
- `scripts/robustness/evaluate_delay_aware_mpc.py`

它们复用冻结的 task-space references，保存逐 rollout CSV/NPZ、汇总表、bootstrap
结果和图片，并支持多个 seed。Direct IK 与四种 MPC 因而可以在相同 trajectory、
condition 和 seed 下配对。

早期 virtual planning 标定得到 planning P95 = 40.98 ms：

```text
ceil((40.98 ms + 5 ms guard) / 10 ms) = 5 steps
```

这产生了 D5，但它只测 virtual/synchronous planning time，没有覆盖真实 worker
snapshot、排队和 publication 的 E2E latency，因此后来没有作为 threaded 最终值。
证据见
[`outputs/robustness/timing/gru_20260717_182930.json`](../outputs/robustness/timing/gru_20260717_182930.json)。

### 3.2 Planner projection v1：完整 projection 的延迟问题

第一轮 projection on/off 测试固定 H20、D6、两个轨迹、三个 seed，每个 rollout
500 步。此时 `on` 表示对全部 `128 × 20 × 6` candidate commands 做完整递归
physical projection。

| 模式 | Projection | TCP RMSE | Solve P95 | E2E P95 | Planner | Late | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|
| Virtual | off | 54.74 mm | 43.62 ms | n/a | n/a | 0% | 1.2% |
| Virtual | on | 55.39 mm | 59.97 ms | n/a | n/a | 0% | 1.2% |
| Threaded | off | 59.76 mm | 45.46 ms | 55.98 ms | 21.60 Hz | 12% | 1.2% |
| Threaded | on | 64.50 mm | 63.26 ms | 74.19 ms | 15.68 Hz | 100% | 100% |

Virtual 中 projection on 降低了 joint RMSE 14.9%、orientation RMSE 12.2% 和
planner–execution mismatch 73.5%，但 solve P95 增加 37.5%。真实 threaded 中，
D6 的 publication deadline 为 55 ms，而 E2E P95 达到 74.19 ms，所有 packet
都迟到，执行完全退回 fallback。

因此问题不是 projection 的数学价值，而是 H20 递归中的大量小 GPU kernel 和同步
开销。完整结果见
[`outputs/planner_projection_v1_test/README.md`](../outputs/planner_projection_v1_test/README.md)。

### 3.3 方案 A：编译完整递归 projection

本地把固定 shape 的 projection core 抽出，并增加：

```text
planner_projection_backend = eager | compiled
```

`compiled` 使用 `torch.compile(fullgraph=True)`，数学计算与 eager full projection
保持一致，仅优化执行图。真实 threaded、500-plan E2E 标定结果为：

| Variant | E2E P95 | 标定 D |
|---|---:|---:|
| projection off | 49.88 ms | 6 |
| full eager | 66.30 ms | 8 |
| full compiled | 51.61 ms | 6 |

方案 A 把 full projection 的 E2E P95 降低 14.69 ms，并把所需 D 从 8 降回 6。
在 calibrated-D tracking 测试中，full compiled 的 TCP RMSE 为 60.28 mm，
比 off 的 59.13 mm 高 1.15 mm。因此它解决了大部分实现开销，但没有成为当前测试
中的最佳精度/延迟组合。

### 3.4 方案 B：两阶段 CEM projection

本地新增：

```text
planner_projection_strategy = full | two_stage
selection_validation = none | exact_final_pool
```

`two_stage` 的数据流为：

```text
第一阶段：
全部 CEM candidates
  → cheap residual bound + joint clip
  → GRU rollout 和 approximate cost

第二阶段：
final elites + approximate best + mean + zero-residual baseline
  → 去重
  → exact physical projection
  → 重新 GRU rollout 和 exact cost
  → 只从 exact-validated pool 中选择
```

H20 实验中每个 plan 实际精确验证约 11–13 条唯一候选；exact re-evaluation 在
49.6% 的成功计划中改变了 approximate selection。

| Variant | TCP RMSE | Joint RMSE | Solve P95 | E2E P95 | Planner | 标定 D |
|---|---:|---:|---:|---:|---:|---:|
| off | 59.13 mm | 0.02376 rad | 39.63 ms | 50.84 ms | 24.55 Hz | 6 |
| full eager | 60.47 mm | 0.02425 rad | 53.34 ms | 64.81 ms | 18.13 Hz | 8 |
| full compiled | 60.28 mm | 0.02330 rad | 39.97 ms | 50.86 ms | 24.43 Hz | 6 |
| two-stage compiled | **57.57 mm** | **0.02170 rad** | **33.35 ms** | **44.44 ms** | **28.87 Hz** | **6** |

相对 projection off，方案 B 的 TCP RMSE 改善 1.56 mm，真实 threaded E2E P95
降低约 6.4 ms。在 common-D8 比较中它仍取得最低 TCP RMSE 56.85 mm 和最低 E2E
P95 44.46 ms，说明优势不只是来自选择了更小的 D。

完整证据见
[`outputs/planner_projection_h20_optimization/README.md`](../outputs/planner_projection_h20_optimization/README.md)
和
[`evaluation manifest`](../outputs/planner_projection_h20_optimization/evaluation/experiment_manifest.json)。

### 3.5 H20、方案 B、D6 下的四种 MPC

固定配置：

```text
horizon = 20
planner_projection = on
planner_projection_backend = compiled
planner_projection_strategy = two_stage
anticipation_delay_steps = 6
```

使用 circle、figure-8、三个 seed、无扰动、每次 500 步，共 24 个 rollout：

| 方法 | TCP RMSE | Joint RMSE | Orientation RMSE | Solve P95 | E2E P95 | Late |
|---|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | 55.46 mm | 0.02134 rad | 2.763° | 31.88 ms | n/a | 0% |
| NaiveDelayed | 122.81 mm | 0.15951 rad | 22.207° | 31.98 ms | n/a | 0% |
| VirtualDelayAware | 56.18 mm | **0.02105 rad** | 2.831° | 32.56 ms | n/a | 0% |
| ThreadedAsync | 55.83 mm | 0.02161 rad | 2.764° | 33.22 ms | 44.19 ms | 0% |

配对 TCP 差值：

- Virtual − Ideal：+0.72 mm，95% bootstrap CI `[-0.30, +1.61]` mm；
- Threaded − Ideal：+0.38 mm，CI `[-0.85, +1.84]` mm；
- Naive − Ideal：+67.36 mm，CI `[+57.86, +76.47]` mm。

在这个六配对 screening test 中，Ideal、Virtual 和 Threaded 的 tracking
统计上无法区分；Threaded 没有 planner failure、late packet 或命令约束违规。
Naive 明显更差，因为它重放过期 absolute commands，没有 future alignment、
residual re-anchor 或 fast feedback。

证据见
[`outputs/mpc_structures_h20_two_stage_test/README.md`](../outputs/mpc_structures_h20_two_stage_test/README.md)。

### 3.6 与 Direct IK 对比

同样使用两个轨迹、三个 seed、无扰动和 500 步：

| 方法 | TCP RMSE | 相对 Direct IK | Joint RMSE | Orientation RMSE |
|---|---:|---:|---:|---:|
| Direct IK | 64.50 mm | — | 0.02832 rad | 3.129° |
| IdealZeroDelay | 55.46 mm | -14.0% | 0.02134 rad | 2.763° |
| NaiveDelayed | 122.81 mm | +90.4% | 0.15951 rad | 22.207° |
| VirtualDelayAware | 56.18 mm | -12.9% | 0.02105 rad | 2.831° |
| ThreadedAsync | 55.83 mm | -13.4% | 0.02161 rad | 2.764° |

Ideal、Virtual 和 Threaded 在所有配对实验中均优于 Direct IK。Direct IK 的优势是
计算开销：control compute P99 约 0.2 ms，而方案 B MPC solve P95 约 32–33 ms，
Threaded E2E P95 为 44.19 ms。

配对数据见
[`mpc_vs_direct_ik_summary.csv`](../outputs/mpc_structures_h20_two_stage_test/mpc_vs_direct_ik_summary.csv)
和
[`mpc_vs_direct_ik_paired.json`](../outputs/mpc_structures_h20_two_stage_test/mpc_vs_direct_ik_paired.json)。

### 3.7 为什么最终固定 H20，而不是 H25

远端基础 CEM-MPC 最初就是 H20；anchored residual 阶段曾短暂使用 H10，delay-aware
ASAP 又恢复为 H20。本地 robustness 和 Model-C 辅助脚本此前残留 H25 默认值，容易
让 reference padding、delay calibration 和正式 controller 使用不同 horizon。

当前工作区已把主 runner、robustness reference/manifest/calibration 和 Model-C 的
MPC horizon 默认值统一为 20。`branch_horizon=25` 仍保留，因为它是 Model-C 数据
分支标签长度，不是 MPC rollout horizon。

需要严格说明：仓库中没有 H20 与 H25、相同轨迹和 seed 的完整配对 sweep。因此
H20 的确定依据是：

1. 与主 runner 和 delay-aware 架构历史保持一致；
2. projection 方案 A/B、真实 E2E 标定和四结构对比全部在 H20 下完成；
3. 避免不同工作流静默使用不同 horizon；
4. H20 已在当前测试中满足 tracking、实时性和安全要求。

所以 H20 是**经过当前完整证据链验证的统一默认配置**，不是已经证明的全局最优
horizon。

---

## 4. 当前默认配置

当前 `scripts/run_cem_mpc.py` 的核心默认配置如下：

| 类别 | 参数 | 当前默认值 | 确定依据 |
|---|---|---|---|
| 控制架构 | `controller_mode` | `mpc` | learned MPC 主入口 |
| MPC action | `mpc_policy` | `residual` | IK nominal + bounded correction |
| dynamics | `dynamics_backend` | `learned` | 在线主控制器 |
| 调度 | `multirate_mode` | `threaded_asap` | 真实 100 Hz executor + 后台 planner |
| horizon | `horizon` | **20** | 所有最终标定和对比统一使用 H20 |
| CEM | `num_samples` | 128 | 当前实时预算 |
| CEM | `cem_iters` | 2 | 当前实时预算 |
| CEM | `rollout_batch_size` | 128 | 与 population 对齐 |
| virtual replanning | `replan_interval_steps` | 5 | virtual/synchronous 使用；threaded 完成即重规划 |
| delay | `anticipation_delay_steps` | **6** | two-stage 真实 E2E P95 45.81 ms + 5 ms guard |
| delay semantics | `delay_protocol` | `full` | future alignment + re-anchor + feedback |
| projection | `planner_projection` | **`on`** | planner 与 execution 更一致 |
| projection backend | `planner_projection_backend` | **`compiled`** | 避免 eager 小 kernel 开销 |
| projection strategy | `planner_projection_strategy` | **`two_stage`** | 当前精度、延迟和 planner rate 最佳 |
| nominal | `nominal_command_semantics` | `raw_ik` | 保持 zero residual 的基准语义 |
| residual cost | `residual_cost_semantics` | `requested` | cost 与请求的 MPC correction 对应 |
| delayed packet | `packet_residual_semantics` | `requested` | 激活时重新锚定 |
| feasibility | `residual_feasibility_semantics` | `finite` | 避免旧 projected-offset bound 混淆 |
| fast feedback | `feedback_kq` | 0.30 | delay-aware tube correction |
| fast feedback | `feedback_kdq` | 0.015 s | velocity correction |
| fast feedback | `feedback_max` | 0.015 rad | correction bound |
| deadline | `planner_guard_ms` | 5 ms | D 标定和 late-drop 安全余量 |
| Direct IK | `ik_preview_steps` | 0 | 无预览基线 |
| Direct IK | `ik_command_projection` | `raw` | 历史 Direct IK 基线 |

方案 B 对 IdealZeroDelay、NaiveDelayed、VirtualDelayAware 和 ThreadedAsync 四种
learned residual MPC 都有影响，因为它们共享 CEM planner 和
`LearnedDynamicsPlanner`。它们的差别是 delay、packet 和 executor 语义，而不是使用
不同的 candidate projection。Direct IK 不经过 CEM planner，因此不读取方案 B；
它只受独立的 `ik_command_projection` 控制。

执行层 physical projection 始终开启。把 planner projection 设为 `off` 并不会关闭
最终命令的 joint、velocity、acceleration 和 braking 安全约束。

这些值是默认配置，不是删除消融能力。命令行仍可显式选择 `synchronous`、
`virtual_asap`、projection `off`、`full/eager` 或其他 delay protocol，以复现实验和
诊断回归。

---

## 5. 默认值决策链总结

最终配置不是由单个 RMSE 数字决定，而是依次排除了几个问题：

1. 基础 CEM-MPC 缺少 task-space nominal 和显式 planner latency；
2. anchored residual 把 IK 作为安全、可解释的 baseline；
3. delay-aware packet 引入 future alignment、re-anchor 和 100 Hz feedback；
4. threaded ASAP 暴露真实 E2E latency 和 late packet 问题；
5. 完整 planner projection 改善语义一致性，但 eager 实现使 D6 deadline 失效；
6. 方案 A 用 compile 把 full projection 从 D8 优化到 D6；
7. 方案 B 只对 final pool 做 exact validation，同时取得更低 E2E latency 和更好
   tracking；
8. 真实 E2E 标定固定 D6；
9. 四结构和 Direct IK 配对实验确认 H20 + two-stage/compiled 下的 Virtual 与
   Threaded 接近 Ideal，且明显优于 Direct IK；
10. 因而把 **H20、D6、threaded ASAP、planner projection on、
    two-stage/compiled** 固定为当前默认组合。

后续如果模型、GPU、candidate 数、CEM iterations 或 control period 改变，D 必须
重新通过真实 threaded E2E 标定，不能直接沿用 6。若要改变 horizon，也应重新执行
projection latency 标定、四结构配对和 Direct IK 对照，而不是只比较单次 solve time。
