# ASAP Residual CEM-MPC：计算延迟、多速率执行与实验结果

## 1. 结论与适用范围

本文记录本项目针对 GRU residual CEM-MPC 计算延迟实现的 **ASAP 风格多速率控制器**，并与旧的“低频 CEM + 缓存绝对命令”GRU-MPC 对照。

在相同圆轨迹、GRU checkpoint 和 `128 samples × 2 CEM iterations` 的条件下：

| 对照 | TCP RMSE | CEM planning p95 | 结果 |
|---|---:|---:|---|
| 旧 GRU H20@20 Hz | 71.6 ± 23.2 mm | 47.4 ± 0.1 ms | 低频缓存绝对命令，劣于 Direct IK |
| **ASAP H20@20 Hz** | **38.5 ± 0.3 mm** | **32.0 ± 0.3 ms** | 默认配置；优于 Direct IK |
| 旧 GRU H30@10 Hz | 79.1 ± 19.0 mm | 67.0 ± 2.2 ms | 低频缓存绝对命令，劣于 Direct IK |
| **ASAP H30@10 Hz** | **37.8 ± 0.2 mm** | **44.1 ± 0.7 ms** | 明显改善，但更新频率较低 |
| Direct IK | 53.2 mm | — | 无模型、100 Hz nominal 基线 |

ASAP 的误差收益来自三项结构变化：**在计划真正生效的未来状态处优化、缓存 residual 而不是过时的绝对 `q_ref`、以及在每个 10 ms 控制步使用当前状态作小幅反馈修正**。这使慢速 CEM 只负责预测性补偿，快速层持续保留 Direct IK 的实时跟踪能力。

本文中的实现名为 `virtual_asap`：CEM 仍在单进程中同步测时，但其结果被安排在固定的未来控制步才可生效。这能可重复地验证计算延迟补偿逻辑，**并不等同于**论文中的后台线程/进程异步求解、真实通信时延或真机实时调度。

相关方法学参考：Dirckx 等提出的 [ASAP-MPC](https://arxiv.org/abs/2402.06263) 将慢速最优轨迹生成与快速线性状态反馈结合，并在新解完成后异步更新、平滑衔接轨迹。本文实现借鉴其“慢规划 + 快反馈 + 未来状态锚定”的原则，但尚未实现真实后台求解和连续轨迹拼接。

---

## 2. 系统、基线与时间尺度

### 2.1 机器人和接口

对象为 6 关节 ABB IRB 2400 MuJoCo 模型：

\[
x_t=[q_t,\dot q_t]\in\mathbb R^{12},\qquad u_t=q_{\mathrm{ref},t}\in\mathbb R^6.
\]

上层命令是 position actuator 的关节位置目标，而不是力矩命令。GRU 预测闭环离散转移：

\[
\hat x_{t+1}=f_\theta(\mathcal H_t,q_{\mathrm{ref},t}),
\]

其中历史 \(\mathcal H_t\) 含 16 个 `[q, dq, q_ref]` token。checkpoint 为 `gru_20260717_152930`，GRU hidden size 为 256，模型与控制时间步均为 10 ms。

| 层次 | 周期/频率 | 职责 |
|---|---:|---|
| MuJoCo 物理积分 | 2 ms / 500 Hz | position actuator、阻尼和重力补偿 |
| 快速命令层 | 10 ms / 100 Hz | 读取状态、重建 nominal、反馈、投影与下发 `q_ref` |
| GRU 动力学模型 | 10 ms / 100 Hz | CEM rollout 的状态预测粒度 |
| CEM 重规划层 | 10–20 Hz（本实验） | 预测 future residual 序列 |

本项目没有单独手写的底层 PD 循环；MuJoCo XML 的 position actuator 以 `kp` 和 `dampratio=1` 在 500 Hz 物理积分内实现位置伺服。

### 2.2 Direct IK 基线

任务空间轨迹经连续 DLS IK 生成 `q_des, dq_des`。Direct IK 每个 10 ms 直接发送后继目标：

\[
q_{\mathrm{ref},t}=q_{\mathrm{des},t+1}.
\]

它保留执行器跟踪滞后，但没有模型预测或 CEM 产生的修正。因此它也是 residual MPC 的安全零修正基线。

---

## 3. 旧式低频 GRU-MPC：问题是什么

旧多速率方案在时刻 \(t\) 从实测状态求解完整绝对命令序列，并连续执行前 \(K\) 项：

\[
q^{\mathrm{plan}}_{\mathrm{ref},t:t+K-1}.
\]

```text
t = 0 ms:   read x_t -> CEM(x_t) -> cache absolute q_ref[0:K]
t = 10 ms:  send cached q_ref[1]
...
t = K×10 ms: read latest state -> solve next CEM plan
```

即使命令以 100 Hz 逐项发送，缓存中的绝对 `q_ref` 仍由旧状态、旧 nominal 和旧参考窗口生成。缓存期间真实状态、参考轨迹和前一实际命令继续变化，而命令不能重新对齐。对有执行器滞后和模型偏差的系统，这会造成：

1. 预测状态与真实状态偏离；
2. residual 补偿以过时的绝对命令形式继续执行；
3. 新计划直到下一个低频重规划点才利用实际反馈；
4. 当缓存误差大于 CEM 的预测补偿收益时，性能会低于 Direct IK。

旧实验的 H20@20 Hz 和 H30@10 Hz 正是这一问题的直接证据：两者均无 planner failure，却分别得到 71.6 mm 和 79.1 mm TCP RMSE，均差于 53.2 mm 的 Direct IK。

---

## 4. ASAP residual MPC 的具体实现

### 4.1 设计原则

控制器将慢速部分限定为“生成未来 correction”，将快速部分限定为“用最新状态安全地执行 correction”。CEM 不再拥有完整绝对命令的长期控制权。

一次 CEM 计划返回：

- future residual 序列 \(r^{\mathrm{plan}}_{t_a:t_a+H-1}\)；
- 对应的预测状态序列 \(\hat x^{\mathrm{plan}}_{t_a:t_a+H}\)；
- 计划开始时刻 \(t_a\) 与测得的 CEM 墙钟耗时。

其中 \(t_a=t+D\)，\(D\) 是按 10 ms 控制步表示的预期计算延迟。默认 H20@20 Hz 使用 `D=6`（60 ms）；已有 H30@10 Hz 实验使用 `D=7`。

### 4.2 未来状态锚定与计划 packet

在时刻 \(t\) 发起规划时，系统预期仍会执行未来 \(D\) 步已有命令。运行器先用 GRU 将当前历史向前 rollout：

\[
\hat x_{t+D\mid t}=f_\theta^{(D)}\left(\mathcal H_t,
q_{\mathrm{ref},t:t+D-1}^{\mathrm{active}}\right).
\]

然后 CEM 从 \(\hat x_{t+D\mid t}\) 开始优化，并把 `q_des`、`dq_des` 的窗口整体平移到 \(t+D\)：

\[
r^{\mathrm{plan}}_{t+D:t+D+H-1}=
\operatorname{CEM}\left(
\hat x_{t+D\mid t},q_{\mathrm{des},t+D+1:t+D+H}
\right).
\]

结果保存在 `DelayedPlanPacket` 中，只有到达 activation step \(t+D\) 时才成为 active packet。若 `planning_time > D\Delta t`，该 packet 被标记为 `late_plan_dropped`，控制器继续使用已有 packet 或零 residual nominal。

```text
launch at t
  -> forecast D active commands
  -> optimize a residual plan anchored at t+D
  -> schedule packet for t+D

every 10 ms
  -> activate due packet
  -> read current x
  -> use residual and predicted state at packet age
  -> feedback + projection + env.step(q_ref)
```

当前 `virtual_asap` 按 `replan_interval_steps` 的固定周期启动新 CEM，而非论文中“上一求解一结束就立即启动下一求解”。该差异必须在解释结果时保留。

### 4.3 100 Hz residual 重锚定与快速反馈

每个 10 ms 控制步重新读取实测状态，并根据当前 IK 参考构造 nominal：

\[
q_{\mathrm{nom},k}=q_{\mathrm{des},k+1}.
\]

若 active packet 存在，取其当前年龄对应的 residual \(r^{\mathrm{plan}}_k\) 与预测状态 \(\hat x_k=[\hat q_k,\hat{\dot q}_k]\)。快速反馈为：

\[
\delta r_k=
\operatorname{clip}\left(
K_q(\hat q_k-q_k)+K_{\dot q}(\hat{\dot q}_k-\dot q_k),
-\delta r_{\max},\delta r_{\max}
\right).
\]

本实验默认 `feedback_kq=0.30`、`feedback_kdq=0.015 s`、`feedback_max=0.015 rad`。最终下发命令为：

\[
q_{\mathrm{ref},k}=
\Pi\left(q_{\mathrm{nom},k}+r^{\mathrm{plan}}_k+\delta r_k\right),
\]

其中 \(\Pi\) 对非零 correction 以**上一条实际下发命令**为约束初值，重新执行关节范围、速度和加速度投影。零 correction 则直接发送 joint-limit-clipped nominal，从而保持 Direct IK 回退的精确语义，而不继承陈旧 MPC 命令。

若 packet 尚未激活、过期、求解失败或 residual 不可行，则令 \(r=\delta r=0\)。此时命令退化为 Direct IK nominal；不会继续发送一段旧绝对命令。

### 4.4 与旧方案的差异

| 项目 | 旧低频 GRU-MPC | ASAP residual MPC |
|---|---|---|
| 缓存对象 | 绝对 `q_ref` | residual 序列和预测状态 |
| 计划初始状态 | 当前测量 `x_t` | 预计生效时的 `x̂_{t+D|t}` |
| 快速层状态反馈 | 无 | 100 Hz `Kq/Kdq` 有界反馈 |
| nominal | 规划时固定 | 每 10 ms 根据当前参考重建 |
| 约束投影 | CEM population 内的完整序列投影 | 非零 correction 在执行层逐步投影；零 correction 精确回退 Direct IK |
| 无有效计划时 | 继续缓存/等待重规划 | 立即退化为 Direct IK nominal |

### 4.5 CEM 与可行性边界

residual CEM 仍在 \([-1,1]\) 归一化 residual 空间中采样，并使用 `lowest_cost` 比较 baseline、best sample 与 CEM mean。residual 上限保持：

```text
[0.12, 0.10, 0.12, 0.15, 0.15, 0.20] rad
```

ASAP 路径将“全候选、全 horizon 的速度/加速度投影”移出 CEM population rollout；CEM 内部对候选进行 residual 限幅和关节范围裁剪，而 100 Hz 执行层再对实际将要下发的一步做物理速度/加速度投影。这样减少了 CEM 每次 rollout 的张量操作，但它也是与旧方案不同的控制实现。

---

## 5. 计时口径：什么叫“规划更快”

| 指标 | 定义 | 是否包含未来锚点预测 |
|---|---|---|
| `planning_time` / `replan_time` | `CEMMPCController.plan()` 的墙钟耗时 | 否 |
| planning mean / p95 | 只在实际 CEM 重规划步统计 | 否 |
| `control_step_wall_time` | 一个 10 ms 快速控制步的完整脚本耗时 | 是；还含命令投影、MuJoCo、诊断和日志 |
| control p95 | 所有 100 Hz 控制步的 wall time P95 | 是 |
| deadline | `planning_time > D × 10 ms` | 使用 CEM 计时，不是完整端到端时间 |

因此，ASAP 的 32 ms planning p95 表示 CEM 内部优化时间，而非新的 packet 从测量到可下发的完整端到端延迟。完整控制 p95 应同时报告，例如 H20@20 Hz 为 35.7 ms。

此外，ASAP 不会仅因“降低重规划频率”而让一次 CEM 求解变快。当前结果中的 planning-time 优势来自以下可观测实现条件的组合：

1. H20/H25/H30 及 `128×2` 的计算预算本身不同；
2. candidate sequence 的速度/加速度投影从 population rollout 转移到逐步执行层；
3. 同一 GPU、CUDA 状态和代码路径下重新测量，且每个 rollout 丢弃一次 CUDA warm-up plan；
4. future-anchor rollout 不计入 `planning_time`，但计入 `control_step_wall_time`。

因此应表述为：**当前 ASAP 实现在所测配置下同时取得更低的 CEM 计时和更低的跟踪误差**；不能将其简化为“ASAP 理论上必然降低 CEM 求解复杂度”。

---

## 6. 实验设置

除 Direct IK 外，每个配置均使用 3 个 CEM seed（0、1、2）：

| 项目 | 设置 |
|---|---|
| 轨迹 | 三圈 task-space circle |
| checkpoint | `dynamics_modeling/outputs/checkpoints/gru_20260717_152930/best_model.pt` |
| normalizer | 同目录 `normalizer.pt` |
| 模型时间步 | 10 ms |
| MPC policy | residual + `lowest_cost` |
| CEM budget | 128 samples × 2 iterations，batch 128 |
| CEM exploration | elite ratio 0.08，uniform ratio 0.15 |
| 快速层 | 100 Hz；MuJoCo position actuator 500 Hz |
| ASAP delay | H20/H25 与 H30 的 14.29–20 Hz 为 D=6；H30@10 Hz 为 D=7 |

默认配置为 **ASAP H20@20 Hz**：`H=20`、每 5 个 10 ms 步重规划、`D=6`。

```bash
python scripts/run_cem_mpc.py \
  --model_type gru \
  --checkpoint dynamics_modeling/outputs/checkpoints/gru_20260717_152930/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints/gru_20260717_152930/normalizer.pt \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --horizon 20 --replan_interval_steps 5 \
  --multirate_mode virtual_asap --anticipation_delay_steps 6 \
  --num_samples 128 --rollout_batch_size 128 --cem_iters 2 \
  --device cuda --mpc_policy residual --cem_execute lowest_cost \
  --save_dir outputs/mpc/asap_h20_20hz_circle
```

---

## 7. ASAP 完整 horizon × 频率网格（圆轨迹）

下表为 3 seed 的均值 ± 标准差。planning 一列为 `mean / p95`，control 一列为 100 Hz 全部控制步的 wall-time P95。误差均相对相同的 IK 参考计算。

| 配置 | Joint RMSE (rad) | TCP RMSE (mm) | Orientation RMSE (deg) | Planning mean / p95 (ms) | Control p95 (ms) |
|---|---:|---:|---:|---:|---:|
| H20 @ 10 Hz | 0.01660 ± 0.00096 | 43.21 ± 1.51 | 2.31 ± 0.12 | 32.1 ± 0.9 / 33.5 ± 1.7 | 36.0 ± 0.7 |
| H20 @ 14.29 Hz | 0.01514 ± 0.00005 | 40.13 ± 0.10 | 2.12 ± 0.00 | 31.9 ± 1.1 / 32.7 ± 1.1 | 36.4 ± 1.1 |
| H20 @ 16.67 Hz | 0.01478 ± 0.00015 | 39.41 ± 0.38 | 2.08 ± 0.02 | 31.4 ± 0.1 / 32.3 ± 0.3 | 36.2 ± 0.3 |
| **H20 @ 20 Hz** | **0.01453 ± 0.00009** | **38.53 ± 0.28** | **2.01 ± 0.01** | **31.2 ± 0.3 / 32.0 ± 0.3** | **35.7 ± 0.3** |
| H25 @ 10 Hz | 0.01569 ± 0.00086 | 40.78 ± 1.49 | 2.18 ± 0.13 | 37.5 ± 0.4 / 38.9 ± 0.4 | 41.5 ± 0.5 |
| H25 @ 14.29 Hz | 0.01416 ± 0.00006 | 37.19 ± 0.14 | 1.97 ± 0.01 | 37.4 ± 0.2 / 38.3 ± 0.4 | 41.8 ± 0.1 |
| H25 @ 16.67 Hz | 0.01370 ± 0.00006 | 36.14 ± 0.12 | 1.88 ± 0.01 | 37.4 ± 0.1 / 38.5 ± 0.3 | 41.9 ± 0.2 |
| H25 @ 20 Hz | 0.01762 ± 0.00287 | 42.42 ± 4.99 | 2.44 ± 0.39 | 37.9 ± 0.7 / 38.8 ± 0.7 | 42.6 ± 0.7 |
| H30 @ 10 Hz | 0.01439 ± 0.00004 | 37.77 ± 0.24 | 2.00 ± 0.02 | 43.3 ± 0.7 / 44.1 ± 0.7 | 47.6 ± 0.7 |
| H30 @ 14.29 Hz | 0.01441 ± 0.00137 | 36.32 ± 2.33 | 1.98 ± 0.20 | 43.2 ± 0.6 / 44.6 ± 1.2 | 47.5 ± 0.6 |
| H30 @ 16.67 Hz | 0.01401 ± 0.00158 | 35.55 ± 2.27 | 1.94 ± 0.21 | 44.2 ± 1.2 / 45.6 ± 1.5 | 48.8 ± 1.3 |
| H30 @ 20 Hz | 0.01373 ± 0.00150 | 34.21 ± 2.20 | 1.88 ± 0.20 | 43.4 ± 0.5 / 44.3 ± 0.5 | 48.1 ± 0.6 |

### 7.1 网格结论

- **默认 H20@20 Hz** 的计划 p95 为 32.0 ms、控制 p95 为 35.7 ms，低于 50 ms 重规划周期，且有较小的 seed 方差，是当前实时裕度和精度最平衡的配置。
- H25@16.67 Hz 的圆轨迹误差更低，但曾出现超过 60 ms 周期的端到端单次尾部值；它适合作为精度候选，不应按严格 16.67 Hz 实时配置部署。
- H25@20 Hz 的误差和跨 seed 方差明显变差，说明该 horizon 在 20 Hz 下缺少足够调度/模型鲁棒性。
- H30@20 Hz 的圆轨迹平均 TCP RMSE 最低，但 control p95 已接近 50 ms，且跨 seed 波动大于 H20；它没有 H20@20 Hz 的实时裕度。

---

## 8. 为什么 ASAP 同时改善误差和当前计时

### 8.1 误差改善的因果链

1. **消除时间错位。** 新 residual 计划从预计生效时的 \(\hat x_{t+D|t}\) 和未来参考开始，而不是从已经过去的 \(x_t\) 开始。
2. **保留 nominal 的实时性。** 每个 10 ms 都根据当前 `q_des` 重建 `q_nom`；即使 CEM packet 已有数十毫秒历史，命令不会整体滞后于 IK 参考。
3. **快速反馈闭环模型偏差。** \(K_q/K_{\dot q}\) 项用最新测量压缩预测轨迹与真实轨迹的偏差，而不是盲目执行计划前缀。
4. **安全回退不会传播旧计划。** packet 不可用时 residual 为零，系统立即等价于 Direct IK，而不是继续使用过时绝对命令。
5. **非零修正按实际历史投影。** 速度、加速度和 joint limit 以实际前一命令为基准重算；零修正则保持 Direct IK nominal，避免缓存命令与当前执行历史不一致。

这解释了为何 ASAP H20@20 Hz 将旧 H20@20 Hz 的 71.6 mm TCP RMSE 降至 38.5 mm，并超过 Direct IK 的 53.2 mm。

### 8.2 当前 planning-time 改善的原因与边界

旧 H20@20 Hz 的 planning p95 为 47.4 ms，当前 ASAP H20@20 Hz 为 32.0 ms；旧 H30@10 Hz 为 67.0 ms，当前 ASAP H30@10 Hz 为 44.1 ms。该差异不能只归因于“异步”。

- ASAP 把对每个 CEM candidate、每个 horizon step 的速度/加速度投影移出 population rollout，减少了 CEM 内部张量计算；
- 每 10 ms 的实际单命令投影仍执行，因此这是一种**计算位置迁移**，不是取消约束；
- future-anchor rollout 不计入 `planning_time`，但包含在 control time；
- H、CEM budget、CUDA 预热和代码路径也会影响测得时间。

所以合理结论是：在当前实现与计时定义下，ASAP 配置同时取得更低 CEM planning time 和更低 tracking error；如需证明纯算法加速，应在完全一致的 candidate projection、horizon、budget、设备状态和计时边界下做额外消融。

---

## 9. 实现定位、数据与限制

| 内容 | 位置 |
|---|---|
| CLI、默认 H20@20 Hz ASAP 配置 | `scripts/run_cem_mpc.py` |
| 未来锚定 packet 调度、反馈与快速执行 | `mpc/delay_aware_runner.py` |
| packet、反馈和单步重锚定投影 | `mpc/delay_aware.py` |
| residual sequence 构造和 CEM rollout | `mpc/planner_rollout.py`, `mpc/cem_controller.py` |
| ASAP 圆轨迹频率网格 | `outputs/mpc/delay_aware/horizon_frequency_grid_circle/` |
| H30 ASAP 频率实验 | `outputs/mpc/delay_aware/virtual_asap_h30*/` |
| H25@16.67 Hz 全轨迹实验 | `outputs/mpc/delay_aware/h25_16p7hz_d6_all_tasks/` |
| 旧低频 GRU 对照 | `outputs/mpc/multirate_gru_circle/` |

本结果只覆盖进程内 MuJoCo、当前 GPU、当前 GRU checkpoint 和所列轨迹。真实 ASAP 部署仍需实现独立 planner worker、线程安全的 packet 发布、时钟/通信延迟测量、实际末端状态估计、packet 超时策略以及真机安全约束验证。
