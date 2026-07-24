# H20 四种 MPC 架构对比：旧 GRU 与历史 Ideal 基线

## 实验目的与日期

实验完成于 **2026-07-24**。本实验复用历史
`cem_horizon_grid_gru/h20` 的 checkpoint、task-space reference 和 CEM seed，
比较当前实现中的四种 MPC 时序架构和 raw Direct IK，并将当前
`IdealZeroDelay` 与 2026-07-18 保存的旧同步 Ideal 结果进行配对比较。

四种方法为：

| 方法 | 实现 | 逻辑/真实延迟 |
|---|---|---:|
| `IdealZeroDelay` | `virtual_asap + full` | D=0 |
| `NaiveDelayed` | `virtual_asap + naive_delayed` | D=5 |
| `VirtualDelayAware` | `virtual_asap + full` | D=5 |
| `ThreadedAsync` | `threaded_asap + full` | 实际异步执行，补偿参数 D=5 |

## 冻结配置

| 项目 | 设置 |
|---|---|
| checkpoint | `dynamics_modeling/outputs/checkpoints/gru_20260717_152930/best_model.pt` |
| normalizer | 同目录 `normalizer.pt` |
| 轨迹 | `circle_3laps`、`ellipse_3laps`、`figure8_3laps`、`square_3laps` |
| CEM seeds | 0、1、2 |
| horizon | 20 |
| samples / batch | 128 / 128 |
| CEM iterations | 2 |
| virtual replan interval | 5 个 10 ms tick，即 20 Hz |
| planner projection | on，compiled，two-stage（方案 B） |
| 扰动 | 无 |

历史 reference 只为 H20 预留了 future points，没有额外预留 D5。延迟方法至少需要
`execution_steps + H + D + 1` 个点，因此四种方法统一执行前 **2102 steps**。这比
历史 Ideal 的 2107 steps 少最后 5 步，即 0.05 s。所有当前方法使用完全相同的
2102-step 区间，当前四方法之间仍是严格配对比较。

输入清单、标定和结果分别位于：

- `outputs/mpc/four_architectures_h20_old_gru/benchmark.json`
- `outputs/mpc/four_architectures_h20_old_gru/delay_calibration.json`
- `outputs/mpc/four_architectures_h20_old_gru/results_common_2102/`
- `outputs/mpc/four_architectures_h20_old_gru/direct_ik_common_2102/`

## 先标定 D

使用 circle 和 figure-8、真实 `threaded_asap`、相同 checkpoint、H20、128×2 和
two-stage projection 收集 500 个 planner update。结果为：

| 指标 | 数值 |
|---|---:|
| Threaded E2E P95 | 44.22 ms |
| planner guard | 5 ms |
| control tick | 10 ms |
| 标定结果 | `ceil((44.22 + 5) / 10) = D5` |

因此 Naive、Virtual 和 Threaded 均使用 D5；Ideal 固定使用 D0。

## 四种架构结果

下表为四条轨迹、三个 seed，共 12 个 rollout 的均值：

| 方法 | TCP RMSE | Joint RMSE | Orientation RMSE | Solve P95 | E2E P95 | Planner rate | Late packet |
|---|---:|---:|---:|---:|---:|---:|---:|
| `IdealZeroDelay` | **28.52 mm** | **0.01012 rad** | **1.493°** | 31.41 ms | n/a | 固定 20 Hz 逻辑调度 | 0% |
| `NaiveDelayed` | 63.47 mm | 0.05896 rad | 6.951° | 31.40 ms | n/a | 固定 20 Hz 逻辑调度 | 0% |
| `VirtualDelayAware` | 28.67 mm | 0.01014 rad | 1.506° | **31.31 ms** | n/a | 固定 20 Hz 逻辑调度 | 0% |
| `ThreadedAsync` | 29.90 mm | 0.01055 rad | 1.574° | 32.82 ms | 43.64 ms | 29.80 Hz | 1.03% |

所有 48 个 rollout 均无 planner failure、joint-limit violation、command velocity
violation 或 command acceleration violation。三个 D5 方法的平均 fallback
step rate 均为 0.238%，对应启动阶段尚无可用 packet；Ideal 为 0。

### 分轨迹 TCP RMSE

均值 ± seed 标准差：

| 轨迹 | Ideal | Naive | Virtual | Threaded |
|---|---:|---:|---:|---:|
| Circle | 38.07 ± 0.35 mm | 97.06 ± 2.32 mm | 38.06 ± 0.46 mm | 39.95 ± 0.23 mm |
| Ellipse | 25.61 ± 0.15 mm | 51.70 ± 6.58 mm | 26.08 ± 0.10 mm | 26.94 ± 0.14 mm |
| Figure-8 | 25.22 ± 0.07 mm | 53.37 ± 4.72 mm | 25.28 ± 0.13 mm | 26.57 ± 0.28 mm |
| Square | 25.19 ± 0.10 mm | 51.75 ± 5.54 mm | 25.27 ± 0.13 mm | 26.13 ± 0.07 mm |

### 配对差值

以相同 trajectory × seed 配对，并对 12 个差值进行 5000 次 bootstrap：

| 对比 | TCP RMSE 平均差值 | 95% CI |
|---|---:|---:|
| Naive − Ideal | +34.95 mm | [+27.15, +43.82] mm |
| Virtual − Ideal | +0.15 mm | [−0.11, +0.38] mm |
| Threaded − Ideal | +1.37 mm | [+1.11, +1.67] mm |
| Threaded − Virtual | +1.22 mm | [+0.94, +1.54] mm |

Virtual 与 Ideal 的区间跨 0，说明 D5 下 future alignment、execution-time
re-anchor 和 fast feedback 几乎消除了固定逻辑延迟的 tracking 代价。Naive
虽然使用相同模型、CEM budget 和 projection，但重放过期 absolute command，使
TCP RMSE 增加约 35 mm，证明主要问题是时序语义而不是 CEM 求解质量。

Threaded 相对 Virtual 平均增加 1.22 mm。它同时具有约 29.8 Hz 的实际 planner
更新率和 1.03% late-packet rate；真实 worker 调度、packet activation/drop 和
非确定性求解完成时刻共同形成了这一级别的差距。其控制线程 compute P99 仅
0.82 ms、control-period P99 为 10.15 ms，且没有控制 deadline miss。

## 与 Direct IK 比较

Direct IK 使用历史 `raw` command 口径、无 preview、同步执行，不经过 MPC
planner。它使用相同四条 reference 和共同的 2102-step 区间。Direct IK 没有 CEM
随机性，因此同一轨迹的 seeds 0/1/2 得到完全相同的结果；仍保留三个 seed 条目，
是为了和 MPC 的 trajectory × seed 索引严格配对。

| 方法 | TCP RMSE | Joint RMSE | Orientation RMSE | Control compute P99 |
|---|---:|---:|---:|---:|
| `DirectIK` | 37.11 mm | 0.01399 rad | 1.900° | **0.23 ms** |
| `IdealZeroDelay` | **28.52 mm** | **0.01012 rad** | **1.493°** | 35.16 ms |
| `NaiveDelayed` | 63.47 mm | 0.05896 rad | 6.951° | 34.44 ms |
| `VirtualDelayAware` | 28.67 mm | 0.01014 rad | 1.506° | 38.67 ms |
| `ThreadedAsync` | 29.90 mm | 0.01055 rad | 1.574° | 0.82 ms |

Virtual 方法的 control compute 包含同步阻塞的 planner，只是逻辑时间不受 wall
time 推进；只有 Direct IK 和 Threaded 的 control-compute 数值可用于真实 100 Hz
控制线程对比。

### 分轨迹 TCP RMSE

| 轨迹 | Direct IK | Ideal | Naive | Virtual | Threaded |
|---|---:|---:|---:|---:|---:|
| Circle | 53.27 mm | 38.07 mm | 97.06 mm | 38.06 mm | 39.95 mm |
| Ellipse | 32.48 mm | 25.61 mm | 51.70 mm | 26.08 mm | 26.94 mm |
| Figure-8 | 31.39 mm | 25.22 mm | 53.37 mm | 25.28 mm | 26.57 mm |
| Square | 31.30 mm | 25.19 mm | 51.75 mm | 25.27 mm | 26.13 mm |

### 相对 Direct IK 的配对差值

| 对比 | TCP RMSE 平均差值 | 95% CI | 优于 Direct 的配对数 |
|---|---:|---:|---:|
| Ideal − Direct | −8.59 mm | [−10.86, −6.47] mm | 12/12 |
| Naive − Direct | +26.36 mm | [+20.36, +32.99] mm | 0/12 |
| Virtual − Direct | −8.44 mm | [−10.76, −6.23] mm | 12/12 |
| Threaded − Direct | −7.21 mm | [−9.31, −5.27] mm | 12/12 |

因此，Ideal、Virtual 和 Threaded 的提升不只来自某一条轨迹或某一个 CEM seed：
它们在全部 12 个配对中都优于 Direct IK。Threaded 的 pooled TCP RMSE 相对
Direct IK 降低约 **19.4%**，Circle 上降低 13.32 mm，其余三条轨迹降低约
4.7–5.5 mm。

该 tracking 收益不是免费的。Direct IK 的控制计算 P99 仅 0.23 ms，也不需要
checkpoint、GPU planner、delay 标定或 packet 管理；其 command acceleration RMS
为 0.44 rad/s²，而四种 MPC 会频繁触及配置的 command acceleration projector。
因此 Direct IK 仍是更简单、更确定且命令更平滑的基线，Threaded MPC 的优势是用
额外模型和规划复杂度换取约 7.2 mm 的平均 TCP tracking 改善。

## 与 2026-07-18 旧 Ideal 数据比较

旧数据位于 `outputs/mpc/cem_horizon_grid_gru/h20/`。它同样使用 GRU
`gru_20260717_152930`、四条轨迹、seeds 0/1/2、H20 和 128×2，但属于 ASAP
引入前的同步控制循环。

| 轨迹 | 旧同步 Ideal | 当前 IdealZeroDelay | 当前 − 旧 |
|---|---:|---:|---:|
| Circle | 34.70 ± 0.58 mm | 38.07 ± 0.35 mm | +3.37 mm |
| Ellipse | 21.81 ± 0.08 mm | 25.61 ± 0.15 mm | +3.80 mm |
| Figure-8 | 21.46 ± 0.46 mm | 25.22 ± 0.07 mm | +3.76 mm |
| Square | 21.43 ± 0.12 mm | 25.19 ± 0.10 mm | +3.76 mm |
| **Pooled** | **24.85 mm** | **28.52 mm** | **+3.67 mm** |

12 个 trajectory × seed 配对差值的 bootstrap 95% CI 为
**[+3.43, +3.88] mm**。当前 Ideal 的 tracking 明确差于旧同步结果，但这不是
“相同控制器发生回归”的直接证据，因为两者的调度与 projection 语义不同：

1. 旧同步实现每个 10 ms 逻辑控制步都重新求解，相当于 **100 Hz logical replan**；
   当前四架构基准为了与 Virtual/Naive 公平比较，Ideal 使用每 5 步一次的
   **20 Hz logical replan**。
2. 旧实现使用当时的完整 candidate physical projection 路径；当前实现使用
   compiled two-stage projection。
3. 当前统一执行 2102 steps，旧结果执行 2107 steps。仅相差末尾 0.05 s，影响应小，
   但仍不是完全相同的采样区间。
4. 旧 planning mean/P95 为 45.65/46.80 ms；当前 Ideal solve P50/P95 为
   29.90/31.41 ms。当前方案明显降低单次求解耗时，但较低的 20 Hz 更新频率牺牲了
   约 3.7 mm pooled TCP RMSE。

因此，旧结果更适合作为“每步重规划、忽略计算延迟”的算法上界；当前
`IdealZeroDelay` 是四种现代架构在相同 20 Hz virtual launch schedule 下的因果
上界。二者回答的问题不同，不应只根据名称都叫 Ideal 而直接视为同一基线。

## 结论

- D5 的 VirtualDelayAware 基本恢复了当前 Ideal 的 tracking，平均只差 0.15 mm。
- NaiveDelayed 明显失败，说明延迟下直接重放 cached absolute commands 不可取。
- ThreadedAsync 相对 Virtual 有约 1.2 mm 的真实调度代价，但保持了约 29.8 Hz
  planner rate、亚毫秒级控制线程 P99 和零控制 deadline miss。
- Ideal、Virtual 和 Threaded 在全部 12 个 trajectory × seed 配对中都优于
  Direct IK；Threaded pooled TCP RMSE 低 7.21 mm，约改善 19.4%。
- Direct IK 仍具有最低的计算量和更平滑的 command，MPC tracking 收益需要用
  checkpoint、GPU planning、D 标定和异步 packet 管理换取。
- 当前 Ideal 比旧同步 Ideal 差约 3.7 mm，主要比较的是 20 Hz 与 100 Hz logical
  replanning 以及新旧 projection 路径的组合差异，不能归因于 checkpoint、轨迹、
  seed、H 或 CEM budget。
