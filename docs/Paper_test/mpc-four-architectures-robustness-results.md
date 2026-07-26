# 四种 Delay-Aware MPC 架构鲁棒性实验结果

> **测试完成日期：2026-07-26（Asia/Shanghai）**。汇总 CSV 的最后修改时间为
> `2026-07-26 00:25:13 +0800`。

## 1. 测试对象与冻结配置

本测试完成 **720/720** 个 learned-MPC rollout，结果位于：

```text
outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/
```

测试了四种 MPC 架构/延迟协议：

| 名称 | Planner 模式 | 延迟协议 | D (10 ms steps) | 含义 |
|---|---|---|---:|---|
| IdealZeroDelay | `virtual_asap` | `full` | 0 | 相同 CEM budget 下的零逻辑延迟上界，不代表 100 Hz CEM |
| NaiveDelayed | `virtual_asap` | `naive_delayed` | 6 | launch-time state/reference，激活后重放绝对命令，无反馈 |
| VirtualDelayAware | `virtual_asap` | `full` | 6 | 确定性固定延迟下的完整 future alignment、re-anchor 与 feedback |
| ThreadedAsync | `threaded_asap` | `full` | 6 | 真实异步 GPU planner 下的完整方法 |

所有 rollout 使用同一冻结配置：

| 配置 | 值 |
|---|---|
| Dynamics model | GRU，history length 16 |
| Checkpoint | `dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt` |
| Checkpoint SHA-256 | `9b7f65384f892bde1b4465e7d0795d1dafc806cf7bcb4bf7fb0ff965165684e5` |
| Horizon / CEM | H=20；128 candidates；2 CEM iterations；batch=128 |
| Replanning | 每 5 个 10 ms tick（20 Hz） |
| Cost | exact task-space cost=on |
| Stage-one GPU FK | **off**；population CEM 仅使用 joint/base cost，MuJoCo FK 仅用于 exact final pool |
| Planner projection | on；backend=`compiled`；strategy=`two_stage` |
| Feedback gains | Kq=0.3，Kdq=0.015 |
| 不确定性感知 | **off**；未加载 ensemble 或 uncertainty gate |
| 延迟标定 | `delay.json`，`D_cal=6` |

每个 rollout 的 fingerprint 均记录上述参数、checkpoint/normalizer/reference hash、plant XML
和扰动等级。当前报告不混入 Direct IK rollout，也不使用不确定性感知或模型集成。

## 2. 实验矩阵与扰动

矩阵为：

```text
4 MPC 架构 × 4 条论文轨迹 × 9 个单因素条件 × 5 个 seed = 720
```

四条冻结轨迹为 `circle`、`figure8`、`fast_ellipse`、`rounded_square`。每个“方法 ×
条件 × 轨迹”有 5 个 seed。除非另有说明，下表中的数值都是这 5 个 seed 的算术平均；
**不再将四条轨迹先 pooled 后作为唯一主结果**。

| 条件 | L3 | L6 |
|---|---:|---:|
| Payload | 4 kg | 12 kg |
| Actuator gain | Kp=0.70，Kd=0.837 | Kp=0.30，Kd=0.548 |
| Force pulse | 200 N，+Y，持续 0.10 s | 500 N，+Y，持续 0.10 s |
| Observation noise | q=0.001 rad，dq=0.010 rad/s | q=0.005 rad，dq=0.050 rad/s |

所有条件均为单因素；nominal 中四类扰动均为 L0。

## 3. 每条轨迹的架构对比：TCP 位置 RMSE

单位均为 mm，越小越好。每张表仅在该轨迹的五个 seed 内平均，因此可直接看出不同
reference 的难度和各架构的相对表现。

### 3.1 Circle

| 条件 | IdealZeroDelay | NaiveDelayed | VirtualDelayAware | ThreadedAsync |
|---|---:|---:|---:|---:|
| Nominal | **27.44** | 69.39 | 29.29 | 28.40 |
| Payload L3 | 49.10 | 76.31 | 45.41 | **43.44** |
| Payload L6 | 90.01 | 95.96 | **88.30** | 91.10 |
| Gain L3 | **30.12** | 72.63 | 31.17 | 36.69 |
| Gain L6 | **44.30** | 86.10 | 46.42 | 49.82 |
| Force L3 | **26.95** | 71.38 | 29.02 | 28.67 |
| Force L6 | 30.84 | 73.71 | 32.59 | **28.69** |
| Observation noise L3 | 29.05 | 71.29 | **28.58** | 33.21 |
| Observation noise L6 | 35.47 | 73.38 | 35.34 | **34.89** |

### 3.2 Figure-8

| 条件 | IdealZeroDelay | NaiveDelayed | VirtualDelayAware | ThreadedAsync |
|---|---:|---:|---:|---:|
| Nominal | **26.81** | 64.21 | 27.03 | 28.16 |
| Payload L3 | 45.26 | 68.26 | 43.06 | **42.45** |
| Payload L6 | **86.40** | 90.98 | 89.90 | 94.50 |
| Gain L3 | **28.91** | 68.13 | 30.54 | 36.31 |
| Gain L6 | **42.13** | 77.60 | 44.95 | 46.02 |
| Force L3 | **26.10** | 64.60 | 27.54 | 27.93 |
| Force L6 | 28.47 | 65.34 | 29.04 | **28.28** |
| Observation noise L3 | **25.90** | 65.18 | 27.72 | 29.49 |
| Observation noise L6 | 37.99 | 69.46 | 35.27 | **32.55** |

### 3.3 Fast ellipse

| 条件 | IdealZeroDelay | NaiveDelayed | VirtualDelayAware | ThreadedAsync |
|---|---:|---:|---:|---:|
| Nominal | 34.69 | 73.91 | 33.71 | **32.92** |
| Payload L3 | 50.93 | 77.88 | 50.04 | **48.72** |
| Payload L6 | 94.96 | 96.40 | 98.13 | **93.25** |
| Gain L3 | **35.97** | 76.67 | 37.15 | 41.39 |
| Gain L6 | **50.02** | 86.19 | 51.32 | 51.65 |
| Force L3 | 33.77 | 75.87 | 31.85 | **31.58** |
| Force L6 | 33.53 | 75.85 | **31.97** | 35.67 |
| Observation noise L3 | 36.56 | 69.12 | **34.68** | 38.14 |
| Observation noise L6 | 43.16 | 75.51 | 47.12 | **40.76** |

### 3.4 Rounded square

| 条件 | IdealZeroDelay | NaiveDelayed | VirtualDelayAware | ThreadedAsync |
|---|---:|---:|---:|---:|
| Nominal | 23.66 | 64.19 | **23.44** | 24.22 |
| Payload L3 | 43.15 | 63.67 | 41.29 | **37.91** |
| Payload L6 | **85.40** | 90.91 | 88.24 | 97.22 |
| Gain L3 | **26.55** | 65.98 | 28.29 | 32.52 |
| Gain L6 | **38.31** | 73.66 | 40.80 | 41.85 |
| Force L3 | 23.83 | 63.54 | **22.60** | 23.97 |
| Force L6 | **25.10** | 64.34 | 25.24 | 30.67 |
| Observation noise L3 | 24.25 | 62.70 | **23.91** | 30.05 |
| Observation noise L6 | 30.44 | 58.85 | 31.90 | **29.57** |

## 4. 每条轨迹的综合误差与控制性能

下表将该轨迹的 9 个条件 × 5 个 seed（45 cases）平均。TCP P95 是每个 rollout 的
TCP P95 再平均，不是将 45 个 rollout 拼接后的全局 P95。fallback 是执行 tick 比例。

### 4.1 Circle：跨条件平均

| 方法 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Torque RMS (Nm) | Fallback (%) |
|---|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | **40.36** | **76.19** | **2.13** | 0.017 | 29.60 | 0.00 |
| NaiveDelayed | 76.68 | 152.90 | 11.24 | 0.094 | 13.90 | 0.33 |
| VirtualDelayAware | 40.68 | 76.79 | 2.26 | 0.018 | 27.25 | 0.33 |
| ThreadedAsync | 41.66 | 80.61 | 2.26 | 0.018 | 25.12 | 2.10 |

### 4.2 Figure-8：跨条件平均

| 方法 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Torque RMS (Nm) | Fallback (%) |
|---|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | **38.66** | **74.37** | **2.06** | 0.016 | 28.01 | 0.00 |
| NaiveDelayed | 70.42 | 148.59 | 11.76 | 0.092 | 13.72 | 0.33 |
| VirtualDelayAware | 39.45 | 75.91 | 2.20 | 0.017 | 25.66 | 0.33 |
| ThreadedAsync | 40.63 | 80.65 | 2.24 | 0.017 | 24.38 | 1.90 |

### 4.3 Fast ellipse：跨条件平均

| 方法 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Torque RMS (Nm) | Fallback (%) |
|---|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | **45.95** | 88.41 | **2.44** | 0.020 | 33.29 | 0.00 |
| NaiveDelayed | 78.60 | 157.80 | 12.59 | 0.098 | 13.98 | 0.40 |
| VirtualDelayAware | 46.22 | 88.33 | 2.53 | 0.020 | 28.76 | 0.40 |
| ThreadedAsync | 46.01 | **88.06** | 2.51 | 0.020 | 26.29 | 2.47 |

### 4.4 Rounded square：跨条件平均

| 方法 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Torque RMS (Nm) | Fallback (%) |
|---|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | **35.63** | **71.29** | **1.88** | 0.014 | 26.87 | 0.00 |
| NaiveDelayed | 67.54 | 150.27 | 12.05 | 0.087 | 13.64 | 0.33 |
| VirtualDelayAware | 36.19 | 73.56 | 2.02 | 0.014 | 23.88 | 0.33 |
| ThreadedAsync | 38.67 | 81.68 | 2.16 | 0.014 | 21.71 | 5.90 |

NaiveDelayed 在每条轨迹上均显著恶化，尤其是姿态与 TCP P95；这表明问题不是由某一条
reference 特有的几何结构造成。VirtualDelayAware 在四条轨迹上都接近 IdealZeroDelay，
而 ThreadedAsync 的额外退化主要集中在软实时状态更差的条件。

## 5. ThreadedAsync：逐轨迹、逐条件的误差与实时性能

只有 ThreadedAsync 的 wall-time 指标可用于部署性能结论。表中 `solve P95` 和
`E2E P95` 都是对应条件下 5 个 rollout 的每-rollout P95 平均；late/fallback 为执行
tick 比例，deadline miss 为每 rollout 平均次数。所有行的 failure rate、planner failure、
joint/velocity/acceleration violation 均为零。

### 5.1 Circle

| 条件 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Planner Hz | Solve P95 (ms) | E2E P95 (ms) | Late (%) | Fallback (%) | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 28.40 | 60.60 | 1.42 | 0.012 | 25.35 | 38.00 | 48.78 | 0.22 | 0.33 | 0.00 |
| Payload L3 | 43.44 | 81.16 | 2.54 | 0.018 | 24.23 | 40.13 | 51.14 | 0.50 | 0.37 | 2.40 |
| Payload L6 | 91.10 | 143.71 | 5.55 | 0.029 | 24.38 | 39.46 | 50.96 | 0.77 | 0.33 | 0.00 |
| Gain L3 | 36.69 | 78.96 | 1.89 | 0.014 | 20.80 | 50.40 | 60.41 | 29.50 | 4.33 | 0.00 |
| Gain L6 | 49.82 | 110.07 | 2.41 | 0.018 | 23.67 | 41.54 | 52.32 | 3.67 | 0.76 | 0.00 |
| Force L3 | 28.67 | 57.27 | 1.44 | 0.011 | 24.41 | 39.42 | 50.88 | 0.14 | 0.33 | 0.00 |
| Force L6 | 28.69 | 58.96 | 1.45 | 0.011 | 24.05 | 42.27 | 53.26 | 3.13 | 1.29 | 0.00 |
| Observation noise L3 | 33.21 | 69.06 | 1.72 | 0.014 | 20.87 | 54.10 | 63.94 | 29.81 | 10.86 | 0.00 |
| Observation noise L6 | 34.89 | 65.72 | 1.92 | 0.015 | 24.16 | 40.29 | 51.49 | 0.37 | 0.33 | 0.00 |

### 5.2 Figure-8

| 条件 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Planner Hz | Solve P95 (ms) | E2E P95 (ms) | Late (%) | Fallback (%) | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 28.16 | 57.09 | 1.45 | 0.012 | 25.10 | 39.01 | 49.54 | 0.49 | 0.37 | 2.40 |
| Payload L3 | 42.45 | 80.27 | 2.53 | 0.018 | 24.37 | 39.75 | 50.99 | 0.64 | 0.33 | 0.00 |
| Payload L6 | 94.50 | 152.56 | 5.91 | 0.029 | 23.64 | 44.23 | 54.83 | 5.94 | 2.08 | 0.00 |
| Gain L3 | 36.31 | 81.32 | 1.89 | 0.014 | 20.80 | 49.41 | 59.45 | 30.61 | 4.88 | 0.00 |
| Gain L6 | 46.02 | 105.73 | 2.24 | 0.017 | 24.17 | 40.28 | 51.51 | 0.73 | 0.33 | 0.00 |
| Force L3 | 27.93 | 60.80 | 1.41 | 0.011 | 24.35 | 39.68 | 51.05 | 0.64 | 0.34 | 2.40 |
| Force L6 | 28.28 | 60.23 | 1.43 | 0.011 | 23.36 | 46.71 | 56.46 | 5.98 | 1.32 | 0.00 |
| Observation noise L3 | 29.49 | 61.28 | 1.53 | 0.013 | 21.23 | 49.83 | 60.37 | 23.53 | 5.59 | 0.00 |
| Observation noise L6 | 32.55 | 66.53 | 1.78 | 0.014 | 23.77 | 43.63 | 53.99 | 3.86 | 1.84 | 0.00 |

### 5.3 Fast ellipse

| 条件 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Planner Hz | Solve P95 (ms) | E2E P95 (ms) | Late (%) | Fallback (%) | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 32.92 | 64.90 | 1.68 | 0.013 | 25.44 | 37.67 | 48.81 | 0.05 | 0.40 | 0.00 |
| Payload L3 | 48.72 | 92.54 | 2.78 | 0.019 | 24.38 | 39.78 | 51.01 | 0.60 | 0.40 | 0.00 |
| Payload L6 | 93.25 | 151.47 | 5.74 | 0.030 | 23.79 | 42.04 | 52.43 | 1.75 | 0.48 | 0.00 |
| Gain L3 | 41.39 | 84.52 | 2.11 | 0.016 | 21.32 | 48.33 | 58.65 | 26.58 | 4.83 | 0.00 |
| Gain L6 | 51.65 | 108.78 | 2.53 | 0.019 | 24.11 | 40.03 | 51.39 | 0.72 | 0.40 | 0.00 |
| Force L3 | 31.58 | 61.90 | 1.62 | 0.012 | 24.09 | 41.52 | 52.20 | 1.87 | 0.97 | 0.00 |
| Force L6 | 35.67 | 73.29 | 1.85 | 0.013 | 23.26 | 48.08 | 57.63 | 8.81 | 4.23 | 0.00 |
| Observation noise L3 | 38.14 | 75.77 | 2.02 | 0.016 | 20.94 | 49.92 | 60.52 | 29.85 | 6.72 | 0.00 |
| Observation noise L6 | 40.76 | 79.39 | 2.21 | 0.017 | 23.01 | 52.21 | 61.44 | 9.76 | 3.85 | 0.00 |

### 5.4 Rounded square

| 条件 | TCP RMSE (mm) | TCP P95 (mm) | 姿态 RMSE (°) | Joint RMSE (rad) | Planner Hz | Solve P95 (ms) | E2E P95 (ms) | Late (%) | Fallback (%) | Deadline miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 24.22 | 53.70 | 1.20 | 0.010 | 25.32 | 38.24 | 49.20 | 0.74 | 0.33 | 0.00 |
| Payload L3 | 37.91 | 73.70 | 2.26 | 0.014 | 24.45 | 39.19 | 50.78 | 0.27 | 0.34 | 2.40 |
| Payload L6 | 97.22 | 150.25 | 6.30 | 0.031 | 20.90 | 49.92 | 60.30 | 29.65 | 5.42 | 0.00 |
| Gain L3 | 32.52 | 79.29 | 1.63 | 0.012 | 20.99 | 49.18 | 59.59 | 30.45 | 7.22 | 0.00 |
| Gain L6 | 41.85 | 105.28 | 1.99 | 0.017 | 24.34 | 39.53 | 50.98 | 0.46 | 0.33 | 0.00 |
| Force L3 | 23.97 | 56.62 | 1.22 | 0.010 | 23.68 | 46.07 | 56.09 | 5.28 | 2.39 | 0.00 |
| Force L6 | 30.67 | 74.90 | 1.62 | 0.012 | 20.07 | 54.88 | 64.37 | 44.86 | 16.34 | 0.00 |
| Observation noise L3 | 30.05 | 74.35 | 1.57 | 0.012 | 20.75 | 51.29 | 61.37 | 36.07 | 12.36 | 0.00 |
| Observation noise L6 | 29.57 | 67.01 | 1.64 | 0.013 | 22.16 | 52.66 | 62.76 | 19.15 | 8.37 | 0.00 |

## 6. 每条轨迹的实时汇总与解释

| 轨迹 | Planner Hz | Solve P95 (ms) | E2E P95 (ms) | Late (%) | Fallback (%) | Deadline miss / rollout | Threaded TCP RMSE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Circle | 23.55 | 42.85 | **53.69** | **7.57** | **2.10** | 0.27 | 41.66 |
| Figure-8 | 23.42 | 43.62 | 54.24 | 8.05 | 1.90 | **0.53** | 40.63 |
| Fast ellipse | 23.37 | 44.40 | 54.90 | 8.89 | 2.47 | **0.00** | 46.01 |
| Rounded square | 22.52 | 46.77 | 57.27 | 18.55 | 5.90 | 0.27 | 38.67 |

Rounded square 是 ThreadedAsync 的实时性瓶颈：跨条件的 E2E P95 为 57.27 ms，且 late 与
fallback 比例最高。其恶化并非均匀发生：Force L6、observation-noise L3 和 payload L6
分别产生 44.86%、36.07% 和 29.65% 的 late packet。Circle 的总体软实时表现最稳定。

Gain L3 和 observation-noise L3 在四条轨迹上都使 planner Hz 降至约 21 Hz，并使 E2E P95
接近或超过 60 ms；因此 ThreadedAsync 在这些条件下相对 VirtualDelayAware 的 tracking
差距应解释为实际异步 packet/fallback 行为，而不是 CEM 目标本身失效。

## 7. 与已完成 Preview IK baseline 的交叉比较

Direct IK 的独立实验使用完全相同的四条 reference、九个条件和五个 seed；其中
PreviewIK 固定使用独立标定的 `P_cal=7`。下表保留为跨轨迹总览，不能替代上文逐轨迹结果：

| 方法 | MPC / Preview 平均 TCP RMSE (mm) | 相对 Preview 的均值差 (mm) | MPC 更优的 case 比例 |
|---|---:|---:|---:|
| PreviewIK | 48.49 | 0.00 | — |
| IdealZeroDelay | 40.15 | -8.34 | 86.7% |
| VirtualDelayAware | 40.64 | -7.86 | 88.9% |
| ThreadedAsync | 41.74 | -6.75 | 84.4% |
| NaiveDelayed | 73.31 | +24.82 | 11.1% |

因此，在这组固定配置与测试矩阵中，完整的 delay-aware MPC 并未输给 PreviewIK：
Ideal、Virtual 和 Threaded 的 pooled mean 均优于 PreviewIK；只有 NaiveDelayed 明显较差。
不过，PreviewIK 仍是一个很强的低计算量基线，应在论文中单独报告，而不应与 RawIK
或 PhysicalIK 混合。

## 8. 结论与论文表述边界

- 可以支持的结论：四条轨迹上，future alignment、execution re-anchor 和 feedback 使
  VirtualDelayAware 在固定 D=6 下均接近 IdealZeroDelay，且显著优于 NaiveDelayed。
- 必须按轨迹报告的限制：ThreadedAsync 的软实时退化并不均匀。Rounded square 跨条件
  E2E P95 为 57.27 ms、late packet rate 为 18.55%，明显高于 Circle 的 53.69 ms 与
  7.57%；论文中不能只报告四轨迹 pooled 的单一 latency。
- 不能支持的结论：不应声称 threaded 已达到无 late-drop 的硬实时保证，也不能将 high-
  payload 误差完全归因于 delay，因为 IdealZeroDelay 也在每条轨迹的 payload L6 下显著退化。

## 9. 结果文件与图件

- [MPC experiment manifest](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/experiment_manifest.json)
- [逐 rollout summary](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/delay_aware_mpc_robustness_summary.csv)
- [聚合统计](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/delay_aware_mpc_robustness_aggregate.csv)
- [配对 bootstrap](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/paired_bootstrap.json)
- [TCP RMSE 曲线](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/plots/tcp_rmse_vs_perturbation.png)
- [方法配对差值](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/plots/paired_method_tcp_delta.png)
- [实时性与可靠性](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/plots/reliability_and_timing.png)
- [外力响应](../../outputs/robustness/paper_three_ik_l036_s5_v1/mpc_architectures/plots/force_response_summary.png)
- [Direct IK 对照报告](direct-ik-three-variant-robustness-results.md)
