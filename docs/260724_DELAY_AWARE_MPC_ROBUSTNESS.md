# Delay-aware MPC 四协议鲁棒性实验

## 2026-07-24：H20 + 方案 B 正式实验

### 必须先标定 D

正式鲁棒性实验不得直接复用其他配置或其他日期的 delay。每次冻结 checkpoint、
horizon、CEM budget 或 planner projection 策略后，必须先在完全相同的 planner
配置下重新运行真实 `threaded_asap` E2E 标定，再根据

```text
D = ceil((E2E P95 + planner_guard) / control_dt)
```

确定本次实验的 `anticipation_delay_steps`。本轮固定配置为 H20、128 candidates、
2 CEM iterations、`planner_projection=on`、`backend=compiled`、
`strategy=two_stage`，使用 circle 和 figure-8 收集 500 个真实 threaded plans：

```bash
conda run -n pendulum-rl python \
  scripts/experiments/planner_projection/calibrate_h20.py \
  --manifest outputs/robustness_h20_d10/benchmark.json \
  --case_ids circle_00,figure8_00 \
  --plans 500 \
  --calibration_delay 10 \
  --planner_projection on \
  --planner_projection_backend compiled \
  --planner_projection_strategy two_stage \
  --output_path \
    outputs/robustness/calibration/h20_two_stage_compiled_20260724.json
```

只有标定文件生成并通过配置、样本数和有限数值检查后，才允许启动 432-rollout
正式实验。正式实验必须引用本次新生成的标定文件，不能引用此前的
`two_stage_compiled.json`。

### 本轮标定结果

500 个 plan 全部采集成功，配置和有限数值检查通过：

| 指标 | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| Solve latency | 30.41 ms | 32.52 ms | 33.71 ms | 38.39 ms |
| Threaded E2E latency | 39.61 ms | 44.72 ms | 45.74 ms | 47.19 ms |

本轮标定无 late packet。由
`ceil((44.72 + 5) / 10) = 5` 得到 **D=5**。标定文件为
[`outputs/robustness/calibration/h20_two_stage_compiled_20260724.json`](../outputs/robustness/calibration/h20_two_stage_compiled_20260724.json)。
这也说明 D 是运行时平台和当次冻结配置的测量结果，不能直接沿用此前得到的 D6。

### 正式实验

使用 12 个固定 case、9 个 nominal/单因素扰动条件和 4 种方法，共运行 432 个
rollout：

```bash
conda run -n pendulum-rl python \
  scripts/robustness/evaluate_delay_aware_mpc.py \
  --manifest outputs/robustness_h20_d10/benchmark.json \
  --delay_calibration \
    outputs/robustness/calibration/h20_two_stage_compiled_20260724.json \
  --planner_projection on \
  --planner_projection_backend compiled \
  --planner_projection_strategy two_stage \
  --save_dir \
    outputs/robustness/delay_aware_mpc_h20_two_stage_d5_20260724 \
  --resume
```

完整性检查结果：

- 432/432 个唯一 `method × condition × case` 组合；
- 每种方法 108 个 rollout，每个 rollout 均为 500 steps；
- manifest 固定 H20、D5、projection on、compiled、two-stage；
- planner failure、joint/velocity/acceleration violation 均为 0。

原始数据、bootstrap 和图片位于
[`outputs/robustness/delay_aware_mpc_h20_two_stage_d5_20260724`](../outputs/robustness/delay_aware_mpc_h20_two_stage_d5_20260724)。

### 总体结果

下表对每种方法的 108 个 rollout 做 pooled mean。不同扰动强度混合后的 pooled
结果用于概览，具体结论应结合后面的 condition 表。

| 方法 | TCP RMSE | Joint RMSE | Orientation RMSE | Solve P95 | E2E P95 | Planner | Late | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IdealZeroDelay | **61.79 mm** | **0.02295 rad** | **3.300°** | 31.34 ms | n/a | n/a | 0% | 0% |
| NaiveDelayed | 129.95 mm | 0.14191 rad | 19.068° | 31.51 ms | n/a | n/a | 0% | 1.00% |
| VirtualDelayAware | 62.24 mm | 0.02312 rad | 3.377° | 31.44 ms | n/a | n/a | 0% | 1.00% |
| ThreadedAsync | 63.59 mm | 0.02361 rad | 3.436° | 33.26 ms | 43.87 ms | 29.56 Hz | 2.03% | 1.02% |

Virtual 只比 Ideal 高 0.45 mm，说明 D5 下 future alignment、residual re-anchor 和
fast feedback 基本补偿了逻辑 delay。Threaded 比 Virtual 高 1.35 mm；差距很小，
但不应忽略其真实调度产生的 late packet 和少量 packet expiration。Naive 比 Ideal
高 68.16 mm，主要问题仍是重放 stale absolute commands，而不是 CEM 单次求解质量。

Nominal 条件下结果为：

| 方法 | TCP RMSE | Joint RMSE | Orientation RMSE |
|---|---:|---:|---:|
| IdealZeroDelay | 53.50 mm | 0.01948 rad | 2.686° |
| NaiveDelayed | 130.46 mm | 0.13632 rad | 17.939° |
| VirtualDelayAware | 53.79 mm | **0.01944 rad** | 2.744° |
| ThreadedAsync | 54.74 mm | 0.02008 rad | 2.761° |

Nominal 配对 bootstrap：

- Naive − Ideal：+76.96 mm，95% CI `[+62.13, +93.98]`；
- Virtual − Ideal：+0.29 mm，CI `[-0.56, +1.51]`；
- Threaded − Ideal：+1.23 mm，CI `[+0.37, +2.12]`；
- Threaded − Virtual：+0.95 mm，CI `[-0.15, +2.10]`。

Virtual 与 Ideal、Threaded 与 Virtual 的区间跨 0；当前 12 个 nominal case 下不能
认为它们存在稳定的 tracking 差异。Threaded 相对 Ideal 的小幅差值区间未跨 0，
说明真实异步时序仍带来约 1 mm 级代价。

### 不同扰动下的 TCP RMSE

| Condition | Ideal | Naive | Virtual | Threaded |
|---|---:|---:|---:|---:|
| nominal | 53.50 | 130.46 | 53.79 | 54.74 |
| actuator gain L3 | 61.15 | 135.27 | 61.35 | 62.85 |
| actuator gain L6 | 83.57 | 141.76 | 83.56 | 84.70 |
| force pulse L3 | 54.14 | 130.33 | 55.10 | 55.54 |
| force pulse L6 | 59.72 | 131.55 | 60.11 | 61.69 |
| observation noise L3 | 52.51 | 130.39 | 53.84 | 54.77 |
| observation noise L6 | 54.26 | 129.23 | 54.12 | 56.05 |
| payload L3 | 51.07 | 116.89 | 52.06 | 53.38 |
| payload L6 | 86.17 | 123.64 | 86.19 | 88.58 |

单位均为 mm。主要观察如下：

- **Payload L6 是 full delay-aware 方法最严重的扰动。** Ideal/Virtual/Threaded
  分别达到 86.17/86.19/88.58 mm。大质量负载改变重力和惯量，形成训练模型之外的
  plant mismatch；delay compensation 无法消除模型本身的系统误差。
- **Actuator gain L6 次之。** 三种 full 方法约 83.6–84.7 mm，说明低增益造成的
  actuator lag 同样是主要误差来源。
- **Force pulse 的影响相对局部。** L6 下 full 方法为 59.7–61.7 mm，仍远低于
  payload/gain L6。Threaded 的 L6 peak error 为 106.8 mm，平均恢复时间 1.47 s；
  Virtual 为 106.8 mm/1.65 s，Ideal 为 107.3 mm/1.65 s。
- **Observation noise 退化较小。** Full 方法在 L6 下为 54.1–56.0 mm，fast
  feedback 的限幅避免了明显发散；Threaded 比 Virtual 高约 1.93 mm。
- Payload L3 偶然低于 nominal，不能解释为“增加 payload 有益”。它反映有限 case
  下 plant dynamics、轨迹和 controller correction 的相互作用；L6 的强退化才显示
  清晰的强度趋势。
- Naive 在 nominal 已经很差，因此某些扰动下的增量反而较小。特别是 force recovery
  的“excess above nominal baseline”会被其很高的 baseline error 压低，不能据此
  认为 Naive 恢复更快。

### 真实 Threaded 时序与 D5 解释

Threaded 的 108 个 rollout：

| 指标 | Mean | Median | Worst run |
|---|---:|---:|---:|
| E2E P50 | 38.66 ms | 38.64 ms | 39.45 ms |
| E2E P95 | 43.87 ms | 43.80 ms | 46.12 ms |
| E2E P99 | 45.93 ms | 45.40 ms | 58.54 ms |
| Planner rate | 29.56 Hz | 29.60 Hz | 30.27 Hz（max） |
| Late packet rate | 2.03% | 2.00% | 7.64% |

总计出现 34 次 control deadline miss、2 次 packet expiration、0 次 planner
failure；93/108 个 rollout 至少有一个 late packet。D5 的可发布 deadline 是
`50 - 5 = 45 ms`，而正式实验 E2E P95 约 43.87 ms，因此约 2% late packet 与
P95 标定目标一致。D5 是按 P95 规则得到的吞吐/新鲜度折中，不代表零 late。

如果部署要求所有 packet 均按时，应改用更保守的 P99/max 标定或人为增加到 D6，
并重新运行正式测试；不能把当前 D5 结果描述为 hard real-time guarantee。尽管存在
少量 late，Threaded pooled TCP 只比 Virtual 高 1.35 mm，且没有安全约束违规，
说明 executor fallback 和 fast feedback 在本实验中有效限制了调度抖动的影响。

### 轨迹差异

Nominal TCP RMSE 中 circle 对 full 方法最难（Ideal 59.01 mm），square 最低
（Ideal 48.58 mm）。Naive 对 figure-8 最差，达到 158.91 mm，说明带方向反转和
交叉段的轨迹对 stale absolute command 尤其敏感。

---

## 历史记录（2026-07-23：H25 + D5 + planner projection off）

该工作流在同一批固定 task-space references、相同扰动和相同 CEM seed 下比较：

| 方法 | 运行模式 | delay protocol | 逻辑 delay |
| --- | --- | --- | --- |
| `IdealZeroDelay` | `virtual_asap` | `full` | 0 |
| `NaiveDelayed` | `virtual_asap` | `naive_delayed` | 标定值 D |
| `VirtualDelayAware` | `virtual_asap` | `full` | 标定值 D |
| `ThreadedAsync` | `threaded_asap` | `full` | 标定值 D |

当前冻结模型为
`dynamics_modeling/outputs/checkpoints/gru_20260717_182930`。500-plan 标定得到
planning P95 = 40.98 ms，加入 5 ms guard 后 D = 5 个 10 ms 控制步，记录在
`outputs/robustness/timing/gru_20260717_182930.json`。

## 正式中等规模实验

默认使用 circle、figure-8、ellipse、square 各 3 个 case，运行共享 nominal，以及
payload、actuator gain、force pulse、observation noise 的 level 3/6。总计：

```text
12 cases × 9 conditions × 4 protocols = 432 rollouts
```

```bash
conda run -n pendulum-rl python scripts/robustness/evaluate_delay_aware_mpc.py \
  --manifest outputs/robustness/benchmark.json \
  --checkpoint dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt \
  --delay_calibration outputs/robustness/timing/gru_20260717_182930.json \
  --save_dir outputs/robustness/delay_aware_mpc_medium
```

中断后添加 `--resume`。每个 rollout 的 fingerprint 包括 checkpoint、normalizer、
delay calibration、reference、plant XML、扰动和完整控制参数，不匹配时拒绝复用。

## 冒烟测试

先在独立目录以一个 reference、20 步和 force level 6 检查四协议：

```bash
conda run -n pendulum-rl python scripts/robustness/evaluate_delay_aware_mpc.py \
  --case_ids circle_00 --levels 0,6 --perturbations force_pulse \
  --max_execution_steps 20 --bootstrap_samples 100 \
  --save_dir /tmp/delay_aware_mpc_robustness_smoke
```

## 输出

每个 `<method>/<condition>/<case_id>/` 目录包含完整 `rollout.csv/.npz`、
`run_summary.json`、`task_tracking_summary.json`、planner events（threaded）和跟踪图。
实验根目录包含逐运行 CSV、分组 aggregate CSV、配对 bootstrap JSON、冻结 manifest
以及 TCP、姿态、平滑性、可靠性、外力响应和方法配对差值图。

只有 `ThreadedAsync` 的 wall-clock planner rate、late packet、控制周期和 deadline
指标具有真实并发执行含义；三个 virtual 方法的 wall time 只作为计算诊断，不能解释为
实时部署性能。
