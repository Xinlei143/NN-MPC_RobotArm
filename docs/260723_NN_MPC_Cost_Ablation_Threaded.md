# Threaded-ASAP Cost Function 消融实验操作手册

本手册复现实验文档 [260723_NN_MPC_Cost_Ablation.md](260723_NN_MPC_Cost_Ablation.md) 的 cost function 消融矩阵，但将控制协议从 `virtual_asap` 改为真实墙钟 100 Hz 的 `threaded_asap`。

它测试的是：在相同 Model A、相同 reference、相同 CEM 预算和相同逻辑延迟下，cost 项变化对**实际异步控制系统**的影响。

不要把本实验与既有 `virtual_asap` 的数字混在同一张统计表中。`threaded_asap` 会受 GPU 规划耗时、操作系统调度和 packet late drop 影响；每次运行都必须保留并报告这些实时性指标。

## 1. 固定实验设计

核心三组均运行 seed `0, 1, 2`，最终报告 mean ± sample SD（n=3）：

| 组别 | 相对 Full 的唯一修改 |
| --- | --- |
| Full | 完整默认 cost |
| No Smoothness | `w_residual_velocity=0`、`w_residual_acceleration=0`、`w_first=0` |
| No Residual Anchor | `w_residual=0` |

下面三组只运行 seed `0`，作为补充材料：

| 组别 | 相对 Full 的唯一修改 |
| --- | --- |
| No Servo Proxy | `w_servo=0` |
| No Velocity Tracking | `w_dq=0` |
| Best Rollout Checkpoint | checkpoint 改为 `best_rollout_model.pt` |

除上述唯一修改外，所有实验固定为：

- Model A：`gru_20260717_182930`；GRU history length 16；
- reference：`outputs/references/circle_3laps/reference.npz`；
- residual CEM-MPC、blackbox cost profile；
- horizon 20，128 samples，2 CEM iterations；
- `threaded_asap`、100 Hz 控制、`aligned` history、`tick_start` snapshot；
- GPU warm-up 1 次、planner guard 5 ms、strict ASAP（minimum interval 0 ms）；
- 标称 plant：payload、actuator gain、force pulse、observation noise 均为 0。

`threaded_asap` 的 worker 每次完成规划后立即发起下一次规划。因此 `--replan_interval_steps 5` 对该模式不是重规划频率控制，仅作为兼容性元数据保留；不要将它解释为“每 5 步才规划一次”。

## 2. 进入环境并定义路径

在项目根目录运行。以下命令使用一个全新的输出目录，避免覆盖已有结果。

```bash
cd ~/Data/RL_Projects/NN-MPC_RobotArm
conda activate pendulum-rl

export DEVICE=cuda
export COST_CKPT=dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt
export COST_ROLLOUT_CKPT=dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_rollout_model.pt
export COST_NORM=dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt
export COST_REF=outputs/references/circle_3laps/reference.npz
export COST_ROOT=outputs/cost_ablation_threaded
```

如果 `outputs/cost_ablation_threaded` 已经包含正式结果，请改用新的目录名，例如 `outputs/cost_ablation_threaded_rerun1`；同一路径的 `--save_dir` 会覆盖同名运行输出。

确认文件和 CUDA 可用：

```bash
test -f "$COST_CKPT" && test -f "$COST_ROLLOUT_CKPT" && test -f "$COST_NORM" && test -f "$COST_REF"
python -c "import torch; assert torch.cuda.is_available(), 'threaded_asap requires CUDA'; print(torch.cuda.get_device_name(0))"
python -c "import numpy as np; d=np.load('$COST_REF'); print('execution_steps =', int(d['execution_steps'])); print('q_des shape =', d['q_des'].shape)"
```

最后一条命令打印本机 reference 的实际执行步数。不要沿用旧 virtual-asap 报告中的 `2097`；正式结果以该文件实际记录的 `execution_steps` 为准。

## 3. 针对本 cost 配置标定逻辑 delay

先在 `virtual_asap` 下测 Model A 的 planning p95，仅用于选择 threaded-asap 的逻辑 anticipation delay；这一步不是 cost 消融结果的一部分。使用 horizon 20、128×2，并限制每个校准 rollout 为 500 个控制步以控制耗时。

```bash
mkdir -p "$COST_ROOT"

python scripts/robustness/calibrate_model_a_delay.py \
  --checkpoint "$COST_CKPT" --normalizer "$COST_NORM" \
  --model_type gru --history_len 16 --device "$DEVICE" \
  --reference_mode task --reference_file "$COST_REF" \
  --horizon 20 --num_samples 128 --cem_iters 2 \
  --max_execution_steps 500 --plans 500 --seed 0 \
  --output_path "$COST_ROOT/timing.json"

export COST_DELAY=$(python -c "import json; print(json.load(open('$COST_ROOT/timing.json'))['anticipation_delay_steps'])")
echo "Using anticipation_delay_steps=$COST_DELAY"
cat "$COST_ROOT/timing.json"
```

这里的计算规则是 `ceil((planning p95 + 5 ms) / 10 ms)`。本次所有 cost 组必须使用同一个 `$COST_DELAY`；不能为某个消融组单独调 delay。

## 4. 定义完全固定的公共参数

在**同一个 shell**中执行以下命令。之后所有运行均展开同一个 `COMMON_ARGS`，所以每组只会改变表格规定的 cost 项或 checkpoint。

```bash
COMMON_ARGS=(
  --normalizer "$COST_NORM"
  --model_type gru --history_len 16 --device "$DEVICE"
  --controller_mode mpc --mpc_policy residual --cost_profile blackbox
  --reference_mode task --reference_file "$COST_REF"
  --multirate_mode threaded_asap --anticipation_delay_steps "$COST_DELAY"
  --asap_history_mode aligned --asap_snapshot_mode tick_start
  --planner_guard_ms 5 --planner_min_interval_ms 0 --mpc_warmup_plans 1
  --horizon 20 --num_samples 128 --cem_iters 2 --elite_ratio 0.08
  --init_std 0.5 --min_std 0.25 --smoothing_alpha 0.2 --temporal_noise_alpha 0.8
  --uniform_sample_ratio 0.15 --rollout_batch_size 128 --cem_execute lowest_cost
  --replan_interval_steps 5
  --feedback_kq 0.30 --feedback_kdq 0.015 --feedback_max 0.015
  --q_ref_velocity_limit auto --q_ref_acceleration_limit auto
  --temporal_discount 0.95 --joint_limit_margin 0.02 --barrier_max_weight 2.0
  --recovery_error_ratio 1.25 --recovery_min_tracking_error 0.05
  --recovery_residual_fraction 0.95 --recovery_consecutive_steps 3 --recovery_cooldown_steps 5
  --payload_level 0 --actuator_gain_level 0 --force_pulse_level 0 --observation_noise_level 0
  --w_q 1.0 --w_dq 0.10 --w_residual 0.20 --w_servo 0.05
  --w_residual_velocity 0.05 --w_residual_acceleration 0.02 --w_first 0.20
  --w_terminal 0 --w_joint_limit 10 --w_dq_limit 5 --velocity_cost_mode track
)
```

不要在正式 threaded-asap 运行中使用 `--visualize`。该模式有实时控制线程和 CUDA planner worker，visualizer 会干扰墙钟时序。

## 5. 先做一个 Full smoke test

这一步检查 CUDA worker、参考、保存路径和实时日志是否正常。它只运行 300 个执行步，不进入正式统计。

```bash
python scripts/run_cem_mpc.py \
  --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
  --seed 0 --max_execution_steps 300 \
  --save_dir "$COST_ROOT/smoke/full_seed0"

python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["COST_ROOT"]) / "smoke/full_seed0/run_summary.json"
d = json.loads(p.read_text())
print("mode:", d["replanning"]["interval_steps"], "(None is expected for threaded_asap)")
print("planner:", d["planner"])
print("timing p95 (s):", d["timing"]["planning_time_s"].get("p95"))
print("safety:", d["safety"])
PY
```

继续前应确认：`replanning.interval_steps` 是 `None`、`planner.solve_count` 大于 0、没有 `controller_failure_count`。`late_drop_count` 可以非零，但必须在每个正式结果中保留，不能忽略。

## 6. 正式核心消融：三组 × 三个 seed

这些命令不设置 `--max_execution_steps`，因此每次执行完整的 immutable circle reference。

### 6.1 Full

```bash
for SEED in 0 1 2; do
  python scripts/run_cem_mpc.py \
    --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
    --seed "$SEED" \
    --save_dir "$COST_ROOT/core/full/seed_$SEED"
done
```

### 6.2 No Smoothness

```bash
for SEED in 0 1 2; do
  python scripts/run_cem_mpc.py \
    --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
    --seed "$SEED" \
    --w_residual_velocity 0 --w_residual_acceleration 0 --w_first 0 \
    --save_dir "$COST_ROOT/core/no_smoothness/seed_$SEED"
done
```

### 6.3 No Residual Anchor

```bash
for SEED in 0 1 2; do
  python scripts/run_cem_mpc.py \
    --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
    --seed "$SEED" --w_residual 0 \
    --save_dir "$COST_ROOT/core/no_residual_anchor/seed_$SEED"
done
```

每个完成目录都应至少包含：

```text
run_summary.json
task_tracking_summary.json
rollout.npz
```

## 7. 补充消融：seed 0

这三项不应与核心三组一起做 n=3 的统计推断；将其作为单 seed 的探索性结果或附录。

```bash
python scripts/run_cem_mpc.py \
  --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
  --seed 0 --w_servo 0 \
  --save_dir "$COST_ROOT/supplementary/no_servo_proxy/seed_0"

python scripts/run_cem_mpc.py \
  --checkpoint "$COST_CKPT" "${COMMON_ARGS[@]}" \
  --seed 0 --w_dq 0 \
  --save_dir "$COST_ROOT/supplementary/no_velocity_tracking/seed_0"

python scripts/run_cem_mpc.py \
  --checkpoint "$COST_ROLLOUT_CKPT" "${COMMON_ARGS[@]}" \
  --seed 0 \
  --save_dir "$COST_ROOT/supplementary/best_rollout_checkpoint/seed_0"
```

## 8. 汇总核心结果和实时性审计

下面命令读取九个核心运行，生成一行一个 seed 的 CSV，并打印各组 mean ± sample SD。它同时输出 tracking、力矩、规划 p95、update rate、late-drop rate 和控制 deadline miss，避免把 cost 差异与实时调度差异混为一谈。

```bash
python - <<'PY'
import csv, json, math, os, statistics
from pathlib import Path

root = Path(os.environ["COST_ROOT"]) / "core"
groups = ["full", "no_smoothness", "no_residual_anchor"]
rows = []

for group in groups:
    for seed in range(3):
        run = root / group / f"seed_{seed}"
        summary = json.loads((run / "run_summary.json").read_text())
        task = json.loads((run / "task_tracking_summary.json").read_text())
        overall = task["overall"]
        joint = task.get("joint_tracking", {})
        timing = summary["timing"]
        planner = summary["planner"]
        rows.append({
            "group": group,
            "seed": seed,
            "tcp_rmse_mm": 1000.0 * overall["position_rmse_m"],
            "tcp_max_mm": 1000.0 * overall["max_position_error_m"],
            "joint_rmse_rad": joint.get("position_rmse_rad", float("nan")),
            "orientation_rmse_deg": overall["orientation_rmse_rad"] * 180.0 / math.pi,
            "torque_rms_nm": summary["actuator"]["torque_rms_nm"],
            "planning_p95_ms": 1000.0 * timing["planning_time_s"].get("p95", float("nan")),
            "planner_update_hz": planner["actual_update_rate_hz"],
            "late_drop_count": planner["late_drop_count"],
            "late_drop_rate": planner["late_drop_rate"],
            "control_deadline_miss_count": summary["replanning"]["control_deadline_miss_count"],
            "controller_failure_count": summary["safety"]["controller_failure_count"],
        })

out = root.parent / "core_three_seed_runs.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out}")

metrics = ["tcp_rmse_mm", "joint_rmse_rad", "orientation_rmse_deg", "tcp_max_mm", "torque_rms_nm",
           "planning_p95_ms", "planner_update_hz", "late_drop_rate"]
for group in groups:
    subset = [r for r in rows if r["group"] == group]
    print("\n" + group)
    for key in metrics:
        values = [r[key] for r in subset]
        print(f"  {key}: {statistics.mean(values):.5g} ± {statistics.stdev(values):.5g}")
    print("  late_drop_count:", [r["late_drop_count"] for r in subset])
    print("  control_deadline_miss_count:", [r["control_deadline_miss_count"] for r in subset])
    print("  controller_failure_count:", [r["controller_failure_count"] for r in subset])
PY
```

上述命令从 shell 的 `$COST_ROOT` 读取路径。若在新的 shell 中单独运行汇总命令，请先重新执行：

```bash
export COST_ROOT=outputs/cost_ablation_threaded
```

## 9. 结果如何解释

主表报告以下五项为 mean ± sample SD（n=3）：

- TCP RMSE（mm）；
- joint RMSE（rad）；
- orientation RMSE（deg）；
- TCP max error（mm）；
- torque RMS（Nm）。

同表或紧邻表格必须报告实时审计：planning p95、planner update rate、late-drop count/rate、control deadline miss，以及 controller failure count。

在 `threaded_asap` 下，即使 seed 固定，线程调度也可能导致不同 packet 的激活/丢弃时刻不同。因此：

1. 固定 seed、GPU、checkpoint、reference、`$COST_DELAY` 和所有公共参数；
2. 以三个 seed 的趋势和方差解释 cost 作用，而不是只挑选单次最好结果；
3. 若某组的 late-drop rate 或 planner update rate 与其他组明显不同，明确说明该组的 tracking 差异同时受到实时调度影响；
4. 若出现 controller failure、关节限位违规或大量 control deadline miss，将该运行标注为失败/不稳定，而不是只保留误差数字；
5. 不要将本手册得到的 threaded-asap 数字与旧 `virtual_asap` 数字直接比较优劣，它们回答的是不同问题。

## 10. 实际运行环境与标定结果

### 10.1 运行环境

本次 threaded-ASAP cost ablation 在以下环境中完成：

- Operating system：Windows；
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6 GB；
- Python：3.9.23；
- PyTorch：2.0.0+cu118；
- CUDA：available；
- Model A checkpoint：`dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt`；
- Best-rollout checkpoint：`dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_rollout_model.pt`；
- Normalizer：`dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt`；
- Reference：`outputs/references/circle_3laps/reference.npz`。

为兼容当前 PyTorch 2.0 环境，对 `planner_rollout.py` 和 `cost_functions.py` 中不受该版本支持的多维 `torch.all` / `torch.any` reduction 进行了等价改写：先使用 `flatten(start_dim=1)` 展平非 batch 维度，再沿 `dim=1` 完成 reduction。

该兼容性修改不改变 cost function、约束条件或 reduction 的数学含义。

### 10.2 Anticipation delay 标定

使用 Model A、horizon 20、128 samples、2 CEM iterations 和 500 个 calibration plans 进行 delay calibration，结果如下：

| 指标 | 数值 |
| --- | ---: |
| Calibration mode | `virtual_asap` |
| Control period | 10 ms |
| Planner guard | 5 ms |
| Planning p50 | 93.46 ms |
| Planning p95 | 100.91 ms |
| Anticipation delay | 11 steps |

Anticipation delay 按操作手册规定计算：

`ceil((100.91 ms + 5 ms) / 10 ms) = 11 steps`

因此，本次所有正式 threaded-ASAP cost ablation 统一使用：

`anticipation_delay_steps=11`

所有组使用相同 anticipation delay，没有针对单独的 cost configuration 重新标定 delay。

### 10.3 Reference 与有效执行长度

Reference 文件中记录：

- `execution_steps=2107`；
- `q_des.shape=(2128, 6)`；
- prediction horizon 为 20；
- anticipation delay 为 11。

正式 threaded-ASAP 运行的有效执行长度为：

`2128 - 20 - 11 - 1 = 2096 steps`

因此，每个正式实验均记录 2096 个控制步。

这里的 2096 并不表示运行提前终止。Controller 需要为 20-step prediction horizon、11-step anticipation delay 和一步 look-ahead 保留后续 reference samples，因此有效闭环执行长度小于 reference metadata 中记录的 2107 steps。

### 10.4 实验与结果文件完整性

本次共完成 12 个正式运行。

核心实验：

- Full：seed 0、1、2；
- No Smoothness：seed 0、1、2；
- No Residual Anchor：seed 0、1、2。

补充实验：

- No Servo Proxy：seed 0；
- No Velocity Tracking：seed 0；
- Best Rollout Checkpoint：seed 0。

此外还完成了一个 300-step Full smoke test，该测试仅用于检查 threaded worker、CUDA、reference 和输出流程，不计入正式统计。

所有 12 个正式实验均记录 2096 个有效控制步。

根据各运行目录中的 `run_summary.json` 和 `task_tracking_summary.json`，所有正式实验均满足：

- planner failure count：0；
- controller failure count：0；
- joint-limit violation count：0；
- command-velocity violation count：0；
- command-acceleration violation count：0；
- recovery trigger count：0。

核心与补充实验的汇总结果保存在：

```text
outputs/cost_ablation_threaded/core_three_seed_runs.csv
outputs/cost_ablation_threaded/core_three_seed_summary.csv
outputs/cost_ablation_threaded/supplementary_seed0_runs.csv
```

Delay calibration 结果保存在：

```text
outputs/cost_ablation_threaded/timing.json
```

---

## 11. 核心消融实验结果

核心三组分别运行 seed 0、1、2。

以下结果报告 mean ± sample SD，样本数为 n=3。

### 11.1 Tracking accuracy 与 torque RMS

| Configuration | Joint RMSE (rad) | TCP RMSE (mm) | Orientation RMSE (deg) | TCP max error (mm) | Torque RMS (Nm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 0.020947 ± 0.000091 | 52.842 ± 0.229 | 2.725 ± 0.013 | 98.484 ± 0.304 | 7.860 ± 0.285 |
| No Smoothness | 0.020858 ± 0.000426 | 52.457 ± 0.748 | 2.707 ± 0.034 | 98.302 ± 0.102 | 8.479 ± 1.276 |
| No Residual Anchor | 0.020995 ± 0.000163 | 52.678 ± 0.442 | 2.721 ± 0.011 | 97.932 ± 0.698 | 8.430 ± 1.135 |

相对于 Full 的均值变化如下：

| Configuration | Joint RMSE | TCP RMSE | Orientation RMSE | TCP max error | Torque RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| No Smoothness | -0.43% | -0.73% | -0.67% | -0.18% | +7.88% |
| No Residual Anchor | +0.23% | -0.31% | -0.15% | -0.56% | +7.25% |

负的 tracking error 百分比表示误差减小，正的百分比表示误差增加。

No Smoothness 的 mean joint RMSE、TCP RMSE 和 orientation RMSE 均略低于 Full，但改善幅度只有约 0.43%–0.73%。

与此同时，No Smoothness 的 torque RMS 从 Full 的 `7.860 ± 0.285 Nm` 增加到 `8.479 ± 1.276 Nm`，平均增加 7.88%。

因此，删除 smoothness terms 没有带来明显的 tracking improvement，却增加了平均控制力矩，并使不同 seed 之间的结果波动明显增大。

No Residual Anchor 的所有 mean tracking metrics 与 Full 的差异也均小于 1%。其中 joint RMSE 略微增加 0.23%，而 TCP RMSE、orientation RMSE 和 TCP max error 略微降低。

但是其 torque RMS 从 `7.860 ± 0.285 Nm` 增加到 `8.430 ± 1.135 Nm`，平均增加 7.25%。

因此，residual anchor 对平均 tracking accuracy 的影响较小，但能够降低控制力矩并减小不同运行之间的波动。

### 11.2 实时性审计

Threaded-ASAP 的 tracking 结果会受到实际 wall-clock planning latency 和 packet activation/drop behavior 的影响，因此实时性指标必须与 tracking results 一起报告。

| Configuration | Planning p95 (ms) | Planner update rate (Hz) | Late-drop rate | Late-drop count | Control deadline misses | Controller failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 112.55 ± 19.84 | 9.147 ± 0.615 | 90.69% ± 2.50% | 174.0 ± 10.0 | 758.0 ± 2.0 | 0 / 0 / 0 |
| No Smoothness | 106.77 ± 10.23 | 9.417 ± 0.273 | 90.86% ± 6.58% | 179.7 ± 8.1 | 762.7 ± 2.9 | 0 / 0 / 0 |
| No Residual Anchor | 114.40 ± 19.56 | 9.156 ± 0.548 | 91.94% ± 3.91% | 176.7 ± 9.3 | 759.7 ± 3.1 | 0 / 0 / 0 |

三个核心组的 planning p95 均约为 100–115 ms，planner update rate 均约为 9–9.4 Hz。

因此，从总体统计上看，三个核心组的 planner computational load 和实际 update rate 处于相近范围，没有某一个 cost configuration 明显获得更快的规划频率。

但是，三个组的 mean late-drop rate 均超过 90%：

- Full：90.69%；
- No Smoothness：90.86%；
- No Residual Anchor：91.94%。

这说明 threaded-ASAP 系统处于较高的实时规划压力下，大量完成的 planning packets 无法按照预期的 logical activation timing 生效。

虽然三个组均没有发生 planner failure 或 controller failure，但小于 1% 的 tracking differences 不能完全解释为 cost function 本身的独立作用。

Wall-clock planning latency、线程调度和 packet activation/drop pattern 同样会影响最终的闭环轨迹。

### 11.3 每个 seed 的结果

Full 三个 seed：

| Seed | Joint RMSE (rad) | TCP RMSE (mm) | Torque RMS (Nm) | Planning p95 (ms) | Late-drop rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.020989 | 52.838 | 8.145 | 135.42 | 92.66% |
| 1 | 0.020843 | 52.615 | 7.860 | 102.29 | 87.88% |
| 2 | 0.021010 | 53.073 | 7.575 | 99.95 | 91.54% |

No Smoothness 三个 seed：

| Seed | Joint RMSE (rad) | TCP RMSE (mm) | Torque RMS (Nm) | Planning p95 (ms) | Late-drop rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.020921 | 52.517 | 8.187 | 102.07 | 90.95% |
| 1 | 0.020403 | 51.681 | 9.876 | 99.74 | 84.24% |
| 2 | 0.021249 | 53.174 | 7.375 | 118.50 | 97.40% |

No Residual Anchor 三个 seed：

| Seed | Joint RMSE (rad) | TCP RMSE (mm) | Torque RMS (Nm) | Planning p95 (ms) | Late-drop rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.021053 | 52.842 | 7.665 | 136.92 | 94.41% |
| 1 | 0.021120 | 53.014 | 7.890 | 101.71 | 93.97% |
| 2 | 0.020810 | 52.178 | 9.733 | 104.57 | 87.44% |

完整的一行一个 seed 的数据保存在：

```text
outputs/cost_ablation_threaded/core_three_seed_runs.csv
```

三组 mean ± sample SD 的数据保存在：

```text
outputs/cost_ablation_threaded/core_three_seed_summary.csv
```

---

## 12. 补充消融实验结果

三个补充实验均只运行 seed 0，因此仅作为 exploratory results，不进行 n=3 的统计推断。

为便于比较，下表同时列出 Full seed 0。

### 12.1 Tracking 与实时性结果

| Configuration | Joint RMSE (rad) | TCP RMSE (mm) | Orientation RMSE (deg) | TCP max error (mm) | Torque RMS (Nm) | Planning p95 (ms) | Update rate (Hz) | Late-drop rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full, seed 0 | 0.020989 | 52.838 | 2.716 | 98.500 | 8.145 | 135.42 | 8.441 | 92.66% |
| No Servo Proxy, seed 0 | 0.021061 | 52.961 | 2.717 | 98.420 | 7.629 | 104.88 | 9.244 | 97.42% |
| No Velocity Tracking, seed 0 | 0.021156 | 53.182 | 2.724 | 98.243 | 7.292 | 104.39 | 9.209 | 98.96% |
| Best Rollout Checkpoint, seed 0 | 0.021029 | 52.995 | 2.728 | 98.421 | 7.564 | 101.09 | 9.454 | 93.47% |

对应的 late-drop、deadline miss 和 failure 指标为：

| Configuration | Late-drop count | Control deadline misses | Planner failures | Controller failures |
| --- | ---: | ---: | ---: | ---: |
| Full, seed 0 | 164 | 756 | 0 | 0 |
| No Servo Proxy, seed 0 | 189 | 769 | 0 | 0 |
| No Velocity Tracking, seed 0 | 191 | 771 | 0 | 0 |
| Best Rollout Checkpoint, seed 0 | 186 | 768 | 0 | 0 |

### 12.2 No Servo Proxy

相对于 Full seed 0，删除 servo proxy 后：

- joint RMSE 增加约 0.34%；
- TCP RMSE 增加约 0.23%；
- orientation RMSE 基本不变；
- torque RMS 降低约 6.33%；
- late-drop rate 从 92.66% 增加到 97.42%。

因此，在这一单次运行中，删除 servo proxy 没有改善 tracking accuracy，但降低了 torque RMS。

不过，该实验只有一个 seed，而且其 late-drop rate 明显高于 Full seed 0，因此不能把全部变化都归因于 `w_servo=0`。

该结果应作为探索性观察，而不是确定性的统计结论。

### 12.3 No Velocity Tracking

相对于 Full seed 0，删除 velocity tracking 后：

- joint RMSE 增加约 0.79%；
- TCP RMSE 增加约 0.65%；
- orientation RMSE 增加约 0.28%；
- torque RMS 降低约 10.47%；
- late-drop rate 从 92.66% 增加到 98.96%。

从单次运行结果看，velocity tracking term 可能使用更高的 control effort 换取小幅 position 和 orientation tracking improvement。

但是，该运行的 late-drop rate 接近 99%，因此不能把 tracking 和 torque 变化完全解释为 `w_dq=0` 的独立作用。

该实验同样只能作为 exploratory result。

### 12.4 Best Rollout Checkpoint

相对于使用 `best_model.pt` 的 Full seed 0，使用 `best_rollout_model.pt` 后：

- joint RMSE 增加约 0.19%；
- TCP RMSE 增加约 0.30%；
- orientation RMSE 增加约 0.43%；
- torque RMS 降低约 7.13%；
- planning p95 从 135.42 ms 降低到 101.09 ms；
- late-drop rate 从 92.66% 增加到 93.47%。

在本次 threaded-ASAP 闭环测试中，`best_rollout_model.pt` 没有改善 tracking accuracy。

因此，目前没有实验依据支持使用 `best_rollout_model.pt` 替换默认的 `best_model.pt`。

三个补充运行的汇总结果保存在：

```text
outputs/cost_ablation_threaded/supplementary_seed0_runs.csv
```

---

## 13. 结果讨论

### 13.1 Smoothness terms 的作用

No Smoothness 的 mean joint RMSE、TCP RMSE 和 orientation RMSE 均略低于 Full，但改善幅度只有约 0.43%–0.73%。

这些差异都小于 1%，而且三个 seed 之间存在明显波动，因此不能据此认为删除 smoothness terms 能够稳定提高 tracking accuracy。

相比之下，control effort 的变化更加明显。

Full 的 torque RMS 为：

`7.860 ± 0.285 Nm`

No Smoothness 的 torque RMS 为：

`8.479 ± 1.276 Nm`

即平均 torque RMS 增加约 7.88%。

此外，torque RMS 的 sample SD 从 0.285 Nm 增加到 1.276 Nm。Joint RMSE 和 TCP RMSE 的跨 seed 波动也明显增大。

因此，本次实验更支持以下解释：

Smoothness terms 对平均 tracking accuracy 的影响较小，但能够降低 control effort，并提高不同运行之间的一致性。

当前结果不支持从默认 cost function 中删除 smoothness terms。

### 13.2 Residual anchor 的作用

No Residual Anchor 与 Full 的所有 mean tracking metrics 差异均小于 1%。

其中 joint RMSE 从：

`0.020947 rad`

变为：

`0.020995 rad`

仅增加约 0.23%。

与此同时，torque RMS 从：

`7.860 ± 0.285 Nm`

增加到：

`8.430 ± 1.135 Nm`

平均增加约 7.25%。

Torque RMS 的跨 seed 波动也明显增大。

因此，residual anchor 对平均 tracking accuracy 的影响较小，但能够对 residual control action 起到正则化作用，从而降低平均控制力矩并减小运行间波动。

当前实验结果支持保留 residual anchor。

### 13.3 Full cost 的综合表现

Full configuration 并没有在每一个 tracking metric 上取得数值最小值。

例如，No Smoothness 的 mean joint RMSE 和 TCP RMSE 都略低于 Full。

但是这些 tracking differences 均小于 1%。

与此同时，Full configuration 具有：

- 与两个消融组基本相同的 mean tracking accuracy；
- 三个核心组中最低的 mean torque RMS；
- 三个核心组中最低的 torque RMS sample SD；
- 较低的 joint RMSE sample SD；
- 与其他组处于同一数量级的 planning p95；
- 与其他组接近的 planner update rate；
- 0 planner failures；
- 0 controller failures；
- 0 joint-limit violations；
- 0 command-limit violations。

因此，从 tracking accuracy、control effort 和 across-seed consistency 三方面综合考虑，Full cost configuration 提供了更均衡的闭环表现。

本次实验没有提供足够证据支持删除 smoothness terms 或 residual anchor。

因此，Full cost configuration 应继续作为默认配置。

### 13.4 Threaded-ASAP 的实时性限制

本次实验最重要的限制是较高的 late-drop rate。

三个核心组的 mean late-drop rate 分别为：

```text
Full:               90.69%
No Smoothness:      90.86%
No Residual Anchor: 91.94%
```

Delay calibration 阶段得到的 planning p95 为：

`100.91 ms`

正式实验中三个核心组的 mean planning p95 为：

```text
Full:               112.55 ms
No Smoothness:      106.77 ms
No Residual Anchor: 114.40 ms
```

实际 planner update rate 约为：

`9–9.4 Hz`

而 control loop 的目标频率为：

`100 Hz`

这说明 planner 的 wall-clock computation 明显慢于 control loop 的更新周期，并且大量规划 packets 无法按照预期的 logical activation timing 生效。

因此，在 threaded-ASAP 条件下，cost function 并不是决定最终 tracking performance 的唯一因素。

以下实时因素同样会影响最终结果：

- operating-system thread scheduling；
- GPU planning duration；
- planner completion timing；
- packet activation timing；
- packet expiration；
- late packet dropping。

三个核心配置的实时性统计总体处于相近范围，因此当前结果仍然能够用于比较不同 cost configurations 在真实异步运行条件下的整体行为趋势。

但是，小于 1% 的 tracking differences 不应被解释为严格的、无混杂因素的 cost-function 因果效应。

这也是本 threaded-ASAP 实验与此前 virtual-ASAP cost ablation 的重要区别。

---

## 14. 总结

本次实验完成了三个核心 cost configurations 的 threaded-ASAP cost ablation，以及三个单 seed 补充实验。

核心配置为：

- Full；
- No Smoothness；
- No Residual Anchor。

每个核心配置均运行 seed 0、1、2，共完成 9 个核心正式运行。

补充配置为：

- No Servo Proxy；
- No Velocity Tracking；
- Best Rollout Checkpoint。

每个补充配置运行 seed 0，共完成 3 个补充正式运行。

因此，本次共完成 12 个正式 threaded-ASAP runs。

所有正式运行均记录预期的 2096 个有效控制步，并且未发生：

- planner failure；
- controller failure；
- joint-limit violation；
- command-velocity violation；
- command-acceleration violation；
- recovery trigger。

核心实验中，删除 smoothness terms 或 residual anchor 后，mean joint-space 和 task-space tracking error 的变化均小于 1%。

因此，当前实验没有证据表明删除这两类 cost terms 能够带来稳定而明显的 tracking improvement。

No Smoothness 的 mean joint RMSE 和 TCP RMSE 分别比 Full 低约 0.43% 和 0.73%。

但是其 mean torque RMS 增加约 7.88%，同时 tracking 和 torque metrics 的跨 seed 波动明显增大。

因此，smoothness terms 的主要作用更可能是降低 control effort 并提高闭环运行一致性，而不是直接降低平均 tracking error。

No Residual Anchor 的 tracking performance 与 Full 基本相同。

但是，其 mean torque RMS 增加约 7.25%，并且 torque RMS 的跨 seed variation 明显增大。

因此，residual anchor 对 tracking accuracy 的影响较小，但能够对 residual control action 起到有效的正则化作用。

综合 tracking accuracy、torque RMS 和 across-seed consistency，Full cost configuration 在三个核心配置中提供了最均衡的总体表现。

因此，本次 threaded-ASAP cost ablation 不支持删除 smoothness terms 或 residual anchor，Full cost 应继续作为默认 cost configuration。

补充实验中，删除 servo proxy 或 velocity tracking term 均降低了 torque RMS，但同时伴随轻微的 tracking degradation。

由于这两个实验都只有一个 seed，并且 late-drop rate 分别达到约 97.42% 和 98.96%，因此相关结果只能作为 exploratory observations。

使用 `best_rollout_model.pt` 也没有改善 threaded-ASAP 闭环 tracking accuracy，因此当前数据没有提供证据支持使用该 checkpoint 替换默认的 `best_model.pt`。

最后，三个核心组的 mean late-drop rate 均约为 91%。

因此，本实验观察到的小幅 tracking differences 不能完全归因于 cost function。

Real-time planning latency、thread scheduling 和 packet activation/drop behavior 是 threaded-ASAP 实验中不可忽略的影响因素。

总体而言，本次实验支持继续保留当前 Full cost configuration。Smoothness terms 和 residual anchor 在基本不牺牲 mean tracking accuracy 的情况下，能够降低 control effort，并提高不同运行之间的总体一致性。

## 15. 最终产物清单

完成后建议保留以下小文件进入结果归档或论文补充材料；不要提交每个运行的大型 `rollout.npz`，除非需要复现原始曲线。

```text
outputs/cost_ablation_threaded/
  timing.json
  core_three_seed_runs.csv
  core/
    full/seed_{0,1,2}/run_summary.json
    no_smoothness/seed_{0,1,2}/run_summary.json
    no_residual_anchor/seed_{0,1,2}/run_summary.json
  supplementary/
    no_servo_proxy/seed_0/run_summary.json
    no_velocity_tracking/seed_0/run_summary.json
    best_rollout_checkpoint/seed_0/run_summary.json
```

原始 virtual-asap 实验的背景、cost 定义和旧结果见 [260723_NN_MPC_Cost_Ablation.md](260723_NN_MPC_Cost_Ablation.md)。
