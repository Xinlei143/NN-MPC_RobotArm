# Threaded-ASAP Cost Function 消融实验操作手册

本手册复现实验文档 [NN_MPC_Cost_Ablation.md](NN_MPC_Cost_Ablation.md) 的 cost function 消融矩阵，但将控制协议从 `virtual_asap` 改为真实墙钟 100 Hz 的 `threaded_asap`。

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

## 10. 最终产物清单

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

原始 virtual-asap 实验的背景、cost 定义和旧结果见 [NN_MPC_Cost_Ablation.md](NN_MPC_Cost_Ablation.md)。
