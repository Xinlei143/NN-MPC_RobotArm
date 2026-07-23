# Delay-Aware MPC 论文实验操作手册

本文档对应 ROBIO 论文的五控制器主比较、因果消融、delay sweep、Preview IK、GRU 验证和 oracle-dynamics upper bound。所有新产物均写入：

```text
outputs/paper_delay_aware/
```

Model C 不进入本文主线。Payload、观测噪声、执行器失配和外力实验继续使用 [`MODEL_A_ROBUSTNESS.md`](MODEL_A_ROBUSTNESS.md)，不复制进本工作流。

## 1. 环境和冻结模型检查

```bash
conda activate pendulum-rl
cd ~/Data/RL_Projects/NN-MPC_RobotArm

export PAPER_OUT=outputs/paper_delay_aware
export PAPER_CKPT=outputs/checkpoints/gru_20260720_202923/best_model.pt
export PAPER_NORM=outputs/checkpoints/gru_20260720_202923/normalizer.pt
mkdir -p "$PAPER_OUT/logs"

test -f "$PAPER_CKPT"
test -f "$PAPER_NORM"
nvidia-smi
git status --short
```

论文配置显式使用 GRU、history 16、H20、128 candidates、2 CEM iterations 和每5个逻辑 tick 重规划一次。不要依赖根 CLI 的 Transformer 默认值。正式 manifest 必须在 clean worktree 上生成，并记录 commit、checkpoint/reference hash、Python、PyTorch、CUDA、MuJoCo、GPU 和 driver。

## 2. 控制协议

底层只提供五个不重复的 `--delay_protocol`：

| ID | Activation state/reference | 执行方式 | Feedback |
|---|---|---|---|
| `full` | 均对齐到 `k+D` | 当前 IK nominal + age-aligned residual | 开 |
| `naive_delayed` | 均停留在 launch time | 激活后从 age 0 重放 planner-projected absolute commands | 关 |
| `no_future_alignment` | 均停留在 launch time | 当前 nominal + residual | 开 |
| `no_reanchor` | 均对齐到 `k+D` | 重放 planner-projected absolute commands | 开 |
| `no_feedback` | 均对齐到 `k+D` | 当前 nominal + residual | 关 |

Ideal zero-delay 定义为 `full + D=0`。它保持相同的逻辑20 Hz重规划频率和 CEM budget，不代表100 Hz实时 CEM。

virtual runner 是 deterministic fixed-delay simulation：wall time 只记录为诊断，绝不决定 virtual packet 是否激活。真实 late drop 只由 `threaded_asap` 报告。

`naive_delayed` 和 `no_reanchor` 的 absolute sequence 已经在 CEM candidate rollout 中依据预测锚点做过完整运动学投影；执行时仅做 joint-limit clip，不再按真实上一拍命令做第二次速度/加速度投影。否则第二次投影本身就是 execution-time reconciliation，会让 `no_reanchor` 与 Full 退化为同一控制器。相应的实际速度/加速度违例会原样记录并进入结果表。

## 3. 生成独立 calibration reference

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  generate-calibration-reference
```

生成的 joint chirp 只用于平台延迟标定，不进入 circle、figure-eight、fast ellipse 或 rounded-square 正式测试。

## 4. 用真实 planner E2E 标定 D

标定期间不要同时运行其他 GPU 工作负载：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  calibrate-delay \
  --samples 500 \
  --provisional-delay 10 \
  --guard-ms 5
```

标定读取 `planner_end_to_end_latency_s`：

```text
snapshot launch -> future forecast/history -> CEM -> packet construction -> publication
```

冻结公式：

```text
D_cal = ceil((P95(E2E) + 5 ms) / 10 ms)
```

查看结果：

```bash
cat "$PAPER_OUT/calibration/delay.json"
```

所有 virtual 主比较、消融和 threaded full 共用该 D；禁止为某个消融单独重新标定。

## 5. 生成正式 immutable references

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  generate-references
```

生成：

- `circle`：3圈平滑圆；
- `figure8`：3圈 Gerono figure-eight；
- `fast_ellipse`：3圈高速椭圆；
- `rounded_square`：直线与圆弧连续相切并按弧长重采样；
- `preview_calibration`：不进入正式测试的独立慢速椭圆。

旧 `square` 实现及历史 robustness references 不会被修改。所有 reference 都带 horizon、最大 sweep delay 和 preview padding，并将 SHA-256 写入 reference manifest。

## 6. 标定统一 Preview IK

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  calibrate-preview \
  --preview-values 0,1,2,3,4
```

选择规则为 calibration TCP RMSE 最小；完全相同时选择更小 preview。选出的同一个 preview 用于全部四条正式轨迹，禁止在测试轨迹上重新选择。

## 7. GRU 冻结模型验证

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  validate-model \
  --num-rollouts 20 \
  --rollout-len 200
```

输出包括：

- one-step prediction with ground-truth history；
- 1/5/10/20-step open-loop q/dq RMSE；
- 每个关节的 RMSE、NMSE、R² 和 amplitude ratio；
- divergence rate；
- 冻结 evaluation action/state rollouts 及 hash。

这些 rollout 在 checkpoint 冻结后新生成，不参与训练。不要在论文中称为 sequence-decoder teacher forcing。

## 8. 构建唯一 paper manifest

完成上述步骤并提交实现后，在 clean worktree 上执行：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  build-manifest \
  --profile paper
```

正式 manifest 路径：

```text
outputs/paper_delay_aware/manifests/paper.json
```

文件不可覆盖。配置或 commit 改变时必须使用新的输出根目录，例如 `outputs/paper_delay_aware_rerun1/`。

## 9. Smoke test

Smoke 使用短轨迹、H3、8 candidates 和1次 CEM iteration，覆盖 D=0、五个 virtual protocol、Direct IK 和 threaded full：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  smoke 2>&1 | tee "$PAPER_OUT/logs/smoke.log"
```

Smoke 只验证数据流和输出字段，不能写入论文结果。
如果当前进程的 PyTorch 看不到 CUDA，virtual 和 Direct IK smoke 会改用 CPU，threaded smoke 会明确记录为 `skipped_cuda_unavailable`；正式实验机必须在 CUDA 可见时重新运行，状态保存在 `smoke/environment_status.json`。

回归测试：

```bash
python -m unittest \
  mpc.test_paper_experiments \
  mpc.test_residual_mpc \
  mpc.test_asap_planner_worker \
  mpc.test_asap_timing \
  mpc.test_logging_threaded \
  mpc.model_c.test_oracle \
  mpc.test_robustness \
  mpc.test_model_a_robustness_evaluate \
  -v 2>&1 | tee "$PAPER_OUT/logs/unit_tests.log"
```

## 10. 正式 P0 主实验

### 五控制器主比较：84 cases

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite main --resume
```

矩阵：

```text
Direct IK                         4 trajectories × 1
Ideal full, D=0                  4 × 5 paired CEM seeds
Naive delayed, D=D_cal           4 × 5
Full virtual, D=D_cal            4 × 5
Full threaded, D=D_cal           4 × 5
```

### 核心消融：80个表格 case，只有60个新增 rollout

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite ablation --resume
```

`FullVirtual` 的20个 case 与 main 完全相同，通过 fingerprint cache 复用；只新增 NoFutureAlignment、NoReanchor、NoFeedback 各20次。

## 11. 正式 P1 实验

### Delay sweep：60 cases

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite delay_sweep --resume
```

```text
protocols: full, naive_delayed
trajectories: circle, fast_ellipse
D: 0, 2, 4, 6, 8
paired CEM seeds: 0, 1, 2
```

与 main 完全相同的 D=0 或 D=`D_cal` case 会自动复用。

### Preview IK

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite preview --resume
```

Preview IK 单独成表，不扩充论文五控制器主表。

## 12. P2 Oracle upper bound

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite oracle --resume
```

该组为 learned/MuJoCo-oracle × circle/fast-ellipse × 3 seeds，共12次。它是 oracle-dynamics upper bound，不是保持控制动作不变的纯模型误差消融。

## 13. 汇总与恢复运行

单独重建全部 summary：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  summarize --suite all --bootstrap-samples 10000
```

`runs/cache/<fingerprint>/` 是 canonical rollout；`runs/indexes/` 将论文方法名称映射到 cache。`--resume` 只接受精确 fingerprint，缺文件或 hash 不一致会立即失败。

主要结果位于：

```text
summaries/main.csv
summaries/main_aggregate.csv
summaries/main_paired_bootstrap.json
summaries/latency_recovery.json
summaries/ablation.csv
summaries/ablation_aggregate.csv
summaries/ablation_paired_bootstrap.json
summaries/delay_sweep.csv
summaries/delay_sweep_aggregate.csv
summaries/preview.csv
summaries/oracle.csv
```

统计单位是 trajectory-seed case。五个 seed 只表示 CEM sampling stochasticity，不表示五个随机机器人环境。10 ms timestep 不能作为独立 bootstrap 样本。
`failure_rate` 是每个 case 是否出现过 planner failure 的二值量；瞬时 fallback 频率另存为 `planner_failure_step_rate`。

Latency recovery 定义为：

```text
(E_naive - E_full) / (E_naive - E_ideal)
```

仅当分母大于 `1e-6 m` 时报告，不截断到 `[0,1]`。

## 14. 指标定义

| 字段 | 定义 |
|---|---|
| `requested_mpc_residual` | CEM 请求并写入 packet 的 bounded residual；用于 warm start、mean shift 和 packet 传输 |
| `buffered_residual` | packet 实际用于在线重锚定的 residual；正式配置与 `requested_mpc_residual` 相同 |
| `residual_cost_semantics` | 正式配置为 `requested`：residual magnitude/velocity/acceleration cost 只作用于 CEM 请求的 MPC residual，不混入 feedback 或 safety projection lag |
| `requested_feedback_correction` | 执行层根据预测状态与实测状态计算并限幅后的在线反馈；planner rollout 假设未来该项为零 |
| `requested_total_correction` | `clip(requested_mpc_residual + requested_feedback_correction)` |
| `requested_absolute_command` | 当前 IK nominal 加 `requested_total_correction` |
| `safety_projection_offset` | `actual_command - requested_absolute_command`；表示物理执行投影修改了多少 |
| `command_nominal_offset` | `actual_command - current_nominal`，同时包含 MPC、反馈与安全投影 lag |
| `planned_q_ref` | planner 当初送入 dynamics rollout 和 tracking cost 的 actuator command；是否经过 planner-side projection 由 manifest 明确记录 |
| `planner_execution_qref_error` | `actual_command - planned_q_ref[packet_age]`；表示预测动作与真实执行动作不一致 |
| `feedback_raw` | feedback bound 前的状态反馈 |
| `projection_discrepancy` | 兼容字段，恒等于 `-safety_projection_offset` |
| `projection_active` | safety projection offset 任一关节大于 `1e-6 rad` |

所有新 rollout 写入：

```text
control_semantics_version = 2
projection_semantics_version = 2
projection_backend = shared_physical_v2
```

汇总脚本拒绝混合不同 semantics version。absolute-command 消融和
`constraint_projected_direct_ik` fallback 也使用同一个物理投影；`raw_direct_ik`
仅用于复现旧结果。planner projection 由 `--planner_projection on|off` 显式记录。
正式配置使用 `off`，但 100 Hz 执行层的 shared physical projection 始终开启。

正式 residual 数据流固定为：

```text
planner_projection = off
residual_cost_semantics = requested
packet_residual_semantics = requested
residual_feasibility_semantics = finite
nominal_command_semantics = raw_ik
```

planner rollout 假设未来在线 feedback 为零。实际执行命令仍由同一 braking-aware
physical projector 限制关节位置、速度和加速度；其修改量通过
`safety_projection_offset` 记录，planner 预测动作与实际执行动作的差异通过
`planner_execution_qref_error` 记录。因此 `off` 不表示绕过安全约束，而表示将
physical projection 明确放在 100 Hz execution safety filter 中。

选择 `off` 的依据不是让 threaded 数字看起来更好，而是同一旧 GRU、circle、D7
的配对诊断。`on` 在 seed 1 将真实 worker rate 降到约 18.8 Hz，并产生 late drop；
`off` 在两个已复核 seed 中维持约 25.9 Hz、0 late、0 expiration，同时所有命令
加速度仍不超过 12.5 rad/s²：

| planner projection | seed | virtual full / lap (mm) | threaded full / lap (mm) | threaded E2E p95 | threaded rate |
|---|---:|---:|---:|---:|---:|
| on | 0 | 36.62 / 39.91 | 41.15 / 43.99 | 62.50 ms | 19.08 Hz |
| on | 1 | 36.49 / 39.53 | 44.95 / 49.78 | 63.58 ms | 18.77 Hz |
| off | 0 | 37.07 / 37.49 | 37.61 / 39.83 | 48.72 ms | 25.82 Hz |
| off | 1 | 44.04 / 49.03 | 41.92 / 45.85 | 48.36 ms | 25.94 Hz |

`off` 的两个 paired threaded-minus-virtual 差值分别为 `+0.54/-2.13 mm`
（full）和 `+2.35/-3.17 mm`（lap）。这也说明此前观察到的“差距缩小”不是靠
把 virtual 恶化到 threaded：seed 1 中 threaded 实际优于其 paired virtual。
该配置下 planner 侧只有 joint-limit clip；诊断 rollout 中
`planned_q_ref - nominal - requested_mpc_residual` 的最大绝对误差为
`2.98e-8 rad`，所以这里的 `requested` 与 planner 实际使用的 offset 数值一致。
正式五 seed 结果仍必须写入新目录并单独汇总，不能把这两组诊断当作论文结果。

threaded 另存 `planner_events.jsonl`。每次异步结果具有唯一 `result_id`，类型为
`success_published`、`success_late_dropped`、`planner_failure` 或 `worker_fatal`；
事件只记录一次，不应对逐 tick 的持久 worker 状态求和。启动等待不计 packet
expiration，只有已激活 packet 真正耗尽且 replacement 尚未激活时才产生
`packet_expired_event`。

virtual 的固定逻辑 D 与 threaded 的真实 snapshot-to-publication E2E 必须分开报告。只有 threaded 结果可以用于 soft-real-time、late packet、planner rate、period、jitter 和 deadline-miss 结论。

## 15. Projection v2 修复后的旧 GRU D7 复核

planner/execution projection 的问题演化、zero-correction bug、seed 方差和当前尚未
冻结的设计选择详见
[`ASAP_MPC_PROJECTION_AND_SEED_VARIANCE_ISSUES.md`](ASAP_MPC_PROJECTION_AND_SEED_VARIANCE_ISSUES.md)。

下面的命令用同一旧 GRU、同一正式 circle reference、D=7 和 seeds 0--4
比较 fixed-delay virtual 与真实 threaded。结果必须写入新的 v2 目录，不得覆盖或
与旧语义日志合并。

```bash
MODEL_DIR=dynamics_modeling/outputs/checkpoints/gru_20260717_152930
REF=outputs/references/circle_3laps/reference.npz
OUT=outputs/old_gru_circle_d7_projection_v2_off

for SEED in 0 1 2 3 4; do
  python scripts/run_cem_mpc.py \
    --model_type gru --reference_mode task --device cuda \
    --multirate_mode virtual_asap --delay_protocol full \
    --anticipation_delay_steps 7 --planner_projection off \
    --residual_cost_semantics requested \
    --packet_residual_semantics requested --residual_feasibility_semantics finite \
    --nominal_command_semantics raw_ik \
    --checkpoint "$MODEL_DIR/best_model.pt" \
    --normalizer "$MODEL_DIR/normalizer.pt" \
    --reference_file "$REF" --seed "$SEED" \
    --max_execution_steps 1807 \
    --horizon 20 --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
    --feedback_kq 0.3 --feedback_kdq 0.015 \
    --save_dir "$OUT/virtual_asap/seed_$SEED"

  python scripts/run_cem_mpc.py \
    --model_type gru --reference_mode task --device cuda \
    --multirate_mode threaded_asap --delay_protocol full \
    --anticipation_delay_steps 7 --planner_guard_ms 5 \
    --planner_min_interval_ms 0 --planner_projection off \
    --residual_cost_semantics requested \
    --packet_residual_semantics requested --residual_feasibility_semantics finite \
    --nominal_command_semantics raw_ik \
    --checkpoint "$MODEL_DIR/best_model.pt" \
    --normalizer "$MODEL_DIR/normalizer.pt" \
    --reference_file "$REF" --seed "$SEED" \
    --max_execution_steps 1807 \
    --horizon 20 --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
    --feedback_kq 0.3 --feedback_kdq 0.015 \
    --save_dir "$OUT/threaded_asap/seed_$SEED"
done
```

安全硬标准为：无 NaN/Inf、无 worker fatal、命令关节/速度/加速度约束零违规，
并且注入 packet gap 时无命令尖峰。正式 D7 的可用性目标为
`all_costs_invalid=0`、首包后 expiration=0、late drop 接近零和 control deadline
miss 接近零。性能报告同时给出 paired lap-RMSE 均值差、95% bootstrap CI 与
worst-seed 差值；目标是 threaded 相对 virtual 不超过 5 mm。
