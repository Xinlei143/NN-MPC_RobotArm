# SO101 CUDA Graph active 实机测试记录

日期：2026-08-09
机器人：SO101 follower；机器人控制频率：30 Hz
模型：u-q epoch14（`delta_state`，GRU，history length 16）
参考轨迹：`circle_p0`，H=6，D=2

## 1. 本次完成的修改

本次测试没有修改模型、cost function 或 H。使用的是前面完成的 executable-command semantics 修复和 CEM 延迟优化：

1. Planner projection、ASAP D=2 forecast 和 SO101 backend 统一调用 canonical executable-command state machine。
2. 状态使用上一拍真正 transmitted 的 q_ref 和由 transmitted q_ref 计算的 command velocity。
3. Planner packet 保存 `requested_q_ref_sequence`、canonical `q_ref_sequence` 和 `expected_raw_sequence`。
4. CUDA CEM rollout 改为 GPU-resident 的 projection → GRU 交错执行；固定形状使用 CUDA Graph replay。
5. D=2 anchor forecast 使用一次批量 GPU 路径，避免逐步 Python/GPU 同步调用。
6. exact final pool 直接携带选中轨迹的 raw Goal_Position，避免 worker 再次重放选中轨迹。
7. `mpc_preview_nominal_steps=0` 明确表示 `tick t -> reference[t]`，与 Direct IK 同 tick。

## 2. 实机测试命令

```bash
conda run --no-capture-output -n lerobot python scripts/run_real_cem_mpc.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --real-mode active_mpc \
  --checkpoint dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/normalizer.pt \
  --model_type gru \
  --history_len 16 \
  --reference_mode joint_file \
  --reference_file outputs/hardware/so101_pre_mpc/20260808_refs_6phase/circle_p0/joint_reference_mpc.npz \
  --reference-manifest outputs/hardware/so101_pre_mpc/20260808_refs_6phase/mpc_manifest.json \
  --delay-calibration outputs/hardware/so101_pre_mpc/20260808_shadow/delay_calibration.json \
  --ood-envelope outputs/hardware/so101_pre_mpc/20260808_shadow/ood_envelope.json \
  --horizon 6 \
  --num_samples 128 \
  --cem_iters 2 \
  --rollout_batch_size 128 \
  --planner_projection on \
  --planner_projection_backend compiled \
  --planner_projection_strategy two_stage \
  --executable_rollout_backend cuda_graph \
  --mpc_warmup_plans 3 \
  --mpc_preview_nominal_steps 0 \
  --nominal_command_semantics executable_ik \
  --asap_history_mode aligned \
  --asap_snapshot_mode tick_start \
  --mpc_policy residual \
  --cem_execute lowest_cost \
  --feedback_kdq 0 \
  --feedback_max 0.00872665 \
  --residual_max 0.034906585 \
  --device cuda \
  --seed 10 \
  --save_dir outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active \
  --enable-motion \
  --operator-supported-shutdown
```

模型文件和 normalizer 必须来自同一个目录：

```text
dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/
```

## 3. 测试结果

结果目录：

```text
outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/
```

### 3.1 安全与 deadline

| 指标 | 结果 |
|---|---:|
| 记录 tick | 1087 |
| planner publication | 1004 |
| startup Direct gate | 前 83 tick |
| planner failure | 0 |
| late drop | 0 / 1004 |
| packet expiry | 0 |
| control deadline miss | 0 |
| OOD invalid tick | 0 |

D=2 的发布 deadline 为：

```text
2 × 33.333 ms − 5 ms guard = 61.667 ms
```

本次 planner 端到端延迟为：

| 指标 | 结果 |
|---|---:|
| mean | 20.19 ms |
| P50 | 19.53 ms |
| P95 | 26.62 ms |
| P99 | 28.67 ms |
| max | 36.80 ms |

因此最大值距离 deadline 仍有约 24.9 ms 裕量。

### 3.2 延迟分解

| 阶段 | mean | P95 |
|---|---:|---:|
| CEM search | 16.89 ms | 18.77 ms |
| D=2 anchor forecast | 1.33 ms | 1.71 ms |
| worker queue wait | 1.78 ms | 6.71 ms |
| packet postprocess | 0.044 ms | 0.047 ms |

CEM search 延迟与 CUDA Graph 离线 benchmark（约 16.7 ms P95）一致，说明 `cuda_graph` backend 在本次实机运行中生效。

### 3.3 executable-command 一致性

| 检查项 | 结果 |
|---|---:|
| planner expected raw match | 1087 / 1087 |
| live reproject | 0 |
| Direct fallback | 0 |
| projected q_ref → transmitted q_ref P95 | 约 0.04° |

请求经过安全 projector 的偏移仍存在，但这是速度/加速度约束产生的有意 projection；最终 projected command 到 transmitted command 的偏差已经接近 encoder quantization，而不是旧链路的 projector state mismatch。

### 3.4 跟踪效果

| 指标 | 结果 |
|---|---:|
| overall position RMSE | **0.931°** |
| maximum position error | 3.579° |
| shoulder pan RMSE | 1.344° |
| shoulder lift RMSE | 0.891° |
| elbow RMSE | 1.097° |
| wrist flex RMSE | 0.719° |
| wrist roll RMSE | 0.120° |

Active 运动段约 90% tick 使用了非零 residual，residual L2 平均约 0.44°、最大约 1.90°，没有达到 2° residual 上限饱和。

与此前同一 circle reference 的 Direct baseline（约 0.920°）相比，本次 Active 为 0.931°，暂时没有改善；shoulder pan 仍然是主要退化关节。单次运行不能据此判断统计显著性，但可以确认延迟和 command projection 已不是当前主要瓶颈。

## 4. 结论

本次测试确认：

- CUDA Graph 优化将 planner 维持在约 20 ms mean、26.6 ms P95，deadline 完全通过；
- planner、ASAP 和 hardware 的 raw-count executable command 一致性已通过；
- active residual 已经实际执行，并非全部回退 Direct；
- 但模型预测的 residual 在真机上没有稳定降低 tracking error，尤其是 shoulder pan。

因此下一步重点应放在 model/cost 的 counterfactual ranking 和 residual trust gate，而不是继续优化通信或 CUDA 延迟。

说明：实机入口目前没有把完整 CEM 配置字段写入 `run_summary.json`，所以其中部分顶层字段显示为 `NaN` 或 `not_applicable`；本报告的 planner 与 timing 数字来自同目录的 `rollout.npz` 和 `planner_events.jsonl`。
