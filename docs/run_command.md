# CEM-MPC 运行命令

从仓库根目录使用 `pendulum-rl` 环境。以下命令对应当前默认 residual MPC，而非历史的 unanchored acceleration policy。

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
```

## 任务空间 residual MPC

先生成并验证参考：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --shape figure8 --repeat_count 3 \
  --save_dir outputs/references/figure8

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/figure8/reference.npz
```

推荐运行：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --reference_mode task \
  --reference_file outputs/references/figure8/reference.npz \
  --horizon 20 --replan_interval_steps 5 \
  --multirate_mode virtual_asap --anticipation_delay_steps 6 \
  --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --cem_execute lowest_cost \
  --save_dir outputs/mpc/task_figure8_residual
```

task 模式使用 reference 的 `execution_steps`，因此不需要也不会使用 `--episode_len`。本地图形环境可在命令末尾添加 `--visualize`。

## IK direct baseline

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --controller_mode ik_direct \
  --reference_mode task \
  --reference_file outputs/references/figure8/reference.npz \
  --save_dir outputs/mpc/task_figure8_ik_direct
```

该模式直接将后继 `q_des` 发送给 position actuator；不需要 checkpoint、normalizer 或 CEM。与 residual run 对比 `task_tracking_summary.json` 的 TCP 误差和 `run_summary.json` 的 joint tracking。

## 关节空间 smoke run

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer --reference_mode multi_joint_sine \
  --episode_len 200 --horizon 10 --num_samples 128 --cem_iters 3 \
  --rollout_batch_size 128 --mpc_policy residual --cem_execute lowest_cost \
  --save_dir outputs/mpc/joint_sine_residual
```

## CEM 参数

- `--cem_execute lowest_cost`：比较 zero-residual baseline、best sample、最终 mean；推荐默认。
- `--cem_execute best`：执行本轮最低 cost sample，探索更激进。
- `--cem_execute mean`：执行最终分布均值，仅建议作为消融。
- `--uniform_sample_ratio 0.15`：默认将 15% 的非 forced candidates 从 `[-1,1]` 均匀采样。
- `--reset_std_each_step`：可选地每拍恢复 `init_std`；默认不加，即继承 warm-start std，但不会低于 `min_std=0.25`。
- `--rollout_batch_size` 可以大于 `num_samples`，但只决定一次模型前向的最大 batch，不增加搜索候选。

## Recovery 与诊断

默认 recovery 参数为：

```text
recovery_error_ratio=1.25
recovery_min_tracking_error=0.05 rad
recovery_residual_fraction=0.95
recovery_consecutive_steps=3
recovery_cooldown_steps=5
```

planner failure、持续 tracking-error growth、持续 residual saturation 会触发 recovery；命令速度或加速度贴近上限只作为 violation/diagnostic 记录。结束报告和 `run_summary.json` 提供 recovery 的总触发次数、原因分布和 active steps。

## 历史消融

`--mpc_policy legacy_acceleration`、`--cem_execute mean` 与旧的 absolute-command smoothness 权重仅用于复现实验。它们不是当前方法的推荐起点；使用时请单独记录 policy、cost profile 和全部 CLI 参数。

## 不确定性感知 threaded ASAP-MPC

该流程固定 primary Model-A checkpoint 做完整 CEM 搜索；四个相同结构、不同训练 seed 的 GRU 只在 CEM 选出 baseline/best/mean/selected 分支后进行少量 rollout。五模型预测的归一化状态分歧超过阈值时，系统发布 zero-residual packet，严格执行可行 IK nominal `q_nom`，并重置 CEM warm start。

先训练四个 replica（这不会修改 baseline checkpoint）：

```bash
conda run -n pendulum-rl python scripts/train_uncertainty_ensemble.py
```

训练完成后，`outputs/uncertainty_ensemble/ensemble_manifest.json` 会记录四个 checkpoint 与 normalizer 路径。将其中的四对路径传入 threaded ASAP：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint outputs/checkpoints/gru_20260720_202923/best_model.pt \
  --normalizer outputs/checkpoints/gru_20260720_202923/normalizer.pt \
  --model_type gru \
  --multirate_mode threaded_asap \
  --horizon 20 --replan_interval_steps 5 --anticipation_delay_steps 6 \
  --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --cem_execute lowest_cost \
  --uncertainty_mode ensemble_gate \
  --uncertainty_checkpoints REPLICA_1.pt REPLICA_2.pt REPLICA_3.pt REPLICA_4.pt \
  --uncertainty_normalizers REPLICA_1_normalizer.pt REPLICA_2_normalizer.pt REPLICA_3_normalizer.pt REPLICA_4_normalizer.pt \
  --uncertainty_threshold 0.10 \
  --reference_mode multi_joint_sine \
  --save_dir outputs/mpc/uncertainty_threaded_asap
```

`uncertainty_score`、`uncertainty_max_score`、`uncertainty_evaluation_time`、`uncertainty_gate_flags` 会写入 rollout。报告 planning time 时包含集成检查耗时；threaded ASAP 仍用 packet publish deadline 判定 late drop，因此不满足实时预算的 plan 不会被错误执行。`0.10` 只是初始安全阈值，训练完成后应利用 development benchmark 分布标定阈值，再锁定 final benchmark。
