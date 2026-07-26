# CEM-MPC 运行命令

从仓库根目录使用 `pendulum-rl` 环境。以下命令对应当前默认 residual MPC，而非历史的 unanchored acceleration policy。

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
```

## 默认配置：Paper H20 residual MPC（无 stage-one GPU FK）

先生成并验证参考：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --shape figure8 --repeat_count 3 \
  --save_dir outputs/references/figure8

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/figure8/reference.npz
```

以下是当前默认、也是 Paper robustness 使用的 MPC 配置。它保留最终 exact final-pool
MuJoCo task-space scorer，但**不**在 CEM population stage 使用 Torch/GPU FK：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt \
  --model_type gru \
  --reference_mode task \
  --reference_file outputs/paper_delay_aware_two_stage_v1/references/figure8/reference.npz \
  --horizon 20 --residual_parameterization full \
  --multirate_mode threaded_asap --delay_protocol full --anticipation_delay_steps 6 \
  --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --cem_execute lowest_cost \
  --planner_projection on --planner_projection_backend compiled \
  --planner_projection_strategy two_stage --exact_task_space_cost on \
  --stage_one_task_space_cost off --stage_one_task_compile off \
  --save_dir outputs/mpc/paper_default_figure8
```

H20 full-residual MPC 是默认方法；CEM 直接搜索每个 10 ms 控制步的 residual，
不使用 control-point 插值，也不使用 stage-one GPU FK。代码层面的 CLI 默认同样为
`--stage_one_task_space_cost off --stage_one_task_compile off`。

以下命令才是可选的 GPU stage-one task-space 消融（只在最后一次 CEM iteration、
7 个时间点参与搜索），不能替代上述默认配置：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt \
  --model_type gru \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --horizon 20 \
  --stage_one_task_space_cost gpu_budgeted \
  --stage_one_task_steps 0,3,6,9,12,15,19 \
  --planner_projection on \
  --planner_projection_strategy two_stage \
  --exact_task_space_cost on \
  --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --multirate_mode threaded_asap \
  --anticipation_delay_steps <H20_FULL_GPU_CALIBRATED_D> \
  --save_dir outputs/mpc/h20_full_gpu_circle
```

该方法的 CEM mean/std、projection、GRU rollout、cost 和 delay-aware packet
均使用完整 `[20,6]` 序列；GPU scorer 仅在最后一轮对 `[128,7,6]` sparse pose
计算 task cost。启用前必须单独标定 E2E delay；GPU stage-one 仅支持 CUDA task
reference，并继续由 MuJoCo exact final pool 最终裁决。若测试 compile，附加
`--stage_one_task_compile on`，并将该首轮 warm-up 排除在标定外。

task 模式使用 reference 的 `execution_steps`，因此不需要也不会使用 `--episode_len`。`threaded_asap` 需要 CUDA，且不支持 `--visualize`；主实验请记录 control deadline miss、planner update rate、late packet drop 和 fallback。若要进行确定性逻辑延迟消融，显式使用 `--multirate_mode virtual_asap`。

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
  --multirate_mode threaded_asap \
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
