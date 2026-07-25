# NN-MPC_RobotArm

面向 ABB IRB 2400 的 MuJoCo 学习动力学与 CEM-MPC 项目。动力学模型学习位置执行器的闭环状态转移；默认控制器是 **residual MPC**：以 IK nominal command 为锚点，只搜索有界补偿，而不是重新生成一条无锚定的绝对命令轨迹。最终发送到执行器的命令始终经过共享的物理安全投影。

## 快速开始

在仓库根目录运行：

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
```

依赖与动力学数据采集、训练和开环评估说明见 [dynamics_modeling/README.md](dynamics_modeling/README.md)。

冻结 Model A 在 payload、执行器失配、外力和观测噪声下的 threaded-asap 鲁棒性评测见 [Model A 鲁棒性手册](docs/guides/model-a-robustness.md)。

完整的文档分区与入口见 [文档导航](docs/README.md)。

## 当前 MPC 方法

MuJoCo 和 learned dynamics 的动作语义始终是绝对 position-actuator target：

```text
state   = [q(6), dq(6)]
action  = q_ref(6)     # absolute actuator reference, rad
target  = delta_dq(6)
```

任务空间参考先经连续 DLS IK 生成并验证 `q_des`、`dq_des`。在时刻 `t`，默认 residual MPC 的流程是：

```text
q_des[t+1:t+H]
    -> raw IK nominal q_nom
    -> CEM samples normalized residual r / r_max in [-1, 1]
    -> planner projection of q_nom + r
    -> learned dynamics rollout
    -> residual joint-space cost
    -> 100 Hz physical projection -> execute q_ref in MuJoCo
```

默认 `--nominal_command_semantics raw_ik` 保留原始 IK nominal；零 residual 因而对应 raw Direct IK nominal。planner 会在候选 rollout 中考虑约束，执行层则在每个 100 Hz tick 对最终命令施加位置、速度、加速度与 braking 安全投影。默认 `--cem_execute lowest_cost` 会比较 baseline、CEM best sample 和最终 CEM mean，执行预测 cost 最低的候选。

`--mpc_policy legacy_acceleration` 仍可复现实验中的无锚定命令加速度动作空间，但不是默认方法。

## 推荐运行

推荐的本地部署基线是 CUDA 上的 **threaded ASAP residual MPC**：预测 horizon 20、128 candidates、2 次 CEM iteration、batch 128。主线程以 100 Hz 持续重锚定 residual 并施加状态反馈；后台 CUDA worker 在每次求解完成后使用最新 snapshot 尽快启动下一次规划，因此实际 planner update rate 由 GPU 负载决定，而不是固定 20 Hz。下面的 6 步激活延迟是本地 fallback；论文正式实验必须使用 E2E 延迟标定得到的 `D_cal`。

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint outputs/checkpoints/gru_20260720_202923/best_model.pt \
  --normalizer outputs/checkpoints/gru_20260720_202923/normalizer.pt \
  --model_type gru --history_len 16 \
  --reference_mode multi_joint_sine \
  --episode_len 200 \
  --horizon 20 \
  --multirate_mode threaded_asap \
  --anticipation_delay_steps 6 \
  --num_samples 128 \
  --cem_iters 2 \
  --rollout_batch_size 128 \
  --mpc_policy residual \
  --exact_task_space_cost off \
  --cem_execute lowest_cost \
  --save_dir outputs/mpc/joint_sine_residual
```

`threaded_asap` 需要 `--device cuda`（默认值）。主实验除 tracking 指标外，应同时报告 planner update rate、control deadline miss、late packet drop 和 Direct-IK fallback。若需要可重复的逻辑延迟消融，可显式传入 `--multirate_mode virtual_asap`；该模式不是默认部署控制器。`--exact_task_space_cost` 默认 `on`，只能与 task-space reference 一起使用；因此上面的 joint-space 示例显式关闭它。

参数含义：

- `episode_len`：非 task 模式的执行控制步数；task 模式改用 reference 的 `execution_steps`。
- `horizon`：每拍预测与优化的未来控制步数。
- `num_samples`：每个 CEM iteration 的候选序列数量；residual 模式含 forced baseline 和 mean 候选。
- `cem_iters`：每个控制步的 CEM 更新次数。
- `rollout_batch_size`：一次模型前向计算的候选上限；可大于 `num_samples`，但不会增加候选数。
- `cem_execute`：`mean` 执行最终分布均值，`best` 执行最低 cost sample，`lowest_cost` 比较 baseline、best、mean 后执行最低者；推荐后者。

输出目录包含 `rollout.npz`、`rollout.csv`、跟踪/控制图、`run_summary.json`。结束报告中的 `recovery triggers` 是 recovery 的总触发次数；`recovery active steps` 是 nominal 回退实际执行的步数。

## 任务空间参考与 Direct IK

生成并验证三圈圆形参考：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --shape circle --repeat_count 3 \
  --save_dir outputs/references/circle_3laps

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/circle_3laps/reference.npz
```

运行 residual MPC：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint outputs/checkpoints/gru_20260720_202923/best_model.pt \
  --normalizer outputs/checkpoints/gru_20260720_202923/normalizer.pt \
  --model_type gru --history_len 16 \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --multirate_mode threaded_asap \
  --mpc_policy residual --cem_execute lowest_cost \
  --save_dir outputs/mpc/task_circle_residual
```

Direct IK baseline 不加载 learned dynamics 或 CEM。默认 `raw` 语义直接发送后继 `q_des`：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --controller_mode ik_direct \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --save_dir outputs/mpc/task_circle_ik_direct
```

如需将 Direct IK 目标先通过共享的物理投影，添加 `--ik_command_projection physical`。Preview IK 仅用于单独对照：`--ik_preview_steps P` 会发送未来第 `P` 步的 IK 参考，用于检验 MPC 收益是否仅来自相位提前；它不运行 learned dynamics 或 CEM。标准 Direct IK 基线保持 `--ik_preview_steps 0`。

对比两次运行的 `task_tracking_summary.json` 和 `run_summary.json`。task 模式忽略 `episode_len`，使用参考文件的 `execution_steps`。本地图形桌面可添加 `--visualize`，关闭窗口会停止并保存部分 rollout。

## 安全与 recovery

默认 residual bound 为 `[0.12, 0.10, 0.12, 0.15, 0.15, 0.20] rad`。命令速度/加速度上限是 MuJoCo 规划上限而非 ABB 硬件额定值；达到这些上限只记录诊断，不会单独触发 recovery。

Residual MPC 在以下情况下回退到 `q_nom` 并 reset CEM warm start：planner failure 立即触发；跟踪误差持续恶化或 residual 持续接近 bound 时按 `recovery_consecutive_steps` 触发。默认持续步数为 3，冷却期为 5 步。

最新的可选不确定性感知安全监督器以 Model A 与多个独立 GRU replica 对已选轨迹的预测分歧作为信号。`ensemble_monitor` 只记录诊断；`ensemble_soft_gate` 在持续高分歧且存在具体物理风险时限制 residual，必要时回退到 nominal。它默认关闭，使用前应先训练 replica 并在 ID 数据上标定阈值；完整流程见[不确定性软安全监督器](docs/safety/uncertainty-soft-supervisor.md)与[replica 训练手册](docs/guides/model-a-replica-training-5090.md)。

## 目录与测试

```text
dynamics_modeling/  ABB XML、数据采集、训练与开环评估
mpc/                CEM、nominal projection、rollout、cost、recovery、IK
scripts/            闭环 MPC 与参考生成命令
docs/               当前规范、历史设计和实验材料
outputs/            生成的模型、参考和运行结果
```

快速验证 residual MPC 单元测试：

```bash
conda run -n pendulum-rl python -m unittest mpc/tests/test_residual_mpc.py -v
```

完整项目测试入口见 [dynamics_modeling/README.md](dynamics_modeling/README.md)。

## 文档入口

- 理解系统：[项目结构](docs/architecture/project-structure.md)、[Cost function](docs/architecture/cost-function.md)、[MPC 伪代码](docs/architecture/mpc-pseudocode.md)、[planner projection](docs/architecture/planner-projection.md)。
- 运行与复现：[运行命令](docs/guides/run-commands.md)、[Direct IK 鲁棒性](docs/guides/direct-ik-robustness.md)、[Model A 鲁棒性](docs/guides/model-a-robustness.md)。
- 论文实验：[论文实验总计划](docs/experiments/paper-test-plan.md)、[Delay-Aware MPC 操作手册](docs/experiments/paper-delay-aware-experiments.md)。
- 历史设计与实验记录：[历史资料索引](docs/archive/README.md)。历史资料不定义当前默认控制语义。
