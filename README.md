# NN-MPC_RobotArm

面向 **ABB IRB 2400** 与 **UR5e** 的 MuJoCo 学习动力学、任务空间轨迹跟踪和延迟感知 CEM-MPC 仿真项目。

项目学习位置伺服机械臂的闭环动力学，并在 100 Hz 控制周期内运行以 IK 参考为锚点的 **residual CEM-MPC**。控制器可在 CUDA 后台线程中异步规划，同时在主线程持续执行物理安全投影、状态反馈与延迟补偿。除标准 Model-A learned dynamics 外，仓库还提供 Model-C 数据闭环、鲁棒性评测、Direct IK 基线和可选的集成模型不确定性感知安全监督器。

> 这是 MuJoCo 仿真研究代码。代码中的速度、加速度和安全边界是规划/仿真约束，不是 ABB 或 UR 的硬件额定值；将方法迁移到真机前需要完成独立的硬件安全审查、碰撞检测、急停和实时性验证。

## 项目能力

- 支持 ABB IRB 2400（默认）与 UR5e；`RobotSpec` 统一绑定 XML、关节/执行器顺序、home 位姿、TCP、控制周期、重力补偿和约束。
- 在闭环 position actuator 动力学上训练 MLP、GRU 或 Transformer；当前论文/鲁棒性工作流默认使用 GRU Model A。
- 支持关节空间参考和经过连续 DLS IK 验证的任务空间参考（圆、椭圆、8 字、方形等）。
- 默认使用 full-horizon residual CEM-MPC；也保留旧的无锚定 acceleration policy 以复现实验。
- 提供 `threaded_asap` 实时部署模式，以及 `virtual_asap`、`synchronous` 等确定性/消融模式。
- 支持 payload、执行器增益、外力脉冲和观测噪声扰动；实验产物带有 robot、数据集和 checkpoint 的身份校验。
- 可选的集成不确定性监督器只监控 CEM 已选轨迹，不参与 CEM 优化或模型平均，并在高风险时限制 residual 或回退到 nominal/IK。

## 系统总览

```text
task-space trajectory
        │
        ├─ continuous DLS IK + validation ───────► q_des, dq_des
        │                                             │
        │                                  raw/executable IK nominal
        │                                             │
state [q, dq] ─► Model-A CEM residual planning ─► q_nom + residual
        │                  │                          │
        │                  └─ learned dynamics rollout + cost
        │                                             │
        └──── 100 Hz feedback / delay alignment ◄────┤
                                                      ▼
                                physical position/velocity/acceleration projection
                                                      │
                                                      ▼
                                          MuJoCo position actuators

optional stage 2: primary + replica predictions of the selected trajectory
                  └─ uncertainty monitor / soft safety supervisor
```

动作始终是绝对位置执行器目标，而不是速度或力矩命令：

```text
state  x_t = [q_t(6), dq_t(6)]
action u_t = q_ref,t(6)                         # rad
target     = delta_dq,t
```

数据采集、训练、normalizer、reference 和 checkpoint 都会记录并校验机器人身份。不要把不同 RobotSpec、XML 或 actuator 语义的数据与模型混用。

## 目录

```text
configs/robots/          ABB IRB 2400 与 UR5e 的 RobotSpec YAML
dynamics_modeling/       MuJoCo XML/资产、数据采集、动力学训练与开环评估
mpc/                     CEM、rollout、约束、延迟/ASAP、IK、cost、recovery、uncertainty
scripts/                 闭环运行、参考生成、鲁棒性、Model-C 与论文实验入口
docs/                    当前架构、运行手册、安全说明与实验材料
outputs/                 根目录下的 MPC/reference/实验输出（按需生成）
```

动力学数据集、checkpoint 和开环诊断通常位于 `dynamics_modeling/outputs/`；MPC、reference 与论文实验输出位于根目录的 `outputs/`。生成结果不应写入源代码目录。

## 环境与快速检查

所有命令从仓库根目录执行。项目当前使用 conda 环境 `pendulum-rl`；依赖清单位于 [`dynamics_modeling/requirements.txt`](dynamics_modeling/requirements.txt)。

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
conda run -n pendulum-rl python -c "import mujoco, torch; print('mujoco', mujoco.__version__); print('torch', torch.__version__)"
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
```

先检查机器人配置及 MuJoCo 模型：

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --num_episodes 1 --episode_len 3 --num_envs 1 \
  --save_path dynamics_modeling/outputs/datasets/abb_smoke.npz
```

该命令还会生成同名 `.manifest.json`。后续训练会严格检查 manifest 与当前 `RobotSpec` 是否一致。

## 从动力学到闭环控制

### 1. 采集闭环位置伺服数据

采集目标是 `x_{t+1}=f(x_t,q_ref,t)`，其中 `q_ref` 是绝对关节目标。采集器会先 settle，再混合 hold、小阶跃、平滑 waypoint 和正弦目标；多环境采集只用于无图形的数据生成。

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --num_episodes 200 --episode_len 300 --num_envs 8 --action_std 0.5 \
  --save_path dynamics_modeling/outputs/datasets/abb_model_a.npz
```

新平台应传入完整的 `--robot_config`，而不是仅替换 `--model_xml`。例如 UR5e 使用 `configs/robots/ur5e.yaml`；其独立的端到端流程见[UR5e 工作流](docs/guides/ur5e-end-to-end-workflow.md)。

### 2. 训练与评估动力学

训练脚本支持 `mlp`、`gru`、`transformer`。GRU/Transformer 需要 history；checkpoint 内的模型结构、history length、target mode 与控制周期会在加载时校验。

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --data_path dynamics_modeling/outputs/datasets/abb_model_a.npz \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --epochs 100 --save_dir dynamics_modeling/outputs/checkpoints

conda run -n pendulum-rl python dynamics_modeling/scripts/eval_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --checkpoint <CHECKPOINT>/best_model.pt \
  --normalizer <CHECKPOINT>/normalizer.pt \
  --model_type gru --history_len 16
```

### 3. 生成任务空间参考并验证 IK

任务空间控制只接受预先验证过的 reference 文件。reference 中保存 TCP 轨迹、`q_des`、`dq_des`、执行步数及其身份信息；task 模式会使用其中的 `execution_steps`，而非 `--episode_len`。

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --shape figure8 --repeat_count 3 \
  --save_dir outputs/references/figure8

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/figure8/reference.npz
```

### 4. 运行默认 residual CEM-MPC

以下配置对应当前默认的 H=20、128 candidates、2 次 CEM iteration 的 full-residual 控制器。`threaded_asap` 需要 CUDA，不支持 `--visualize`；论文级实验应以端到端延迟标定值替换示例中的 `6`。

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --checkpoint <CHECKPOINT>/best_model.pt \
  --normalizer <CHECKPOINT>/normalizer.pt \
  --model_type gru --history_len 16 \
  --reference_mode task \
  --reference_file outputs/references/figure8/reference.npz \
  --horizon 20 --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --residual_parameterization full \
  --multirate_mode threaded_asap --delay_protocol full \
  --anticipation_delay_steps 6 \
  --planner_projection on --planner_projection_backend compiled \
  --planner_projection_strategy two_stage \
  --exact_task_space_cost on --stage_one_task_space_cost off \
  --cem_execute lowest_cost \
  --save_dir outputs/mpc/figure8_residual
```

输出目录包含 `rollout.npz`、`rollout.csv`、tracking/control 图、`run_summary.json`，任务空间运行还会生成 `task_tracking_summary.json`。除了跟踪误差，还应记录 planner update rate、control deadline miss、late packet drop 和 Direct-IK fallback。

## 当前 MPC 方法

### Residual CEM-MPC

默认策略 `--mpc_policy residual` 不重新搜索无约束的绝对命令轨迹，而是优化 IK nominal 周围的有界补偿：

```text
q_des[t+1:t+H]
  → q_nom (raw IK by default)
  → CEM samples r / r_max ∈ [-1, 1]
  → planner projection(q_nom + r)
  → learned dynamics rollout and cost
  → select baseline / best / mean
  → 100 Hz physical projection and execution
```

`--nominal_command_semantics raw_ik` 是默认值，因此零 residual 与 raw Direct IK nominal 一致。`--cem_execute lowest_cost` 会在 zero-residual baseline、最佳样本和最终分布均值间选择预测 cost 最低者。`legacy_acceleration`、`linear_control_points`、stage-one GPU FK 等仍可用于历史复现或消融，但不是默认方法。

### 延迟与多速率

`threaded_asap` 是本地 CUDA 部署模式：主线程以 100 Hz 施加物理投影和状态反馈，后台 worker 在完成一次 CEM 后立即利用最新 snapshot 发起下一次规划。其 planner update rate 由实际 GPU 负载决定，并非固定 20 Hz。

- `threaded_asap`：默认实时异步模式；仅支持完整 `delay_protocol`。
- `virtual_asap`：确定性的逻辑延迟消融模式，适合可重复比较。
- `synchronous` / `virtual_smooth`：固定重规划间隔的对照模式。
- `anticipation_delay_steps`：planner-to-activation 延迟；默认 6 仅是本地 fallback，正式实验必须完成 E2E 标定。

### 约束、recovery 与 Direct IK

planner projection 会让候选命令满足规划运动学；执行层还会在每个 tick 对最终 `q_ref` 应用位置、速度、加速度与 braking 安全投影。默认 residual 上界为 `[0.12, 0.10, 0.12, 0.15, 0.15, 0.20] rad`（可由 RobotSpec/CLI 覆盖）。

planner failure 会立即回退到 nominal 并 reset CEM warm start；持续 tracking-error 恶化或 residual 接近上界也会触发 recovery。命令速度/加速度达到规划上限只记作诊断，不能单独视为 recovery 事件。

Direct IK 是不加载 learned dynamics、不运行 CEM 的对照：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --controller_mode ik_direct --reference_mode task \
  --reference_file outputs/references/figure8/reference.npz \
  --ik_command_projection raw \
  --save_dir outputs/mpc/figure8_ik_direct
```

`raw` 重现历史 Direct IK 基线；需要共享物理投影时改为 `--ik_command_projection physical`。`--ik_preview_steps` 只用于单独的 Preview-IK 对照，不能把它与标准 Direct IK 混为同一基线。

## 不确定性感知安全监督器

### 设计边界

不确定性功能是 residual MPC 的**后验安全监督层**，默认关闭（`--uncertainty_mode off`）。当前实现仅支持 learned backend 的 `--controller_mode mpc --multirate_mode threaded_asap`。Model A 仍是唯一参与 CEM 优化的动力学模型；replica 不输出控制命令，也不与 Model A 平均。这样可以把模型分歧作为模型失配信号，而不改变优化目标或把多个偏差预测直接混进控制动作。

每次 CEM 选出可执行命令序列后，监督器只重放该条短 horizon 轨迹：

```text
Model-A CEM chooses one q_ref sequence
        │
        ├─ Model A cached rollout
        └─ each compatible replica rolls out the same q_ref sequence
                         │
                         ▼
          normalized inter-model RMS disagreement score
                         │
                         ▼
    monitor only / hysteretic residual limiting / nominal fallback
```

在线 score 是所有模型在已选轨迹未来状态上的逐维标准差，按 primary normalizer 的 `state_std` 标准化后，在 horizon 和 state 维度求 RMS：

```text
score = sqrt(mean_{k,d}((std_models[x̂_{t+k,d}] / state_std[d])²))
```

因此阈值与 replica 数量、训练协议、normalizer 和 `--uncertainty_horizon` 绑定，不能跨配置照搬。所有成员必须有相同的 model type、状态维度、target mode、history length 与 control period；运行时至少需要 primary 之外的两个 replica。

### 模式与状态机

| 模式 | 计算 | 对命令的影响 |
| --- | --- | --- |
| `off` | 不计算 | 标准 Model-A residual MPC |
| `ensemble_monitor` | 对已选轨迹计算分歧与风险标记 | 仅记录，不改命令 |
| `ensemble_soft_gate` | 同上 | 带迟滞的 `normal → suspected → limited → fallback` 监督 |

`ensemble_gate` 仍是 `ensemble_soft_gate` 的弃用别名。soft gate 的关键行为如下：

- score 低于 low threshold：维持正常 residual；多次低分歧后从受限/回退状态恢复，防止频繁抖动。
- score 位于 low/high 之间：进入或保持 `suspected`，只记录，不持续削弱一个尚可用的控制器。
- 连续高分歧达到 `--uncertainty_confirm_steps`：进入 `limited`，将 residual 缩放为 `--uncertainty_limited_residual_scale`（默认 0.75），并使用 reference-tracking feedback。
- 高分歧同时出现明确物理风险时：立即进入 `fallback`，residual 置零，执行 nominal/IK。
- active packet 的预测创新超过校准阈值并持续确认时：也会促成保守 `limited`；它本身不被当作立即的物理紧急事件。
- replica rollout 超过 `--uncertainty_budget_ms` 或返回非有限 score：无法信任部分结果，保守地立即 nominal fallback。

明确物理风险由 Model-A 已选预测轨迹给出，任一条件成立即可：预测关节进入 joint-limit margin、residual 接近配置上界，或下一步预测 tracking error 相对当前误差增长超过指定比例。监督器记录的 `packet_prediction_q_innovation` 是激活 packet 的预测关节位置与观测位置之差；它用于把“模型有分歧”与“模型已在当前执行中失配”区分开。

### 训练、标定与启用

先训练与 Model A 使用同一数据和超参数、但随机 seed 不同的 GRU replica。训练脚本会冻结 primary normalizer，使 score 处于共同坐标系：

```bash
conda run -n pendulum-rl python scripts/train_uncertainty_ensemble.py \
  --data_path <DATASET>.npz \
  --baseline_checkpoint <MODEL_A>/best_model.pt \
  --baseline_normalizer <MODEL_A>/normalizer.pt \
  --output_root dynamics_modeling/outputs/uncertainty_ensemble \
  --replica_seeds 101,211,307
```

在干净的 ID rollout 上计算分歧分布，再选择阈值；改变 replica、训练协议、normalizer 或 horizon 后必须重新标定：

```bash
conda run -n pendulum-rl python scripts/calibrate_uncertainty_threshold.py \
  --rollout outputs/mpc/id_nominal/rollout.npz \
  --checkpoint <MODEL_A>/best_model.pt --normalizer <MODEL_A>/normalizer.pt \
  --replica_checkpoints <R1>/best_model.pt <R2>/best_model.pt <R3>/best_model.pt \
  --replica_normalizers <R1>/normalizer.pt <R2>/normalizer.pt <R3>/normalizer.pt \
  --horizon 5 --quantile 0.95

conda run -n pendulum-rl python scripts/calibrate_uncertainty_innovation.py \
  --rollouts outputs/mpc/id_run_1/rollout.npz outputs/mpc/id_run_2/rollout.npz \
  --output outputs/calibration/innovation.json
```

建议先使用 monitor，测量其额外时延和 OOD 检出表现；确认阈值后再启用 gate：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint <MODEL_A>/best_model.pt --normalizer <MODEL_A>/normalizer.pt \
  --model_type gru --history_len 16 \
  --reference_mode task --reference_file outputs/references/figure8/reference.npz \
  --multirate_mode threaded_asap \
  --uncertainty_mode ensemble_monitor \
  --uncertainty_horizon 5 --uncertainty_budget_ms 15 \
  --uncertainty_checkpoints <R1>/best_model.pt <R2>/best_model.pt <R3>/best_model.pt \
  --uncertainty_normalizers <R1>/normalizer.pt <R2>/normalizer.pt <R3>/normalizer.pt \
  --save_dir outputs/mpc/figure8_uncertainty_monitor
```

将 `--uncertainty_mode` 改为 `ensemble_soft_gate` 后，显式提供校准得到的 `--uncertainty_low_threshold`、`--uncertainty_high_threshold`，并按需加入 `--uncertainty_innovation_threshold` 与较低的 `--uncertainty_innovation_recovery_threshold`。rollout 会记录 `uncertainty_score`、`uncertainty_state`、`uncertainty_residual_scale`、`uncertainty_high_risk_flags`、`uncertainty_gate_flags`、`packet_prediction_q_innovation` 和 stage-2 用时。

详细的安全语义与参数说明见[不确定性软安全监督器](docs/safety/uncertainty-soft-supervisor.md)。

## 多机器人、鲁棒性与实验

默认 `configs/robots/abb_irb2400.yaml`；UR5e 使用 `configs/robots/ur5e.yaml`。每个平台都需要各自独立采集数据、训练模型、生成 reference 和进行统计；跨平台结果仅应做描述性汇总。

鲁棒性配置覆盖 payload、执行器失配、外力脉冲和观测噪声等级。可复现的评测入口位于 `scripts/robustness/`，论文矩阵与图表工具位于 `scripts/paper_experiments/`；Model-C 的数据收集、dataset、benchmark 与评测工作流位于 `scripts/model_c/`。建议从以下文档开始：

- [运行命令与默认 H20 配置](docs/guides/run-commands.md)
- [Model A 鲁棒性评测](docs/guides/model-a-robustness.md)
- [Direct IK 鲁棒性评测](docs/guides/direct-ik-robustness.md)
- [Model-C 数据闭环](docs/guides/model-c-workflow.md)
- [延迟感知论文实验](docs/experiments/paper-delay-aware-experiments.md)

## 测试与文档

运行 MPC 测试套件：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl pytest -q mpc/tests
```

架构、cost、投影与伪代码见[文档导航](docs/README.md)。其中 `docs/archive/` 与 `docs/Paper_test/` 用于追溯历史设计和冻结证据，不定义当前默认控制语义。

## 许可

本仓库采用 [Apache License 2.0](LICENSE.md)。UR5e MuJoCo 资产的来源与许可证信息见 `configs/robots/ur5e.yaml` 及 `dynamics_modeling/robots/ur5e/`。
