# NN-MPC_RobotArm 已完成内容整理

> **历史实现快照（部分控制细节已过时）**：本文大部分内容记录的是 residual MPC 引入前的 `delta_q_ref` / command-acceleration 实现，不是当前控制器规范。当前默认实现为 `q_ref=q_nom+r`：先投影 IK nominal `q_nom`，CEM 优化有界 residual `r`，并默认以 `lowest_cost` 比较 baseline、best 和 mean。当前规范请见 [README](../README.md)、[CostFunction](CostFunction.md) 和 [MPC 伪代码](mpc-pseudocode.md)。本文保留旧描述以便追溯已完成模块和历史实验。

本文档基于当时工作区代码和已有文档整理，目标是说明项目在该阶段完成了哪些模块、每个模块做到什么程度，以及这些模块之间如何串成研究/实验流程。

当前项目已经不只是一个单独的 learned dynamics 训练仓库，而是形成了从 MuJoCo 机械臂建模、数据采集、神经网络动力学训练、open-loop 评估，到 closed-loop learned CEM-MPC 控制与模型对比分析的完整雏形。

## 1. 项目整体阶段

当前仓库已经完成了两条主线：

1. `dynamics_modeling/` 中的 learned dynamics 主线：负责 ABB IRB 2400 机械臂 MuJoCo 环境、数据采集、数据诊断、模型训练和 open-loop 评估。
2. 顶层 `mpc/` 与 `scripts/` 中的 learned CEM-MPC 主线：负责加载训练好的 learned dynamics，在 MuJoCo 中闭环运行 CEM-MPC，并保存 rollout、图像和对比指标。

从功能上看，项目已经具备：

- 一个基于 ABB IRB 2400 的 6 自由度 MuJoCo 仿真环境。
- 以 position actuator 目标关节角 `q_ref` 为动作的数据采集系统。
- MLP、GRU、Transformer 三类 learned dynamics 模型。
- 单步预测和多步 rollout 训练/评估能力。
- 顶层 CEM-MPC 闭环控制 pipeline。
- MPC-induced 数据收集、Model A/B/C 对比和 OOD 分析脚本。
- 较完整的单元测试覆盖，当前测试在正确环境配置下可以通过。

## 2. 仓库组织与职责划分

项目已经完成了比较清晰的模块拆分。

顶层目录的主要职责如下：

- `mpc/`：CEM-MPC 的可复用控制逻辑，包括 CEM 控制器、planner rollout、cost function、约束、参考轨迹和日志保存。
- `scripts/`：顶层 MPC 实验入口，包括运行闭环 MPC、收集 MPC 诱导数据、比较多个模型和分析 OOD。
- `tests/`：顶层 MPC pipeline 的单元测试。
- `docs/`：项目说明、结构说明、MPC 伪代码和当前完成情况文档。
- `dynamics_modeling/`：MuJoCo 机械臂模型、learned dynamics 包、数据采集/训练/评估脚本、诊断工具和对应测试。

`dynamics_modeling/` 内部也已经拆分为几类内容：

- `ABB_IRB2400.xml`：默认机械臂 MuJoCo XML。
- `abb_irb2400_assets/`：XML 使用的 ABB IRB 2400 visual STL mesh。
- `neural_dynamics/`：当前主线 dynamics package，被顶层 MPC pipeline 使用。
- `scripts/`：当前主线数据采集、训练、评估、诊断和可视化脚本。
- `tests/`：learned dynamics、MuJoCo 环境和工具函数测试。

当前推荐默认路径已经明确：

- dynamics 数据采集、训练和 open-loop 评估从 `dynamics_modeling/` 目录运行。
- 顶层 MPC 实验从仓库根目录运行。
- 新的 MPC 输出写到顶层 `outputs/...`。
- dynamics 旧数据、旧 checkpoint 和旧图像仍然可以显式引用 `dynamics_modeling/outputs/...`。

## 3. MuJoCo 机械臂环境

项目已经完成了 ABB IRB 2400 机械臂的 MuJoCo 环境封装，核心实现位于 `dynamics_modeling/neural_dynamics/mujoco_env.py`。

已经完成的环境能力包括：

- 默认加载 `ABB_IRB2400.xml` 作为 6 关节机械臂模型。
- 环境状态定义为 `[q, dq]`，即 6 个关节角和 6 个关节速度。
- 动作定义为 6 维 absolute actuator target joint angle，也就是 `q_ref`。
- 当前只支持 `position` actuator 控制模式，旧的 velocity control 已经被显式拒绝。
- 自动读取 actuator ctrlrange 和 joint range，并暴露 `action_low/action_high`、`joint_low/joint_high`。
- 在 `step()` 中对动作做 actuator range 裁剪。
- 每个 control step 内按 `frame_skip` 调用 MuJoCo step。
- 默认启用重力补偿，在每个 substep 前根据当前位姿计算 `qfrc_bias` 并写入 `qfrc_applied`。
- 对第 6 关节的重力补偿做特殊置零处理，避免不合理补偿引入问题。
- 每次 step 后检查真实 `qpos` 是否越过 joint limit，越界时抛出清晰错误。
- 支持 `compute_torque_components()`，可以拆分并记录：
  - `actuator_tau`
  - `gravity_tau`
  - `total_tau`
- 支持末端位置读取，尝试 site/body 名称包括 `ee_site`、`tool0`、`flange`、`end_effector`。

XML 和 asset 层面已经完成：

- `ABB_IRB2400.xml` 使用 6 个 position actuator，且 actuator ctrlrange 与 joint range 对齐。
- XML 中保留了 visual ground plane。
- 机器人 visual mesh 使用 `abb_irb2400_assets/` 下的 STL 文件。
- 当前 STL mesh 主要用于显示，不作为质量、惯量或碰撞的主要来源。
- XML 中使用显式 inertial 参数，测试中验证总质量在 ABB IRB 2400 合理范围内。
- `link_5` 和 `link_6` 的质量/惯量经过特别检查，避免腕部关节速度尖峰。

## 4. 数据采集系统

项目已经完成了 current position-control 数据采集流程，核心实现在 `dynamics_modeling/neural_dynamics/parallel_collector.py`，入口脚本包括：

- `dynamics_modeling/scripts/collect_data.py`
- `dynamics_modeling/scripts/collect_transformer_data.py`

已完成的数据采集能力包括：

- 支持单环境采集，便于调试 MuJoCo XML、actuator 和状态维度。
- 支持多进程并行采集，提高大规模数据生成速度。
- 支持 `--append`，可以把新采集数据追加到已有 `.npz` 数据集。
- append 前会校验已有数据集结构，避免拼接不兼容数据。
- 支持 `settle_steps`，每个 episode 开始后先用 `q_ref = q` 稳定机械臂，再开始记录样本。
- reset 时使用安全工作空间，避免从极端关节位姿开始。
- 每个 episode 采样结构化而不是完全随机的 `q_ref` 轨迹。
- `q_ref` 轨迹包含多种 motion mode，例如 hold、小阶跃、平滑随机 waypoint 和正弦轨迹。
- `q_ref` 会经过归一化空间采样和一阶滤波，避免每一步目标角剧烈跳变。
- 采集时保存真实状态转移 `x_t, q_ref_t, x_{t+1}`。
- 采集时可保存力矩分解和轨迹诊断字段。

当前主要数据 schema 已经包括：

- `states`：当前状态 `[q, dq]`。
- `actions`：动作，当前语义为 absolute `q_ref`。
- `next_states`：下一步状态 `[q_next, dq_next]`。
- `episode_ids`：episode 边界，用于 GRU/Transformer 历史窗口。
- `q_ref`：与 `actions` 等价，显式记录当前 position-control 语义。
- `delta_q_ref`：相邻控制目标变化量。
- `tau_actuator`：position actuator 产生的关节力矩。
- `tau_gravity`：重力补偿力矩。
- `tau_total`：总力矩。
- `action_std_normalized`：样本对应的归一化动作采样强度。
- `settle_steps`：episode 开始前稳定步数。
- `motion_mode_ids`：采样轨迹模式。
- `termination_reasons`：采集终止原因。

这意味着当前数据已经不只是简单的 `(state, action, next_state)`，而是带 episode 边界、控制语义、力矩信息和数据覆盖诊断元数据的训练数据。

## 5. 数据处理与诊断工具

项目已经完成了多种数据辅助工具：

- `merge_datasets.py`：合并多个 `.npz` 数据集。
- `plot_dataset_q_dq.py`：绘制数据集中 `q/dq` 分布或轨迹。
- `diagnose_dynamics_data.py`：对采集数据做覆盖范围、joint limit margin、归一化分布、q_ref tracking 和计数统计诊断。
- `diagnose_rollout_spikes.py`：分析 learned dynamics rollout 中的尖峰、误差来源、limit 距离和 Jacobian 相关信息。
- `convert_collada_to_stl.py`：把 Collada DAE mesh 转成 MuJoCo 可加载的 binary STL。
- `rollout_visualize.py`：打开 MuJoCo viewer 可视化结构化 `q_ref` 运动。
- `dataset_merge.py`：包内提供 `.npz` dataset merge 逻辑。

这些工具覆盖了数据生成后常见的检查流程：

- 数据是否覆盖足够的 joint/action 范围。
- 轨迹是否靠近或越过 joint limit。
- `q_ref` 和真实 `q` 是否存在明显异常。
- 力矩是否存在极端分布。
- episode 边界和历史窗口是否可用于序列模型。
- rollout 误差是否集中在某些维度或某些时刻。

## 6. Learned Dynamics 模型

项目已经完成了三类神经网络动力学模型，实现在 `dynamics_modeling/neural_dynamics/models.py`。

### 6.1 MLPDynamics

MLP 模型用于单步输入：

```text
[q_t, dq_t, q_ref_t] -> target
```

已完成特性：

- 输入维度为 `state_dim + action_dim`。
- 默认使用多层 Linear + SiLU。
- 输出维度可配置，支持完整 `delta_state` 或只输出 `delta_dq`。
- 输入 shape 校验明确。

### 6.2 GRUDynamics

GRU 模型用于历史序列输入：

```text
history_len steps of [q, dq, q_ref] -> target
```

已完成特性：

- 使用 `nn.GRU` 处理历史窗口。
- 默认读取最后一个 hidden output 做预测。
- 支持可配置层数和输出维度。
- 输入必须是 `[batch, history_len, input_dim]`。

### 6.3 TransformerDynamics

Transformer 模型用于历史序列输入：

```text
history_len steps of [q, dq, q_ref] -> target
```

已完成特性：

- 使用 Linear embedding + learnable positional embedding。
- 使用 `nn.TransformerEncoder`。
- 最后一个 token 的 encoded 表征送入 head。
- 支持 `max_history_len` 检查，避免超出位置编码范围。
- 输出维度可配置。

## 7. Dataset 与历史窗口处理

项目已经完成了通用数据集封装，核心在 `dynamics_modeling/neural_dynamics/dataset.py`。

已完成能力包括：

- `DynamicsDataset` 支持 MLP、GRU、Transformer 三类模型。
- 支持 `target_mode="delta_state"` 和 `target_mode="delta_dq"`。
- MLP 直接读取当前样本。
- GRU/Transformer 根据 `history_len` 构造连续历史窗口。
- 如果数据集包含 `episode_ids`，序列窗口不会跨越 episode reset 边界。
- 对没有 `episode_ids` 的旧数据保留 backward compatibility。
- 严格检查 `states/actions/next_states` rank、长度和 shape。
- `RolloutDynamicsDataset` 支持为 rollout loss 提供未来动作序列和真实下一状态序列。
- `split_dataset()` 支持 train/val split。
- split 时支持 `train_sample_stride` 和 `val_sample_stride`，可以加速大数据集训练。
- 如果有 episode 边界，stride 会尽量按 episode 内部采样，避免破坏序列语义。

## 8. Normalizer 与状态重建

项目已经完成标准化工具 `StandardNormalizer`，用于训练和推理一致的数据缩放。

已完成能力包括：

- 分别拟合：
  - `state_mean/state_std`
  - `action_mean/action_std`
  - `delta_mean/delta_std`
- 支持单步输入归一化。
- 支持序列输入归一化。
- 支持 target delta 的归一化和反归一化。
- 支持保存到 `normalizer.pt` 和从 checkpoint 加载。
- 对未 fit 或缺字段的 normalizer 给出明确错误。

状态重建逻辑位于 `neural_dynamics/integration.py`，已经支持根据模型输出 target mode 重建下一状态。当前训练和 MPC rollout 中都复用同一套重建逻辑，避免训练/推理不一致。

## 9. 训练系统

训练入口为 `dynamics_modeling/scripts/train_dynamics.py`，当前已经支持比较完整的训练流程。

已完成训练能力包括：

- 支持 `--model_type mlp/gru/transformer`。
- MLP 自动使用 `history_len=1`。
- 支持 `target_mode=delta_state` 和 `target_mode=delta_dq`，当前默认是 `delta_dq`。
- 支持 `control_dt`，用于从 `delta_dq` 重建下一状态。
- 训练前默认校验当前数据集是否包含 position-control 必需字段：
  - `states`
  - `actions`
  - `next_states`
  - `q_ref`
  - `delta_q_ref`
- 默认检查 `actions == q_ref`，确保动作语义是 absolute target joint angle。
- 对 GRU/Transformer 默认要求 `episode_ids`，避免历史窗口跨 reset。
- 支持 `--no_require_q_ref_dataset` 兼容旧实验数据。
- 支持 MSE 和 Huber loss。
- 支持 `q_weight`、`dq_weight` 以及 per-joint extra weights。
- 支持 rollout loss：
  - `rollout_loss_steps`
  - `rollout_loss_weight`
  - `rollout_loss_discount`
- 支持 AMP 混合精度训练。
- 支持 DataLoader worker 和 pin memory。
- 支持从完整 checkpoint resume。
- 支持从旧模型权重初始化新训练。
- 禁止同时使用 `--resume_checkpoint` 和 `--init_from_checkpoint`。
- 保存训练 config 到 `config.yaml`。
- 保存 normalizer 到 `normalizer.pt`。
- 保存 `best_model.pt`、`latest_model.pt`。
- 使用 rollout loss 时还可保存 `best_rollout_model.pt`。
- 每个 epoch 打印 train/val loss 和分关节 RMSE 表。

checkpoint 中已经保存：

- `model_state_dict`
- `config`
- optimizer state
- AMP scaler state
- metadata，包括 epoch、best val loss、checkpoint type 等。

## 10. Learned Dynamics 推理与 Rollout

项目已经完成统一的 learned dynamics rollout 工具，核心在 `dynamics_modeling/neural_dynamics/rollout.py`。

已完成能力包括：

- `load_dynamics_bundle()` 可一次性加载：
  - model
  - normalizer
  - model type
  - history length
  - state/action dimension
  - target mode
  - control dt
  - checkpoint config
- 加载 checkpoint 时检查 checkpoint 中的 state/action dimension 是否与 `n_joints` 匹配。
- 序列模型优先使用 checkpoint config 中的 `history_len`。
- `rollout_dynamics_batch()` 支持批量预测多条候选未来 `q_ref` 序列。
- 支持 `rollout_batch_size` chunking，避免 CEM 大 batch 预测时显存/内存过大。
- 对 MLP 使用当前预测 state 和当前 action 作为输入。
- 对 GRU/Transformer 使用历史窗口，并在 rollout 中把最后一个 token 的 action 替换为候选 absolute `q_ref`。
- rollout 每一步用统一的 `reconstruct_next_state()` 重建预测下一状态。
- 返回 shape 为 `[batch, horizon + 1, state_dim]` 的预测轨迹。

这一部分已经直接被顶层 MPC planner 使用。

## 11. Open-loop 评估系统

项目已经完成 learned dynamics 的 open-loop 和 teacher-forcing 评估脚本，主要入口是 `dynamics_modeling/scripts/eval_dynamics.py`。

已完成能力包括：

- 从 checkpoint 和 normalizer 加载 MLP/GRU/Transformer。
- 自动读取 checkpoint config 中的：
  - `target_mode`
  - `output_dim`
  - `control_dt`
  - `history_len`
- 在 MuJoCo 中采集真实 rollout。
- 支持 warmup steps。
- 支持多种 horizon，例如 `1,5,10,20,50,200`。
- 支持 open-loop prediction。
- 支持 teacher forcing prediction。
- 输出每个 rollout 的：
  - 关节角 `q` 对比图。
  - 关节速度 `dq` 对比图。
  - state L2 error 图。
  - torque component 图。
- 计算整体 RMSE、q RMSE、dq RMSE、per-dimension RMSE、NMSE、R2 和 amplitude ratio。
- 支持把 metric rows 保存为 CSV。

这说明项目已经具备检查 learned model 是否能在真实 MuJoCo rollout 上保持预测稳定性的基础工具。

## 12. CEM-MPC 控制器

顶层 CEM-MPC 控制器位于 `mpc/cem_controller.py`，已经完成可复用的 Cross-Entropy Method optimizer。

已完成能力包括：

- `CEMMPCConfig` 封装 horizon、action_dim、num_samples、elite ratio、CEM iteration、初始标准差、最小标准差、平滑系数、时间相关噪声系数、随机种子和设备。
- `CEMMPCController` 维护跨 timestep 的 sampling mean/std。
- 每次规划会采样 `[num_samples, horizon, action_dim]` 的候选 `delta_q_ref` 序列。
- 噪声支持 temporal correlation，使相邻 timestep 的候选动作更平滑。
- 每轮 CEM 调用 planner evaluate 得到 cost。
- 自动过滤非 finite cost。
- 选择 elite samples 更新 mean/std。
- 跨 control step warm-start：规划结束后把 best sequence 左移作为下一步 mean。
- 如果 planner 异常、cost shape 错误、所有 cost 无效或选中动作无效，会 fallback 到上一拍 `previous_q_ref`。
- `CEMMPCResult` 返回：
  - 实际执行的 `q_ref`
  - 第一拍 `delta_q_ref`
  - best cost
  - elite mean cost
  - planning time
  - failure flag
  - failure reason
  - best sequence

控制器本身不关心 learned dynamics，也不关心 cost 细节，只要求 planner 提供 `evaluate(candidate_delta_q_ref)`。这个边界已经比较清楚。

## 13. Planner Rollout 与动作语义

顶层 planner 位于 `mpc/planner_rollout.py`，负责把 CEM 候选动作翻译为 learned dynamics 可预测的未来轨迹。

已经完成的关键设计是：

```text
candidate_delta_q_ref -> actuator_q_ref_sequence -> learned dynamics rollout -> tracking cost
```

具体能力包括：

- `construct_actuator_q_ref_sequence()` 支持两种模式：
  - `delta`：候选序列表示相对增量。
  - `absolute`：候选序列直接表示 absolute `q_ref`。
- delta 模式支持两种 base：
  - `previous_q_ref`
  - `current_q`
- 支持 `delta_q_ref_max` 限幅。
- 支持 `delta_rate_limit`。
- 支持 `q_ref_rate_limit`。
- 支持 joint limit margin，并最终裁剪到安全 joint range 内。
- `LearnedDynamicsPlanner.evaluate()` 会：
  - 从 history 中读取当前 `q`。
  - 构造 absolute actuator `q_ref_sequences`。
  - 调用 `rollout_dynamics_batch()` 得到 predicted states。
  - 调用 `joint_space_tracking_cost()` 计算 cost。
  - 返回 cost、q_ref sequences 和 predicted states。

这里最重要的完成点是：CEM 优化变量和真实 actuator 命令已经分离。CEM 可以优化平滑增量 `delta_q_ref`，而 learned dynamics 和 MuJoCo 仍然接收真实可执行的 absolute `q_ref`。

## 14. Cost Function

顶层 cost function 位于 `mpc/cost_functions.py`。

已经完成的 cost 结构为：

```text
cost =
  w_q * normalized position tracking error
+ w_dq * normalized velocity tracking or damping error
+ w_u_offset * normalized actuator-to-predicted-position offset
+ w_dqref * normalized first difference of q_ref
+ w_ddqref * normalized second difference of q_ref
+ w_terminal * normalized terminal position tracking error
+ w_joint_limit * softplus joint-limit safety barrier
```

已完成能力包括：

- 预测状态输入 shape 检查。
- 自动从 `pred_states` 中拆分 `q_pred` 和 `dq_pred`。
- `q_des` 和 `dq_des` 支持 horizon 对齐；位置尺度由参考轨迹振幅和 `q_tol` 确定。
- 默认 `velocity_cost_mode=track` 跟踪 `dq_des`；未提供 `dq_des` 或使用 `damping` 模式时，惩罚预测速度大小。
- `w_u_offset` 惩罚 `actuator_q_ref - q_pred`，避免命令与预测关节位置出现过大偏置。
- `w_dqref` 惩罚 `q_ref` 相邻步的一阶差分，第一项相对 `previous_q_ref`；`w_ddqref` 惩罚其二阶差分。
- `w_terminal` 强调 horizon 末端的归一化位置跟踪误差。
- `w_joint_limit` 根据预测 `q` 到最近上下限的距离施加 softplus barrier，并在进入 `joint_limit_safe_margin` 前开始增加。
- `joint_space_tracking_cost()` 接收 `delta_q_ref` 参数，但当前实现不直接用它计算代价。

因此当前 MPC 目标已经不是单纯追踪，而是在 tracking、速度趋势、控制幅度、控制平滑性和 joint safety 之间折中。

## 15. Constraints 与 Reference Trajectory

顶层约束工具位于 `mpc/constraints.py`。

已完成能力包括：

- `joint_bounds_with_margin()`：根据 joint limit margin 得到收缩后的可用范围，并检查 margin 不会让范围为空。
- `clip_to_joint_limits()`：把 `q_ref` 裁剪到 joint limit 内。
- `apply_rate_limit()`：限制 absolute `q_ref` 相对上一拍的变化。
- `apply_delta_rate_limit()`：限制 `delta_q_ref` 序列内部的变化。

参考轨迹工具位于 `mpc/reference.py`。

已完成 reference mode 包括：

- `hold`：保持初始关节角。
- `step`：在 episode 约三分之一处切换到随机目标。
- `joint_sine`：只让第一个关节做正弦运动。
- `multi_joint_sine`：多个关节使用随机频率、相位和幅值做正弦运动。
- `task`：加载离线验证通过的任务空间 `ReferenceBundle`；其中的 continuous DLS IK 输出仍是 joint-space `q_des`、`dq_des`。

另外已经实现：

- `finite_difference_dq()`：从 `q_des` 通过有限差分生成 `dq_des`。
- reference 最终会裁剪到 joint low/high 内。
- 任务空间 reference 支持 circle、ellipse、figure8 和 square，包含零位保持、joint-space 安全离开、task-space approach、三圈图形、return、joint-space 返回零位和 horizon padding。
- 当前 XML 的 TCP 为 `ee_site`，与兼容 site `tool0` 共位；mesh geom 没有碰撞位，因此 reference 诊断将 self-collision 明确标记为 `not_available`。

## 16. MPC Logging 与输出

顶层日志保存位于 `mpc/logging.py` 和 `mpc/utils.py`。

已完成能力包括：

- `save_mpc_run()` 会保存：
  - `rollout.npz`
  - `rollout.csv`
  - diagnostic figures
- 输出图包括：
  - `q_tracking.png`：真实 `q`、期望 `q_des` 和控制输入 `actuator_q_ref`。
  - `dq.png`：真实关节速度。
  - `tracking_error.png`：`||q - q_des||`。
  - `control.png`：各关节 `q_ref`。
  - `planning_diagnostics.png`：planning time 和 best cost。
- `build_history_tensor()` 可以把最近状态和 `q_ref` 拼接成序列模型需要的 history。
- history 不足时会用最早一帧前填充，保证长度固定。
- `write_csv_rows()` 支持把每步诊断 row 写为 CSV。
- task mode 还保存 desired/actual TCP pose、position/orientation error、segment/lap id，以及 task-space plots 和 `task_tracking_summary.json`。

## 17. 顶层 Closed-loop CEM-MPC 脚本

闭环 MPC 主入口是 `scripts/run_cem_mpc.py`。

已经完成的 closed-loop 流程包括：

1. 解析 checkpoint、normalizer、model type、MPC 参数、cost 权重和保存路径。
2. 处理顶层路径和 `dynamics_modeling/` 旧输出路径。
3. 自动选择 CPU/CUDA。
4. 加载 learned dynamics bundle。
5. 创建 MuJoCoArmEnv。
6. reset 到固定零位 home pose `[0, 0, 0, 0, 0, 0] rad`。
7. 用 `q_ref = current q` settle 若干步。
8. 生成 joint-space reference，或加载预验证的 task-space reference file。
9. 每个 control step 构造 history tensor。
10. 创建/更新 LearnedDynamicsPlanner。
11. 使用 CEMMPCController 规划动作。
12. 执行选中的第一拍 `q_ref`。
13. 记录实际状态、下一状态、reference、动作、delta action、planning time、best cost、elite cost、failure flag、joint limit violation 和 torque。
14. 循环执行 receding horizon closed-loop control。
15. 保存 rollout arrays、CSV 和图像。

保存的主要 arrays 包括：

- `actual_states`
- `next_states`
- `q_des`
- `dq_des`
- `actuator_q_ref`
- `delta_q_ref`
- `planning_time`
- `best_cost`
- `elite_mean_cost`
- `failure_flags`
- `joint_limit_violation_flags`
- `realized_tracking_error`
- `predicted_real_error_gap`
- `tau_actuator`
- `tau_gravity`
- `tau_total`
- task mode 额外包含 desired/actual TCP pose、TCP position/orientation errors、segment ids 和 lap ids。

这说明当前项目已经具备完整的 learned dynamics closed-loop 控制实验入口。

## 18. MPC-induced 数据收集

脚本 `scripts/collect_mpc_data.py` 已经完成把 closed-loop MPC rollout 转成训练数据的流程。

已完成能力包括：

- 复用 `run_cem_mpc.py` 的参数和闭环运行逻辑。
- 把 rollout 中的完整 transition 转成：
  - `states`
  - `actions`
  - `next_states`
- `actions` 使用 MPC 实际输出的 `actuator_q_ref`。
- 保存 `episode_ids`。
- 保存额外字段：
  - `q_ref`
  - `delta_q_ref`
  - `tau_actuator`
  - `tau_gravity`
  - `tau_total`
  - `motion_mode_ids`
  - `reference_mode_ids`
  - `mpc_cost`
  - `planning_time`
  - `failure_flags`
  - `source_policy`
- 支持 `--append` 追加到已有数据集。

这一部分为后续 Model C 或 MPC-induced distribution 再训练提供了基础。

## 19. Model A/B/C 对比脚本

脚本 `scripts/evaluate_model_abc.py` 已经完成多个 checkpoint 在统一 MPC 设置下的比较框架。

已完成能力包括：

- 支持重复传入 `--model_spec`。
- 每个 model spec 格式为：

```text
label,checkpoint,normalizer,model_type[,dataset_path]
```

- 对每个模型使用相同的 CEM-MPC 参数运行 closed-loop rollout。
- 每个模型单独保存完整 MPC 输出到对应 label 目录。
- 汇总每个模型的：
  - steps
  - tracking_error_mean
  - tracking_error_final
  - failure_rate
  - planning_time_mean
  - best_cost_mean
  - predicted_real_error_gap_mean
  - dataset_path
- 输出 `model_abc_summary.csv`。

这说明项目已经有了用于比较 baseline 模型、追加数据模型和 MPC-induced 模型的实验框架。

## 20. MPC OOD 分析

脚本 `scripts/analyze_ood_mpc.py` 已经完成 MPC 查询分布与训练数据分布的对比工具。

已完成能力包括：

- 支持多个 `--pair`，格式为：

```text
label,mpc_rollout_npz,training_dataset_npz
```

- 支持额外传入 `--baseline_dataset`。
- 从训练数据计算 state/action mean 和 std。
- 对 MPC rollout 中的 `actual_states` 和 `actuator_q_ref` 计算 z-score。
- 汇总：
  - state z-score mean/max
  - state OOD fraction
  - action z-score mean/max
  - action OOD fraction
  - failure rate
  - predicted-real error gap mean
- 支持自定义 `z_threshold`。
- 输出 CSV summary。

这部分用于判断 MPC 规划时查询的状态/动作是否偏离训练数据分布，是分析 closed-loop 性能下降的重要工具。

## 21. 已有文档

当前 `docs/` 下已经有几份文档：

- `docs/PROJECT_STRUCTURE.md`：说明仓库结构、路径约定、dynamics 与 MPC 的职责划分。
- `docs/mpc-pseudocode.md`：把当前 MPC 主线翻译成伪代码，解释 closed-loop receding horizon、planner、CEM、cost 和脚本关系。
- `docs/Note.md`：记录 2026-07-03 的历史问题和 cost function 理解笔记；其中的 cost 表述不作为当前实现规范。
- `docs/PROJECT_COMPLETION_STATUS.md`：本文档，整理当前项目已经完成的内容。

`dynamics_modeling/README.md` 也已经比较详细地记录了：

- 环境准备。
- 数据采集。
- 多环境并行。
- 模型训练。
- Transformer 数据。
- 继续训练。
- 模型评估。
- 数据格式。
- 常见命令。

## 22. 测试覆盖与当前验证状态

项目已有任务空间 IK/reference 测试以及原有 MPC、dynamics 测试：

- `tests/test_cost_functions.py`
- `tests/test_mpc_pipeline.py`
- `tests/test_ik_solver.py`
- `tests/test_task_space_reference.py`
- `tests/test_reference_pipeline.py`
- `tests/test_task_space_mpc_smoke.py`
- `dynamics_modeling/tests/test_core.py`

顶层 MPC 测试覆盖内容包括：

- cost 的位置归一化、actuator offset、命令一阶/二阶平滑、joint-limit barrier 和 velocity damping。
- 顶层 runtime path 解析。
- legacy dynamics 输出路径解析。
- bare XML 路径解析。
- OOD 输出路径解析。
- `delta_q_ref` 到 cumulative absolute `q_ref` 的转换。
- learned dynamics batch rollout 是否使用 absolute `q_ref`。
- rollout chunked 与 unchunked 结果一致性。
- `delta_dq` 状态重建一致性。
- CEM 遇到无效 cost 时 fallback 到上一拍 `q_ref`。

learned dynamics 核心测试覆盖内容包括：

- 默认模型路径解析。
- 缺失 XML 的错误提示。
- ABB IRB 2400 XML inertial 合理性。
- visual mesh asset 引用。
- ground plane。
- position actuator 和 joint range 对齐。
- 默认 gravity compensation。
- velocity control 被拒绝。
- torque component 分解。
- joint position limit 检查。
- Collada 到 STL 转换。
- dataset、rollout dataset、normalizer、checkpoint、loss、诊断和 v2 package 的多项核心行为。

当前验证命令和结果：

```bash
python -m pytest tests/test_cost_functions.py tests/test_mpc_pipeline.py dynamics_modeling/tests/test_core.py -q
```

结果：失败。当前 shell 中没有 `python` 命令。

```bash
python3 -m pytest tests/test_cost_functions.py tests/test_mpc_pipeline.py dynamics_modeling/tests/test_core.py -q
```

结果：失败。系统 Python 缺少 `torch`。

```bash
conda run -n pendulum-rl python -m pytest tests/test_cost_functions.py tests/test_mpc_pipeline.py dynamics_modeling/tests/test_core.py -q
```

结果：失败。pytest 自动加载 ROS `launch_testing` 插件时缺少 `lark`。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl python -m pytest tests/test_cost_functions.py tests/test_mpc_pipeline.py tests/test_ik_solver.py tests/test_task_space_reference.py tests/test_reference_pipeline.py tests/test_task_space_mpc_smoke.py dynamics_modeling/tests/test_core.py -q
```

结果：通过。

```text
115 passed, 8 subtests passed in 20.69s
```

因此当前项目测试本身在 `pendulum-rl` 环境下可以通过，但建议运行 pytest 时禁用外部插件自动加载，避免 ROS 环境插件污染。

## 23. 当前已经形成的完整实验链路

目前项目已经能串成下面这条完整链路：

1. 在 MuJoCo ABB IRB 2400 position-control 环境中采集 `q_ref` 数据。
2. 对数据做覆盖、joint limit、q_ref tracking 和 torque 诊断。
3. 用 `.npz` 数据训练 MLP/GRU/Transformer learned dynamics。
4. 保存 checkpoint、normalizer 和 config。
5. 对 learned dynamics 做 open-loop 或 teacher-forcing 评估。
6. 在顶层 CEM-MPC 中加载 learned dynamics。
7. CEM 采样候选 `delta_q_ref` 序列。
8. planner 把候选增量转换成 absolute `q_ref`。
9. learned dynamics 预测每条候选控制序列的未来状态。
10. cost function 对候选未来轨迹评分。
11. 控制器执行最优序列第一拍。
12. 在真实 MuJoCo 环境中闭环推进。
13. 保存 rollout、CSV、tracking 图、control 图和 planning diagnostics。
14. 把 MPC rollout 转成 MPC-induced training dataset。
15. 比较多个模型的 closed-loop MPC 表现。
16. 分析 MPC 查询分布相对训练数据是否 OOD。

这条链路说明项目已经从“训练一个动力学模型”推进到了“用 learned dynamics 做闭环控制实验”的阶段。

## 24. 目前还不应算作已完成的内容

下面这些内容虽然已经有代码入口或自然延伸方向，但不能根据当前仓库状态直接算作已经完成的实验结论：

- 尚未在本文档验证具体 checkpoint 的 closed-loop tracking 性能。
- 尚未在本文档确认某个 Transformer checkpoint 是最终最佳模型。
- 尚未在本文档确认 Model A/B/C 的最终实验结论。
- 尚未在本文档确认 MPC-induced 数据再训练后一定优于 baseline。
- 尚未在本文档确认 OOD 分析的具体数值结论。
- 尚未在本文档确认真实机器人部署能力；当前项目仍是 MuJoCo 仿真 pipeline。

当前最自然的后续工作是：

1. 确认 `dynamics_modeling/outputs/` 下可用 checkpoint 和 dataset。
2. 运行一个小规模 `scripts/run_cem_mpc.py` smoke rollout。
3. 如果闭环运行正常，收集 MPC-induced 数据。
4. 训练或比较 Model A/B/C。
5. 用 `scripts/analyze_ood_mpc.py` 判断 MPC 查询是否偏离训练分布。
6. 根据 closed-loop 结果回到数据采集、loss 权重、cost 权重或 reference 设计继续迭代。
