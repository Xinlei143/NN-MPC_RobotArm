# NN-MPC_RobotArm

这是一个面向 ABB IRB 2400 的 MuJoCo 学习动力学与 CEM-MPC 项目。系统以六轴位置执行器的绝对关节目标 `q_ref` 为控制输入，学习关节状态转移，并在闭环中用 CEM 优化未来的 `q_ref` 增量序列。

当前支持两类参考：

- 关节空间参考：`hold`、`step`、`joint_sine`、`multi_joint_sine`。
- 任务空间参考：`circle`、`ellipse`、`figure8`、`square`。该类参考先离线执行固定姿态连续 DLS IK，生成并验证关节空间 `q_des`、`dq_des`，再接入现有 MPC。

## 快速开始

在仓库根目录运行顶层 MPC 命令：

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
```

项目使用 Python 3.10+、MuJoCo、PyTorch、NumPy 和 Matplotlib。依赖清单位于：

```bash
conda run -n pendulum-rl pip install -r dynamics_modeling/requirements.txt
```

完整的数据采集、模型训练和开环评估说明见 [dynamics_modeling/README.md](dynamics_modeling/README.md)。

## 系统概览

MuJoCo 状态、动作和学习目标使用以下语义：

```text
state   = [q(6), dq(6)]
action  = q_ref(6)，绝对位置执行器目标，单位为 rad
target  = delta_dq(6)
```

Transformer/GRU 的输入是由 `[q, dq, q_ref]` 组成的历史序列 token。学习动力学输出 `delta_dq`，滚动预测通过速度积分重建未来状态。CEM 采样未来的 `delta_q_ref`，将其转换为满足约束的绝对 `q_ref`，用学习动力学滚动预测未来，并通过关节空间代价选出第一拍命令。

```text
q_des, dq_des
    -> CEM 候选 delta_q_ref
    -> 满足约束的绝对 q_ref 序列
    -> 学习动力学滚动预测
    -> 关节空间跟踪代价
    -> 在 MuJoCo 执行第一拍 q_ref
```

默认 MPC 初始配置为：

```text
q0 = [0, 0, 0, 0, 0, 0] rad
dq0 = [0, 0, 0, 0, 0, 0] rad/s
q_ref0 = [0, 0, 0, 0, 0, 0] rad
```

## 目录结构

```text
dynamics_modeling/  ABB XML、资产、数据采集、训练与评估
mpc/                      CEM 控制器、规划器、代价、约束、IK 与参考轨迹
scripts/                  闭环 MPC 与离线任务参考命令
tests/                    MPC、IK、参考轨迹与集成测试
outputs/                  生成的模型、参考、滚动结果与图像
docs/                     设计笔记和详细项目文档
```

关键模块：

- `mpc/cem_controller.py`：CEM 采样、elite 更新、warm start 与 fallback。
- `mpc/planner_rollout.py`：候选命令约束和学习动力学滚动预测。
- `mpc/cost_functions.py`：归一化位置/速度跟踪、命令偏置、平滑、终端和关节限位代价。
- `mpc/kinematics_utils.py` 与 `mpc/ik_solver.py`：私有 `MjData` FK/Jacobian 和有界连续 DLS IK。
- `mpc/reference_pipeline.py`：任务参考组装、离线验证与 `.npz` 持久化。

## 关节空间 MPC

从固定初始位姿运行一次闭环 MPC：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --reference_mode multi_joint_sine \
  --horizon 20 \
  --num_samples 1024 \
  --cem_iters 4 \
  --rollout_batch_size 256 \
  --save_dir outputs/mpc/joint_sine_run
```

输出目录包含 `rollout.npz`、`rollout.csv`、关节跟踪图、控制图和规划诊断图。

## 任务空间参考与 MPC

ABB XML 中的 `ee_site` 与兼容 site `tool0` 共位。任务空间参考必须离线生成并通过验证，MPC 只加载已验证的参考文件。

生成三圈圆形参考：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --shape circle \
  --repeat_count 3 \
  --save_dir outputs/references/circle_3laps
```

独立验证已有参考：

```bash
conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/circle_3laps/reference.npz
```

使用已验证的任务空间参考运行 MPC：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --save_dir outputs/mpc/task_circle
```

`--reference_mode task` 使用参考文件中的 `execution_steps`；该模式下 `--episode_len` 不生效。参考文件包含 `q_des`、`dq_des`、`ddq_des`、期望 TCP 位姿、IK 诊断、阶段 id、圈数 id 和预测时域尾部填充。

任务空间模式还会保存 TCP 位置/姿态误差、`ee_trajectory_3d.png`、平面投影图、按阶段/圈数统计的图像以及 `task_tracking_summary.json`。

## 任务空间参考验证

每条参考都从零关节位姿开始并返回零关节位姿。由于 ABB 全零配置的完整 6D Jacobian 存在奇异性，流程会先通过确定性的离线搜索寻找满足 `sigma_min >= 0.10` 的邻近 `q_safe`，随后采用五次关节空间离开和返回段。

参考生成器会拒绝以下情况：

- DLS 不收敛或 FK 误差超过阈值。
- 硬关节限位违规。
- 单关节速度或加速度超过阈值。
- 任务空间段的 `sigma_min < 0.005`。
- 关节空间不连续，或每圈 TCP/关节闭合失败。
- 未返回零位姿和零速度。

当前 MuJoCo XML 为所有 mesh geom 关闭了 collision bit。因此自碰撞会标记为 `not_available`，不构成已验证的安全保证。

## 数据、训练和模型对比

动力学数据集的核心字段为：

```text
states
actions
next_states
episode_ids
q_ref
delta_q_ref
tau_actuator
tau_gravity
tau_total
```

使用 `scripts/collect_mpc_data.py` 将闭环 MPC 滚动结果转成额外的动力学数据。使用 `scripts/evaluate_model_abc.py` 在相同 MPC 设置下比较多个 checkpoint，使用 `scripts/analyze_ood_mpc.py` 比较滚动结果的状态/动作查询与训练数据分布。

进行任务空间实验时，Model A/B/C 必须复用同一个已验证 `reference.npz`。任务空间路径可能超出当前学习动力学训练分布；解释跟踪结果前应先进行 OOD 分析。

## 测试

禁用外部 pytest 插件后运行完整相关测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl python -m pytest \
  tests/test_cost_functions.py \
  tests/test_mpc_pipeline.py \
  tests/test_ik_solver.py \
  tests/test_task_space_reference.py \
  tests/test_reference_pipeline.py \
  tests/test_task_space_mpc_smoke.py \
  dynamics_modeling/tests/test_core.py -q
```

当前实现下，该测试集结果为 `115 passed, 8 subtests passed`。

## 当前限制

- 学习动力学与 CEM 目标仍处于关节空间；TCP 跟踪只作为评估指标，不是 MPC 代价。
- 任务空间 IK 使用 previous-solution warm start 的局部 DLS，不枚举 ABB analytic IK 多解，也不联合优化整条轨迹。
- 当前 XML 没有启用自碰撞或外部障碍物模型。
- 现有 checkpoint 使用关节空间数据训练。大范围任务空间参考可能 OOD，需要降低速度、在新任务轨迹下采集数据，并训练 Model C。
- 通过参考/IK 验证不代表已有学习动力学 checkpoint 一定能在闭环中稳定跟踪。

## 相关文档

- [项目结构](docs/PROJECT_STRUCTURE.md)
- [MPC 伪代码](docs/mpc-pseudocode.md)
- [完成状态](docs/PROJECT_COMPLETION_STATUS.md)
- [命令示例](docs/run_command.md)
