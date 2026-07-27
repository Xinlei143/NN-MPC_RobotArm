# UR5e：从数据采集到 CEM-MPC 的完整流程

本文是 UR5e 第二机械臂验证的可执行手册。目标是在 **独立的 UR5e
MuJoCo plant、数据集、GRU、reference 和延迟标定** 上，运行与 ABB 共用的
residual CEM-MPC 算法。它不训练跨机器人共享模型，也不替换 ABB 的正式冻结流程。

所有命令均从仓库根目录执行，并假设 Conda 环境名为 `pendulum-rl`：

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
conda run -n pendulum-rl python --version
```

下面用四个变量简化路径。Shell 关闭、新开终端或重新连接 SSH 后需重新设置：

```bash
UR_CFG=configs/robots/ur5e.yaml
DATA=dynamics_modeling/outputs/datasets/ur5e_model_a.npz
CKPT_ROOT=dynamics_modeling/outputs/checkpoints/ur5e
EXP=outputs/paper_ur5e_v1

printf 'UR_CFG=<%s>\nDATA=<%s>\nCKPT_ROOT=<%s>\nEXP=<%s>\n' \
  "$UR_CFG" "$DATA" "$CKPT_ROOT" "$EXP"
```

输出中的每一项都必须在尖括号内有值；若例如出现 `EXP=<>`，停止执行后续命令并重新设置变量。否则 `--output-root "$EXP"` 会被传入空字符串，后续阶段会在错误位置寻找 reference、标定和输出文件。

## 1. 先理解哪些东西必须成套使用

UR5e 的运行链如下：

```text
RobotSpec
  ├─> XML + TCP + home + 重力补偿 + 物理限制
  ├─> dataset.npz + dataset.manifest.json
  ├─> GRU checkpoint + normalizer + artifact_manifest.json
  └─> task references + delay calibration + paper manifest
                                      └─> CEM-MPC runs + summaries
```

`configs/robots/ur5e.yaml` 是这一链的根。它固定了六个关节和 actuator 的前缀顺序、
非零 home pose、`ee_site`、10 ms 控制周期、全轴重力补偿、数据采集范围和 UR5e
专属物理限制。训练或运行时，脚本会比较 RobotSpec/XML/dataset manifest 的 SHA256；
因此不要把 ABB 的数据、normalizer、checkpoint 或 reference 传给 UR5e 命令。

初始 UR5e profile 的数据区间相对 home 分别为：reset
`[0.35, 0.30, 0.35, 0.45, 0.45, 0.60] rad`，target
`[0.70, 0.55, 0.70, 0.90, 0.90, 1.20] rad`。采集器还会和 XML joint limit
及 margin 求交集。

## 2. 阶段 0：模型和 reference smoke test

先确认 UR5e 模型本身可用；这一步不需要数据或神经网络：

```bash
conda run -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" \
  validate-robot

conda run -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" \
  generate-references --calibration-steps 2400
```

验收文件：

```text
$EXP/diagnostics/robot_contract.json
$EXP/references/manifest.json
$EXP/references/{circle,figure8,fast_ellipse,rounded_square,delay_calibration}/reference.npz
```

`delay_calibration` 使用与 ABB 工作流相同的独立小椭圆和长度计算规则。默认 `--calibration-steps 2400`，生成约 2400 个实际控制步，与现有 ABB 标定 reference 的 2405 步基本一致，使 500-sample 正式标定通常能在单个 threaded worker 生命周期内完成；该轨迹不进入正式 tracking 结果。

`robot_contract.json` 应显示 `nq=nv=nu=6`、`control_dt=0.01`，并记录 500 步 home
hold。每条 reference 的首点和执行段末点必须是 UR5e home；生成失败时先处理 IK 或
关节限制问题，不要绕过验证或改用 ABB reference。

四条正式测试任务与 ABB 工作流使用相同的 task-space 定义：`circle`（半径 0.05 m、3 圈、3 s/圈）、`figure8`（轴长 0.05/0.03 m、3 圈、3 s/圈）、`fast_ellipse`（半轴 0.055/0.03 m、3 圈、2 s/圈）和 `rounded_square`（半边长 0.03 m、圆角 0.008 m、3 圈、3 s/圈）。因此跨平台比较的是同样的 TCP 任务尺度和速度，而不是同一组关节角。

不要复用 ABB 的 `reference.npz`：它包含 ABB XML、TCP、home、IK 分支及关节轨迹的身份信息，UR5e loader 会正确拒绝它。UR5e reference 由同一 task-space 参数在其自身的 safe departure pose、运动学与关节限制下重新求解 IK，故其 `q_des`、`dq_des` 和绝对 TCP 坐标可以不同。

若只想检查一条较短的轨迹，可直接调用生成器：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --robot_config "$UR_CFG" \
  --shape circle --repeat_count 1 --circle_radius 0.02 --lap_duration 2.0 \
  --horizon 20 --lookahead_steps 16 \
  --save_dir outputs/references/ur5e/circle_smoke
```

## 3. 阶段 1：小数据 smoke test

先采一个很小的数据集，确认环境、采集 workspace、manifest 和数据字段没有问题。这个
数据集仅用于管线验证，不应用于正式论文模型：

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config "$UR_CFG" \
  --num_episodes 2 --episode_len 32 --settle_steps 10 --num_envs 1 --seed 0 \
  --save_path dynamics_modeling/outputs/datasets/ur5e_smoke.npz
```

检查同目录中自动生成的 `ur5e_smoke.manifest.json`。它至少应包含：

- `robot_identity.robot_id == "ur5e"`；
- dataset SHA256、`state_dim=12`、`action_dim=6`；
- `control_dt=0.01`；
- home-centered collection workspace 和 motion-mode counts。

可做覆盖诊断；该工具尚未直接读取 RobotSpec，所以显式传入 UR5e XML：

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/diagnose_dynamics_data.py \
  --data_path dynamics_modeling/outputs/datasets/ur5e_smoke.npz \
  --model_xml dynamics_modeling/robots/ur5e/ur5e_project.xml --n_joints 6 \
  --coverage_dir outputs/diagnostics/ur5e_smoke_coverage
```

若 smoke collect 失败，优先检查 `$UR_CFG` 中的 home、joint/actuator names、XML 的
`ee_site` 和控制周期。不要继续到训练阶段。

## 4. 阶段 2：正式采集 Model A 数据

正式数据应是 UR5e 独立数据，不能 append 到 ABB 数据集。以下是起始配置；`num_envs`
应按机器可用 CPU 调整，通常取物理核心数的一半附近：

```bash
conda run --no-capture-output -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config "$UR_CFG" \
  --num_episodes 2000 --episode_len 300 --settle_steps 50 \
  --num_envs 8 --action_std 0.5 --seed 20260727 \
  --save_path "$DATA"
```

这会生成：

```text
$DATA
${DATA%.npz}.manifest.json
```

数据保存的是 position-actuator 闭环转移：

```text
state      = [q, dq]          # 12 dimensions
action     = q_ref            # 6 dimensions, absolute joint target
next_state = f(state, q_ref)
```

每个 episode 先从 profile 的 reset 区间开始，`q_ref=q` settle 后，再混合 hold、step、
smooth random、sine、delta-reference 和 correlated-random 动作。正式采集后至少检查：

1. `termination_reasons` 中没有大量异常终止；
2. 各关节的 q、dq、q_ref 覆盖包含后续 reference 附近的区域；
3. target 没有长期卡在 joint/actuator limit；
4. `motion_mode_sample_counts` 不是单一模式；
5. dataset manifest 的 `robot_identity` 与 `$UR_CFG` 相同。

要增加样本时，使用完全相同的 RobotSpec 向 **同一个 UR5e 文件** append：

```bash
conda run --no-capture-output -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config "$UR_CFG" \
  --num_episodes 1000 --episode_len 300 --settle_steps 50 \
  --num_envs 8 --action_std 0.5 --seed 20260728 \
  --append --save_path "$DATA"
```

append 会重新写 dataset hash，并保留 `collection_runs`。若 profile/XML 已变，严格身份
检查会拒绝 append；这是预期行为，应新建数据集并重新训练。

## 5. 阶段 3：训练独立 GRU

推荐先训练一个与 MPC 匹配的 GRU：history 16、`delta_dq` target、10 ms 控制周期、
5-step rollout loss。`rollout_loss_weight` 必须大于零才会真的启用 rollout loss。

```bash
conda run --no-capture-output -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config "$UR_CFG" \
  --data_path "$DATA" \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --epochs 100 --batch_size 1024 --lr 1e-3 --seed 20260727 \
  --rollout_loss_steps 5 --rollout_loss_weight 0.1 \
  --num_workers 4 --pin_memory --amp \
  --save_dir "$CKPT_ROOT"
```

如出现 `DataLoader worker ... exited unexpectedly`，先用 `--num_workers 0` 重跑以排除多进程数据加载问题；稳定后再尝试 `2` 或 `4`。不要一开始设置为很大的 worker 数（例如 32），它可能耗尽共享内存或进程资源。

训练脚本默认从 `${DATA%.npz}.manifest.json` 读取数据身份。每个训练 run 会创建带时间戳
的目录，例如：

```text
$CKPT_ROOT/gru_YYYYMMDD_HHMMSS/
├── best_model.pt
├── best_rollout_model.pt            # 只有 rollout 指标改善时存在
├── latest_model.pt
├── normalizer.pt
├── config.yaml
├── artifact_manifest.json
└── validation_by_source.jsonl
```

令该目录为 `RUN_DIR`。训练完成后先核对：

```bash
RUN_DIR=dynamics_modeling/outputs/checkpoints/ur5e/gru_YYYYMMDD_HHMMSS
test -f "$RUN_DIR/best_model.pt" && test -f "$RUN_DIR/normalizer.pt" && \
  test -f "$RUN_DIR/artifact_manifest.json"
```

不要把 ABB checkpoint 用作 UR5e 的 `--init_from_checkpoint`，也不要为了绕过检查修改
manifest。当前论文验证的设计要求每台机器人独立训练。

## 6. 阶段 4：动力学模型验证

先运行 1/5/10/20-step rollout 验证。该命令同时检查 checkpoint 与 normalizer 是否和
UR5e RobotSpec 及同一 dataset manifest 匹配：

```bash
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" \
  --checkpoint "$RUN_DIR/best_model.pt" \
  --normalizer "$RUN_DIR/normalizer.pt" \
  validate-model --num-rollouts 20 --rollout-len 200
```

输出在：

```text
$EXP/diagnostics/gru_validation/
├── validation_manifest.json
├── rollout_metrics.csv
└── figures and per-rollout diagnostics
```

验收时关注高幅 action 组的误差是否显著发散，而不仅是 one-step loss。若 10/20-step
rollout 已明显失真，先扩充/调整 UR5e 数据覆盖或重训，再调 CEM；不要用 MPC 参数补偿
一个失配的动力学模型。

## 7. 阶段 5：延迟标定与 MPC preflight

UR5e 必须独立标定 threaded planner 的 snapshot-to-publication 延迟。先使用少样本 smoke
命令确认链路，再运行正式标定。**这一阶段依赖阶段 0 在同一个 `$EXP` 下生成的 reference；先执行下列检查。**

```bash
test -s "$EXP/references/delay_calibration/reference.npz" || {
  echo "缺少延迟标定 reference：请先运行阶段 0 的 generate-references，并确认 --output-root 为 $EXP"
  exit 1
}
test -s "$EXP/references/manifest.json" || {
  echo "缺少 reference manifest：请先运行阶段 0 的 generate-references"
  exit 1
}
```

若上述检查失败，运行：

```bash
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" \
  generate-references --calibration-steps 2400
```

只在尚未产生标定或正式 manifest 时重新生成 reference。若已有 `$EXP/calibration/delay_smoke.json`、`$EXP/calibration/delay.json` 或 `$EXP/manifests/paper.json`，不要用 `--overwrite` 静默替换 reference；应建立新的 `$EXP` 实验目录，保证 artifact chain 可追溯。

检查通过后，运行：

```bash
# 快速检查：只采最多 3 个有效 planner sample
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" --checkpoint "$RUN_DIR/best_model.pt" \
  --normalizer "$RUN_DIR/normalizer.pt" \
  calibrate-delay --smoke --samples 3

# 正式标定：建议至少 500 个有效 planner sample
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" --checkpoint "$RUN_DIR/best_model.pt" \
  --normalizer "$RUN_DIR/normalizer.pt" \
  calibrate-delay --samples 500 --guard-ms 5.0 --provisional-delay 10

# 对 Projected Direct IK 和四个 MPC variant 做短闭环检查
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" --checkpoint "$RUN_DIR/best_model.pt" \
  --normalizer "$RUN_DIR/normalizer.pt" \
  preflight --max-steps 5
```

`delay.json` 使用
`ceil((P95(snapshot-to-publication) + guard) / control_dt)` 计算 anticipation delay。
不要把 ABB 的 delay 复制给 UR5e。`preflight/report.json` 必须为 `passed`，且五种方法
均没有 NaN、joint-limit、command velocity 或 command acceleration violation。

如需手动只跑一个 task-space MPC smoke test，可使用通用 runner：

```bash
conda run --no-capture-output -n pendulum-rl python scripts/run_cem_mpc.py \
  --robot_config "$UR_CFG" \
  --checkpoint "$RUN_DIR/best_model.pt" --normalizer "$RUN_DIR/normalizer.pt" \
  --model_type gru --history_len 16 --device cuda \
  --reference_mode task --reference_file "$EXP/references/circle/reference.npz" \
  --controller_mode mpc --multirate_mode virtual_asap \
  --delay_protocol full --anticipation_delay_steps 6 \
  --horizon 20 --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --planner_projection on \
  --planner_projection_backend compiled --planner_projection_strategy two_stage \
  --exact_task_space_cost on --save_dir outputs/mpc/ur5e_circle_smoke
```

这里的 `6` 仅是手动 smoke placeholder；正式实验必须用 `$EXP/calibration/delay.json`
中的值。无 CUDA 时可把 `--device cuda` 改为 `cpu`，但 threaded timing 不再代表正式
部署测量。

## 8. 阶段 6：冻结并运行正式 UR5e 矩阵

在干净 worktree 和固定 commit 上建立 manifest。正式运行不建议使用 `--allow-dirty`：

```bash
git status --short

conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" --checkpoint "$RUN_DIR/best_model.pt" \
  --normalizer "$RUN_DIR/normalizer.pt" \
  --dataset-manifest "${DATA%.npz}.manifest.json" \
  build-manifest
```

然后运行正式矩阵：四条轨迹 × 五个 CEM seeds × 四个 MPC variants，外加每条轨迹一个
确定性的 Projected Direct IK。这一共是 `4 + 4 × 5 × 4 = 84` 个 run：

```bash
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" \
  run --manifest "$EXP/manifests/paper.json" --resume
```

运行中断后重复同一命令并保留 `--resume`。单机先试一个 case 可加
`--case-limit 1`，但不要把该 limited index 当正式结果。

完成后汇总：

```bash
conda run --no-capture-output -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  --output-root "$EXP" summarize --bootstrap-samples 10000
```

主要输出：

```text
$EXP/manifests/paper.json
$EXP/runs/indexes/main.json
$EXP/runs/<method>/<trajectory>/seed_<n>/
$EXP/summaries/main.csv
$EXP/summaries/main_aggregate.csv
$EXP/summaries/main_paired_bootstrap.json
$EXP/summaries/worst_seed.json
```

每次检查 `run_summary.json`、projection activity、requested/executed residual RMS、
residual saturation、velocity/acceleration saturation、torque RMS、Direct-IK fallback、
planner update rate、deadline miss 和 late packet drop。CEM 结构、cost 权重、history
length 和 horizon 应与 ABB 保持一致；只有 robot-specific physical limits 可以在有这些
诊断证据时调整。若调整 `ur5e.yaml`，其 spec SHA 会改变，必须重新采集/训练或建立新的
完整 artifact chain。

## 9. 阶段 7：与 ABB 做描述性跨平台汇总

仅在 ABB 与 UR5e 都已完成各自的平台内正式统计后，生成合并表和 effect-size 图：

```bash
conda run -n pendulum-rl python scripts/paper_experiments/multi_robot_summary.py \
  --abb-summary outputs/<abb_formal_run>/summaries/main.csv \
  --ur5e-summary "$EXP/summaries/main.csv" \
  --output-dir outputs/multi_robot_summary \
  --bootstrap-samples 10000
```

输出 `platforms.csv`、`tracking_table.csv`、`effect_sizes.csv`、`effect_sizes.json` 和
`multi_robot_effect_sizes.png`。这里的 bootstrap 只在每个机器人内部按 trajectory-seed
配对；不要把 ABB 和 UR5e 合并为一个显著性检验。

## 10. 常见失败及处理顺序

| 症状 | 常见原因 | 处理 |
| --- | --- | --- |
| `RobotSpec ... mismatch` | 机器人 profile、XML、数据或 checkpoint 不同 | 检查 manifest SHA；使用同一条 artifact chain，不要强制加载。 |
| `control_dt` mismatch | XML timestep 或 frame skip 被改动 | 恢复 2 ms × 5 = 10 ms，或重新采集、重训和重标定。 |
| reference IK 失败/首尾不回 home | UR5e home、TCP 或任务尺度不适配 | 先跑 `validate-robot`；降低轨迹尺度或增大 lap duration，保留 RobotSpec home。 |
| preflight violation | physical caps、reference 导数或模型预测不匹配 | 查看 projection/residual/velocity/acceleration 诊断；先修 reference 或数据覆盖，再有证据地调 UR5e physical limits。 |
| 20-step rollout 发散 | 训练数据在 MPC 区域不足或 GRU 不足 | 扩展该 workspace/motion mode 的独立 UR5e 数据并重训。 |
| threaded delay 为空/异常高 | GPU、worker 或 timing 环境不稳定 | 检查 CUDA、planner events 和 deadline metrics；重新做 UR5e delay calibration。 |

## 11. 两道验收门

在耗费正式计算资源前，确认以下两道门都通过：

1. **工程门**：`validate-robot`、reference generation、collect smoke、模型 identity
   check、delay smoke 与 `preflight` 全部通过，无 NaN/约束违例。
2. **证据门**：UR5e 1/5/10/20-step rollout 可接受；正式 delay 已标定；84 个 run 的
   manifest、hash、reference、checkpoint、normalizer、dataset 和 commit 都可追溯。

通过后，论文中可作出的窄结论是：同一 activation-aligned asynchronous residual MPC
架构能在两个独立训练、独立标定的 6-DoF MuJoCo 机械臂上运行。它不等价于“同一个 GRU
跨机器人迁移成功”。
