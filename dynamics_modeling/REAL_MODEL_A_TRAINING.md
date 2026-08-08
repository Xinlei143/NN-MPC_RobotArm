# SO101 Model-A 真实动力学训练指南

本文档描述如何用已冻结的 SO101 Model-A 真实数据集训练 GRU 动力学模型。
数据采集、校验与冻结见 [`docs/hardware/so101-model-a-real-collection-protocol.md`](../docs/hardware/so101-model-a-real-collection-protocol.md)。

## 1. 前置条件

- canonical 数据集与 manifest 已冻结：`outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz` + `.manifest.json`
- 48 个 session 已严格 append，`validate_real_dataset.py` 全布尔项通过
- robot config：`configs/robots/so101.yaml`（`robot_id=so101_real_v1`，`domain=real`，5 关节，30 Hz）

**Session split 约定（固定，不可按 transition 随机切分）：**

| 组 | session index | 用途 |
|---|---|---|
| train | 0–37 | 训练窗口 |
| validation | 38–42 | 每 epoch 验证 |
| test | 43–47 | 全程冻结，禁止进入 train/val、禁止用于调参 |

## 2. 训练命令（Model-A 标准命令，含全部参数接口）

下面命令逐参数拼写，覆盖 `train_dynamics.py` 的**全部**参数接口（已使用的显式给出值；未使用的随后列出其默认/关闭状态）。`--real_defaults` 保留作为安全网，但每个参数值都已显式写出，不依赖默认值。

```bash
conda run --no-capture-output -n lerobot python dynamics_modeling/scripts/train_dynamics.py \
  --data_path outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz \
  --dataset_manifest outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.manifest.json \
  --robot_config configs/robots/so101.yaml \
  --model_type gru \
  --history_len 16 \
  --batch_size 8192 \
  --epochs 200 \
  --lr 1e-5 \
  --pin_memory \
  --save_dir dynamics_modeling/outputs/checkpoints_real \
  --seed 10 \
  --num_workers 2 \
  --amp \
  --loss_type huber \
  --huber_delta 1.0 \
  --train_sample_stride 2 \
  --target_mode delta_state \
  --control_dt 0.033333333333333333 \
  --test_group_ids 43,44,45,46,47 \
  --rollout_loss_steps 6 \
  --rollout_loss_weight 0.09 \
  --validation_group_ids 38,39,40,41,42 \
  --init_from_checkpoint dynamics_modeling/outputs/checkpoints_real/gru_20260807_154049/best_rollout_model.pt \

  |

```

**其余参数未使用（保持默认/关闭）：**

```text
--pin_memory                 off（默认）
--freeze_normalizer          off（默认，需配合 --normalizer_path 才可开启）
--no_require_q_ref_dataset   off（默认，q_ref 校验强制开启）
--resume_checkpoint          None（未指定）
--init_from_checkpoint       None（未指定）
--normalizer_path            None（未指定）
--source_weights             None（未指定，单采集源）
--steps_per_epoch            None（未指定）
--q_extra_weights            None（未指定，使用关节平均权重）
--dq_extra_weights           None（未指定，使用关节平均权重）
```

对应展开后的等效配置：

```text
model_type        gru
history_len       16
target_mode       delta_state
control_dt        1/30 s
loss_type         huber
q_weight          1.0
dq_weight         0.35
rollout_steps     5
rollout_weight    1.0
val_fraction      0.2   （被 --validation_group_ids 显式分割覆盖）
```

## 3. 参数接口（train_dynamics.py 全部）

### 3.1 数据与身份

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data_path` | **必填** | canonical NPZ 路径。训练要求 `states/actions/next_states/q_ref/delta_q_ref` 存在，且 `actions == q_ref`（position-control 语义）。 |
| `--robot_config` | `configs/robots/abb_irb2400.yaml` | RobotSpec。`robot_identity` 必须与 dataset manifest 完全一致，`control_dt` 必须与 `expected_control_dt` 一致。 |
| `--dataset_manifest` | `data_path` 同名 `.manifest.json` | 严格 manifest；校验 `robot_identity`、`plant_identity`、`dataset.sha256`、`collection_mode`。 |
| `--no_require_q_ref_dataset` | off | 跳过 q_ref 强制校验（仅旧数据集使用，不推荐）。 |

### 3.2 模型与窗口

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model_type` | `transformer` | `mlp` / `gru` / `transformer`。GRU 单层 hidden=256；Transformer 3 层 d_model=256。 |
| `--history_len` | `1` | 递归窗口长度。GRU/Transformer 要求窗口内每条 transition 均 `valid_target=true`。 |
| `--target_mode` | `delta_dq` | `delta_state`（10 维）/ `delta_dq`（5 维）。Model-A 用 `delta_state`。 |
| `--control_dt` | `0.01` | 名义 dt，用于 delta_state 积分与 rollout 推进。Model-A 用 `1/30`。 |

### 3.3 优化与损失

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch_size` | `1024` | DataLoader batch。 |
| `--epochs` | `100` | 总 epoch 数。 |
| `--lr` | `1e-3` | AdamW 学习率。 |
| `--loss_type` | `mse` | `mse` / `huber`。 |
| `--huber_delta` | `1.0` | Huber 阈值（loss_type=huber 时生效）。 |
| `--q_weight` | `1.0` | 位置分量损失权重。 |
| `--dq_weight` | `1.0` | 速度分量损失权重。Model-A 用 `0.35`。 |
| `--q_extra_weights` | None | 每关节位置权重，逗号分隔 5 个值。 |
| `--dq_extra_weights` | None | 每关节速度权重，逗号分隔 5 个值。 |
| `--rollout_loss_steps` | `1` | rollout 步数（>1 且 weight>0 时启用 rollout 损失）。 |
| `--rollout_loss_weight` | `0.0` | rollout 损失权重。 |
| `--rollout_loss_discount` | `1.0` | rollout 时间折扣。 |

### 3.4 数据分割

| 参数 | 默认 | 说明 |
|---|---|---|
| `--test_group_ids` | None | 逗号分隔 split_group_id，从 train 和 val 中排除（Model-A：`43,44,45,46,47`）。real domain 必填。 |
| `--validation_group_ids` | None | 逗号分隔 split_group_id，仅用于验证（Model-A：`38,39,40,41,42`）。real domain 必填，且与 test 不相交。 |
| `--val_fraction` | `0.1` | 无显式 group 时的随机验证比例（real 数据集被显式 group 覆盖）。 |
| `--train_sample_stride` | `1` | train 窗口抽样步长。 |
| `--val_sample_stride` | `1` | val 窗口抽样步长。 |

### 3.5 采样与续训

| 参数 | 默认 | 说明 |
|---|---|---|
| `--seed` | `10` | 随机种子。 |
| `--num_workers` | `0` | DataLoader 进程数。 |
| `--pin_memory` | off | CUDA pinned memory。 |
| `--amp` | off | 混合精度训练。 |
| `--source_weights` | None | `source_id:weight,...` 重采样（多采集源时用）。 |
| `--steps_per_epoch` | None | 每 epoch 采样步数（配 source_weights）。 |
| `--resume_checkpoint` | None | 从 checkpoint 续训（模型+优化器+scaler+epoch 计数）。 |
| `--init_from_checkpoint` | None | 仅用 checkpoint 初始化权重，重新开始训练（与 resume 互斥）。 |
| `--normalizer_path` | None | 冻结 normalizer 时使用。 |
| `--freeze_normalizer` | off | 冻结 normalizer（需 `--normalizer_path`，校验 robot/dataset/plant identity）。 |
| `--real_defaults` | off | 一键展开 Model-A 冻结语义（见第 2 节）。**注意：在 parse 之后覆盖模型/损失参数，history_len 固定 16。** |
| `--save_dir` | `outputs/checkpoints` | 输出目录，子目录 `{model_type}_{YYYYMMDD_HHMMSS}/`。 |

## 4. 运行校验（训练前自动执行）

- manifest 存在且 `dataset.sha256` 匹配文件
- `robot_identity` 与 `robot_config` 一致；`collection_mode == model_a_excitation`
- real domain 必须有显式 `--test_group_ids` 和 `--validation_group_ids`，不相交、都存在、至少留一个 train 组
- `control_dt` 与 RobotSpec 一致；`--freeze_normalizer` 必须配 `--normalizer_path`
- q_ref 校验：`actions == q_ref`，否则报错

## 5. 输出产物（save_dir/{run}/）

| 文件 | 内容 |
|---|---|
| `config.yaml` | 完整训练配置（含 identity、dataset sha256、split 组）。 |
| `normalizer.pt` | 由 train 窗口拟合（val/test 统计不泄漏）。 |
| `best_model.pt` | val_loss 最优 checkpoint。 |
| `best_rollout_model.pt` | val rollout loss 最优 checkpoint（启用 rollout 损失时）。 |
| `latest_model.pt` | 最后一个 epoch。 |
| `validation_by_source.jsonl` | 按 source 分组的每 epoch 验证损失。 |
| `artifact_manifest.json` | 产物 sha256 + 数据集/plant identity。 |

## 6. 评估（模型门禁）

门禁命令（MPC 工作范围：1/3/5/6/10 步）：

```bash
conda run -n lerobot python dynamics_modeling/scripts/evaluate_real_model.py \
  --dataset outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz \
  --checkpoint dynamics_modeling/outputs/checkpoints_real/<RUN>/best_rollout_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_real/<RUN>/normalizer.pt \
  --test-group-ids 43,44,45,46,47 \
  --horizons 1,3,5,6,10 \
  --output dynamics_modeling/outputs/checkpoints_real/<RUN>/real_model_gate_mpc_range.json
```

评估参数：

| 参数 | 说明 |
|---|---|
| `--dataset` | canonical NPZ。 |
| `--checkpoint` | 模型 checkpoint。 |
| `--normalizer` | 对应 normalizer。 |
| `--test-group-ids` | test session 组（Model-A：`43,44,45,46,47`）。 |
| `--horizons` | 逗号分隔的 rollout horizons（最大值为 rollout 长度）；默认 `1,3,5,10,20,25,30`。 |
| `--output` | 门禁 JSON 输出路径。 |

报告内容：每个 horizon 的 q **和 dq** RMSE/P95/max、逐关节 q/dq RMSE，对 learned、persistence、constant-velocity、first-order 四种方法。

通过标准：1/3/5/10 步 q RMSE 全部优于 persistence、constant-velocity 和一阶舵机 baseline
（`model_gate_passed: true`）；逐关节 RMSE、q P95、每 motion mode、commanded-motion 子集（避免 hold 稀释）、
10 步排序能力、test 完全隔离、checkpoint/normalizer/plant identity 完全匹配（见下方审查命令）。
未通过则禁止 shadow/active 论文实验。

### 6.1 F 阶段审查（论文清单 §2 学习动力学验证的实机对应）

`audit_real_model_f.py` 输出 F 阶段全部审查项：逐关节 NMSE/R²、amplitude ratio、divergence rate（判据：
max-joint \|q error\| 超过阈值）、per-motion-mode 分模式 RMSE、commanded-motion 子集、worst-window 分析、
test-session 隔离与 artifact identity 审计、逐步 error-growth 曲线（时间轴秒，`T_H = H·Δt`）：

```bash
conda run -n lerobot python dynamics_modeling/scripts/audit_real_model_f.py \
  --dataset outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz \
  --checkpoint dynamics_modeling/outputs/checkpoints_real/<RUN>/best_rollout_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_real/<RUN>/normalizer.pt \
  --max-horizon 30 \
  --output-dir dynamics_modeling/outputs/checkpoints_real/<RUN>
```

输出 `audit_f_<时间戳>.json` + `error_growth_curve_<时间戳>.csv`（不进训练产物目录，不覆盖旧证据）。

### 6.2 实测结论（gru_20260807_130024，2026-08-07）

- MPC 范围（1/3/5/6/10 步）门禁通过；20/25/30 步（0.67/0.83/1.0 s）输给 first-order —— 与 ABB 主实验的
  预测窗口不对等：ABB H=20@100 Hz = 0.2 s ≈ SO101 H=6@30 Hz。跨平台比较必须用 `T_H = H·Δt` 统一横轴。
- dq 通道：H1 dq RMSE ≈ 0.014 rad/s，受 Δdq 目标噪声地板（0.019–0.023 rad/s）限制，只影响 dq 通道，不影响 q。
- 五种配置（delta_dq / delta_state × rollout × lr）在 H6 统计不可区分（0.204–0.210°），全部优于 first_order（0.263°）。
- 实机首轮 active 权限（2026-08-07 用户决定）：**`--horizon 6`**、`--feedback_kdq 0`、`--feedback_max 0.00872665`（0.5°），
  residual 由实机执行层硬限制 ≤2°；**不得继承 sim 的 `--horizon 20` / `--feedback_kdq 0.015` 默认值**。
  H=6 对应 0.2 s，与 ABB 主实验 H=20@100 Hz 的预测窗口一致；H6 q_rmse 0.204°（优于 first_order 22%）。
