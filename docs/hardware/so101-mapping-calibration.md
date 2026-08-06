# SO101 实机 ↔ fine MuJoCo 映射校准（同步窗口）

> 本文档描述如何校准 `q_ctrl_to_q_kin`（`joint_sign` / `joint_offset`），使轨迹安全审查（fine 网格模型）的
> 姿态与实机一致。**结论背景见 `docs/hardware/so101-pre-mpc-experiment-checklist.md` 与
> `docs/guides/so101-real-hardware.md`。**

> **当前状态（2026-08-05）**：校准已完成并应用到
> `configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml`
> （`joint_offset[3] = -0.2516342834 rad`，`joint_sign` 全 `+1`，
> `source = operator_verified_sync_window_calibration_20260805`）。以下步骤是执行记录与复现方法。

## 为什么要做

已确认（校准前）：当时所有 table-safety profile 都是 identity 映射（`joint_offset: [0,0,0,0,0]`），而 wrist_flex 的软件
home 刻意设在 raw 2048 → `q_ctrl[3] = +0.2516 rad = +14.42°`（距标定中点高 164 编码）。同步窗口和运行时
clearance 门（`MeshTableClearanceChecker`，fine 网格模型）共用同一映射，因此：

- 实机 home 手腕水平；
- fine 模型 `q_kin[3]=0` 时手腕在水平下方约 **2.82°**（CAD 腕部仰角）；
- identity 把 `+14.42°` 原样写入模型 → 模型显示手腕向下约 **17°**。

即**当前安全审查是在"手腕下折约 17°"的错误姿态上做碰撞判断**。校准后审查姿态才与实机一致。
其余四关节 home 在 0，sign（正方向）未在实物上验证过，一并测试。

**不改动实机 home**（保持 raw 2048 水平位）；修正通过 `joint_offset` 实现。

## 安全要求（每次必读）

- 操作者必须在场、扶机/随时可卸力，执行 `--enable-motion --operator-supported-shutdown`。
- 每个关节先从 ±0.5° 开始；方向无误后才可升到 ±1° 复核。
- 镜像窗口（`--visualize-mujoco`）是**纯观测**的：关闭它不影响实机运行，它也不发送任何命令。
- 测试产生的 NPZ 保留在 `outputs/hardware/calibration/` 作为证据。

## 环境

```bash
conda run -n lerobot python scripts/run_real_direct_control.py --help   # 确认含 --visualize-mujoco
```

## 第一步：逐关节方向测试（每个关节 positive + negative）

对 5 个关节分别执行（`<joint>` ∈ `shoulder_pan` | `shoulder_lift` | `elbow_flex` | `wrist_flex` | `wrist_roll`），
先用**已批准**的 identity profile（本步只确认方向，不预览偏移）：

```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --joint <joint> --direction positive --amplitude-deg 0.5 --seconds 10 \
  --visualize-mujoco \
  --output outputs/hardware/calibration/20260805/<joint>_positive.npz \
  --enable-motion --operator-supported-shutdown
```

```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --joint shoulder_lift --direction positive --amplitude-deg 0.5 --seconds 10 \
  --visualize-mujoco \
  --output outputs/hardware/calibration/20260805/shoulder_lift_positive.npz \
  --enable-motion --operator-supported-shutdown
```

把 `--direction positive` 换成 `negative` 再跑一次。

**通过判据**：实机该关节做小幅动作时，镜像窗口里 fine 模型同一关节向**相同方向**转动且幅度相近。

- 方向一致 → 该关节 `joint_sign = +1`（identity 成立）。
- 方向相反 → 该关节 `joint_sign = -1`。
- 幅度明显不成比例（远大于/远小于预期）→ 记下，可能是该关节零位或模型关节定义问题，先停下。

## 第二步：wrist_flex 的 offset 校准（目视）

确认方向后，用 `--mapping-preview-offset-rad` 预览候选 offset（只影响渲染，不改 profile、不发命令）。
identity 下模型手腕显示向下约 17°。先试第一近似（把实机 home 映射到模型零位）：

```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --joint wrist_flex --direction positive --amplitude-deg 0.5 --seconds 10 \
  --visualize-mujoco --mapping-preview-offset-rad 0 0 0 -0.2516342834 0 \
  --output outputs/hardware/calibration/20260805/wrist_flex_offset_preview.npz \
  --enable-motion --operator-supported-shutdown
```

- 若模型手腕已与实机水平位一致 → 记录 `joint_offset[3] = -0.2516342834`。
- 若模型仍显示约 2.8° 下折（fine 模型自身腕部仰角），把最后一个数调到 `-0.301` 再预览，
  直到目视完全一致，记录最终值。**以目视一致为准，不必拘泥于 -0.2516 或 -0.301。**

## 第三步：把结论写入已批准 profile

本校准直接修改已批准的 `mesh_guard20_runtime10.local.yaml`（`configs/hardware/*.local.yaml` 已被
gitignore，不会误提交）。操作者确认方向与 offset 后，编辑该文件：

- `q_ctrl_to_q_kin.joint_sign`: 填入第一步结论（默认全部 `1`；有反向关节改 `-1`）。
- `q_ctrl_to_q_kin.joint_offset`: 填入第二步结论；其余关节保持 `0`（方向测试未发现偏置）。
- `source: operator_verified_sync_window_calibration_20260805`
- `status: approved`（保持不变；这是操作者对校准映射的签署）。
- 保留其余字段（`model_xml_sha256`、`mesh_collision.bundle_sha256`、`table_probe` 等）原样。

## 第四步：重跑离线预检

预检脚本要求 `status: approved`（第三步已保持）。对要采集的每个 `--session-index` 单独预检：

```bash
conda run -n lerobot python dynamics_modeling/scripts/preflight_real_workspace_model_a_table.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --session-index 6 --seed 20260805 --workspace-margin-deg 1 \
  --output outputs/hardware/preflight/20260805/session6_calib.json
```

**通过判据**（与 `docs/hardware/so101-model-a-real-collection-protocol.md` §3.4 对齐）：
输出 `status: pass` 且 `overall_min_effective_clearance_m >= 0.005`（脚本实际执行的离线门槛是 profile 的
`offline_minimum_clearance_m=0.0`，即离 35 mm 保护面 ≥0 / 离桌面 ≥35 mm；0.005 是协议文档的更严判据）。
采集器/MPC 继续使用同一个 `mesh_guard20_runtime10.local.yaml`（修正映射已在其中）。

## 验证清单（端到端）

- [ ] 5 关节 × 正负方向测试：镜像转动方向 == 实机方向。
- [ ] home 下镜像显示的手腕姿态与实机目视一致。
- [ ] 新 profile `status: approved` 后预检 `pass` 且 `overall_min_effective_clearance_m >= 0.005`。
- [ ] （可选）随机目标点用镜像抽查，模型姿态与实物吻合。

## 相关代码

- `robot_runtime/mujoco_mirror.py`：可复用被动镜像（渲染用 `profile.mapping`，可叠加预览 offset）。
- `scripts/run_real_direct_control.py`：B4 逐关节直接控制入口 + `--visualize-mujoco` /
  `--mapping-preview-offset-rad`。
- `robot_runtime/kinematics.py`：`JointCoordinateMapping`。
- `robot_runtime/table_safety.py`：`MeshTableClearanceChecker`（安全审查用 fine 网格模型）。
