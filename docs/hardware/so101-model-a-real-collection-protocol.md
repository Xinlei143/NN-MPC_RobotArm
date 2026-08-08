# SO101 实机 Model-A 数据采集协议（48 × 15 min）

本文是 SO101 实机 Model-A 工作空间数据集的唯一操作流程。它覆盖离线轨迹预检、温度检查、单次采集、严格追加、失败恢复与最终验收。所有真实动作均由操作者执行；任何异常声、线缆拉扯、碰撞风险或失去支撑时，立刻 `Ctrl+C` 或断电。

## 1. 固定采集计划

本轮数据集计划版本为 `so101_model_a_48x15min_v1`。采集开始后，硬件 YAML、PID、Acceleration、控制周期、校准、关节范围、精细模型/STL bundle、工作空间边距或本计划不能改变后继续追加到同一 canonical 数据集。桌面安全 profile 可按 session 变化，但每个 shard、canonical manifest 和每行 transition 都记录其 profile hash；验证器会按各自行对应的阈值验证，绝不把 profile 身份抹掉。

| 项目 | 固定值 |
|---|---:|
| 控制频率 | 30 Hz |
| 单次有效采集时长 | 15 min = 900 s |
| 单次控制 tick | 27,000 |
| 单次 transition（通常） | 26,995 |
| 总 session | 48 |
| train / validation / test | 38 / 5 / 5 |
| 总 transition（通常） | 1,295,760 |
| train transition（通常） | 1,025,810 |
| 有效运动总时长 | 12 h |

session index 与 split 固定对应：`0–37` 为 train，`38–42` 为 validation，`43–47` 为 test。不得把 transition 随机拆分到不同 split。

每个 session 的五个记录 episode 为：

| 顺序 | 时长 | 模式 | 说明 |
|---:|---:|---|---|
| 1 | 135 s | `hold` | 工作空间随机目标保持 |
| 2 | 270 s | `single_joint_sine` | `session-index % 5` 决定测试的关节 |
| 3 | 225 s | `multi_joint_sine` | 五关节同时正弦 |
| 4 | 180 s | `smooth_random` | 平滑随机 knot/reference |
| 5 | 90 s | `step`（偶数）/`delta_ref_random`（奇数） | 不同的快速变化参考 |

每段开始前均有一个约 8 s 的预定位轨迹；预定位、启动回 home、结束回 home 不写入训练数据。因此操作者应为每个 session 预留约 16–18 min，而非仅 15 min。

## 2. 当前桌面安全模型

E3 不使用 `so101_nominal.xml` 的 capsule 作安全判断，而使用 `dynamics_modeling/robots/so101_fine/` 的精细 STL MuJoCo 模型。

- 真实桌面在模型基座坐标 `z=0`；当前已批准的前提是基座水平，且 `q_ctrl_to_q_kin` 为 2026-08-06 同步窗口标定映射（wrist_flex `joint_offset=-0.2516342834 rad`，全部 `joint_sign=+1`）。
- `scene_table_guard_25mm.xml` 在 `z=25 mm` 放置虚拟桌面保护平面（2026-08-06 由 `scene_table_guard_20mm.xml` 就地改名并升高到 25 mm）。
- 离线 MuJoCo 预检和参考/预定位轨迹生成使用 `offline_minimum_clearance_m=0 mm`：`upper_arm`、`lower_arm`、wrist、gripper、活动夹爪不得触碰 25 mm 虚拟面，即模型桌面上方至少 25 mm。生成阶段比实机更保守 10 mm。
- 实机启动、实时实际状态、待发送指令和回收使用 `minimum_clearance_m=-10 mm`：最多可低至虚拟面下方 10 mm，即模型桌面上方至少 15 mm。它是跟踪/模型误差的运行时紧急门限，不是离线放行标准；这 10 mm 带宽吸收被发送指令相对参考的投影滞后（该滞后曾在旧 20 mm 面下把 `e3_23_retry00/retry01` 锁定在 −0.4 mm）。
- `shoulder_pan` 与第 1→2 关节之间的支架（MuJoCo body `shoulder`）明确不在此门禁中。
- TCP 不再有独立高度门限。

当前批准的 profile 是：

```text
configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml
```

该 profile 的 `q_ctrl_to_q_kin` 已于 2026-08-06 通过同步窗口标定（wrist_flex `joint_offset = -0.2516342834 rad`，
全部 `joint_sign = +1`，`source = operator_verified_sync_window_calibration_20260806`），见
`docs/hardware/so101-mapping-calibration.md`。不要使用旧的 `so101_table_safety.model_tcp110.local.yaml` 或
`mesh_guard15` profile，也不要把旧 capsule 的预检结果当作 E3 放行依据。

本协议下面统一使用新的 canonical 路径，避免与此前的失败 pilot shard 混淆：

```text
outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz
```

## 3. 每次 session 前的检查

### 3.1 物理检查

1. 确认基座没有移动、翘起或垫高；桌面相对基座仍对应模型的 `z=0`。
2. 确认桌面及工作空间没有新障碍物，线缆有余量，机械臂卸力后有支撑或软垫。
3. 断电方式必须可随手触及；操作者全程在场。
4. 确认当前硬件 YAML、精细 mesh profile 和计划未修改。若有任意修改，本轮 canonical 数据集必须重新开始。

### 3.2 温度与只读诊断

先运行一次只读执行器诊断并保存 JSON。它不发送 Goal Position、不使能扭矩、不修改寄存器。检查：

- 六个电机的 `Present_Temperature` 都低于 50 °C；建议开始下一 session 前最高温度不高于 45 °C。
- `Present_Voltage`、状态读取和串口通信正常。
- 运行时温度每约 0.5 s 读取一次；阈值使用最近四次**独立且已确认**温度读取的逐电机平均值，而非重复使用同一条 30 Hz 缓存。平均值达到 50 °C 时进入 warning/降权，达到 55 °C 时锁存故障；平均值达到 60 °C 后，第二次独立平均值仍确认时请求关扭矩。
- 新读数相对该电机已确认温度中位数跳变超过 8 °C（或首条读数 ≥50 °C）时，采集器立即对 `Present_Temperature` 额外复读两次。至少一次与候选值相差不超过 3 °C，才将其作为真实跳变纳入四样本均值；两次复读均不一致则丢弃候选异常值，以两次复读的中位数替代。复读失败时保持上一安全目标，直到下次成功诊断；连续 3 s 没有有效温度仍会停止。
- 原始温度、过滤后四样本均值和每电机状态（`accepted`、`confirmed_jump`、`outlier_rejected` 等）随 transition 保存，便于事后区分真实温升和寄存器/总线异常。

15 min 连续采集后必须再次做只读诊断并记录最高温度。不要无间隔连续运行 48 个 session；建议按设备温度安排批次，并在每两个 session 后至少观察一次温度趋势。只有温度回到可接受的启动范围后才进入下一次采集。

session 0 的采集前温度记录示例：

```bash
conda run -n lerobot python scripts/diagnose_so101_actuation.py \
  --read-only \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output outputs/hardware/so101_pre_mpc/20260804_e_stage/temperature_pre_e3_01_retry05.json
```

### 3.3 结束、异常与 Ctrl+C 的回收位

当前 `so101_follower.local.yaml` 固定了 E3 采集器的六电机回收 raw 姿态（ID 1 至 ID 6）：

```text
[2048, 1296, 2173, 2830, 988, 2048]
```

正常完成一个 session 后，采集器会先经过普通硬件 q 安全范围回到该姿态；保持目标并以 0.25 s 间隔最多等待 5 s，直到实际 raw 误差不超过 12 count，随后关闭全部扭矩。超时仍不满足该精度时，session 标记为 `failed_recovery`，不会追加到 canonical 数据集。常规 E3 参考轨迹不会使用这一个回收姿态。

发生报错或按 `Ctrl+C` 时，脚本也会尝试该回收，但**仅**在下面条件全部满足时才发送任何回收动作：当前状态有效、位置与 Goal Position 回读通信正常、诊断不超过 3 s、六个温度均低于 50 °C、以及从当前姿态到完整回收 raw 姿态的每个离散点都满足实机 `-10 mm` 相对虚拟 25 mm 面的门限。任一条件不满足时，脚本不再尝试移动，直接关闭扭矩，并把原因写入 session manifest 的 `shutdown_recovery` 字段。

因此，在机械臂已越过 raw/q 范围、通信失败、碰桌后状态异常或过温时，不要期待脚本强行把机械臂拉回该姿态。先断电/支撑机械臂并人工恢复，再做只读诊断。

### 3.4 离线精细网格预检

每个 session index 都要单独离线预检。预检对本次 seed 生成的五段轨迹、每段预定位、home 与结束回 home 的所有 30 Hz 离散点执行精细网格距离检查；不连接机械臂。

预检通过条件：

- 输出 `status: pass`；
- `overall_min_effective_clearance_m >= 0.0`，即离线参考/预定位轨迹必须清空 25 mm 虚拟面（距离真实模型桌面至少 25 mm）；
- profile 中记录的 `mesh_collision.bundle_sha256` 与当前模型资产一致。

预检只验证计划轨迹。实机程序仍会在启动时读取未知的当前姿态，并对启动路径、预定位、当前状态和待发送目标重新执行相同检查。

session 0 的预检示例：

```bash
conda run -n lerobot python dynamics_modeling/scripts/preflight_real_workspace_model_a_table.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --session-index 0 --seed 20260805 \
  --workspace-margin-deg 1 \
  --output outputs/hardware/so101_pre_mpc/20260804_e_stage/preflight_guard20_offline25_e3_01_retry05.json
```

## 4. 单次正式采集

对于每个 index，使用与 split 对应的唯一 `session-id`。session 0 已完成并追加为 `e3_00_retry09`；session 1 的 `e3_01*` 至 `e3_01_retry04` 是失败 evidence，下一次 session 1 应使用 `e3_01_retry05`，但保持同一个 `session-index` 与 split。

正式采集程序依次执行：

1. 连接硬件，读取状态，验证状态有效、关节处于绝对硬件范围内。
2. 从当前姿态平滑进入 home；整段路径也接受精细网格检查。
3. 对每个 episode 生成工作空间内、速度/加速度受限的参考。不能通过网格检查的候选整段轨迹会重新采样。
4. 预定位至该段起点（不录入数据），随后以 30 Hz direct control 记录 15 min 有效数据。
5. 每个 tick 检查测量状态、待发送命令与桌面门禁；记录 q、估计 dq、实际 dt、动作、回读、温度、诊断和 table-safety 字段。
6. 五段完成后先平滑回 home，再执行上节的受保护 raw 回收；回收验证通过才写出不可覆盖的 `completed` session shard，随后关闭扭矩。

运行中遇到碰撞风险、线缆张紧、异响、异常抖动、温升或异常状态时，立即 `Ctrl+C`；不要等待脚本自动恢复。脚本只会按上节条件决定是否执行受保护回收，失败数据绝不会追加到 canonical。若在第一条 transition 前失败，没有 `.npz` shard 是正常的：会留下同名的 `.failure.json`，其中也包含 `shutdown_recovery` 的决定和理由；重试必须使用新的 `session-id`。

session 0 的正式采集示例：

```bash
conda run -n lerobot python dynamics_modeling/scripts/collect_real_workspace_model_a.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml \
  --session-index 47 --session-id e3_47_retry00 --split test --seed 20260806 \
  --workspace-margin-deg 1 \
  --output outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz \
  --append --enable-motion --operator-supported-shutdown \
  --visualize-mujoco
```

`--visualize-mujoco` 会打开精细 MuJoCo 的被动观察窗口（含 35 mm 虚拟面）。每个有效的实机测量状态都会写入这个**独立**的 MuJoCo model/data 并刷新窗口；它不修改发送指令、轨迹生成或任何安全阈值。窗口中模型与实物的方向、零位或桌面相对高度明显不一致时，立即 `Ctrl+C`，先重新校验 `q_ctrl_to_q_kin`
映射再采集（当前映射为 2026-08-05 sync-window 标定，见 `docs/hardware/so101-mapping-calibration.md`）。
关闭窗口本身不会中断采集。

## 5. 严格 append 与每次采集后的验收

canonical 输出路径在整个 48-session 计划内保持不变。每次成功采集先生成：

```text
sessions/model_a_workspace_<session-id>.npz
sessions/model_a_workspace_<session-id>.manifest.json
```

只有以下条件均满足时，shard 才会原子追加到 canonical `.npz`：

- session 的结构和字段有效；
- session 状态为 `completed`；
- robot identity、plant identity、工作空间和 48×15 min collection plan 与 canonical 完全一致；
  桌面 table_safety profile **不要求字节一致**：append 把各阶段实际使用的 profile 累计进
  `table_safety_profiles`（例如 0-5 在 identity 映射下采集、6+ 在标定映射下采集）；
- session ID 未存在；
- episode ID 不与既有数据重叠；
- 所有数组名称、形状、dtype 一致。

每次 append 后都运行数据验证器。对所有行必须保证样本有限、timestamp 单调、动作/状态对齐、诊断新鲜、session 字段完整、manifest hash 正确；对 `valid_target=true` 的训练 transition，还必须保证 dt 在窗口内且不跨 history/estimator generation。采集器会保留 `valid_target=false` 的短暂调度抖动/重置行作为审计证据，但训练和有效 transition 统计会排除它们。若验证失败，停止继续采集；不要手动删除、覆盖或合并失败证据。

GRU/Transformer 训练时，数据加载器还会要求整个 history window 的每一条 transition 都具有 `valid_target=true`；不能只检查窗口最后一条。以 `history_len=16` 为例，任何包含一条审计无效行的 16 步窗口都会自动排除。canonical NPZ 不做删行或重写，以保留完整审计证据。

每次采集后还应保存一次只读温度诊断 JSON。这样可以把 dataset 的在线温度字段与独立寄存器读数相互核对。

session 0 成功结束后的两个必做动作是：

```bash
conda run -n lerobot python dynamics_modeling/scripts/validate_real_dataset.py \
  outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz
```

```bash
conda run -n lerobot python scripts/diagnose_so101_actuation.py \
  --read-only \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output outputs/hardware/so101_pre_mpc/20260804_e_stage/temperature_post_e3_01_retry05.json
```

若 session 1 成功，session 2 仅替换 `--session-index 2`、`--session-id e3_02`；`--split train`、seed、profile 与 canonical output 路径保持不变。session 38 开始将 split 改为 `validation`，session 43 开始改为 `test`。

## 6. 建议的批次安排

48 个 session 的有效运动时间为 12 h，不应连续执行。建议：

1. 先完成 session 0 作为 15 min pilot，验证预检、启动、完整 append、数据验证和温升记录。
2. pilot 通过后，每批执行 2–4 个 session；每个 session 后记录温度，每批结束后充分冷却。
3. 每完成 8 个 train session，检查累计 dataset 的大小、每个关节 q/dq/action 分布、有效率、温度趋势和安全 flag；不要等到 48 个全完成后才发现覆盖缺失。
4. train 全部完成后，再采 validation 与 test；绝不把 validation/test session 用于调参或重新训练。

## 7. 完成标准与后续步骤

E3 完成需要同时满足：48 个独立 completed session、38/5/5 session split、所有 shard 已严格追加、每次后验证通过、温度记录完整、无不明安全事件、且本轮配置 identity 一致。

完成后冻结 canonical 数据集和 manifest；随后进入真实 dynamics 训练。训练仅使用 train group `0–37`，validation 使用 `38–42`，最终 test 使用 `43–47`。不得将 MPC 相关或其他额外采集数据混入这份 Model-A 数据集。
