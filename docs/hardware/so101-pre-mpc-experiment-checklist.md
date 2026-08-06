# SO-ARM101 正式接入 MPC 前的测试与验收清单

本文档规定本项目在 SO-ARM101 实机上开始正式 MPC 论文实验之前必须完成的测试、证据产物、通过标准和晋级条件。所有步骤以当前仓库实现为准，不把 nominal MJCF 当作真实动力学，也不把 `sync_write` 当作逐电机确认。

最后更新：2026-08-05。

## 1. 适用范围与总原则

控制链路采用：

```text
tick k 开始读取 x_k
    -> 本地安全投影
    -> 向六个电机发送 transmitted_u_k
tick k+1 读取 x_{k+1}
    -> 提交 (x_k, transmitted_u_k, x_{k+1})
```

其中前五个电机进入状态与动作，ID6 夹爪只锁存 raw 目标。`transmitted_u_k` 是交给 Feetech `sync_write` 的软件目标，不代表每个电机已经确认接收。

以下规则贯穿全部测试：

- 只有 control/I/O thread 可以访问 Feetech bus；
- planner/GPU thread 不得访问串口；
- 机械臂必须有物理支撑或承接软垫，退出程序后卸力不能自由坠落；
- 必须准备独立于软件的断电手段；
- 每一级测试必须人工晋级，不允许脚本自动从 direct 晋级到 active；
- 任何 calibration、PID、Acceleration、Maximum_Acceleration、控制频率、安全范围、状态估计器或 LeRobot commit 变化，都必须重新执行相应测试并生成新的 plant identity；
- `q_ctrl` 用于实机采集、learned dynamics 和 joint-space MPC；`q_kin` 只用于 MJCF FK、TCP、IK 和任务空间代价；
- 首轮论文实验只做 5-DOF joint-space 控制。未完成独立运动学校准前，不允许以 full-pose IK/FK 结果作为论文实机结论。

## 2. 当前已知配置与未决项

本机配置文件为：

```text
configs/hardware/so101_follower.local.yaml
```

当前已确认：

```text
port                 /dev/ttyACM0
robot_id             my_follower
power variant        12 V
LeRobot commit       2aba372b4e217cc47db28e0f836859b20d1456c9
Acceleration         32
Maximum_Acceleration 32
gripper_hold_raw     2048
wrist_roll raw       300..3300
control rate         30 Hz
```

当前 calibration SHA-256：

```text
e5420b6914b3c43a405556f68b6964f87bc4839991d590ee74789a34b61b0d04
```

当前 calibration envelope：

```yaml
raw_low:  [984, 868, 867, 869, 300, 1742]
raw_high: [3246, 3273, 3071, 2899, 3300, 3155]
```

注意：除 wrist_roll 外，上述数值主要是 LeRobot calibration envelope，不等于已人工验证的最终物理安全范围。

当前阻塞正式运动测试的事项：

1. `home_q_ctrl=[0,0,0,0,0]` 表示 LeRobot 各关节标定中点，不表示五个 raw 都等于 2048；手动启动姿态不精确等于 home 是正常现象，本身不是故障或阻塞；
2. 最近一次手动启动姿态为：

   ```text
   q_ctrl_rad = [-0.09359568, -0.02685122, 0.12735149, 0.25623736, -0.00076718]
   ```

   它不在当前 `home=0 +/-3 deg` 的实验包络内。这本应由独立的初始化状态机在较宽、已验证的硬件安全范围内保持当前位置，再沿速度/加速度受限轨迹移动到 home；不应要求操作者手工精确对准 home；
3. `hardware_joint_low/high` 已作为独立配置字段和启动包络实现，但当前值仍为 `null`。必须在 B2 人工验证后填写；所有 motion 入口在其为 `null` 时拒绝运行。`joint_low/high` 保留为home附近的实验包络；
4. 电压阈值仍为 `null`，因此 active MPC 按设计不能启动；
5. 动态负载电压、持续温升和逐关节最终软件限位尚未冻结。

## 3. 证据目录与运行记录

所有实机验收产物统一放在：

```text
outputs/hardware/so101_pre_mpc/<YYYYMMDD_HHMMSS>/
```

每次运行至少保存：

- 使用的 hardware YAML 副本或 SHA-256；
- calibration SHA-256；
- plant identity；
- 执行命令；
- JSON/NPZ/manifest；
- 操作者、日期、负载、供电、环境温度和异常说明；
- 是否发生人工断电、支撑、碰撞、通信重试或异常声音。

论文正式实验前不得覆盖已有产物。失败运行也应保留并标明失败原因。

## 4. 阶段 A：离线软件与配置门禁

### A1. 硬件相关自动测试

运行：

```bash
python3 -m pytest -q tests/hardware
```

覆盖范围包括：

- 安全连接顺序与单读单写后端；
- MuJoCo backend 协议兼容性；
- 状态估计器因果性和 reset；
- `(x_k,u_k,x_{k+1})` transition 对齐；
- ASAP history generation，拒绝 reset 前的迟到 packet；
- OOD envelope；
- real dataset loading。

通过标准：全部通过。当前 `lerobot` conda 环境未必安装 pytest，可在项目开发环境运行；这不授权为了测试而替换已固定的 LeRobot 包。

### A2. 本地配置与 calibration identity

```bash
conda run -n lerobot python scripts/validate_so101_config.py \
  --hardware-config configs/hardware/so101_follower.local.yaml
```

通过标准：输出 `status: valid`，且 calibration hash、PID、Acceleration、Maximum_Acceleration、30 Hz、估计器版本和 LeRobot commit 与计划一致。

任何一次 LeRobot 重新标定都会修改 calibration 文件。修改后必须重新检查 raw 范围、更新哈希，并重新执行本文档从 A2 开始的相关测试。

### A3. rad/degree/raw 量化一致性

```bash
conda run -n lerobot python scripts/validate_so101_quantization.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --samples 1000
```

通过标准：

```text
max_raw_count_error <= 1 count
max_roundtrip_degree_error <= 360/4095 degree
```

### A4. 依赖和设备检查

记录以下输出：

```bash
conda run -n lerobot python --version
conda run -n lerobot python -c "import lerobot; print(lerobot.__file__)"
ls -l /dev/ttyACM0
nvidia-smi
```

通过标准：串口存在、用户具有访问权限、本地 NVIDIA GPU 可用于 planner、LeRobot 来源与固定 commit 一致。不得通过互联网或远程云主机闭环控制真机。

### A5. 启动拒绝与配置负向测试

在 fake bus/断电硬件条件下验证以下情况均拒绝启用力矩：

- calibration 文件缺失；
- calibration SHA-256 不匹配；
- EEPROM calibration 与固定文件不匹配；
- 六个电机中任意一个缺失；
- PID、Acceleration、Maximum_Acceleration 或 Return Delay 读回不一致；
- raw 软件范围超出 calibration envelope；
- gripper hold raw 越界；
- 只填写部分电压门限；
- 非固定 LeRobot identity 或错误 robot ID。

还必须验证连接顺序为：bus connect、disable torque、calibration检查、读取当前位置、预装当前位置目标、配置及读回、最后enable torque。失败路径必须保持torque disabled并关闭串口。

## 5. 阶段 B：物理安全和坐标确认

### B1. 上电前检查

逐项人工签字：

- follower 与 STS3215 确认为 12 V 版本；
- 电源额定电压、额定电流、极性和接头正确；
- 所有结构件、舵盘、线缆和夹爪紧固；
- 完整工作区无人员和硬物；
- 有支撑架或软垫承接卸力机械臂；
- 独立断电开关可在一秒内触达；
- USB 与供电线不会被腕部运动拉扯；
- 初始负载、夹爪开度和环境温度已记录。

### B2. raw 安全范围

逐关节、低速、人工监督验证：

```text
calibration envelope
  intersect MJCF nominal range
  intersect manually verified collision-free range
  minus safety margin
```

不得通过一次 `+2 deg` 运动自动生成 offset 或完整限位。wrist_roll 当前使用 `300..3300`，首次实验不得跨越 raw 的 `0/4095` 回绕点。

在 `hardware_joint_low/high` 尚未冻结时，只可使用 B2 专用的人工监督 jog 来逐步建立证据。首先尝试0.5度；若已确认处在行程中部而编码器仍无响应，可用一次1.0度诊断步进。它仅检查现有raw范围，绝不构成 direct/MPC 放行：

```bash
conda run -n lerobot python scripts/jog_so101_raw_safety.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint shoulder_pan --direction positive --step-deg 0.5 --hold-seconds 5 \
  --output outputs/hardware/so101_pre_mpc/<RUN>/b2_shoulder_pan_positive_001.json \
  --enable-manual-jog --operator-supported-shutdown
```

每次只允许一个0.5度步长；仅在该步无响应且起始姿态居中时才允许单次1.0度诊断。操作者观察到任何干涉、线缆拉扯、异常声或方向错误时，立即断电、记录raw值并停止扩大该方向。

通过标准：每个关节都有人工验证的 raw low/high、验证日期和操作者记录；任何方向错误、线缆扭转或机械干涉都必须先修复。

### B3. home 与 direct 包络

必须明确区分：

```text
raw 2048                    电机编码值
calibration midpoint        q_ctrl = 0 的编码值，逐关节不同
home_q_ctrl                 人工选定的安全起始姿态，可非零
joint_low/high              当前实验允许的绝对 q_ctrl 软件范围
```

不要求操作者人工精确移动到某个 raw，也不应把一次随手摆放的测量值强行定义为 home。正式实现应区分：

```text
hardware_joint_low/high     人工验证的较宽绝对安全范围
startup trajectory          在硬件安全范围内从当前测量姿态缓慢移动到home
experiment_low/high         home_q_ctrl附近的首轮+/-3 deg实验权限
```

启动流程应先读取并原样保持当前位置，验证当前位置位于 `hardware_joint_low/high`，然后执行速度/加速度受限的 home trajectory；到达并稳定后才启用 `experiment_low/high`。通过标准：home与整条启动路径均无自碰撞、无桌面碰撞、卸力时有支撑、夹爪raw安全；hardware YAML与 `configs/robots/so101.yaml` 中用于 joint reference 的home语义一致。

当前候选的首次 B3 配置为：所有电机 `P=20, D=32`，默认 `I=0`；`shoulder_lift` 使用 `I=1`，重力负载较大的 `elbow_flex` 使用 `I=3`；`Acceleration=32`，`Maximum_Acceleration=32`。wrist_flex 的软件 `home_q_ctrl` 对应 raw `2048`（不修改 LeRobot calibration midpoint）。每次 B3 运行前先核验配置身份：

```bash
conda run -n lerobot python scripts/validate_so101_config.py \
  --hardware-config configs/hardware/so101_follower.local.yaml
```

预期输出包含：

```text
pid: [20, 0, 32]
pid_i_by_motor.shoulder_lift: 1
pid_i_by_motor.elbow_flex: 3
hardware_startup_envelope_configured: true
```

然后在支撑和独立断电条件下运行一次 B3 startup-only 测试：

```bash
conda run -n lerobot python scripts/check_so101_startup.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --startup-seconds 3 \
  --startup-timeout-seconds 15 \
  --home-tolerance-deg 1 \
  --settle-seconds 3 \
  --output outputs/hardware/so101_pre_mpc/<RUN>/b3_startup_001.json \
  --enable-motion \
  --operator-supported-shutdown
```

通过标准：五个受控关节均在 `home_tolerance_deg=1` 内收敛；wrist_flex 的最终 Goal Position 回读应为 raw `2048`。失败输出必须保留，不能覆盖以前的运行文件。

### B4. 方向测试

在 home 确认后逐关节单独执行 `+2 deg`，其他关节保持。验证：

- 预期 ID 运动；
- 实体方向与 q_ctrl 正方向一致；
- 回读方向一致；
- 没有跳动、啸叫或线缆拉扯；
- 命令投影和 raw 目标都可审计。

当前仓库的 `SafeExcitation` 是五关节多频同时运动，不适合替代首次逐关节方向测试。若没有逐关节检查入口，该项属于实现阻塞，禁止直接跳到多关节 direct。

当前已有逐关节入口。每个关节从0.5度开始，逐级运行：

```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint shoulder_pan --direction positive --amplitude-deg 0.5 --seconds 10 \
  --output outputs/hardware/so101_pre_mpc/20260804_b_stage/b4_p20_i_elbow3_shoulder_pan_0p5deg_001.npz \
  --enable-motion --operator-supported-shutdown
```
```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint shoulder_lift --direction positive --amplitude-deg 0.5 --seconds 10 \
  --output outputs/hardware/so101_pre_mpc/20260804_b_stage/b4_p20_i_elbow3_shoulder_lift_0p5deg_001.npz \
  --enable-motion --operator-supported-shutdown
```
```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint elbow_flex --direction positive --amplitude-deg 0.5 --seconds 10 \
  --output outputs/hardware/so101_pre_mpc/20260804_b_stage/b4_p20_i_elbow3_elbow_flex_0p5deg_001.npz \
  --enable-motion --operator-supported-shutdown
```
```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint wrist_flex --direction positive --amplitude-deg 0.5 --seconds 10 \
  --output outputs/hardware/so101_pre_mpc/20260804_b_stage/b4_p20_i_elbow3_wrist_flex_0p5deg_001.npz \
  --enable-motion --operator-supported-shutdown
```
```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --joint wrist_roll --direction positive --amplitude-deg 0.5 --seconds 10 \
  --output outputs/hardware/so101_pre_mpc/20260804_b_stage/b4_p20_i_elbow3_wrist_roll_0p5deg_001.npz \
  --enable-motion --operator-supported-shutdown
```

### B5. home重复性与安全启动状态机

在至少三次独立的“卸力手动放置/重新上电/连接”过程中测试：

- 任意安全初始姿态能被读取并原样保持；
- 启动力矩时不会跳向EEPROM旧目标；
- 初始姿态位于已验证的hardware joint范围；
- startup trajectory不离开hardware joint范围；
- 到达home后的稳态误差和重复性可量化；
- 只有到达并稳定后才启用experiment envelope；
- Ctrl-C、异常和正常退出均在支撑条件下卸力。

当前实现已提供 `backend.startup_to_home()`：连接后先读取并保持当前位置，验证较宽的 `hardware_joint_low/high`，再以硬件包络中的限速/限加速度轨迹移动到 `home_q_ctrl`，到达实验包络后才允许 runner 开始。填写 hardware范围前，motion入口会拒绝执行。

启动轨迹不复用 `relative_target_limit` 的实验期±3度语义；它以宽hardware envelope、速度和加速度限幅保护，使存在静态跟随误差时仍可把目标持续推进到 home。

在不叠加direct激励的情况下验证该路径：

```bash
conda run -n lerobot python scripts/check_so101_startup.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --startup-seconds 3 --startup-timeout-seconds 15 --home-tolerance-deg 1 --settle-seconds 3 \
  --output outputs/hardware/so101_pre_mpc/<RUN>/b3_b5_startup_<RUN_INDEX>.json \
  --enable-motion --operator-supported-shutdown
```

## 6. 阶段 C：静止单读单写与总线基准

使用修订后的 raw-echo benchmark。它读取任意当前安全姿态，并把同一次读取的六个 raw 位置原样写回；它不使用 home 投影，也不主动把机械臂拉向 `joint_low/high`。

```bash
conda run -n lerobot python \
  dynamics_modeling/scripts/benchmark_so101_io.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --ticks 900 \
  --output outputs/hardware/so101_pre_mpc/<RUN>/hold_30s.json \
  --enable-hardware \
  --operator-supported-shutdown
```

通过标准：

```text
fatal_state_count                      0
control_path_execution P99             < 25 ms
wake_lateness P99                      < 2 ms
deadline_miss_rate                     < 1%
skipped_tick_rate                      < 0.1%
Goal Position readback mismatch        0
读取全部六电机                         100%
```

`software_range` 可以在纯 I/O benchmark 中报告，但不能被解释为通信失败；进入 direct 前必须解决。

`sample_count` 当前是按30 Hz tick对低频诊断缓存进行时间加权后的数量，不是独立的总线诊断读取次数。正式证据还必须记录诊断更新时间戳或独立诊断样本数，并证明诊断sample age始终满足安全监督要求。

旧版900 tick预检查曾得到：执行 P99 约6.85 ms、wake P99 约0.179 ms、0 deadline miss、静止电压12.0--12.3 V、最高温度41 C。但旧脚本曾经过 command projector，不能作为最终论文前正式 I/O 证据，必须使用当前 raw-echo 版本重跑。

### C1. 长时总线、诊断与退出测试

raw-echo版本依次运行30秒、10分钟和论文计划的最长单次持续时间，检查：

- read/write P50/P95/P99/max；
- retry次数和六电机完整返回率；
- Goal Position低频回读一致性；
- 诊断独立样本数与sample age；
- 电压、温度、电流和负载；
- 内存/线程无持续增长；
- Ctrl-C和人工异常退出后力矩关闭、串口释放；
- 重新连接不需要重启电脑，且不会跳动。

当前后端会自行执行单次读取重试并记录实际重试次数；benchmark 会保存独立诊断样本数、更新时间戳、sample age、重试统计和 Goal Position 回读不一致次数。仅在诊断全量读取成功时，诊断时间戳才更新。

### C2. 受控故障注入

先在fake bus完成，再在机械支撑和独立断电条件下进行实机低风险验证：

- 拔除USB/制造读取失败；
- Goal Position回读不一致；
- 诊断超过3秒未更新；
- 时间戳异常、跳过tick和严重deadline miss；
- 模拟warning/hard电压；
- 模拟50/55/60 C温度和持续大电流；
- runner异常抛出和Ctrl-C；
- history reset期间旧GPU packet延迟到达。

通过标准：分别进入预期的HOLDING、FAULT_LATCHED或TORQUE_DISABLED；清除packet、递增generation、禁止自动恢复锁存故障，并保留可审计原因。实机拔线测试不得在无支撑状态进行。

## 7. 阶段 D：direct 控制、动态电压和热测试

只有 B2--B4 与 C 全部通过后才能进入本阶段。

### D1. 逐关节 direct

顺序为：

```text
单关节 0.5 deg -> 1 deg -> 2 deg
每一级先运行5--10 s
每次只改变一个关节
```
```bash
  conda run -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --joint shoulder_pan \
    --direction oscillate \
    --amplitude-deg 2 \
    --seconds 10 \
    --output outputs/hardware/so101_pre_mpc/20260804_d_stage/d1_shoulder_pan_2deg_001.npz \
    --enable-motion \
    --operator-supported-shutdown
```

```bash
  conda run -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --joint shoulder_lift \
    --direction oscillate \
    --amplitude-deg 2 \
    --seconds 10 \
    --output outputs/hardware/so101_pre_mpc/20260804_d_stage/d1_shoulder_lift_2deg_001.npz \
    --enable-motion \
    --operator-supported-shutdown
```

```bash
  conda run -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --joint elbow_flex \
    --direction oscillate \
    --amplitude-deg 2 \
    --seconds 10 \
    --output outputs/hardware/so101_pre_mpc/20260804_d_stage/d1_elbow_flex_2deg_001.npz \
    --enable-motion \
    --operator-supported-shutdown
```

```bash
  conda run -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --joint wrist_flex \
    --direction oscillate \
    --amplitude-deg 2 \
    --seconds 10 \
    --output outputs/hardware/so101_pre_mpc/20260804_d_stage/d1_wrist_flex_2deg_001.npz \
    --enable-motion \
    --operator-supported-shutdown
```

```bash
  conda run -n lerobot python scripts/run_real_direct_control.py \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    --joint wrist_roll \
    --direction oscillate \
    --amplitude-deg 2 \
    --seconds 10 \
    --output outputs/hardware/so101_pre_mpc/20260804_d_stage/d1_wrist_roll_2deg_001.npz \
    --enable-motion \
    --operator-supported-shutdown
```

通过标准：方向正确、无可见振荡、无异常声音、无投影饱和、无 deadline miss、无 goal readback mismatch、能安全返回 home。

当前 `scripts/run_real_direct_control.py` 会让五个关节同时执行不同频率的正弦，只能用于逐关节方向测试完成后的多关节 direct：

```bash
conda run -n lerobot python scripts/run_real_direct_control.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --amplitude-deg 2 \
  --seconds 10 \
  --enable-motion \
  --operator-supported-shutdown
```

先通过2度，才能尝试最多3度。禁止扩大 `joint_low/high` 来消除尚未解释的投影或状态错误。

### D2. 动态电压表征

静止电压不能独立确定门限。至少记录：

- 静止卸力；
- 静止保持；
- 单关节2度；
- 多关节2度；
- 多关节3度；
- 最接近论文实验负载的轨迹。

每种状态报告每个电机及全体的 min/P1/median/P99/max。STS3215 返回电压 raw 的换算为 `raw * 0.1 V`。

当前12 V系统的候选值仅供动态数据审核：

```yaml
voltage_warning_low: 10.8
voltage_warning_high: 13.2
voltage_hard_low: 10.0
voltage_hard_high: 13.8
```

冻结前必须确认正常动态轨迹不会频繁跨越 warning；若出现明显压降，应先检查 PSU 容量、线缆、接头和电源板，不得仅通过放宽门限掩盖问题。四个门限必须同时填写，且满足：

```text
hard_low < warning_low < warning_high < hard_high
```

### D3. 温升测试

依次进行5、10、30分钟低权限 direct/hold，逐电机记录温度和电流。安全策略：

```text
50 C  residual归零，降低速度/加速度权限
55 C  FAULT_LATCHED，停止轨迹
温度继续上升或保持电流过大  提前请求支撑并卸力
60 C  支撑条件下立即卸力
```

通过标准：论文计划的最长单次运行内无关节达到50 C，诊断不过期，温度趋势稳定。当前30秒预检查中夹爪最高41 C，需要在长时测试中特别关注。

### D4. Acceleration 扫描

首轮固定：

```text
Acceleration=32
Maximum_Acceleration=32
```

只有32通过全部 direct、动态电压和热测试后，才允许分别测试64、128。不得经过 LeRobot 默认254再覆盖；每次连接必须在自定义安全 connect 内直接写入冻结值并读回。参数变化后必须重新采集数据和训练 real dynamics。

### D5. 真实状态估计器验收

训练和部署必须使用同一个4 Hz因果后向差分估计器。实机上至少验证：

- 静止时每关节q抖动、raw dq和filtered dq分布；
- 单关节慢速正弦中的速度符号、相位延迟和峰值；
- 编码器单count跳变不会频繁触发错误急停；
- 真正不可信的跳变、wrap和超速能够触发状态无效；
- reset后首帧dq语义一致；
- offline dataset重放得到的dq与在线记录逐样本一致；
- 记录并比较STS3215 Present_Velocity raw，仅作为诊断，不在未标定单位前替换部署估计器。

当前后端已采集五个受控关节的 `Present_Velocity` raw 作为诊断字段；它不替代部署中的4 Hz因果估计器。对采集文件运行：

```bash
conda run -n lerobot python scripts/report_so101_state_estimator.py \
  <SESSION.npz> --output outputs/hardware/so101_pre_mpc/<RUN>/state_estimator_report.json
```

### D6. 夹爪锁存专项测试

虽然夹爪不进入5-DOF模型，仍需验证：

- 每tick六电机写入时ID6始终为冻结的 `gripper_hold_raw`；
- 夹爪不会因单位混用把“度”解释为0--100；
- Goal Position回读与锁存raw一致；
- 10分钟及论文最长运行中的夹爪电流和温升；
- 卸力后夹爪/负载不会造成危险。

夹爪曾在30秒预检查中达到41 C，因此必须单独留档，不得只看五个受控关节。

## 8. 阶段 E：真实数据采集与 transition 审计

### E1. 小规模冒烟采集

在正式采集前先运行短 session，并保存独立文件：

```bash
conda run -n lerobot python \
  dynamics_modeling/scripts/collect_real_data.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output outputs/hardware/so101_pre_mpc/<RUN>/smoke_session.npz \
  --minutes 0.5 \
  --session-id smoke_001 \
  --enable-motion \
  --operator-supported-shutdown
```

采集器要求显式的 `--motion-mode`、`--amplitude-deg`、`--joint`、`--seed` 和 `--episode-id`，并在manifest中固定这些值；首次 smoke 建议使用 `--motion-mode multi_joint_sine --amplitude-deg 2`。在2度多关节 direct 未通过前，禁止采集。

### E2. 数据结构验证

```bash
conda run -n lerobot python \
  dynamics_modeling/scripts/validate_real_dataset.py \
  outputs/hardware/so101_pre_mpc/<RUN>/smoke_session.npz
```

必须满足：

- `actions == q_ref == transmitted_q_ref`；
- `next_state_timestamp_ns > state_timestamp_ns`；
- 无跨 history/estimator generation transition；
- command-delivery uncertain transition 默认不训练；
- Goal Position 回读不一致区间被追溯标记；
- `actual_dt` 被记录；训练只使用冻结的 nominal-30-Hz有效窗口；
- torque、true state 和外力字段不被伪造；
- 电压、温度、电流、负载、raw goal、requested/projected/transmitted命令可审计。

训练有效率、按 flag 的无效率和各关节 q/dq/action 分布必须单独报告。只有布尔结构检查通过不等于数据质量通过。

`validate_real_dataset.py` 现在实际计算history/estimator generation一致性，并检查shape、数组长度、NaN/Inf、dt窗口、诊断新鲜度、session/episode字段和manifest SHA-256。正式采集前仍须为每条规则保留损坏数据负向测试证据。

### E3. Model A 数据集

> **当前执行版本：** 旧的 12-session / 5-minute E3.1–E3.4 内容已被替代。请以 [SO101 实机 Model-A 数据采集协议（48 × 15 min）](so101-model-a-real-collection-protocol.md) 为唯一操作依据：48 个 session、每个 15 min、38/5/5 split、15 mm 精细网格虚拟桌面门禁。

首版局部模型目标约60分钟、12个独立 session；按 session 分为8 train、2 validation、2 test。每个 split 必须覆盖相同的 motion-mode 类型，禁止随机 transition 切分。正式工作空间采集使用 `hardware_joint_low/high` 各向内收缩 **1度** 后的范围；不得使用 B4 的 `home +/- 3 deg` 小包络替代。

> **桌面安全前置条件：** 关节范围合法不代表夹爪或连杆不会碰桌面。E3 必须使用官方 SO101 的精细 MuJoCo 碰撞网格；旧 `so101_nominal.xml` 的 capsule 仅保留给既有动力学实验，不能再作实机桌面安全结论。桌面平面暂按基座安装面 `z=0`，仅在基座无垫高、无翘起时成立。桌面 clearance **明确忽略** `shoulder_pan`（第1关节）及第1到第2关节之间、同属 MuJoCo `shoulder` body 的支架；`upper_arm`、`lower_arm`、wrist、gripper 与活动夹爪均受精细网格检查。TCP 中心不再单独设置高度门限。

#### E3.0 桌面 50 mm clearance 标定与离线预检

精细模型位于 `dynamics_modeling/robots/so101_fine/`，包含 STL 网格和启用的碰撞几何。`scene_table_guard_15mm.xml` 将**虚拟**桌面保护平面放在真实桌面（`z=0`）上方 15 mm：每个受保护网格必须与该平面保持严格正距离；接触或穿透即拒绝。因此该门禁要求真实桌面上方至少 15 mm，而不是额外的 50 mm clearance。基座视觉网格穿过该虚拟平面是预期现象，基座和已排除的 `shoulder` 不参与门禁。MuJoCo 对网格与平面的距离使用凸包，因而对桌面 clearance 是保守的；profile 会固定 XML 与全部 STL 的 bundle SHA-256，防止模型资产被静默替换。

当前 `home` 中被保护部件的最低点是 `upper_arm`，到真实桌面约 `104.6 mm`、到 15 mm 虚拟保护平面约 `89.6 mm`，因此未接触虚拟平面。`shoulder` 的约 `16.2 mm` 高度是已明确排除的第1关节/第1到第2关节支架，不参与此门限。`configs/hardware/so101_table_safety.mesh_guard15.local.yaml` 是操作者已批准的“基座水平、`q_ctrl` 与精细模型关节坐标相同”配置；不得使用 `so101_table_safety.model_tcp110.local.yaml` 或旧 capsule 预检启动 E3。

完成垫高和物理 table registration 后，创建并批准一个含 `mesh_collision` 的 local profile，再运行无硬件 I/O 的确定性轨迹预检。预检覆盖 home、五段参考、所有30 Hz离散点、每段预定位路径和回 home 路径；未知的实际开机姿态仍由采集器在启动时重新检查。通过标准：输出 `status: pass`，且 `overall_min_effective_clearance_m >= 0.050`。预检不替代实机运行时门禁；任何实测状态、预定位点或待发送目标不满足网格阈值时，采集器锁存故障、保持最后一个已验证安全目标且不再发送新的随机目标。

#### E3.1 单次 session 的内容、数据量和时长

每一个 session 的**有效记录时长固定为5分钟**，以30 Hz运行，共9000个控制 tick。每个激励段的首 tick 没有可配对的上一状态，因此通常产生约8995条 `(x_k, q_ref_k, x_k+1)` transition。启动回home、五段的预定位和结束回home不进入训练数据；实际操作者占用时间通常约5分50秒，若home收敛需要继续保持，则可能接近6分钟。

单次 session 的固定顺序如下。所有目标先在软件中按70%的冻结速度/加速度权限限幅，之后才经已验证的硬件安全包络发送。

| 顺序 | 有效记录时长 | 激励 | 说明 |
|---:|---:|---|---|
| 1 | 45 s | `hold` | 在有效工作空间内采样一个随机目标并保持 |
| 2 | 90 s | `single_joint_sine` | 单关节 sine；随 `session-index % 5` 轮换五个受控关节 |
| 3 | 75 s | `multi_joint_sine` | 五关节同时 sine |
| 4 | 60 s | `smooth_random` | 平滑随机 knot/reference 轨迹 |
| 5 | 30 s | `step` 或 `delta_ref_random` | 偶数 session 使用 step；奇数 session 使用 delta-reference random |

每段开始前，采集器会在宽硬件包络内预定位到该段起点；预定位动作不写入 dataset。任何干涉、线缆拉扯、异常声、失去支撑或异常温升均应立即断电或 `Ctrl+C`，不要等待该段结束。

#### E3.2 第一次采集

正式采集命令中，`session-index` 同时就是固定的 `split_group_id`。第一次使用 session 0：

```bash
# 0..7 使用 --split train；8..9 使用 --split validation；10..11 使用 --split test。
conda run -n lerobot python dynamics_modeling/scripts/collect_real_workspace_model_a.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output outputs/hardware/so101_model_a/model_a_workspace.npz \
  --session-index 0 --session-id e3_00 --split train --seed 20260805 \
  --workspace-margin-deg 1 --append \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard15.local.yaml \
  --enable-motion --operator-supported-shutdown
```

成功时，目录结构类似：

```text
outputs/hardware/so101_model_a/
├── model_a_workspace.npz
├── model_a_workspace.manifest.json
└── sessions/
    ├── model_a_workspace_e3_00.npz
    └── model_a_workspace_e3_00.manifest.json
```

其中：

- `sessions/model_a_workspace_e3_00.npz` 是本次不可覆盖的原始 session shard；
- `model_a_workspace.npz` 是所有已完成 session 的累计 canonical dataset；
- 两份 manifest 分别保存 session 证据和累计数据集的 identity、SHA-256、工作空间、split、模式及配置记录。

只有完整 session 通过结构验证后，shard 才会原子追加到 canonical `.npz`。失败、中断和重复 session 不会进入 canonical 训练集，但应保留其 shard 作为证据。每次追加后运行：

```bash
conda run -n lerobot python dynamics_modeling/scripts/validate_real_dataset.py \
  outputs/hardware/so101_model_a/model_a_workspace.npz
```

输出中的所有结构布尔项必须为 `true`。采集器默认以 `--append` 拒绝覆盖 canonical dataset 或已有 shard；不得手动删除或覆盖旧 session 来掩盖失败证据。

#### E3.3 下一次追加到同一个数据集

下一次不需要合并脚本，也不要复制或重命名旧 `.npz`。只要保持下列两项不变，即可严格追加：

```text
--output outputs/hardware/so101_model_a/model_a_workspace.npz
--append
```

同时必须换成新的 `--session-index` 和唯一的 `--session-id`。例如追加第二个训练 session：

```bash
conda run -n lerobot python dynamics_modeling/scripts/collect_real_workspace_model_a.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output outputs/hardware/so101_model_a/model_a_workspace.npz \
  --session-index 1 --session-id e3_01 --split train --seed 20260805 \
  --workspace-margin-deg 1 --append \
  --table-safety-config configs/hardware/so101_table_safety.mesh_guard15.local.yaml \
  --enable-motion --operator-supported-shutdown
```

该命令先创建 `sessions/model_a_workspace_e3_01.npz`，验证后才将其追加到原来的 `model_a_workspace.npz`。追加器会拒绝以下情况：不同的 robot/plant identity、不同工作空间或数组 schema、已存在的 session ID、重复的 episode ID、缺少 manifest 或未完成的 shard。因此不允许静默混入不同PID、标定、control_dt或安全范围的数据。

若某次在追加前失败，该 shard 不会污染 canonical dataset。重试该 `session-index` 时应使用新的 session ID，例如 `e3_03_retry1`；因为失败 shard 未追加，成功重试仍可使用同一 split group。若该 index 已成功追加，则不得再次采集并追加同一 index。

#### E3.4 12-session 顺序和 split

按以下顺序逐次人工运行；每完成一次都验证累计 dataset，再开始下一次。不得并行连接同一机械臂或自动连续运行12次。

| session index / split group | `--split` | 建议 session ID |
|---:|---|---|
| 0–7 | `train` | `e3_00` … `e3_07` |
| 8–9 | `validation` | `e3_08`, `e3_09` |
| 10–11 | `test` | `e3_10`, `e3_11` |

完成12次后，训练使用：validation group `8,9`，test group `10,11`。训练程序会明确拒绝随机transition切分、validation/test重叠，或不是12个独立session group的数据集。

推荐激励覆盖：

```text
hold/小扰动                    15%
单关节 sine/chirp              30%
多关节 sine                    25%
smooth random                  20%
step / delta-reference random  10%
```

工作空间采集器记录每一段的 mode、seed、episode、session、split、有效工作空间、目标回读和完整 plant identity。MPC-correlated local 数据只能在后续聚合阶段单独采集，不能混入原始Model A训练/测试集。

## 9. 阶段 F：real dynamics 训练与模型门禁

首版固定模型语义：

```text
GRU
history_len       8
state_dim         10
action_dim        5
target_mode       delta_state
loss              Huber
rollout steps     5
nominal dt        1/30 s
actual_dt input   否
```

训练示例：

```bash
conda run -n lerobot python dynamics_modeling/scripts/train_dynamics.py \
  --data_path <MODEL_A_DATASET.npz> \
  --robot_config configs/robots/so101.yaml \
  --real_defaults \
  --validation_group_ids <VALIDATION_GROUP_1>,<VALIDATION_GROUP_2> \
  --test_group_ids <TEST_GROUP_1>,<TEST_GROUP_2> \
  --epochs 100 \
  --save_dir outputs/so101_model_a
```

评估：

```bash
conda run -n lerobot python dynamics_modeling/scripts/evaluate_real_model.py \
  --dataset <MODEL_A_DATASET.npz> \
  --checkpoint outputs/so101_model_a/best_rollout_model.pt \
  --normalizer outputs/so101_model_a/normalizer.pt \
  --test-group-ids <TEST_GROUP_1>,<TEST_GROUP_2> \
  --output outputs/so101_model_a/real_model_gate.json
```

通过标准：learned model 在1、3、5、10步的 q RMSE 全部优于 persistence、constant velocity 和一阶舵机 baseline；同时审查：

- 每关节 RMSE；
- q P95 和最大误差；
- 每种 motion mode；
- commanded-motion 子集，避免 hold 样本稀释误差；
- 10步排序能力与异常 rollout；
- test session 完全隔离；
- checkpoint、normalizer 和 plant identity 完全匹配。

模型门禁未通过时禁止 shadow/active 论文实验。不得复用 ABB、UR5e 或 SO101 simulation normalizer。

## 10. 阶段 G：joint reference 与 direct 基线

正式 MPC 前必须冻结 joint-space reference：

- 所有点位于已验证的绝对软件包络；
- q、dq、ddq 满足当前速度/加速度权限；
- 起点与已验证 home/当前姿态兼容；
- reference 长度包含 execution、horizon、delay 和 preview padding；
- reference 文件、生成参数和 SHA-256 固定；
- 同一 reference 先完成 direct baseline；
- direct 输出 tracking、命令平滑性、deadline、温度和电压。

首轮不得使用任意6D位姿目标。任务空间实验必须另行完成至少五个有仪器参考构型的：

```text
q_kin = sign * q_ctrl + offset
```

校准，并采用 position-only、position+tool-axis 或显式 task mask。

当前 `run_real_direct_control.py` 不保存rollout、电压、温度或时序产物，不能独立形成论文direct baseline证据。正式基线前必须让direct入口与MPC入口保存同语义字段、相同reference和相同统计摘要。

## 11. 阶段 H：shadow MPC

shadow 模式运行完整 GRU、CEM、ASAP packet、feedback correction 和 OOD 记录，但实际发送 nominal，不发送 residual。

调用入口：

```bash
conda run -n lerobot python scripts/run_real_cem_mpc.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --real-mode shadow_mpc \
  --checkpoint <MODEL_A_CHECKPOINT> \
  --normalizer <MODEL_A_NORMALIZER> \
  --reference_mode joint_file \
  --reference_file <FROZEN_JOINT_REFERENCE> \
  --save_dir outputs/hardware/so101_pre_mpc/<RUN>/shadow \
  <其余已冻结CEM参数> \
  --enable-motion \
  --operator-supported-shutdown
```

运行前用 `--help` 核对继承自 simulation CLI 的参数名称，不得盲目照抄其他机器人命令。

shadow 至少收集2,000次有效 planner publication。审查：

- planner failure；
- packet late drop；
- packet expiry；
- packet history generation；
- state midpoint 到 publication/activation 的完整年龄；
- residual建议值及饱和率；
- prediction innovation；
- executed、selected action、predicted state OOD coverage；
- 30 Hz control thread 不因CUDA planner发生 deadline回归。

### H1. 延迟校准

```bash
conda run -n lerobot python scripts/calibrate_real_delay.py \
  <SHADOW_EVENTS.npz> \
  --control-dt 0.03333333333333333 \
  --guard-ms 5 \
  --output outputs/hardware/so101_pre_mpc/<RUN>/delay_calibration.json
```

active门禁：

```text
samples >= 2000
method = p99.5
late_drop_rate < 1%
packet_expiry_rate < 1%
```

不足2,000样本时脚本使用P99+5 ms guard，只能继续shadow调试，不能通过当前active入口门禁。

### H2. OOD包络

准备包含以下数组的 NPZ：

```text
training_tokens
validation_tokens
executed_tokens
selected_action_tokens
predicted_state_tokens
```

执行：

```bash
conda run -n lerobot python scripts/calibrate_real_ood.py \
  <OOD_TOKENS.npz> \
  --output outputs/hardware/so101_pre_mpc/<RUN>/ood_envelope.json
```

active门禁：executed history、selected action 和 predicted state coverage 均不低于99%。CEM选中的未来动作也必须在训练分布内，不能只检查当前实测状态。

### H3. shadow故障与恢复测试

在不施加residual的shadow模式中重复C2的时序、诊断和旧packet故障，验证：

- invalid state或严重dt异常后不继续使用旧GRU history；
- generation reset前已在GPU运行的结果发布后被拒绝；
- packet过期执行nominal，不执行旧residual；
- planner failure、late drop和OOD均不影响30 Hz I/O thread；
- 连续有效history重新积累前不恢复planner authority；
- shadow输出包含生成、发布时间、激活tick和状态midpoint年龄。

`run_real_cem_mpc.py` 现在连接并完成 `startup_to_home()` 后，以实际读取的硬件状态预热worker；它还拒绝 `configs/robots/so101.yaml` 的 `home_q` 与 hardware YAML 的 `home_q_ctrl` 不一致的运行。

## 12. 阶段 I：低权限 active smoke

以下全部通过后才能运行：

- A--H全部通过；
- 电压四阈值已冻结；
- delay calibration通过；
- OOD coverage通过；
- Model A artifact identity匹配；
- direct与shadow无未解释故障；
- 有独立断电和机械支撑；
- active数据写入独立目录，不进入Model A训练/测试集。

首轮权限：

```text
residual max       <= 2 deg
feedback max       <= 0.5 deg
kdq                0
nominal envelope   <= home附近3 deg
control rate       30 Hz
horizon            8..12
samples            64..128
CEM iterations     2
```

入口：

```bash
conda run -n lerobot python scripts/run_real_cem_mpc.py \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --real-mode active_mpc \
  --checkpoint <MODEL_A_CHECKPOINT> \
  --normalizer <MODEL_A_NORMALIZER> \
  --reference_mode joint_file \
  --reference_file <FROZEN_JOINT_REFERENCE> \
  --delay-calibration <DELAY_JSON> \
  --ood-envelope <OOD_JSON> \
  --save_dir outputs/hardware/so101_pre_mpc/<RUN>/active_smoke \
  <其余已冻结参数> \
  --enable-motion \
  --operator-supported-shutdown
```

第一次 active 只证明安全、时序和语义正确，不要求优于 direct。任何一次以下事件立即停止晋级：

- FAULT_LATCHED 或 TORQUE_DISABLED；
- command delivery uncertain；
- packet generation不匹配；
- residual持续饱和；
- OOD抑制频繁；
- 温度/电压越界；
- tracking明显恶化、振荡或异常声音；
- deadline或packet门禁不再满足。

active smoke还必须进行“运行中人工请求停止”测试：停止planner、停止新命令、进入安全hold、操作者确认支撑后卸力；不得把关闭进程后机械臂自由下坠视为正常停止。

## 13. 阶段 J：Model C 数据聚合与论文实验放行

低权限 active smoke 通过后：

1. 将 active-induced transition 单独保存；
2. 不修改或污染 Model A test sessions；
3. 用基础真实数据与 active数据训练 Model C；
4. 重新拟合 real normalizer；
5. 重复模型门禁、shadow、delay和OOD校准；
6. 再与同 reference 的 direct baseline 正式比较。

论文正式实验放行标准：

- 全部配置、代码 commit、calibration、plant identity、reference和模型 artifact 已冻结；
- direct、shadow、active使用相同合法 reference 和运行时权限；
- 至少报告 tracking、命令平滑性/振动代理、时序、packet、OOD、电压、温度和安全事件；
- 预先定义 seeds/session/重复次数和排除规则；
- 失败运行不删除；
- active相对direct有控制收益，或在相近tracking下显著改善平滑性/振动；
- 若只能安全运行但没有收益，只能报告“安全可运行”，不能宣称MPC性能优越。

## 14. 一页式晋级清单

```text
[ ] A1 hardware tests全部通过
[ ] A2 config/calibration identity通过
[ ] A3 quantization通过
[ ] A4 serial/GPU/software版本归档
[ ] A5 配置错误与安全连接负向测试通过
[ ] B1 物理、电源、支撑、断电检查通过
[ ] B2 每关节raw安全范围人工验证
[ ] B3 home和绝对joint envelope冻结
[ ] B4 五关节逐一方向测试通过
[ ] B5 跨上电home重复性和初始化状态机通过
[ ] C  raw-echo 30 Hz benchmark通过
[ ] C1 长时总线、独立诊断样本和异常退出通过
[ ] C2 故障注入与锁存/卸力语义通过
[ ] D1 逐关节direct 0.5/1/2 deg通过
[ ] D1 多关节direct 2 deg，最多3 deg通过
[ ] D2 动态电压完成并冻结四阈值
[ ] D3 长时热测试通过
[ ] D4 Acceleration/Maximum_Acceleration冻结
[ ] D5 实机q/dq状态估计器验收通过
[ ] D6 夹爪锁存、回读和长时温升通过
[ ] E1 smoke dataset通过
[ ] E2 transition和manifest审计通过
[ ] E3 12-session Model A数据集冻结
[ ] F  real dynamics 1/3/5/10步门禁通过
[ ] G  joint reference与direct baseline冻结
[ ] H  shadow >=2000 publication通过
[ ] H1 P99.5 delay、late drop、expiry通过
[ ] H2 三类OOD coverage均>=99%
[ ] H3 shadow故障、history reset和旧packet拒绝通过
[ ] I  低权限active smoke通过
[ ] J  Model C与全部门禁重跑
[ ] 论文实验配置与统计方案冻结
```

只有最后一项之前的全部门禁都有可追溯证据，才允许开始正式 MPC 论文实验。
