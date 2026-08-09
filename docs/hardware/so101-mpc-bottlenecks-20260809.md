# SO101 实机 MPC 当前瓶颈与近期测试总结

更新时间：2026-08-09   
对象：SO101 follower、30 Hz、u-q（`q_ref_minus_q`）GRU 模型、H=6 residual MPC。

## 结论先行

当前 active MPC 已经能够对命令产生明显响应，但在真机圆轨迹上仍没有超过 Direct IK：

- Direct IK shadow：位置 RMSE **0.920°**。
- 最新 active v3：位置 RMSE **0.935°**。
- 最新结果没有 planner deadline miss、packet expiry 或真正的 late drop；问题不在 CUDA 推理速度或串口发送。
- 启动阶段的无意义修正已经通过 gate 消除。剩余主要问题是：**CEM 选择的修正对真机跟踪没有稳定收益，且 planner 内部投影与真机安全投影仍不完全一致。**

因此目前不应把问题归结为“模型完全命令盲”。模型已经能看到命令，瓶颈转移到了模型精度、cost 权衡和在线命令投影的一致性。

## 1. 当前部署模型

### 1.1 u-q epoch14

| 项目 | 值 |
|---|---|
| checkpoint | `dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt` |
| normalizer | 同目录 `normalizer.pt` |
| 架构 | 单层 GRU，hidden size 256，单步输出 10 维 |
| history | 16 步 |
| 输入状态 | `[q, dq]`，10 维 |
| 输入命令 | `u-q = q_ref - q`，5 维 |
| 总输入 | 15 维 token |
| 输出目标 | `delta_state = [Δq, Δdq]`，10 维 |
| dt | 0.033333 s |
| loss | 单步 Huber + 20 步 rollout loss |
| rollout loss weight | 0.025 |
| batch / micro-batch | 8192 / 1024 |
| 学习率 / epoch | 1e-4 / 14 |
| 数据 source weight | source 0:0.2，source 1:0.8 |

epoch14 训练日志中的验证值约为 `val_loss=1.7549`、`val_rollout_loss=7.3340`。该 checkpoint 是当前实机测试使用的模型。

### 1.2 κ（命令敏感度）

κ 是在固定真实 rollout 上施加 counterfactual 命令偏移后，H=6 内预测位置响应的累计增益。它不是训练 loss，也不是直接的跟踪误差；它衡量模型是否相信“改变命令会改变位置”。

理想一阶伺服 `τ=0.3 s` 的参考 κ 约为 **0.49**。

| 模型/epoch | κ 平均 | tick 300 | tick 400 | tick 600 | 诊断 |
|---|---:|---:|---:|---:|---|
| OLD 20260807 | 0.003–0.005 | — | — | — | 基本命令盲 |
| absolute-u epoch14 | 0.042 | 0.041 | 0.044 | 0.042 | 有响应但很弱 |
| absolute-u epoch30 | 0.094 | 0.093 | 0.095 | 0.094 | 仍明显偏弱 |
| absolute-u epoch45 | 0.115 | 0.115 | 0.115 | 0.115 | 逐步增强但远低于理想 |
| u-q epoch1 | 0.348 | 0.346 | 0.352 | 0.347 | 已能看到命令 |
| u-q epoch7 | **0.601** | 0.702 | 0.547 | 0.554 | κ 峰值 |
| u-q epoch14 | **0.475** | 0.591 | 0.411 | 0.423 | 最接近理想参考 |
| u-q epoch20 | 0.360 | 0.497 | 0.276 | 0.307 | 开始衰减 |
| u-q epoch30 | 0.285 | 0.440 | 0.177 | 0.237 | 命令响应继续变弱 |
| u-q epoch45 | 0.233 | 0.388 | 0.125 | 0.187 | 明显衰减 |

完整 κ 结果在 `docs/hardware/so101-input-ablation-20260809/kappa_by_epoch.csv`；epoch14 细节在 `u_minus_q_epoch14/epoch_014/kappa.json`。

这里的趋势说明：继续优化验证 rollout loss 不一定会提高 counterfactual command authority。u-q 在 epoch7–14 的命令敏感度明显优于后期 checkpoint，因此当前实机采用 epoch14 是基于 κ 的选择，而不是单纯选择最后一个 epoch。

## 2. 当前 cost function

在线 planner 使用 residual cost，候选中强制加入 baseline，并以 `lowest_cost` 在 best / mean / baseline 之间选择。

总 cost 可以概括为：

```text
J = J_q + J_dq
  + J_residual + J_servo
  + J_residual_velocity + J_residual_acceleration + J_first
  + J_qref_velocity + J_qref_acceleration
  + J_terminal + J_joint_limit + J_dq_limit
```

当前默认权重：

| 项 | 权重 | 含义 |
|---|---:|---|
| `w_q` | 1.0 | 预测位置跟踪误差 |
| `w_dq` | 0.10 | 预测速度跟踪误差 |
| `w_residual` | 0.20 | residual 幅值惩罚 |
| `w_servo` | 0.05 | 命令相对当前模型状态的偏移 |
| `w_residual_velocity` | 0.05 | residual 速度惩罚 |
| `w_residual_acceleration` | 0.02 | residual 加速度惩罚 |
| `w_first` | 0.20 | 第一拍 residual 速度惩罚 |
| `w_qref_velocity` | 0.05 | q_ref 速度惩罚 |
| `w_qref_acceleration` | 0.02 | q_ref 加速度惩罚 |
| `w_terminal` | 0.0 | 终端项关闭 |
| `w_joint_limit` | 10.0 | 关节位置软边界 |
| `w_dq_limit` | 5.0 | 关节速度软边界 |
| temporal discount | 0.95 | 时间折扣 |

当前物理约束：

- residual 最大权限：每关节 **2°**。
- q_ref 速度上限：**0.25 rad/s**。
- q_ref 加速度上限：**1 rad/s²**。
- state velocity soft limit：**0.5 rad/s**。
- horizon：**6 步 = 0.2 s**。
- anticipation delay：**2 步**。
- `servo_scale=[1,1,1,1,1]`。
- residual scale 为 `0.5 × residual_max`，即约 **1°**；residual velocity scale 约 **1.047 rad/s**，acceleration scale 约 **31.416 rad/s²**。
- 位置跟踪 scale 由 reference 自动校准并限制在 0.04–0.08 rad；速度跟踪 scale 至少为 0.25 rad/s。

离线虚拟 CEM 已证明 H=6 下 cost 对修正非常保守：当前配置在很多 tick 选择 baseline；增大 H 或明显降低 residual 惩罚后才更容易出现非零修正。但把惩罚全部关掉会牺牲真机平滑性和安全裕度，因此不能直接作为部署方案。

## 3. 近期测试时间线

### 3.1 离线模型与 CEM

| 测试 | 产物 | 结果 |
|---|---|---|
| κ epoch sweep | `docs/hardware/so101-input-ablation-20260809/` | u-q epoch7 峰值约 0.601，epoch14 约 0.475，后期逐步下降 |
| u-q epoch14 virtual CEM | `docs/hardware/so101-input-ablation-20260809/u_minus_q_epoch14/virtual_cem_h6/virtual_cem_report.json` | 部分 tick 有修正收益，但不是所有 tick 都优于 baseline |
| 旧模型 κ 对比 | 旧 20260807 / 重训 delta-dq / delta-state | 旧模型 κ≈0.005；重训后约 0.13；u-q epoch14 counterfactual κ 更高，约 0.475 |

### 3.2 实机 shadow 与 active

所有圆轨迹测试均使用：

```text
reference: outputs/hardware/so101_pre_mpc/20260808_refs_6phase/circle_p0/joint_reference_mpc.npz
horizon: 6
num_samples: 128
cem_iters: 2
model: u-q epoch14
```

| 测试 | 主要变化 | RMSE | 最大误差 | 请求→安全投影 P95 | 结论 |
|---|---|---:|---:|---:|---|
| `shadow_circle_p0` | Direct IK 名义轨迹，不应用 MPC | **0.920°** | 3.871° | — | 当前基线 |
| `active_exploratory_circle` | 旧 active 路径，raw residual 重构 | 0.955° | 4.335° | 约 1.28° | 略差于基线 |
| `active_projected_circle` | 发送 planner 的 projected absolute q_ref；速度状态 bug 尚未修复 | 0.925° | 4.065° | **1.87°** | 命令投影偏差很大 |
| `active_projected_circle_v2` | 修复 adapter 初始速度与 `Δq/dt` | 0.943° | 3.971° | **1.04°** | 命令对齐改善，但跟踪未改善 |
| `active_projected_circle_v3` | 增加静态 reference startup gate，tick 83 前只走 Direct IK | **0.935°** | 3.924° | 1.41° | 启动乱动消失，整体仍未超过 baseline |

最新 v3 的运行事实：

- 1087 个控制 tick 完成。
- `planner_applied=True` 为 999 tick；前 83 tick 的 gate 正常生效。
- OOD 无效 3 tick。
- 无 deadline miss、packet expiry、planner failure 或真正的 late drop。
- 安全投影到最终 actuator command 的 P95 只有约 0.145°；大偏差主要发生在 planner 请求和硬件安全投影之间。
- v3 运动段（tick≥83）RMSE 约 0.959°，shadow 对应运动段约 0.940°。
- v3 各关节 RMSE：`[shoulder_pan 1.434°, shoulder_lift 0.812°, elbow 1.040°, wrist_flex 0.745°, wrist_roll 0.141°]`。

## 4. 已发现并修复的问题

### 4.1 rollout loss 梯度

训练路径移除了 `rollout_dynamics_batch()` 的无条件 `torch.no_grad()`，推理路径仍保持 no-grad；增加了 rollout-only 梯度测试。

### 4.2 模型输入

增加 `q_ref_minus_q`，即模型看到相对命令 `u-q`。相比 absolute-u，u-q 显著提高了 κ，epoch14 达到约 0.475。

### 4.3 requested command 与 planner projected command 不一致

planner packet 现在携带 CEM 实际评分过的 absolute projected q_ref，active runner 优先使用该序列，而不是重新计算 `nominal + residual`。

### 4.4 adapter 速度状态错误

旧代码把位置差 `Δq` 直接当作速度；同时第一次提交把 home 位姿相对全零误当成初始速度。现已改为：

```text
previous_velocity = (q_ref[t] - q_ref[t-1]) / control_dt
```

并将第一次 home snapshot 的速度初始化为 0。该修复将请求→安全投影 P95 从约 1.87° 降到约 1.04°（v2）。

### 4.5 静态启动段乱动

reference 前 83 tick 是 home hold，但 CEM 在此期间仍可能输出 residual。v3 增加了自动 startup gate：检测到 reference 首次偏离初始姿态超过 1e-3 rad 前，清空 packet、不提交 planner，只执行 Direct IK。

## 5. 当前瓶颈排序

### P0：active 修正没有稳定的真实收益

最新 v3 仍略差于 Direct IK，尤其 `shoulder_pan` 变差约 0.35° RMSE。说明 CEM 正在输出“模型认为有益、但真机上未证实有益”的修正。

### P1：planner projection 与硬件 projector 仍有动态不一致

v3 的请求→安全投影 P95 仍约 1.41°，且约 929 个 tick 触发速度/加速度投影。最终发送和安全投影之间只有约 0.15°，所以问题主要在 planner 预测的命令序列与真机 projector 状态/重规划重叠之间，而不是通信链路。

### P1：H=6 和 2° 权限限制收益窗口

H=6 只有 0.2 s，模型的响应和真机 servo 都来不及充分体现修正收益；同时 residual 权限只有 2°，cost 中的 residual、first、速度和加速度项会进一步压低可见收益。

### P1：模型 κ 随训练阶段变化，训练 loss 不是充分选择标准

u-q epoch7 κ≈0.601，epoch14≈0.475，epoch30≈0.285，epoch45≈0.233。后期可能继续降低 logged-trajectory validation loss，却压低 counterfactual command authority。因此必须同时监测 κ、离线 CEM 和实机 shadow，而不能只看 val loss。

### P2：OOD 门禁和实时性能

v3 只有 3 个 OOD 无效 tick，规划延迟 P95 约 27 ms，无 deadline miss。当前不是主要瓶颈，但 active 测试仍应保留这些门禁。

### P2：启动 home 收敛边界

曾出现 `shoulder_pan` home 误差 1.055°、超过 1° 门限 0.055° 的启动失败。这发生在 MPC 之前，不属于模型问题；重试后可继续测试，暂不建议直接放宽 home 门限。

## 6. 建议的下一步实验

1. 固定 epoch14 模型和 reference，先做“只改变 cost、不改变模型”的 A/B：提高 `w_residual` 或限制 residual 只在运动段生效，确认 shoulder_pan 的恶化是否消失。
2. 离线逐 tick 记录 planner projected q_ref、硬件 projector 重放结果和真实 transmitted q_ref，定位剩余 1° 级 projection discrepancy 是 packet overlap、delay anchor 还是速度状态不同步。
3. 以 Direct IK 为硬门槛：只有当 active 的运动段 RMSE、P95 和最大误差至少不劣于 shadow，才继续扩大 residual 权限或 horizon。
4. 若要解锁 MPC 收益，优先评估 H=12，而不是继续盲目扩大 GRU；同时保持 2° residual 上限和硬件 projector。
5. 训练模型时把 κ 和离线 CEM 选择结果纳入 checkpoint 选择指标，避免后期 val loss 改善却 κ 衰减。

## 7. 关键产物索引

- κ 汇总：`docs/hardware/so101-input-ablation-20260809/kappa_by_epoch.csv`
- κ 诊断：`docs/hardware/so101-input-ablation-20260809/kappa_diagnosis.md`
- epoch14 κ：`docs/hardware/so101-input-ablation-20260809/u_minus_q_epoch14/epoch_014/kappa.json`
- epoch14 virtual CEM：`docs/hardware/so101-input-ablation-20260809/u_minus_q_epoch14/virtual_cem_h6/virtual_cem_report.json`
- shadow：`outputs/hardware/so101_pre_mpc/20260809_epoch14/shadow_circle_p0/`
- 旧 active：`outputs/hardware/so101_pre_mpc/20260809_epoch14/active_exploratory_circle/`
- projected active：`outputs/hardware/so101_pre_mpc/20260809_epoch14/active_projected_circle/`
- adapter 修复后：`outputs/hardware/so101_pre_mpc/20260809_epoch14/active_projected_circle_v2/`
- startup gate 后：`outputs/hardware/so101_pre_mpc/20260809_epoch14/active_projected_circle_v3/`
