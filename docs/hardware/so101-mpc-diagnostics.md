# SO101 实机 MPC 诊断:训练数据构成、模型输入输出与"零修正"问题

> 状态快照:2026-08-08 20:20。本文档记录当前实机 MPC(Active ≈ Direct,零修正)问题的
> 完整诊断链 —— 训练数据是什么、模型输入输出是什么、为什么 CEM 在 H=6 下始终回到 baseline、
> 以及据此拆出的"两层天花板"。所有数字均来自冻结快照(`TMP/snap_delta/`,epoch 75)下的
> 确定性实验,可在离线复现(见 §8)。

---

## 1. 背景:问题现象

真机上 Active MPC 与 Direct IK baseline 的 TCP 轨迹几乎重合 —— **CEM 在每次 launch 时给出的
修正 residual 全部为 0(即 baseline),Active ≈ Direct**。这既不是传输/延迟问题(该问题已在
shadow 阶段标定并门禁通过),也不是 cost function 定义问题(见 §5.1),而是两层叠加的天花板:

1. **第一层(成本函数,绑定)**:H=6 的规划窗口太短,修正的"收益"永远装不下它的"代价"
   ——即使给一个完美模型,counterfactual 计算也表明它翻不了案。
2. **第二层(模型能力,次要)**:旧 20260807 模型对指令几乎"失明"(κ≈0.003–0.005),重训后
   (delta_dq_full / delta_state)修正了失明(κ≈0.13),但仍比理想一阶伺服(κ=0.49)低约 3.5 倍。

两层分别对应两条可操作的杠杆:H 与残余惩罚权重是第一层;模型质量(κ)是第二层。
本文档先讲清数据与模型(§2–§3),再讲诊断证据(§4–§6),最后是未决决策(§7)。

---

## 2. 训练数据构成

### 2.1 文件与规模

| 项 | 值 |
|---|---|
| 文件 | `outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz` |
| 总行数 | 1,565,690(过渡) |
| session | 58(session id `e3_00_retry09` … `e3_47_retry00` + `workspace_48_*` … `workspace_57_*`) |
| split 组 | 58 个 `split_group_ids`(0–57),每个 = 一个 session |
| 时间步长 | dt ≈ 0.0333 s(30 Hz;实测 `actual_dt` 中位 0.033334) |
| 有效目标 | `valid_target` 99.9% |

### 2.2 两大采集源(来源构成)

| | source 0(基础 48 session)| source 1(扩展 10 session)|
|---|---|---|
| 行数 | 1,295,760(82.8%) | 269,930(17.2%) |
| session id | `e3_00_retry09` … `e3_47_retry00` | `workspace_48_*` … `workspace_57_*` |
| 激励段 | **6 段**(原协议):single_joint_sine 30% / multi_joint_sine 25% / smooth_random 20% / hold 15% / step 5% / delta_ref_random 5% | **4 段**(新激励,任务 #23 追加):fast_sine 40% / fast_walk 40% / step_hold 16.7% / hold 3.3% |
| 目的 | 原 48-session 基线与门禁主体 | 补足 **fast 频率带**激励,让模型在高频段有数据可学 |

### 2.3 每条样本的字段语义(训练直接消费的三个数组)

npz 内与训练直接相关的数组:

- **`states` (N,10)** = `[q(5), dq(5)]`,**实测**状态(q 范围 ±1.7 rad,wrist_roll ±2.6;dq p5–p95 ±0.13–0.17 rad/s)。
- **`actions` (N,5) == `q_ref` == `transmitted_q_ref`**(逐位相等,max diff = 0.0)。
  **模型看到的指令是安全层实际下发到臂的那条**,不是 `requested_q_ref`(原始目标,与 transmitted
  的差可达 1.14 rad —— 因为经过裁剪/投影)。训练脚本 `validate_q_ref_dataset` 强制
  `actions == q_ref`(atol 1e-6)才放行,保证 position-control 语义。
- **`next_states` (N,10)** = 下一拍实测状态,配合 `valid_target` 过滤(99.9% 有效)。

### 2.4 数据划分(关键,且与直觉不同)

| 集合 | split_group_ids | 来源 | 作用 |
|---|---|---|---|
| **test** | 43–47 | source 0 | **构造数据集时整体剔除**(`exclude_split_group_ids`),不参与 train/val,全程冻结 |
| **val** | 38–42 | source 0 | 每 epoch 验证(`split_dataset_by_group_ids`) |
| **train** | **0–37 ∪ 48–57** | source 0 **∪ source 1** | 训练窗口 |

⚠️ **扩展的 10 个 session(groups 48–57)全部进了训练集,不是留出集**。所以测试/验证完全
来自 source 0 的 6 段协议;高频新激励段只以训练集成员身份存在。

### 2.5 采样(过采样,数值修正)

训练命令传 `--source_weights 0:1,1:4`,解析后归一化为 `{0: 0.2, 1: 0.8}`
(config.yaml `sampler.source_weights`)。逐 epoch 采样(WeightedRandomSampler,replacement):

- `samples_per_epoch` = 2,537,475
- source 0:窗口 507,495,权重 0.2 → 期望 ~1.0× 遍历(每个窗口每 epoch 看 ~1 次)
- source 1:窗口 133,659,权重 0.8 → 期望 **2,029,980 样本 / 133,659 窗口 = 15.19×** 重采样

**结论:source 1(高频扩展集)每 epoch 被反复看过 ~15 次**,有效占比约 80%。先前把
"4 倍"挂在嘴边的说法不准确 —— 那是归一化前的权重比,真实重采样倍率是 15.19×。

---

## 3. 模型输入 / 输出

### 3.1 架构

```
GRUDynamics (单层 GRU, hidden=256, num_layers=1)
input:  (batch, history_len=16, 15)  →  GRU(15→256)  →  Linear(256)→SiLU→Linear(256→10)  →  (batch, 10)
```

关键超参(见 `gru_20260808_171118/config.yaml`):`model_type=gru`,`history_len=16`,
`hidden_size=256`(默认),`num_layers=1`,`target_mode=delta_state`,`control_dt=1/30`,
`loss_type=huber`,`lr=1e-4`,`batch_size=8192`,`rollout_loss_steps=20`,`rollout_loss_weight=0.025`,
`q_weight=1.0`,`dq_weight=1.0`,`seed=10`。

### 3.2 输入 token(每条样本 = 16 步历史窗口)

窗口第 t 步的 token:

```
token_t = concat([ states[t](10 维: q+dq), actions[t](5 维: transmitted_q_ref) ])   →  15 维
```

- 输入张量形状 `(batch, 16, 15)`,由 `dataset.py::__getitem__` 构造:
  `x = cat([states[start:end], actions[start:end]], dim=-1)`,即 16 步 × [状态,指令]。
- 窗口边界由 `episode_ids` 限定(不同 episode 断窗);`valid_target` 全窗口掩码
  (`_all_valid_window_mask`)—— 任何含审计行/时序重置的窗口直接丢弃(递归输入代表整段历史,
  不能跨越坏行)。
- 归一化(`normalize_sequence_input`):前 10 维用 `state_mean/std`,后 5 维用
  `action_mean/std`,按通道归一。normalizer 只由 train 窗口拟合,val/test 不泄漏。

### 3.3 输出与目标

`target_mode = delta_state`:目标是窗口末步的**单步状态增量**:

```
delta = next_states[end-1] − states[end-1]   →  [Δq(5), Δdq(5)],共 10 维
```

- 模型输出 10 维**归一化**的 Δq/Δdq 预测;`denormalize_delta` 还原物理单位。
- 推理重建(`integration.py::reconstruct_next_state` 的 delta_state 分支):
  `next = state + pred_target` —— 位置和速度的增量都由模型直接预测,位置增量不经过数值积分。
  (对比 `delta_dq` 模式:`dq_next = dq + pred; q_next = q + dq_next·dt`。)
- **损失** = huber(单步)+ 20 步 rollout 损失(权重 0.025)。长程滚动一致性是训练目标的一部分。

### 3.4 训练现状(2026-08-08 20:20 核实)

- 训练已**停止**于 epoch 76(`latest_model`),未跑完 `epochs=200`;目录内无 crash log,
  停止原因未确认(可能被手动终止)。
- `best_model` / `best_rollout_model` 均冻结于 **epoch 75**(val_loss 2.1464,rollout_loss 14.70),
  epoch 76 略升(2.1495 / 14.75)→ epoch 75 确为当前最优。
- 离线诊断用快照 = `TMP/snap_delta/{best_rollout_model.pt, normalizer.pt}`(epoch 75 的冻结拷贝),
  保证确定性复现。

### 3.5 与 CEM 问题直接相关的两个语义

1. **模型条件在 `transmitted_q_ref`(实际下发)上** → 离线 CEM 测试里用 `reference + residual`
   喂给它做预测,与训练语义一致(指令 = 最终下发)。
2. **模型是"位置增量直接预测"** → 前文测的 κ(对恒定指令偏移的累计位置响应;理想一阶 τ=0.3s
   应为 0.49)直接反映该增量预测的质量:重训后 0.13–0.14,仍比理想低 ~3.5×。这是"第二层天花板"
   的量化证据。

---

## 4. 诊断方法:CEM 离线决策测试

所有"为什么 CEM 返回 baseline"的结论都来自同一套离线测试框架
(`TMP/cem_decision_test.py` + 各 sweep 脚本),冻结真实录制的一个 rollout
(`active_circle_p0/rollout.npz`)和对应的 reference 轨迹,在**固定 tick** 重放完整的 CEM
决策(采样 128、elite 8%、2 次迭代、强制 baseline 候选、按 `(cost, preference)` 选最优点,
`execute=lowest_cost`)。这样把"上机跑一次"压缩成"离线确定性重放",排除了网络抖动与随机性。

关键常量:`HORIZON=6`,`residual_max=2°`,`residual_scale=0.5×residual_max`,
cost 权重与 sim 完全一致(见 §5.1)。

---

## 5. 两层天花板分解

### 5.1 第一层(绑定):成本函数 —— H=6 太短,收益永远装不下代价

**成本函数定义与 MuJoCo 仿真完全一致**(`mpc/cost_functions.py::joint_space_tracking_cost`,
权重相同:`w_q=1.0, w_dq=0.10, w_residual=0.20, w_first=0.20, w_servo=0.05,
w_residual_velocity=0.05, w_residual_acceleration=0.02, w_qref_velocity=0.05,
w_qref_acceleration=0.02, w_terminal=0.0, w_joint_limit=10.0, w_dq_limit=5.0,
temporal_discount=0.95`)。**实机与仿真的差别不是定义,是参数**:

| 参数 | MuJoCo 仿真(UR5e/ABB) | so101 实机 |
|---|---|---|
| residual_max | 0.12–0.20 rad = 7–11.5° | **2°** |
| horizon H | 20 | **6** |
| delay(anticipation) | 6 | 2 |
| servo_scale | 0.025–0.08 | 1.0(=57°) |

**为什么这层是"绑定"的**:一次修正(如 elbow 2°)的代价 ≈ 残余惩罚的平方项(residual 0.20
+ first 0.20 + residual_velocity 0.05 + residual_acceleration 0.02,合计在 2° step 下 ≈ 0.9),
而它的收益(把未来 H 步的位置跟踪误差压低)受两个因素钳制:
- **H=6 只有 0.2 s**,而 plant 时间常数 shoulder_lift ≈ 0.8 s、pan/elbow ≈ 0.3 s
  → 修正的效果在窗口内根本来不及体现;
- 模型预测的位置响应 κ≈0.13,进一步压低收益(见 §5.2)。

证据(counterfactual + 扫描,全部在冻结快照上):

1. **完美模型 counterfactual**:即使假设预测=真值一阶响应,H=6 下修正的收益也 < 代价
   (此前估算代价 ~0.58 仅含 residual 项,实际全代价 ≈0.9,更翻不了)。
2. **设置扫描** `cem_settings_sweep.py`:w_residual 从 0.20 一路砍到 0.025(8×)、w_first
   0.20→0.10,在 tick {300,400,600,800} **全部返回 baseline**。单纯降权不翻转 H=6。
3. **快照扫描** `cem_snapshot_sweep.py`(冻结 e75):H=6 在任何权重组合下都 baseline;
   H=12/20 + 降权(w_res 0.05,w_first 0.10)才开始在 tick 400 出 0.2–0.43° 的非零修正
   (残差全部 ≤2°,安全)。**H 是解锁修正的必要条件,不是充分条件**。

### 5.2 第二层(次要):模型能力 —— κ 从"失明"到"不足"

κ = 模型对恒定指令偏移(0.5°)在 H=6 窗口内的累计位置响应(归一化),理想一阶 τ=0.3s 为 0.49。
用 `TMP/kappa_compare.py` 对同一 real rollout 测三种模型:

| 模型 | 目标模式 | κ_accum(H=6) | 诊断 |
|---|---|---|---|
| **OLD 20260807**(`gru_20260807_130024`) | delta_dq | **0.003–0.005** | **指令失明**:对命令偏移几乎无响应 —— 这才是"候选轨迹完全一致"的真正原因 |
| delta_dq_full(`gru_20260808_135847`) | delta_dq | 0.12–0.125 | 重训修好失明 |
| delta_state e75 快照(`snap_delta`) | delta_state | **0.13–0.14** | 重训修好失明,但仍比理想 0.49 低 **~3.5×** |

即:用户最初问的"为什么模型预测的候选轨迹完全一致",答案是**旧模型对指令失明**;重训
(扩展数据 + source 1 过采样 + delta_state)把 κ 从 0.005 拉到 0.13,问题从"完全没反应"
升级为"反应不足 3.5 倍"。

### 5.3 残余惩罚是"锁":全关后 CEM 敢动了

`cem_residual_off_sweep.py` 把 4 个残余惩罚全关(w_residual=w_first=w_residual_velocity=
w_residual_acceleration=0),在 **H=6 也解开了锁**:

| 权重组合 | H | tick 400 | tick 800 |
|---|---|---|---|
| 生产值(0.20/0.20/0.05/0.02) | 6 | baseline | baseline |
| 全关(0/0/0/0) | **6** | **1.83°**(+0.7%) | **1.45°**(+0.1%) |
| 全关 | 12 | 1.05°(+2.2%) | baseline |
| 全关 | 20 | 1.93°(+3.4%,≤2°安全) | baseline |

→ 残余惩罚就是"锁"。但这把锁是有意义的:**全关 first/velocity 惩罚上真机 = step 抖动风险,
不安全**。现实的候选是中等削减(0.05/0.05/0.02/0.005)+ H=12(在 tick 400 出 0.2–1.38°
非零修正),仍是"有条件解锁"。

---

## 6. 证据链汇总(全部可离线复现)

| # | 实验 | 脚本 | 结论 |
|---|---|---|---|
| 1 | 完美模型 counterfactual | (会话内推演) | H=6 下收益<代价,模型再完美也翻不了 |
| 2 | 重训后 CEM(H=6) | `cem_decision_test.py` MODELS[1]/[2]/[3] | 仍全 baseline —— 模型变好 ≠ 解锁 |
| 3 | 设置扫描 w_res×w_first | `cem_settings_sweep.py` | 8× 降权不翻转 H=6,全 baseline |
| 4 | 快照扫描 H×w(冻结 e75) | `cem_snapshot_sweep.py` | H=6 永远 baseline;H≥12+降权 → tick400 非零,残差 ≤2° |
| 5 | 残余惩罚全关 | `cem_residual_off_sweep.py` | 全关后 **H=6 出 1.45–1.83°**,锁在残余惩罚上 |
| 6 | κ 对比三种模型 | `kappa_compare.py` | OLD κ=0.005(失明);重训后 κ=0.13–0.14,仍比理想 0.49 低 3.5× |
| 7 | 数据构成核实 | (本会话 npz 检查) | actions==transmitted_q_ref;source1 过采样 15.19×;扩展集全在 train |

---

## 7. 未决问题与开放决策

1. **训练收敛未决**(任务 #30):训练停在 epoch 76,rollout_loss 仍在下降(14.75)而 best 在
   epoch 75。κ 是否在更长训练后向 0.3+ 走(改写"3.5× 不足"结论)仍开放。
   需要用户决定:继续训(补到 200)/ 冻结现状 / 用 epoch 75 直接上真机。
2. **上真机的配置选择(决策门,尚未拍板)**:
   - A. H=12 + 中等降权(0.05/0.05/0.02/0.005)—— 实测在 tick 400 出非零修正,安全余量仍够
   - B. 等训练收敛后再定 —— 可能拿到更好的 κ,但需要时间
   - C. 维持 H=6 + 全关残余惩罚 —— 解锁最大但 step 抖动风险,不推荐真机
3. **数据面未决**:source 1 高频激励只存在于 train 集,val/test 无 —— 若后续要评估高频段
   泛化,需要把部分扩展 session 划入留出。
4. **训练为何停在 epoch 76** 需确认(无 crash log)。

---

## 8. 复现路径(关键命令)

```bash
# 冻结快照(已存在 TMP/snap_delta/)
cp dynamics_modeling/outputs/checkpoints_real/gru_20260808_171118/{best_rollout_model.pt,normalizer.pt} /home/xinlei/.claude/jobs/834a4b6e/tmp/snap_delta/

# 冻结快照上重跑全部 sweep(Python 侧需 ROOT/TMP 在 sys.path)
conda run -n lerobot python /home/xinlei/.claude/jobs/834a4b6e/tmp/cem_snapshot_sweep.py
conda run -n lerobot python /home/xinlei/.claude/jobs/834a4b6e/tmp/cem_residual_off_sweep.py
conda run -n lerobot python /home/xinlei/.claude/jobs/834a4b6e/tmp/kappa_compare.py

# 数据核实
conda run -n lerobot python - <<'PY'
import numpy as np
a=np.load('outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz')
print(a['states'].shape, a['actions'].shape)
print('actions==q_ref:', np.allclose(a['actions'], a['q_ref'], atol=1e-6))
print('actions==transmitted:', np.allclose(a['actions'], a['transmitted_q_ref']))
PY
```

冻结快照 / sweep 脚本位于作业临时目录,如需长期保留应复制到仓库内(如
`dynamics_modeling/outputs/checkpoints_real/gru_20260808_171118/` 旁加 `diagnostics/`)。
