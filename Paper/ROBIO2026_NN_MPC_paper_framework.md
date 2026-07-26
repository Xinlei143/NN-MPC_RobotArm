# ROBIO 2026 论文框架

## Delay-Aware Asynchronous Learned Residual CEM-MPC with a 100 Hz Execution Layer for Position-Controlled Manipulators

> 对应仓库：`Xinlei143/NN-MPC_RobotArm`
> 当前论文基线：待正式实验前冻结唯一 clean-worktree commit，并写入 paper manifest。
> 目标会议：IEEE ROBIO 2026
> 目标篇幅：6 页；实验和时序证据完整时最多扩展至 7 页。

---

# 0. 论文定位

## 0.1 核心问题

本文不把“神经网络用于机械臂 MPC”本身作为主要创新，而回答一个更具体的部署问题：

> **Can activation-time alignment recover the stale-plan degradation of a learned CEM planner whose solution time exceeds the command period, while retaining the practical performance of a strong Direct-IK nominal controller?**

系统包含三个时间尺度：

- MuJoCo 物理积分：500 Hz；
- 位置命令执行层：100 Hz；
- GRU/CEM 后台规划器：由实际求解时间决定，预计约 20--30 Hz。

标题中的 100 Hz 只描述 execution layer，不暗示 CEM 以 100 Hz 完成在线优化。

## 0.2 一句话论点

When learned CEM planning is slower than the command period, launch-state plans degrade under delayed activation; future-state alignment and execution-time residual reanchoring recover this stale-plan loss under an absolute-position interface. Timestamped packets, bounded feedback, reranking, and projection support deployment but are not claimed to provide equal independent tracking gains.

论文的成功条件按以下层级预先固定：

1. learned CEM solve time 明显长于 10 ms execution period；
2. Naive delayed 相对 Ideal zero-delay logical MPC 出现可测的 stale-plan degradation；
3. FullVirtual 显著优于 NaiveDelayed，并恢复一部分 zero-delay 性能；
4. ThreadedASAP 能复现 virtual fixed-delay 的主要趋势；
5. 完整方法在强 Direct-IK baseline 前不显著退化；高动态轨迹或扰动下的小幅、可重复收益属于增强证据，而不是唯一成功条件。

## 0.3 收紧后的贡献

1. **Future activation alignment。** 为历史条件 GRU residual CEM 预测 activation state/history，使慢规划器不再从过期 launch state 开始优化。
2. **Execution-time re-anchor。** 将带时间戳 residual packet 重新锚定到当前 IK nominal，避免绝对位置接口重放过期 absolute command。
3. **Matched virtual/threaded evidence。** 在冻结模型和 CEM budget 下量化 stale-plan degradation、两项核心机制、真实线程复现性和 soft-real-time 边界。

有界 fast feedback 仅称为辅助状态修正；task-space reranking 明确写成准确率—延迟权衡；two-stage projection 写成 computation–consistency trade-off。三者不再与上述两项核心机制并列。

不把 CEM、asynchronous MPC 的总体思想、控制器数量或 Model C 数据聚合作为独立贡献。ASAP-MPC、advanced-step MPC 和 asynchronous MPC 是本文建立其上的相关工作。

## 0.4 Model C 的处理

Model C 从标题、摘要、贡献、方法、结果表和正文图中删除。仅在 Discussion 末尾保留一条可选负结果占位句：

> We additionally evaluated MPC-induced data aggregation under the frozen threaded controller. It did not yield a consistent improvement over the offline Model A across the common benchmark, suggesting that deployment latency and plan alignment, rather than offline data coverage alone, were the dominant bottlenecks in the present setting.

只有在完整统一 benchmark 有数字支持时才保留该句，否则投稿稿中完全删除。

---

# 1. 六页结构与叙事

| Section | 内容 | 建议篇幅 |
|---|---|---:|
| I. Introduction and Related Work | 问题、相关工作、贡献 | 0.9 页 |
| II. Problem and Learned Residual Planner | 状态/输入、GRU、IK nominal、residual CEM、cost | 1.1 页 |
| III. Delay-Aware Asynchronous Execution | future anchor、packet、reanchoring、feedback、fallback | 1.0 页 |
| IV. Experimental Protocol | 轨迹、控制器、seed、timing 和统计定义 | 0.7 页 |
| V. Results | 主比较、软实时、消融、delay sweep | 1.4 页 |
| VI. Discussion and Conclusion | 限制、边界、结论 | 0.4 页 |
| References | 精选且已核验的引用 | 约 0.5 页 |

建议正文使用 3--4 张图和 2 张表：系统图、tracking/control-quality 主表、deployment/timing 表、代表性轨迹/时序图、机制消融图和 delay sweep。若 6 页过满，优先压缩背景和公式说明，不删主比较、时序定义或 latency sweep。

论文叙事顺序：

1. 位置执行器只接收绝对关节位置参考，高层控制器面对的是包含低层伺服行为的闭环对象。
2. GRU 可以学习该闭环转移，residual CEM 可以围绕强 IK nominal 提前修正跟踪误差。
3. CEM 求解明显慢于 10 ms 命令周期；从 launch state 得到的计划在生效时可能已经过时。
4. future-state anchoring 解决“从哪个状态规划”，timestamped residual packet 解决“计划何时生效”，100 Hz reanchoring 和反馈解决“当前执行什么”。
5. 主证据是 FullVirtual 与 NaiveDelayed 的 paired difference；Direct IK 是关键实用 secondary baseline，Ideal zero-delay logical MPC 是不可部署的逻辑性能上界。

---

# 2. Abstract 框架

摘要保持 170--210 词，按背景、缺口、方法、实验、结论组织。正式结果完成前不写承诺式结论。

> Position-controlled manipulators expose absolute joint-reference commands rather than torque inputs, making their effective response depend on low-level servo behavior. Learned model predictive control can capture this closed-loop response, but sampling-based planning may take substantially longer than the command period, so a plan optimized from the launch state can be stale when activated. We study a delay-aware asynchronous learned residual CEM-MPC architecture for this setting. A history-conditioned GRU predicts transitions under actuator-reference commands, while CEM searches bounded corrections around a continuous-IK nominal trajectory. The planner forecasts the state at the intended activation time and publishes a timestamped residual packet; a 100 Hz execution layer retrieves the age-aligned residual, reconstructs the current nominal, applies bounded state feedback, and falls back to Projected Direct IK when no valid packet is available. Experiments on an ABB IRB 2400 MuJoCo model compare raw, projected, and preview IK baselines with Ideal zero-delay logical MPC, naive delayed, virtual delay-aware, and real threaded execution over four trajectories and paired CEM seeds. **[RESULT: stale-plan degradation and recovery, paired tracking uncertainty, Direct-IK non-inferiority or improvement, smoothness/torque trade-off, CEM and end-to-end latency, planner rate, packet/fallback behavior, and control deadline misses.]**

Keywords: learned dynamics, model predictive control, cross-entropy method, asynchronous MPC, residual control, robot manipulators.

---

# 3. I. Introduction and Related Work

## Paragraph 1: position-controlled manipulation

说明工业位置接口下，高层输入是绝对关节位置参考，而不是力矩。实际响应由低层 PD/servo、离散执行、饱和、重力补偿和机器人动力学共同决定。Direct IK 简单可靠，但动态滞后会造成任务空间跟踪误差。

## Paragraph 2: learned residual MPC

说明学习闭环 actuator-reference dynamics 的机会：

\[
x_t=[q_t,\dot q_t],\qquad u_t=q_t^{\mathrm{ref}},
\]

\[
(x_{t-L+1:t},u_{t-L+1:t})\mapsto x_{t+1}.
\]

CEM 围绕 IK nominal 优化 residual，而不是完全替代 nominal controller。零 residual 是显式候选，防止模型利用导致的无意义偏移。

## Paragraph 3: stale-plan gap

指出 learned CEM solve time 超过 10 ms 控制周期时，问题不只是“更新频率低”：计划绑定的 state、reference 和 history 在激活时都可能已经过时。缓存旧 absolute command 无法自动适配当前 nominal。

## Paragraph 4: related work boundary

用一段紧凑文字覆盖 learned dynamics/neural MPC、sampling/residual control、advanced-step/asynchronous MPC，并明确以下边界：

- ROBIO 2024 data-driven robust MPC 强调 constrained pose tracking、模型不确定性和鲁棒性；本文差异在闭环 actuator-reference dynamics、sampling residual CEM 和 activation alignment。
- ICRA 2024 *Smooth Computation without Input Delay* 用 successor-state prediction 和 tube-based model MPC 提前求解；本文使用 history-conditioned learned dynamics，并额外处理 absolute-reference nominal reconstruction、packet age 和 projection mismatch。
- ASAP-MPC 已提出异步完整优化、快速反馈和轨迹衔接；本文只声称将 future-anchor asynchronous execution 适配到 learned residual CEM 的绝对位置参考接口。
- Data-Driven Incremental MPC 使用 TDE、在线 RLS 和真实 3-DoF 机械臂；本文没有在线模型更新或实机验证，必须诚实限定为 MuJoCo study。
- ROBIO 2025 S2S-GACTRNN 表明 one-step loss 不足以支撑 recurrent model，需要报告多步误差、逐关节指标和 divergence。
- constrained sampling-based MPC 与 Feedback-MPPI 强调硬约束/投影和快速局部反馈，因此本文必须量化 projection、candidate selection 和 fast feedback 实际承担的工作。

不声称发明 asynchronous MPC、future anchoring、CEM、GRU dynamics、residual control 或 sampling planner 加 fast feedback。

## Paragraph 5: method and contributions

用 future activation state、timestamped packet、100 Hz nominal reconstruction、bounded feedback 和 Direct-IK fallback 串起方法，然后列出第 0.3 节三项贡献。

---

# 4. II. Problem and Learned Residual Planner

## A. Plant, interface, and time scales

六自由度 ABB IRB 2400 MuJoCo 模型：

\[
x_t=\begin{bmatrix}q_t\\\dot q_t\end{bmatrix}\in\mathbb R^{12},
\qquad
u_t=q_t^{\mathrm{ref}}\in\mathbb R^6.
\]

MuJoCo physics step 为 2 ms，`frame_skip=5`，因此控制周期为

\[
\Delta t=10\ \mathrm{ms}.
\]

论文配置固定使用 GRU；投稿前填入唯一 checkpoint、history length 16、hidden size 256、训练数据、normalizer、CEM budget、硬件和 commit hash。根目录历史默认值不作为论文配置来源。

## B. Learned closed-loop dynamics

历史 token 与预测目标：

\[
z_i=[q_i^\top,\dot q_i^\top,(q_i^{\mathrm{ref}})^\top]^\top,
\]

\[
\widehat{\Delta\dot q}_{t+1}=f_\theta(z_{t-L+1:t}),
\quad
\hat{\dot q}_{t+1}=\dot q_t+\widehat{\Delta\dot q}_{t+1},
\quad
\hat q_{t+1}=q_t+\hat{\dot q}_{t+1}\Delta t.
\]

训练和在线 rollout 都使用对齐的 `[x_t,u_t]` token。模型损失简写为单步损失与多步 rollout loss 的组合；详细权重放入实验配置表或补充材料。

## C. IK nominal and planner-requested command

连续 bounded damped-least-squares IK 产生未来关节参考 `q^IK`。为准确匹配 threaded 实现，不能把它描述成 planner 侧完成速度/加速度投影的 fully executable nominal。

定义关节范围裁剪后的 nominal：

\[
\bar q^{\mathrm{IK}}_k=\operatorname{clip}_{\mathcal Q}
\left(q^{\mathrm{IK}}_{t_a+k+1}\right),
\]

其中 `t_a` 是计划激活步。CEM 搜索

\[
\rho_k\in[-1,1]^6,
\qquad
r_k=\rho_k\odot r_{\max},
\]

planner rollout 中的 requested command 为

\[
\tilde q^{\mathrm{ref}}_k=
\operatorname{clip}_{\mathcal Q}
(\bar q^{\mathrm{IK}}_k+r_k).
\]

最终论文配置采用 compiled two-stage projection：第一阶段用廉价近似筛选全部 CEM population，第二阶段对 final elites、mean、best 和 baseline 执行精确 braking-aware projection 并重新评估。100 Hz 执行层仍会对每条下发命令执行同一 physical projection。

## D. Residual CEM and cost

每轮 CEM 强制包含 zero-residual candidate 和 current mean，并在最终比较 baseline、best sample 和 final mean。零 residual 表示 planner 不请求 learned correction；执行层将其作为经过共享 physical projection 的 Direct-IK fallback。

近似 population evaluation 的 joint-space cost 必须与代码一致。令归一化 discount

\[
\alpha_k=\frac{\gamma^k}{\sum_{j=0}^{H-1}\gamma^j},
\]

则论文配置使用

\[
\begin{aligned}
J_{\mathrm{approx}}={}&
\sum_{k=0}^{H-1}\alpha_k\bigl(
w_q\ell_q(\hat q_{k+1},q^d_k)
+w_{\dot q}\ell_{\dot q}(\hat{\dot q}_{k+1},\dot q^d_k)
+w_r\ell_r(r_k)\\
&+w_s\ell_s(\tilde q_k^{\mathrm{ref}},\hat q_k)
+w_{\dot r}\ell_{\dot r}((r_k-r_{k-1})/\Delta t)
+w_{\ddot r}\ell_{\ddot r}((\dot r_k-\dot r_{k-1})/\Delta t)
\bigr)\\
&+w_{\mathrm{first}}\ell_{\mathrm{first}}
+w_{\mathcal Q}B_{\mathcal Q}
+w_{\dot{\mathcal Q}}B_{\dot{\mathcal Q}} .
\end{aligned}
\]

其中 servo proxy 比较动作前状态 \(\hat q_k\)，residual 速度和加速度使用真实 control \(\Delta t\)。每个 barrier 都是 discounted weighted mean 加 horizon/joint 最大值项；若任一预测 \(q\) 硬越过 joint bounds，该 candidate cost 直接设为 \(+\infty\)。

通用实现保留 terminal、torque 和 torque-slew 项，但冻结的 black-box paper profile 中 torque 项不启用，且当前 `w_terminal=0`，因此不把 terminal 写成论文配置的有效主项。若最终 manifest 改变这些权重，公式必须重新与 manifest 核对。

two-stage exact final pool 还加入基于 MuJoCo forward kinematics 的 TCP position 与 orientation cost：

\[
J_{\mathrm{exact}}=J_{\mathrm{approx}}
+w_p\ell_p^{\mathrm{TCP}}
+w_R\ell_R^{\mathrm{TCP}} .
\]

权重、尺度、population、elite、iteration、horizon、residual limits、discount 和 barrier max weight 放入冻结配置表，不在结果完成前填入测量数据。

## E. Two-stage planner projection and exact selection

最终论文配置采用以下明确流程：

1. 全 CEM population 使用廉价 requested-command approximation 和 learned rollout 排序；
2. exact validation pool 由 final elites、跨迭代 best、final mean 和 zero-residual baseline 组成，并对重复序列去重；
3. 每个候选经过 compiled braking-aware exact projection，再次 learned rollout，并加入 exact TCP pose cost；
4. 选择最低 exact cost；完全同 cost 时依次优先 baseline、mean、best、elite；
5. 记录 valid candidate fraction、exact selection changed 和 selection mode；
6. 100 Hz execution layer 对最终下发命令仍使用同一个 shared physical projector。

因此 planner cost、planner-projected sequence 与 actual execution command 不是默认等价关系，必须通过 projection 和 planner/execution discrepancy 指标验证。

---

# 5. III. Delay-Aware Asynchronous Execution

## A. Future-state anchoring

时刻 `t_l` 发布 planning snapshot 后，执行层继续使用 active packet 或 Direct IK。planner 用预期命令向前预测 `D` 步：

\[
\hat x_{t_a}=\hat F^{(D)}
\left(x_{t_l},u^{\mathrm{active}}_{t_l:t_a-1}\right),
\qquad t_a=t_l+D.
\]

随后从 `t_a` 对应的未来 reference window 构造 residual planning problem。求解完成后发布包含 launch time、activation time、residual sequence 和 predicted state sequence 的 packet；超过 guard deadline 的 packet 丢弃。

## B. Timestamped residual packet and packet age

在 fast tick `t` 激活 due packet，并计算

\[
a=t-t_a.
\]

若 `0\le a<H`，读取 `r_a` 和 `\hat x_a`；否则 packet 无效并回退 Direct IK。这里的 residual 相对规划时的 IK nominal 定义，但执行时重新锚定到当前 `\bar q_t^{IK}`，而不是重放旧 absolute command。

## C. Bounded fast feedback

\[
\delta r_t=\operatorname{clip}
\left(K_q(\hat q_a-q_t)+K_{\dot q}(\hat{\dot q}_a-\dot q_t),
-\delta r_{\max},\delta r_{\max}\right).
\]

requested correction 为

\[
c_t=\operatorname{clip}(r_a+\delta r_t,
-r_{\max}-\delta r_{\max},r_{\max}+\delta r_{\max}).
\]

## D. Execution projection and projected Direct-IK fallback

执行层对所有 requested correction 使用同一投影：

\[
q_t^{\mathrm{cmd}}=
\Pi_{v,a,\mathcal Q}
(\bar q_t^{\mathrm{IK}}+c_t;
q_{t-1}^{\mathrm{cmd}},\dot q_{t-1}^{\mathrm{cmd}}).
\]

零 correction 请求 projected Direct-IK nominal，不绕过速度、加速度、braking 或关节范围投影；非零 correction 使用完全相同的物理约束。

执行后的 residual 定义为

\[
r_t^{\mathrm{exec}}=q_t^{\mathrm{cmd}}-\bar q_t^{\mathrm{IK}}.
\]

因此论文必须报告：

- 非零 correction / execution projection 激活比例；
- requested correction 与 `r_exec` 的 RMS/P95 差值；
- residual-bound 和 feedback-bound saturation ratio；
- command velocity/acceleration violation count。

这些指标说明 planner 请求的命令与机器人实际收到的命令是否存在系统性偏差。

## E. Virtual and threaded modes

- **Virtual delay-aware MPC**：按固定 `replan_interval_steps` 和逻辑 delay 运行的确定性 emulator，用于验证算法逻辑和 delay sensitivity。
- **Threaded asynchronous MPC**：控制线程每 10 ms tick 运行，CUDA worker 完成一次 solve 后获取最新 snapshot 并继续求解；实际更新率由端到端运行时间决定。

不得把 virtual 模式描述为真实后台并行控制器。

---

# 6. IV. Experimental Protocol

## A. Frozen configuration

使用独立 calibration trajectory 选择一次 GRU checkpoint、normalizer、history length、horizon、CEM samples/iterations、constraints、cost weights、anticipation delay 和 feedback bounds，然后冻结。最终论文给出唯一配置标识、仓库 commit hash、硬件和软件版本。

## B. Four scenarios

1. **Circle**：平滑、近恒定曲率；
2. **Figure-eight**：曲率变化和方向切换；
3. **Fast ellipse**：提高速度，放大 actuator lag 与 stale-plan mismatch；
4. **Rounded square**：圆角或 minimum-jerk 平滑后的加速度压力测试。

禁止使用参考不连续的 raw square，否则拐角误差无法与时延失效区分。

## C. Main controller set

| Method | Scientific role |
|---|---|
| Raw Direct IK | 未经共享速度/加速度投影的原始 nominal reference；用于分离 IK 本身与投影收益 |
| Projected Direct IK | 强实用 nominal baseline；经过与执行层相同的 physical projector，无 learned model、无 CEM |
| Preview IK | 在独立 calibration reference 上只选择一次 preview，并在所有测试轨迹冻结；用于排除纯相位提前解释 |
| Ideal zero-delay logical MPC | 暂停逻辑仿真时间并立即应用结果，表示 residual MPC 的逻辑上限，不表示实时可行性 |
| Naive delayed/cached MPC | 从 launch state 规划，延迟后直接执行旧计划，暴露 stale-plan failure |
| Virtual delay-aware MPC | 固定逻辑 delay 下验证 future anchoring 和 residual reanchoring |
| Threaded asynchronous MPC | 真实墙钟双线程最终方法与软实时测量 |

论文 primary endpoint 预先固定为：

> full-trajectory TCP position RMSE 上 `FullVirtual − NaiveDelayed` 的 trajectory-seed paired difference。

Projected Direct IK 比较是关键 secondary endpoint；Raw Direct IK 和 Preview IK 用于解释 baseline 与纯相位补偿。Full threaded 相对 Full virtual 的差异验证真实 threaded 实现是否复现 fixed-delay 结论。三个 component ablation 差值是 matched mechanism-isolation comparisons，不解释为可相加的独立贡献。

## D. Seeds and statistics

- Raw/Projected/Preview IK 在固定 MuJoCo 初始状态下各运行一次，因为它们是确定性的；
- MPC 方法使用相同的 5 个 CEM seeds，各方法按 seed 配对；
- 每条轨迹、aggregate 和 worst seed 均报告结果；
- 主比较报告 paired difference 和 trajectory-seed case bootstrap 95% confidence interval；
- bootstrap 单位是一条 trajectory-seed case，不是每个 10 ms tick；
- 若初始状态也随机化，必须单独定义并使用跨方法配对的 initial-state seeds。

## E. Timing definitions

- **CEM solve latency**：仅 `controller.plan()` 内部时间，报告 p50/p95/p99/max；
- **planner end-to-end latency**：planning snapshot launch 到 packet publication；
- **planner update rate**：连续 solve completion 的实际频率；
- **late packet rate**：超过 activation guard deadline 而丢弃的 solve 比例；
- **packet expiration / fallback duty cycle**：packet 生命周期和无有效 packet 时使用 projected Direct IK 的比例；
- **control compute time**：单次 fast tick 的计算时间；
- **control period / wake-up jitter / deadline miss**：真实 100 Hz 调度质量。

packet 能否按时生效使用 end-to-end latency 判断，不能用 CEM solve latency 替代。

## F. Metrics

Tracking：TCP position RMSE/P95/max、orientation RMSE、joint RMSE。
Control quality：command acceleration RMS、command jerk、actuator torque RMS、residual RMS、projection error、residual/feedback saturation。
Safety：joint/velocity/acceleration violation counts。
Timing：CEM solve 和 E2E p50/p95/p99/max、planner Hz、late/expired packets、fallback duty cycle、control-period p95/p99、wake-up jitter、deadline misses 和 worst seed。

定义 latency degradation recovery：

\[
R_{\mathrm{latency}}=
\frac{E_{\mathrm{naive}}-E_{\mathrm{ASAP}}}
{E_{\mathrm{naive}}-E_{\mathrm{zero-delay}}}.
\]

只有分母显著为正时才报告该比例，并同时给出原始误差和置信区间。

---

# 7. V. Results

## A. Main closed-loop comparison

主结果拆为两张表，避免 IEEE 双栏中过宽：

- **Table I: Tracking and control quality**：TCP RMSE/P95/max、orientation RMSE、joint RMSE、command acceleration RMS、torque RMS。
- **Table II: Deployment and timing**：solve p50/p95/p99/max、E2E p50/p95/p99/max、planner Hz、late packet、packet expiration、fallback duty cycle、control-period p95/p99、wake-up jitter、deadline miss、projection activation 和 projection-offset p95。

结果依次回答：

1. residual MPC 在理想零延迟下是否有价值；
2. naive delayed execution 造成多大 stale-plan degradation；
3. delay-aware virtual/threaded 方法恢复了多少 degradation；
4. threaded 方法是否保持 100 Hz soft-real-time execution；
5. 跟踪收益是否以激进命令或 torque 为代价。

## B. Timing and packet behavior

用 timeline 区分 snapshot、future forecast、CEM solve、publication、activation、drop 与 fast ticks。表或分布图分别报告 CEM solve 和 planner end-to-end latency，不使用含糊的 “Plan p50/p95”。

## C. Controlled mechanism-isolation ablations

正文用图或窄表保留四行：

1. FullVirtual；
2. No future-state anchoring；
3. No nominal/residual reanchoring，执行旧 absolute command；
4. No fast feedback。

每个变体与同 trajectory、同 CEM seed 的 FullVirtual 配对。20 条 FullVirtual 在冻结 commit 下重跑，与其余 60 条构成统一 80-case matrix。NoFeedback 的 CI 跨 0，只能支持辅助机制定位。Legacy shifted history 作为实现验证，不列为核心方法消融；三个 `NoX − FullVirtual` 差值不能解释为相互独立或可相加的贡献。

## D. Planner-delay sweep

在 virtual 模式固定其他配置，测试

\[
D\in\{2,4,6,8\}\ \text{steps}=\{20,40,60,80\}\ \mathrm{ms}.
\]

绘制 TCP RMSE 随 delay 的变化，至少包含：

- naive delayed；
- future anchor only；
- anchor + nominal/residual reanchoring；
- full method（再加 fast feedback）。

Figure 4 固定用于该 delay sweep，不再用于 Model C。

结果不应描述为组件逐级单调改善：Anchor-only 在 $D\geq6$ 崩坏，而加入 re-anchor 后恢复稳定。这说明 re-anchor 是 future alignment 在绝对位置接口下工作的必要配套。

---

# 8. VI. Discussion and Conclusion

Discussion 必须明确：

- 贡献是 learned residual CEM 的 activation-time alignment 与多速率部署，而不是 asynchronous MPC 本身；
- planner requested command 与 execution-projected command 存在边界，相关 projection/saturation 指标决定该近似是否可接受；
- 结果仅来自 MuJoCo ABB IRB 2400，没有实机验证；
- Python threading 是 soft real time，不提供 hard-real-time guarantee；
- checkpoint、平台和 CEM budget 固定，结论不代表所有 learned model 或硬件；
- 方法没有理论稳定性保证。

可选 Model C 负结果句只在完整统一 benchmark 支持时保留。Conclusion 用约 100--130 词总结 stale-plan degradation、恢复比例和真实时序结果，不重复全部方法细节。

---

# 9. 图表与投稿前证据

## 推荐图表顺序

1. **Fig. 1**：slow planner、timestamped packet 与 100 Hz execution layer 系统图；
2. **Table I**：Tracking and control quality；
3. **Table II**：Deployment and timing；
4. **Fig. 2**：代表性轨迹与 TCP error；
5. **Fig. 3**：CEM solve、end-to-end publication、activation 和 fast-loop timing；
6. **Fig. 4 或紧凑窄表**：matched component ablations；
7. **Fig. 5**：20--80 ms latency sweep。

严格 6 页时，将配置表并入正文，并把 per-trajectory 次要指标移到补充材料。

## 必须完成

- 唯一冻结的 GRU/CEM/cost/constraint 配置和 commit hash；
- 四轨迹、5 个配对 CEM seeds、Raw/Projected/Preview IK 与四种 MPC 部署方法；
- FullVirtual vs NaiveDelayed primary comparison 及 paired bootstrap 95% CI；
- FullVirtual 与三项 matched component ablations；
- 20/40/60/80 ms latency sweep；
- one-step 与 5/10/20-step q/dq model validation、逐关节 NMSE/R²、amplitude ratio 和 divergence；
- TCP、orientation、smoothness、torque、projection、saturation 和 constraint-violation 指标；
- CEM solve 与 end-to-end latency 的 p50/p95/p99/max 独立统计；
- planner rate、late/expired packets、fallback、control jitter、deadline misses 和 worst seed；
- projection off/full/two-stage 与 oracle dynamics upper bound；
- payload、gain mismatch、noise 和 force-pulse robustness，至少进入补充材料。

## 不进入本稿主线

- Model A/B/C 独立结果章节；
- MLP/GRU/Transformer 大规模架构比较；
- legacy shifted history 作为核心贡献消融；
- 全量 cost 或 sample-number 网格；
- 未经完整实验支持的鲁棒性或 OOD 声明。

论文最终应支持而不是预设以下结论：

> Under the same 100 Hz command interface and frozen planner configuration, activation-state anchoring, residual reanchoring, and bounded fast feedback recover a measurable fraction of the degradation caused by stale delayed plans, while the threaded execution layer maintains its stated soft-real-time timing boundary.
