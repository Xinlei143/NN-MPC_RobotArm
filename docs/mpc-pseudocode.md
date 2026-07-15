# 当前 residual CEM-MPC 伪代码

本页对应默认 `--mpc_policy residual`。模型 action 一直是绝对 `q_ref`；CEM action 则是相对于 nominal 的归一化 residual，而不是绝对命令增量或命令加速度。

## 每个控制周期

```text
load learned dynamics, normalizer, validated reference
reset MuJoCo and settle with q_ref = current q

for control step t:
    history = recent [q, dq, q_ref]
    future_q_des, future_dq_des = reference[t+1 : t+1+H]

    q_nom = project_nominal_q_ref_sequence(
        future_q_des,
        previous_q_ref, previous_q_ref_velocity,
        velocity_limit, acceleration_limit, joint_limits
    )

    planner = LearnedDynamicsPlanner(
        history, future_q_des, future_dq_des, q_nom,
        previous q_ref/residual history, constraints, residual cost
    )

    if recovery cooldown is active:
        q_ref_command = q_nom[0]
    else:
        result = CEM.plan()
        q_ref_command = result.q_ref

    next_state = mujoco.step(q_ref_command)
    record command, nominal, residual, cost, diagnostics
    update history and previous command/residual values
    evaluate recovery monitor
```

`q_nom` 是从 future IK `q_des` 出发、考虑上一拍真实 command velocity 的可执行序列。即使 CEM 不产生任何有效改善，zero residual 仍提供可执行 nominal baseline。

## Planner：从 CEM residual 到模型 rollout

```text
function evaluate(candidate_normalized_residual[batch, H, joints]):
    r_proposed = clamp(candidate_normalized_residual, -1, 1) * residual_max
    q_ref = project_position_command_sequence(
        q_nom + r_proposed,
        previous_q_ref, previous_q_ref_velocity,
        command velocity/acceleration limits, joint bounds
    )
    r_executed = q_ref - q_nom

    reject candidate if any |r_executed| > residual_max
    pred_states = learned_rollout(history, q_ref)
    reject candidate if predicted q violates hard joint bounds
    cost = residual_joint_space_cost(pred_states, q_ref, q_nom, r_executed)
    return cost, q_ref, pred_states, cost terms
```

投影是安全约束，不是优化目标。速度/加速度 hit limit 允许发生并记录；它们不会单独触发 recovery。

## CEM 与执行候选

```text
function CEM.plan():
    warm-start mean and std from last cycle
    if reset_std_each_step: restore std to init_std

    for iteration in 1..cem_iters:
        population = [zero residual, current mean]
        population += Gaussian(mean, std)
        population += uniform samples in [-1, 1]
        costs = planner.evaluate(population)
        update elite mean and std from finite-cost candidates

    evaluate final mean and retain the lowest-cost sampled sequence
    baseline = zero residual sequence

    execute = {
        mean: final mean,
        best: lowest-cost sampled candidate,
        lowest_cost: argmin(cost(baseline), cost(best), cost(mean))
    }
    shift selected normalized sequence to warm-start next cycle
    return first executable q_ref and diagnostics
```

`lowest_cost` 是默认选择，因为它能显式退回 direct nominal，并不会因 CEM mean 的投影或分布平均而强制执行较差补偿。

## Residual cost

令 `q_hat_minus[k]` 为第 `k` 个动作执行前的预测位置，`gamma=0.95`：

```text
J = sum_k lambda[k] * (
      tracking(q_hat[k+1], dq_hat[k+1], q_des[k], dq_des[k])
    + residual_anchor(r[k])
    + servo_proxy(q_ref[k] - q_hat_minus[k])
    + residual_velocity(dot(r)[k])
    + residual_acceleration(ddot(r)[k])
)
  + first_step_continuity
  + optional_terminal_tracking
  + joint_limit_barrier + velocity_limit_barrier
```

barrier 对时域采用 `weighted mean + max` 聚合；预测硬越界候选直接无效。默认 terminal weight 是零，避免短 horizon 的末端 learned-model 误差被过度放大。完整公式、默认权重和尺度见 [CostFunction.md](CostFunction.md)。

## Recovery

```text
if planner fails:
    immediately execute q_nom[0]
    reset CEM and start cooldown
elif not in cooldown and tracking error grows strictly for N steps
     and latest error >= ratio * first error:
    reset CEM and start cooldown
elif not in cooldown and residual is near r_max for N steps:
    reset CEM and start cooldown

while cooldown:
    execute q_nom[0]
```

默认 `N=3`、error ratio `1.25`、residual fraction `0.95`、cooldown 5 步。输出会区分 recovery trigger 次数与实际 recovery active steps。

## 输出语义

```text
actuator_q_ref       # 实际执行的 absolute command
nominal_q_ref        # projected q_nom
executed_residual    # actuator_q_ref - nominal_q_ref
delta_q_ref          # 实际相邻两拍 absolute command 之差，仅日志量
selection_mode       # baseline / best / mean / recovery_nominal / ...
baseline_cost, best_cost, mean_cost, selected_cost
recovery_*           # trigger reason 与 active flags
```

`legacy_acceleration` 的 normalized command-acceleration 双积分流程仍在代码中，仅用于历史实验复现；不要把该流程用于解释默认 residual 运行。
