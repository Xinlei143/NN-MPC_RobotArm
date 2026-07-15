# Residual CEM-MPC cost function

本文件描述当前默认 `--mpc_policy residual --cost_profile blackbox` 的规范。`legacy_acceleration` 是旧实验复现模式，不应将其动作参数化或 cost 当作默认实现。

## Command parameterization

对控制时刻 `t` 和 horizon `H`，先将 future IK reference 投影为可执行 nominal：

```text
q_nom[k] = project(q_des[t+k+1])
q_ref[k] = project(q_nom[k] + r[k])
```

`r[k]` 是 CEM 直接搜索的 normalized residual 乘以每关节 `r_max`。投影执行命令速度、命令加速度、关节边界和安全 margin；若投影后的 residual 超出 `r_max`，候选判为不可行。零 residual 是必有的 baseline candidate，因此 direct nominal 始终可参与比较。

预测状态 `x_hat[k+1]=[q_hat[k+1], dq_hat[k+1]]` 由 `q_ref[k]` 产生。servo 项使用动作执行**前**的状态 `q_hat_minus[k]`：第 0 拍为真实 `q[t]`，其余拍为 `q_hat[k]`。

## Black-box cost

令时间权重为 `lambda_k = gamma^k / sum(gamma^i)`，默认 `gamma=0.95`。默认总代价是：

```text
J = sum_k lambda_k * [
      w_q       * ||(q_hat[k+1]  - q_des[k])  / s_q||^2
    + w_dq      * ||(dq_hat[k+1] - dq_des[k]) / s_dq||^2
    + w_r       * ||r[k] / s_r||^2
    + w_servo   * ||(q_ref[k] - q_hat_minus[k]) / s_servo||^2
    + w_dr      * ||dot(r)[k] / s_dr||^2
    + w_ddr     * ||ddot(r)[k] / s_ddr||^2
]
  + w_first * C_first
  + w_terminal * C_terminal
  + w_joint_limit * C_q_limit
  + w_dq_limit * C_dq_limit
```

- `C_q`、`C_dq`：预测关节位置/速度跟踪；默认速度模式为 `track`。
- `C_r`：reference anchor，限制补偿距离，而非只限制漂移速度。
- `C_servo`：不读取 `Kp/Kd` 的 actuator-effort proxy，比较执行前状态与 actuator target。
- `C_dr`、`C_ddr`：只平滑 residual；正常的 `q_des` 运动不因本项受罚。第一拍连接真实上一周期的 residual 与 residual velocity。
- `C_first`：第一拍 residual continuity cost。
- `C_terminal`：可选 terminal position tracking；稳定性默认 `w_terminal=0`。

尺度在 episode 开始时固定：`s_q=clip(0.1*(P95-P5), 0.04, 0.08)`，`s_dq=max(P99(|dq_des|), 0.25)`，`s_r=0.5*r_max`。`s_servo` 使用 CLI 的 per-joint servo scale。

## 默认权重和约束

```text
w_q=1.0                 w_dq=0.10
w_residual=0.20         w_servo=0.05
w_residual_velocity=0.05  w_residual_acceleration=0.02
w_first=0.20            w_terminal=0.0
w_joint_limit=10.0      w_dq_limit=5.0
```

默认 residual bound 是 `[0.12, 0.10, 0.12, 0.15, 0.15, 0.20] rad`；servo scale 是 `[0.08, 0.07, 0.08, 0.04, 0.025, 0.05] rad`。

命令 velocity/acceleration limit 可用 `auto` 按 reference 校准，或显式传入。其 upper cap 默认是速度 `[1,1,1,2,2,2.5] rad/s`、加速度 `[5,5,5,10,10,12.5] rad/s^2`。这些是 MuJoCo 的保守**规划上限**，不是 ABB 认证硬件额定值。

预测位置越过真实 joint hard limit 的候选直接无效；进入 safety margin 时使用 softplus barrier。barrier 以 time-weighted mean 加 `barrier_max_weight * max` 聚合，避免单个危险关节/时刻被 horizon 平均稀释。速度 barrier 使用相同聚合方式。

## CEM selection, recovery, and profiles

Residual 模式每拍固定加入 zero-residual baseline 与当前 mean candidate，并按 `uniform_sample_ratio` 混入全局 uniform candidates。`--cem_execute` 的语义：

- `mean`：执行最终 CEM mean；
- `best`：执行本轮最低 cost sample；
- `lowest_cost`：重新比较 baseline、best、mean，执行最低 cost 者。默认并推荐。

Recovery 只针对 planner failure、持续误差恶化和持续 residual saturation。命令速度/加速度接近上限是正常的规划诊断，不是 recovery 条件。

`--cost_profile actuator_aware` 在上述 black-box cost 上增加基于 MuJoCo position actuator `Kp/Kd` 的 torque 与 torque-slew 项；它是可选增强，不是当前主方法。`legacy_acceleration` 保留 absolute-command velocity/acceleration smoothing weight，用于复现旧实验。

## Diagnostics

每次运行保存 cost terms、`nominal_q_ref`、`executed_residual`、baseline/best/mean/selected cost、selection mode、命令速度/加速度、预测 replay 误差和 recovery arrays。`run_summary.json` 与控制台报告同时给出 `recovery_trigger_count`、各原因次数和 `recovery_active_step_count`。
