# SO101 executable-command semantics (2026-08-09)

本次 P0 修复把 planner、ASAP delay forecast 和 SO101 backend 收敛到同一个显式状态机：

```text
requested q_ref
  -> joint / measured-relative limits
  -> velocity limit
  -> acceleration limit
  -> joint-limit braking envelope
  -> nearest encoder-count quantisation
  -> transmitted q_ref and next command velocity
```

状态保存的是上一拍真正 transmitted 的五关节 q_ref 和由 transmitted q_ref 计算的 command velocity。Torch CEM、NumPy ASAP forecast 和 backend 都调用 `robot_runtime/executable_command.py`。planner packet 现在同时保存：

- `requested_q_ref_sequence`：CEM 实际评分的 pre-projection request；
- `q_ref_sequence`：canonical state machine 的 transmitted q_ref；
- `expected_raw_sequence`：planner 预计发送的五个 Goal_Position raw count。

真机 backend 每拍复算 expected raw。planner 的 expected raw 是基于延迟 forecast state 得到的，而真正执行时必须使用 live measured state；两者出现一两个 encoder count 的差异并不代表安全违规。因此默认策略是在 live state 上再次调用同一个 canonical state machine，保留 `planner_raw_mismatch_live_reproject` 诊断并继续执行 MPC request，而不是把每个 stale raw 都降级成 Direct nominal。只有显式设置 `strict_expected_raw=True` 的审计调用才会触发 `planner_raw_mismatch_direct_fallback`。

## zero-residual invariant

`mpc_preview_nominal_steps=0` 现在表示 `tick t -> reference[t]`，和 `JointFilePlayer` 的 Direct IK 同 tick。需要提前一拍时显式设置 `--mpc_preview_nominal_steps 1`。tracking target 仍然是 `reference[t+1]`。

## 离线 replay

旧的 v3 log 没有 raw Goal_Position，因此只能从浮点 `actuator_q_ref` 反推 raw；它不是 raw-count exact parity。运行：

```bash
conda run --no-capture-output -n lerobot python scripts/replay_so101_executable_commands.py \
  --rollout outputs/hardware/so101_pre_mpc/20260809_epoch14/active_projected_circle_v3/rollout.npz \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --output-dir docs/hardware/so101-executable-semantics-20260809/v3_replay
```

该 v3 replay 的主要结果：request 到 canonical transmitted 的 P95 约 `0.07–0.09°`；canonical 与旧日志 transmitted 的 P95 约 `0.26–1.01°`，最大约 `2.37°`，说明旧链路确实存在 projector/state mismatch。下一次真机日志会保存 `transmitted_goal_position_raw` 和 planner expected raw，可以做真正的 raw-count parity。

## κ / sensitivity

旧的整体 L2 `kappa_h6` probe 不再作为 checkpoint 选择依据。新脚本：

- 使用训练等价 `[x_t,u_t]` history；
- 使用 `actuator_q_ref`（没有该字段时才退回 transmitted/actions）；
- 每个 joint 做 ±6 encoder-count 的 impulse 和 held perturbation；
- 输出 H=1…12 的 signed 5×5 sensitivity matrix；
- 从 extension 数据的 mode=7 fast step/hold 提取 empirical response；
- 先要求 H=12 对角符号与 empirical response 一致，再以 held-out rollout RMSE 选模型。

示例：

```bash
conda run --no-capture-output -n lerobot python dynamics_modeling/scripts/evaluate_so101_sensitivity.py \
  --dataset outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz \
  --rollout outputs/hardware/so101_pre_mpc/20260809_epoch14/active_projected_circle_v3/rollout.npz \
  --model u_minus_q_epoch14 \
    dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt \
    dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/normalizer.pt \
  --output-dir docs/hardware/so101-executable-semantics-20260809/sensitivity_e14
```

本次 probe 找到 230 个 fast step/hold events；u-q e14 在 H=12 的五个对角符号全部一致，held-out H=12 q RMSE 为约 `0.00378 rad`。三模型比较中，u-q e45 的 RMSE 更低，但它的 `hardware_config_sha256` 与当前真机配置不一致，因此被部署门禁排除；当前可部署选择仍是 `u_minus_q_e14`。结果保存在 `sensitivity_all/sensitivity.json` 和 `sensitivity_all/checkpoint_selection.json`，e14 单模型结果保存在 `sensitivity_e14/`。

## 自动化测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot python -m pytest -q \
  mpc/tests/test_executable_command.py \
  mpc/tests/test_zero_residual_invariant.py \
  mpc/tests/test_preview_nominal.py \
  tests/hardware/test_so101_backend.py
```

测试覆盖 Torch/NumPy raw-count parity、quantised velocity state、Direct 与 zero-residual MPC 的 requested/transmitted 一致性。

## predictive headroom / candidate ranking

Direct preview/lead sweep、Active residual 方向分析和 candidate-ranking benchmark 记录在
[`headroom_diagnostics.md`](headroom_diagnostics.md)。同一 frozen reference 的实机 preview sweep 已完成：
`preview_0=0.8292°`，`preview_6=0.5872°`，相对改善 `29.19%`，证明当前任务存在明确的 predictive headroom。现有 Active 日志仍显示
shoulder_pan residual 与 `dq_des` 反相关（`corr=-0.811`，同向率约 `15%`）；u-q epoch14 在 7 个 preview 候选、6 个 anchor 上的 candidate-ranking 为 Spearman `0.7136`、pairwise `0.8145`、top-1 `0.6667`。因此后续应优先修正 Active 的候选排序/lead 方向，并把 κ 降为辅助诊断，而不是继续以 scalar κ 单独选择 checkpoint。

本轮新增两个默认关闭的诊断开关：

```text
--analytical_preview_steps 3,6
--directional_residual_gate same_as_dq_des \
--directional_residual_gate_joints shoulder_pan
```

前者把解析 preview 分支加入 CEM 的最终候选比较，后者只移除指定关节中与 `dq_des` 反向的 residual。两者都不改变默认配置，先用于 shadow/paired ablation。
