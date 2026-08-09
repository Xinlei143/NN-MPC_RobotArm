# Analytical preview candidate active test (2026-08-09)

模型：u-q epoch14，`delta_state`，GRU，history 16；轨迹：`circle_p0`；`H=6`；解析候选：`preview:3`、`preview:6`。

## 结果

| 指标 | 原 Active u-q e14 | 本次 analytical preview active |
|---|---:|---:|
| 总 q RMSE | 0.9310° | **0.9020°** |
| shoulder pan RMSE | 1.3438° | **1.2662°** |
| shoulder lift RMSE | 0.8907° | **0.8024°** |
| elbow RMSE | 1.0968° | 1.1166° |
| wrist flex RMSE | 0.7193° | 0.7337° |
| wrist roll RMSE | 0.1197° | 0.1888° |

相对原 Active，总 RMSE 下降约 3.1%，但仍明显高于同一任务 Direct preview sweep 的 `preview_0=0.8292°`。

候选选择次数（1004 次 planner publication）：

| selection mode | 次数 |
|---|---:|
| best | 614 |
| mean | 274 |
| baseline | 74 |
| fixed:preview:3 | 37 |
| fixed:preview:6 | 5 |

因此解析候选只在约 4.2% 的 publication 中实际胜出；当前 cost/CEM 仍主要偏向原有 sampled `best/mean` 分支。

## residual 方向

本次 shoulder_pan planner residual：

- `corr(residual, dq_des)=-0.6831`；
- lead slope `-85.5 ms`；
- 同向率 `23.6%`，反向率 `76.4%`。

实际 transmitted qref 相对 q_des 仍为 `corr=-0.7590`、反向率 `81.1%`。方向问题有所缓解，但没有消失。

## 实时性和 executable parity

- planner failure：0；deadline miss：0；
- planning P95：22.90 ms（原 Active 约 18.26 ms）；
- CEM search P95：23.08 ms（原 Active 约 18.77 ms）；
- expected raw mismatch/live reproject：69/69。

这说明把解析候选放在每次 plan 的最终比较中有实际收益，但当前实现增加了规划开销，并使 executable raw parity 在本次运行中退化。修复 parity/延迟前，不应直接把该分支作为正式 active 配置。

完整原始结果：

- `outputs/hardware/so101_pre_mpc/20260809_diagnostic/analytical_preview_active/run_summary.json`
- `outputs/hardware/so101_pre_mpc/20260809_diagnostic/analytical_preview_active/planner_events.jsonl`
- `residual_alignment_analytical_preview_active/alignment.json`
