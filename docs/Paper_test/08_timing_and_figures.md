# 实时性、代表性图与 planner timeline

汇总表已扩展 solve/E2E max、control compute/period/wakeup/start jitter P50/P95/max、packet age、active packet gap、fallback duty、projection/safety offset、planner-execution error 和 worst seed。

四组正式标定结果：

| variant | P50 | P95 | P99 | max | D |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint-only TwoStage | 40.93 ms | 46.24 ms | 48.27 ms | 49.59 ms | 6 |
| Task-space TwoStage | 47.50 ms | 52.28 ms | 54.14 ms | 68.29 ms | 6 |
| ProjectionOff | 45.54 ms | 49.90 ms | 51.33 ms | 56.66 ms | 6 |
| FullCompiled | 46.45 ms | 51.21 ms | 58.50 ms | 68.03 ms | 6 |

代表性 case 按预注册规则选择：nominal circle 的 5 个 ThreadedASAP seed 中 TCP RMSE 最接近中位数者，最终为 seed 3（28.28 mm）。同一 trajectory/seed 用于 Projected Direct IK、NaiveDelayed、FullVirtual 和 ThreadedASAP。

生成文件：

- `representative_tracking.pdf`
- `representative_errors.pdf`
- `representative_control.pdf`
- `planner_timeline.pdf`
- `delay_sweep.pdf`
- `projection_tradeoff.pdf`
- `representative_case_manifest.json`

图均位于 `outputs/paper_delay_aware_two_stage_v2/figures/`。Timeline 由 planner events 与 rollout activation 数组重建，展示 snapshot、publication、scheduled/actual activation、late/expired packet 和 active packet age。
