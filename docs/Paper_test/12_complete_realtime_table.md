# ThreadedASAP 完整实时性结果

下表把 20 条 nominal ThreadedASAP 的原始逐步数组合并后统计。百分位来自事件或 fast-loop tick 的描述性分布；论文中的 tracking 推断仍以 trajectory×seed case 为单位。

| 指标 | 结果 |
| --- | ---: |
| CEM solve p50 / p95 / p99 / max | 36.57 / 38.27 / 41.96 / 171.30 ms |
| snapshot-to-publication E2E p50 / p95 / p99 / max | 44.92 / 49.04 / 51.94 / 180.48 ms |
| planner rate | 25.30 ± 0.26 Hz |
| fast-loop compute p50 / p95 / p99 / max | 0.470 / 0.887 / 1.089 / 3.758 ms |
| control period p50 / p95 / p99 / max | 9.999 / 10.093 / 10.158 / 132.453 ms |
| wake-up lateness p50 / p95 / p99 / max | 0.160 / 0.266 / 0.339 / 122.577 ms |
| start jitter p50 / p95 / p99 / max | -0.001 / 0.093 / 0.158 / 122.453 ms |
| control deadline misses | 12 / 34,640 ticks (0.0346%) |
| late packets | 34 (0.3877% of solves) |
| expired packets | 1 |
| fallback duty | 0.3551% |
| packet age p50 / p95 / p99 / max | 2 / 4 / 4 / 19 steps |
| projection activity | 85.48% |
| safety projection offset p50 / p95 / p99 / max | 2.09 / 36.47 / 70.56 / 229.06 mrad |

## 分析

CEM solve p50 为 36.57 ms，明确慢于 10 ms command period；后台 planner 的实际更新率约 25.3 Hz。与此同时，fast-loop compute p99 仅 1.09 ms，control-period p95/p99 分别为 10.093/10.158 ms。这支持“慢规划器 + 100 Hz 软实时执行层”的系统描述。

不过 max control period 达 132.45 ms，并出现 12 次 deadline miss。因此只能称为 **Python soft-real-time**，不能称为 hard real-time 或零 deadline miss。projection activity 高达 85.48%，也说明共享执行投影是实际承担约束处理的部署组件，而不是可以忽略的实现细节。

证据文件：

- `outputs/paper_revision_v1/statistics/threaded_realtime_nominal.json`
- `outputs/paper_revision_v1/statistics/threaded_realtime_by_perturbation.json`
