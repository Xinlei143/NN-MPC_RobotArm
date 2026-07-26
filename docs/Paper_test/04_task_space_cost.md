# Exact final-pool task-space cost

完成 24 cases：circle/figure8 × 3 seeds × fixed-D6/deployed × joint-only/task-space。

四组 500-plan 标定均得到 D=6。Joint-only TwoStage P95 为 46.24 ms，Task-space TwoStage P95 为 52.28 ms。

## 配对结果

| 对比 | TCP RMSE 差值（Task − Joint） | 95% CI | 结论 |
| --- | ---: | ---: | --- |
| fixed D6 virtual | -2.20 mm | [-4.40, 0.24] mm | 平均改善，CI 跨 0 |
| calibrated-D threaded | -3.05 mm | [-4.32, -0.96] mm | 部署条件下稳定改善 |

Task-space deployed 的 TCP P95 平均改善 13.65 mm，orientation RMSE 改善 0.00363 rad；exact selection changed rate 增加约 19.6 个百分点。代价是 solve P95 增加 7.04 ms、E2E P95 增加 6.82 ms、exact-pool P95 增加 6.65 ms。

因此应写成 accuracy–latency trade-off：reranking 改善最终候选选择和 TCP 指标，但增加约 6–7 ms 尾延迟。不能把 fixed-D 结果单独解释为无代价的部署全面提升。
