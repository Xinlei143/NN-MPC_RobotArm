# Projection choice

完成 36 cases，并完成 ProjectionOff、FullCompiled、TwoStageCompiled 各 500-plan 延迟标定。common D 固定为 6，部署组使用各自 calibrated D（本机均为 6）。

## 平均结果

| variant | common-D TCP RMSE | deployed TCP RMSE | deployed solve P95 | deployed E2E P95 |
| --- | ---: | ---: | ---: | ---: |
| ProjectionOff | 39.67 mm | 36.78 mm | 38.84 ms | 49.54 ms |
| FullCompiled | 34.64 mm | 35.15 mm | 39.61 ms | 50.61 ms |
| TwoStageCompiled | 31.21 mm | 31.89 mm | 33.17 ms | 44.27 ms |

TwoStage 相对 FullCompiled：

- common-D 平均相对 TCP 差为 -9.91%，单侧 95% 上界 -9.10%；
- deployed 平均相对 TCP 差为 -9.26%，单侧 95% 上界 -8.24%；
- 两组都通过预注册的 +5% 非劣界；
- deployed solve/E2E P95 分别降低约 6.45/6.34 ms；
- 所有组 velocity/acceleration violation 均为 0。

相对 ProjectionOff，TwoStage 明显降低 planner-execution mismatch；相对 FullCompiled，TwoStage 更快且 tracking 未退化。因此本矩阵支持“best computation-consistency trade-off”，也满足预定义的 two-stage 选择标准。
