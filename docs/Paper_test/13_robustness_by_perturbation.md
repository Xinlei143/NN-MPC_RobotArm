# 分扰动鲁棒性结果

下表给出 ThreadedASAP 的 TCP RMSE 相对 IK baseline 的差值；负值表示 MPC 更好。每一扰动类型合并 L3/L6，但 bootstrap cluster 保留 trajectory×condition，避免把两个 level 或控制 tick 当作独立重复。

| 条件 | Threaded − Projected IK | 95% CI | Threaded − Preview IK | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Nominal | -10.84 mm | [-11.80, -9.55] | -4.21 mm | [-5.11, -2.98] |
| Payload | -23.33 mm | [-29.94, -16.54] | -20.07 mm | [-27.67, -12.67] |
| Actuator gain | -11.13 mm | [-13.05, -9.17] | -5.40 mm | [-7.69, -3.11] |
| Force pulse | -10.14 mm | [-11.56, -8.45] | -3.75 mm | [-5.05, -2.27] |
| Observation noise | -5.68 mm | [-7.00, -4.15] | +0.95 mm | [-0.49, +2.56] |

## 分析

ThreadedASAP 相对 Projected IK 在 nominal 和四类扰动中的区间均完全低于 0，支持论文关于实际线程部署具有较强 task-space tracking 的 secondary claim。相对 Preview IK，payload、actuator-gain 和 force-pulse 条件仍有明确优势；observation-noise 条件下均值略差 0.95 mm 且区间跨 0，不能宣称一致胜出。

ThreadedASAP 与 FullVirtual 的分扰动差异进一步限定了“复现虚拟结论”的范围：nominal、payload、force pulse 和 observation noise 的 CI 均跨 0；actuator-gain 条件下 Threaded 高 **3.20 mm**，95% CI **[2.40, 4.05] mm**。因此合适的结论是总体和 nominal 趋势可复现，但真实线程在 actuator-gain 扰动下存在可测部署损失。

证据文件：

- `outputs/paper_revision_v1/statistics/ik/mpc_vs_projected_ik_by_perturbation.json`
- `outputs/paper_revision_v1/statistics/ik/mpc_vs_projected_ik_by_level.json`
- `outputs/paper_revision_v1/statistics/threaded_vs_fullvirtual_by_perturbation.json`
