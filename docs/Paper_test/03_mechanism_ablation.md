# 三项核心机制消融

矩阵为 4 trajectories × 5 seeds × 4 methods，共 80 cases；FullVirtual 20 条来自审计通过的历史兼容 cohort，其余 60 条新跑。统计单位为 trajectory × seed。

## Full 的 matched removal 差异

正值表示移除组件后更差。

| 移除项 | TCP RMSE 差值 | 95% bootstrap CI | 解释 |
| --- | ---: | ---: | --- |
| future alignment | +2.82 mm | [1.26, 4.41] mm | 四条轨迹均恶化；planner-execution q-ref RMS 增加 8.54 mrad |
| re-anchor | +654.60 mm | [526.19, 776.58] mm | 四条轨迹均灾难性恶化；absolute command stale |
| fast feedback | -0.95 mm | [-2.26, 0.29] mm | nominal pooled 平均略优于 Full，CI 跨 0 |

NoFeedback 在 circle、fast ellipse、rounded square 的平均 TCP RMSE 略低，仅 figure8 略高。因此验收规则中“三项移除后 pooled 平均方向均不优于 Full”没有全部满足。

论文措辞必须据此调整：

- future alignment 和 re-anchor 有清晰机制证据；
- re-anchor 是最强且不可缺失的组件；
- bounded fast feedback 在本 nominal 矩阵中没有显示独立 tracking 收益，只能作为低成本状态偏差修正机制或结合旧 packet/扰动结果讨论，不能继续称为已被该消融证明的核心性能贡献；
- 三个差值不可相加。
