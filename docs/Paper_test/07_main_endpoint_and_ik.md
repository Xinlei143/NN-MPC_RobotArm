# 主终点与 IK 配对分析

## 主终点

20 个 trajectory × seed 配对中：

```text
FullVirtual − NaiveDelayed TCP RMSE
= -39.55 mm
95% CI = [-42.10, -36.87] mm
relative improvement = 58.1%
```

四条轨迹的平均改善均为 37.18–40.75 mm，所有分轨迹 CI 均低于 0。worst-seed 表已单独保存，避免只报告平均值。

## MPC vs IK

连接键为 trajectory、perturbation type/level、seed、reference hash。共形成 2,160 pairs、36 个 trajectory × condition clusters。deterministic Direct baseline 在每个 cluster 内从 5 个相同内容 seed 去重为 1 个固定 baseline。

FullVirtual 的 TCP RMSE 差值：

| baseline | MPC − IK | 95% CI |
| --- | ---: | ---: |
| Projected Direct IK | -13.48 mm | [-16.58, -10.80] mm |
| Preview IK | -7.86 mm | [-11.26, -4.81] mm |
| Raw Direct IK | -13.48 mm | [-16.47, -10.78] mm |

这是跨 nominal 与四类扰动 level 3/6 的两级 cluster bootstrap 结果。Projected Direct IK 是公平执行约束 baseline；Raw Direct IK 只作为工程参考。
