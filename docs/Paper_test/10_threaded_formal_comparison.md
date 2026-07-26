# ThreadedASAP 正式对比

## 结论

真实双线程部署复现了 FullVirtual 的 nominal 跟踪水平。20 个 trajectory×seed 配对中，ThreadedASAP 相对新跑 FullVirtual 的 TCP RMSE 差为 **+0.054 mm**，95% paired-bootstrap CI 为 **[-1.094, +1.303] mm**。该区间跨 0，且误差尺度远小于 FullVirtual 相对 NaiveDelayed 的 39.55 mm 改善，因此没有证据表明线程部署造成实质性 tracking loss。

ThreadedASAP 也优于确定性 IK baseline。按 trajectory×condition cluster 做两级 bootstrap 后，全部 36 个条件 cluster 上：

- 相对 Projected Direct IK，TCP RMSE 平均降低 **12.38 mm**；
- 相对 Preview IK，TCP RMSE 平均降低 **6.75 mm**。

分扰动结果见 [13_robustness_by_perturbation.md](13_robustness_by_perturbation.md)。observation-noise 条件下，ThreadedASAP 与 Preview IK 的区间跨 0，因此论文不应声称对 Preview IK 在每一种扰动下都显著更好。

## 统计口径

- ThreadedASAP–FullVirtual：相同 trajectory、CEM seed 的 20 个严格配对，10,000 次 case bootstrap。
- MPC–IK：连接键为 trajectory、condition、level、seed 和 reference hash；确定性 IK 重复只保留一个内容唯一 baseline。
- bootstrap 先重采样 trajectory×condition cluster，再在 cluster 内重采样 MPC seed；没有把 10 ms tick 当作样本。

证据文件：

- `outputs/paper_revision_v1/statistics/threaded_vs_fullvirtual.json`
- `outputs/paper_revision_v1/statistics/ik/mpc_vs_projected_ik_bootstrap.json`
- `outputs/paper_revision_v1/statistics/ik/baseline_deduplication_report.json`

## 可支撑的论文表述

可以写：真实 ThreadedASAP 在 nominal 条件下复现了 fixed-delay FullVirtual 的 tracking 水平，并在总体鲁棒性 benchmark 上优于 Projected IK 和 Preview IK。

不应写：线程实现与虚拟调度完全等价，或 ThreadedASAP 在每个扰动、每个 seed 上都优于所有 IK baseline。
