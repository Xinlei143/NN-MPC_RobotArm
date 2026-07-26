# ROBIO 补充实验总览

本目录汇总 2026-07-26 完成的 ROBIO 补充实验、历史证据审计和最终冻结口径。原始结果位于 `outputs/paper_delay_aware_two_stage_v2`，第一版归档位于 `outputs/paper_final`，论文修订证据位于 `outputs/paper_revision_v1`。

## 完成状态

| 项目 | 结果 |
| --- | --- |
| 历史证据审计 | 720 MPC + 540 IK，审计通过 |
| 回归测试 | 130 passed，1 skipped，12 subtests passed |
| 延迟标定 | 4 组 × 500 plans，均得到 D=6 |
| GRU held-out 验证 | 20 rollouts，action std 0.5/0.8 各 10 |
| 核心机制消融 | 80 cases：冻结 commit 下补跑 20 条 FullVirtual + 原 60 条消融 |
| final-pool task-space cost | 24 cases |
| 四阶段 delay sweep | 96 cases |
| projection choice | 36 cases |
| 新控制 rollout | 198 条 |
| MPC–IK 配对 | 2,160 pairs，10,000 次两级 bootstrap |
| 代表性图与 timeline | 6 个 PDF + case manifest |

## 核心结论

- FullVirtual 相对 NaiveDelayed 的 pooled TCP RMSE 降低 39.55 mm，95% CI 为 [36.87, 42.10] mm，平均相对改善 58.1%。
- 移除 future alignment 后 TCP RMSE 增加 2.82 mm；移除 re-anchor 后增加 654.60 mm。移除 fast feedback 的平均差为 -0.95 mm，CI 跨 0，因此当前 nominal 消融不支持把 feedback 称为决定性核心贡献。
- exact task-space reranking 在 fixed-D6 下改善 2.20 mm，但 CI 跨 0；在各自 calibrated-D threaded 部署下改善 3.05 mm，CI 为 [0.96, 4.32] mm，同时 E2E P95 增加约 6.82 ms。
- TwoStage 相对 FullCompiled 在 common-D 和 deployed 两组均通过预注册的 +5% 非劣界，且平均 TCP RMSE 分别改善约 9.9% 和 9.3%，计算延迟也更低。
- 四阶段 sweep 不支持原先“Naive 对 D 最敏感”的完整预期：Anchor-only 在 D≥6 时发生严重退化；Full 与 Anchor+Reanchor 保持稳定。论文应把这写成 re-anchor 是 future alignment 可部署的必要配套，而非平滑累计收益。

逐项证据见下列报告。

- [证据审计](00_evidence_audit.md)
- [代码、配置与测试冻结](01_freeze_and_regression.md)
- [GRU 验证](02_gru_validation.md)
- [机制消融](03_mechanism_ablation.md)
- [Task-space cost](04_task_space_cost.md)
- [Delay sweep](05_delay_sweep.md)
- [Projection choice](06_projection_choice.md)
- [主终点与 IK 配对](07_main_endpoint_and_ik.md)
- [实时性与图表](08_timing_and_figures.md)
- [最终冻结](09_final_freeze.md)
- [ThreadedASAP 正式对比](10_threaded_formal_comparison.md)
- [冻结 FullVirtual 重跑](11_frozen_fullvirtual_rerun.md)
- [完整实时性表](12_complete_realtime_table.md)
- [分扰动鲁棒性](13_robustness_by_perturbation.md)
