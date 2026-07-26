# 最终证据冻结

最终冻结目录为 `outputs/paper_final/`，由 `freeze_paper_evidence.py` 从以下只读来源构建：

- schema-v6 control manifest；
- 4 组 delay calibration；
- 5 个 suite 的 case index、case-level CSV、aggregate 和 bootstrap；
- GRU held-out 与 formal-command replay；
- 720/540 历史证据审计；
- MPC–IK 配对与 baseline 去重报告；
- 代表性图、delay 图、projection trade-off 和 timeline。

冻结策略：

- 不复制大体积 rollout；manifest 记录来源路径、SHA-256 与大小；
- 论文表格只从冻结 summaries/statistics 读取；
- control commit 与 analysis base commit 分开记录；
- 历史 cohort 的 commit 状态明确标为 unverified，禁止与新 schema-v6 rollout 混写为同一正式版本；
- 报告完成后运行最终 artifact inventory，任何后续控制代码或参数变化都要求新建版本而不是覆盖 v2。

本轮最重要的负面结果也被冻结：NoFeedback 未显示 nominal 独立收益，Anchor-only 在 D≥6 崩坏。最终论文必须保留这些限定，不能只挑选支持原假设的结果。
