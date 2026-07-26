# 冻结 commit 下的 FullVirtual 重跑

## 执行与完整性

在 detached、clean 的 `77505a5` worktree 下重新执行了 4 trajectories × 5 seeds，共 **20 条 FullVirtual**。运行时把 legacy MPC/IK roots 指向空目录，因此 `_run_case` 只能生成新的 fingerprint cache，不能复用历史 rollout。

新 20 条与原 60 条 removal variants 合并后得到完整的 **80-case matched matrix**，每个 label×trajectory×seed 唯一，无缺失、无重复。输出索引为：

`outputs/paper_fullvirtual_rerun_v1/runs/indexes/fullvirtual_frozen.json`

## 重现性检查

新旧 FullVirtual 的逐 case TCP RMSE 完全一致：

- 历史 20 条均值：28.3719 mm；
- 新跑 20 条均值：28.3719 mm；
- mean delta：0；
- 最大逐 case 绝对差：0。

这说明历史兼容数据在控制结果上可重现；论文消融统计仍统一改用新跑 20 条，避免混合 provenance。

## 更新后的消融

使用新 FullVirtual 重新做 10,000 次 paired bootstrap：

| Removal variant − FullVirtual | TCP RMSE 差 | 95% CI |
| --- | ---: | ---: |
| NoFutureAlignment | +2.820 mm | [+1.270, +4.389] mm |
| NoReanchor | +654.596 mm | [+528.679, +778.682] mm |
| NoFeedback | -0.950 mm | [-2.246, +0.276] mm |

证据支持 future alignment，并非常强地支持 execution-time re-anchor 是绝对位置接口下的必要配套。NoFeedback 的区间跨 0，不能据此声称 fast feedback 有独立 tracking 收益。
