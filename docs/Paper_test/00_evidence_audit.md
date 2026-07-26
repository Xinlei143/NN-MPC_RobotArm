# 证据审计结果

正式审计输出：`outputs/paper_evidence_audit/`。

| 检查项 | 结果 |
| --- | ---: |
| MPC rollout | 720 |
| IK rollout | 540 |
| missing formal cases | 0 |
| duplicate conflicting cases | 0 |
| run issues / nonfinite primary metrics | 0 |
| configuration mismatches | 0 |
| mixed control semantics | false |
| mixed projection semantics | false |

审计结论为 `passed=true`。checkpoint、normalizer、reference、XML 与目标配置兼容；新增字段按历史代码能力回填为 full residual、stage-one task cost off、stage-one compile off、preview nominal 0。

限制：历史 fingerprint 没有保存可验证的 Git clean/dirty 状态。因此 720/540 被标记为 `historical_compatible` cohort，只能表述为“配置和语义兼容”，不能声称来自本次冻结 commit。新实验单独使用 schema-v6 和冻结 commit。
