# 最终 GRU checkpoint 独立验证

验证对象为 `gru_20260717_182930` 的同一 checkpoint 与 normalizer。共生成 20 条冻结后 held-out MuJoCo rollout，长度 200，seed 20260730；action std 0.5 和 0.8 各 10 条，匹配训练数据中实际存在的两个 action 标度。

## 关键结果

| action std | horizon | q RMSE mean | q RMSE P95 | divergence mean |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 1 | 0.00337 rad | 0.00954 rad | 0 |
| 0.5 | 5 | 0.01144 rad | 0.03121 rad | 0 |
| 0.5 | 10 | 0.01753 rad | 0.04791 rad | 0 |
| 0.5 | 20 | 0.02164 rad | 0.05969 rad | 0 |
| 0.8 | 1 | 0.00715 rad | 0.01776 rad | 0 |
| 0.8 | 5 | 0.02336 rad | 0.05717 rad | 0 |
| 0.8 | 10 | 0.03559 rad | 0.08725 rad | 0.04 |
| 0.8 | 20 | 0.04411 rad | 0.10757 rad | 0.02 |

Teacher-forcing one-step q RMSE 为 0.00084 rad（std 0.5）和 0.00128 rad（std 0.8）。高幅 action 下多步误差明显增大，因此结论应限定为“在训练覆盖范围内具备可用的短时预测能力”，不能写成全范围无发散。

补充 formal-command replay 已覆盖 80 条 nominal MPC run，并输出每种方法的 1/5/10/20-step aggregate。原始 rollout、hash、逐 horizon CSV/JSON 位于 `outputs/paper_delay_aware_two_stage_v2/diagnostics/`。
