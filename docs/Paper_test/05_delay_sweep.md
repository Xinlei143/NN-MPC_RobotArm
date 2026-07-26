# 四阶段 delay sweep

完成 96 cases：circle/fast ellipse × 4 protocols × D={2,4,6,8} × 3 seeds。12 个历史 D6 Full/Naive 与 6 个跨 suite 相同指纹被复用。

## Pooled TCP RMSE

| 协议 | D2 | D4 | D6 | D8 |
| --- | ---: | ---: | ---: | ---: |
| Naive | 54.5 mm | 54.8 mm | 69.6 mm | 91.0 mm |
| Anchor-only | 30.9 mm | 29.6 mm | 831.4 mm | 941.2 mm |
| Anchor+Reanchor | 30.1 mm | 31.5 mm | 29.1 mm | 32.4 mm |
| Full | 32.1 mm | 30.6 mm | 32.5 mm | 33.2 mm |

Full 的退化斜率接近 0；Naive 为约 5.7–6.7 mm/step。Anchor-only 在 D≥6 时出现非线性崩坏，线性斜率达到 145–208 mm/step，因此原计划的平滑累计恢复链没有成立。

可支持的结论是：

- Full 和 Anchor+Reanchor 在 D2–D8 范围内最稳定；
- Naive 随 D 明显恶化；
- future alignment 单独使用而不 re-anchor absolute command 不是安全的中间设计；
- re-anchor 是把 future alignment 转化为稳定执行的必要组件；
- 不能声称“Anchor-only 已显著降低 delay sensitivity”，也不能声称四阶段严格单调改进。
