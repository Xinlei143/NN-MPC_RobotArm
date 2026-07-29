# Final-Pool Task-Space Reranking

Twenty-four cases compared joint-only scoring with exact task-space reranking.

| Comparison | TCP-RMSE change, task minus joint | 95% CI |
| --- | ---: | ---: |
| Fixed D=6 virtual execution | -2.20 mm | [-4.40, 0.24] mm |
| Calibrated-delay threaded execution | -3.05 mm | [-4.32, -0.96] mm |

Reranking improved deployed TCP metrics but raised solve and end-to-end p95
latency by about 7 ms. It is therefore reported as an accuracy-latency
trade-off, not as an unconditional independent tracking improvement.
