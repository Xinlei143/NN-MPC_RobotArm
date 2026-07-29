# Main Endpoint and IK Comparisons

Across 20 matched trajectory-seed cases, FullVirtual minus NaiveDelayed TCP
RMSE was -39.55 mm (95% CI [-42.10, -36.87] mm), a 58.1% reduction.

Cluster-bootstrap comparisons across nominal and perturbation conditions gave:

| Baseline | FullVirtual minus baseline TCP RMSE | 95% CI |
| --- | ---: | ---: |
| Projected Direct IK | -13.48 mm | [-16.58, -10.80] mm |
| Preview IK | -7.86 mm | [-11.26, -4.81] mm |
| Raw Direct IK | -13.48 mm | [-16.47, -10.78] mm |

Projected Direct IK is the shared-execution-constraint baseline; raw Direct IK
is retained only as an engineering reference.
