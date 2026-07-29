# Core-Claim Evidence Tests

This report adds 99 closed-loop rollouts, 54 candidate snapshots, and 35
passing targeted unit tests.

## History alignment

| Variant | TCP RMSE |
| --- | ---: |
| Full alignment | 30.01 mm |
| Stale history | 29.35 mm |
| No alignment | 32.90 mm |

No alignment minus full alignment was +2.89 mm (95% CI [0.84, 4.97] mm).
Stale history minus full was -0.67 mm (95% CI [-2.02, 0.45] mm). The complete
alignment module is supported, but these nominal trajectories do not establish
an independent tracking gain from advancing recurrent history.

## Tracking and effort

| Setting | TCP RMSE | Command acceleration RMS | Torque RMS |
| --- | ---: | ---: | ---: |
| Projected IK | 40.68 mm | 0.57 rad/s^2 | 5.90 Nm |
| Frozen 1x | 30.70 mm | 6.78 rad/s^2 | 19.31 Nm |
| Post-freeze 2x | 30.30 mm | 6.56 rad/s^2 | 13.40 Nm |
| Post-freeze 8x | 36.74 mm | 5.60 rad/s^2 | 8.37 Nm |

The post-freeze sweep is diagnostic, not a replacement for the primary frozen
configuration. It demonstrates an accuracy-effort trade-off.

## Candidate ranking

On 54 activation snapshots, within-snapshot Spearman correlation was 0.976,
pairwise concordance was 0.988, and mean realized relative regret was 0.192%.
The 20-step branch-replay joint RMSE was 0.889 mrad. The diagnostics cover
retained projection-active candidates, not the complete CEM population.

UR5e reached 23.7% maximum XML torque utilization. ABB has no finite XML
`forcerange`, so this repository does not claim ABB torque headroom.
