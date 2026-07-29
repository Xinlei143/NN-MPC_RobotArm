# Core-Claim Evidence Tests

This report adds 145 closed-loop rollouts, 54 candidate snapshots, and 36
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
| Projected IK | 39.26 mm | 0.58 rad/s^2 | 5.60 Nm |
| Post-freeze 0.5x | 31.73 mm | 6.99 rad/s^2 | 23.52 Nm |
| Frozen 1x | 28.37 mm | 6.69 rad/s^2 | 17.17 Nm |
| Post-freeze 2x | 29.02 mm | 6.49 rad/s^2 | 13.25 Nm |
| Post-freeze 4x | 31.99 mm | 6.18 rad/s^2 | 10.99 Nm |
| Post-freeze 8x | 35.36 mm | 5.53 rad/s^2 | 8.04 Nm |

The post-freeze sweep uses the complete primary FullVirtual grid: four
trajectories, five paired CEM seeds, and the same fixed $D=6$ protocol at every
scale. The 1x rows reproduce the 20 primary FullVirtual cases exactly. Relative
to 1x, 2x changed TCP RMSE by +0.65 mm (95% paired-bootstrap CI
[-0.46, 1.62] mm), while reducing command-acceleration RMS by 0.20 rad/s^2
([-0.28, -0.12]) and torque RMS by 3.92 Nm ([-6.06, -2.17]). The diagnostic
therefore establishes an accuracy--effort sensitivity; it was not used to
replace the primary frozen configuration.

## Candidate ranking

On 54 activation snapshots, within-snapshot Spearman correlation was 0.976,
pairwise concordance was 0.988, and mean realized relative regret was 0.192%.
The 20-step branch-replay joint RMSE was 0.889 mrad. The diagnostics cover
retained projection-active candidates, not the complete CEM population.

UR5e reached 23.7% maximum XML torque utilization. ABB has no finite XML
`forcerange`, so this repository does not claim ABB torque headroom.
