# UR5e Robot-Specific Replication

UR5e is a second independently trained and delay-calibrated MuJoCo manipulator.
Its XML, TCP, home configuration, limits, dataset, checkpoint, normalizer,
references, and activation delay are separate from ABB. This is not a
shared-model generalization experiment.

The nominal matrix contains 84 runs and uses D=7.

| Method | TCP RMSE |
| --- | ---: |
| Projected IK | 15.06 mm |
| NaiveDelayed | 34.78 mm |
| FullVirtual | 10.43 mm |
| ThreadedAsync | 10.42 mm |

FullVirtual minus NaiveDelayed was -24.36 mm (95% CI [-26.08, -22.64] mm),
a 70.0% reduction. ThreadedAsync minus FullVirtual was -0.01 mm (95% CI
[-0.39, 0.41] mm). The freeze audit found all cases and no constraint violations.
UR5e XML force utilization reached 23.7% at most; this is not a hardware rating.
