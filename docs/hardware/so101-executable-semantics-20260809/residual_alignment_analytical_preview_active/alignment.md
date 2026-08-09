# MPC residual / reference-velocity alignment

rollout: `/home/xinlei/Data/RL_Projects/NN-MPC_RobotArm/outputs/hardware/so101_pre_mpc/20260809_diagnostic/analytical_preview_active/rollout.npz`
dt: `0.033333333` s; active_start_tick: `83`

| joint | samples | corr(r,dq_des) | lead slope [s] | same-sign | opposite-sign |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 475 | -0.6831 | -0.0855 | 0.236 | 0.764 |
| shoulder_lift | 276 | 0.5512 | 0.0803 | 0.728 | 0.272 |
| elbow_flex | 523 | -0.3899 | -0.0323 | 0.325 | 0.675 |
| wrist_flex | 398 | 0.5199 | 0.0292 | 0.794 | 0.206 |
| wrist_roll | 0 | nan | nan | nan | nan |

`lead_slope_s` is the least-squares coefficient in `residual ~= lead_slope_s * dq_des`. Positive correlation and same-sign rate above 0.5 are necessary (not sufficient) evidence of anticipatory control.
