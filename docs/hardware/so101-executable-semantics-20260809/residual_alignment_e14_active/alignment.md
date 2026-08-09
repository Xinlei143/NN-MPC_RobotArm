# MPC residual / reference-velocity alignment

rollout: `/home/xinlei/Data/RL_Projects/NN-MPC_RobotArm/outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz`
dt: `0.033333333` s; active_start_tick: `83`

| joint | samples | corr(r,dq_des) | lead slope [s] | same-sign | opposite-sign |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 493 | -0.8111 | -0.0940 | 0.150 | 0.850 |
| shoulder_lift | 260 | 0.5219 | 0.0707 | 0.750 | 0.250 |
| elbow_flex | 540 | -0.3416 | -0.0303 | 0.335 | 0.665 |
| wrist_flex | 406 | 0.4750 | 0.0266 | 0.766 | 0.234 |
| wrist_roll | 0 | nan | nan | nan | nan |

`lead_slope_s` is the least-squares coefficient in `residual ~= lead_slope_s * dq_des`. Positive correlation and same-sign rate above 0.5 are necessary (not sufficient) evidence of anticipatory control.
