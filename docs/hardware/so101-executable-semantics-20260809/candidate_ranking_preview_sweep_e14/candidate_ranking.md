# SO101 counterfactual candidate ranking

base rollout: `/home/xinlei/Data/RL_Projects/NN-MPC_RobotArm/outputs/hardware/so101_pre_mpc/20260809_headroom/direct_preview/preview_0/rollout.npz`

primary selection metric: `candidate_pairwise_accuracy_then_spearman_then_top1`; selected: `u_minus_q_e14`

## Model-vs-real ranking

| model | anchors | Spearman | pairwise accuracy | top-1 accuracy |
|---|---:|---:|---:|---:|
| u_minus_q_e14 | 6 | 0.7136 | 0.8145 | 0.6667 |

## Candidate predicted cost (rad²)

| candidate | mean predicted cost |
|---|---:|
| direct | 0.0002304441 |
| preview:1 | 0.0002079287 |
| preview:2 | 0.00019618927 |
| preview:3 | 0.00018740587 |
| preview:4 | 0.00018519482 |
| preview:5 | 0.00018198753 |
| preview:6 | 0.00017849246 |

The primary checkpoint-selection signal is candidate ranking against repeated real runs; κ/sensitivity is retained only as a secondary safety/diagnostic gate.
