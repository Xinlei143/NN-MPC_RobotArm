# SO101 counterfactual candidate ranking

base rollout: `/home/xinlei/Data/RL_Projects/NN-MPC_RobotArm/outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz`

primary selection metric: `candidate_pairwise_accuracy_then_spearman_then_top1`; selected: `None`

## Model-vs-real ranking

| model | anchors | Spearman | pairwise accuracy | top-1 accuracy |
|---|---:|---:|---:|---:|
| u_minus_q_e14 | 0 | nan | nan | nan |

## Candidate predicted cost (rad²)

| candidate | mean predicted cost |
|---|---:|
| direct | 0.00026701362 |
| preview:1 | 0.00025667888 |
| preview:2 | 0.00023892718 |
| preview:3 | 0.00022477894 |
| active | 0.00023796637 |

The primary checkpoint-selection signal is candidate ranking against repeated real runs; κ/sensitivity is retained only as a secondary safety/diagnostic gate.
