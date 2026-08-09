# SO101 counterfactual candidate ranking

base rollout: `/home/xinlei/Data/RL_Projects/NN-MPC_RobotArm/outputs/hardware/so101_pre_mpc/20260809_executable_semantics/u_minus_q_e14_cuda_graph_active/rollout.npz`

primary selection metric: `candidate_pairwise_accuracy_then_spearman_then_top1`; selected: `u_minus_q_e14`

## Model-vs-real ranking

| model | anchors | Spearman | pairwise accuracy | top-1 accuracy |
|---|---:|---:|---:|---:|
| u_minus_q_e14 | 6 | 0.0000 | 0.5000 | 0.5000 |

## Candidate predicted cost (rad²)

| candidate | mean predicted cost |
|---|---:|
| direct | 0.00026701224 |
| lead:0.05 | 0.00024417258 |
| lead:0.10 | 0.00022420182 |
| lead:0.15 | 0.00021652351 |
| lead:0.20 | 0.00020998259 |
| active | 0.00023796629 |

The primary checkpoint-selection signal is candidate ranking against repeated real runs; κ/sensitivity is retained only as a secondary safety/diagnostic gate.
