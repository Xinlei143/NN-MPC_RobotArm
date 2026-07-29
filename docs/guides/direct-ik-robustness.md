# Direct-IK Robustness Workflow

Direct IK is the learned-model-free baseline. Use `--controller_mode ik_direct`
with the same validated reference and perturbation definition as the MPC run.
`--ik_command_projection raw` reproduces the raw nominal; `physical` applies
the shared physical execution projection. Report these as distinct baselines.
