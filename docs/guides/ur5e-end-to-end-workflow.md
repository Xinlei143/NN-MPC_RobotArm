# UR5e End-to-End Workflow

UR5e uses `configs/robots/ur5e.yaml`. Collect UR5e data, train its own model,
generate its own references, calibrate activation delay, then run the same
residual-MPC protocol used for ABB. Do not reuse ABB checkpoints, normalizers,
references, or delay values.

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py --robot_config configs/robots/ur5e.yaml --help
conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py --robot_config configs/robots/ur5e.yaml --help
conda run -n pendulum-rl python -m scripts.paper_experiments.workflow --help
```

The frozen UR5e results and audit are published in
[`evidence/robio2026/ur5e`](../../evidence/robio2026/ur5e).
