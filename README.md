# NN-MPC_RobotArm

Learned-dynamics, delay-aware residual CEM-MPC for position-controlled
manipulators in MuJoCo. The repository contains the ABB IRB 2400 primary
evaluation, an independently trained and calibrated UR5e replication, task-space
reference generation, and reproducible evidence for the ROBIO 2026 manuscript.

> This is simulation research code. Position, velocity, acceleration, and
> safety limits in this repository are planning or MuJoCo constraints, not
> hardware ratings. A physical-robot deployment requires a separate safety
> review, collision checking, emergency-stop integration, and real-time testing.

## What the controller does

The controller tracks a continuously generated IK nominal with bounded joint
reference residuals. A GRU models the closed-loop position-actuator dynamics;
CEM evaluates residual sequences around the nominal. In threaded operation, a
CUDA planner runs asynchronously while the 100 Hz execution loop continues to
apply physical projection, packet-age indexing, feedback, and fallback logic.

```text
task-space reference -> continuous DLS IK -> absolute joint nominal
                                             |
state/history -> GRU + residual CEM --------+-> projected q_ref -> MuJoCo actuators
                     |                                 ^
                     +-- activation-aligned packet ----+
```

The primary protocol advances the predicted activation state, recurrent context,
and reference window together, then reanchors the timestamped residual at
execution time. The action is always an absolute joint-reference target:

```text
state  x_t = [q_t(6), dq_t(6)]
action u_t = q_ref,t(6)  # rad
```

## Public evidence

The compact [ROBIO 2026 evidence bundle](evidence/robio2026/README.md) contains
70 SHA-256-tracked artifacts: ABB and UR5e summaries, paired bootstrap results,
robustness and timing statistics, model diagnostics, mechanism-isolation
results, figures, and portable audit manifests. It excludes raw rollouts,
caches, checkpoints, normalizers, and paper files.

The reported experiments are MuJoCo-only. UR5e is a robot-specific replication
with its own data, model, IK references, and activation-delay calibration; it is
not a shared-model transfer result.

## Repository layout

```text
configs/robots/          RobotSpec YAML files for ABB IRB 2400 and UR5e
dynamics_modeling/       MuJoCo assets, data collection, training, and validation
mpc/                     CEM, rollout, constraints, delay protocol, IK, and diagnostics
scripts/                 Runners, reference tools, robustness, and paper workflows
docs/                    Current architecture, guides, safety notes, and paper audits
evidence/robio2026/      Compact public evidence bundle and checksums
outputs/                 Generated local outputs; intentionally not tracked
```

## Environment

Commands below assume the `pendulum-rl` conda environment and are run from the
repository root. Dependencies are listed in
[`dynamics_modeling/requirements.txt`](dynamics_modeling/requirements.txt).

```bash
conda run -n pendulum-rl python -c "import mujoco, torch; print(mujoco.__version__); print(torch.__version__)"
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
```

## Quick start

### 1. Collect position-servo data

The learned model predicts \(x_{t+1}=f(x_t,q_{\mathrm{ref},t})\), where
`q_ref` is an absolute joint target.

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --num_episodes 200 --episode_len 300 --num_envs 8 --action_std 0.5 \
  --save_path dynamics_modeling/outputs/datasets/abb_model_a.npz
```

The collector writes a manifest beside the dataset. Training and evaluation
check that manifest against the selected `RobotSpec`.

### 2. Train and validate a GRU

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --data_path dynamics_modeling/outputs/datasets/abb_model_a.npz \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --epochs 100 --save_dir dynamics_modeling/outputs/checkpoints

conda run -n pendulum-rl python dynamics_modeling/scripts/eval_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --checkpoint <CHECKPOINT>/best_model.pt \
  --normalizer <CHECKPOINT>/normalizer.pt \
  --model_type gru --history_len 16
```

### 3. Generate and validate a task-space reference

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --shape figure8 --repeat_count 3 \
  --save_dir outputs/references/figure8

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/figure8/reference.npz
```

### 4. Run residual CEM-MPC

`threaded_asap` is the wall-clock, CUDA-backed execution mode. It replans when
the worker finishes rather than at a fixed planner frequency. Use a calibrated
activation delay for formal experiments; `6` below is only an example.

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --checkpoint <CHECKPOINT>/best_model.pt \
  --normalizer <CHECKPOINT>/normalizer.pt \
  --model_type gru --history_len 16 \
  --reference_mode task --reference_file outputs/references/figure8/reference.npz \
  --horizon 20 --num_samples 128 --cem_iters 2 --rollout_batch_size 128 \
  --mpc_policy residual --residual_parameterization full \
  --multirate_mode threaded_asap --delay_protocol full \
  --anticipation_delay_steps 6 \
  --planner_projection on --planner_projection_backend compiled \
  --planner_projection_strategy two_stage \
  --exact_task_space_cost on --stage_one_task_space_cost off \
  --cem_execute lowest_cost --save_dir outputs/mpc/figure8_residual
```

`threaded_asap` requires CUDA and does not support `--visualize`. Use
`virtual_asap` for deterministic delay-aware ablations and `synchronous` for
fixed-interval baselines.

### Direct-IK baseline

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --controller_mode ik_direct --reference_mode task \
  --reference_file outputs/references/figure8/reference.npz \
  --ik_command_projection raw --save_dir outputs/mpc/figure8_ik_direct
```

`raw` reproduces the direct-IK nominal. Use `physical` to share the physical
execution projection with the MPC path. Preview IK is a separate baseline and
must not be reported as standard Direct IK.

## Robots and reproducibility

Use a complete `RobotSpec` for every robot. Do not substitute an XML alone:
the specification binds actuator and joint order, TCP, home configuration,
control period, gravity compensation, and limits. UR5e workflows are described
in [the dedicated guide](docs/guides/ur5e-end-to-end-workflow.md).

For the manuscript workflow, use the frozen paper runner and inspect its
manifests before reusing a model or reference:

```bash
conda run -n pendulum-rl python -m scripts.paper_experiments.workflow --help
python3 scripts/paper_experiments/publish_evidence.py
```

## Documentation

- [Documentation index](docs/README.md)
- [Current MPC architecture](docs/architecture/current-mpc-architecture.md)
- [Run commands](docs/guides/run-commands.md)
- [UR5e end-to-end workflow](docs/guides/ur5e-end-to-end-workflow.md)
- [Uncertainty soft supervisor](docs/safety/uncertainty-soft-supervisor.md)
- [Paper evidence and audits](docs/Paper_test/README.md)

## License and third-party assets

The project uses the [Apache License 2.0](LICENSE.md). UR5e asset provenance
and license information are retained in `dynamics_modeling/robots/ur5e/`.
