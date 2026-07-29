# Dynamics Modeling

This module collects MuJoCo position-servo transitions, trains learned
closed-loop dynamics, and evaluates open-loop rollout quality. The default
paper model is a GRU predicting `delta_dq` from state-history and absolute joint
reference history.

## Data collection

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --num_episodes 200 --episode_len 300 --num_envs 8 --action_std 0.5 \
  --save_path dynamics_modeling/outputs/datasets/abb_model_a.npz
```

Each dataset has a manifest. A model must only be used with a compatible robot
specification, XML, control period, normalizer, and target definition.

## Training

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --data_path dynamics_modeling/outputs/datasets/abb_model_a.npz \
  --model_type gru --history_len 16 --target_mode delta_dq --control_dt 0.01 \
  --epochs 100 --save_dir dynamics_modeling/outputs/checkpoints
```

## Evaluation

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/eval_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --checkpoint <CHECKPOINT>/best_model.pt \
  --normalizer <CHECKPOINT>/normalizer.pt \
  --model_type gru --history_len 16
```

UR5e requires its own data, model, normalizer, references, and delay
calibration. Do not combine artifacts across robots.

## Environment and smoke checks

Run from the repository root. The project uses the `pendulum-rl` conda
environment. Verify the MuJoCo model before collecting a large dataset:

```bash
conda run -n pendulum-rl python -c "import mujoco, torch; print(mujoco.__version__); print(torch.__version__)"
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py \
  --robot_config configs/robots/abb_irb2400.yaml \
  --num_episodes 1 --episode_len 3 --num_envs 1 \
  --save_path dynamics_modeling/outputs/datasets/abb_smoke.npz
```

Use a writable Matplotlib cache directory if the environment does not provide
one. Viewer support is optional and should not be enabled in headless or
parallel collection jobs.

## Structured collection and parallel environments

The collector settles the plant, then mixes holds, small steps, smooth waypoint
targets, and sinusoidal joint references. Single-environment collection is
useful for visual inspection; `--num_envs > 1` is for headless dataset
generation. Parallel environments change throughput, not the RobotSpec or
actuator semantics.

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/collect_data.py --help
```

Every saved dataset includes state, absolute `q_ref` action, successor target,
episode boundaries, robot identity, and a manifest. Do not train a model on a
dataset collected with a different XML, actuator order, control period, or
target mode.

## Model families

The training script supports MLP, GRU, and Transformer models.

- **MLP** uses the current state and command only; it is a compact baseline.
- **GRU** uses a command-aligned history and is the default model for the paper
  workflow.
- **Transformer** also uses history and requires the same careful history,
  target-mode, and control-period compatibility checks.

Representative commands are:

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml --data_path <DATASET>.npz \
  --model_type mlp --target_mode delta_dq --save_dir dynamics_modeling/outputs/checkpoints

conda run -n pendulum-rl python dynamics_modeling/scripts/train_dynamics.py \
  --robot_config configs/robots/abb_irb2400.yaml --data_path <DATASET>.npz \
  --model_type transformer --history_len 16 --target_mode delta_dq \
  --save_dir dynamics_modeling/outputs/checkpoints
```

GRU and Transformer checkpoints record architecture, history length, target
mode, normalizer, and control period. Loading code rejects incompatible
artifacts rather than silently changing input semantics.

## GPU, workers, AMP, and resume

Use CUDA when available. Increase DataLoader workers only when storage and CPU
throughput support it; too many workers can slow small datasets. Automatic mixed
precision is optional and should be validated against a non-AMP run before use
in a formal workflow.

To continue an interrupted run, use the training script's resume/checkpoint
options and preserve the original dataset and normalizer contract. Initializing
from an older checkpoint is a new training experiment unless model structure,
target definition, and RobotSpec identity all match.

## Evaluation and diagnostics

Evaluate both one-step and multi-step rollout error. A low teacher-forcing
error does not guarantee useful closed-loop rollout ranking. Inspect per-horizon
position/velocity metrics, nonfinite predictions, joint-limit excursions, and
action-scale coverage.

```bash
conda run -n pendulum-rl python dynamics_modeling/scripts/eval_dynamics.py --help
```

For paper experiments, use the separate formal command replay and
candidate-ranking diagnostics in `scripts/paper_experiments/`; their command
distribution is constrained around the IK nominal and should not be compared
directly with broad random-excitation validation.

## Recommended order

1. Run the smoke collection check.
2. Collect a small single-environment dataset and inspect it.
3. Collect the final headless dataset with the intended RobotSpec.
4. Train the selected model family and validate open-loop rollouts.
5. Generate and validate task-space references.
6. Run Direct IK, then learned residual MPC.
7. Freeze manifests before paired robustness or paper evaluation.

## Common problems

**XML, mesh, or STL file not found.** Use a complete `--robot_config` rather
than a bare XML path, and run commands from the repository root.

**Actuator count or order mismatch.** Dataset, checkpoint, XML, and RobotSpec
are incompatible. Recollect or select matching artifacts; do not reorder arrays
manually.

**Viewer fails to open.** Run headless collection or use a local graphical
session. The dataset pipeline itself does not require a viewer.

**Matplotlib cache warning.** Point `MPLCONFIGDIR` to a writable task-specific
directory before generating figures.

## Relationship to MPC workflows

The learned model predicts closed-loop position-actuator dynamics, not an
unconstrained torque plant. Model A is the default GRU workflow. Model B/C
labels refer to separate research workflows and must retain their own manifests
and evidence. The MPC README and root README describe controller execution;
this module owns data, models, normalizers, and open-loop diagnostics.
