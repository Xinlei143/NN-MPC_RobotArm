# NN-MPC_RobotArm

Learned-dynamics, delay-aware residual CEM-MPC for position-controlled
manipulators. The repository contains the ABB IRB 2400 primary MuJoCo
evaluation, an independently trained and calibrated UR5e replication, and a
separate SO101 real-hardware evaluation pipeline for the final paper protocol.

> The MuJoCo workflows are simulation research code. Position, velocity,
> acceleration, and safety limits are not hardware ratings. The SO101 workflow
> is a separately gated real-hardware experiment and requires the configured
> startup/homing procedure, operator-supported shutdown, collision checking,
> an independent emergency-stop path, and a hardware-specific safety review.

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
the SHA-256-tracked artifacts listed in
[`PUBLIC_MANIFEST.json`](evidence/robio2026/PUBLIC_MANIFEST.json): ABB and UR5e summaries, paired bootstrap results,
robustness/timing and paired IK-baseline statistics, true-history GRU model
diagnostics, mechanism-isolation results, figures, delay calibrations,
robot/reference contracts, and portable cohort audits. It excludes raw
rollouts, caches, checkpoints, normalizers, and paper files.

The ROBIO 2026 evidence bundle reports MuJoCo experiments only. UR5e is a
robot-specific replication with its own data, model, IK references, and
activation-delay calibration; it is not a shared-model transfer result. The
SO101 real-hardware results are maintained as local experiment records because
they include raw rollouts and hardware-specific artifacts that are not part of
the compact public bundle.

## SO101 real-hardware paper experiment

The final SO101 protocol is frozen in
[`configs/experiments/so101_final_paper_20260810.yaml`](configs/experiments/so101_final_paper_20260810.yaml).
It compares Direct IK, fixed Preview6 (6 steps / 200 ms), and tracking-dominant
NN-MPC on 63 paired trials: 9 development-circle trials and 54 held-out trials
covering ellipse, back-and-forth, and rounded-square references at nominal and
fast speeds.

The final NN-MPC configuration is 30 Hz, horizon 6, GRU history 16, CEM with
128 samples and 2 iterations, and requested residual authority `|r_j| <= 1°`.
The objective is `J = C_q`; executable velocity, acceleration, joint-limit,
braking, encoder-quantization, startup, and shutdown safeguards remain active.
The requested residual is measured before the executable-command projector;
the executed command deviation can therefore differ from the requested
residual after stateful projection and quantization.

Use the frozen trial wrapper for one registered trial at a time:

```bash
conda run --no-capture-output -n lerobot python scripts/run_so101_paper_trial.py \
  --protocol configs/experiments/so101_final_paper_20260810.yaml \
  --trial-id <TRIAL_ID> \
  --confirm <TRIAL_ID> \
  --operator <OPERATOR>
```

The wrapper prevents non-registered parameters and output overwrites. The
offline analyzer evaluates all completed artifacts without contacting the
robot:

```bash
conda run --no-capture-output -n lerobot python scripts/analyze_so101_paper_trials.py \
  --protocol configs/experiments/so101_final_paper_20260810.yaml \
  --write-doc
```

The current result record is
[`docs/hardware/so101-final-paper-experiment-results-20260810.md`](docs/hardware/so101-final-paper-experiment-results-20260810.md).
It records the held-out aggregate, paired comparisons, residual definitions,
latency, safety fields, and command-activity trade-off. The raw SO101 outputs
are generated under `outputs/hardware/` and are intentionally not tracked.

The evidence bundle supports claim checking and aggregate reanalysis, not a
from-scratch reproduction: large learned artifacts and raw runtime arrays are
deliberately excluded. Its README documents each public directory and the
boundaries of the reported claims.

## Repository layout

```text
configs/robots/          RobotSpec YAML files for ABB IRB 2400 and UR5e
configs/hardware/        Local hardware configurations, including SO101
dynamics_modeling/       MuJoCo assets, data collection, training, and validation
mpc/                     CEM, rollout, constraints, delay protocol, IK, and diagnostics
scripts/                 Runners, reference tools, robustness, and paper workflows
docs/                    Current architecture, guides, safety notes, hardware records, and paper audits
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

## MPC semantics, constraints, and recovery

Residual MPC does not search an unconstrained absolute command trajectory. It
optimizes bounded corrections around the IK nominal:

```text
q_des[t+1:t+H] -> q_nom -> sample r / r_max in [-1, 1]
                  -> planner_projection(q_nom + r)
                  -> learned rollout and cost -> select command
                  -> 100 Hz physical projection and execution
```

With the default `raw_ik` nominal-command semantics, zero residual is exactly
the raw Direct-IK nominal. `--cem_execute lowest_cost` compares the zero
residual baseline, best sample, and final distribution mean. Legacy acceleration
policies, linear control points, and stage-one GPU FK are retained for historical
reproduction and ablations; they are not the default method.

`threaded_asap` uses a CUDA worker and a 100 Hz execution loop. The worker
replans whenever a solve completes; its update rate is therefore hardware and
load dependent, not a fixed 20 Hz setting. `virtual_asap` is the deterministic
logical-delay mode for repeatable ablations, while `synchronous` and
`virtual_smooth` are fixed-replanning baselines. Formal experiments must derive
`--anticipation_delay_steps` from measured end-to-end latency rather than reuse
the local fallback value.

Planner projection enforces configured kinematic and braking-aware constraints
on candidates. The execution layer independently applies position, velocity,
acceleration, and braking projection to every final `q_ref`. The default
residual bounds are `[0.12, 0.10, 0.12, 0.15, 0.15, 0.20]` rad, subject to
RobotSpec and CLI overrides. A planner failure falls back to the nominal and
resets the CEM warm start. Persistent tracking deterioration or residual
saturation can trigger recovery; hitting a command limit is diagnostic by
itself and is not a recovery event.

## Optional model-disagreement monitor and residual gate

The uncertainty feature is an optional post-selection model-disagreement monitor
and residual gate and is disabled by default (`--uncertainty_mode off`). It is
available only for learned MPC in `threaded_asap` mode. Model A remains the only
model used by CEM; replicas evaluate the selected command sequence and never
average predictions into the control command.

```text
Model-A CEM selects q_ref[0:H)
    -> primary cached rollout + compatible replica rollouts
    -> normalized inter-model RMS disagreement
    -> monitor, hysteretic residual limiting, or nominal fallback
```

The score is normalized by the primary normalizer's state standard deviation:

```text
sqrt(mean_{horizon,state}((std_models[x_hat] / state_std)^2))
```

Thresholds depend on replica count, training protocol, normalizer, and horizon;
they cannot be transferred between configurations. All replicas must match the
primary model type, state dimension, target mode, history length, and control
period. At least two replicas beyond the primary model are required.

| Mode | Behavior |
| --- | --- |
| `off` | Standard residual MPC with no uncertainty computation. |
| `ensemble_monitor` | Logs disagreement and risk flags without changing commands. |
| `ensemble_soft_gate` | Uses `normal -> suspected -> limited -> fallback` hysteresis. |

In soft-gate mode, sustained high disagreement limits the residual; high
disagreement plus predicted physical risk causes nominal fallback. Nonfinite
replica output or an expired replica-computation budget also causes conservative
fallback. The active-packet prediction innovation is a confirmation signal, not
an immediate physical emergency. See the
[uncertainty supervisor guide](docs/safety/uncertainty-soft-supervisor.md) for
the full training and calibration procedure.

## Robots and reproducibility

Use a complete `RobotSpec` for every robot. Do not substitute an XML alone:
the specification binds actuator and joint order, TCP, home configuration,
control period, gravity compensation, and limits. UR5e workflows are described
in [the dedicated guide](docs/guides/ur5e-end-to-end-workflow.md).

For the manuscript workflow, use the frozen paper runner and inspect its
manifests before reusing a model or reference:

```bash
conda run -n pendulum-rl python -m scripts.paper_experiments.workflow --help
conda run -n pendulum-rl python scripts/paper_experiments/workflow.py \
  reanalyze-model-validation --overwrite
conda run -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  reanalyze-model-validation --overwrite
python3 scripts/paper_experiments/publish_evidence.py
```

The two reanalysis commands evaluate saved held-out rollouts on windows with a
complete 16-token ground-truth GRU history. They generate the compact ABB and
UR5e model-validation summaries used by the ROBIO evidence bundle without
rerunning training, MuJoCo, or closed-loop controller experiments.

## Robustness, Model-C, tests, and outputs

The robustness workflows cover payload, actuator-gain mismatch, force pulses,
and observation noise. Every rollout manifest records robot, dataset,
checkpoint, normalizer, XML, and reference identity. Model-C collection,
dataset, benchmark, and evaluation workflows live under `scripts/model_c/`;
their outputs must not be mixed with the frozen Model-A evidence without a new
experiment protocol.

Every MPC run writes `rollout.npz`, `rollout.csv`, tracking/control plots, and
`run_summary.json`; task-space runs also write `task_tracking_summary.json`.
Record tracking error together with planner rate, deadline misses, late-packet
drops, packet age, projection activity, and fallback duty cycle.

Run the MPC tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl pytest -q mpc/tests
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
