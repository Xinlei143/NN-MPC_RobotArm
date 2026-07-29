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
