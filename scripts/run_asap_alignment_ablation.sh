#!/usr/bin/env bash
# 2x2 ablation for state-action history alignment and snapshot phase.
set -euo pipefail

asap_env_name="${ASAP_CONDA_ENV:-pendulum-rl}"
asap_output_root="${ASAP_ABLATION_OUTPUT_ROOT:-outputs/mpc/asap_alignment_ablation_h20_seed0}"

run_variant() {
  local asap_label="$1"
  local asap_history_mode="$2"
  local asap_snapshot_mode="$3"

  conda run -n "$asap_env_name" python scripts/run_cem_mpc.py \
    --model_type gru \
    --checkpoint dynamics_modeling/outputs/checkpoints/gru_20260717_152930/best_model.pt \
    --normalizer dynamics_modeling/outputs/checkpoints/gru_20260717_152930/normalizer.pt \
    --reference_mode task \
    --reference_file outputs/references/circle_3laps/reference.npz \
    --multirate_mode threaded_asap \
    --asap_history_mode "$asap_history_mode" \
    --asap_snapshot_mode "$asap_snapshot_mode" \
    --horizon 20 \
    --anticipation_delay_steps 6 \
    --planner_guard_ms 5 \
    --max_execution_steps 1000 \
    --num_samples 128 \
    --rollout_batch_size 128 \
    --cem_iters 2 \
    --seed 0 \
    --device cuda \
    --save_dir "$asap_output_root/$asap_label"
}

# A is the former implementation. D is the corrected physical implementation.
run_variant A_legacy_history_legacy_snapshot legacy_shifted post_step_legacy
run_variant B_aligned_history_legacy_snapshot aligned post_step_legacy
run_variant C_legacy_history_tick_start_snapshot legacy_shifted tick_start
run_variant D_aligned_history_tick_start_snapshot aligned tick_start
