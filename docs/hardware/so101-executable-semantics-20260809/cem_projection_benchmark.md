# CEM projection benchmark (2026-08-09)

The executable projection/GRU loop is now kept on CUDA.  Fixed production
shapes use CUDA Graph replay; eager GPU execution remains available as the
reference backend.  The exact final pool carries its raw counts directly to
the packet, so the worker does not replay the selected trajectory.

Configuration:

- checkpoint: `u_minus_q_epoch14_20260809/.../best_rollout_model.pt`
- SO101 hardware envelope and calibration
- GRU, history length 16, `H=6`
- 128 candidates, 2 CEM iterations, full residual parameterization
- CUDA Graph backend, 3 warm-up plans, 30 measured plans

Measured CEM planner time (candidate search only):

| metric | time |
|---|---:|
| mean | 15.70 ms |
| p50 | 15.80 ms |
| p95 | 16.68 ms |
| p99 | 16.91 ms |
| max | 17.01 ms |

With exact final-pool validation and raw-count handoff, the same run measured
15.71 ms mean, 16.68 ms P95, and 17.02 ms max. No selected action required a
float64 model-rollout fallback. The D=2 exact delay forecast measured 0.84 ms
mean and 1.01 ms P95 after warm-up.

For context, the canonical executable-semantics run before this optimization
reported approximately 30.45 ms mean planning time and 30.77 ms P95 for the
worker-equivalent path. The CUDA Graph result is offline; a physical shadow
run is still required to measure worker scheduling and USB timing.

Eager GPU comparison measured 26.34 ms mean CEM and 31.29 ms P95 including
exact postprocessing, confirming that the reduction comes from removing
per-step Python/kernel-launch overhead rather than changing the model or
search configuration. Exact raw-count parity tests remain passing.

The benchmark is reproducible without hardware:

```bash
conda run --no-capture-output -n lerobot python scripts/benchmark_cem_projection.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/best_rollout_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809/gru_20260809_150038/normalizer.pt \
  --reference outputs/hardware/so101_pre_mpc/20260808_refs_6phase/circle_p0/joint_reference_mpc.npz \
  --hardware-config configs/hardware/so101_follower.local.yaml \
  --horizon 6 --num-samples 128 --cem-iters 2 --plans 30 --warmup 3 \
  --executable-rollout-backend cuda_graph --selection-validation exact_final_pool \
  --delay-steps 2
```
