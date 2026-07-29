# Command Reference

Run all commands from the repository root in the `pendulum-rl` environment.

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
conda run -n pendulum-rl python scripts/generate_task_reference.py --help
conda run -n pendulum-rl python scripts/validate_ik.py --help
```

Use `threaded_asap` for CUDA-backed wall-clock MuJoCo execution, `virtual_asap`
for deterministic delay-aware comparisons, and `synchronous` for fixed-interval
baselines. A formal learned MPC run requires a compatible `RobotSpec`,
checkpoint, normalizer, validated reference, and calibrated activation delay.
The complete example is in the root README.
