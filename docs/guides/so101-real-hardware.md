# SO-ARM101 real-hardware runtime

The SO101 integration deliberately does not emulate `MuJoCoArmEnv.step()`. A
control tick reads `x_k` once, projects and transmits one six-motor raw goal,
then completes `(x_k, transmitted_u_k, x_{k+1})` when the next tick begins.
`transmitted_u_k` means the target handed to Feetech `sync_write`; it is not a
per-motor acknowledgement.

## Safety boundary

- Only the control/I/O thread may access the Feetech bus.
- `SO101Backend.connect()` does not call LeRobot's full `SOFollower.connect()`.
  It verifies the pinned calibration with torque disabled, freezes and reads
  back all behavior-defining registers, preloads the current pose as the goal,
  and enables torque last.
- Every tick performs one `Present_Position` sync read and one six-motor
  `Goal_Position` sync write. The gripper target is latched in raw units and is
  refreshed with the five controlled joints.
- A low-rate Goal Position readback bounds command-delivery uncertainty. All
  transitions since the last matching readback are invalidated after a
  mismatch.
- 50 C removes residual authority and halves motion authority; 55 C latches a
  fault; sustained heating/current requests an early supported torque-off; 60
  C requests torque-off immediately. Provide a physical support/cradle and an
  independent E-stop before any moving test.

The pinned software baseline is LeRobot commit
`2aba372b4e217cc47db28e0f836859b20d1456c9`. Its SO follower implementation
performs a second position read when `max_relative_target` is enabled, so this
backend uses the lower-level bus and performs both safety projections from the
single tick-start measurement. See the [pinned LeRobot source](https://github.com/huggingface/lerobot/blob/2aba372b4e217cc47db28e0f836859b20d1456c9/src/lerobot/robots/so_follower/so_follower.py).

Extend the existing `lerobot` environment with the project runtime before
collection or MPC. If it already uses the pinned editable local checkout,
install the project dependencies without replacing LeRobot:

```bash
conda run -n lerobot python -m pip install -r dynamics_modeling/requirements.txt
```

For a clean environment, `requirements-so101.txt` also pins LeRobot itself.

## Configuration and identities

Copy `configs/hardware/so101_follower.template.yaml` to the gitignored
`configs/hardware/so101_follower.local.yaml`. Replace every `CHECK_ME`, record
the exact calibration SHA-256, and replace the raw and joint safety ranges with
the intersection of EEPROM calibration, nominal model limits, and the manually
validated range.

```bash
conda run -n lerobot python scripts/validate_so101_config.py \
  --hardware-config configs/hardware/so101_follower.local.yaml
conda run -n lerobot python scripts/validate_so101_quantization.py \
  --hardware-config configs/hardware/so101_follower.local.yaml
```

`q_ctrl` is the LeRobot calibrated five-joint coordinate used by collection,
training, limits, and joint-space MPC. `q_kin = sign*q_ctrl + offset` is an
independent mapping used only for FK/TCP/IK. Fit it from at least five
instrumented reference poses:

```bash
conda run -n lerobot python scripts/calibrate_so101_mapping.py poses.npz \
  --output configs/hardware/so101_mapping.local.json
```

Changing PID, either acceleration register, calibration, control rate, safe
limits, estimator version, or LeRobot commit changes `plant_identity` and
invalidates a real checkpoint. Kinematic XML/TCP/mapping changes are tracked
separately and do not invalidate the joint-space dynamics model.

## Promotion sequence

Run each stage under physical supervision. No stage promotes itself.

1. Validate the local configuration and independent E-stop.
2. Benchmark static one-read/one-write I/O:

   ```bash
   conda run -n lerobot python dynamics_modeling/scripts/benchmark_so101_io.py \
     --hardware-config configs/hardware/so101_follower.local.yaml \
     --ticks 900 --enable-hardware --operator-supported-shutdown
   ```

   Accept only after control-path execution P99 has a measured guard (initial
   target: under 25 ms), wake lateness P99 is under 2 ms, deadline misses are
   under 1%, and skipped ticks are under 0.1%. Period P99 is reported, not used
   as an exact 33.333 ms upper bound.
3. Run direct control at 2 degrees, then at most 3 degrees:

   ```bash
   conda run -n lerobot python scripts/run_real_direct_control.py \
     --hardware-config configs/hardware/so101_follower.local.yaml \
     --amplitude-deg 2 --seconds 10 --enable-motion --operator-supported-shutdown
   ```
4. Collect Model-A excitation sessions. Twelve sessions should be split 8/2/2
   by `split_group_ids`, with motion modes represented in every split.
5. Validate the NPZ v2 file, then train the fixed nominal-30-Hz model:

   ```bash
   conda run -n lerobot python dynamics_modeling/scripts/validate_real_dataset.py data.npz
   conda run -n lerobot python dynamics_modeling/scripts/train_dynamics.py \
     --data_path data.npz --robot_config configs/robots/so101.yaml \
     --real_defaults --test_group_ids TEST_GROUP_1,TEST_GROUP_2 \
     --epochs 100 --save_dir outputs/so101_model_a
   conda run -n lerobot python dynamics_modeling/scripts/evaluate_real_model.py \
     --dataset data.npz --checkpoint outputs/so101_model_a/best_rollout_model.pt \
     --normalizer outputs/so101_model_a/normalizer.pt --test-group-ids TEST_GROUP_1,TEST_GROUP_2
   ```
6. Run shadow MPC for at least 2,000 planner publications. Calibrate packet age
   from the state midpoint timestamp; with fewer samples use P99 plus a 5 ms
   guard instead of P99.5.
   Save the calibration JSON with `scripts/calibrate_real_delay.py --output`;
   active mode requires at least 2,000 samples and measured late-drop and packet
   expiry rates below 1%.
7. Promote manually to active only with residual authority at most 2 degrees,
   feedback at most 0.5 degrees, `kdq=0`, and the nominal envelope at most 3
   degrees. Active smoke data is stored separately and is not used to retrain
   Model A.

   Active mode also requires `--ood-envelope` produced by
   `scripts/calibrate_real_ood.py`. All executed-history, selected-action, and
   selected predicted-state coverage values must be at least 99%; an online
   violation suppresses the residual and executes nominal.

The nominal MJCF in `dynamics_modeling/robots/so101` is for kinematics and
contract tests only. It is not a collision-complete safety model or the real
plant.
