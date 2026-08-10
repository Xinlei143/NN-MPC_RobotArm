# SO101 ThreadedAsync injected-delay validation (historical v1 runbook)

> This is the original v1 runbook. The accepted v2 calibration and active
> results are recorded in
> [`so101-threaded-delay-stress-results-20260810.md`](so101-threaded-delay-stress-results-20260810.md).
> The v1 calibration did not pass the active gate and must not be mixed with v2.

This is an isolated physical stress study for the temporal execution path. It
uses only the existing `threaded_asap` / activation-aligned NN-MPC controller;
it does not add a `naive_delayed` controller, accept late packets, or change
the residual authority. The formal 63-trial paper output is never modified.

Before any motion, the real runner performs a torque-disabled planner
preflight through a read-only SO101 connection. The same CUDA/CEM worker must
reach `ready` before the production connection is opened and the arm is sent
to home. Initialization stages are printed with a bounded timeout; on a
failure, hardware is closed before waiting for any remaining CUDA cleanup.
This is a safety/order guard only and does not change the controller or delay
condition.

The only runtime perturbation is a deterministic 33 ms wall-clock sleep after
CEM search, executable projection, and packet post-processing, immediately
before packet publication. The packet still uses the canonical activation
guard: if the publication misses its calibrated deadline it is dropped and
the nominal command remains in charge.

## Frozen parameters

All controller settings are inherited from
`configs/experiments/so101_final_paper_20260810.yaml`:

- 30 Hz, five controlled joints, history 16, `H=6`;
- 128 samples, two CEM iterations, the formal repeat seeds;
- ±1° residual authority, `J=C_q`, compiled two-stage executable projection;
- the same checkpoint, normalizer, hardware configuration, joint references,
  startup gates, velocity/acceleration/braking limits, and safety handling.

The baseline condition uses the original calibration (`D=2`) and OOD envelope.
The injected condition uses a new calibration and OOD envelope produced from
the injected-delay shadow runs. The new `D` is derived from measured p99.5
latency; it must not be edited by hand. The historical v1 pass did not satisfy
the active gate; use the v2 protocol and result record for the accepted run.

## 1. Injected-delay shadow calibration

Run the three frozen calibration references under physical supervision. Each
run records at least the same planner, packet, OOD, and safety fields as the
formal shadow protocol:

```bash
conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage shadow --shadow-label circle_p0 --dry-run
conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage shadow --shadow-label figure8_p0 --dry-run
conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage shadow --shadow-label circle_p1 --dry-run
```

After recovering/restarting the workstation, the planner can be tested once
without enabling torque or commanding motion:

```bash
conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage shadow --shadow-label circle_p0 --preflight-only
```

Only continue to the physical shadow run if this preflight reaches `ready`
and exits cleanly. The resulting `__preflight` directory is a diagnostic
artifact and must not be merged as a trial.

After the dry-run commands have been inspected, run each command again with
the printed confirmation token and operator identity. The runner writes to
`outputs/hardware/so101_delay_stress/20260810_threaded_asap/shadow_delay33/`.

Merge only these three injected-delay rollouts and require the recorded delay
metadata to be 33 ms in every file:

```bash
conda run -n lerobot python scripts/assemble_real_mpc_evidence.py \
  outputs/hardware/so101_delay_stress/20260810_threaded_asap/shadow_delay33/\
shadow__circle_p0__delay33ms/rollout.npz \
  outputs/hardware/so101_delay_stress/20260810_threaded_asap/shadow_delay33/\
shadow__figure8_p0__delay33ms/rollout.npz \
  outputs/hardware/so101_delay_stress/20260810_threaded_asap/shadow_delay33/\
shadow__circle_p1__delay33ms/rollout.npz \
  --expected-injected-planner-delay-ms 33 \
  --output-dir outputs/hardware/so101_delay_stress/20260810_threaded_asap/calibration
```

Then derive the new timing calibration and OOD envelope:

```bash
conda run -n lerobot python scripts/calibrate_real_delay.py \
  outputs/hardware/so101_delay_stress/20260810_threaded_asap/calibration/merged_shadow.npz \
  --control-dt 0.03333333333333333 --guard-ms 5 \
  --injected-planner-delay-ms 33 \
  --output outputs/hardware/so101_delay_stress/20260810_threaded_asap/delay_calibration_33ms.json

conda run -n lerobot python scripts/calibrate_real_ood.py \
  outputs/hardware/so101_delay_stress/20260810_threaded_asap/calibration/OOD_TOKENS.npz \
  --output outputs/hardware/so101_delay_stress/20260810_threaded_asap/ood_envelope_33ms.json
```

The active gate remains unchanged: at least 2,000 finite latency samples,
`method=p99.5`, late-drop and packet-expiry rates below 1%, and all three OOD
coverages at least 99%. If the measured calibration produces `D>=4`, stop and
inspect timing rather than silently extending the reference or changing CEM
settings.

## 2. Six paired active runs

Run the same three held-out `ellipse_fast` phases under both conditions:

```text
ellipse_fast_p0: baseline, delay33
ellipse_fast_p1: delay33, baseline
ellipse_fast_p2: baseline, delay33
```

The order is counterbalanced across phases. For each run use the exact command
printed by the isolated runner and confirm its token:

```bash
conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage active --condition baseline \
  --trial-id ellipse_fast_p0_nn_mpc --dry-run

conda run -n lerobot python scripts/run_so101_delay_stress_trial.py \
  --stage active --condition delay33 \
  --trial-id ellipse_fast_p0_nn_mpc --dry-run
```

Use `p1` and `p2` analogously. Each run is stored in a unique directory under
the isolated active root, with command, protocol, checkpoint/normalizer,
reference, calibration, OOD, and hardware identities hashed in its manifest.
No retry may replace a technically complete run; any retry receives an explicit
`--retry-index` and remains visible in the ledger.

## Interpretation boundary

This six-run paired study tests whether the physical ThreadedAsync deployment
remains executable and retains tracking benefit when planner latency is forced
to span an additional control period. It is not a new `NaiveDelayed` ablation,
and therefore does not by itself identify the causal advantage of alignment
over stale-plan semantics on hardware. The MuJoCo delay/reanchoring ablations
remain the mechanism-isolation evidence; this experiment is hardware stress
confirmation plus deployability evidence.
