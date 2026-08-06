# SO101 fault-state policy

`RUNNING` permits the selected direct/shadow/active mode. `HOLDING` transmits a
safe hold target and pauses planner snapshots. `FAULT_LATCHED` clears packets,
rejects late GPU results through `history_generation`, and requires operator
intervention. `TORQUE_DISABLED` is terminal for the run.

A single 10--25% period deviation invalidates the transition, holds the command,
and skips that recurrent token. Two consecutive deviations rebuild estimator
and history. Read failure, non-monotonic time, skipped ticks, `dt > 1.5 Ts`, or
wake lateness over `0.5 Ts` rebuilds immediately. Missed absolute deadlines
jump to the next future deadline; the runtime never performs catch-up bursts.

State validity requires all six motors, finite values, calibrated raw ranges,
the software envelope plus 0.05 rad tolerance, monotonic timestamps, plausible
read duration, no encoder jump/wrap, and plausible measured speed. Command,
plausibility, and emergency velocity limits are separate concepts.
