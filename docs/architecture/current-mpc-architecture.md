# Current Residual CEM-MPC Architecture

## Scope

The controller is evaluated in MuJoCo on position-controlled ABB IRB 2400 and
UR5e manipulators. It does not establish real-robot performance, hardware
torque feasibility, or hard real-time scheduling.

## Control interface

The actuator interface consumes absolute joint references. At each 10 ms tick,
the execution layer applies a projected `q_ref`; it is not a velocity or torque
controller. A continuous DLS IK solver supplies the evolving nominal
`q_nom(t)`, and residual MPC selects bounded corrections around that nominal.

```text
task reference -> continuous IK -> q_nom(t)
state + command history -> GRU rollout -> CEM residual sequence r[0:H)
q_nom(t) + r[a] -> execution-time projection -> q_ref(t)
```

The default learned model is a history-conditioned GRU predicting closed-loop
actuator-reference dynamics. Formal paper settings use history length 16,
horizon 20, 128 candidates, and two CEM iterations.

## Delay-aware semantics

Planning and execution are asynchronous. A plan launched at `t_i` may activate
at `t_a`, after the plant, command history, and IK nominal have advanced. The
full protocol therefore:

1. forecasts the activation state;
2. advances recurrent context and the reference window consistently to that
   activation time;
3. publishes a timestamped residual packet;
4. indexes the residual by packet age; and
5. reanchors it to the current IK nominal immediately before execution.

`virtual_asap` is the deterministic fixed-delay emulator used for controlled
ablations. `threaded_asap` is the wall-clock MuJoCo mode: the CPU execution loop
runs at nominal 100 Hz while a CUDA worker solves CEM plans as soon as it can.
The latter requires the full delay protocol and CUDA.

## Projection and recovery

The planner first projects candidate sequences to respect the configured joint,
velocity, acceleration, and braking-aware constraints. The execution loop then
applies the shared physical projection again to the selected command. The
two-stage compiled planner projection is the frozen paper configuration.

If planning fails, the runner executes the nominal and resets the CEM warm
start. Direct IK is a separate baseline that does not load a learned model or
run CEM.

## Evidence boundaries

The public evidence bundle reports primary tracking results, effort statistics,
timing, robustness, mechanism isolation, and UR5e replication. The evidence
supports the full activation-alignment and execution-time-reanchoring protocol.
It does not isolate an independent nominal-tracking gain from advancing the
recurrent history alone, nor an independent gain from fast feedback or
task-space reranking.
