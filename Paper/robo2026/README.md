# ROBIO 2026 LaTeX scaffold

The manuscript source is [main.tex](main.tex). It uses the `conference` option of IEEEtran. The core bibliography is in [references.bib](references.bib); experimental values and conclusions remain visible placeholders until the benchmark is complete.

## One-sentence argument

When learned CEM planning is slower than the command period, future activation alignment and execution-time residual reanchoring recover stale-plan degradation under an absolute-position interface. Feedback, reranking, and projection are reported as auxiliary mechanisms or deployment trade-offs.

## Terminology ledger

| Canonical term | Definition | Avoid |
|---|---|---|
| Raw Direct IK | Continuous IK nominal dispatched without the shared velocity/acceleration projector; diagnostic baseline only. | practical fallback |
| Projected Direct IK | Continuous IK nominal dispatched through the shared physical projector; zero correction is this practical fallback. | raw projection bypass |
| Preview IK | One preview calibrated independently and frozen across all test trajectories. | per-trajectory tuned preview |
| `q_ref` / `q_cmd` | Absolute position-actuator command actually sent to MuJoCo. | torque, action increment |
| clipped IK nominal | Current IK reference clipped to joint bounds; not planner-side velocity/acceleration projected. | fully executable nominal |
| requested residual | CEM residual plus bounded fast feedback before execution projection. | executed actuator correction |
| executed residual | Actual command minus the current clipped IK nominal after the execution rule. | raw CEM residual |
| ideal zero-delay logical MPC | MPC result applied without advancing simulated time; a logical upper bound. | real-time synchronous MPC |
| naive delayed/cached MPC | Launch-state plan applied after delay without current-nominal reconciliation. | low-rate ASAP |
| virtual delay-aware MPC | Deterministic fixed-delay emulator. | threaded online controller |
| threaded asynchronous MPC | CUDA planner worker plus measured 100 Hz Python soft-real-time loop. | hard real-time MPC |
| CEM solve latency | Time inside `controller.plan()`. | total planner latency |
| planner end-to-end latency | Snapshot launch to packet publication. | CEM solve latency |

## Evidence checklist

Every `\datareq{...}` marker must be resolved or its enclosing claim removed before submission. The 16 core references in `references.bib` support the position-interface, learned-dynamics, CEM/residual, delay-aware MPC, IK, and MuJoCo background claims.

| Claim | Required evidence |
|---|---|
| Delay-aware execution mitigates stale plans | Pre-specified FullVirtual--NaiveDelayed full-trajectory TCP RMSE paired difference on four trajectories and five paired CEM seeds; trajectory--seed bootstrap 95% CI. |
| Full remains practically competitive | Raw, Projected, and Preview IK secondary comparisons; Direct IK is deterministic and is not replicated across fake seeds. |
| The gain comes from timing alignment and reanchoring | Frozen 80-case matrix; feedback is auxiliary because its removal CI crosses zero. |
| The behavior is robust to planner delay | Virtual sweep at 20, 40, 60, and 80 ms for naive, anchor-only, reanchored, and full variants. |
| The execution layer remains viable | CEM solve and end-to-end p50/p95/p99/max, actual planner Hz, late/expired packets, fallback duty cycle, control period/jitter, deadline misses, and worst seed. |
| Tracking is not bought with aggressive control | Command acceleration RMS, actuator torque RMS, requested/executed residual discrepancy, saturation, and constraint violations. |
| The model supports finite-horizon rollout | One-step and 5/10/20-step q/dq RMSE, per-joint NMSE or R2, amplitude ratio, divergence, and train/evaluation isolation. |
| The model and budget are frozen | One GRU checkpoint and normalizer, history length 16, hidden size 256, CEM/cost/constraint configuration, hardware, and repository commit. |

Model C is not a paper contribution or result section. A one-sentence negative result may remain in Discussion only if a complete common-controller benchmark supports it.

## Required experimental protocol

1. Select one GRU/CEM configuration on a separate calibration trajectory, record it in a unique paper configuration/manifest, and freeze it.
2. Run each deterministic Raw/Projected/Preview IK baseline once under the fixed initial condition. Run all MPC methods with the same five paired CEM seeds.
3. Preserve the same reference, initial-state protocol, model, cost, constraints, and CEM budget unless a row explicitly ablates one component.
4. Treat virtual delay-aware MPC as an emulator and threaded asynchronous MPC as the final wall-clock system.
5. Report planner end-to-end latency, rather than CEM solve time, when deciding whether a packet meets its activation deadline.
6. Treat the three matched component contrasts as non-additive mechanism-isolation evidence, not causal identification.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For validation without updating generated files in the repository:

```bash
latexmk -pdf -outdir=/tmp/robo2026-latex -interaction=nonstopmode -halt-on-error main.tex
```

`latexmk` runs BibTeX automatically when `references.bib` changes. The target is a six-page IEEE conference paper after real figures and values replace the remaining scaffold markers. Recheck page count after those assets are inserted.
