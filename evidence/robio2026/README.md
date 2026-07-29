# ROBIO 2026 public evidence

This directory contains the compact evidence supporting the reported results
for activation-aligned asynchronous residual MPC. It is intended for claim
checking and reanalysis of released aggregate evidence without committing hundreds of megabytes of runtime
caches to Git history.

## Contents

- `abb/summaries`: nominal endpoints, mechanism ablations, delay sweeps,
  projection/task-cost comparisons, paired statistics, and IK comparisons.
- `abb/model_validation`: held-out GRU validation, including the
  complete-history window analysis used in Table I, historical padded-history
  startup diagnostics retained for audit, and formal MPC replay diagnostics.
- `abb/figures`: compact diagnostic and representative-result figures.
- `abb/manifests`: frozen analysis/control manifests and evidence inventory.
- `abb/calibration`: the selected ABB latency calibration that fixes $D=6$.
- `abb/audit`: frozen-cohort completeness, duplicate, configuration, and run
  audits; the empty issue tables correspond to the zero counts in the summary.
- `ur5e`: nominal robot-specific replication, complete-history model-validation
  summaries, paired bootstrap results, configuration manifest, and freeze audit.
- `ur5e/calibration`: the selected UR5e latency calibration that fixes $D=7$.
- `ur5e/contracts`: machine-readable UR5e robot contract and four-reference
  generation manifest.
- `claim_evidence`: recurrent-history isolation, tracking--effort Pareto,
  candidate-ranking/realized-regret, peak-effort summaries, the historical
  claim-evidence test log, and the final full-regression record.
- `robustness_and_timing`: single-factor perturbation, threaded-versus-virtual,
  wall-clock timing, and `ik_baselines` paired statistics versus Projected and
  Preview IK.
- `PUBLIC_MANIFEST.json`: source path, byte size, and SHA-256 for every public
  artifact in this bundle.

The detailed interpretation and evidence boundaries are documented in
[`docs/Paper_test/15_core_claim_evidence_tests.md`](../../docs/Paper_test/15_core_claim_evidence_tests.md).
The conference-paper implementation details, full controller configuration,
and compressed diagnostics are collected in
[`technical_supplement.pdf`](technical_supplement.pdf), with its auditable
LaTeX source in [`technical_supplement.tex`](technical_supplement.tex).

## Deliberate exclusions

Raw rollouts, cache files, candidate snapshot arrays, model checkpoints,
normalizers, and the paper source/PDF are not included. They are unnecessary
for checking the aggregate claims and would make the repository unnecessarily
large. The public copies of legacy manifests replace the local repository
prefix with `.`; numerical values, hashes, configuration fields, and source
commit identifiers are otherwise unchanged.

## Evidence boundaries

- ABB is the primary robot; robustness and mechanism-isolation experiments are
  confined to ABB.
- UR5e is an independently trained and calibrated, nominal within-robot
  replication. It is not a shared-model transfer experiment.
- In the UR5e manifest, `base_run_args` are retained workflow arguments from
  before RobotSpec resolution because they are part of the frozen fingerprints.
  The executed UR5e XML, TCP, home configuration, limits, and formal per-method
  delays are in `resolved_robot_contract` and `formal_case_delay_steps`.
- The history-only ablation does not establish an independent nominal tracking
  gain from advancing recurrent history.
- The public calibration payloads record the guard-adjusted latency rule,
  samples, selected delay, and input hashes. The UR5e contract/reference files
  are explicit audit aids; the resolved contract in the frozen paper manifest
  remains the authoritative formal-case record.
- The baseline robustness files are aggregate and paired statistics, not raw
  rollouts. They support claim checking under the stated perturbations but do
  not add a new robustness guarantee.
- Table I model errors are an offline reanalysis of saved held-out rollouts.
  Every reported window begins after 16 ground-truth recurrent tokens; no data
  collection, training, MuJoCo rollout, or closed-loop controller result was
  rerun. The original padded-history startup diagnostics remain public only for
  audit.
- The effort sweep was performed after freezing the primary configuration,
  uses the same four-trajectory, five-seed FullVirtual matrix at every scale,
  and was not used to replace the matched primary evidence.
- Candidate-ranking diagnostics use retained, projection-active candidates.
- Results are from MuJoCo; they do not establish physical-robot performance or
  hard-real-time guarantees.

## Rebuilding this directory

From the repository root, regenerate the two derived model-validation outputs
before publishing the bundle:

```bash
conda run -n pendulum-rl python scripts/paper_experiments/workflow.py \
  reanalyze-model-validation --overwrite
conda run -n pendulum-rl python scripts/paper_experiments/ur5e_workflow.py \
  reanalyze-model-validation --overwrite
python3 scripts/paper_experiments/publish_evidence.py
```

The build is deterministic for a fixed set of local frozen outputs.
