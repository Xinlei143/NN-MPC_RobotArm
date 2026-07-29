# ROBIO 2026 public evidence

This directory contains the compact evidence supporting the reported results
for activation-aligned asynchronous residual MPC. It is intended for claim
checking and reproduction without committing hundreds of megabytes of runtime
caches to Git history.

## Contents

- `abb/summaries`: nominal endpoints, mechanism ablations, delay sweeps,
  projection/task-cost comparisons, paired statistics, and IK comparisons.
- `abb/model_validation`: held-out GRU validation and formal MPC replay
  diagnostics.
- `abb/figures`: compact diagnostic and representative-result figures.
- `abb/manifests`: frozen analysis/control manifests and evidence inventory.
- `ur5e`: nominal robot-specific replication, paired bootstrap results,
  configuration manifest, and freeze audit.
- `claim_evidence`: recurrent-history isolation, tracking--effort Pareto,
  candidate-ranking/realized-regret, peak-effort summaries, and test log.
- `robustness_and_timing`: single-factor perturbation, threaded-versus-virtual,
  and wall-clock timing statistics.
- `PUBLIC_MANIFEST.json`: source path, byte size, and SHA-256 for every public
  artifact in this bundle.

The detailed interpretation and evidence boundaries are documented in
[`docs/Paper_test/15_core_claim_evidence_tests.md`](../../docs/Paper_test/15_core_claim_evidence_tests.md).

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
- The history-only ablation does not establish an independent nominal tracking
  gain from advancing recurrent history.
- The effort sweep was performed after freezing the primary configuration,
  uses the same four-trajectory, five-seed FullVirtual matrix at every scale,
  and was not used to replace the matched primary evidence.
- Candidate-ranking diagnostics use retained, projection-active candidates.
- Results are from MuJoCo; they do not establish physical-robot performance or
  hard-real-time guarantees.

## Rebuilding this directory

From the repository root:

```bash
python3 scripts/paper_experiments/publish_evidence.py
```

The build is deterministic for a fixed set of local frozen outputs.
