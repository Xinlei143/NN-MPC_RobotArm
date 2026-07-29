# GRU true-history window reanalysis — 2026-07-29

## Purpose

This analysis checked whether the non-monotone ABB velocity values in manuscript
Table I arose from the frozen model or from the original held-out validation
protocol. It is now the formal Table I validation workflow for ABB and UR5e.
It uses only saved evaluation rollouts, frozen checkpoints, and normalizers; no
data collection, training, MuJoCo rollout, or controller result was rerun.

## Source and protocol

The ABB source is `outputs/paper_final/diagnostics/gru_validation`; the UR5e
source is `outputs/paper_ur5e_v2/diagnostics/gru_validation`. Each contains:

- a frozen robot-specific GRU checkpoint and normalizer;
- 20 held-out 200-step rollouts generated with seed 20260730;
- ten rollouts at each of $\sigma_u=0.5$ and $0.8$.

The original Table I starts at step zero with `warmup_steps=0`. A GRU with
history length 16 therefore receives a history padded from its first token.
The original h=1 result consequently measures the first command transient
after the 50-step settle phase, not a one-step prediction with complete
ground-truth recurrent context.

The reanalysis evaluates common non-overlapping windows beginning at steps
16, 36, 56, 76, 96, 116, 136, 156, and 176. Every window has 16 preceding
ground-truth state/action tokens and fits all reported horizons. For each
rollout and horizon, equal-length windows are pooled by RMS; estimates below
are means across the ten rollouts in each excitation group.

Run the formal wrappers:

```bash
/home/xinlei/Data/robotics_ws/miniconda3/envs/pendulum-rl/bin/python \
  scripts/paper_experiments/workflow.py reanalyze-model-validation --overwrite
/home/xinlei/Data/robotics_ws/miniconda3/envs/pendulum-rl/bin/python \
  scripts/paper_experiments/ur5e_workflow.py reanalyze-model-validation --overwrite
```

The outputs are `outputs/paper_model_validation_history_windows_v1` and
`outputs/paper_ur5e_history_windows_v1`; each contains `window_metrics.csv`,
`rollout_metrics.csv`, `aggregate.csv`, and an input-hash manifest.

## Results

| $\sigma_u$ | Horizon | $q$ RMSE (rad) | $\dot q$ RMSE (rad/s) | Divergence |
|---:|---:|---:|---:|---:|
| 0.5 | 1 | 0.000670 | 0.001990 | 0% |
| 0.5 | 5 | 0.002016 | 0.012236 | 0% |
| 0.5 | 10 | 0.003354 | 0.025956 | 0% |
| 0.5 | 20 | 0.004852 | 0.034472 | 0% |
| 0.8 | 1 | 0.001069 | 0.005570 | 0% |
| 0.8 | 5 | 0.003226 | 0.017878 | 0% |
| 0.8 | 10 | 0.005038 | 0.036071 | 0% |
| 0.8 | 20 | 0.006876 | 0.051398 | 0% |

| Robot | $\sigma_u$ | Horizon | $q$ RMSE (rad) | $\dot q$ RMSE (rad/s) | Divergence |
|---|---:|---:|---:|---:|---:|
| UR5e | 0.5 | 1 | 0.000132 | 0.000465 | 0% |
| UR5e | 0.5 | 20 | 0.001334 | 0.002745 | 0% |
| UR5e | 0.8 | 1 | 0.000160 | 0.000532 | 0% |
| UR5e | 0.8 | 20 | 0.001436 | 0.003023 | 0% |

For comparison, the original $\sigma_u=0.5$ start-of-rollout values in Table I
were $q/\dot q$ RMSE $=0.003366/0.558419$ at $h=1$ and
$0.021644/0.264904$ at pooled $h=20$. The original values are reproduced by
the frozen CSV and are not a transcription or aggregation error. Their
decreasing velocity component is driven by the padded-history startup segment:
the mean $\dot q$ RMSE is 0.558419 at prediction step 1, 0.602173 at step 2,
then declines to 0.046934 at step 10 and 0.112239 at step 20.

## Interpretation and publication decision

The true-history reanalysis restores the expected monotone degradation with
horizon and is the more appropriate model-validation diagnostic for a
history-conditioned planner. It does **not** alter any frozen closed-loop MPC,
threaded, UR5e, robustness, ablation, or effort result.

The reanalysis is exposed through both robot workflows and is published in the
public evidence bundle with its input-hash manifests. Manuscript Table I and
supplementary Table 2 use these complete-history values. The original
padded-history startup CSVs remain public for audit but do not support the
reported model-validation claim.
