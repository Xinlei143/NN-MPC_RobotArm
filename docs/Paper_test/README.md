# ROBIO Evidence Reports

These reports explain the compact data published in
[`evidence/robio2026`](../../evidence/robio2026/README.md). They are results
and audit records, not the source of current controller defaults.

The original formal outputs are under `outputs/paper_delay_aware_two_stage_v2`;
the compact freeze is under `outputs/paper_final`; and the threaded revision
evidence is under `outputs/paper_revision_v1`.

## Completed evidence

| Item | Result |
| --- | --- |
| Historical evidence audit | 720 MPC and 540 IK rollouts; passed |
| Regression suite | 143 passed, 12 subtests passed |
| Delay calibration | Four groups of 500 plans; D=6 in each group |
| GRU held-out validation | 20 rollouts; ten each at action std. 0.5 and 0.8 |
| Core ablation | 80 cases |
| Final-pool task cost | 24 cases |
| Four-stage delay sweep | 96 cases |
| Projection choice | 36 cases |
| New controller rollouts | 198 |
| MPC--IK pairing | 2,160 pairs; 10,000 hierarchical bootstrap resamples |

## Main conclusions

- FullVirtual reduced pooled TCP RMSE by 39.55 mm relative to NaiveDelayed
  (95% CI [36.87, 42.10] mm; 58.1%).
- Removing future alignment increased TCP RMSE by 2.82 mm; removing reanchoring
  increased it by 654.60 mm. Fast-feedback removal changed it by -0.95 mm with
  a confidence interval crossing zero.
- Exact task-space reranking improved threaded deployed TCP RMSE by 3.05 mm
  (95% CI [0.96, 4.32] mm) and increased E2E p95 by about 6.82 ms.
- Two-stage projection passed the pre-registered 5% non-inferiority margin
  against full compiled projection and reduced both tracking error and latency.
- Anchor-only execution became unstable for D >= 6. Reanchoring is therefore
  necessary to deploy future alignment with absolute joint references.

| Reports | Contents |
| --- | --- |
| 00–02 | Evidence audit, freeze/regression checks, and GRU validation |
| 03–08 | Mechanism isolation, task-cost, delay, projection, endpoint, and timing analyses |
| 09–13 | Evidence freeze, threaded replication, rerun, real-time, and robustness results |
| 14 | UR5e independently trained and calibrated replication |
| 15 | History alignment, effort sensitivity, and candidate-ranking diagnostics |
| 16 | Submission preflight |
| 17 | Final publication preflight and release checklist |
| 18 | GRU true-history window reanalysis of Table I metrics |

The primary result is activation alignment plus execution-time reanchoring.
The reports preserve the MuJoCo-only scope, the history-ablation limitation,
and the tracking-effort trade-off.
