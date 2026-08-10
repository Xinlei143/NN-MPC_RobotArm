# ROBIO 2026 v2 release preparation

This checklist freezes the repository for the `robio2026-submission-v2`
release. It does not publish the GitHub Release page.

## Scope

The v2 paper adds the physical SO101 validation while preserving the original
ABB/UR5e MuJoCo mechanism study. The SO101 result is a hardware-specific
instantiation of the residual MPC principle, not zero-shot transfer of the
simulation hyperparameters.

## Contents of the tagged repository

- `evidence/robio2026/PUBLIC_MANIFEST.json` and
  `evidence/robio2026/technical_supplement.pdf`;
- `evidence/robio2026/so101/analysis/public_summary.json` and
  `public_trial_ledger.csv`;
- `evidence/robio2026/so101/delay_stress/` with the v2 protocol, pooled timing
  calibration/OOD records, six-run paired summary, and sanitized result note;
- `evidence/robio2026/so101/figures/fig2_representative_tracking.{pdf,svg,png}`
  and its source manifest;
- `configs/experiments/so101_final_paper_20260810.yaml`, the offline analyzer,
  and the figure-generation script.

The GitHub Release is intentionally created without manually uploaded assets.
The manuscript source and final PDF remain local submission materials because
`Paper/` is excluded by `.gitignore`; the tagged repository therefore contains
the code, compact evidence, protocol files, and technical supplement only.

## Pre-publish checks

1. Re-run the offline SO101 analyzer and verify 63/63 complete trials, 54
   held-out trials, 18/18 paired wins against both baselines, 20,956 planner
   events, and zero deadline misses, planner failures, and runtime safety
   violations.
2. Re-run the offline delay-stress analyzer and verify six complete runs,
   three paired delay-worse blocks, pooled p95 latency of 28.33/94.34 ms for
   baseline/delay33, and zero delay-stress safety/deadline counters.
3. Rebuild the compact evidence bundle and verify that all public text and SVG
   files contain portable paths only; raw rollouts, checkpoints, normalizers,
   and hardware-local runtime caches remain excluded.
4. Compile the 8-page main paper and 4-page technical supplement, check the
   page budget and undefined references, and visually inspect all pages and
   Fig. 2 at publication scale.
5. Confirm that the manuscript and README describe TCP as FK-derived secondary
   evidence and do not claim hard-real-time guarantees or zero-shot transfer.
6. Record the final commit, manuscript PDF, supplement PDF, and manifest
   SHA-256 values in the GitHub Release notes. The paper footnote must target
   the immutable URL
   `releases/tag/robio2026-submission-v2`, not `tree/main`.

## Known boundary

The current bundle supports aggregate checking and reanalysis, not a
from-scratch reproduction of the physical runs. Broader physical-platform
replication, external Cartesian metrology, and hard-real-time certification
remain future work.
