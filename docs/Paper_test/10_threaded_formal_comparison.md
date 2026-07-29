# ThreadedASAP Formal Comparison

ThreadedASAP reproduced the nominal tracking level of FullVirtual: the matched
TCP-RMSE difference was +0.054 mm (95% paired-bootstrap CI [-1.094, +1.303]
mm). Across the 36 robustness clusters, ThreadedASAP reduced TCP RMSE by
12.38 mm versus Projected Direct IK and by 6.75 mm versus Preview IK.

This supports nominal and aggregate replication, not exact equivalence between
threaded and virtual schedules or superiority over every baseline in every
perturbation condition.
