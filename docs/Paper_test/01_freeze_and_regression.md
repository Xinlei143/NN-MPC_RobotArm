# Freeze and Regression Checks

The control and analysis provenance was frozen before the final evidence run.
Schema-v6 fixes full residual MPC, lowest-cost execution, batch size 128,
0.01 s control steps, zero preview nominal, and stage-one task-space scoring
and compilation disabled. The suite also fixes the projection non-inferiority
margin at a 5% TCP-RMSE degradation.

Regression result:

```text
130 passed, 1 skipped, 12 subtests passed
```

The skipped test is optional by design. A GPU workflow smoke test also passed.
