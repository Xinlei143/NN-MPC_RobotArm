# Final full-regression record — 2026-07-29

## Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl pytest -q mpc/tests
```

## Result

The release-candidate run passed with **147 tests**, **2 warnings**, and
**12 subtests** in 2.48 s. The warnings are two upstream PyTorch
`torch.jit.script` deprecation warnings from
`test_compiled_projection_matches_eager_at_h20`; no project test failed.

## Scope

This is the full `mpc/tests` regression result used by the final publication
preflight and release notes. It supersedes older 143-test documentation.

`evidence/robio2026/claim_evidence/manifests/unit_tests.log` is retained as a
historical 36-test claim-evidence subset. It is not a substitute for this
full-regression record and should not be relabelled as a 147-test run.
