# Final ROBIO 2026 publication preflight — 2026-07-29

## Scope

This report records the final local preflight for the conference manuscript,
technical supplement, and compact public evidence bundle. It supersedes the
historical preflight in
[`16_submission_preflight_20260729.md`](16_submission_preflight_20260729.md).

## Repository release record

The final submission must be tagged `robio2026-submission-v1` after this
preflight commit is pushed. The associated GitHub Release should record the
tagged commit SHA, the manuscript PDF SHA-256, the technical-supplement PDF
SHA-256, the SHA-256 of `PUBLIC_MANIFEST.json`, the manifest artifact count,
the frozen environment summaries, the manuscript title, and submission date.
Attach the manuscript PDF, technical supplement PDF, and public manifest (or
link them directly in the release notes). The manuscript repository footnote
must target the resulting immutable release URL rather than `main`.

## Required final checks

| Check | Expected result |
|---|---|
| Manuscript compilation | Successful `latexmk` build; 7 pages; A4 PDF |
| Supplement compilation | Successful `latexmk` build; 2 pages; A4 PDF |
| PDF fonts | All fonts embedded; no Type 3 fonts |
| LaTeX integrity | No undefined citations/references; final PDF visually checked for clipping or overlap |
| Evidence manifest | Every listed artifact matches its SHA-256 digest |
| Public evidence | Manifest, supplement source/PDF, UR5e manifest, and evidence report return HTTP 200 |
| UR5e freeze audit | 84 expected and observed cases; zero fingerprint, input-hash, or constraint mismatches |
| Table I source check | Published values match complete-history window aggregate source files |
| Working tree | `git diff --check` passes before creating the tag |

## Local verification record

- The manuscript compiles as a seven-page A4 PDF; the technical supplement
  compiles as a two-page A4 PDF.
- Both PDFs use embedded Type 1 fonts, with no Type 3 fonts or undefined
  references. The final PDF was visually checked for clipping and overlap.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl pytest -q mpc/tests`
  passed with 147 tests and 12 subtests; the only warnings were two upstream
  PyTorch `torch.jit.script` deprecation warnings.
- The UR5e freeze audit passed with 84 expected/observed cases, zero fingerprint
  mismatches, zero input-hash mismatches, and zero constraint-violation runs.
- Table I's ABB and UR5e $h=1/20$, $\sigma_u=0.5$ values were checked
  against the complete-history window aggregates. The ABB
  values are $q$ RMSE 0.000670/0.004852 and $\dot q$ RMSE 0.001990/0.034472;
  the UR5e values are 0.000132/0.001334 and 0.000465/0.002745 before
  manuscript rounding.

## Manual submission checks

- Confirm the current ROBIO portal's anonymity policy before upload.
- Run the final PDF through IEEE PDF eXpress when the conference ID is
  available and use the certified PDF if required.
- Verify title, authors, affiliations, emails, and abstract against the
  submission form.
- Re-open the uploaded PDF and the fixed release URL without authentication.
