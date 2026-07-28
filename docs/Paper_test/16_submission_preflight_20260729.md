# ROBIO 2026 submission preflight — 2026-07-29

## Scope

This check covers the current manuscript at `Paper/robo2026/main.tex` and
`Paper/robo2026/main.pdf`, the frozen ABB/UR5e/claim-evidence outputs, the public
repository URL used by the paper, and the ROBIO 2026 information publicly
available on the conference website as of 2026-07-29.

## Manuscript changes completed

- The title is now **“Activation-Aligned Asynchronous Residual MPC for
  Position-Controlled Manipulators with Learned Dynamics.”**
- The abstract foregrounds the stronger Projected IK comparison:
  - ABB: 39.26 to 28.43 mm;
  - UR5e: 15.06 to 10.42 mm;
  - the direction is consistent over all four evaluated trajectories;
  - the higher ABB command effort is stated explicitly.
- The NaiveDelayed comparison remains, but is identified as a stale-semantics
  comparator rather than the only headline baseline.
- The abstract identifies the experiments as MuJoCo experiments immediately.
- The Introduction, contribution statement, Discussion, and Conclusion use
  the following vocabulary consistently:
  - activation alignment or activation consistency;
  - execution-time reanchoring;
  - absolute position-reference interface;
  - robot-specific replication;
  - wall-clock threaded MuJoCo execution.
- Cross-robot results are not described as transfer, portability, or
  shared-model generalization.
- The Discussion was retained, including the effort trade-off, the absence of
  an isolated recurrent-history benefit, the absence of an isolated fast
  feedback benefit, the missing ABB torque limit, the MuJoCo-only scope, and
  uncertainty-aware future work.

## PDF and LaTeX checks

| Check | Result |
|---|---|
| Clean `latexmk` build | Pass |
| Page count | 8 |
| Page size | A4, 595.276 x 841.890 pt |
| PDF version | 1.7 |
| Embedded fonts | Pass: every font reported by `pdffonts` is embedded and subset |
| Type 3 fonts | None |
| Undefined citations/references | None |
| Overfull boxes | None |
| Bibliography font override | None; IEEEtran bibliography default is used |
| Visual inspection | Title, abstract, equations, subscripts, minus signs, figures, and final page render correctly |

The log contains ordinary Times bold-shape substitutions and several
underfull-box notices. These are cosmetic TeX diagnostics; no missing glyph,
overflow, or unresolved-reference error was found.

IEEE PDF eXpress itself cannot be completed locally because it requires the
conference ID and an author login. Upload the final PDF there after the
submission portal supplies the conference ID; use the returned certified PDF
if the portal requires it.

## Numerical consistency checks

- ABB headline values in the paper match the frozen primary summaries and
  paired analyses:
  - Projected IK 39.26 mm;
  - ThreadedAsync 28.43 mm;
  - FullVirtual--NaiveDelayed difference -39.55 mm (58.1%);
  - ThreadedAsync--FullVirtual difference 0.054 mm.
- UR5e headline values match the frozen UR5e evidence:
  - Projected IK 15.06 mm;
  - NaiveDelayed 34.78 mm;
  - FullVirtual 10.43 mm;
  - ThreadedAsync 10.42 mm;
  - FullVirtual--NaiveDelayed difference -24.36 mm.
- Candidate-ranking values match
  `outputs/paper_claim_evidence_v1/summaries/candidate_ranking.json`:
  - within-snapshot Spearman 0.976;
  - pairwise concordance 0.988;
  - realized relative regret 0.192%;
  - 20-step branch replay joint RMSE 0.889 mrad.
- The post-freeze effort sweep matches
  `outputs/paper_claim_evidence_v1/summaries/effort_pareto_aggregate.csv`.
  The manuscript correctly labels this sweep as diagnostic and does not
  substitute the 2x setting into the frozen primary evidence.
- The UR5e freeze audit passes with all 84 expected cases present and no
  constraint-violation runs.
- The ABB analysis manifest contains 78 artifacts; all local paths, sizes, and
  hashes validate. The referenced commits exist locally.

## References

- 20 bibliography entries are present and all 20 are cited.
- No citation key is missing and no bibliography entry is orphaned.
- The DOI and publication-field audit has been completed; the two conference
  items without retained DOI URLs no longer carry long redundant web links.
- The bibliography uses the IEEEtran style and its default font size.

## Public artifact check — action required

The repository URL in the manuscript is publicly reachable:

`https://github.com/Xinlei143/NN-MPC_RobotArm`

However, the repository's public `main` branch does **not** currently expose
the evidence files named by the manuscript/reproducibility material. The
following URLs returned 404 during this preflight:

- `outputs/paper_final/manifests/analysis_manifest.json`
- `outputs/paper_ur5e_v2/freeze_audit.json`
- `outputs/paper_claim_evidence_v1/claim_manifest.json`
- `docs/Paper_test/15_core_claim_evidence_tests.md`

The local `.gitignore` excludes `/outputs` and `/Paper`, and the evidence
directories are large. Do not commit the entire output trees blindly.
Before submission, choose one of these defensible solutions:

1. publish a compact, path-normalized evidence bundle through a GitHub Release
   or an archival service and change the paper footnote to its persistent URL;
2. commit only compact summaries, manifests, configuration files, hashes, and
   the evidence report, while publishing large rollouts separately;
3. until the artifacts are public, change “Code and frozen evidence” to
   “Code” and remove claims that the public repository already contains the
   frozen evidence.

The current analysis manifest contains absolute local paths, so a released
copy should additionally store repository-relative paths.

## ROBIO 2026 portal checks

The official public website states:

- initial paper deadline: 1 August 2026;
- full-paper length: 6 pages, with at most 2 paid extra pages;
- maximum length: 8 pages.

The public pages inspected did not state whether initial review is single- or
double-blind. Therefore anonymity must be checked in the actual submission
portal or current author instructions before upload. Do not remove or retain
authors based on historical ROBIO practice alone.

The following items require a manual portal-side comparison because they are
not available in the local workspace:

- exact title matches the revised PDF title;
- author order is Xinlei Lin, Chenyu Lin, Zhang Yue;
- emails match the manuscript;
- abstract entered in the portal matches the revised abstract;
- anonymous/non-anonymous mode matches the portal instruction;
- uploaded PDF is the final certified PDF;
- supplementary/repository URL resolves without authentication.

## Submission blockers and remaining manual actions

1. **Resolve the public frozen-evidence 404s or weaken the repository footnote.**
2. **Confirm anonymity in the live submission portal.**
3. Run the final PDF through IEEE PDF eXpress when the conference ID/login is
   available.
4. Synchronize the revised title, author order, emails, and abstract with the
   submission form.
5. Re-open the exact uploaded PDF once from the portal and verify its page
   count, equations, figures, and repository link.

No additional experiment is recommended before submission.
