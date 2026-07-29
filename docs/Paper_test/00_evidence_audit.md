# Evidence Audit

Formal audit output: `outputs/paper_evidence_audit/`.

| Check | Result |
| --- | ---: |
| MPC rollouts | 720 |
| IK rollouts | 540 |
| Missing formal cases | 0 |
| Conflicting duplicates | 0 |
| Nonfinite primary metrics | 0 |
| Configuration mismatches | 0 |

The audit passed. Checkpoint, normalizer, references, XML, and target
configuration were compatible. Historical rollouts are labelled
`historical_compatible`: their configuration and semantics match, but their
clean/dirty Git state was not recorded. New evidence uses schema-v6 and a
frozen commit.
