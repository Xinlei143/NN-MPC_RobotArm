# Final Evidence Freeze

`outputs/paper_final/` is a compact frozen evidence directory built from the
schema-v6 control manifest, calibrated runs, summaries, bootstrap statistics,
GRU validation, MPC-IK pairing, and figures. Large rollouts are not copied;
the manifest records source paths, SHA-256 values, and sizes.

Tables must be derived from the frozen summaries. Historical cohorts remain
explicitly marked as provenance-unverified. The freeze also retains negative
results: no independent nominal feedback gain and anchor-only instability at
longer delays.
