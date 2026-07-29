# Model-A Robustness Workflow

Robustness conditions vary payload, actuator gain, force pulse, and observation
noise. Each run must retain robot, model, normalizer, reference, and checkpoint
identity in its manifest. Use the paper workflow for frozen, paired evaluation;
do not compare unmatched rollouts as independent evidence.

Published aggregate statistics are in
[`evidence/robio2026/robustness_and_timing`](../../evidence/robio2026/robustness_and_timing).
