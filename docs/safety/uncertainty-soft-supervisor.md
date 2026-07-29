# Uncertainty Soft Supervisor

The optional supervisor is disabled by default. It evaluates disagreement among
compatible learned-model replicas on the CEM-selected command sequence; replicas
do not optimize CEM and do not average their predictions into the command.

`ensemble_monitor` records disagreement only. `ensemble_soft_gate` uses
hysteresis to move between normal, suspected, limited, and nominal-fallback
states. High disagreement combined with predicted physical risk, nonfinite
results, or an exhausted replica-computation budget triggers the conservative
nominal fallback.

Thresholds depend on the normalizer, model family, replica training protocol,
and uncertainty horizon. This is a MuJoCo safety monitor, not a certified
hardware safety system.
