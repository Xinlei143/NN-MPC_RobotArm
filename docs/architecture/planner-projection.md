# Planner Projection

Planner projection maps a requested joint-reference sequence to one that
respects configured joint, velocity, acceleration, and braking-aware limits.
The execution loop applies the shared physical projection again to its selected
command. This keeps model rollout semantics close to executed semantics.

The frozen paper configuration uses two-stage compiled projection. It passed the
pre-specified non-inferiority criterion against full compiled projection while
reducing p95 solve and end-to-end latency. Projection remains a simulation and
planning constraint mechanism, not a hardware certification.
