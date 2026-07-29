# Residual CEM-MPC Cost Function

The planner scores projected joint-reference sequences around the continuous IK
nominal. The stage cost combines joint tracking, velocity tracking, residual
magnitude, residual velocity and acceleration, joint-limit barriers, and a
first-command term. The exact task-space score, when enabled, reranks only the
final valid pool; it is not used in stage-one sampling.

All terms are evaluated on projected commands. This distinction matters because
the execution layer can alter an infeasible requested residual before it reaches
the actuator interface. The public evidence reports task-space reranking as an
accuracy-latency trade-off rather than an unconditional gain.
