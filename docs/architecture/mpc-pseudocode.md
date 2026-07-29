# Residual CEM-MPC Pseudocode

```text
at each execution tick:
    update state and executed-command history
    compute the continuously evolving IK nominal q_nom(t)
    if a new packet is active:
        a = packet age
        q_req = q_nom(t) + packet.residual[a] + bounded feedback
    else:
        q_req = q_nom(t)
    execute physical_projection(q_req)

in the asynchronous planner worker:
    snapshot current state and history
    forecast activation state and activation-time context
    sample residual sequences with CEM around the future IK nominal
    project candidates, roll out the GRU, and score them
    optionally rerank the final pool in task space
    publish a timestamped residual packet
```

The virtual protocol uses deterministic activation delay for ablations; the
threaded protocol uses wall-clock worker completion and packet deadlines.
