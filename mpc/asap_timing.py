"""Pure wall-clock scheduling measurements for the ASAP control loop."""
from __future__ import annotations

import numpy as np


def control_timing_sample(
    tick_start_s: float,
    previous_tick_start_s: float | None,
    scheduled_start_s: float,
    control_dt: float,
) -> tuple[float, float, float]:
    """Return actual period, positive wake-up lateness, and signed jitter."""
    period = float("nan") if previous_tick_start_s is None else tick_start_s - previous_tick_start_s
    wakeup_lateness = max(0.0, tick_start_s - scheduled_start_s)
    jitter = float("nan") if not np.isfinite(period) else period - control_dt
    return period, wakeup_lateness, jitter
