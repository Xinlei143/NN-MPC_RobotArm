"""Recovery decisions for residual MPC.

Command velocity and acceleration limits remain hard planning constraints and
logged diagnostics.  Reaching either limit is normal for a feasible projected
nominal trajectory, so it is intentionally not a recovery trigger.
"""

from __future__ import annotations

from collections.abc import Sequence


def residual_recovery_reason(
    tracking_errors: Sequence[float],
    *,
    residual_saturation_streak: int,
    consecutive_steps: int,
    error_ratio: float,
    min_tracking_error: float,
    recovery_active: bool,
) -> str:
    """Return a sustained-failure reason, or an empty string when MPC continues.

    The caller handles planner failures separately because they are immediate
    failures rather than a condition that must persist over several cycles.
    """
    if recovery_active:
        return ""
    if len(tracking_errors) >= consecutive_steps + 1:
        recent = tracking_errors[-(consecutive_steps + 1) :]
        error_worsening = (
            all(right > left for left, right in zip(recent, recent[1:]))
            and recent[-1] >= error_ratio * recent[0]
            and recent[-1] >= min_tracking_error
        )
        if error_worsening:
            return "tracking_error_growth"
    if residual_saturation_streak >= consecutive_steps:
        return "residual_saturation"
    return ""
