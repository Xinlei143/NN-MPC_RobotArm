from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class DeadlineAdvance:
    next_deadline_ns: int
    skipped_ticks: int


def advance_absolute_deadline(deadline_ns: int, now_ns: int, period_ns: int) -> DeadlineAdvance:
    if period_ns <= 0:
        raise ValueError("period_ns must be positive")
    skipped = max(0, math.floor((int(now_ns) - int(deadline_ns)) / period_ns))
    return DeadlineAdvance(int(deadline_ns + (skipped + 1) * period_ns), int(skipped))


def sleep_until_ns(deadline_ns: int) -> int:
    remaining = (int(deadline_ns) - time.perf_counter_ns()) * 1e-9
    if remaining > 0:
        time.sleep(remaining)
    return time.perf_counter_ns()
