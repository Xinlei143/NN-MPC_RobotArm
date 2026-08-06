#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.factory import make_so101_backend
from robot_runtime.backends.so101_backend import ALL_MOTORS
from robot_runtime.timing import advance_absolute_deadline, sleep_until_ns


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {key: float(np.percentile(a, p)) for key, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


def channel_stats(samples: list[np.ndarray]) -> dict[str, object]:
    """Summarize time-weighted diagnostic samples for all six motors."""
    if not samples:
        return {"sample_count": 0, "per_motor": {}}
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ALL_MOTORS):
        raise ValueError(f"expected diagnostic samples shaped [N, {len(ALL_MOTORS)}], got {values.shape}")
    return {
        "sample_count": int(values.shape[0]),
        "per_motor": {
            name: {
                "min": float(np.min(values[:, index])),
                "p01": float(np.percentile(values[:, index], 1)),
                "median": float(np.percentile(values[:, index], 50)),
                "p99": float(np.percentile(values[:, index], 99)),
                "max": float(np.max(values[:, index])),
            }
            for index, name in enumerate(ALL_MOTORS)
        },
        "all_motors": {
            "min": float(np.min(values)),
            "p01": float(np.percentile(values, 1)),
            "median": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one-read/one-write SO101 control I/O.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--ticks", type=int, default=900)
    parser.add_argument("--output", help="Optional JSON output path. Written only after a complete benchmark.")
    parser.add_argument("--enable-hardware", action="store_true", help="Required acknowledgement: connects and enables torque.")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.enable_hardware:
        raise SystemExit("refusing hardware connection without --enable-hardware")
    if not args.operator_supported_shutdown:
        raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    backend = make_so101_backend(args.hardware_config)
    execution, lateness, periods = [], [], []
    voltage_samples: list[np.ndarray] = []
    temperature_samples: list[np.ndarray] = []
    diagnostic_timestamps_ns: list[int] = []
    diagnostic_retry_counts: list[int] = []
    diagnostic_sample_ages_s: list[float] = []
    q_samples: list[np.ndarray] = []
    invalid_state_count = 0
    fatal_state_count = 0
    validity_flag_counts: Counter[str] = Counter()
    misses = skipped = 0
    goal_readback_mismatch_count = 0
    try:
        backend.connect()
        period = int(backend.control_dt * 1e9)
        deadline = previous = time.perf_counter_ns()
        for tick in range(args.ticks):
            wake = time.perf_counter_ns()
            lateness.append(max(0, wake - deadline) * 1e-9)
            periods.append((wake - previous) * 1e-9)
            previous = wake
            state = backend.read_state(tick_index=tick)
            diagnostic_sample_ages_s.append(float(state.diagnostics.get("diagnostic_sample_age_s", float("inf"))))
            invalid_state_count += int(not state.valid)
            validity_flag_counts.update(state.validity_flags)
            q_samples.append(np.asarray(state.q_ctrl, dtype=np.float64))
            voltage = state.diagnostics.get("motor_voltage_v")
            diagnostic_timestamp = int(state.diagnostics.get("diagnostic_update_timestamp_ns", 0))
            if voltage is not None and diagnostic_timestamp and (not diagnostic_timestamps_ns or diagnostic_timestamp != diagnostic_timestamps_ns[-1]):
                voltage_samples.append(np.asarray(voltage, dtype=np.float64))
                temperature = state.diagnostics.get("motor_temperature")
                if temperature is not None:
                    temperature_samples.append(np.asarray(temperature, dtype=np.float64))
                diagnostic_timestamps_ns.append(diagnostic_timestamp)
                diagnostic_retry_counts.append(int(state.diagnostics.get("diagnostic_retry_count", 0)))
            goal_readback_mismatch_count += int(bool(state.diagnostics.get("goal_readback_mismatch", False)))
            fatal_flags = tuple(flag for flag in state.validity_flags if flag != "software_range")
            fatal_state_count += int(bool(fatal_flags))
            if fatal_flags:
                raise RuntimeError(
                    f"benchmark aborted at tick {tick}: fatal state {fatal_flags}; "
                    f"q_ctrl_rad={state.q_ctrl.tolist()}"
                )
            result = backend.retransmit_measured_position(tick_index=tick)
            if not result.tx_local_success:
                raise RuntimeError(f"benchmark write failed at tick {tick}")
            end = time.perf_counter_ns()
            execution.append((end - wake) * 1e-9)
            misses += int(end > deadline + period)
            advance = advance_absolute_deadline(deadline, end, period)
            skipped += advance.skipped_ticks
            deadline = advance.next_deadline_ns
            sleep_until_ns(deadline)
        q_values = np.asarray(q_samples, dtype=np.float64)
        payload = {"ticks": args.ticks, "control_path_execution_s": stats(execution),
                   "wake_lateness_s": stats(lateness), "period_s": stats(periods),
                   "deadline_miss_rate": misses / args.ticks, "skipped_tick_rate": skipped / args.ticks,
                   "invalid_state_count": invalid_state_count,
                   "fatal_state_count": fatal_state_count,
                   "validity_flag_counts": dict(sorted(validity_flag_counts.items())),
                   "q_ctrl_rad": {
                       "joint_names": list(ALL_MOTORS[:5]),
                       "first": q_values[0].tolist(), "last": q_values[-1].tolist(),
                       "min": np.min(q_values, axis=0).tolist(), "max": np.max(q_values, axis=0).tolist(),
                   },
                   "motor_voltage_v": channel_stats(voltage_samples),
                   "motor_temperature_c": channel_stats(temperature_samples),
                   "diagnostics": {
                       "independent_sample_count": len(diagnostic_timestamps_ns),
                       "update_timestamps_ns": diagnostic_timestamps_ns,
                       "sample_age_max_s_after_first_sample": float(max(
                           (age for age in diagnostic_sample_ages_s if np.isfinite(age)), default=float("inf"))),
                       "read_retry_count_total": int(sum(diagnostic_retry_counts)),
                       "read_retry_count_max": int(max(diagnostic_retry_counts, default=0)),
                       "goal_position_readback_mismatch_count": goal_readback_mismatch_count,
                   },
                   "acceptance": {"execution_p99_lt_25ms": bool(np.percentile(execution, 99) < .025),
                                  "wake_p99_lt_2ms": bool(np.percentile(lateness, 99) < .002),
                                  "deadline_miss_lt_1pct": bool(misses / args.ticks < .01),
                                  "fatal_state_count_is_zero": fatal_state_count == 0,
                                  "goal_readback_mismatch_is_zero": goal_readback_mismatch_count == 0}}
        output_text = json.dumps(payload, indent=2) + "\n"
        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.write_text(output_text, encoding="utf-8")
            temporary_path.replace(output_path)
        print(output_text, end="")
    finally:
        backend.close()


if __name__ == "__main__": main()
