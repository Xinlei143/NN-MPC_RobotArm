#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate activation delay from shadow planner timestamps.")
    parser.add_argument("events", help="NPZ containing state_timestamp_ns and packet_publish_timestamp_ns")
    parser.add_argument("--control-dt", type=float, default=1/30)
    parser.add_argument("--guard-ms", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    data = np.load(args.events)
    if "planner_end_to_end_latency_s" in data.files:
        latency = np.asarray(data["planner_end_to_end_latency_s"], dtype=np.float64)
        latency = latency[np.isfinite(latency)]
    else:
        latency = (data["packet_publish_timestamp_ns"] - data["state_timestamp_ns"]) * 1e-9
    if latency.size >= 2000:
        estimate, method = float(np.percentile(latency, 99.5)), "p99.5"
    else:
        estimate, method = float(np.percentile(latency, 99) + args.guard_ms / 1000), "p99_plus_guard"
    steps = math.ceil(estimate / args.control_dt)
    late_key = "planner_late_drop" if "planner_late_drop" in data.files else "late_drop"
    late_rate = float(np.mean(data[late_key])) if late_key in data.files else float("nan")
    expiry_rate = float(np.mean(data["packet_expired"])) if "packet_expired" in data.files else float("nan")
    payload = {"samples": int(latency.size), "method": method, "latency_s": estimate,
               "anticipation_delay_steps": steps, "late_drop_rate": late_rate, "packet_expiry_rate": expiry_rate}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__": main()
