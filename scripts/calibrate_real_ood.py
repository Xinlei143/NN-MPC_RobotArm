#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.ood import RobustEnvelope


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze robust [q,dq,q_ref] OOD gate from Model-A and shadow tokens.")
    parser.add_argument("tokens", help="NPZ with training/validation/executed/selected_action/predicted_state tokens")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.tokens)
    required = {"training_tokens", "validation_tokens", "executed_tokens", "selected_action_tokens", "predicted_state_tokens"}
    missing = required - set(data.files)
    if missing: raise SystemExit(f"missing token arrays: {sorted(missing)}")
    envelope = RobustEnvelope.fit(data["training_tokens"], data["validation_tokens"], 99.5)
    payload = {"schema_version": 1, "token_semantics": "q_ctrl,dq_ctrl,transmitted_q_ref",
               "median": envelope.median.tolist(), "scale": envelope.scale.tolist(), "threshold": envelope.threshold,
               "executed_history_coverage": float(np.mean(envelope.contains(data["executed_tokens"]))),
               "selected_action_coverage": float(np.mean(envelope.contains(data["selected_action_tokens"]))),
               "predicted_state_coverage": float(np.mean(envelope.contains(data["predicted_state_tokens"])))}
    text = json.dumps(payload, indent=2)
    Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
