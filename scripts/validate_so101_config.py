#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.config import file_sha256, load_hardware_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local SO101 hardware configuration without opening serial I/O.")
    parser.add_argument("--hardware-config", required=True)
    args = parser.parse_args()
    config = load_hardware_config(args.hardware_config)
    if not config.calibration_path.is_file():
        raise SystemExit(f"missing calibration file: {config.calibration_path}")
    actual = file_sha256(config.calibration_path)
    if actual != config.calibration_sha256:
        raise SystemExit(f"calibration hash mismatch: config={config.calibration_sha256}, actual={actual}")
    print(json.dumps({"status": "valid", "plant_identity": config.plant_identity()}, indent=2))


if __name__ == "__main__":
    main()
