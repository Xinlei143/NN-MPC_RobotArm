from __future__ import annotations

from pathlib import Path

from robot_runtime.backends.so101_backend import SO101Backend
from robot_runtime.config import load_hardware_config


def make_so101_backend(hardware_config: str | Path) -> SO101Backend:
    return SO101Backend(load_hardware_config(hardware_config))
