"""Shared runtime helpers for Model-C command-line tools."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"


def ensure_import_paths() -> None:
    """Make repository packages importable for direct script execution."""
    for path in (ROOT, DYNAMICS_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_runner(module_name: str) -> ModuleType:
    """Load the generic MPC runner without making it a Model-C module."""
    ensure_import_paths()
    runner_path = ROOT / "scripts" / "run_cem_mpc.py"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
