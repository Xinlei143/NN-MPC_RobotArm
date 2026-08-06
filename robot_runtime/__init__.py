"""Hardware/runtime boundary for simulation and real robot control."""

from .interfaces import CommandResult, RobotBackend, RobotState

__all__ = ["CommandResult", "RobotBackend", "RobotState"]
