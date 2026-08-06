from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


# Operator-defined gripper raw -> fine-model joint angle mapping.
# raw 1755 (jaw closed)  -> model -10 deg (closed limit)
# raw 3155 (jaw open max)-> model +100 deg (model gripper joint upper limit)
GRIPPER_RAW_CLOSED = 1755.0
GRIPPER_RAW_OPEN = 3155.0
GRIPPER_MODEL_CLOSED_DEG = -10.0
GRIPPER_MODEL_OPEN_DEG = 100.0


def gripper_model_angle_rad(hold_raw: float) -> float:
    """Map a gripper raw position to the fine model's gripper joint angle (rad).

    Used to place the fine model's gripper at the collection hold position
    when rendering the mirror and running table-clearance checks, so the
    moving-jaw height matches the real arm.
    """
    deg = (GRIPPER_MODEL_CLOSED_DEG
           + (float(hold_raw) - GRIPPER_RAW_CLOSED) / (GRIPPER_RAW_OPEN - GRIPPER_RAW_CLOSED)
           * (GRIPPER_MODEL_OPEN_DEG - GRIPPER_MODEL_CLOSED_DEG))
    return float(np.deg2rad(deg))


@dataclass(frozen=True)
class JointCoordinateMapping:
    joint_sign: np.ndarray
    joint_offset: np.ndarray
    source: str = "unverified"

    def to_kinematics(self, q_ctrl: np.ndarray) -> np.ndarray:
        return (self.joint_sign * np.asarray(q_ctrl) + self.joint_offset).astype(np.float32)

    def to_control(self, q_kin: np.ndarray) -> np.ndarray:
        return ((np.asarray(q_kin) - self.joint_offset) / self.joint_sign).astype(np.float32)

    def identity(self) -> dict[str, object]:
        return {"joint_sign": self.joint_sign.tolist(), "joint_offset": self.joint_offset.tolist(), "source": self.source}


def kinematics_identity(model_xml: str | Path, ee_site: str, mapping: JointCoordinateMapping,
                        joint_names: tuple[str, ...]) -> dict[str, object]:
    path = Path(model_xml)
    return {"model_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ee_site": ee_site,
            "joint_order": list(joint_names), "q_ctrl_to_q_kin": mapping.identity()}
