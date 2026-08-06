#!/usr/bin/env python3
"""Read-only SO101 table-clearance calibration evidence tool.

It never moves the arm.  During capture the operator manually supports and
places the arm at an already safe pose, measures a named component's actual
distance to the tabletop, and the script records the corresponding encoder
state and FK prediction.  A separate known-height gauge workflow fits the
physical TCP point from encoder readings, so the operator never needs to know
the model's internal ``tool`` coordinates.  Finalization rejects an optimistic
geometry model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.robot_config import load_robot_spec
from robot_runtime.factory import make_so101_backend
from robot_runtime.table_safety import (
    DEFAULT_SAFETY_BODIES,
    load_table_safety_profile,
    make_table_clearance_checker,
    profile_with_finalized_calibration,
)


def _write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def initialize(args: argparse.Namespace) -> None:
    if abs(args.base_roll_deg) > .5 or abs(args.base_pitch_deg) > .5:
        raise SystemExit("base roll/pitch must be measured within +/-0.5 deg")
    if args.geometry_padding_mm < 0:
        raise SystemExit("geometry padding must be non-negative")
    robot = load_robot_spec(args.robot_config, validate_model=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing table profile: {output}")
    use_model_tcp_assumption = bool(args.assume_level_base_model_tcp)
    initial_probe_local = [0.0, 0.0, 0.08] if use_model_tcp_assumption else [0.0, 0.0, 0.0]
    profile = {
        "schema_version": 1,
        "status": "approved" if use_model_tcp_assumption else "draft",
        "model_xml_sha256": robot.model_xml_sha256,
        "table_plane": {"normal": [0.0, 0.0, 1.0], "offset_m": 0.0},
        "minimum_clearance_m": .050,
        "calibrated_overprediction_m": 0.0,
        "static_margin_m": .010,
        "geometry_padding_m": float(args.geometry_padding_mm) / 1000.0,
        "base_level": {"roll_deg": float(args.base_roll_deg), "pitch_deg": float(args.base_pitch_deg)},
        "q_ctrl_to_q_kin": {
            "joint_sign": [1, 1, 1, 1, 1], "joint_offset": [0, 0, 0, 0, 0],
            "source": "identity_candidate_pending_table_validation",
        },
        "safety_body_names": list(DEFAULT_SAFETY_BODIES),
        "table_probe": {
            # This draft value is unusable for collection.  --fit-probe
            # replaces it from physical known-height gauge contacts.
            "body": "tool", "local_position_m": initial_probe_local, "radius_m": 0.0,
        },
        # ee_site is defined in so101_nominal.xml as tool-local [0, 0, .08].
        # This is a model-coordinate condition, not a physical TCP survey.
        "tcp_center": {
            "body": "tool", "local_position_m": [0.0, 0.0, 0.08],
            "minimum_height_m": float(args.tcp_center_minimum_height_mm) / 1000.0,
        },
        "tcp_probe_calibration": {"status": "pending_known_height_contacts"},
    }
    if use_model_tcp_assumption:
        profile["verification_mode"] = "assumed_level_base_model_tcp_center"
        profile["assumption_note"] = (
            "Operator asserts base is directly seated on a level table; no physical TCP/table registration was performed."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": "model_assumption_profile_created" if use_model_tcp_assumption else "draft_created",
                      "profile": str(output), "model_xml_sha256": robot.model_xml_sha256,
                      "tcp_center_minimum_height_m": profile["tcp_center"]["minimum_height_m"]}, indent=2))


def capture_contact(args: argparse.Namespace) -> None:
    """Record one read-only contact of an identifiable TCP point on a gauge."""
    if args.reference_height_mm < 50:
        raise SystemExit("reference gauge height must be at least 50 mm")
    robot = load_robot_spec(args.robot_config, validate_model=True)
    profile = load_table_safety_profile(args.table_safety_config, n_joints=robot.n_joints,
                                        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=False)
    checker = make_table_clearance_checker(robot.model_xml, profile, robot.n_joints)
    backend = make_so101_backend(args.hardware_config)
    try:
        backend.connect_read_only()
        state = backend.read_state(tick_index=0)
        if not state.valid:
            raise RuntimeError(f"refusing contact capture from invalid state: {state.validity_flags}")
    finally:
        backend.close()
    position, rotation = checker.body_world_pose(state.q_ctrl, profile.table_probe_body)
    payload = {
        "schema_version": 1,
        "sample_id": args.sample_id,
        "kind": "known_height_tcp_contact",
        "reference_height_m": float(args.reference_height_mm) / 1000.0,
        "q_ctrl_rad": state.q_ctrl.tolist(),
        "tool_body_position_m": position.tolist(),
        "tool_body_rotation": rotation.tolist(),
        "present_position_raw": np.asarray(state.diagnostics["present_position_raw"], dtype=int).tolist(),
        "table_safety_identity": profile.identity(),
        "robot_identity": robot.artifact_identity(),
        "note": ("Read-only contact capture: operator placed the same identifiable TCP/"
                 "gripper point lightly on a >=50 mm gauge; no motor command was sent."),
    }
    _write_new_json(Path(args.output).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def fit_probe(args: argparse.Namespace) -> None:
    """Fit a tool-local TCP point and table height from known-height contacts."""
    robot = load_robot_spec(args.robot_config, validate_model=True)
    profile_path = Path(args.table_safety_config).expanduser().resolve()
    profile = load_table_safety_profile(profile_path, n_joints=robot.n_joints,
                                        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=False)
    files = [Path(item).expanduser().resolve() for item in args.contact_files.split(",") if item.strip()]
    if len(files) < 6:
        raise SystemExit("--contact-files needs at least six known-height contact JSON files")
    checker = make_table_clearance_checker(robot.model_xml, profile, robot.n_joints)
    rows, target, sample_ids = [], [], []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != "known_height_tcp_contact":
            raise SystemExit(f"not a known-height contact evidence file: {path}")
        if payload.get("robot_identity") != robot.artifact_identity():
            raise SystemExit(f"contact robot identity mismatch: {path}")
        q_ctrl = np.asarray(payload.get("q_ctrl_rad"), dtype=float)
        position, rotation = checker.body_world_pose(q_ctrl, profile.table_probe_body)
        # z^T (R p + x) - table_offset = measured gauge height.
        rows.append(np.r_[rotation[2, :], -1.0])
        target.append(float(payload["reference_height_m"]) - position[2])
        sample_ids.append(str(payload.get("sample_id", path.stem)))
    matrix, rhs = np.asarray(rows), np.asarray(target)
    if np.linalg.matrix_rank(matrix) < 4:
        raise SystemExit("contact poses do not vary tool orientation enough; capture six more distinct poses")
    solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    residuals = matrix @ solution - rhs
    max_residual = float(np.max(np.abs(residuals)))
    local_position, table_offset = solution[:3], float(solution[3])
    if max_residual > .005:
        raise SystemExit(f"contact fit residual is {max_residual * 1000:.1f} mm (>5 mm); repeat contacts more carefully")
    if float(np.linalg.norm(local_position)) > .25:
        raise SystemExit("fitted TCP point is implausibly far from the tool body; check that every contact used the same point")
    source = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise SystemExit("table safety profile must be a mapping")
    target_path = Path(args.output).expanduser().resolve()
    if target_path.exists():
        raise SystemExit(f"refusing to overwrite existing fitted profile: {target_path}")
    source["status"] = "draft"
    source["table_plane"] = {"normal": [0.0, 0.0, 1.0], "offset_m": table_offset}
    source["table_probe"] = {"body": profile.table_probe_body,
                             "local_position_m": local_position.tolist(),
                             # The 10 mm global padding is also applied to this point.
                             "radius_m": 0.0}
    source["tcp_center"] = {"body": profile.table_probe_body,
                            "local_position_m": local_position.tolist(),
                            "minimum_height_m": .110}
    source["tcp_probe_calibration"] = {
        "status": "fitted_from_known_height_contacts", "contact_count": len(files),
        "sample_ids": sample_ids, "max_residual_m": max_residual,
        "contact_evidence_sha256": hashlib.sha256(
            json.dumps([json.loads(path.read_text(encoding="utf-8")) for path in files], sort_keys=True).encode()
        ).hexdigest(),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": "probe_fitted", "profile": str(target_path),
                      "table_offset_m": table_offset, "max_residual_mm": max_residual * 1000,
                      "fitted_local_position_m": local_position.tolist()}, indent=2))


def capture(args: argparse.Namespace) -> None:
    if args.measured_clearance_mm < 50:
        raise SystemExit("refusing a capture below the 50 mm physical clearance requirement")
    robot = load_robot_spec(args.robot_config, validate_model=True)
    profile = load_table_safety_profile(args.table_safety_config, n_joints=robot.n_joints,
                                        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=False)
    checker = make_table_clearance_checker(robot.model_xml, profile, robot.n_joints)
    backend = make_so101_backend(args.hardware_config)
    try:
        backend.connect_read_only()
        state = backend.read_state(tick_index=0)
        if not state.valid:
            raise RuntimeError(f"refusing capture from invalid state: {state.validity_flags}")
    finally:
        backend.close()
    result = checker.evaluate(state.q_ctrl)
    if args.component not in result.component_clearances_m:
        raise SystemExit(f"unknown component {args.component!r}; choices: {sorted(result.component_clearances_m)}")
    measured = float(args.measured_clearance_mm) / 1000.0
    payload = {
        "schema_version": 1,
        "sample_id": args.sample_id,
        "component": args.component,
        "q_ctrl_rad": state.q_ctrl.tolist(),
        "present_position_raw": np.asarray(state.diagnostics["present_position_raw"], dtype=int).tolist(),
        "predicted_clearance_m": float(result.component_clearances_m[args.component]),
        "measured_clearance_m": measured,
        "prediction_minus_measurement_m": float(result.component_clearances_m[args.component] - measured),
        "all_component_clearances_m": result.component_clearances_m,
        "table_safety_identity": profile.identity(),
        "robot_identity": robot.artifact_identity(),
        "note": "Read-only capture: no Goal_Position or motor configuration write was sent.",
    }
    _write_new_json(Path(args.output).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    robot = load_robot_spec(args.robot_config, validate_model=True)
    files = [Path(item).expanduser().resolve() for item in args.capture_files.split(",") if item.strip()]
    if not files:
        raise SystemExit("--capture-files must contain one or more JSON evidence paths")
    captures = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    for capture_payload, path in zip(captures, files, strict=True):
        if capture_payload.get("robot_identity") != robot.artifact_identity():
            raise SystemExit(f"capture robot identity mismatch: {path}")
    profile_with_finalized_calibration(
        args.table_safety_config, captures, output_path=args.output,
        expected_model_xml_sha256=robot.model_xml_sha256,
        required_components=(*DEFAULT_SAFETY_BODIES, "table_probe"),
    )
    print(json.dumps({"status": "approved", "profile": str(Path(args.output).expanduser().resolve()),
                      "capture_count": len(captures)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and verify a 50 mm SO101 tabletop safety profile.")
    parser.add_argument("--hardware-config")
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-contact", action="store_true")
    parser.add_argument("--fit-probe", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--table-safety-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-roll-deg", type=float)
    parser.add_argument("--base-pitch-deg", type=float)
    parser.add_argument("--geometry-padding-mm", type=float, default=10.0)
    parser.add_argument("--assume-level-base-model-tcp", action="store_true",
                        help="Create an approved model-only profile: table z=0 and ee_site TCP center >=110 mm.")
    parser.add_argument("--tcp-center-minimum-height-mm", type=float, default=110.0)
    parser.add_argument("--sample-id")
    parser.add_argument("--component")
    parser.add_argument("--measured-clearance-mm", type=float)
    parser.add_argument("--capture-files")
    parser.add_argument("--reference-height-mm", type=float)
    parser.add_argument("--contact-files")
    args = parser.parse_args()
    selected = sum((args.initialize, args.capture, args.capture_contact, args.fit_probe, args.finalize))
    if selected != 1:
        raise SystemExit("choose exactly one calibration mode")
    if args.initialize:
        if args.base_roll_deg is None or args.base_pitch_deg is None:
            raise SystemExit("--initialize requires --base-roll-deg and --base-pitch-deg")
        if args.tcp_center_minimum_height_mm < 110:
            raise SystemExit("TCP center minimum height must be at least 110 mm")
        initialize(args)
    elif args.capture:
        if not args.hardware_config or not args.table_safety_config or not args.sample_id or not args.component:
            raise SystemExit("--capture requires --hardware-config, --table-safety-config, --sample-id, and --component")
        if args.measured_clearance_mm is None:
            raise SystemExit("--capture requires --measured-clearance-mm")
        capture(args)
    elif args.capture_contact:
        if not args.hardware_config or not args.table_safety_config or not args.sample_id:
            raise SystemExit("--capture-contact requires --hardware-config, --table-safety-config, and --sample-id")
        if args.reference_height_mm is None:
            raise SystemExit("--capture-contact requires --reference-height-mm")
        capture_contact(args)
    elif args.fit_probe:
        if not args.table_safety_config or not args.contact_files:
            raise SystemExit("--fit-probe requires --table-safety-config and --contact-files")
        fit_probe(args)
    else:
        if not args.table_safety_config or not args.capture_files:
            raise SystemExit("--finalize requires --table-safety-config and --capture-files")
        finalize(args)


if __name__ == "__main__":
    main()
