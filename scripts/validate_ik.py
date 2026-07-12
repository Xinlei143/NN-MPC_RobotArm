"""Independently revalidate a saved task-space reference bundle."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "learned_mujoco_dynamics"
for import_path in (ROOT, DYNAMICS_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import mujoco

from mpc.ik_solver import IKConfig
from mpc.kinematics_utils import MujocoKinematics
from mpc.reference_pipeline import (
    REFERENCE_FILE_NAME,
    ReferenceConfig,
    load_reference_bundle,
    reference_summary,
    validate_reference_bundle,
)


def resolve_runtime_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute FK/IK safety diagnostics for a saved task-space reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference_file", required=True)
    parser.add_argument("--model_xml", default="learned_mujoco_dynamics/ABB_IRB2400.xml")
    parser.add_argument("--ee_site_name", default=None)
    parser.add_argument("--summary_path", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def config_from_metadata(metadata: dict[str, Any], ee_site_name: str | None = None) -> ReferenceConfig:
    """Restore supported validation settings from generated bundle metadata."""

    raw_config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(raw_config, dict):
        config = ReferenceConfig()
    else:
        values = dict(raw_config)
        raw_ik = values.pop("ik_config", {})
        ik_fields = {field.name for field in fields(IKConfig)}
        ik_values = {key: value for key, value in raw_ik.items() if key in ik_fields} if isinstance(raw_ik, dict) else {}
        values["ik_config"] = IKConfig(**ik_values)
        config_fields = {field.name for field in fields(ReferenceConfig)}
        config = ReferenceConfig(**{key: value for key, value in values.items() if key in config_fields})
    if ee_site_name is None:
        return config
    values = {field.name: getattr(config, field.name) for field in fields(ReferenceConfig)}
    values["ee_site_name"] = ee_site_name
    return ReferenceConfig(**values)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reference_path = resolve_runtime_path(args.reference_file)
    if reference_path.is_dir():
        reference_path = reference_path / REFERENCE_FILE_NAME
    model_path = resolve_runtime_path(args.model_xml)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    bundle = load_reference_bundle(reference_path, expected_n_joints=model.nu)
    config = config_from_metadata(bundle.metadata, args.ee_site_name)
    kinematics = MujocoKinematics(model, ee_site_name=config.ee_site_name, n_joints=model.nu)
    diagnostics = validate_reference_bundle(bundle, kinematics, config)
    bundle.metadata["validation"] = diagnostics
    summary = reference_summary(bundle)
    summary["reference_file"] = str(reference_path)
    if args.summary_path is None:
        summary_path = reference_path.with_name("validation_summary.json")
    else:
        summary_path = resolve_runtime_path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Reference validation passed: {reference_path}")
    print(f"Wrote validation summary to {summary_path}")


if __name__ == "__main__":
    main()

