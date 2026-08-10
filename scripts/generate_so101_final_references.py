#!/usr/bin/env python3
"""Generate the frozen held-out SO101 references for the final paper protocol.

The geometry, phase set, speed utilization targets, and safety gates are part
of the protocol.  This script never reads a rollout and never tunes a
reference from controller performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dynamics_modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc.ik_solver import IKConfig
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle
from mpc.task_space_reference import SEGMENT_SHAPE_LOOP


DT = 1.0 / 30.0
HORIZON = 6
PADDING_ROWS = 20
PHASES_DEG = (0.0, 120.0, 240.0)
P99_DEG_S = np.asarray([10.89, 11.46, 10.89, 10.31, 10.89], dtype=np.float64)
HARD_ACCEL_RAD_S2 = np.ones(5, dtype=np.float64)
SPEED_TARGETS = {"nominal": 0.65, "fast": 0.85}
KIN_OFFSET = np.asarray([0.0, 0.00872665, -0.1396263, -0.2516342834, 0.0], dtype=np.float64)
CTRL_HOME = np.asarray([0.0, 0.0, 0.0, 0.2516342834, 0.0], dtype=np.float64)
KIN_HOME = CTRL_HOME + KIN_OFFSET
MODEL = ROOT / "dynamics_modeling/robots/so101_fine/scene_table_guard_25mm.xml"
COLLECTION_NPZ = ROOT / "outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz"
HARDWARE_CONFIG = ROOT / "configs/hardware/so101_follower.local.yaml"

SHAPES: dict[str, dict[str, float]] = {
    "ellipse": {"ellipse_axis_a": 0.04, "ellipse_axis_b": 0.025},
    "back_and_forth": {"back_and_forth_half_length": 0.04},
    "rounded_square": {"square_half_side": 0.025, "rounded_square_corner_radius": 0.008},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_config(shape: str, duration: float, phase_deg: float) -> ReferenceConfig:
    return ReferenceConfig(
        shape_name=shape,
        repeat_count=3,
        start_hold_duration=0.5,
        joint_departure_duration=2.0,
        approach_duration=2.0,
        lap_duration=float(duration),
        return_duration=2.0,
        joint_return_duration=2.0,
        final_hold_duration=0.5,
        center_mode="relative",
        center_offset=(0.0, 0.0, 0.0),
        plane_axis_u=(0.0, 1.0, 0.0),
        plane_axis_v=(0.0, 0.0, 1.0),
        fixed_orientation="safe",
        ee_site_name="gripperframe",
        start_phase=float(np.deg2rad(phase_deg)),
        safe_departure_mode="always",
        safe_sigma_threshold=0.04,
        safe_search_samples=2048,
        safe_joint_limit_margin=0.05,
        safe_q=tuple(KIN_HOME),
        ik_config=IKConfig(orientation_mode="position_only"),
        max_joint_velocity=(0.25, 0.25, 0.25, 0.25, 0.25),
        max_joint_acceleration=tuple(HARD_ACCEL_RAD_S2),
        **SHAPES[shape],
    )


def _build(model: mujoco.MjModel, shape: str, duration: float, phase_deg: float):
    return build_reference(
        config=_reference_config(shape, duration, phase_deg),
        model=model,
        initial_q=KIN_HOME,
        control_dt=DT,
        horizon=HORIZON,
        lookahead_steps=0,
    )


def _metrics(bundle) -> dict[str, object]:
    mask = np.asarray(bundle.segment_ids) == SEGMENT_SHAPE_LOOP
    max_dq_deg_s = np.rad2deg(np.max(np.abs(bundle.dq_des[mask]), axis=0))
    max_ddq = np.max(np.abs(bundle.ddq_des[mask]), axis=0)
    q_ctrl = bundle.q_des[: bundle.execution_steps] - KIN_OFFSET[None, :]
    position_ok = True
    position_report: dict[str, dict[str, float | bool]] = {}
    if COLLECTION_NPZ.exists():
        measured = np.load(COLLECTION_NPZ, allow_pickle=False)["measured_q"]
        for index, name in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")):
            low, high = np.percentile(measured[:, index], [0.1, 99.9])
            ref_low, ref_high = float(q_ctrl[:, index].min()), float(q_ctrl[:, index].max())
            inside = ref_low >= low and ref_high <= high
            position_ok = position_ok and bool(inside)
            position_report[name] = {
                "reference_min_rad": ref_low,
                "reference_max_rad": ref_high,
                "collection_p0.1_rad": float(low),
                "collection_p99.9_rad": float(high),
                "inside": bool(inside),
            }
    else:
        position_ok = False
    return {
        "max_lap_dq_deg_s": max_dq_deg_s.tolist(),
        "max_lap_ddq_rad_s2": max_ddq.tolist(),
        "position_distribution_ok": bool(position_ok),
        "position_distribution": position_report,
        "reference_finite": bool(np.all(np.isfinite(bundle.q_des))),
    }


def _passes(metrics: dict[str, object], utilization: float) -> bool:
    max_dq = np.asarray(metrics["max_lap_dq_deg_s"], dtype=np.float64)
    max_ddq = np.asarray(metrics["max_lap_ddq_rad_s2"], dtype=np.float64)
    return bool(
        np.all(max_dq <= P99_DEG_S * utilization + 1e-9)
        and np.all(max_ddq <= HARD_ACCEL_RAD_S2 * 0.80 + 1e-9)
        and metrics["position_distribution_ok"]
        and metrics["reference_finite"]
    )


def _find_duration(model: mujoco.MjModel, shape: str, utilization: float) -> tuple[float, dict[str, object], list[object]]:
    """Find the shortest discrete lap duration passing every registered phase."""
    cache: dict[int, tuple[dict[str, object], list[object]]] = {}

    def evaluate(samples: int) -> tuple[bool, dict[str, object], list[object]]:
        if samples not in cache:
            bundles = [_build(model, shape, samples * DT, phase) for phase in PHASES_DEG]
            metrics = [_metrics(bundle) for bundle in bundles]
            aggregate = {
                "max_lap_dq_deg_s": np.max(np.asarray([m["max_lap_dq_deg_s"] for m in metrics]), axis=0).tolist(),
                "max_lap_ddq_rad_s2": np.max(np.asarray([m["max_lap_ddq_rad_s2"] for m in metrics]), axis=0).tolist(),
                "position_distribution_ok": bool(all(m["position_distribution_ok"] for m in metrics)),
                "reference_finite": bool(all(m["reference_finite"] for m in metrics)),
                "phase_metrics": metrics,
            }
            cache[samples] = aggregate, bundles
        aggregate, bundles = cache[samples]
        return _passes(aggregate, utilization), aggregate, bundles

    low, high = 60, 1200
    ok, _, _ = evaluate(high)
    if not ok:
        raise RuntimeError(f"{shape}: no duration up to {high * DT:.2f}s passes the frozen envelope")
    while low < high:
        middle = (low + high) // 2
        ok, _, _ = evaluate(middle)
        if ok:
            high = middle
        else:
            low = middle + 1
    ok, aggregate, bundles = evaluate(low)
    if not ok:
        raise RuntimeError(f"{shape}: duration search ended without a passing reference")
    return low * DT, aggregate, bundles


def _write_mpc_reference(bundle, target: Path) -> tuple[Path, Path]:
    q_ctrl = (bundle.q_des[: bundle.execution_steps] - KIN_OFFSET[None, :]).astype(np.float32)
    hold = np.repeat(q_ctrl[-1][None, :], PADDING_ROWS, axis=0)
    q_full = np.concatenate([q_ctrl, hold], axis=0)
    dq_full = np.concatenate([bundle.dq_des[: bundle.execution_steps].astype(np.float32), np.zeros((PADDING_ROWS, 5), dtype=np.float32)])
    ddq_full = np.concatenate([bundle.ddq_des[: bundle.execution_steps].astype(np.float32), np.zeros((PADDING_ROWS, 5), dtype=np.float32)])
    np.save(target / "q_des_ctrl.npy", q_ctrl)
    np.savez(
        target / "joint_reference_mpc.npz",
        q_des=q_full,
        dq_des=dq_full,
        ddq_des=ddq_full,
        execution_steps=np.asarray(bundle.execution_steps, dtype=np.int64),
        padding_rows=np.asarray(PADDING_ROWS, dtype=np.int64),
        source=np.asarray("derived from frozen task-space reference.npz"),
    )
    return target / "q_des_ctrl.npy", target / "joint_reference_mpc.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/hardware/so101_paper_refs_20260810")
    parser.add_argument("--hardware-config", default=str(HARDWARE_CONFIG.relative_to(ROOT)))
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    manifest: dict[str, object] = {
        "format_version": 1,
        "protocol": "so101_final_paper_20260810_tracking1deg_v1",
        "control_dt_s": DT,
        "horizon": HORIZON,
        "padding_rows": PADDING_ROWS,
        "phases_deg": list(PHASES_DEG),
        "speed_targets": SPEED_TARGETS,
        "geometry": SHAPES,
        "mapping": {"q_kin_to_q_ctrl_offset": KIN_OFFSET.tolist(), "q_ctrl_home": CTRL_HOME.tolist()},
        "model_xml": str(MODEL.relative_to(ROOT)),
        "model_xml_sha256": sha256_file(MODEL),
        "collection_npz_sha256": sha256_file(COLLECTION_NPZ) if COLLECTION_NPZ.exists() else None,
        "artifacts": {},
    }
    for shape, _ in SHAPES.items():
        for speed, utilization in SPEED_TARGETS.items():
            duration, aggregate, _ = _find_duration(model, shape, utilization)
            manifest.setdefault("speed_design", {})
            manifest["speed_design"][f"{shape}_{speed}"] = {
                "lap_duration_s": duration,
                "utilization_target": utilization,
                "envelope": aggregate,
            }
            for phase_index, phase_deg in enumerate(PHASES_DEG):
                bundle = _build(model, shape, duration, phase_deg)
                label = f"{shape}_{speed}_p{phase_index}"
                target = out_dir / label
                target.mkdir(parents=True, exist_ok=True)
                bundle_path = save_reference_bundle(bundle, target)
                q_path, mpc_path = _write_mpc_reference(bundle, target)
                manifest["artifacts"][label] = {
                    "shape": shape,
                    "speed": speed,
                    "phase_index": phase_index,
                    "start_phase_deg": phase_deg,
                    "lap_duration_s": duration,
                    "execution_steps": int(bundle.execution_steps),
                    "rows": int(mpc_path and np.load(mpc_path, allow_pickle=False)["q_des"].shape[0]),
                    "reference_sha256": sha256_file(bundle_path),
                    "q_des_ctrl_sha256": sha256_file(q_path),
                    "joint_reference_mpc_sha256": sha256_file(mpc_path),
                    "q_des_ctrl.npy": {"bytes": q_path.stat().st_size, "sha256": sha256_file(q_path)},
                    "joint_reference_mpc.npz": {"bytes": mpc_path.stat().st_size, "sha256": sha256_file(mpc_path)},
                    "envelope": _metrics(bundle),
                }
                print(f"[{label}] duration={duration:.3f}s steps={bundle.execution_steps} ref={target}")
    manifest_path = out_dir / "mpc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path} sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
