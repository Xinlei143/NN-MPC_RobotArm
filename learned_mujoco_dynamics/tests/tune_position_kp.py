from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


DEFAULT_SAFE_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
ZERO_GRAVITY_COMPENSATION_JOINT_INDICES = (5,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize per-joint MuJoCo position-actuator Kp dynamics with dampratio fixed at 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--joint", default="all", type=str, help="'all' or 1-based joint numbers like '2' or '2,3'")
    parser.add_argument("--kp_values", default="800,1200,2000,3000,5000,10000", type=str)
    parser.add_argument("--base_kp", default=2000.0, type=float, help="Kp used for non-tested joints")
    parser.add_argument("--duration", default=3.0, type=float)
    parser.add_argument("--frame_skip", default=1, type=int)
    parser.add_argument("--step_fraction", default=-0.12, type=float, help="Step size as a fraction of joint range")
    parser.add_argument("--settling_tolerance", default=0.02, type=float)
    parser.add_argument(
        "--no_gravity_compensation",
        action="store_true",
        help="Disable qfrc_bias feedforward during the response simulation.",
    )
    parser.add_argument("--output_dir", default="outputs/diagnostics/kp_tuning", type=str)
    return parser.parse_args(argv)


def parse_float_list(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one numeric value")
    return items


def parse_joint_selection(value: str, n_joints: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(n_joints))
    joints: list[int] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        joint_number = int(text)
        if joint_number < 1 or joint_number > n_joints:
            raise ValueError(f"Joint selection must be between 1 and {n_joints}, got {joint_number}")
        joints.append(joint_number - 1)
    if not joints:
        raise ValueError("Expected 'all' or at least one joint number")
    return sorted(dict.fromkeys(joints))


def format_number(value: float) -> str:
    return f"{value:.8g}"


def joint_specs_from_xml(xml_text: str) -> tuple[list[str], list[tuple[float, float]]]:
    pattern = re.compile(r'<joint\s+[^>]*name="([^"]+)"[^>]*range="([^"]+)"[^>]*/?>')
    specs: list[tuple[str, tuple[float, float]]] = []
    for match in pattern.finditer(xml_text):
        name = match.group(1)
        if not name.startswith("joint_"):
            continue
        parts = [float(item) for item in match.group(2).split()]
        if len(parts) != 2:
            raise ValueError(f"Joint {name} range must have two values")
        specs.append((name, (parts[0], parts[1])))
    specs.sort(key=lambda item: int(item[0].split("_")[1]))
    return [item[0] for item in specs], [item[1] for item in specs]


def build_position_actuator_xml(
    xml_text: str,
    joint_names: list[str],
    joint_ranges: list[tuple[float, float]],
    kps: list[float],
) -> str:
    if len(joint_names) != len(joint_ranges) or len(joint_names) != len(kps):
        raise ValueError("joint_names, joint_ranges, and kps must have the same length")
    actuator_lines = ["  <actuator>"]
    for name, joint_range, kp in zip(joint_names, joint_ranges, kps):
        lower, upper = joint_range
        actuator_lines.append(
            f'    <position name="{name}_position" joint="{name}" kp="{format_number(kp)}" '
            f'dampratio="1" ctrllimited="true" '
            f'ctrlrange="{format_number(lower)} {format_number(upper)}"/>'
        )
    actuator_lines.append("  </actuator>")
    actuator_block = "\n".join(actuator_lines)
    return re.sub(r"\s*<actuator>.*?</actuator>", "\n" + actuator_block, xml_text, count=1, flags=re.DOTALL)


def make_position_model_xml(source_xml: Path, kps: list[float]) -> tuple[Path, list[str], list[tuple[float, float]]]:
    xml_text = source_xml.read_text(encoding="utf-8")
    joint_names, joint_ranges = joint_specs_from_xml(xml_text)
    if len(kps) != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} Kp values, got {len(kps)}")
    position_xml = build_position_actuator_xml(xml_text, joint_names, joint_ranges, kps)
    temp = tempfile.NamedTemporaryFile(suffix=".xml", prefix="tmp_position_kp_", mode="w", delete=False, dir=ROOT)
    with temp:
        temp.write(position_xml)
    return Path(temp.name), joint_names, joint_ranges


def gravity_compensation_force(model: mujoco.MjModel, source_data: mujoco.MjData, scratch_data: mujoco.MjData, n_joints: int) -> np.ndarray:
    mujoco.mj_resetData(model, scratch_data)
    scratch_data.qpos[: model.nq] = source_data.qpos[: model.nq]
    scratch_data.qvel[: model.nv] = 0.0
    mujoco.mj_forward(model, scratch_data)
    gravity_tau = np.asarray(scratch_data.qfrc_bias[:n_joints], dtype=np.float64).copy()
    for joint_idx in ZERO_GRAVITY_COMPENSATION_JOINT_INDICES:
        if joint_idx < n_joints:
            gravity_tau[joint_idx] = 0.0
    return gravity_tau


def torque_components(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scratch_data: mujoco.MjData,
    q_ref: np.ndarray,
    n_joints: int,
    gravity_compensation: bool,
) -> dict[str, np.ndarray]:
    ctrlrange = np.asarray(model.actuator_ctrlrange[:n_joints], dtype=np.float64)
    data.ctrl[:n_joints] = np.clip(np.asarray(q_ref, dtype=np.float64), ctrlrange[:, 0], ctrlrange[:, 1])
    if gravity_compensation:
        gravity_tau = gravity_compensation_force(model, data, scratch_data, n_joints)
    else:
        gravity_tau = np.zeros(n_joints, dtype=np.float64)
    data.qfrc_applied[:n_joints] = gravity_tau
    mujoco.mj_forward(model, data)
    actuator_tau = np.asarray(data.qfrc_actuator[:n_joints], dtype=np.float64).copy()
    return {
        "actuator_tau": actuator_tau,
        "gravity_tau": gravity_tau.copy(),
        "total_tau": actuator_tau + gravity_tau,
    }


def safe_initial_q(joint_ranges: list[tuple[float, float]]) -> np.ndarray:
    q = DEFAULT_SAFE_Q[: len(joint_ranges)].copy()
    for idx, (lower, upper) in enumerate(joint_ranges):
        margin = 0.1 * (upper - lower)
        q[idx] = float(np.clip(q[idx], lower + margin, upper - margin))
    return q


def step_target(q0: np.ndarray, joint_idx: int, joint_ranges: list[tuple[float, float]], step_fraction: float) -> np.ndarray:
    q_ref = q0.copy()
    lower, upper = joint_ranges[joint_idx]
    step = step_fraction * (upper - lower)
    direction = 1.0 if q0[joint_idx] + step <= upper - 0.05 else -1.0
    q_ref[joint_idx] = float(np.clip(q0[joint_idx] + direction * step, lower + 0.05, upper - 0.05))
    return q_ref


def compute_metrics(
    t: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_ref: np.ndarray,
    joint_idx: int,
    joint_range: tuple[float, float],
    settling_tolerance: float,
) -> dict[str, float | bool]:
    error = q[:, joint_idx] - q_ref[joint_idx]
    abs_error = np.abs(error)
    lower, upper = joint_range
    step_size = abs(q_ref[joint_idx] - q[0, joint_idx])
    overshoot = max(0.0, float(np.max(q[:, joint_idx] - q_ref[joint_idx]))) if q_ref[joint_idx] >= q[0, joint_idx] else max(0.0, float(np.max(q_ref[joint_idx] - q[:, joint_idx])))
    tolerance = max(settling_tolerance * max(step_size, 1e-6), 1e-4)
    settling_time = float("nan")
    for idx in range(len(t)):
        if np.all(abs_error[idx:] <= tolerance):
            settling_time = float(t[idx])
            break
    return {
        "max_abs_error": float(np.max(abs_error)),
        "rms_error": float(np.sqrt(np.mean(np.square(error)))),
        "overshoot": overshoot,
        "max_abs_dq": float(np.max(np.abs(dq[:, joint_idx]))),
        "settling_time": settling_time,
        "near_limit": bool(np.any((q[:, joint_idx] >= upper - 0.02) | (q[:, joint_idx] <= lower + 0.02))),
    }


def run_response(
    source_xml: Path,
    joint_idx: int,
    kp: float,
    base_kp: float,
    duration: float,
    frame_skip: int,
    step_fraction: float,
    gravity_compensation: bool,
) -> dict[str, Any]:
    xml_text = source_xml.read_text(encoding="utf-8")
    joint_names, joint_ranges = joint_specs_from_xml(xml_text)
    kps = [base_kp] * len(joint_names)
    kps[joint_idx] = kp
    temp_xml, _, _ = make_position_model_xml(source_xml, kps)
    try:
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        data = mujoco.MjData(model)
        gravity_data = mujoco.MjData(model)
        q0 = safe_initial_q(joint_ranges)
        q_ref = step_target(q0, joint_idx, joint_ranges, step_fraction)
        mujoco.mj_resetData(model, data)
        data.qpos[: len(joint_names)] = q0
        data.qvel[: len(joint_names)] = 0.0
        data.ctrl[: len(joint_names)] = q_ref
        mujoco.mj_forward(model, data)
        steps = int(duration / (model.opt.timestep * frame_skip))
        q_records: list[np.ndarray] = []
        dq_records: list[np.ndarray] = []
        bias_records: list[np.ndarray] = []
        applied_records: list[np.ndarray] = []
        actuator_tau_records: list[np.ndarray] = []
        gravity_tau_records: list[np.ndarray] = []
        total_tau_records: list[np.ndarray] = []
        for _ in range(steps + 1):
            torque = torque_components(model, data, gravity_data, q_ref, len(joint_names), gravity_compensation)
            q_records.append(np.asarray(data.qpos[: len(joint_names)], dtype=np.float64).copy())
            dq_records.append(np.asarray(data.qvel[: len(joint_names)], dtype=np.float64).copy())
            bias_records.append(np.asarray(data.qfrc_bias[: len(joint_names)], dtype=np.float64).copy())
            applied_records.append(np.asarray(data.qfrc_applied[: len(joint_names)], dtype=np.float64).copy())
            actuator_tau_records.append(torque["actuator_tau"])
            gravity_tau_records.append(torque["gravity_tau"])
            total_tau_records.append(torque["total_tau"])
            data.ctrl[: len(joint_names)] = q_ref
            for _substep in range(frame_skip):
                if gravity_compensation:
                    data.qfrc_applied[: len(joint_names)] = gravity_compensation_force(
                        model, data, gravity_data, len(joint_names)
                    )
                else:
                    data.qfrc_applied[: len(joint_names)] = 0.0
                mujoco.mj_step(model, data)
        t = np.arange(len(q_records), dtype=np.float64) * model.opt.timestep * frame_skip
        return {
            "t": t,
            "q": np.asarray(q_records),
            "dq": np.asarray(dq_records),
            "bias": np.asarray(bias_records),
            "applied": np.asarray(applied_records),
            "actuator_tau": np.asarray(actuator_tau_records),
            "gravity_tau": np.asarray(gravity_tau_records),
            "total_tau": np.asarray(total_tau_records),
            "q0": q0,
            "q_ref": q_ref,
            "joint_ranges": joint_ranges,
            "joint_names": joint_names,
            "kps": kps,
            "gravity_compensation": gravity_compensation,
        }
    finally:
        temp_xml.unlink(missing_ok=True)


def plot_joint_sweep(output_dir: Path, joint_idx: int, responses: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for response, metric in zip(responses, metrics):
        kp = metric["kp"]
        t = response["t"]
        q = response["q"][:, joint_idx]
        dq = response["dq"][:, joint_idx]
        applied = response["applied"][:, joint_idx]
        q_ref = response["q_ref"][joint_idx]
        axes[0].plot(t, q, label=f"kp={kp:g}")
        axes[1].plot(t, q - q_ref, label=f"kp={kp:g}")
        axes[2].plot(t, dq, label=f"kp={kp:g}")
        axes[3].plot(t, applied, label=f"kp={kp:g}")
    q_ref = responses[0]["q_ref"][joint_idx]
    lower, upper = responses[0]["joint_ranges"][joint_idx]
    gravity_compensation = bool(responses[0]["gravity_compensation"])
    axes[0].axhline(q_ref, color="k", linestyle="--", linewidth=1, label="q_ref")
    axes[0].axhline(lower, color="gray", linestyle=":", linewidth=1, label="limit")
    axes[0].axhline(upper, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel(f"q{joint_idx} rad")
    axes[1].set_ylabel("error rad")
    axes[2].set_ylabel(f"dq{joint_idx} rad/s")
    axes[3].set_ylabel("qfrc_applied")
    axes[3].set_xlabel("time s")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
    suffix = "with gravity compensation" if gravity_compensation else "without gravity compensation"
    fig.suptitle(f"Joint {joint_idx + 1} position Kp sweep (dampratio=1, {suffix})")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = "gravity_comp" if gravity_compensation else "no_gravity_comp"
    fig.savefig(output_dir / f"joint_{joint_idx + 1:02d}_kp_sweep_{file_suffix}.png", dpi=150)
    plt.close(fig)


def plot_joint_torque_sweep(output_dir: Path, joint_idx: int, responses: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    components = [
        ("total_tau", "total_tau Nm"),
        ("actuator_tau", "actuator_tau Nm"),
        ("gravity_tau", "gravity_tau Nm"),
    ]
    for response, metric in zip(responses, metrics):
        kp = metric["kp"]
        t = response["t"]
        for axis, (key, _ylabel) in zip(axes, components):
            axis.plot(t, response[key][:, joint_idx], label=f"kp={kp:g}")
    for axis, (_key, ylabel) in zip(axes, components):
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("time s")
    gravity_compensation = bool(responses[0]["gravity_compensation"])
    suffix = "with gravity compensation" if gravity_compensation else "without gravity compensation"
    fig.suptitle(f"Joint {joint_idx + 1} bottom-controller torque ({suffix})")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = "gravity_comp" if gravity_compensation else "no_gravity_comp"
    fig.savefig(output_dir / f"joint_{joint_idx + 1:02d}_kp_sweep_torque_{file_suffix}.png", dpi=150)
    plt.close(fig)


def write_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "kp_tuning_metrics.csv"
    fieldnames = [
        "joint",
        "kp",
        "target",
        "initial",
        "max_abs_error",
        "rms_error",
        "overshoot",
        "max_abs_dq",
        "settling_time",
        "near_limit",
        "gravity_compensation",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_xml = resolve_project_path(args.model_xml, ROOT)
    xml_text = source_xml.read_text(encoding="utf-8")
    joint_names, joint_ranges = joint_specs_from_xml(xml_text)
    kp_values = parse_float_list(args.kp_values)
    selected_joints = parse_joint_selection(args.joint, len(joint_names))
    output_dir = Path(args.output_dir)
    gravity_compensation = not args.no_gravity_compensation
    metric_rows: list[dict[str, Any]] = []
    for joint_idx in selected_joints:
        responses: list[dict[str, Any]] = []
        joint_metrics: list[dict[str, Any]] = []
        for kp in kp_values:
            response = run_response(
                source_xml,
                joint_idx,
                kp,
                args.base_kp,
                args.duration,
                args.frame_skip,
                args.step_fraction,
                gravity_compensation,
            )
            metric = compute_metrics(
                response["t"],
                response["q"],
                response["dq"],
                response["q_ref"],
                joint_idx,
                joint_ranges[joint_idx],
                args.settling_tolerance,
            )
            row = {
                "joint": joint_idx + 1,
                "kp": kp,
                "target": float(response["q_ref"][joint_idx]),
                "initial": float(response["q0"][joint_idx]),
                "gravity_compensation": gravity_compensation,
                **metric,
            }
            responses.append(response)
            joint_metrics.append(row)
            metric_rows.append(row)
        plot_joint_sweep(output_dir, joint_idx, responses, joint_metrics)
        plot_joint_torque_sweep(output_dir, joint_idx, responses, joint_metrics)
    write_metrics(output_dir, metric_rows)
    print(f"saved Kp tuning plots and metrics to {output_dir}")
    for row in metric_rows:
        print(
            f"joint={row['joint']} kp={row['kp']:g} max_err={row['max_abs_error']:.4f} "
            f"rms={row['rms_error']:.4f} max_dq={row['max_abs_dq']:.4f} "
            f"settling={row['settling_time']} near_limit={row['near_limit']}"
        )


if __name__ == "__main__":
    main()
