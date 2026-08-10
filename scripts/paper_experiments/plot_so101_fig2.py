#!/usr/bin/env python3
"""Render the paper's three-panel Fig. 2 from the frozen SO101 rollouts.

The figure keeps the original representative-condition data while adding the
provided SO101 setup photograph and changing the time-series panel to the
primary encoder-space joint RMS error.  TCP curves remain forward-kinematics
quantities computed from encoder joint states; no external tracker is implied.
The script is deliberately offline and writes a source manifest alongside
PDF/SVG/PNG exports for figure provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.kinematics_utils import MujocoKinematics


DT = 1.0 / 30.0
SEGMENT_SHAPE_LOOP = 3
Q_OFFSET = np.asarray([0.0, 0.00872665, -0.1396263, -0.2516342834, 0.0], dtype=np.float64)
METHODS = {
    "direct": {"label": "Direct IK", "color": "#666666", "linestyle": "-.", "linewidth": 1.35},
    "preview6": {"label": "Fixed Preview6", "color": "#CC79A7", "linestyle": ":", "linewidth": 1.45},
    "nn_mpc": {"label": "NN-MPC", "color": "#0072B2", "linestyle": "-", "linewidth": 1.65},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _positions(fk: MujocoKinematics, q: np.ndarray) -> np.ndarray:
    return np.asarray([fk.forward(row + Q_OFFSET)[0] for row in np.asarray(q, dtype=np.float64)], dtype=np.float64)


def _rollout_positions(fk: MujocoKinematics, path: Path, n: int) -> np.ndarray:
    arrays = _load_npz(path)
    states = arrays.get("actual_states", arrays.get("states"))
    if states is None:
        raise ValueError(f"no actual_states/states in {path}")
    return _positions(fk, np.asarray(states[:n, :5], dtype=np.float64))


def _rollout_q_and_target(path: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_npz(path)
    states = arrays.get("actual_states", arrays.get("states"))
    if states is None:
        raise ValueError(f"no actual_states/states in {path}")
    q_des = np.asarray(arrays.get("q_des"), dtype=np.float64)
    if q_des.ndim != 2 or q_des.shape[1] != 5:
        raise ValueError(f"q_des must have shape (T,5) in {path}")
    return np.asarray(states[:n, :5], dtype=np.float64), q_des[:n]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": "#b9b9b9",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def render(output_dir: Path, protocol_id: str, overwrite: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "fig2_representative_tracking.png",
        "fig2_representative_tracking.pdf",
        "fig2_representative_tracking.svg",
        "fig2_representative_tracking.source_manifest.json",
    )
    existing_outputs = [output_dir / name for name in output_names if (output_dir / name).exists()]
    if existing_outputs and not overwrite:
        joined = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite existing Figure 2 outputs in {output_dir}: {joined}. "
            "Use --force only after backing up any manually edited assets."
        )
    ref_path = ROOT / "outputs/hardware/so101_paper_refs_20260810/ellipse_nominal_p0/reference.npz"
    output_root = ROOT / "outputs/hardware/so101_paper_final/20260810_tracking1deg_v1"
    rollout_paths = {
        key: output_root / f"ellipse_nominal_p0_{key}" / "rollout.npz" for key in METHODS
    }
    photo_path = ROOT / "Paper/robo2026/figures/so101.png"
    model_path = ROOT / "dynamics_modeling/robots/so101_fine/scene_table_guard_25mm.xml"
    if not photo_path.exists():
        raise FileNotFoundError(f"SO101 setup photo not found: {photo_path}")
    reference = _load_npz(ref_path)
    execution_steps = int(np.asarray(reference["execution_steps"]).reshape(-1)[0])
    n = min(execution_steps, len(reference["task_positions_des"]))
    task_positions = np.asarray(reference["task_positions_des"][:n], dtype=np.float64)
    segment_ids = np.asarray(reference["segment_ids"][:n], dtype=np.int64)
    lap_ids = np.asarray(reference["lap_ids"][:n], dtype=np.int64)
    shape_mask = segment_ids == SEGMENT_SHAPE_LOOP
    first_lap_mask = shape_mask & (lap_ids == 0)
    if not np.any(shape_mask) or not np.any(first_lap_mask):
        raise ValueError("reference does not contain a complete SEGMENT_SHAPE_LOOP lap")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    fk = MujocoKinematics(model, "gripperframe", n_joints=5)
    positions = {key: _rollout_positions(fk, path, n) for key, path in rollout_paths.items()}
    q_and_targets = {key: _rollout_q_and_target(path, n) for key, path in rollout_paths.items()}
    center = np.mean(task_positions[shape_mask], axis=0)
    photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    # Use a centered square display crop so the portrait hardware photograph
    # can occupy roughly half of the composite figure width without being
    # reduced to a narrow strip.  The source file remains unchanged and the
    # exact post-orientation crop is recorded in the manifest below.
    photo_width, photo_height = photo.size
    crop_side = min(photo_width, photo_height)
    crop_top = int(round((photo_height - crop_side) * 0.52))
    photo_crop_box = (0, crop_top, photo_width, crop_top + crop_side)
    photo_display = photo.crop(photo_crop_box)

    _style()
    fig = plt.figure(figsize=(3.55, 3.35), constrained_layout=False)
    # Give the hardware photograph more visual weight while retaining a
    # readable path panel and a full-width primary error panel.
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.30, 0.70), height_ratios=(1.60, 1.20),
        hspace=0.34, wspace=0.30,
    )
    ax_photo = fig.add_subplot(grid[0, 0])
    ax_path = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[1, :])

    # Panel (a): the provided setup photograph. EXIF orientation is normalized
    # for portable PDF/SVG embedding; only the documented display crop is
    # applied, with no contrast or color adjustment.
    ax_photo.imshow(photo_display, interpolation="none")
    ax_photo.axis("off")
    ax_photo.set_box_aspect(1.0)
    ax_photo.text(
        -0.08, 0.98, "(a)", transform=ax_photo.transAxes, color="black", fontsize=10.5,
        fontweight="bold", va="top", ha="left", clip_on=False,
        bbox={"facecolor": "white", "alpha": 1.0, "edgecolor": "none", "pad": 1.5},
    )

    # Panel (b): first complete physical ellipse lap, centered in the desired
    # y-z plane as in the original figure.
    desired_yz = (task_positions[first_lap_mask][:, 1:3] - center[None, 1:3]) * 1000.0
    ax_path.plot(
        desired_yz[:, 0], desired_yz[:, 1], color="#111111", linestyle="--", linewidth=1.75,
        label="Desired", zorder=3,
    )
    for key in METHODS:
        curve = (positions[key][first_lap_mask][:, 1:3] - center[None, 1:3]) * 1000.0
        style = METHODS[key]
        ax_path.plot(curve[:, 0], curve[:, 1], color=style["color"], linestyle=style["linestyle"],
                     linewidth=style["linewidth"], label=style["label"], zorder=2)
    ax_path.set_xlabel("TCP $y$ [mm]")
    ax_path.set_ylabel("TCP $z$ [mm]")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True)
    ax_path.text(0.03, 1.18, "(b)", transform=ax_path.transAxes, fontsize=10.5, fontweight="bold", va="top", clip_on=False)

    # Panel (c): encoder-space joint RMS error for the complete episode. The
    # shaded interval is the primary shape-loop window used by the analyzer.
    time = np.arange(n, dtype=np.float64) * DT
    for key in METHODS:
        q_actual, q_target = q_and_targets[key]
        error_deg = (q_actual - q_target) * (180.0 / np.pi)
        joint_rms_deg = np.sqrt(np.mean(np.square(error_deg), axis=1))
        style = METHODS[key]
        ax_error.plot(time, joint_rms_deg, color=style["color"], linestyle=style["linestyle"],
                      linewidth=style["linewidth"], label=style["label"], zorder=3)
    shape_indices = np.flatnonzero(shape_mask)
    start, stop = int(shape_indices[0]), int(shape_indices[-1] + 1)
    ax_error.axvspan(start * DT, stop * DT, color="#eeeeee", zorder=0)
    ax_error.axvline(start * DT, color="#777777", linewidth=0.9, zorder=1)
    ax_error.axvline(stop * DT, color="#777777", linewidth=0.9, zorder=1)
    ymax = max(
        float(np.nanmax(np.sqrt(np.mean(np.square((q_and_targets[key][0] - q_and_targets[key][1]) * (180.0 / np.pi)), axis=1))))
        for key in METHODS
    )
    ax_error.text(start * DT + 0.05, ymax * 0.92, "periodic tracking", color="#555555", fontsize=7.0, va="top")
    ax_error.set_xlabel("Time [s]")
    ax_error.set_ylabel("Encoder joint RMS\nerror [deg]")
    ax_error.grid(True)
    ax_error.set_xlim(time[0], time[-1])
    ax_error.set_ylim(bottom=0.0, top=ymax * 1.08)
    ax_error.text(-0.08, 1.14, "(c)", transform=ax_error.transAxes, fontsize=10.5, fontweight="bold", va="top")

    # One shared legend preserves the method mapping without repeating it in
    # the quantitative panels.
    handles, labels = ax_path.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 1.01), ncol=4,
               fontsize=6.6, handlelength=1.65, columnspacing=0.65, handletextpad=0.32,
               frameon=False)

    fig.savefig(output_dir / "fig2_representative_tracking.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "fig2_representative_tracking.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "fig2_representative_tracking.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # Hashes are repository-relative so the manifest is portable and safe to
    # publish; no raw rollout is copied into the evidence bundle.
    def rel(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    manifest = {
        "figure": "fig2_representative_tracking",
        "backend": "matplotlib/Python",
        "protocol_id": protocol_id,
        "selection_rule": "held-out matched condition whose NN-MPC minus Fixed Preview6 joint-RMSE difference is closest to the median over 18 pairs",
        "representative_condition": {
            "trial_block": "ellipse_nominal_p0",
            "shape": "ellipse",
            "speed": "nominal",
            "phase_index": 0,
            "controllers": ["direct", "preview6", "nn_mpc"],
        },
        "panel_a": "Provided SO101 physical setup photograph; EXIF orientation normalized and a centered square display crop applied, with no contrast, brightness, gamma, or color adjustment.",
        "panel_b": "Desired and encoder-state FK TCP y-z paths over the first complete SEGMENT_SHAPE_LOOP lap, centered by the desired loop mean.",
        "panel_c": "Encoder-space joint RMS error over the complete episode; shaded interval is SEGMENT_SHAPE_LOOP and the RMS is over five controlled joints at each tick.",
        "tcp_measurement": "encoder_state_forward_kinematics; no external Cartesian tracker",
        "coordinate_transform": "world TCP y/z coordinates centered by desired shape-loop mean and converted m to mm",
        "q_ctrl_to_q_kin_offset_rad": Q_OFFSET.tolist(),
        "raw_curve_smoothing": "none",
        "photo_integrity": {
            "path": rel(photo_path),
            "sha256": _sha256(photo_path),
            "processing": "EXIF orientation transpose plus centered square display crop; no tonal/color adjustment",
            "crop_box_after_exif_xyxy_px": list(photo_crop_box),
            "scale_bar": "not applicable",
        },
        "source_data": [
            {"path": rel(ref_path), "sha256": _sha256(ref_path)},
            *[{"path": rel(path), "sha256": _sha256(path)} for path in rollout_paths.values()],
            {"path": rel(model_path), "sha256": _sha256(model_path)},
        ],
        "execution_steps": n,
        "shape_loop_step_range": [start, stop],
        "joint_rms_definition": "sqrt(mean_j((q_actual_j - q_des_j)^2)) converted from rad to deg at each tick",
        "raster_dpi": 600,
        "exports": ["pdf", "svg", "png"],
    }
    manifest_path = output_dir / "fig2_representative_tracking.source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Paper/robo2026/figures")
    parser.add_argument("--protocol-id", default="so101_final_paper_20260810_tracking1deg_v1")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing Figure 2 exports after backing them up",
    )
    args = parser.parse_args()
    manifest = render(args.output_dir, args.protocol_id, overwrite=args.force)
    print(json.dumps({"figure": manifest["figure"], "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
