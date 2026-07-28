"""Build paper-ready descriptive ABB/UR5e portability tables and effect plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpc.robot_config import load_robot_spec


METHOD_MAP = {
    "DirectIK": "ProjectedDirectIK",
    "ProjectedDirectIK": "ProjectedDirectIK",
    "NaiveDelayed": "NaiveDelayed",
    "FullVirtual": "FullVirtual",
    "ThreadedASAP": "ThreadedAsync",
    "ThreadedAsync": "ThreadedAsync",
}
INFERENTIAL_COMPARISONS = (
    ("FullVirtual − NaiveDelayed", "NaiveDelayed", "FullVirtual"),
    ("ThreadedAsync − FullVirtual", "FullVirtual", "ThreadedAsync"),
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--abb-summary", required=True)
    value.add_argument("--ur5e-summary", required=True)
    value.add_argument("--output-dir", default="outputs/multi_robot_summary")
    value.add_argument("--bootstrap-samples", type=int, default=10000)
    value.add_argument("--bootstrap-seed", type=int, default=20260727)
    return value


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        label = METHOD_MAP.get(str(row.get("label")))
        if label is None:
            continue
        parsed = dict(row)
        parsed["label"] = label
        parsed["seed"] = int(row["seed"])
        for name in (
            "tcp_rmse_m",
            "tcp_p95_m",
            "orientation_rmse_rad",
            "joint_rmse_rad",
            "command_acceleration_rms_rad_s2",
            "torque_rms_nm",
            "e2e_p95_s",
        ):
            if name in row and row[name] not in {"", "nan", "None"}:
                parsed[name] = float(row[name])
        output.append(parsed)
    return output


def _index(rows: list[dict], label: str) -> dict[tuple[str, int], dict]:
    return {
        (str(row["trajectory"]), int(row["seed"])): row
        for row in rows if row["label"] == label
    }


def paired_differences(rows: list[dict], left: str, right: str) -> np.ndarray:
    right_rows = _index(rows, right)
    left_rows = _index(rows, left)
    common = sorted(set(left_rows).intersection(right_rows))
    if not common:
        raise ValueError(f"No matched rows for {right} − {left}")
    return np.asarray([
        float(right_rows[key]["tcp_rmse_m"]) - float(left_rows[key]["tcp_rmse_m"])
        for key in common
    ], dtype=np.float64)


def bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(values, size=(samples, len(values)), replace=True),
        axis=1,
    )
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def platform_rows() -> list[dict]:
    rows = []
    for label, config in (
        ("ABB IRB2400", ROOT / "configs" / "robots" / "abb_irb2400.yaml"),
        ("UR5e", ROOT / "configs" / "robots" / "ur5e.yaml"),
    ):
        spec = load_robot_spec(config)
        rows.append({
            "robot": label,
            "robot_id": spec.robot_id,
            "dof": spec.n_joints,
            "control_interface": "absolute joint-position reference",
            "control_period_ms": 1000.0 * spec.expected_control_dt,
            "tcp_site": spec.ee_site_name,
            "home_q_rad": json.dumps(spec.home_q.tolist()),
            "history_len": 16,
            "cem_horizon": 20,
            "cem_samples": 128,
            "cem_iterations": 2,
            "model_xml_sha256": spec.model_xml_sha256,
            "robot_spec_sha256": spec.spec_sha256,
        })
    return rows


def tracking_rows(robot_rows: dict[str, list[dict]]) -> list[dict]:
    output = []
    for robot, rows in robot_rows.items():
        for method in ("ProjectedDirectIK", "NaiveDelayed", "FullVirtual", "ThreadedAsync"):
            members = [row for row in rows if row["label"] == method]
            if not members:
                continue
            output.append({
                "robot": robot,
                "method": method,
                "cases": len(members),
                "tcp_rmse_mm": 1000.0 * float(np.mean([row["tcp_rmse_m"] for row in members])),
                "tcp_p95_mm": 1000.0 * float(np.mean([row["tcp_p95_m"] for row in members])),
                "orientation_rmse_deg": float(np.rad2deg(np.mean([
                    row["orientation_rmse_rad"] for row in members
                ]))),
            })
    return output


def direct_ik_descriptive_rows(robot_rows: dict[str, list[dict]]) -> list[dict]:
    output = []
    for robot, rows in robot_rows.items():
        direct = {
            str(row["trajectory"]): float(row["tcp_rmse_m"])
            for row in rows if row["label"] == "ProjectedDirectIK"
        }
        threaded: dict[str, list[float]] = {}
        for row in rows:
            if row["label"] == "ThreadedAsync":
                threaded.setdefault(str(row["trajectory"]), []).append(float(row["tcp_rmse_m"]))
        common = sorted(set(direct).intersection(threaded))
        if not common:
            continue
        direct_values = np.asarray([direct[name] for name in common], dtype=np.float64)
        threaded_values = np.asarray([np.mean(threaded[name]) for name in common], dtype=np.float64)
        delta = threaded_values - direct_values
        output.append({
            "robot": robot,
            "comparison": "ThreadedAsync − ProjectedDirectIK",
            "trajectory_cases": len(common),
            "threaded_tcp_rmse_mm": 1000.0 * float(np.mean(threaded_values)),
            "projected_direct_ik_tcp_rmse_mm": 1000.0 * float(np.mean(direct_values)),
            "mean_delta_tcp_rmse_mm": 1000.0 * float(np.mean(delta)),
            "relative_improvement_pct": 100.0 * float(-np.mean(delta) / np.mean(direct_values)),
            "all_trajectory_deltas_favor_threaded": bool(np.all(delta < 0.0)),
            "inference_scope": "descriptive four-trajectory comparison; no seed-paired bootstrap",
        })
    return output


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output_dir)
    robot_rows = {
        "ABB IRB2400": read_rows(Path(args.abb_summary)),
        "UR5e": read_rows(Path(args.ur5e_summary)),
    }
    write_csv(output / "platforms.csv", platform_rows())
    write_csv(output / "tracking_table.csv", tracking_rows(robot_rows))
    direct_ik_descriptive = direct_ik_descriptive_rows(robot_rows)
    write_csv(output / "direct_ik_descriptive.csv", direct_ik_descriptive)

    effects = []
    for robot_index, (robot, rows) in enumerate(robot_rows.items()):
        for comparison_index, (name, left, right) in enumerate(INFERENTIAL_COMPARISONS):
            values = paired_differences(rows, left, right)
            mean, low, high = bootstrap(
                values,
                args.bootstrap_samples,
                args.bootstrap_seed + 100 * robot_index + comparison_index,
            )
            effects.append({
                "robot": robot,
                "comparison": name,
                "matched_cases": len(values),
                "mean_delta_tcp_rmse_mm": 1000.0 * mean,
                "ci95_low_mm": 1000.0 * low,
                "ci95_high_mm": 1000.0 * high,
                "inference_scope": "within-robot matched trajectory-seed bootstrap",
            })
    write_csv(output / "effect_sizes.csv", effects)
    (output / "effect_sizes.json").write_text(
        json.dumps({
            "schema_version": 1,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "effects": effects,
            "direct_ik_descriptive": direct_ik_descriptive,
            "cross_robot_inference": (
                "descriptive direction/effect-size comparison only; no pooled significance test"
            ),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, len(INFERENTIAL_COMPARISONS), figsize=(7.0, 3.3), sharey=True)
    robots = ("ABB IRB2400", "UR5e")
    colors = ("#0072B2", "#D55E00")
    for axis, (comparison, _, _) in zip(axes, INFERENTIAL_COMPARISONS, strict=True):
        members = [row for row in effects if row["comparison"] == comparison]
        for y, (robot, color) in enumerate(zip(robots, colors, strict=True)):
            row = next(value for value in members if value["robot"] == robot)
            center = row["mean_delta_tcp_rmse_mm"]
            axis.errorbar(
                center, y,
                xerr=[[center - row["ci95_low_mm"]], [row["ci95_high_mm"] - center]],
                fmt="o", color=color, capsize=3,
            )
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(comparison, fontsize=9)
        axis.set_xlabel("Δ TCP RMSE [mm]")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(robots)))
    axes[0].set_yticklabels(robots)
    figure.suptitle("Within-robot portability effects (negative favors the right-hand method)")
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "multi_robot_effect_sizes.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
