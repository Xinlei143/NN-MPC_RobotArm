"""Generate pre-registered representative and trade-off figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--paper-root", required=True)
    value.add_argument("--legacy-ik-root", required=True)
    value.add_argument("--output-dir", required=True)
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "rollout.npz", allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _representative(paper: Path, ik_root: Path, output: Path) -> tuple[dict[str, Any], Path]:
    rows = [
        row for row in _rows(paper / "summaries" / "main.csv")
        if row["trajectory"] == "circle" and row["label"] == "ThreadedASAP"
    ]
    ordered = sorted(rows, key=lambda row: float(row["tcp_rmse_m"]))
    selected = ordered[len(ordered) // 2]
    seed = int(selected["seed"])
    index = json.loads((paper / "runs" / "indexes" / "main.json").read_text())["entries"]
    methods = {}
    for label in ("NaiveDelayed", "FullVirtual", "ThreadedASAP"):
        entry = next(
            item for item in index
            if item["trajectory"] == "circle" and int(item["seed"]) == seed and item["label"] == label
        )
        methods[label] = Path(entry["run_dir"])
    ik_fingerprint = next(
        path for path in ik_root.glob("physical/nominal/circle_seed_*/run_fingerprint.json")
        if json.loads(path.read_text())["payload"]["case_id"] == f"circle_seed_{seed}"
    )
    methods["ProjectedDirectIK"] = ik_fingerprint.parent
    arrays = {label: _load(path) for label, path in methods.items()}
    desired = arrays["ThreadedASAP"]["desired_ee_positions"]

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*desired.T, color="black", linewidth=2.0, label="Desired")
    for label, values in arrays.items():
        axis.plot(*values["actual_ee_positions"].T, linewidth=1.2, label=label)
    axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]"); axis.set_zlabel("z [m]")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "representative_tracking.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, values in arrays.items():
        time = np.arange(len(values["ee_position_errors"])) * 0.01
        axes[0].plot(time, values["ee_position_errors"] * 1000, label=label)
        axes[1].plot(time, np.rad2deg(values["ee_orientation_errors"]), label=label)
    axes[0].set_ylabel("TCP error [mm]"); axes[1].set_ylabel("Orientation error [deg]")
    axes[1].set_xlabel("Time [s]"); axes[0].legend(ncol=2, fontsize=8)
    for axis in axes: axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "representative_errors.pdf")
    plt.close(fig)

    threaded = arrays["ThreadedASAP"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    signals = (
        ("requested_mpc_residual", "Requested residual [rad]"),
        ("executed_residual", "Executed residual [rad]"),
        ("feedback_correction", "Feedback [rad]"),
        ("safety_projection_offset", "Projection offset [rad]"),
    )
    for axis, (key, ylabel) in zip(axes, signals):
        values = threaded[key]
        axis.plot(np.arange(len(values)) * 0.01, values, linewidth=0.7)
        axis.set_ylabel(ylabel); axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output / "representative_control.pdf")
    plt.close(fig)
    manifest = {
        "rule": "nominal circle ThreadedASAP seed with TCP RMSE closest to the five-seed median",
        "selected_seed": seed,
        "threaded_tcp_rmse_m": float(selected["tcp_rmse_m"]),
        "runs": {label: str(path) for label, path in methods.items()},
    }
    (output / "representative_case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, methods["ThreadedASAP"]


def _timeline(run_dir: Path, output: Path) -> None:
    events = [
        json.loads(line)
        for line in (run_dir / "planner_events.jsonl").read_text(encoding="utf-8").splitlines()
    ][:20]
    delay = int(_load(run_dir)["anticipation_delay_steps"])
    fig, axis = plt.subplots(figsize=(11, 7))
    for index, event in enumerate(events):
        snapshot = float(event["observed_control_step"]) * 0.01
        publication = snapshot + float(event.get("end_to_end_latency_s", np.nan))
        activation = snapshot + delay * 0.01
        axis.plot([snapshot, publication], [index, index], color="tab:blue", linewidth=3)
        axis.scatter(snapshot, index, marker="o", color="black", s=22)
        axis.scatter(publication, index, marker="s", color="tab:blue", s=24)
        late = "late" in str(event.get("result_type", ""))
        axis.scatter(activation, index, marker="x" if late else "^", color="red" if late else "tab:green", s=35)
    axis.set_xlabel("Wall-clock time from rollout start [s]")
    axis.set_ylabel("Planner packet")
    axis.set_title("Snapshot → publication → scheduled activation")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "planner_timeline.pdf")
    plt.close(fig)


def _tradeoff_figures(paper: Path, output: Path) -> None:
    delay_path = paper / "summaries" / "delay_sweep_components.csv"
    if delay_path.is_file():
        rows = _rows(delay_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for axis, trajectory in zip(axes, ("circle", "fast_ellipse")):
            members = [row for row in rows if row["trajectory"] == trajectory]
            for protocol in ("naive_delayed", "anchor_only", "no_feedback", "full"):
                protocol_rows = [row for row in members if row["delay_protocol"] == protocol]
                delays = sorted({int(row["delay_steps"]) for row in protocol_rows})
                means, lows, highs = [], [], []
                for delay in delays:
                    values = np.asarray([float(row["tcp_rmse_m"]) for row in protocol_rows if int(row["delay_steps"]) == delay])
                    means.append(np.mean(values)); lows.append(np.percentile(values, 2.5)); highs.append(np.percentile(values, 97.5))
                axis.plot(delays, np.asarray(means) * 1000, marker="o", label=protocol)
                axis.fill_between(delays, np.asarray(lows) * 1000, np.asarray(highs) * 1000, alpha=0.12)
            axis.set_title(trajectory); axis.set_xlabel("Delay D [steps]"); axis.grid(alpha=0.25)
        axes[0].set_ylabel("TCP RMSE [mm]"); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output / "delay_sweep.pdf"); plt.close(fig)
    projection_path = paper / "summaries" / "projection_choice.csv"
    if projection_path.is_file():
        rows = _rows(projection_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for axis, evaluation_set in zip(axes, ("common_d", "deployed")):
            members = [row for row in rows if row["evaluation_set"] == evaluation_set]
            for variant in ("ProjectionOff", "FullCompiled", "TwoStageCompiled"):
                variant_rows = [row for row in members if row["variant"] == variant]
                tracking = np.mean([float(row["tcp_rmse_m"]) for row in variant_rows]) * 1000
                timing_key = "solve_p95_s" if evaluation_set == "common_d" else "e2e_p95_s"
                timing = np.nanmean([float(row[timing_key]) for row in variant_rows]) * 1000
                axis.scatter(timing, tracking, s=65, label=variant)
                axis.annotate(variant, (timing, tracking), fontsize=8)
            axis.set_title(evaluation_set); axis.set_xlabel("Latency [ms]"); axis.set_ylabel("TCP RMSE [mm]")
            axis.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(output / "projection_tradeoff.pdf"); plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    paper = Path(args.paper_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _, threaded = _representative(paper, Path(args.legacy_ik_root).resolve(), output)
    _timeline(threaded, output)
    _tradeoff_figures(paper, output)


if __name__ == "__main__":
    main()
