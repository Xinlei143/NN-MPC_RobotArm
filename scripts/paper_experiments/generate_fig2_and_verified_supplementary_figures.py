"""Regenerate the verified ROBIO 2026 Fig. 2 and supplementary figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_prism_data import sha256, write_bundle
from paper_style import (
    DOUBLE_COLUMN_IN, SINGLE_COLUMN_IN, METHOD_STYLES, apply_style, finish_axis,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True, help="Frozen paper_revision_v1 root")
    parser.add_argument("--suite-root", type=Path, required=True, help="Frozen paper_delay_aware_two_stage_v2 root")
    parser.add_argument("--legacy-ik-root", type=Path, required=True, help="Audited legacy IK/MPC root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-data-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "rollout.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing rollout arrays: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def index_entries(root: Path, suite: str) -> list[dict[str, Any]]:
    for candidate in (root / "indexes" / f"{suite}.json", root / "runs" / "indexes" / f"{suite}.json"):
        if candidate.is_file():
            return load_json(candidate)["entries"]
    raise FileNotFoundError(f"No index for suite {suite!r} beneath {root}")


def run_path(entry: dict[str, Any], *, legacy_root: Path) -> Path:
    value = Path(entry["run_dir"])
    if value.is_dir():
        return value
    marker = "outputs/robustness/"
    text = str(value).replace("\\", "/")
    if marker in text:
        candidate = legacy_root.parent / text.split(marker, 1)[1]
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Indexed run directory is unavailable and cannot be remapped: {value}")


def source_metadata(paths: list[Path], *, selection: str, statistic_unit: str, transform: str = "none") -> dict[str, Any]:
    return {
        "original_inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "selection_rule": selection,
        "statistical_unit": statistic_unit,
        "bootstrap": "Uses frozen case/cluster bootstrap where confidence intervals are plotted; no 10-ms tick is an inferential sample.",
        "coordinate_transform": transform,
        "method_styles": METHOD_STYLES,
    }


def write_source_manifest(output: Path, figure: str, metadata: dict[str, Any]) -> None:
    (output / f"{figure}.source_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def representative(main_rows: pd.DataFrame, entries: list[dict[str, Any]], legacy_root: Path) -> tuple[int, dict[str, Path], pd.DataFrame]:
    rows = main_rows[(main_rows.trajectory == "circle") & (main_rows.label == "ThreadedASAP")].copy()
    if len(rows) != 5:
        raise ValueError(f"Expected five nominal circle ThreadedASAP cases, found {len(rows)}")
    median = float(rows.tcp_rmse_m.median())
    selected = rows.assign(distance=(rows.tcp_rmse_m - median).abs()).sort_values(["distance", "seed"]).iloc[0]
    seed = int(selected.seed)
    if seed != 3:
        raise ValueError(f"Pre-registered representative selection changed unexpectedly: seed {seed}")
    wanted = {"NaiveDelayed", "FullVirtual", "ThreadedASAP"}
    paths = {
        entry["label"]: run_path(entry, legacy_root=legacy_root)
        for entry in entries if entry["trajectory"] == "circle" and int(entry["seed"]) == seed and entry["label"] in wanted
    }
    if set(paths) != wanted:
        raise ValueError(f"Representative MPC methods missing: {wanted - set(paths)}")
    paths["Projected Direct IK"] = legacy_root / "physical" / "nominal" / f"circle_seed_{seed}"
    paths["Preview IK"] = legacy_root / "preview" / "nominal" / f"circle_seed_{seed}"
    if not all(path.is_dir() for path in paths.values()):
        raise FileNotFoundError("Representative Projected/Preview IK run directories are missing")
    return seed, paths, rows


def fig2(output: Path, data_root: Path, arrays: dict[str, dict[str, np.ndarray]], seed: int, paths: dict[str, Path]) -> None:
    desired = arrays["ThreadedASAP"]["desired_ee_positions"]
    # `lap_ids` is emitted by the reference generator: -1 marks the transition
    # and non-negative values mark complete periodic laps.  The main panel
    # shows the first complete lap from the periodic tracking phase.
    lap_ids = arrays["ThreadedASAP"]["lap_ids"]
    tracking = np.flatnonzero(lap_ids >= 0)
    periodic = np.flatnonzero(lap_ids == 0)
    if len(tracking) == 0 or len(periodic) == 0:
        raise ValueError("Representative circle rollout has no first periodic lap metadata")
    periodic_start, periodic_stop = int(periodic[0]), int(periodic[-1]) + 1
    tracking_start, tracking_stop = int(tracking[0]), int(tracking[-1]) + 1
    periodic_slice = slice(periodic_start, periodic_stop)

    # The circle is generated in the robot's physical y-z plane (the x range is
    # numerically zero here), so retain that interpretable plane rather than
    # introducing an abstract PCA rotation.  The desired periodic center is the
    # common origin for every method.
    center_yz = np.mean(desired[periodic_slice, 1:3], axis=0)
    project = lambda values: (values[:, 1:3] - center_yz) * 1000.0
    fig, axes = plt.subplots(
        2, 1, figsize=(SINGLE_COLUMN_IN, 2.75),
        gridspec_kw={"height_ratios": [.85, 1.35], "hspace": .58},
    )
    ax = axes[0]
    spatial_styles = {
        "Desired": {"color": "#000000", "linestyle": (0, (5, 2)), "linewidth": 1.25},
        "NaiveDelayed": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.15},
        "ThreadedASAP": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.25},
    }
    display_labels = {"Desired": "Desired", "NaiveDelayed": "NaiveDelayed", "ThreadedASAP": "ThreadedAsync"}
    error_styles = {
        "NaiveDelayed": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.05},
        "FullVirtual": {"color": "#009E73", "linestyle": (0, (5, 2)), "linewidth": 1.0},
        "ThreadedASAP": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.15},
        "Projected Direct IK": {"color": "#555555", "linestyle": "-.", "linewidth": .9},
        "Preview IK": {"color": "#CC79A7", "linestyle": (0, (1.5, 1.5)), "linewidth": .9},
    }
    spatial_methods = ("Desired", "NaiveDelayed", "ThreadedASAP")
    for name in spatial_methods:
        values = desired if name == "Desired" else arrays[name]["actual_ee_positions"]
        xy = project(values)[periodic_slice]
        ax.plot(xy[:, 0], xy[:, 1], label=display_labels[name], marker=None, **spatial_styles[name])
    desired_periodic = project(desired)[periodic_slice]
    span = np.ptp(desired_periodic, axis=0)
    pad = max(float(np.max(span)) * .10, 1.0)
    xlim = (float(desired_periodic[:, 0].min() - pad), float(desired_periodic[:, 0].max() + pad))
    ylim = (float(desired_periodic[:, 1].min() - pad), float(desired_periodic[:, 1].max() + pad))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(""); ax.set_ylabel("TCP z [mm]", fontsize=7); ax.set_aspect("equal", adjustable="box")
    finish_axis(ax)
    ax.text(1.06, -.10, "TCP y [mm]", transform=ax.transAxes, fontsize=7,
            va="top", ha="left", clip_on=False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.08), ncol=3, fontsize=5.2,
              handlelength=1.8, columnspacing=.7, borderaxespad=0.)

    ax = axes[1]
    ordered = ("NaiveDelayed", "FullVirtual", "ThreadedASAP", "Projected Direct IK", "Preview IK")
    error_labels = {
        "NaiveDelayed": "NaiveDelayed", "FullVirtual": "FullVirtual", "ThreadedASAP": "ThreadedAsync",
        "Projected Direct IK": "Projected", "Preview IK": "Preview",
    }
    for name in ordered:
        values = arrays[name]
        time = np.arange(len(values["ee_position_errors"])) * 0.01
        ax.plot(time, values["ee_position_errors"] * 1000, label=error_labels[name], marker=None, **error_styles[name])
    tracking_start_time, tracking_stop_time = tracking_start * .01, tracking_stop * .01
    ax.axvspan(tracking_start_time, tracking_stop_time, color="#D9D9D9", alpha=.10, zorder=0)
    ax.axvline(tracking_start_time, color="#8A8A8A", linewidth=.7, zorder=1)
    ax.axvline(tracking_stop_time, color="#8A8A8A", linewidth=.7, zorder=1)
    ax.text(tracking_start_time + .08, .96, "periodic tracking", transform=ax.get_xaxis_transform(),
            fontsize=5.8, va="top", color="#555555")
    ax.set_xlabel(""); ax.set_ylabel("TCP error [mm]", fontsize=7)
    finish_axis(ax)
    ax.text(1., -.18, "Time [s]", transform=ax.transAxes, fontsize=7,
            va="top", ha="right", clip_on=False)
    ax.legend(loc="lower center", bbox_to_anchor=(.52, 1.015), ncol=5, fontsize=5.2,
              handlelength=1.2, columnspacing=.28, borderaxespad=0.)
    fig.subplots_adjust(left=.20, right=.99, bottom=.13, top=.93)
    panel_x = .145
    panel_a_y = axes[0].get_position().y0 + 1.08 * axes[0].get_position().height
    panel_b_y = axes[1].get_position().y0 + 1.015 * axes[1].get_position().height
    fig.text(panel_x, panel_a_y, "(a)", fontweight="bold", fontsize=7.4,
             va="bottom", ha="left")
    fig.text(panel_x, panel_b_y, "(b)", fontweight="bold", fontsize=7.4,
             va="bottom", ha="left")
    save_figure(fig, output / "fig2_representative_tracking")
    source = [path / "rollout.npz" for path in paths.values()]
    metadata = source_metadata(source, selection=f"Pre-registered nominal circle seed {seed}, closest ThreadedASAP RMSE to five-seed median.", statistic_unit="Representative trajectory; plotted 10-ms samples are descriptive, not inferential.", transform="Physical TCP y-z plane; all methods use the desired periodic-circle center as the common origin; metres converted to mm.")
    metadata.update({
        "methods": ["Desired", *ordered], "seed": seed, "raw_curve_smoothing": "none",
        "periodic_segment": {"start_step": periodic_start, "stop_step_exclusive": periodic_stop,
                             "source": "lap_ids == 0 (first complete periodic lap)"},
        "tracking_segment": {"start_step": tracking_start, "stop_step_exclusive": tracking_stop,
                             "source": "lap_ids >= 0 (all complete periodic laps)"},
    })
    tidy = []
    for name in ("Desired", *ordered):
        positions = desired if name == "Desired" else arrays[name]["actual_ee_positions"]
        for step, (x, y) in enumerate(project(positions)):
            tidy.append({"panel": "task_plane_periodic" if periodic_start <= step < periodic_stop else "task_plane_complete", "method": name, "step": step, "coordinate_1_mm": x, "coordinate_2_mm": y})
        if name != "Desired":
            for step, err in enumerate(arrays[name]["ee_position_errors"] * 1000):
                tidy.append({"panel": "tcp_error", "method": name, "step": step, "time_s": step*.01, "tcp_error_mm": err})
    write_bundle(data_root / "fig2", tidy=tidy, wide=pd.DataFrame(tidy).pivot_table(index=["panel", "step"], columns="method", values=["coordinate_1_mm", "coordinate_2_mm", "tcp_error_mm"], aggfunc="first").reset_index().to_dict("records"), metadata=metadata, readme="Fig. 2 source data. The spatial main panel contains the first complete periodic lap selected from lap_ids; the error panel retains the full episode. No error smoothing is applied, and all task-plane coordinates use the physical TCP y-z plane.")
    write_source_manifest(output, "fig2_representative_tracking", metadata)


def threaded_comparison(group: dict[str, Any], target: str) -> dict[str, Any]:
    for name, values in group["comparisons"].items():
        if name.startswith(("ThreadedASAP_minus_", "ThreadedAsync_minus_")) and target in name:
            return values["tcp_rmse_m"]
    raise KeyError(f"Missing ThreadedASAP comparison containing {target}")


def supplementary_robustness_forest(output: Path, data_root: Path, paper_root: Path) -> None:
    path=paper_root/"statistics"/"ik"/"mpc_vs_projected_ik_by_perturbation.json"; groups=load_json(path)["groups"]
    order=[("nominal","Nominal"),("payload","Payload"),("actuator_gain","Actuator gain"),("force_pulse","Force pulse"),("observation_noise","Observation noise")]
    fig,ax=plt.subplots(figsize=(SINGLE_COLUMN_IN,2.55)); tidy=[]; ypos=np.arange(len(order))[::-1]
    for offset,target,color,marker,label in [(-.15,"physical","#666666","D","Projected Direct IK"),(.15,"preview","#CC79A7","v","Preview IK")]:
        for y,(key,display) in zip(ypos,order):
            metric=threaded_comparison(groups[key],target); mean=metric["mean_delta_mpc_minus_ik"]*1000; lo,hi=np.asarray(metric["ci95"])*1000
            ax.plot([lo,hi],[y+offset,y+offset],color=color,lw=1.35); ax.plot(mean,y+offset,marker=marker,color=color,ms=4.3,label=label if y==ypos[0] else None)
            tidy.append({"condition":display,"baseline":label,"delta_tcp_rmse_mm":mean,"ci95_low_mm":lo,"ci95_high_mm":hi,"n_clusters":metric["n_clusters"],"n_pairs":metric["n_pairs"]})
    ax.axvline(0,color="#555555",ls="--",lw=.8); ax.axvspan(-32,0,color="#EAF3EA",alpha=.38,zorder=0); ax.text(-30.5,4.55,"ThreadedAsync better",fontsize=6.1,color="#4D4D4D")
    ax.set_yticks(ypos); ax.set_yticklabels([x[1] for x in order]); ax.set_xlabel("Δ TCP RMSE [mm] (ThreadedAsync − IK)"); ax.set_xlim(-32,6); finish_axis(ax,grid=False); ax.grid(axis="x",color="#B9B9B9",alpha=.18,lw=.55); ax.legend(loc="lower left",fontsize=6.5)
    fig.subplots_adjust(left=.31,right=.99,bottom=.22,top=.95); save_figure(fig,output/"fig5_robustness_forest")
    metadata=source_metadata([path],selection="Frozen five perturbation families; L3/L6 are clustered by trajectory × condition.",statistic_unit="Two-level cluster bootstrap; plotted intervals are frozen 95% confidence intervals.")
    write_bundle(data_root/"fig5",tidy=tidy,wide=pd.DataFrame(tidy).pivot(index="condition",columns="baseline",values="delta_tcp_rmse_mm").reset_index().to_dict("records"),metadata=metadata,readme="Fig. 5 forest-plot data. Negative values favor ThreadedAsync; observation noise versus Preview IK crosses zero.")
    write_source_manifest(output,"fig5_robustness_forest",metadata)


def supplementary(output: Path, data_root: Path, suite_root: Path) -> None:
    sup=output/"supplementary"; sup.mkdir(parents=True,exist_ok=True)
    aggregate=suite_root/"diagnostics"/"gru_validation"/"gru_validation_aggregate.csv"; df=pd.read_csv(aggregate); fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COLUMN_IN,2.25),sharex=True)
    for ax,metric,label in zip(axes,["q_rmse","dq_rmse"],["q RMSE [rad]","dq RMSE [rad/s]"]):
        rows=df[(df["mode"].eq("open_loop"))&(df["metric"].eq(metric))]
        for std,color,marker in [(0.5,"#0072B2","o"),(0.8,"#D55E00","s")]:
            x=rows[rows.action_std.eq(std)].sort_values("horizon"); ax.plot(x.horizon,x["mean"],color=color,marker=marker,label=f"action std {std}")
        ax.set_xticks([1,5,10,20]); ax.set_xlabel("Prediction horizon [steps]"); ax.set_ylabel(label); finish_axis(ax); ax.legend(fontsize=6.5)
    fig.subplots_adjust(wspace=.28,left=.075,right=.995,bottom=.23,top=.95); save_figure(fig,sup/"s1_gru_multistep_validation")
    metadata=source_metadata([aggregate],selection="20 held-out post-freeze MuJoCo rollouts.",statistic_unit="rollout aggregate; divergence recorded in frozen aggregate.")
    write_bundle(data_root/"supplementary"/"s1",tidy=df.to_dict("records"),wide=df.pivot_table(index=["action_std","horizon","mode"],columns="metric",values="mean").reset_index().to_dict("records"),metadata=metadata,readme="S1 GRU validation source data.")
    task=suite_root/"summaries"/"task_cost.csv"; rows=pd.read_csv(task); fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COLUMN_IN,2.25))
    pairs=[("JointOnlyFixedD6","TaskSpaceFixedD6","Fixed D6 virtual"),("JointOnlyDeployed","TaskSpaceDeployed","Calibrated threaded")]
    for ax,(joint,taskspace,title) in zip(axes,pairs):
        left=rows[rows.label.eq(joint)].set_index(["trajectory","seed"]); right=rows[rows.label.eq(taskspace)].set_index(["trajectory","seed"]); common=left.index.intersection(right.index)
        for i,key in enumerate(common): ax.plot([0,1],[left.loc[key,"tcp_rmse_m"]*1000,right.loc[key,"tcp_rmse_m"]*1000],color="#B0B0B0",lw=.75); ax.scatter([0,1],[left.loc[key,"tcp_rmse_m"]*1000,right.loc[key,"tcp_rmse_m"]*1000],color=["#666666","#0072B2"],s=13)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Joint-only","Task-space"]); ax.set_ylabel("TCP RMSE [mm]"); ax.set_title(title); finish_axis(ax)
    fig.subplots_adjust(wspace=.28,left=.075,right=.995,bottom=.23,top=.90); save_figure(fig,sup/"s2_task_space_reranking")
    projection=suite_root/"summaries"/"projection_choice.csv"; rows=pd.read_csv(projection); fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COLUMN_IN,2.25))
    for ax,setting,xcol,xlabel in zip(axes,["common_d","deployed"],["solve_p95_s","e2e_p95_s"],["Solve P95 [ms]","E2E P95 [ms]"]):
        sub=rows[rows.evaluation_set.eq(setting)]
        for variant in ["ProjectionOff","FullCompiled","TwoStageCompiled"]:
            data=sub[sub.variant.eq(variant)]; ax.scatter(data[xcol]*1000,data.tcp_rmse_m*1000,label=variant,s=18,alpha=.7)
        ax.set_xlabel(xlabel); ax.set_ylabel("TCP RMSE [mm]"); ax.set_title("Common-D" if setting=="common_d" else "Deployed"); finish_axis(ax); ax.legend(fontsize=6.2)
    fig.subplots_adjust(wspace=.28,left=.075,right=.995,bottom=.23,top=.90); save_figure(fig,sup/"s3_projection_choice")


def main() -> None:
    args=parse_args(); apply_style()
    output=args.output_dir.resolve(); data=args.figure_data_dir.resolve(); output.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)
    main_csv=args.paper_root/"summaries"/"main.csv"; main_rows=pd.read_csv(main_csv); entries=index_entries(args.paper_root,"main")
    seed,paths,_=representative(main_rows,entries,args.legacy_ik_root)
    arrays={name:load_npz(path) for name,path in paths.items()}
    print(f"Representative case: nominal circle seed {seed}")
    for name,path in paths.items(): print(f"  {name}: {path}")
    fig2(output,data,arrays,seed,paths)
    supplementary_robustness_forest(output,data,args.paper_root)
    supplementary(output,data,args.suite_root)
    limitation=output.parent/"PRISM_EXPORT_LIMITATION.md"
    limitation.write_text("# GraphPad Prism export limitation\n\nNo GraphPad Prism executable and no `.prism`, `.pzf`, or `.pzfx` template was found in the audited workspace. pyPRISM is a Polymer Reference Interaction Site Model calculation package, not a GraphPad Prism project writer. The figure-data bundles provide tidy and wide CSV files plus metadata so that a supplied Prism 10/11 project or PZFX template can be populated without altering statistics. No native Prism file was fabricated.\n",encoding="utf-8")
    print(f"Wrote verified Fig. 2 and supplementary figures to {output}; source data to {data}")


if __name__=="__main__": main()
