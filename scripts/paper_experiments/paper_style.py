"""Shared IEEE/ROBIO figure style and portable vector-export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE_COLUMN_IN = 3.45
DOUBLE_COLUMN_IN = 7.10
PNG_DPI = 300

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "Desired": {"color": "#111111", "linestyle": "--", "linewidth": 1.8, "marker": None},
    "Raw Direct IK": {"color": "#A0A0A0", "linestyle": ":", "linewidth": 1.15, "marker": "x"},
    "Projected Direct IK": {"color": "#666666", "linestyle": "-.", "linewidth": 1.2, "marker": "D"},
    "Preview IK": {"color": "#CC79A7", "linestyle": ":", "linewidth": 1.2, "marker": "v"},
    "Ideal zero-delay logical MPC": {"color": "#56B4E9", "linestyle": (0, (7, 3)), "linewidth": 1.3, "marker": "P"},
    "NaiveDelayed": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.65, "marker": "o"},
    "FullVirtual": {"color": "#0072B2", "linestyle": "--", "linewidth": 1.65, "marker": "s"},
    "ThreadedASAP": {"color": "#009E73", "linestyle": "-", "linewidth": 1.7, "marker": "^"},
    "Alignment-only": {"color": "#E69F00", "linestyle": "--", "linewidth": 1.5, "marker": "X"},
    "Alignment+Reanchor": {"color": "#009E73", "linestyle": "-.", "linewidth": 1.5, "marker": "d"},
}


def apply_style() -> None:
    """Apply the sole plotting style used by every final-paper figure."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.8,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.1,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.markersize": 4.1,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.formatter.useoffset": False,
    })


def method_style(name: str, **overrides: Any) -> dict[str, Any]:
    value = dict(METHOD_STYLES[name])
    value.update(overrides)
    return value


def finish_axis(ax: Any, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color="#B9B9B9", alpha=0.18, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: Any, label: str) -> None:
    ax.text(-0.13, 1.035, label, transform=ax.transAxes, fontweight="bold", fontsize=8.4,
            va="bottom", ha="left")


def save_figure(fig: Any, destination: Path, *, png: bool = True) -> list[Path]:
    """Write editable SVG/PDF and the mandated 300-dpi review PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs = [destination.with_suffix(".pdf"), destination.with_suffix(".svg")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.015)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.015)
    if png:
        outputs.append(destination.with_suffix(".png"))
        fig.savefig(outputs[-1], dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return outputs
