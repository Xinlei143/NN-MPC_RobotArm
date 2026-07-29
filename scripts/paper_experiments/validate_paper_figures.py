"""Fail-fast validation for the final ROBIO figure bundle and manuscript integration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_style import METHOD_STYLES


MAIN = ("fig1_activation_aligned_architecture", "fig2_representative_tracking", "fig4_delay_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, help="Figure bundle to validate; defaults to <paper-dir>/figures")
    parser.add_argument("--figure-data-dir", type=Path, help="Source-data bundle; defaults to <paper-dir>/figure_data")
    args = parser.parse_args(); paper=args.paper_dir.resolve(); figures=(args.figures_dir or paper/"figures").resolve(); data=(args.figure_data_dir or paper/"figure_data").resolve(); tex=(paper/"main.tex").read_text(encoding="utf-8")
    failures=[]
    for name in MAIN:
        suffixes = (".svg", ".png", ".source_manifest.json") if name == "fig1_activation_aligned_architecture" else (".pdf", ".svg", ".png", ".source_manifest.json")
        for suffix in suffixes:
            if not (figures/f"{name}{suffix}").is_file(): failures.append(f"missing {name}{suffix}")
        if name.startswith("fig2") and not (data/"fig2"/"metadata.json").is_file(): failures.append("missing Fig. 2 source data")
    if "no_feedback" in tex: failures.append("main.tex still exposes no_feedback rather than Alignment+Reanchor")
    if "Latency [ms]" in tex: failures.append("main.tex contains ambiguous Latency label")
    forbidden = {
        "77505a5": "obsolete evidence-only commit",
        "clean commit": "unverifiable clean-worktree claim",
        "logical upper bound": "ZeroDelay must be described as a logical reference",
        "ThreadedASAP": "ambiguous legacy method name",
    }
    for phrase, reason in forbidden.items():
        if phrase in tex:
            failures.append(f"main.tex contains {phrase!r}: {reason}")
    required = ("tab:gru-validation", "tab:protocol-semantics", "tab:ablation", "fig2_representative_tracking", "fig4_delay_sweep")
    for phrase in required:
        if phrase not in tex:
            failures.append(f"main.tex is missing required manuscript marker {phrase!r}")
    for name in MAIN:
        if name not in tex: failures.append(f"main.tex does not reference {name}")
    source=load_json(figures/"fig2_representative_tracking.source_manifest.json")
    if source.get("seed") != 3: failures.append("representative seed is not the frozen seed 3")
    for folder in ("fig2", "fig3", "fig4", "fig5", "supplementary/s1"):
        for filename in ("tidy.csv", "wide.csv", "metadata.json", "README.md"):
            if not (data/folder/filename).is_file(): failures.append(f"missing source-data file {folder}/{filename}")
    encodings = [(style["linestyle"], style["marker"]) for style in METHOD_STYLES.values()]
    if len(encodings) != len(set(encodings)):
        failures.append("method styles do not retain unique line/marker encodings for grayscale/CVD reading")
    luminance = {}
    for name, style in METHOD_STYLES.items():
        rgb = tuple(int(style["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        luminance[name] = round((.2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]) / 255, 3)
    (figures / "figure_validation.json").write_text(json.dumps({
        "grayscale_check": "Every method has a unique line-style/marker pair; luminance is descriptive only.",
        "deuteranopia_check": "Method identity remains encoded by line style and marker, not color alone.",
        "luminance": luminance,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures: raise SystemExit("Figure validation failed:\n- " + "\n- ".join(failures))
    print("Figure bundle validation passed.")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__": main()
