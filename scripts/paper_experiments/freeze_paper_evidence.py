"""Assemble immutable final ROBIO evidence and hash every published artifact."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from scripts.experiment_utils import file_identity, load_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--paper-root", required=True)
    value.add_argument("--audit-root", required=True)
    value.add_argument("--ik-comparison-root", required=True)
    value.add_argument("--figure-root", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--report-commit", default=None)
    return value


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    args = parser().parse_args()
    paper = Path(args.paper_root).resolve()
    audit = Path(args.audit_root).resolve()
    ik = Path(args.ik_comparison_root).resolve()
    figures = Path(args.figure_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen evidence: {output}")
    manifest = load_json(paper / "manifests" / "paper.json")
    _copy(paper / "manifests" / "paper.json", output / "manifests" / "control_manifest.json")
    _copy(audit / "audit_summary.json", output / "manifests" / "evidence_inventory.json")
    for source in sorted((paper / "summaries").glob("*")):
        if source.suffix == ".csv":
            _copy(source, output / "summaries" / source.name)
        elif source.suffix == ".json":
            _copy(source, output / "statistics" / source.name)
    for source in sorted(ik.glob("*")):
        if source.suffix == ".csv":
            _copy(source, output / "summaries" / source.name)
        elif source.suffix == ".json":
            _copy(source, output / "statistics" / source.name)
    for source in sorted(figures.glob("*")):
        if source.suffix in {".pdf", ".json"}:
            _copy(source, output / "figures" / source.name)
    identities = [
        file_identity(path)
        for path in sorted(output.glob("**/*"))
        if path.is_file()
    ]
    analysis_manifest = {
        "schema_version": 1,
        "paper_control_commit": manifest["paper_control_commit"],
        "analysis_commit": manifest["analysis_commit"],
        "report_commit": args.report_commit or _commit(),
        "source_paper_manifest": file_identity(paper / "manifests" / "paper.json"),
        "artifacts": identities,
    }
    path = output / "manifests" / "analysis_manifest.json"
    path.write_text(json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
