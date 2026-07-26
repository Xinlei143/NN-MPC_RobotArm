"""Run exactly the 20 FullVirtual ablation anchors without legacy-cache reuse."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.paper_experiments.revision_evidence import select_fullvirtual_cases
from scripts.paper_experiments.workflow import _run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if checkout != args.expected_commit or dirty:
        raise ValueError(
            f"Frozen rerun requires clean checkout {args.expected_commit}; "
            f"got commit={checkout}, dirty={bool(dirty)}"
        )
    actual = manifest["environment"]["git_commit"]
    if actual != args.expected_commit:
        raise ValueError(f"Manifest commit {actual} does not match {args.expected_commit}")
    output = Path(args.output_root).resolve()
    empty = output / "_no_legacy_reuse"
    empty.mkdir(parents=True, exist_ok=True)
    entries = [
        _run_case(output, manifest, case, args.resume, "ablation", empty, empty)
        for case in select_fullvirtual_cases(manifest)
    ]
    path = output / "runs/indexes/fullvirtual_frozen.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"suite": "fullvirtual_frozen", "entries": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(entries)} fresh FullVirtual cases to {path}")


if __name__ == "__main__":
    main()
