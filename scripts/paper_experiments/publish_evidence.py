"""Build the compact public evidence bundle for the ROBIO 2026 paper."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "evidence" / "robio2026"

# Raw rollouts and caches are intentionally excluded. These files contain the
# compact statistics, audits, and figures needed to check the paper's claims.
SOURCES = {
    "abb/summaries": (
        "outputs/paper_final/summaries/*",
        "outputs/paper_final/statistics/*",
    ),
    "abb/model_validation": (
        "outputs/paper_final/diagnostics/formal_mpc_replay/*.csv",
        "outputs/paper_final/diagnostics/gru_validation/*.csv",
        "outputs/paper_final/diagnostics/gru_validation/*.json",
    ),
    "abb/figures": ("outputs/paper_final/figures/*",),
    "abb/manifests": ("outputs/paper_final/manifests/*.json",),
    "ur5e/summaries": ("outputs/paper_ur5e_v2/summaries/*",),
    "ur5e/manifests": (
        "outputs/paper_ur5e_v2/manifests/*.json",
        "outputs/paper_ur5e_v2/freeze_audit.json",
    ),
    "claim_evidence/summaries": (
        "outputs/paper_claim_evidence_v2/summaries/*",
    ),
    "claim_evidence/manifests": (
        "outputs/paper_claim_evidence_v2/claim_manifest.json",
        "outputs/paper_claim_evidence_v2/unit_tests.log",
    ),
    "robustness_and_timing": (
        "outputs/paper_revision_v1/statistics/*.json",
        "outputs/paper_revision_v1/statistics/*.csv",
    ),
}

TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".txt"}
PRESERVED_PUBLIC_FILES = {"README.md", "technical_supplement.tex", "technical_supplement.pdf"}
MANIFESTED_PUBLIC_FILES = ("technical_supplement.tex", "technical_supplement.pdf")


def _portable_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() not in TEXT_SUFFIXES:
        shutil.copy2(source, destination)
        return
    text = source.read_text(encoding="utf-8")
    # Frozen manifests retain their original hashes but use portable paths in
    # the public copy. No numerical or configuration field is changed.
    text = text.replace(str(ROOT), ".")
    destination.write_text(text, encoding="utf-8")


def build() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for child in DESTINATION.iterdir():
        if child.name in PRESERVED_PUBLIC_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    copied: list[dict[str, object]] = []
    for relative_directory, patterns in SOURCES.items():
        seen: set[Path] = set()
        for pattern in patterns:
            for source in sorted(ROOT.glob(pattern)):
                if not source.is_file() or source in seen:
                    continue
                seen.add(source)
                destination = DESTINATION / relative_directory / source.name
                _portable_copy(source, destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                copied.append(
                    {
                        "path": destination.relative_to(DESTINATION).as_posix(),
                        "bytes": destination.stat().st_size,
                        "sha256": digest,
                        "source": source.relative_to(ROOT).as_posix(),
                    }
                )
    for name in MANIFESTED_PUBLIC_FILES:
        public_file = DESTINATION / name
        if not public_file.is_file():
            continue
        copied.append(
            {
                "path": name,
                "bytes": public_file.stat().st_size,
                "sha256": hashlib.sha256(public_file.read_bytes()).hexdigest(),
                "source": "repository-maintained technical supplement",
            }
        )
    manifest = {
        "schema_version": 1,
        "description": "Compact public evidence for the ROBIO 2026 paper",
        "artifact_count": len(copied),
        "artifacts": copied,
        "excluded": [
            "raw rollouts",
            "candidate snapshot NPZ files",
            "runtime caches",
            "model checkpoints and normalizers",
            "paper source and PDF",
        ],
    }
    (DESTINATION / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
