"""Write transparent Prism-import CSV bundles; never fabricate Prism project files."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def write_bundle(directory: Path, *, tidy: list[dict[str, Any]], wide: list[dict[str, Any]],
                 metadata: dict[str, Any], readme: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "tidy.csv", tidy)
    write_csv(directory / "wide.csv", wide)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "README.md").write_text(readme.rstrip() + "\n", encoding="utf-8")
