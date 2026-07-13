from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_XML = "ABB_IRB2400.xml"


def resolve_project_path(path: str, project_root: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return project_root / expanded
