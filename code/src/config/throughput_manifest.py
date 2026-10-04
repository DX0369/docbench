"""Read the single canonical fixed-subset manifest; paths are relative to DATA_ROOT."""
from __future__ import annotations

import json
from pathlib import Path

from paths import DATA_ROOT, SRC_ROOT

DEFAULT_THROUGHPUT_MANIFEST = SRC_ROOT.parent / "config" / "throughput_manifests" / "rtx3090_20.json"


def load_throughput_manifest(path: Path, *, data_root: Path = DATA_ROOT) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1 or payload["path_root"] != "GDB_DATA_ROOT":
        raise ValueError(f"Unsupported throughput manifest: {path}")
    selected = payload["selected"]
    if not selected or payload["selected_count"] != len(selected):
        raise ValueError("Manifest must contain the declared, nonempty document list")
    paths = []
    for entry in selected:
        relative = Path(entry["pdf_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF path relative to GDB_DATA_ROOT: {relative}")
        if type(entry["page_count"]) is not int or entry["page_count"] <= 0:
            raise ValueError(f"Invalid page_count: {relative}")
        source = (data_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.append(source)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate PDFs in fixed throughput manifest")
    return paths
