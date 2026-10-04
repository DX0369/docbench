from __future__ import annotations

import argparse
from pathlib import Path

from execution.commands import conda_python
from paths import SRC_ROOT


def build_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    if type(args.marker_workers) is not int or args.marker_workers < 1:
        raise ValueError("--marker-workers must be a positive integer")
    cmd = conda_python("gdb_marker") + [
        str(SRC_ROOT / "methods" / "marker" / "worker.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / "marker"),
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--torch-device",
        "cuda",
        "--workers",
        str(args.marker_workers),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd
