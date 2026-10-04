from __future__ import annotations

import argparse
from pathlib import Path

from execution.commands import conda_python
from paths import SRC_ROOT

def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_runtime_args(parser: argparse.ArgumentParser, *, prefix: str = "pp-", include_batch_scale: bool = True) -> None:
    parser.add_argument(f"--{prefix}render-dpi", type=positive_int, default=200,
                        help="Target official PDF render DPI; oversized pages retain PaddleX's pixel limit.")
    if include_batch_scale:
        parser.add_argument(f"--{prefix}batch-scale", type=positive_int, default=1,
                            help="Page-batch route: multiply page/layout and selected recognition batches; never add model replicas.")


def build_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    cmd = conda_python("gdb_paddle") + [
        str(SRC_ROOT / "methods" / "pp_structurev3" / "worker.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / "pp_structurev3"),
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--device",
        args.pp_device,
        "--render-dpi",
        str(args.pp_render_dpi),
        "--batch-scale",
        str(args.pp_batch_scale),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd
