from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python
from paths import SRC_ROOT

YOUTU = get_method_version("youtu")


def build_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    if args.youtu_ocr_batch_size < 1:
        raise ValueError("Youtu OCR query batch size must be positive")
    return conda_python(args.youtu_env) + [
        str(SRC_ROOT / "methods" / "youtu" / "worker.py"),
        "--input-root", args.input_root,
        "--output-root", str(Path(args.output_root_base) / "youtu"),
        "--queue-dir", str(queue_dir),
        "--worker-id", worker_id,
        "--model-path", args.youtu_model_path or str(YOUTU.model_path),
        "--dpi", "200",
        "--ocr-batch-size", str(args.youtu_ocr_batch_size),
        *(["--skip-existing"] if args.skip_existing else []),
    ]
