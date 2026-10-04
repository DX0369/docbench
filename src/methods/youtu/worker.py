#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from execution.workers import run_pdf_queue_worker
from methods.youtu.parser import load_youtu_runtime, process_pdf_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent Youtu-Parsing worker consuming a shared PDF queue."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ocr-batch-size", type=int, default=5)
    parser.set_defaults(max_pixels=None, enable_angle_correct=False, angle_correct_model_path=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.ocr_batch_size < 1:
        parser.error("--ocr-batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    args.model_path = args.model_path.resolve()

    runtime = load_youtu_runtime(args)

    should_exit_failed = run_pdf_queue_worker(
        queue_dir=args.queue_dir,
        worker_id=args.worker_id,
        input_root=args.input_root,
        output_root=args.output_root,
        fail_fast=args.fail_fast,
        process_job=lambda _claimed_path, pdf_paths: process_pdf_paths(args, pdf_paths, runtime=runtime),
    )
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
