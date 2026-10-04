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
from methods.pp_structurev3.commands import add_runtime_args
from methods.pp_structurev3.pipeline import load_ppstructure_runtime, process_pdf_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent PP-StructureV3 worker consuming a shared PDF queue."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--lang", type=str)
    add_runtime_args(parser, prefix="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()

    runtime = load_ppstructure_runtime(args)

    def process_job(_claimed_path, pdf_paths):
        status = {}
        failed = process_pdf_paths(args, pdf_paths, runtime=runtime, job_status=status)
        return failed, status

    should_exit_failed = run_pdf_queue_worker(
        queue_dir=args.queue_dir,
        worker_id=args.worker_id,
        input_root=args.input_root,
        output_root=args.output_root,
        fail_fast=args.fail_fast,
        process_job=process_job,
    )
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
