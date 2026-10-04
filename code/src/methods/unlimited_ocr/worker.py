#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.workers import run_pdf_queue_worker  # noqa: E402
from methods.unlimited_ocr.client import client_runtime, parse_args, process_pdf_paths  # noqa: E402


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    # Imports, processor serialization and HTTP session setup precede ready.
    # All jobs share this client; only the service owns the GPU model.
    with client_runtime() as (session, logit_processor):
        should_exit_failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir,
            worker_id=args.worker_id,
            input_root=args.input_root,
            output_root=args.output_root,
            fail_fast=args.fail_fast,
            process_job=lambda _claimed_path, pdf_paths: process_pdf_paths(
                args, pdf_paths, session=session, logit_processor=logit_processor
            ),
        )
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
