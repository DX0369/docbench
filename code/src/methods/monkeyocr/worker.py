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
from methods.monkeyocr.pipeline import (
    DEFAULT_CONFIG_PATH,
    close_monkeyocr_runtime,
    load_monkeyocr_runtime,
    prepare_runtime_config,
    process_pdf_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent MonkeyOCR worker. Initializes MonkeyOCR once and "
            "consumes PDF job files from a shared queue directory."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-id", type=str, required=True)
    parser.add_argument("--method-display-name", type=str, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--region-concurrency", type=int, default=64,
                        help="vllm_async in-flight region limit; official default: 64.")
    parser.add_argument("--model-parallelism", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--chat-weight-path", type=str)
    parser.add_argument("--pred-abandon", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    args.config_path = args.config_path.resolve()
    if args.models_dir is not None:
        args.models_dir = args.models_dir.resolve()

    runtime_config_path = prepare_runtime_config(args)
    runtime = None
    try:
        runtime = load_monkeyocr_runtime(runtime_config_path)
        should_exit_failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir,
            worker_id=args.worker_id,
            input_root=args.input_root,
            output_root=args.output_root,
            fail_fast=args.fail_fast,
            process_job=lambda _claimed_path, pdf_paths: process_pdf_paths(
                args,
                pdf_paths,
                runtime=runtime,
            ),
        )
        if should_exit_failed:
            raise SystemExit(1)
    finally:
        try:
            close_monkeyocr_runtime(runtime)
        finally:
            runtime_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
