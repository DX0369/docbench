#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import error_context, get_doc_output_dir, runtime_unusable  # noqa: E402
from execution.trial import get_pdf_page_count  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from methods.teleocr.output import write_document_error, write_document_outputs  # noqa: E402
from methods.teleocr.runtime import load_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent TeleOCR vLLM async worker.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm-async-engine",), default="vllm-async-engine")
    parser.add_argument("--layout-mode", choices=("Detection",), default="Detection")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.95)
    parser.add_argument("--pdf-tools", default="pypdfium2", choices=("pypdfium2", "PyMuPDF"))
    parser.add_argument("--pdf-tools-workers", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.pdf_tools_workers < 1:
        parser.error("--pdf-tools-workers must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def process_pdf_paths(args: argparse.Namespace, pdf_paths: list[Path], *, runtime) -> bool:
    had_failure = False
    for pdf_path in pdf_paths:
        try:
            with error_context("process_document"):
                if args.skip_existing:
                    from common import document_completed
                    output = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
                    if document_completed(output):
                        continue
                expected_pages = get_pdf_page_count(pdf_path)
                started = time.perf_counter()
                middle_json = runtime.process_pdf(pdf_path)
                write_document_outputs(
                    args,
                    pdf_path,
                    middle_json,
                    expected_pages,
                    time.perf_counter() - started,
                )
                print(f"[done] {pdf_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            had_failure = True
            write_document_error(args, pdf_path, exc)
            print(f"[error] {pdf_path}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast or runtime_unusable(exc):
                raise
    return had_failure


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    args.model_path = args.model_path.resolve()

    runtime = load_runtime(args)
    try:
        failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir,
            worker_id=args.worker_id,
            input_root=args.input_root,
            output_root=args.output_root,
            fail_fast=args.fail_fast,
            process_job=lambda _claimed_path, pdf_paths: process_pdf_paths(
                args, pdf_paths, runtime=runtime
            ),
        )
        if failed:
            raise SystemExit(1)
    finally:
        warnings = runtime.close()
        for warning in warnings:
            print(f"[warning] {warning}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
