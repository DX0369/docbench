#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from common import (
    add_shard_args,
    apply_shard_to_pdf_paths,
    collect_pdf_paths,
    load_pdf_paths_from_file,
)
from image_vllm_client import build_openai_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qianfan-OCR through a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model-name", type=str, default="qianfan-ocr")
    parser.add_argument("--page-concurrency", type=int, default=1)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--allow-partial-success", action="store_true")
    parser.add_argument("--enable-think", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--limit", type=int)
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-cache-root", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if bool(args.queue_dir) != bool(args.worker_id):
        parser.error("--queue-dir and --worker-id must be provided together")
    return args


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()

    client = build_openai_client(base_url=args.server_url, timeout=args.timeout)
    if args.queue_dir is not None:
        from execution.workers import run_pdf_queue_worker
        with client:
            failed = run_pdf_queue_worker(
                queue_dir=args.queue_dir, worker_id=args.worker_id,
                fail_fast=args.fail_fast, input_root=args.input_root, output_root=args.output_root,
                allow_partial_success=args.allow_partial_success,
                process_job=lambda _path, pdfs: process_pdf_paths(args, client, pdfs),
            )
        if failed:
            raise SystemExit(1)
        return
    if args.pdf_list_file is not None:
        args.pdf_list_file = args.pdf_list_file.resolve()
        pdf_paths = load_pdf_paths_from_file(args.pdf_list_file, limit=args.limit)
    else:
        pdf_paths = collect_pdf_paths(args.input_root, limit=args.limit)
    pdf_paths = apply_shard_to_pdf_paths(
        pdf_paths,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    with client:
        had_failure = process_pdf_paths(args, client, pdf_paths)
    if had_failure:
        raise SystemExit(1)


def process_pdf_paths(args, client, pdf_paths):
    from methods.qianfan_ocr.page_pool import process_page_pool
    return process_page_pool(args, client, pdf_paths)


if __name__ == "__main__":
    main()
