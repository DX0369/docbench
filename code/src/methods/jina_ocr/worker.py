#!/usr/bin/env python3
"""Resident Jina FastMTP worker with bounded global page requests."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (
    append_error_payload, build_error_payload, build_image_boxes_payload,
    build_page_records, concat_document_markdown, document_completed, error_context,
    get_doc_output_dir, pdf_page_count, render_pdf_page, save_markdown_bundle, write_json,
)
from execution.page_pool import run_page_pool
from execution.workers import run_pdf_queue_worker
from methods.jina_ocr.runtime import JinaRuntime, STRICT_PROMPT
from paths import TMP_ROOT

METHOD = "jina_ocr_v1"
PROFILE = "jina_ocr_v1_vllm_fastmtp_page_pool"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("input-root", "output-root", "queue-dir", "model-path"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--page-concurrency", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if min(args.dpi, args.max_tokens, args.page_concurrency) < 1 or args.max_model_len <= args.max_tokens:
        parser.error("Require positive DPI/tokens/P and context greater than generation limit")
    for key in ("input_root", "output_root", "queue_dir", "model_path"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def process_pdf_paths(args, runtime, pdfs):
    def output(pdf):
        return get_doc_output_dir(args.input_root, args.output_root, pdf)

    def should_skip(pdf):
        directory = output(pdf)
        if document_completed(directory):
            if not args.skip_existing:
                raise FileExistsError(f"Existing result requires --skip-existing: {directory}")
            return True
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Preserve partial/failed output; use a new run: {directory}")
        return False

    def run_page(pdf, page):
        from PIL import Image
        started = time.perf_counter()
        with error_context("page", page_number=page):
            with render_pdf_page(pdf, page, args.dpi, TMP_ROOT / "runtime/jina_ocr") as path:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                try:
                    text, reason = runtime.generate(image)
                finally:
                    image.close()
        return text, time.perf_counter() - started, reason

    def export(pdf, results, started):
        pages = [item[0] for item in results]
        directory = output(pdf)
        save_markdown_bundle(directory, pages, concat_document_markdown(pages))
        write_json(directory / "metrics.json", {
            "method": "Jina-OCR-v1", "method_id": METHOD, "execution_profile": PROFILE,
            "backend": "vllm_async_fastmtp", "pdf_path": str(pdf),
            "output_dir": str(directory), "model_path": str(args.model_path),
            "page_count": len(pages), "page_concurrency": args.page_concurrency,
            "dpi": args.dpi, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
            "prompt_id": "official_benchmark_strict", "prompt": STRICT_PROMPT,
            "temperature": 0.0, "repetition_penalty": 1.05,
            "repetition_detection": {"max_pattern_size": 35, "min_pattern_size": 35, "min_count": 10},
            "speculative_method": "eagle", "num_speculative_tokens": 3,
            "mtp_heads": 1, "mtp_recursive": True,
            "doc_total_seconds": round(time.perf_counter() - started, 6),
            "document_markdown_path": "document.md",
            "pages": build_page_records(pages, [item[1] for item in results]),
            "warnings": [{"page_number": i, "reason": "generation_length_limit"}
                         for i, item in enumerate(results, 1) if item[2] == "length"],
            "asset_policy": "no_image_assets",
        })
        write_json(directory / "image_boxes.json", build_image_boxes_payload(
            method="Jina-OCR-v1", pdf_path=pdf, entries=[],
            notes=["Strict official prompt ignores graphical content; no stable boxes exposed."],
        ))

    def record_error(pdf, exc):
        append_error_payload(output(pdf) / "error.json", build_error_payload(
            "Jina-OCR-v1", pdf, exception=exc,
            extra={"method_id": METHOD, "execution_profile": PROFILE},
        ))
        print(f"[error] {pdf}: {exc}", file=sys.stderr, flush=True)

    return run_page_pool(
        pdfs, concurrency=args.page_concurrency, page_count=pdf_page_count,
        should_skip=should_skip, run_page=run_page, export_document=export,
        record_error=record_error, fail_fast=args.fail_fast,
    )


def main():
    args = parse_args()
    runtime = JinaRuntime(args)
    try:
        failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir, worker_id=args.worker_id,
            input_root=args.input_root, output_root=args.output_root, fail_fast=args.fail_fast,
            process_job=lambda _job, pdfs: process_pdf_paths(args, runtime, pdfs),
        )
    finally:
        runtime.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
