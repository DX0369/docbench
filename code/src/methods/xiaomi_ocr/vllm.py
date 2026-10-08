#!/usr/bin/env python3
"""Resident Xiaomi whole-page worker using official input and postprocessing."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from common import (
    append_error_payload, build_error_payload, build_image_boxes_payload,
    build_page_records, concat_document_markdown, document_completed, error_context,
    get_doc_output_dir, pdf_page_count, remove_file_if_exists, render_pdf_page,
    save_markdown_bundle, write_json,
)
from config.execution_profiles import resolve_execution_profile
from execution.page_pool import run_page_pool
from execution.workers import run_pdf_queue_worker
from image_vllm_client import build_openai_client
from method_paths import method_repo_path
from paths import TMP_ROOT

# Import official helpers before the worker's ready barrier. No model, CLI,
# MCP server or region pipeline is started by these imports.
sys.path.insert(0, str(method_repo_path("xiaomi_ocr")))
from pipeline.e2e_img2md import PROMPTS, encode_image_base64
from postprocess.document import finalize_document

METHOD_ID = "xiaomi_ocr_0"
DISPLAY_NAME = "Xiaomi-OCR-0"
PROFILE_ID = resolve_execution_profile(METHOD_ID).profile_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--page-concurrency", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if min(args.dpi, args.max_tokens, args.page_concurrency) < 1:
        parser.error("DPI, max tokens and page concurrency must be positive")
    return args


def request_page(args, client, image_path):
    # Match official batch e2e messages, sampling and thinking settings. Do
    # not use infer_one(): it turns a successful empty response into failure.
    response = client.chat.completions.create(
        model=METHOD_ID,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPTS["document"]},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encode_image_base64(image_path),
            }},
        ]}],
        temperature=0.0, top_p=0.8, presence_penalty=0.0,
        max_tokens=args.max_tokens, timeout=600,
        extra_body={"repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    content = choice.message.content
    if content is not None and not isinstance(content, str):
        raise TypeError("Xiaomi response content must be text or null")
    # Empty text and length termination are execution successes; SDK/request
    # errors and malformed response structure are never suppressed.
    return finalize_document("" if content is None else content.strip()), choice.finish_reason


def run_page(args, client, pdf, page):
    started = time.perf_counter()
    with error_context("page", page_number=page):
        with render_pdf_page(pdf, page, args.dpi, TMP_ROOT / "runtime" / "xiaomi_ocr") as image:
            rendered = time.perf_counter() - started
            text, finish_reason = request_page(args, client, image)
    return text, time.perf_counter() - started, rendered, finish_reason


def export_document(args, pdf, results, started):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    pages = [item[0] for item in results]
    warnings = [{"page_number": page, "reason": "generation_length_limit"}
                for page, item in enumerate(results, 1) if item[3] == "length"]
    save_markdown_bundle(output, pages, concat_document_markdown(pages))
    write_json(output / "metrics.json", {
        "method": DISPLAY_NAME, "method_id": METHOD_ID,
        "execution_profile": PROFILE_ID, "backend": "vllm",
        "input_mode": "official_batch_single_page_image",
        "pdf_path": str(pdf), "output_dir": str(output),
        "server_url": args.server_url, "model_name": METHOD_ID,
        "dpi": args.dpi, "max_image_size": 2048,
        "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 0.8,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "enable_thinking": False, "request_timeout_seconds": 600,
        "page_concurrency": args.page_concurrency, "page_count": len(pages),
        "page_render_seconds_sum": round(sum(item[2] for item in results), 6),
        "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md",
        "pages": build_page_records(pages, [item[1] for item in results]),
        "warnings": warnings, "asset_policy": "no_image_assets",
        "notes": [
            "Official batch document prompt, RGB/2048px encoding and finalize_document postprocessing.",
            "Project PDF rendering is 200 DPI by default; the official PDF demo uses 150 DPI.",
            "One image per request; global page pool restores per-PDF page order. No region/joint route.",
            "Empty/length responses are execution successes; SDK default transport retries remain enabled.",
            "No raw response or image asset is retained.",
        ],
    })
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=DISPLAY_NAME, pdf_path=pdf, entries=[],
        notes=["The official whole-page route does not expose stable image boxes."],
    ))
    remove_file_if_exists(output / "error.json")


def should_skip(args, pdf):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not document_completed(output):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Preserve partial/failed output; use a new run: {output}")
        return False
    if not args.skip_existing:
        raise FileExistsError(f"Existing result requires --skip-existing: {output}")
    return True


def record_error(args, pdf, exc):
    append_error_payload(
        get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
        build_error_payload(DISPLAY_NAME, pdf, exception=exc, extra={
            "method_id": METHOD_ID, "execution_profile": PROFILE_ID,
            "backend": "vllm", "page_concurrency": args.page_concurrency,
        }),
    )
    print(f"[error] {pdf}: {exc}", file=sys.stderr, flush=True)


def process_pdf_paths(args, client, pdf_paths):
    return run_page_pool(
        pdf_paths, concurrency=args.page_concurrency, page_count=pdf_page_count,
        should_skip=lambda pdf: should_skip(args, pdf),
        run_page=lambda pdf, page: run_page(args, client, pdf, page),
        export_document=lambda pdf, results, started: export_document(args, pdf, results, started),
        record_error=lambda pdf, exc: record_error(args, pdf, exc), fail_fast=args.fail_fast,
    )


def main():
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    client = build_openai_client(args.server_url, 600)
    try:
        failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir, worker_id=args.worker_id,
            fail_fast=args.fail_fast, input_root=args.input_root, output_root=args.output_root,
            process_job=lambda _job, pdfs: process_pdf_paths(args, client, pdfs),
        )
    finally:
        client.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
