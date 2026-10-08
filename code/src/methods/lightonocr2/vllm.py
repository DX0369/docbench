#!/usr/bin/env python3
"""Official LightOnOCR-2 single-page requests through a resident vLLM service."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import (  # noqa: E402
    append_error_payload,
    build_error_payload,
    build_image_boxes_payload,
    build_page_records,
    concat_document_markdown,
    document_completed,
    get_doc_output_dir,
    remove_file_if_exists,
    pdf_page_count,
    render_pdf_page,
    save_markdown_bundle,
    write_json,
)
from config.execution_profiles import resolve_execution_profile  # noqa: E402
from execution.page_pool import run_page_pool  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from image_vllm_client import build_openai_client, content_to_text, image_to_data_url  # noqa: E402
from paths import TMP_ROOT  # noqa: E402


METHOD_ID = "lightonocr2_1b"
DISPLAY_NAME = "LightOnOCR-2-1B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightOnOCR-2 one-page-at-a-time through an OpenAI-compatible vLLM service."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="lightonocr2")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-image-size", type=int, default=1540)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--page-concurrency", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1 or args.max_image_size < 1:
        parser.error("--dpi and --max-image-size must be positive")
    if args.max_tokens < 1 or args.page_concurrency < 1:
        parser.error("--max-tokens and --page-concurrency must be positive")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be in [0, 2]")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    return args


def request_page(args: argparse.Namespace, client: Any, image_path: Path) -> str:
    # The official LightOnOCR-2 vLLM example sends an image-only user message;
    # do not inject the generic OCR prompt used by other VLM adapters.
    response = client.chat.completions.create(
        model=args.model_name,
        messages=[{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)},
            }],
        }],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=False,
    )
    return content_to_text(response.choices[0].message.content).strip()


def run_page(args: argparse.Namespace, client: Any, pdf: Path, page: int):
    started = time.perf_counter()
    render_root = TMP_ROOT / "runtime" / "lightonocr2"
    with render_pdf_page(pdf, page, args.dpi, render_root) as image_path:
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        longest = max(image.size)
        try:
            if longest > args.max_image_size:
                scale = args.max_image_size / longest
                resized = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                resized.save(image_path, format="PNG")
                resized.close()
        finally:
            image.close()
        rendered = round(time.perf_counter() - started, 6)
        text = request_page(args, client, image_path)
    return text, round(time.perf_counter() - started, 6), rendered


def export_document(args: argparse.Namespace, pdf: Path, results, started: float) -> None:
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    pages = [item[0] for item in results]
    latencies = [item[1] for item in results]
    save_markdown_bundle(output, pages, concat_document_markdown(pages))
    write_json(output / "metrics.json", {
        "method": DISPLAY_NAME,
        "method_id": METHOD_ID,
        "execution_profile": resolve_execution_profile(METHOD_ID).profile_id,
        "backend": "vllm",
        "input_mode": "official_single_page_image",
        "pdf_path": str(pdf),
        "output_dir": str(output),
        "server_url": args.server_url,
        "model_name": args.model_name,
        "dpi": args.dpi,
        "max_image_size": args.max_image_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "page_concurrency": args.page_concurrency,
        "page_count": len(pages),
        "page_render_seconds_sum": round(sum(item[2] for item in results), 6),
        "page_inference_seconds_sum": round(sum(item[1] - item[2] for item in results), 6),
        "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md",
        "pages": build_page_records(pages, latencies),
        "warnings": [],
        "asset_policy": "no_image_assets",
        "notes": [
            "One resident official vLLM service; each request contains exactly one page image.",
            "The global page pool may submit pages from different PDFs concurrently and restores per-PDF order.",
            "Official LightOnOCR-2 vLLM generation settings are max_tokens=4096, temperature=0.2, top_p=0.9.",
            "Rendered pages are resized to the official longest-edge target when necessary.",
            "No raw response or image asset is retained.",
        ],
    })
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=DISPLAY_NAME,
        pdf_path=pdf,
        entries=[],
        notes=["LightOnOCR-2 does not expose benchmark-stable image boxes in this route."],
    ))
    remove_file_if_exists(output / "error.json")


def should_skip(args: argparse.Namespace, pdf: Path) -> bool:
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not document_completed(output):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Existing partial/failed output must be retained; use a new run: {output}")
        return False
    if not args.skip_existing:
        raise FileExistsError(f"Existing result requires --skip-existing: {output}")
    return True


def record_error(args: argparse.Namespace, pdf: Path, exc: BaseException) -> None:
    append_error_payload(
        get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
        build_error_payload(
            DISPLAY_NAME,
            pdf,
            exception=exc,
            extra={
                "method_id": METHOD_ID,
                "execution_profile": resolve_execution_profile(METHOD_ID).profile_id,
                "backend": "vllm",
                "page_concurrency": args.page_concurrency,
            },
        ),
    )
    print(f"[error] {pdf}: {exc}", file=sys.stderr, flush=True)


def process_pdf_paths(args: argparse.Namespace, client: Any, pdf_paths: list[Path]) -> bool:
    return run_page_pool(
        pdf_paths,
        concurrency=args.page_concurrency,
        page_count=pdf_page_count,
        should_skip=lambda pdf: should_skip(args, pdf),
        run_page=lambda pdf, page: run_page(args, client, pdf, page),
        export_document=lambda pdf, results, started: export_document(args, pdf, results, started),
        record_error=lambda pdf, exc: record_error(args, pdf, exc),
        fail_fast=args.fail_fast,
    )


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    client = build_openai_client(args.server_url, args.timeout)
    try:
        failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir,
            worker_id=args.worker_id,
            fail_fast=args.fail_fast,
            input_root=args.input_root,
            output_root=args.output_root,
            process_job=lambda _claimed_path, pdfs: process_pdf_paths(args, client, pdfs),
        )
    finally:
        client.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
