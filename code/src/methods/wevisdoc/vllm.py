#!/usr/bin/env python3
"""Official WeVisDoc single-page requests through a resident vLLM service."""
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
    pdf_page_count,
    remove_file_if_exists,
    render_pdf_page,
    save_markdown_bundle,
    write_json,
)
from config.execution_profiles import resolve_execution_profile  # noqa: E402
from execution.page_pool import run_page_pool  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from image_vllm_client import (  # noqa: E402
    build_openai_client,
    content_to_text,
)
from method_paths import method_repo_path  # noqa: E402
from paths import TMP_ROOT  # noqa: E402

# Reuse the official prompt and message construction rather than introducing a
# benchmark-specific OCR prompt. The upstream checkout is read-only here.
WEVISDOC_ROOT = method_repo_path("wevisdoc")
if str(WEVISDOC_ROOT) not in sys.path:
    sys.path.insert(0, str(WEVISDOC_ROOT))
from wevisdoc.common import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, build_messages  # noqa: E402


METHOD_ID = "wevisdoc_2b"
DISPLAY_NAME = "WeVisDoc-2B"
REQUEST_TIMEOUT = 600.0
DEFAULT_DPI = 200
MAX_TOKENS = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WeVisDoc one-page-at-a-time through an OpenAI-compatible vLLM service."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="wevisdoc")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--page-concurrency", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1 or args.page_concurrency < 1 or args.timeout <= 0:
        parser.error("DPI, page concurrency and timeout must be positive")
    return args


def request_page(args: argparse.Namespace, client: Any, image_path: Path) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=args.model_name,
        messages=build_messages(
            image_path,
            DEFAULT_PROMPT,
            DEFAULT_SYSTEM_PROMPT,
        ),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stream=False,
    )
    if not response.choices:
        raise RuntimeError("WeVisDoc service returned no choices")
    choice = response.choices[0]
    finish_reason = str(getattr(choice, "finish_reason", None) or "unknown")
    # Empty content and length termination are recorded as output warnings;
    # transport, response-shape and service failures still propagate.
    text = content_to_text(getattr(choice.message, "content", None)).strip()
    return text, finish_reason


def run_page(args: argparse.Namespace, client: Any, pdf: Path, page: int):
    started = time.perf_counter()
    render_root = TMP_ROOT / "runtime" / "wevisdoc"
    with render_pdf_page(pdf, page, args.dpi, render_root) as image_path:
        rendered = time.perf_counter() - started
        text, finish_reason = request_page(args, client, image_path)
    return text, time.perf_counter() - started, rendered, finish_reason


def export_document(args: argparse.Namespace, pdf: Path, results, started: float) -> None:
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    pages = [item[0] for item in results]
    warnings = []
    for page_number, item in enumerate(results, start=1):
        finish_reason = item[3]
        if finish_reason != "stop":
            warnings.append({"page_number": page_number, "reason": finish_reason})
        if not item[0].strip():
            warnings.append({"page_number": page_number, "reason": "empty_content"})
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
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "limit_mm_per_prompt": {"image": 1, "video": 0},
        "page_concurrency": args.page_concurrency,
        "page_count": len(pages),
        "page_render_seconds_sum": round(sum(item[2] for item in results), 6),
        "page_inference_seconds_sum": round(sum(item[1] - item[2] for item in results), 6),
        "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md",
        "pages": build_page_records(pages, [item[1] for item in results]),
        "warnings": warnings,
        "asset_policy": "no_image_assets",
        "notes": [
            "One resident official vLLM service; each request contains exactly one page image.",
            "The page pool can submit pages from different PDFs and restores per-PDF order.",
            "The document output is page-wise concatenation, not whole-document joint inference.",
            "No raw response or image asset is retained.",
        ],
    })
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=DISPLAY_NAME,
        pdf_path=pdf,
        entries=[],
        notes=["The official WeVisDoc page route does not expose stable image boxes."],
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
