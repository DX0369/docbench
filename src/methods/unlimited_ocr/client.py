#!/usr/bin/env python3
"""Shared Unlimited-OCR request/output implementation for the persistent worker."""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import contextmanager
import json
import mimetypes
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import requests

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import (  # noqa: E402
    append_error_payload,
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_image_boxes_payload,
    build_page_records,
    cleanup_render_dir,
    concat_document_markdown,
    get_doc_output_dir,
    get_image_size,
    remove_file_if_exists,
    remove_dir_if_exists,
    render_pdf_pages,
    render_pdf_page,
    save_markdown_bundle,
    write_json,
    write_text,
)
from execution.processes import http_service_ready  # noqa: E402
from execution.page_pool import run_page_pool  # noqa: E402
from execution.trial import get_pdf_page_count  # noqa: E402
from paths import TMP_ROOT  # noqa: E402


METHOD_ID = "unlimited_ocr_sglang"
METHOD_NAME = "Unlimited-OCR SGLang"
PROMPT = "Multi page parsing."
MAX_TOKENS = 16384
REFERENCE_RE = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL,
)
DET_ONLY_RE = re.compile(
    r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)", re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Unlimited-OCR through its official SGLang multi-page API."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-name", default="Unlimited-OCR")
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--request-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=1024)
    parser.add_argument("--request-mode", choices=("joint", "page_pool"), default="joint")
    parser.add_argument("--page-concurrency", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


@contextmanager
def client_runtime():
    from sglang.srt.sampling.custom_logit_processor import (
        DeepseekOCRNoRepeatNGramLogitProcessor,
    )

    logit_processor = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
    with requests.Session() as session:
        session.trust_env = False
        yield session, logit_processor


def reference_matches(text: str) -> list[tuple[str, str, str]]:
    matches = list(REFERENCE_RE.findall(text))
    seen = {match[0] for match in matches}
    for match in DET_ONLY_RE.findall(text):
        if match[0] not in seen:
            matches.append(match)
            seen.add(match[0])
    return matches


def parse_boxes(raw_boxes: str) -> list[list[float]]:
    try:
        parsed = ast.literal_eval(raw_boxes)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list) or not parsed:
        return []
    if len(parsed) == 4 and all(isinstance(value, (int, float)) for value in parsed):
        return [[float(value) for value in parsed]]
    boxes: list[list[float]] = []
    for item in parsed:
        if isinstance(item, list) and len(item) == 4 and all(
            isinstance(value, (int, float)) for value in item
        ):
            boxes.append([float(value) for value in item])
    return boxes


def split_raw_pages(raw_output: str) -> list[str]:
    """Return observed output segments, without inventing input-page coverage."""
    return [part.strip() for part in raw_output.split("<PAGE>") if part.strip()]


def clean_page_markdown(raw_page: str, page_index: int) -> tuple[str, list[dict[str, Any]]]:
    markdown = raw_page
    image_entries: list[dict[str, Any]] = []
    image_index = 0
    for full_match, raw_label, raw_boxes in reference_matches(raw_page):
        label = raw_label.strip()
        if label == "image" or "<|ref|>image<|/ref|>" in full_match:
            marker_name = f"page_{page_index - 1}_{image_index}.jpg"
            marker_expression = f"![](images/{marker_name})\n"
            markdown = markdown.replace(full_match, marker_expression, 1)
            for box in parse_boxes(raw_boxes):
                image_entries.append(
                    {
                        "page_index": page_index,
                        "marker_name": marker_name,
                        "marker_type": "markdown_image",
                        "page_marker_expression": marker_expression.strip(),
                        "page_marker_src": f"images/{marker_name}",
                        "relative_box_0_1000": [int(round(value)) for value in box],
                        "box_source": "unlimited_ocr_det_reference",
                        "block_label": label,
                    }
                )
            image_index += 1
        else:
            markdown = markdown.replace(full_match, "", 1)
    markdown = markdown.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:").strip()
    return (markdown + "\n" if markdown else ""), image_entries


def add_absolute_boxes(entries: list[dict[str, Any]], page_sizes: Sequence[tuple[int, int]]) -> None:
    for entry in entries:
        page_index = int(entry["page_index"])
        if page_index < 1 or page_index > len(page_sizes):
            continue
        page_width, page_height = page_sizes[page_index - 1]
        relative_box = entry["relative_box_0_1000"]
        entry["absolute_box"] = [
            int(round(relative_box[0] * page_width / 999.0)),
            int(round(relative_box[1] * page_height / 999.0)),
            int(round(relative_box[2] * page_width / 999.0)),
            int(round(relative_box[3] * page_height / 999.0)),
        ]
        entry["page_size"] = [page_width, page_height]


def encode_image(image_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def build_payload(
    *,
    model_name: str,
    image_paths: Sequence[Path],
    logit_processor: str,
    no_repeat_ngram_size: int,
    ngram_window: int,
) -> dict[str, Any]:
    content = [{"type": "text", "text": PROMPT}]
    content.extend(encode_image(image_path) for image_path in image_paths)
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
        "images_config": {"image_mode": "base"},
        "custom_logit_processor": logit_processor,
        "custom_params": {
            "ngram_size": no_repeat_ngram_size,
            "window_size": ngram_window,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def generate(
    *,
    session: requests.Session,
    server_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    endpoint = f"{server_url.rstrip('/')}/v1/chat/completions"
    response = session.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout_seconds,
        stream=True,
    )
    chunks: list[str] = []
    generation: dict[str, Any] = {"finish_reason": None, "matched_stop": None, "usage": None}
    stream_completed = False
    try:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                stream_completed = True
                break
            event = json.loads(data)
            if not isinstance(event, dict):
                raise ValueError("SGLang SSE event must be an object")
            if "error" in event:
                raise requests.RequestException(f"SGLang SSE error: {event['error']}")
            usage = event.get("usage")
            if isinstance(usage, dict):
                generation["usage"] = usage
            choices = event["choices"]
            if not isinstance(choices, list):
                raise ValueError("SGLang SSE choices must be a list")
            if not choices:
                continue
            if len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("SGLang SSE requires exactly one choice")
            if choices[0].get("finish_reason") is not None:
                generation["finish_reason"] = choices[0]["finish_reason"]
                generation["matched_stop"] = choices[0].get("matched_stop")
            delta = choices[0]["delta"]
            if not isinstance(delta, dict):
                raise ValueError("SGLang SSE delta must be an object")
            content = delta.get("content")  # Role/finish frames may omit content or send null.
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("SGLang SSE delta content must be a string or null")
                chunks.append(content)
        if not stream_completed:
            raise requests.exceptions.ChunkedEncodingError("SGLang stream ended before [DONE]")
    except Exception as exc:
        exc.gdb_generation = generation
        raise
    finally:
        response.close()
    return "".join(chunks), generation


def run_one_pdf(
    args: argparse.Namespace,
    *,
    session: requests.Session,
    logit_processor: str,
    pdf_path: Path,
) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if should_skip(args, pdf_path):
        print(f"[skip] {pdf_path}")
        return

    doc_total_start = time.perf_counter()
    with error_context("render"):
        page_paths, render_dir, created_temp_dir = render_pdf_pages(pdf_path, dpi=args.dpi)
    try:
        page_sizes = [get_image_size(page_path) for page_path in page_paths]
        payload = build_payload(
            model_name=args.model_name,
            image_paths=page_paths,
            logit_processor=logit_processor,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            ngram_window=args.ngram_window,
        )
        with error_context("inference"):
            raw_output, generation = generate(
                session=session,
                server_url=args.server_url,
                payload=payload,
                timeout_seconds=args.request_timeout_seconds,
            )
    finally:
        cleanup_render_dir(render_dir, created_temp_dir)

    with error_context("validate_output"):
        try:
            if generation["finish_reason"] not in ("stop", "length"):
                reason = generation["finish_reason"]
                raise RuntimeError(f"Unlimited-OCR unconfirmed complete output: finish_reason={reason!r}")
            raw_pages = split_raw_pages(raw_output)
        except RuntimeError as exc:
            exc.gdb_generation = generation
            raise
    page_markdowns: list[str] = []
    image_entries: list[dict[str, Any]] = []
    page_mismatch = len(raw_pages) != len(page_paths)
    quality_warnings = []
    if generation["finish_reason"] == "length":
        quality_warnings.append("generation_length")
    if page_mismatch:
        quality_warnings.append("page_count_mismatch")
        # No reliable mapping from segments to physical pages: keep the entire
        # model response, including tags, without fabricated page views/boxes.
        remove_dir_if_exists(doc_output_dir / "pages")
        write_text(doc_output_dir / "document.md", raw_output)
    else:
        for page_index, raw_page in enumerate(raw_pages, start=1):
            page_markdown, page_entries = clean_page_markdown(raw_page, page_index)
            page_markdowns.append(page_markdown)
            image_entries.extend(page_entries)
        add_absolute_boxes(image_entries, page_sizes)
        save_markdown_bundle(doc_output_dir, page_markdowns, concat_document_markdown(page_markdowns))
    doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
    usage = generation["usage"]
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    metrics = {
        "method": METHOD_NAME,
        "method_id": METHOD_ID,
        "execution_profile": "unlimited_ocr_sglang_service",
        "request_mode": "joint",
        "backend": "sglang",
        "input_mode": "native_pdf_multi_page",
        "pdf_path": str(pdf_path),
        "output_dir": str(doc_output_dir),
        "server_url": args.server_url,
        "model_name": args.model_name,
        "prompt": PROMPT,
        "image_mode": "base",
        "dpi": args.dpi,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "ngram_window": args.ngram_window,
        **generation,
        "output_tokens": completion_tokens if completion_tokens is not None else "N/A",
        "response_characters": len(raw_output),
        "request_completed": True,
        "input_page_count": len(page_paths),
        "page_count": len(page_paths),
        "output_segment_count": len(raw_pages),
        "quality_warnings": quality_warnings,
        "output_scope": "document_only" if page_mismatch else "document_and_pages",
        "page_boundary_source": "unavailable" if page_mismatch else "model_page_tokens",
        "render_seconds": "included_in_doc_total",
        "page_inference_seconds_sum": "N/A",
        "doc_total_seconds": doc_total_seconds,
        "document_markdown_path": "document.md",
        "pages": build_page_records(page_markdowns, ["N/A"] * len(page_markdowns)),
        "notes": [
            "One PDF is sent as one official SGLang multi-image request.",
            "PDF rendering uses benchmark 200 DPI by default, not the official example's 300 DPI.",
            "All pages use image_mode=base with the official Multi page parsing prompt.",
            "page_count counts input pages of a completed request, not correctly recognized output pages.",
            "On segment-count mismatch, document.md preserves the raw response; no page attribution or absolute image boxes are invented.",
            "output_tokens is N/A unless the service returns an explicit completion_tokens usage field; SSE chunks are not counted as tokens.",
        ],
    }
    write_json(doc_output_dir / "metrics.json", metrics)
    write_json(
        doc_output_dir / "image_boxes.json",
        build_image_boxes_payload(
            method=METHOD_NAME,
            pdf_path=pdf_path,
            entries=image_entries,
            notes=[
                "Coordinates come from Unlimited-OCR det tags on a 0-999 relative scale.",
                "Absolute coordinates are reconstructed using the corresponding rendered page size.",
            ],
        ),
    )
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}" + (f" quality_warnings={quality_warnings}" if quality_warnings else ""))


def write_pdf_error(args: argparse.Namespace, pdf_path: Path, exc=None) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    exc = sys.exc_info()[1] if exc is None else exc
    generation = getattr(exc, "gdb_generation", {})
    append_error_payload(
        doc_output_dir / "error.json",
        build_error_payload(
            method_name=METHOD_NAME,
            pdf_path=pdf_path,
            exception=exc,
            extra={
                "method_id": METHOD_ID,
                "backend": "sglang",
                "execution_profile": execution_profile(args),
                "request_mode": args.request_mode,
                "server_url": args.server_url,
                "model_name": args.model_name,
                **generation,
            },
        ),
    )


def process_pdf_paths(
    args: argparse.Namespace,
    pdf_paths: Sequence[Path],
    *,
    session: requests.Session,
    logit_processor: str,
) -> bool:
    if args.request_mode == "page_pool":
        return process_page_pool(args, pdf_paths, logit_processor=logit_processor)
    had_failure = False
    for pdf_path in pdf_paths:
        try:
            run_one_pdf(
                args,
                session=session,
                logit_processor=logit_processor,
                pdf_path=pdf_path,
            )
        except Exception as exc:
            had_failure = True
            if isinstance(exc, requests.RequestException) and not http_service_ready(
                args.server_url.rstrip("/") + "/health"
            ):
                exc.gdb_runtime_unusable = True
            write_pdf_error(args, pdf_path)
            print(f"[error] {pdf_path}")
            if args.fail_fast or runtime_unusable(sys.exc_info()[1]):
                raise
    return had_failure


def execution_profile(args):
    return ("unlimited_ocr_sglang_page_pool" if args.request_mode == "page_pool"
            else "unlimited_ocr_sglang_service")


def should_skip(args, pdf):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not args.skip_existing or not document_completed(output):
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics["execution_profile"] != execution_profile(args):
        raise ValueError(f"Cannot reuse another Unlimited request route: {output}")
    return True


def process_page_pool(args, pdf_paths, *, logit_processor):
    local = threading.local()
    sessions = []
    session_lock = threading.Lock()
    render_root = TMP_ROOT / "runtime" / "unlimited_ocr"
    render_root.mkdir(parents=True, exist_ok=True)

    def run_page(pdf, page):
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.trust_env = False
            with session_lock:
                sessions.append(local.session)
        started = time.perf_counter()
        with error_context("render", page), render_pdf_page(pdf, page, args.dpi, render_root) as image:
            size = get_image_size(image)
            payload = build_payload(model_name=args.model_name, image_paths=[image],
                logit_processor=logit_processor, no_repeat_ngram_size=args.no_repeat_ngram_size,
                ngram_window=args.ngram_window)
            with error_context("inference", page):
                raw, generation = generate(session=local.session, server_url=args.server_url,
                    payload=payload, timeout_seconds=args.request_timeout_seconds)
                if generation["finish_reason"] not in ("stop", "length"):
                    exc = RuntimeError(f"Unlimited-OCR unconfirmed complete output: {generation['finish_reason']!r}")
                    exc.gdb_generation = generation
                    raise exc
        # The input supplies the page identity. Model page tokens have no
        # authority to move content to another PDF/page in this route.
        markdown, boxes = clean_page_markdown(raw.replace("<PAGE>", ""), page)
        return {"markdown": markdown, "boxes": boxes, "size": size, "generation": generation,
                "seconds": time.perf_counter() - started, "response_characters": len(raw)}

    def export(pdf, ordered, started):
        output = get_doc_output_dir(args.input_root, args.output_root, pdf)
        pages = [r["markdown"] for r in ordered]
        boxes = [box for r in ordered for box in r["boxes"]]
        add_absolute_boxes(boxes, [r["size"] for r in ordered])
        save_markdown_bundle(output, pages, concat_document_markdown(pages))
        records = build_page_records(pages, [round(r["seconds"], 6) for r in ordered])
        for record, result in zip(records, ordered):
            record.update(result["generation"], page_size=list(result["size"]), request_completed=True)
        warnings = [{"page_number": i, "warning": "generation_length"}
                    for i, r in enumerate(ordered, 1) if r["generation"]["finish_reason"] == "length"]
        write_json(output / "metrics.json", {
            "method": METHOD_NAME, "method_id": METHOD_ID, "backend": "sglang",
            "execution_profile": execution_profile(args), "request_mode": "page_pool",
            "input_mode": "independent_pages", "pdf_path": str(pdf), "output_dir": str(output),
            "server_url": args.server_url, "model_name": args.model_name, "prompt": PROMPT,
            "image_mode": "base", "dpi": args.dpi, "temperature": 0, "skip_special_tokens": False,
            "no_repeat_ngram_size": args.no_repeat_ngram_size, "ngram_window": args.ngram_window,
            "page_concurrency": args.page_concurrency, "request_completed": True,
            "page_count": len(ordered), "input_page_count": len(ordered),
            "quality_warnings": warnings, "output_scope": "document_and_pages",
            "page_boundary_source": "input_pdf", "render_seconds": "included_in_doc_total",
            "page_inference_seconds_sum": "N/A", "doc_total_seconds": round(time.perf_counter() - started, 6),
            "document_markdown_path": "document.md", "pages": records,
            "notes": ["Independent one-image requests share one service and one global page window.",
                      "Page ownership comes from the input, never model page tokens; no raw/image assets saved."]})
        write_json(output / "image_boxes.json", build_image_boxes_payload(
            method=METHOD_NAME, pdf_path=pdf, entries=boxes,
            notes=["Coordinates use each input page's rendered size; no image assets exported."]))
        remove_file_if_exists(output / "error.json")

    def record_error(pdf, exc):
        if isinstance(exc, requests.RequestException) and not http_service_ready(args.server_url.rstrip("/") + "/health"):
            exc.gdb_runtime_unusable = True
        write_pdf_error(args, pdf, exc)

    try:
        return run_page_pool(pdf_paths, concurrency=args.page_concurrency,
            page_count=get_pdf_page_count, should_skip=lambda pdf: should_skip(args, pdf),
            run_page=run_page, export_document=export, record_error=record_error, fail_fast=args.fail_fast)
    finally:
        for session in sessions:
            session.close()
