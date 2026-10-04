"""Qianfan document requests and exports; no automatic splitting or fallback."""
from __future__ import annotations

import json
import math
import re
import tempfile
import time
from pathlib import Path

from PIL import Image

from common import (
    append_error_payload, build_error_payload, build_image_boxes_payload, build_page_records,
    document_completed, error_context,
    get_doc_output_dir, render_pdf_pages, save_markdown_bundle, write_json,
    pdf_page_count as page_count, render_pdf_page,
)
from image_vllm_client import image_to_data_url
from methods.qianfan_ocr.prompt import DOCUMENT_PARSING_PROMPT, PROMPT_ID
from paths import TMP_ROOT

PAGE_MARKER = re.compile(r"^--- Page (\d+) ---[ \t]*$", re.MULTILINE)
IMAGE_MARKER = re.compile(r"!\[(image|chart|seal)\]\(<box>(.*?)</box>\)", re.DOTALL)


def temp_root(args):
    root = args.render_cache_root or TMP_ROOT / "runtime" / "qianfan"
    root.mkdir(parents=True, exist_ok=True)
    return root


def split_pages(text, count):
    """Only exact, complete model page markers define page views."""
    markers = list(PAGE_MARKER.finditer(text))
    if count == 1 and not markers:
        return [text]
    if [int(m[1]) for m in markers] != list(range(1, count + 1)):
        return None
    if text[:markers[0].start()].strip():
        return None
    pages = [
        text[m.end():markers[i + 1].start() if i + 1 < count else len(text)].strip()
        for i, m in enumerate(markers)
    ]
    return pages if all(pages) else None


def request(args, client, images):
    # The local model chat template appends <think> when enable_thinking=True.
    # Do not also append it to the user text (that produces two triggers).
    prompt = DOCUMENT_PARSING_PROMPT
    # Official CLI order: text, then each image. Let the chat template insert
    # image placeholders; a manually added single <image> is wrong for joint.
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_to_data_url(p)}} for p in images)
    extra = {"skip_special_tokens": False}
    if args.enable_think:
        extra["chat_template_kwargs"] = {"enable_thinking": True}
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model_name, messages=[{"role": "user", "content": content}],
        temperature=0.0, max_tokens=args.max_tokens, stream=False, extra_body=extra,
    )
    seconds = time.perf_counter() - started
    choice = response.choices[0]
    raw = choice.message.content
    usage = response.usage.model_dump() if response.usage is not None else None
    record = {
        "finish_reason": choice.finish_reason, "usage": usage,
        "image_count": len(images), "request_seconds": seconds,
    }
    if choice.finish_reason not in {"stop", "length"}:
        raise ValueError(f"Unconfirmed response: finish_reason={choice.finish_reason!r}")
    if not isinstance(raw, str):
        raise ValueError("Qianfan response content must be a string")
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", raw, count=1, flags=re.DOTALL).strip()
    return text, record


def extract_image_boxes(text, image_path, page):
    entries, warnings = [], []
    with Image.open(image_path) as image:
        width, height = image.size

        for index, match in enumerate(IMAGE_MARKER.finditer(text), 1):
            reason = None
            try:
                coords = json.loads(match[2])
            except json.JSONDecodeError:
                reason = "invalid_json"
            else:
                if (not isinstance(coords, list) or len(coords) != 1
                        or not isinstance(coords[0], list) or len(coords[0]) != 4):
                    reason = "invalid_shape"
                elif any(type(v) not in (int, float) for v in coords[0]):
                    reason = "non_numeric_coordinate"
                elif any(type(v) is float and not math.isfinite(v) for v in coords[0]):
                    reason = "non_finite_coordinate"
                else:
                    x1, y1, x2, y2 = coords[0]
                    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                        reason = "out_of_range_or_unordered"
            if reason is not None:
                warnings.append({"warning": "invalid_image_box", "page_index": page,
                                 "box_index": index, "reason": reason})
                continue
            x1, y1, x2, y2 = coords[0]
            absolute = [int(x1 * width / 1000), int(y1 * height / 1000),
                        int(x2 * width / 1000), int(y2 * height / 1000)]
            entries.append({"page_index": page, "relative_box_0_1000": coords[0],
                            "absolute_box": absolute, "page_size": [width, height],
                            "page_marker_expression": match[0],
                            "box_source": "official_box_marker"})
    return entries, warnings


def export(args, pdf, count, document, pages, records, boxes, started, *, warnings=()):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    save_markdown_bundle(output, pages or [], document)
    joint = args.request_mode == "joint"
    metrics = {
        "method": "Qianfan-OCR", "method_id": "qianfan_ocr_think" if args.enable_think else "qianfan_ocr",
        "backend": "vllm", "request_mode": args.request_mode,
        "execution_profile": ("qianfan_ocr_think" if args.enable_think else "qianfan_ocr") + "_vllm_" + args.request_mode,
        "prompt_id": PROMPT_ID, "prompt": DOCUMENT_PARSING_PROMPT,
        "thinking_enabled": args.enable_think,
        "thinking_trigger": "chat_template_only" if args.enable_think else "disabled",
        "pdf_path": str(pdf),
        "model_name": args.model_name, "server_url": args.server_url,
        "dpi": args.dpi, "temperature": 0.0, "max_tokens_per_request": args.max_tokens,
        "page_count": count, "input_page_count": count,
        "output_scope": "document_only" if pages is None else "document_and_pages",
        "page_boundary_source": ("single_page_input" if count == 1 else "model_markers") if joint and pages is not None else
                                ("unavailable" if joint else "independent_requests"),
        "pages": build_page_records(pages or [], ["N/A"] * count if joint else
                                   [r["request_seconds"] for r in records]),
        "requests": records, "doc_total_seconds": time.perf_counter() - started,
        "quality_warnings": list(warnings), "invalid_image_box_count": len(warnings),
        "document_markdown_path": "document.md",
        "notes": ["Joint input page count is not independently verified output content coverage.",
                  "Joint request latency is not a per-page inference latency."] if joint else [],
    }
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method="Qianfan-OCR", pdf_path=pdf, entries=boxes,
        notes=["No page assignment or crops are guessed when joint page markers are unavailable."] if pages is None else [],
    ))
    write_json(output / "metrics.json", metrics)
    if warnings:
        print(f"[warning] {pdf}: {len(warnings)} invalid image box(es); see metrics.json", flush=True)
    (output / "error.json").unlink(missing_ok=True)


def should_skip(args, pdf):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not document_completed(output):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Existing partial/failed output must be retained; use a new run: {output}")
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    same = (metrics.get("request_mode") == args.request_mode and metrics.get("prompt_id") == PROMPT_ID
            and metrics["thinking_enabled"] == args.enable_think
            and metrics["dpi"] == args.dpi and metrics["max_tokens_per_request"] == args.max_tokens
            and metrics["model_name"] == args.model_name
            and (not args.enable_think or metrics.get("thinking_trigger") == "chat_template_only"))
    if not args.skip_existing or not same:
        raise FileExistsError(f"Existing result must not be overwritten or reused across profiles: {output}")
    return True


def record_error(args, pdf, exc):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    append_error_payload(output / "error.json", build_error_payload(
        "Qianfan-OCR", pdf, exception=exc,
        extra={"backend": "vllm", "request_mode": args.request_mode, "prompt_id": PROMPT_ID},
    ))
    print(f"[error] {pdf}: {exc}", flush=True)


def run_joint(args, client, pdf):
    if should_skip(args, pdf):
        return
    started = time.perf_counter()
    count = page_count(pdf)
    with tempfile.TemporaryDirectory(prefix="joint_", dir=temp_root(args)) as temp:
        with error_context("render"):
            images, _, _ = render_pdf_pages(pdf, args.dpi, Path(temp))
            if len(images) != count:
                raise ValueError("Rendered page count differs from PDF page count")
        with error_context("inference"):
            document, record = request(args, client, images)
        with error_context("export"):
            pages = split_pages(document, count)
            boxes, warnings = [], []
            if pages is not None:
                for n, (page, image) in enumerate(zip(pages, images), 1):
                    page_boxes, page_warnings = extract_image_boxes(page, image, n)
                    boxes.extend(page_boxes)
                    warnings.extend(page_warnings)
            export(args, pdf, count, document, pages, [record], boxes, started, warnings=warnings)


def run_page(args, client, pdf, page):
    with render_pdf_page(pdf, page, args.dpi, temp_root(args)) as image:
        with error_context("inference", page):
            text, record = request(args, client, [image])
        with error_context("export", page):
            boxes, warnings = extract_image_boxes(text, image, page)
        record["page_index"] = page
        return text, record, boxes, warnings
