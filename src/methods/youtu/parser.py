#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path, require_failure_patch  # noqa: E402
from paths import MODEL_ROOT  # noqa: E402

DEFAULT_CACHE_ROOT = MODEL_ROOT / "runtime_cache" / "youtu_parsing"

YOUTU_REPO_ROOT = method_repo_path("youtu")
if str(YOUTU_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOUTU_REPO_ROOT))

from common import (  # noqa: E402
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_image_boxes_payload,
    build_page_records,
    concat_document_markdown,
    get_doc_output_dir,
    normalize_box_0_1000,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return




def normalize_markdown_text(markdown_text: str) -> str:
    normalized = str(markdown_text or "").strip()
    if not normalized:
        return ""
    return normalized + "\n"


def build_youtu_page_markdown(page_result: List[Dict[str, Any]]) -> str:
    contents: List[str] = []
    for item in page_result:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if text:
            contents.append(text)
    return normalize_markdown_text("\n\n".join(contents))


def is_youtu_image_like_type(layout_type: Any) -> bool:
    normalized = str(layout_type or "").strip().lower()
    return normalized in {
        "figure",
        "chart",
        "chart_data",
        "chart_logic",
        "image",
        "picture",
        "photo",
        "header_figure",
        "footer_figure",
    }


def youtu_box_to_xyxy(bbox: Any) -> List[float] | None:
    if not isinstance(bbox, list):
        return None
    if len(bbox) == 4:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return [x1, y1, x2, y2]
    if len(bbox) == 8:
        xs = [float(value) for value in bbox[0::2]]
        ys = [float(value) for value in bbox[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def extract_youtu_image_boxes(
    *,
    pdf_path: Path,
    page_results: List[List[Dict[str, Any]]],
    page_sizes: List[List[int]],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for page_index, (page_result, page_size) in enumerate(
        zip(page_results, page_sizes), start=1
    ):
        page_width, page_height = page_size
        for item in page_result:
            if not isinstance(item, dict):
                continue
            layout_type = item.get("type")
            if not is_youtu_image_like_type(layout_type):
                continue
            absolute_box = youtu_box_to_xyxy(item.get("bbox"))
            if absolute_box is None:
                continue
            relative_box = None
            if page_width > 0 and page_height > 0:
                relative_box = normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                )
            content = str(item.get("content", "") or "").strip()
            entries.append(
                {
                    "page_index": page_index,
                    "marker_name": None,
                    "marker_type": None,
                    "page_marker_expression": content or None,
                    "page_marker_src": None,
                    "document_marker_expressions": [content] if content else [],
                    "document_marker_sources": [],
                    "absolute_box": absolute_box,
                    "relative_box_0_1000": relative_box,
                    "page_size": [page_width, page_height],
                    "box_source": "youtu_page_result_bbox",
                    "block_label": str(layout_type or ""),
                    "block_content_preview": content[:200],
                }
            )

    return build_image_boxes_payload(
        method="Youtu-Parsing",
        pdf_path=pdf_path,
        entries=entries,
        notes=[
            "Youtu-Parsing is run with its official Hugging Face parser flow, which converts PDFs to page images at the specified DPI and parses pages sequentially.",
            "Image coordinates are extracted from the official page_result bbox fields and normalized to a 0-1000 page-relative scale.",
            "Youtu-Parsing's official markdown export is text-content-only, so image marker expressions may be empty or fall back to the recognized figure/chart text.",
        ],
    )


def run_one_pdf(args: argparse.Namespace, parser: Any, load_images_from_pdf: Any, pdf_path: Path) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if args.skip_existing and document_completed(doc_output_dir):
        print(f"[skip] {pdf_path}")
        return

    doc_total_start = time.perf_counter()
    render_start = time.perf_counter()
    with error_context("render"):
        images = load_images_from_pdf(str(pdf_path), dpi=args.dpi)
    render_seconds = round(time.perf_counter() - render_start, 6)

    page_markdowns: List[str] = []
    page_latencies: List[float] = []
    page_results: List[List[Dict[str, Any]]] = []
    page_sizes: List[List[int]] = []
    page_angles: List[Any] = []

    for page_number, image in enumerate(images, start=1):
        page_sizes.append([int(image.size[0]), int(image.size[1])])
        page_start = time.perf_counter()
        with error_context("inference", page_number):
            page_result, page_angle, _hierarchy = parser._parse_single_image(
                image,
                enable_angle_corrector=args.enable_angle_correct,
                batch_size=args.ocr_batch_size,
            )
        page_seconds = round(time.perf_counter() - page_start, 6)
        page_results.append(page_result)
        page_angles.append(page_angle)
        page_markdowns.append(build_youtu_page_markdown(page_result))
        page_latencies.append(page_seconds)
        clear_cuda_cache()

    with error_context("export"):
        document_markdown = concat_document_markdown(page_markdowns)
        save_markdown_bundle(
            doc_output_dir=doc_output_dir,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
        )

        doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
        metrics: Dict[str, Any] = {
            "method": "Youtu-Parsing",
            "backend": "native_hf_parser",
            "input_mode": "native_pdf_to_images",
            "pdf_path": str(pdf_path),
            "output_dir": str(doc_output_dir),
            "model_path": str(args.model_path),
            "dpi": args.dpi,
            "max_pixels": args.max_pixels,
            "ocr_batch_size": args.ocr_batch_size,
            "enable_angle_correct": args.enable_angle_correct,
            "angle_correct_model_path": (
                str(args.angle_correct_model_path) if args.angle_correct_model_path else None
            ),
            "page_count": len(page_markdowns),
            "render_seconds": render_seconds,
            "page_inference_seconds_sum": round(sum(page_latencies), 6),
            "doc_total_seconds": doc_total_seconds,
            "document_markdown_path": "document.md",
            "pages": build_page_records(page_markdowns, page_latencies),
            "page_angles": page_angles,
            "notes": [
                "Youtu-Parsing is run through its official YoutuOCRParserHF flow.",
                "The runner follows the official parse_pdf logic: PDF pages are rendered at the specified DPI and then parsed sequentially page by page.",
                "The parser uses Youtu-Parsing's built-in prompt-guided framework and does not require a user-authored document prompt.",
                "Page markdown uses the same text-only export convention as the official parse_file implementation.",
            ],
        }
        write_json(doc_output_dir / "metrics.json", metrics)
        write_json(
            doc_output_dir / "image_boxes.json",
            extract_youtu_image_boxes(
                pdf_path=pdf_path,
                page_results=page_results,
                page_sizes=page_sizes,
            ),
        )
        clear_cuda_cache()
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")


def write_pdf_error(args: argparse.Namespace, pdf_path: Path) -> None:
    doc_output_dir = get_doc_output_dir(
        args.input_root,
        args.output_root,
        pdf_path,
    )
    write_json(
        doc_output_dir / "error.json",
        build_error_payload(
            method_name="Youtu-Parsing",
            pdf_path=pdf_path,
            extra={
                "model_path": str(args.model_path),
                "dpi": args.dpi,
                "max_pixels": args.max_pixels,
                "ocr_batch_size": args.ocr_batch_size,
                "enable_angle_correct": args.enable_angle_correct,
            },
        ),
    )
    print(f"[error] {pdf_path}")


def load_youtu_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    require_failure_patch("youtu")
    cache_root = DEFAULT_CACHE_ROOT
    hf_home = cache_root / "hf_home"
    hf_modules = cache_root / "hf_modules"
    huggingface_hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    for path in (
        cache_root,
        hf_home,
        hf_modules,
        huggingface_hub_cache,
        transformers_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(huggingface_hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_modules))

    from youtu_hf_parser.youtu_ocr_parser_hf import YoutuOCRParserHF  # noqa: E402
    from youtu_parsing_utils import load_images_from_pdf  # noqa: E402

    parser_kwargs = {
        "model_path": str(args.model_path),
        "enable_angle_correct": args.enable_angle_correct,
        "angle_correct_model_path": (
            str(args.angle_correct_model_path) if args.angle_correct_model_path else ""
        ),
    }
    # Preserve the official parser default when the user does not override
    # max_pixels instead of passing None through to downstream processors.
    if args.max_pixels is not None:
        parser_kwargs["max_pixels"] = args.max_pixels
    parser = YoutuOCRParserHF(**parser_kwargs)
    return {
        "parser": parser,
        "load_images_from_pdf": load_images_from_pdf,
    }


def process_pdf_paths(
    args: argparse.Namespace,
    pdf_paths: List[Path],
    *,
    runtime: Dict[str, Any],
) -> bool:
    had_failure = False
    for pdf_path in pdf_paths:
        try:
            run_one_pdf(
                args,
                runtime["parser"],
                runtime["load_images_from_pdf"],
                pdf_path,
            )
        except Exception:
            had_failure = True
            write_pdf_error(args, pdf_path)
            clear_cuda_cache()
            if args.fail_fast or runtime_unusable(sys.exc_info()[1]):
                raise
    return had_failure
