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

from method_paths import method_repo_path  # noqa: E402
from paths import MODEL_ROOT  # noqa: E402

DEFAULT_CACHE_ROOT = MODEL_ROOT / "runtime_cache" / "marker"
# Benchmark-wide rendering policy, overriding Marker's native 96/192 DPI defaults.
BENCHMARK_RENDER_DPI = 200

MARKER_REPO_ROOT = method_repo_path("marker")
if str(MARKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MARKER_REPO_ROOT))

from common import (  # noqa: E402
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_image_boxes_payload,
    build_page_records,
    concat_document_markdown,
    extract_markdown_image_markers,
    get_doc_output_dir,
    normalize_box_0_1000,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)


# Fixed quality settings shared by the converter and output metadata.
QUALITY_OPTIONS = {
    "use_llm": False,
    "llm_service": None,
    "force_ocr": False,
    "strip_existing_ocr": False,
    "format_lines": False,
    "disable_image_extraction": False,
}


def normalize_markdown_text(markdown_text: str) -> str:
    normalized = str(markdown_text or "").strip()
    if not normalized:
        return ""
    return normalized + "\n"


def build_marker_image_boxes(
    *,
    pdf_path: Path,
    document: Any,
    image_block_types: Any,
    page_markdowns: List[str],
    document_markdown: str,
) -> Dict[str, Any]:
    document_markers = extract_markdown_image_markers(document_markdown)
    entries: List[Dict[str, Any]] = []

    for page_index, page in enumerate(document.pages, start=1):
        page_markdown = (
            page_markdowns[page_index - 1]
            if page_index - 1 < len(page_markdowns)
            else ""
        )
        page_width = int(round(page.polygon.width))
        page_height = int(round(page.polygon.height))
        page_markers = extract_markdown_image_markers(page_markdown)

        for block in page.contained_blocks(document, image_block_types):
            block_key = block.id.to_path()
            page_matches = [
                marker
                for marker in page_markers
                if Path(marker["src"]).stem == block_key
            ]
            document_matches = [
                marker
                for marker in document_markers
                if Path(marker["src"]).stem == block_key
            ]
            absolute_box = [float(value) for value in block.polygon.bbox]
            entries.append(
                {
                    "page_index": page_index,
                    "marker_name": Path(page_matches[0]["src"]).name if page_matches else None,
                    "marker_type": page_matches[0]["marker_type"] if page_matches else None,
                    "page_marker_expression": (
                        page_matches[0]["match_text"] if page_matches else None
                    ),
                    "page_marker_src": page_matches[0]["src"] if page_matches else None,
                    "document_marker_expressions": [
                        marker["match_text"] for marker in document_matches
                    ],
                    "document_marker_sources": [
                        marker["src"] for marker in document_matches
                    ],
                    "absolute_box": absolute_box,
                    "relative_box_0_1000": normalize_box_0_1000(
                        absolute_box,
                        page_width=page_width,
                        page_height=page_height,
                    ),
                    "page_size": [page_width, page_height],
                    "box_source": "marker_block_polygon_bbox",
                    "block_label": str(block.block_type),
                    "block_description": str(getattr(block, "description", "") or ""),
                    "block_id": str(block.id),
                }
            )

    return build_image_boxes_payload(
        method="Marker",
        pdf_path=pdf_path,
        entries=entries,
        notes=[
            "Marker is run through its official native PdfConverter pipeline.",
            "Document and per-page markdown are rendered from the same Marker document object; page concatenation is used only when whole-document markdown is empty.",
            "Image coordinates are extracted from Marker figure/picture block polygons and normalized to a 0-1000 page-relative scale.",
        ],
    )


def run_one_pdf(
    args: argparse.Namespace,
    converter: Any,
    markdown_renderer: Any,
    html_renderer: Any,
    cleanup_text: Any,
    image_block_types: Any,
    pdf_path: Path,
) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if args.skip_existing and document_completed(doc_output_dir):
        print(f"[skip] {pdf_path}")
        return

    doc_total_start = time.perf_counter()
    with error_context("inference"):
        document = converter.build_document(str(pdf_path))
        document_output = document.render()

    document_html, _document_images = html_renderer.extract_html(document, document_output)
    document_markdown_raw = cleanup_text(markdown_renderer.md_cls.convert(document_html))

    page_markdowns: List[str] = []
    for page_number, page_output in enumerate(document_output.children, start=1):
        with error_context("postprocess", page_number):
            page_html, _page_images = html_renderer.extract_html(document, page_output, level=1)
            page_markdown_raw = cleanup_text(markdown_renderer.md_cls.convert(page_html))
            page_markdown = normalize_markdown_text(page_markdown_raw)
            page_markdowns.append(page_markdown)

    if document_markdown_raw:
        document_markdown = normalize_markdown_text(document_markdown_raw)
    else:
        document_markdown = concat_document_markdown(page_markdowns)

    save_markdown_bundle(
        doc_output_dir=doc_output_dir,
        page_markdowns=page_markdowns,
        document_markdown=document_markdown,
    )

    doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
    metrics: Dict[str, Any] = {
        "method": "Marker",
        "backend": "native_converter",
        "input_mode": "native_pdf",
        "pdf_path": str(pdf_path),
        "output_dir": str(doc_output_dir),
        "torch_device": args.torch_device,
        **QUALITY_OPTIONS,
        "page_count": len(page_markdowns),
        "lowres_image_dpi": converter.config["lowres_image_dpi"],
        "highres_image_dpi": converter.config["highres_image_dpi"],
        "render_seconds": "N/A",
        "page_inference_seconds_sum": "N/A",
        "doc_total_seconds": doc_total_seconds,
        "document_markdown_path": "document.md",
        "pages": build_page_records(page_markdowns, ["N/A"] * len(page_markdowns)),
        "notes": [
            "Marker is run through its official ConfigParser + PdfConverter flow.",
            "The runner keeps Marker's native whole-document conversion path and derives page markdown from the same built document object without re-running model inference per page.",
            "Marker does not expose reliable page-level inference latency in this native whole-document flow, so page latencies are recorded as 'N/A'.",
            "Marker does not require a user-authored prompt in its default native document conversion mode.",
            "Both rendering paths use the benchmark's 200 DPI policy, not Marker's native 96/192 DPI defaults.",
        ],
    }
    write_json(doc_output_dir / "metrics.json", metrics)
    write_json(
        doc_output_dir / "image_boxes.json",
        build_marker_image_boxes(
            pdf_path=pdf_path,
            document=document,
            image_block_types=image_block_types,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
        ),
    )
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
            method_name="Marker",
            pdf_path=pdf_path,
            extra={
                "torch_device": args.torch_device,
                "use_llm": QUALITY_OPTIONS["use_llm"],
                "force_ocr": QUALITY_OPTIONS["force_ocr"],
            },
        ),
    )
    print(f"[error] {pdf_path}")


def configure_marker_runtime(args: argparse.Namespace) -> None:
    cache_root = DEFAULT_CACHE_ROOT
    xdg_cache_home = cache_root / "xdg_cache"
    hf_home = cache_root / "hf_home"
    hf_modules = cache_root / "hf_modules"
    huggingface_hub_cache = hf_home / "hub"
    transformers_cache = hf_home / "transformers"
    model_cache_dir = xdg_cache_home / "datalab" / "models"
    for path in (
        cache_root,
        xdg_cache_home,
        hf_home,
        hf_modules,
        huggingface_hub_cache,
        transformers_cache,
        model_cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(huggingface_hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_modules))
    os.environ.setdefault("MODEL_CACHE_DIR", str(model_cache_dir))

    if args.torch_device:
        os.environ["TORCH_DEVICE"] = args.torch_device

    import torch  # noqa: E402

    # Enable cuDNN before the parent creates models and children use them.
    torch.backends.cudnn.enabled = True


def load_marker_runtime(args: argparse.Namespace, *, artifact_dict: Dict[str, Any]) -> Dict[str, Any]:
    configure_marker_runtime(args)
    from marker.config.parser import ConfigParser  # noqa: E402
    from marker.renderers.html import HTMLRenderer  # noqa: E402
    from marker.renderers.markdown import MarkdownRenderer, cleanup_text  # noqa: E402
    from marker.schema import BlockTypes  # noqa: E402

    cli_options = {
        "output_format": "markdown",
        "lowres_image_dpi": BENCHMARK_RENDER_DPI,
        "highres_image_dpi": BENCHMARK_RENDER_DPI,
        **QUALITY_OPTIONS,
        "debug": False,
        # Match the native multi-document CLI: no nested pdftext process pool.
        "disable_multiprocessing": True,
    }
    config_parser = ConfigParser(cli_options)
    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    markdown_renderer = converter.resolve_dependencies(MarkdownRenderer)
    html_renderer = converter.resolve_dependencies(HTMLRenderer)
    image_block_types = (BlockTypes.Picture, BlockTypes.Figure)
    return {
        "converter": converter,
        "markdown_renderer": markdown_renderer,
        "html_renderer": html_renderer,
        "cleanup_text": cleanup_text,
        "image_block_types": image_block_types,
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
                args=args,
                converter=runtime["converter"],
                markdown_renderer=runtime["markdown_renderer"],
                html_renderer=runtime["html_renderer"],
                cleanup_text=runtime["cleanup_text"],
                image_block_types=runtime["image_block_types"],
                pdf_path=pdf_path,
            )
        except Exception:
            had_failure = True
            write_pdf_error(args, pdf_path)
            if args.fail_fast or runtime_unusable(sys.exc_info()[1]):
                raise
    return had_failure
