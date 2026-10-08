"""Convert TeleOCR middle JSON into the minimal GlobalDocBench bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from common import (
    build_error_payload,
    build_image_boxes_payload,
    build_page_records,
    get_doc_output_dir,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)


METHOD_ID = "teleocr_vllm_async"
EXECUTION_PROFILE = METHOD_ID


def write_document_outputs(
    args: Any,
    pdf_path: Path,
    middle_json: dict[str, Any],
    input_page_count: int,
    processing_seconds: float,
) -> None:
    pdf_info = middle_json.get("pdf_info")
    if not isinstance(pdf_info, list) or len(pdf_info) != input_page_count:
        raise ValueError(
            f"TeleOCR returned {len(pdf_info) if isinstance(pdf_info, list) else 'invalid'} "
            f"pages for input with {input_page_count} pages"
        )

    output = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make

    page_markdowns = [union_make([page], "") for page in pdf_info]
    document_markdown = union_make(pdf_info, "")
    save_markdown_bundle(output, page_markdowns, document_markdown)

    write_json(output / "metrics.json", {
        "method": "TeleOCR vLLM Async",
        "method_id": METHOD_ID,
        "execution_profile": EXECUTION_PROFILE,
        "backend": args.backend,
        "model_path": str(args.model_path),
        "input_mode": "official_pdf_page_pipeline",
        "pdf_path": str(pdf_path),
        "output_dir": str(output),
        "input_page_count": input_page_count,
        "recognized_page_count": len(pdf_info),
        "page_count": len(page_markdowns),
        "layout_mode": args.layout_mode,
        "max_concurrency": args.max_concurrency,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "pdf_tools": args.pdf_tools,
        "render_dpi": 200,
        "pdf_tools_workers": args.pdf_tools_workers,
        "doc_total_seconds": round(processing_seconds, 6),
        "document_markdown_path": "document.md",
        "pages": build_page_records(page_markdowns, ["N/A"] * len(page_markdowns)),
        "warnings": [],
        "asset_policy": "no_image_assets",
        "notes": [
            "One resident official TeleOCR vllm-async-engine runtime is reused across documents.",
            "Documents are submitted sequentially; layout and region requests within the current document use max_concurrency.",
            "The adapter calls aio_doc_analyze directly and passes image_writer=None.",
            "No raw response, cropped image, or layout PDF asset is retained.",
        ],
    })
    # The common completion validator requires this metadata file.  It is not an
    # image asset and intentionally contains no copied pixels.
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method="TeleOCR vLLM Async",
        pdf_path=pdf_path,
        entries=[],
        notes=["No image assets are retained by the benchmark adapter."],
    ))
    remove_file_if_exists(output / "error.json")


def write_document_error(args: Any, pdf_path: Path, exc: BaseException) -> None:
    output = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    write_json(
        output / "error.json",
        build_error_payload(
            "TeleOCR vLLM Async",
            pdf_path,
            exception=exc,
            extra={
                "method_id": METHOD_ID,
                "execution_profile": EXECUTION_PROFILE,
                "backend": args.backend,
                "max_concurrency": args.max_concurrency,
            },
        ),
    )
