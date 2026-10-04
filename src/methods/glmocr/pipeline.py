"""Configuration and unified output for the persistent GLM-OCR SDK worker."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path, require_failure_patch  # noqa: E402

GLM_REPO_ROOT = method_repo_path("glmocr")
DEFAULT_CONFIG_PATH = GLM_REPO_ROOT / "glmocr" / "config.yaml"
if str(GLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GLM_REPO_ROOT))

from common import (  # noqa: E402
    error_context,
    build_image_boxes_payload,
    build_page_records,
    extract_markdown_image_markers,
    get_doc_output_dir,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)
from execution.trial import get_pdf_page_count  # noqa: E402
from paths import TMP_ROOT  # noqa: E402


def is_glm_image_like_label(label: Any) -> bool:
    normalized = str(label).strip().lower()
    return normalized in {"image", "figure", "chart", "photo", "picture", "diagram"}


def normalize_glm_json_result(raw_json_result: Any) -> List[List[Dict[str, Any]]]:
    if isinstance(raw_json_result, str):
        parsed = json.loads(raw_json_result)
    else:
        parsed = raw_json_result
    if not isinstance(parsed, list):
        raise TypeError("GLM-OCR json_result is expected to be a list of pages.")
    normalized_pages: List[List[Dict[str, Any]]] = []
    for page in parsed:
        if isinstance(page, list):
            normalized_pages.append(page)
        else:
            raise TypeError("Each GLM-OCR page result should be a list of blocks.")
    return normalized_pages


def render_glm_page_markdown(page_blocks: List[Dict[str, Any]], *, image_prefix: str = "") -> str:
    markdown_parts: List[str] = []
    for block in page_blocks:
        label = str(block.get("label", "")).strip().lower()
        content = str(block.get("content", "") or "").strip()
        image_path = str(block.get("image_path", "") or "").strip()
        if image_path and is_glm_image_like_label(label):
            markdown_parts.append(f"![]({image_prefix}{image_path})")
        elif content:
            markdown_parts.append(content)
    if not markdown_parts:
        return ""
    return "\n\n".join(markdown_parts).rstrip() + "\n"


def extract_glm_image_boxes(
    *,
    pdf_path: Path,
    page_markdowns: List[str],
    document_markdown: str,
    page_blocks_list: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    document_markers = extract_markdown_image_markers(document_markdown)
    document_markers_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for marker in document_markers:
        document_markers_by_name.setdefault(marker["name"], []).append(marker)

    entries: List[Dict[str, Any]] = []
    for page_index, page_markdown in enumerate(page_markdowns, start=1):
        page_blocks = page_blocks_list[page_index - 1] if page_index - 1 < len(page_blocks_list) else []
        page_markers = extract_markdown_image_markers(page_markdown)
        page_markers_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for marker in page_markers:
            page_markers_by_name.setdefault(marker["name"], []).append(marker)

        for block in page_blocks:
            label = str(block.get("label", "")).strip().lower()
            bbox = block.get("bbox_2d")
            image_path = str(block.get("image_path", "") or "").strip()
            if not is_glm_image_like_label(label):
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            marker_name = Path(image_path).name if image_path else None
            page_marker = None
            if marker_name and page_markers_by_name.get(marker_name):
                page_marker = page_markers_by_name[marker_name][0]

            entries.append(
                {
                    "page_index": page_index,
                    "marker_name": marker_name,
                    "marker_type": page_marker["marker_type"] if page_marker else None,
                    "page_marker_expression": (
                        page_marker["match_text"] if page_marker else (f"![]({image_path})" if image_path else None)
                    ),
                    "page_marker_src": page_marker["src"] if page_marker else image_path,
                    "document_marker_expressions": (
                        [marker["match_text"] for marker in document_markers_by_name.get(marker_name, [])]
                        if marker_name
                        else []
                    ),
                    "document_marker_sources": (
                        [marker["src"] for marker in document_markers_by_name.get(marker_name, [])]
                        if marker_name
                        else []
                    ),
                    "relative_box_0_1000": [int(round(float(value))) for value in bbox],
                    "absolute_box": None,
                    "page_size": None,
                    "box_source": "glm_json_bbox_2d",
                    "block_label": label,
                    "block_content_preview": str(block.get("content", ""))[:200],
                }
            )

    return build_image_boxes_payload(
        method="GLM-OCR",
        pdf_path=pdf_path,
        entries=entries,
        notes=[
            "GLM-OCR exposes bbox_2d directly in its structured JSON output.",
            "GLM-OCR bbox_2d coordinates are already normalized to a 0-1000 page-relative scale.",
            "GLM-OCR is run through its official self-hosted SDK flow, which keeps the input as a whole PDF document.",
        ],
    )


def prepare_runtime_config(args: argparse.Namespace) -> Path:
    require_failure_patch("glmocr")
    config_data = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    pipeline_cfg = config_data.setdefault("pipeline", {})
    pipeline_cfg.setdefault("maas", {})["enabled"] = False

    ocr_api_cfg = pipeline_cfg.setdefault("ocr_api", {})
    ocr_api_cfg["api_host"] = args.ocr_api_host
    ocr_api_cfg["api_port"] = args.ocr_api_port
    ocr_api_cfg["model"] = args.ocr_api_model_name
    if args.ocr_connection_pool_size is not None:
        ocr_api_cfg["connection_pool_size"] = args.ocr_connection_pool_size

    layout_cfg = pipeline_cfg.setdefault("layout", {})
    if args.layout_model_dir is not None:
        layout_cfg["model_dir"] = args.layout_model_dir
    if args.layout_device is not None:
        layout_cfg["device"] = args.layout_device
    if args.max_workers is not None:
        pipeline_cfg["max_workers"] = args.max_workers

    runtime_dir = TMP_ROOT / "runtime" / "glmocr"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=runtime_dir,
        suffix=".yaml",
        prefix="gdb_glmocr_config_",
        delete=False,
    ) as tmp:
        runtime_config_path = Path(tmp.name)
    runtime_config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return runtime_config_path


def export_document(args, result, pdf_path, started):
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)

    with error_context("export"):
        page_blocks_list = normalize_glm_json_result(result.json_result)
        # Keep the SDK's native image markers as text, not as a promise that
        # this export includes the corresponding image assets.
        page_markdowns = [render_glm_page_markdown(page) for page in page_blocks_list]
        document_markdown = result.markdown_result
        if not isinstance(document_markdown, str):
            raise TypeError("GLM-OCR markdown_result must be a string")

        expected_page_count = get_pdf_page_count(pdf_path)
        if len(page_markdowns) != expected_page_count:
            raise RuntimeError(
                "GLM-OCR output page count mismatch: "
                f"expected {expected_page_count}, got {len(page_markdowns)}"
            )
        save_markdown_bundle(
            doc_output_dir=doc_output_dir,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
        )

        metrics: Dict[str, Any] = {
            "method": "GLM-OCR",
            "method_id": "glmocr",
            "execution_profile": "glmocr_multi_pdf_regions",
            "input_feed": "official_multi_pdf_stream",
            "backend": "vllm",
            "input_mode": "native_pdf",
            "pdf_path": str(pdf_path),
            "output_dir": str(doc_output_dir),
            "config_path": str(args.config_path),
            "ocr_api_host": args.ocr_api_host,
            "ocr_api_port": args.ocr_api_port,
            "ocr_api_model_name": args.ocr_api_model_name,
            "layout_device": args.layout_device,
            "layout_model_dir": args.layout_model_dir,
            "page_count": len(page_markdowns),
            "render_seconds": "N/A",
            "page_inference_seconds_sum": "N/A",
            "doc_total_seconds": round(time.perf_counter() - started, 6),
            "document_markdown_path": "document.md",
            "pages": build_page_records(page_markdowns, ["N/A"] * len(page_markdowns)),
            "notes": [
                "GLM-OCR is run through its official self-hosted SDK flow: whole PDF input, local layout detection, and a vLLM-backed OCR API.",
                "The official whole-document GLM-OCR SDK flow does not expose reliable page-level latency, so page latencies are recorded as 'N/A'.",
                "Standard document parsing uses the official SDK flow and does not require a user-authored prompt.",
                "Page Markdown comes from official formatted JSON; document.md preserves SDK Markdown, including normal empty output. Native image markers are retained, but image assets are not exported.",
            ],
        }
        write_json(doc_output_dir / "metrics.json", metrics)
        write_json(
            doc_output_dir / "image_boxes.json",
            extract_glm_image_boxes(
                pdf_path=pdf_path,
                page_markdowns=page_markdowns,
                document_markdown=document_markdown,
                page_blocks_list=page_blocks_list,
            ),
        )
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")
