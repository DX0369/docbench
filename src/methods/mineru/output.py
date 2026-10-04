"""Shared MinerU output conversion and document language selection (no models)."""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from common import build_image_boxes_payload, extract_markdown_image_markers


def mineru_identity() -> Dict[str, str]:
    try:
        from mineru.version import __version__ as runtime_version
    except (ImportError, AttributeError):
        runtime_version = os.environ.get("GDB_MINERU_VERSION", "unknown")
    return {
        "method_id": os.environ.get("GDB_METHOD_ID", "mineru"),
        "method_family": "mineru",
        "mineru_version": runtime_version,
        "source_revision": os.environ.get("GDB_MINERU_SOURCE_REVISION", "unknown"),
        "checkpoint": os.environ.get("GDB_MINERU_CHECKPOINT", "unknown"),
    }


def normalize_markdown_text(markdown_text: str) -> str:
    normalized = str(markdown_text or "").strip()
    if not normalized:
        return ""
    return normalized + "\n"


def infer_mineru_lang(input_root: Path, pdf_path: Path, requested_lang: str) -> str:
    if requested_lang and requested_lang != "auto":
        return requested_lang
    rel_parts = [part.lower() for part in pdf_path.relative_to(input_root).parts]
    if "en" in rel_parts:
        return "en"
    if "zh" in rel_parts:
        return "ch"
    return "ch"


def resolve_backend_name(
    requested_backend: str, get_vlm_engine: Callable[..., str], *, is_async: bool = True
) -> str:
    if requested_backend.startswith("hybrid-"):
        suffix = requested_backend[7:]
        if suffix in ("auto-engine", "engine"):
            return f"hybrid-{get_vlm_engine(inference_engine='auto', is_async=is_async)}"
        return requested_backend
    return requested_backend


def build_page_markdowns(
    pdf_info: Sequence[Dict[str, Any]],
    union_make: Callable[..., Any],
) -> List[str]:
    page_markdowns: List[str] = []
    for page_info in pdf_info:
        page_markdown = union_make([page_info], "mm_markdown", "images")
        page_markdowns.append(normalize_markdown_text(page_markdown))
    return page_markdowns


def build_mineru_content_list(
    pdf_info: Sequence[Dict[str, Any]],
    union_make: Callable[..., Any],
) -> List[Dict[str, Any]]:
    content_list = union_make(list(pdf_info), "content_list", "images")
    if not isinstance(content_list, list):
        return []
    return [item for item in content_list if isinstance(item, dict)]


def build_page_size_map(pdf_info: Sequence[Dict[str, Any]]) -> Dict[int, List[int]]:
    page_sizes: Dict[int, List[int]] = {}
    for page_info in pdf_info:
        page_idx = int(page_info.get("page_idx", len(page_sizes)))
        page_size = page_info.get("page_size")
        if isinstance(page_size, list) and len(page_size) == 2:
            page_sizes[page_idx] = [int(page_size[0]), int(page_size[1])]
    return page_sizes


def entry_fallback_expression(entry: Dict[str, Any]) -> Optional[str]:
    text_parts: List[str] = []
    text_value = entry.get("text")
    if isinstance(text_value, str) and text_value.strip():
        text_parts.append(text_value.strip())

    table_body = entry.get("table_body")
    if isinstance(table_body, str) and table_body.strip():
        text_parts.append(table_body.strip())

    generic_content = entry.get("content")
    if isinstance(generic_content, str) and generic_content.strip():
        text_parts.append(generic_content.strip())

    for key in [
        "image_caption",
        "table_caption",
        "chart_caption",
        "image_footnote",
        "table_footnote",
        "chart_footnote",
        "code_caption",
    ]:
        value = entry.get(key)
        if isinstance(value, list):
            for item in value:
                item_text = str(item).strip()
                if item_text:
                    text_parts.append(item_text)
        elif isinstance(value, str) and value.strip():
            text_parts.append(value.strip())

    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def denormalize_box(
    relative_box: Sequence[float],
    page_size: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if not page_size or len(page_size) != 2:
        return None
    page_width, page_height = int(page_size[0]), int(page_size[1])
    if page_width <= 0 or page_height <= 0:
        return None
    x0, y0, x1, y1 = [float(value) for value in relative_box]
    return [
        int(round(x0 * page_width / 1000.0)),
        int(round(y0 * page_height / 1000.0)),
        int(round(x1 * page_width / 1000.0)),
        int(round(y1 * page_height / 1000.0)),
    ]


def extract_mineru_image_boxes(
    *,
    pdf_path: Path,
    backend_requested: str,
    backend_resolved: str,
    page_markdowns: Sequence[str],
    document_markdown: str,
    content_list: Sequence[Dict[str, Any]],
    page_sizes: Dict[int, List[int]],
) -> Dict[str, Any]:
    document_markers = extract_markdown_image_markers(document_markdown)
    document_markers_by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for marker in document_markers:
        document_markers_by_src[marker["src"]].append(marker)

    page_markers_by_page_src: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for page_index, page_markdown in enumerate(page_markdowns, start=1):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for marker in extract_markdown_image_markers(page_markdown):
            grouped[marker["src"]].append(marker)
        page_markers_by_page_src[page_index] = grouped

    entries: List[Dict[str, Any]] = []
    for item in content_list:
        img_path = str(item.get("img_path", "") or "").strip()
        if not img_path:
            continue

        page_idx = int(item.get("page_idx", 0))
        page_index = page_idx + 1
        page_markers = page_markers_by_page_src.get(page_index, {}).get(img_path, [])
        page_marker = page_markers[0] if page_markers else None
        document_marker_matches = document_markers_by_src.get(img_path, [])
        relative_box = item.get("bbox")
        if not (isinstance(relative_box, list) and len(relative_box) == 4):
            relative_box = None

        page_size = page_sizes.get(page_idx)
        fallback_expression = entry_fallback_expression(item)

        entries.append(
            {
                "page_index": page_index,
                "marker_name": Path(img_path).name,
                "marker_type": page_marker["marker_type"] if page_marker else None,
                "page_marker_expression": (
                    page_marker["match_text"]
                    if page_marker
                    else fallback_expression
                ),
                "page_marker_src": page_marker["src"] if page_marker else img_path,
                "document_marker_expressions": (
                    [marker["match_text"] for marker in document_marker_matches]
                    if document_marker_matches
                    else ([fallback_expression] if fallback_expression else [])
                ),
                "document_marker_sources": (
                    [marker["src"] for marker in document_marker_matches]
                    if document_marker_matches
                    else ([img_path] if img_path else [])
                ),
                "relative_box_0_1000": relative_box,
                "absolute_box": (
                    denormalize_box(relative_box, page_size)
                    if relative_box is not None
                    else None
                ),
                "page_size": page_size,
                "box_source": "mineru_content_list_bbox",
                "block_label": str(item.get("type", "")),
                "block_content_preview": (
                    (fallback_expression or str(item.get("type", "")))[:200]
                ),
            }
        )

    return build_image_boxes_payload(
        method="MinerU",
        pdf_path=pdf_path,
        entries=entries,
        notes=[
            "MinerU is run with its official whole-document parsing flow and direct PDF input.",
            f"Requested backend: {backend_requested}. Resolved backend: {backend_resolved}.",
            "Page markdown is generated page by page with MinerU's own union_make renderer using the official middle_json output.",
            "Image boxes are extracted from MinerU content_list bbox fields, which are already normalized to a 0-1000 page-relative scale.",
            "Intermediate image assets created by MinerU are kept only in a temporary workspace and are not preserved in the final benchmark output.",
        ],
    )
