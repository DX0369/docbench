"""Convert the typed 4.x result without altering official document rendering."""
from __future__ import annotations

from pathlib import Path

from common import build_image_boxes_payload, extract_markdown_image_markers


def convert_result(result, *, input_page_count: int, pdf_path: Path):
    from docvortex.render.fragments import normalize_image_source, resolve_image_source
    from mineru.render import render_markdown

    if [page.page_idx for page in result.pages] != list(range(input_page_count)):
        raise ValueError("MinerU 4 result does not cover every original PDF page in order")
    # Rendering the full tree preserves official cross-page joins. Per-page
    # views are projections, not the source of document.md.
    document = result.markdown()
    pages = [render_markdown(result.middle_json.model_copy(
        update={"pages": [page], "is_full_document": False},
    )) for page in result.pages]
    document_markers = extract_markdown_image_markers(document)
    entries = []
    for page, markdown in zip(result.pages, pages):
        markers = extract_markdown_image_markers(markdown)
        for block in page.blocks:
            if block.type not in {"image", "table", "chart", "equation"}:
                continue
            body = (block if block.type == "equation" else
                    next(child for child in block.content if child.type == f"{block.type}_body"))
            source = resolve_image_source(body)
            source = normalize_image_source(source) if source else None
            matches = [marker for marker in markers if source and marker["src"] == source]
            document_matches = [marker for marker in document_markers if source and marker["src"] == source]
            marker = matches[0] if matches else None
            entries.append({
                "page_index": page.page_idx + 1,
                "marker_name": Path(source).name if source and not source.startswith("data:") else None,
                "marker_type": marker["marker_type"] if marker else None,
                "page_marker_expression": marker["match_text"] if marker else None,
                "page_marker_src": marker["src"] if marker else None,
                "document_marker_expressions": [m["match_text"] for m in document_matches],
                "document_marker_sources": [m["src"] for m in document_matches],
                "relative_box_0_1000": [int(value * 1000) for value in block.bbox],
                "absolute_box": None, "page_size": None,
                "block_label": str(block.type), "box_source": "mineru4_middle_json_bbox",
            })
    boxes = build_image_boxes_payload(
        method="mineru_4011_standard", pdf_path=pdf_path, entries=entries,
        notes=["Official normalized 0–1 parent-block boxes converted to 0–1000; no pixel dimensions in PageInfo.",
               "Visual blocks without an exported image still retain their boxes; no synthetic image markers."],
    )
    return document, pages, boxes
