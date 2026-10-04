#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from common import (
    add_shard_args,
    apply_shard_to_pdf_paths,
    build_image_boxes_payload,
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_page_records,
    collect_pdf_paths,
    ensure_dir,
    extract_markdown_image_markers,
    get_doc_output_dir,
    load_pdf_paths_from_file,
    normalize_box_0_1000,
    parse_box_from_image_name,
    remove_file_if_exists,
    write_json,
    write_text,
)
from paths import TMP_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PaddleOCR-VL version with its official PaddleOCR pipeline backed by a vLLM server."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--method-id", default="paddleocr_vl_15")
    parser.add_argument("--method-display-name", default="PaddleOCR-VL-1.5")
    parser.add_argument("--pipeline-version", default="v1.5", choices=["v1.5", "v1.6"])
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8081/v1")
    parser.add_argument(
        "--api-model-name",
        type=str,
        default="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    parser.add_argument("--limit", type=int)
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--render-dpi", type=int, default=200,
        help="Target PDF DPI used by the official PaddleX renderer; oversized pages may be downscaled.",
    )
    parser.add_argument(
        "--vl-rec-max-concurrency",
        type=int,
        help="Maximum concurrent VLM recognition requests inside the official PaddleOCR-VL pipeline.",
    )
    parser.add_argument(
        "--use-queues",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable PaddleOCR-VL pipeline queues. If omitted, keep the official default.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.render_dpi <= 0:
        parser.error("--render-dpi must be positive")
    return args


def is_image_like_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    explicit_labels = {
        "image",
        "chart",
        "figure",
        "photo",
        "picture",
        "diagram",
        "graph",
    }
    return (
        normalized in explicit_labels
        or normalized.startswith("image_")
        or normalized.endswith("_image")
    )


def load_result_json_via_tempfile(result) -> Dict[str, Any]:
    ensure_dir(TMP_ROOT)
    with tempfile.NamedTemporaryFile(dir=TMP_ROOT, suffix=".json", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        result.save_to_json(temp_path)
        return json.loads(temp_path.read_text(encoding="utf-8"))
    finally:
        remove_file_if_exists(temp_path)


def extract_image_boxes_from_paddle_page(
    *,
    page_index: int,
    page_markdown: str,
    document_markers_by_name: Dict[str, List[Dict[str, Any]]],
    page_size: Any,
    page_result_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    page_width, page_height = page_size
    page_markers = extract_markdown_image_markers(page_markdown)
    entries: List[Dict[str, Any]] = []
    seen_keys = set()

    for page_marker in page_markers:
        absolute_box = parse_box_from_image_name(page_marker["name"])
        if absolute_box is None:
            continue
        key = tuple(round(value, 3) for value in absolute_box)
        seen_keys.add(key)
        entries.append(
            {
                "page_index": page_index,
                "marker_name": page_marker["name"],
                "marker_type": page_marker["marker_type"],
                "page_marker_expression": page_marker["match_text"],
                "page_marker_src": page_marker["src"],
                "document_marker_expressions": [
                    marker["match_text"]
                    for marker in document_markers_by_name.get(page_marker["name"], [])
                ],
                "document_marker_sources": [
                    marker["src"]
                    for marker in document_markers_by_name.get(page_marker["name"], [])
                ],
                "absolute_box": absolute_box,
                "relative_box_0_1000": normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_size": [page_width, page_height],
                "box_source": "markdown_image_name",
            }
        )

    for block in page_result_json.get("parsing_res_list", []):
        label = block.get("block_label")
        box = block.get("block_bbox")
        if not is_image_like_label(label) or not isinstance(box, list) or len(box) != 4:
            continue
        absolute_box = [float(value) for value in box]
        key = tuple(round(value, 3) for value in absolute_box)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entries.append(
            {
                "page_index": page_index,
                "marker_name": None,
                "marker_type": None,
                "page_marker_src": None,
                "document_marker_sources": [],
                "absolute_box": absolute_box,
                "relative_box_0_1000": normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_size": [page_width, page_height],
                "box_source": "paddle_parsing_res_list",
                "block_label": label,
                "block_content_preview": str(block.get("block_content", ""))[:200],
            }
        )

    return entries


def export_document(args, pipeline, pdf_path, page_results, doc_total_start):
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)

    notes: List[str] = [
        f"{args.method_display_name} is run through the official PaddleOCRVL pipeline with "
        "vl_rec_backend='vllm-server', so the input stays as a whole PDF document.",
        "PaddleOCR-VL internal VLM request concurrency and queue mode are recorded in metrics.json.",
        "The official whole-document pipeline does not expose reliable page-level "
        "latencies, so page latencies are recorded as 'N/A'.",
        "Native Markdown and image coordinates are retained; output image assets are not saved.",
        "PDF rendering uses the official PaddleX renderer at the target render_dpi; oversized pages may be downscaled.",
    ]

    page_markdowns: List[str] = []
    page_result_jsons: List[Dict[str, Any]] = []
    page_sizes: List[List[int]] = []
    pages_dir = doc_output_dir / "pages"
    ensure_dir(pages_dir)
    for page_index, res in enumerate(page_results, start=1):
        with error_context("page_export", page_index):
            page_path = pages_dir / f"page_{page_index:04d}.md"
            markdown = res.markdown["markdown_texts"]
            if not isinstance(markdown, str):
                raise TypeError("Official markdown_texts must be a string")
            write_text(page_path, markdown)
            page_markdowns.append(markdown.strip())
            page_result_json = load_result_json_via_tempfile(res)
            page_size = [page_result_json["width"], page_result_json["height"]]
            if any(type(value) is not int or value <= 0 for value in page_size):
                raise ValueError("Official page width and height must be positive integers")
            page_result_jsons.append(page_result_json)
            page_sizes.append(page_size)

    restructure_start = time.perf_counter()
    with error_context("restructure"):
        final_results = pipeline.restructure_pages(
            page_results,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=True,
        )
        final_results = list(final_results)
        if not final_results:
            raise RuntimeError("restructure_pages returned no results")
        document_markdown = final_results[0].markdown["markdown_texts"]
        if not isinstance(document_markdown, str):
            raise TypeError("Official markdown_texts must be a string")
        write_text(doc_output_dir / "document.md", document_markdown)
        document_markdown = document_markdown.strip()
        restructure_seconds: Any = round(time.perf_counter() - restructure_start, 6)

    document_markers = extract_markdown_image_markers(document_markdown)
    document_markers_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for document_marker in document_markers:
        document_markers_by_name.setdefault(document_marker["name"], []).append(
            document_marker
        )
    image_box_entries: List[Dict[str, Any]] = []
    for page_index, page_markdown in enumerate(page_markdowns, start=1):
        page_result_json = page_result_jsons[page_index - 1]
        page_size = page_sizes[page_index - 1]
        image_box_entries.extend(
            extract_image_boxes_from_paddle_page(
                page_index=page_index,
                page_markdown=page_markdown,
                document_markers_by_name=document_markers_by_name,
                page_size=page_size,
                page_result_json=page_result_json,
            )
        )

    page_latencies: List[str] = ["N/A"] * len(page_markdowns)
    pages = build_page_records(page_markdowns, page_latencies)
    for page_record, page_size in zip(pages, page_sizes):
        page_record["page_size"] = page_size
    doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
    metrics: Dict[str, Any] = {
        "method": args.method_display_name,
        "method_id": args.method_id,
        "execution_profile": args.method_id + "_multi_pdf_queue",
        "input_feed": "official_multi_pdf_predict_iter",
        "backend": "vllm",
        "input_mode": "native_pdf",
        "pdf_path": str(pdf_path),
        "output_dir": str(doc_output_dir),
        "server_url": args.server_url,
        "api_model_name": args.api_model_name,
        "pipeline_version": args.pipeline_version,
        "vl_rec_max_concurrency": args.vl_rec_max_concurrency,
        "use_queues": args.use_queues,
        "page_count": len(page_markdowns),
        "render_seconds": "N/A",
        "render_dpi": args.render_dpi,
        "pdf_render_scale": args.render_dpi / 72,
        "page_inference_seconds_sum": "N/A",
        "restructure_seconds": restructure_seconds,
        "doc_total_seconds": doc_total_seconds,
        "document_markdown_path": "document.md",
        "pages": pages,
        "notes": notes,
    }
    if args.max_new_tokens is not None:
        metrics["max_new_tokens"] = args.max_new_tokens
    write_json(doc_output_dir / "metrics.json", metrics)
    write_json(
        doc_output_dir / "image_boxes.json",
        build_image_boxes_payload(
            method=args.method_display_name,
            pdf_path=pdf_path,
            entries=image_box_entries,
            notes=[
                "Image coordinates are extracted from official PaddleOCR-VL outputs.",
                "Coordinate boxes are normalized to a 0-1000 page-relative scale using each official page result's width and height, in the layout image coordinate system.",
                "Markdown image assets are not saved; native references remain as markers only.",
            ],
        ),
    )
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")


def load_paddle_runtime(args):
    # PaddleX reads this flag once, at import. Set it before importing the SDK
    # in every standalone or persistent pipeline worker, never in the VLM service.
    os.environ["PADDLE_PDX_PDF_RENDER_SCALE"] = str(args.render_dpi / 72)
    from paddleocr import PaddleOCRVL  # noqa: E402

    pipeline_kwargs = {
        "pipeline_version": args.pipeline_version,
        "vl_rec_backend": "vllm-server",
        "vl_rec_server_url": args.server_url,
        "vl_rec_api_model_name": args.api_model_name,
    }
    if args.vl_rec_max_concurrency is not None:
        pipeline_kwargs["vl_rec_max_concurrency"] = args.vl_rec_max_concurrency
    if args.use_queues is not None:
        pipeline_kwargs["use_queues"] = args.use_queues
    return PaddleOCRVL(**pipeline_kwargs)


def write_pdf_error(args, pdf, exc, *, batch=None):
    extra = {"backend": "vllm", "method_id": args.method_id,
             "execution_profile": args.method_id + "_multi_pdf_queue"}
    if batch is not None:
        # Native queue errors do not identify a PDF-local page.
        extra.update(stage="batch_inference", page_number=None, page_numbers=[],
                     affected_inputs=[str(path) for path in batch])
    write_json(get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
               build_error_payload(args.method_display_name, pdf, exception=exc, extra=extra))
    print(f"[error] {pdf}: {exc}", flush=True)


def process_pdf_paths(args, pdf_paths, *, pipeline):
    paths = {str(path.resolve()): path for path in pdf_paths}
    if len(paths) != len(pdf_paths):
        raise ValueError("Duplicate input PDF in Paddle batch")
    for key, path in list(paths.items()):
        output = get_doc_output_dir(args.input_root, args.output_root, path)
        if args.skip_existing and document_completed(output):
            if json.loads((output / "metrics.json").read_text())["execution_profile"] != args.method_id + "_multi_pdf_queue":
                raise ValueError(f"Cannot reuse a different Paddle execution profile: {output}")
            del paths[key]
    if not paths:
        return False
    started = time.perf_counter()
    pages = {key: {} for key in paths}
    counts, seen, finished = {}, {key: set() for key in paths}, set()
    had_failure = False
    kwargs = {} if args.max_new_tokens is None else {"max_new_tokens": args.max_new_tokens}
    try:
        for result in pipeline.predict_iter(list(paths), **kwargs):
            key = str(Path(result["input_path"]).resolve())
            if key not in paths:
                raise ValueError(f"Paddle returned an unknown input_path: {key}")
            index, count = result["page_index"], result["page_count"]
            if (type(index) is not int or type(count) is not int or count < 1
                    or not 0 <= index < count or index in seen[key]
                    or (key in counts and counts[key] != count)):
                exc = ValueError(f"Invalid or repeated Paddle page identity: {key}, {index}, {count}")
                write_pdf_error(args, paths[key], exc)
                finished.add(key)
                raise exc
            counts[key] = count
            seen[key].add(index)
            pages[key][index] = result
            if len(pages[key]) == count:
                try:
                    export_document(args, pipeline, paths[key],
                                    [pages[key][i] for i in range(count)], started)
                except Exception as exc:
                    had_failure = True
                    write_pdf_error(args, paths[key], exc)
                    if args.fail_fast or runtime_unusable(exc):
                        raise
                finally:
                    finished.add(key)
                    pages[key].clear()
        remaining = [paths[key] for key in paths if key not in finished]
        if remaining:
            raise RuntimeError("Paddle batch ended before every input produced all declared pages")
    except Exception as exc:
        had_failure = True
        remaining = [paths[key] for key in paths if key not in finished]
        for pdf in remaining:
            write_pdf_error(args, pdf, exc, batch=remaining)
        # A trailing iterator/protocol error belongs to the job, not a PDF
        # already exported. Propagate it so the queue records the actual cause.
        if not remaining or args.fail_fast or runtime_unusable(exc):
            raise
    return had_failure


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.queue_dir is not None and (not args.worker_id or args.pdf_list_file is not None or args.limit is not None or args.num_shards != 1):
        raise ValueError("Queue mode requires worker-id and cannot be combined with PDF list, limit or sharding")
    pipeline = load_paddle_runtime(args)
    try:
        if args.queue_dir is not None:
            from execution.workers import run_pdf_queue_worker
            had_failure = run_pdf_queue_worker(
                queue_dir=args.queue_dir.resolve(), worker_id=args.worker_id, fail_fast=args.fail_fast,
                input_root=args.input_root, output_root=args.output_root,
                process_job=lambda _, pdfs: process_pdf_paths(args, pdfs, pipeline=pipeline))
        else:
            if args.pdf_list_file is not None:
                pdf_paths = load_pdf_paths_from_file(args.pdf_list_file.resolve(), limit=args.limit)
            else:
                pdf_paths = collect_pdf_paths(args.input_root, limit=args.limit)
            pdf_paths = apply_shard_to_pdf_paths(pdf_paths, num_shards=args.num_shards, shard_index=args.shard_index)
            had_failure = process_pdf_paths(args, pdf_paths, pipeline=pipeline)
    finally:
        pipeline.close()
    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
