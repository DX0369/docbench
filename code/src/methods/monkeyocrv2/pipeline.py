"""MonkeyOCRv2 official AR pipeline, resident across the benchmark queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import (  # noqa: E402
    append_error_payload, build_error_payload, build_image_boxes_payload,
    build_page_records, concat_document_markdown, document_completed, error_context,
    extract_markdown_image_markers, get_doc_output_dir, normalize_box_0_1000,
    pdf_page_count, runtime_unusable, save_markdown_bundle, write_json,
)
from config.method_versions import get_method_version  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from method_paths import method_repo_path  # noqa: E402
from paths import TMP_ROOT  # noqa: E402

METHOD_ID = "monkeyocrv2_b_parsing"
DISPLAY_NAME = "MonkeyOCRv2-B-Parsing"
PROFILE_ID = "monkeyocrv2_b_parsing_vllm_ar"


def export_document(args, pdf, native, record, expected_pages, output, started, official):
    sizes = record["image_size"]
    # Official JSON represents a single page as [width, height].
    if expected_pages == 1:
        sizes = [sizes]
    if len(sizes) != expected_pages:
        raise ValueError("MonkeyOCRv2 image_size does not cover the input PDF")
    if any(len(size) != 2 or any(type(value) is not int or value <= 0 for value in size) for size in sizes):
        raise ValueError("MonkeyOCRv2 image_size must contain positive page width and height")
    layouts = record["layouts"]
    grouped = [[] for _ in range(expected_pages)]
    boxes = []
    for item in layouts:
        page = item["page_num"]
        if type(page) is not int or not 1 <= page <= expected_pages:
            raise ValueError(f"Invalid MonkeyOCRv2 page_num: {page!r}")
        item["content"]  # Required even if a legitimate block is empty.
        grouped[page - 1].append(item)
        if item["label"] == "Picture":
            markers = extract_markdown_image_markers(item["content"])
            if len(markers) != 1:
                raise ValueError("Picture block must reference exactly one image")
            name = markers[0]["name"]
            if not (native / "images" / name).is_file():
                raise FileNotFoundError(f"Missing official picture asset: {name}")
            boxes.append({
                "page_index": page, "marker_name": name,
                "marker_type": markers[0]["marker_type"],
                "page_marker_expression": markers[0]["match_text"],
                "page_marker_src": markers[0]["src"],
                "document_marker_expressions": [markers[0]["match_text"].replace("../images/", "images/")],
                "document_marker_sources": [f"images/{name}"],
                "absolute_box": item["bbox"],
                "relative_box_0_1000": normalize_box_0_1000(item["bbox"], *sizes[page - 1]),
                "page_size": sizes[page - 1], "box_source": "monkeyocrv2_preprocessed_layout_bbox",
            })
    pages = official.result2md([str(i) for i in range(expected_pages)], grouped)
    images = native / "images"
    if images.exists():
        shutil.copytree(images, output / "images")
    # Page files have the same depth as official markdowns; document.md is one
    # level higher, so only its image reference prefix changes.
    document = concat_document_markdown(pages).replace("../images/", "images/")
    save_markdown_bundle(output, pages, document)
    write_json(output / "layouts.json", {
        **record, "layouts": [{**item, "content": item["content"].replace("../images/", "images/")} for item in layouts],
    })
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=DISPLAY_NAME, pdf_path=pdf, entries=boxes,
        notes=["Coordinates use official preprocessed page sizes; referenced image assets are retained."],
    ))
    write_json(output / "metrics.json", {
        "method": DISPLAY_NAME, "method_id": METHOD_ID, "execution_profile": PROFILE_ID,
        "backend": "vllm_ar", "model_runner": "v2", "input_mode": "official_layout_recognition_pipeline",
        "pdf_path": str(pdf), "output_dir": str(output), "model_path": str(get_method_version(METHOD_ID).model_path),
        "page_count": expected_pages, "pages": build_page_records(pages, []),
        "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md", "page_max_inflight": args.page_max_inflight,
        "server_max_inflight": 1024, "preprocess_batch_size": 32, "max_pixels": 1003520,
        "request_timeout": 300, "http_max_retries": 5, "http_retry_backoff": 1.0,
        "dpi": 200, "asset_policy": "referenced_images_retained", "warnings": [],
    })


def process_document(args, pdf, pool, backend, official):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    started = time.perf_counter()
    with error_context("input_pdf"):
        expected = pdf_page_count(pdf)
    with tempfile.TemporaryDirectory(prefix="document_", dir=args.temp_root) as temporary:
        native = Path(temporary)
        with error_context("official_pipeline"):
            result = pool.run(official.PipelineConfig(
                input_path=str(pdf), output_path=str(native), backend=backend,
                page_max_inflight=args.page_max_inflight,
            ))
        with error_context("output_conversion"):
            records = json.loads(Path(result["all_results_path"]).read_text(encoding="utf-8"))
            # ServicePipelinePool records input_path.name in image_path.
            if len(records) != 1 or records[0]["image_path"] != pdf.name:
                raise ValueError("Official output does not identify the submitted PDF")
            output.mkdir(parents=True, exist_ok=True)
            export_document(args, pdf, native, records[0], expected, output, started, official)


def process_pdf_paths(args, pdf_paths, pool, backend, official):
    pending_paths = []
    for pdf in pdf_paths:
        output = get_doc_output_dir(args.input_root, args.output_root, pdf)
        if output.exists() and any(output.iterdir()):
            if args.skip_existing and document_completed(output):
                continue
            raise FileExistsError(f"Use a new run directory for existing output: {output}")
        pending_paths.append(pdf)
    failed = False
    # A fixed bounded document feed fills the shared official page window.
    # These threads own no model; P changes only the pool's page limit.
    with ThreadPoolExecutor(max_workers=min(8, len(pending_paths)) or 1) as feed:
        pending = {feed.submit(process_document, args, pdf, pool, backend, official): pdf for pdf in pending_paths}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                pdf = pending[future]
                append_error_payload(get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
                    build_error_payload(DISPLAY_NAME, pdf, exception=exc,
                        extra={"method_id": METHOD_ID, "execution_profile": PROFILE_ID}))
                failed = True
                if args.fail_fast or runtime_unusable(exc):
                    for other in pending:
                        other.cancel()
                    raise
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--page-max-inflight", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.page_max_inflight < 1:
        parser.error("--page-max-inflight must be positive")
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    parsing = method_repo_path("monkeyocrv2") / "parsing"
    sys.path.insert(0, str(parsing))
    import core_runner as official

    parsing_page = ContextVar("monkeyocrv2_parsing_page", default=None)

    class PipelinePool(official.ServicePipelinePool):
        def _parse_page(self, page):
            token = parsing_page.set(page["page_idx"] + 1)
            try:
                super()._parse_page(page)
            finally:
                parsing_page.reset(token)

        def _fail_job(self, job, stage, exc):
            if not hasattr(exc, "gdb_stage"):
                exc.gdb_stage = stage
            if parsing_page.get() is not None and not hasattr(exc, "gdb_page_number"):
                exc.gdb_page_number = parsing_page.get()
            super()._fail_job(job, stage, exc)

    backend = official.BackendConfig(
        model_path=str(get_method_version(METHOD_ID).model_path), server_url=args.server_url,
    )
    manager = official.BackendManager()
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="monkeyocrv2_ar_", dir=TMP_ROOT) as temporary:
        args.temp_root = Path(temporary)
        pool = None
        try:
            pool = PipelinePool(backend, args.page_max_inflight, backend_manager=manager)
            # Model and preprocessor initialization complete before queue ready.
            failed = run_pdf_queue_worker(
                queue_dir=args.queue_dir, worker_id=args.worker_id, fail_fast=args.fail_fast,
                input_root=args.input_root, output_root=args.output_root,
                process_job=lambda _claimed, paths: process_pdf_paths(args, paths, pool, backend, official),
            )
        finally:
            if pool is not None:
                pool.close()
            manager.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
