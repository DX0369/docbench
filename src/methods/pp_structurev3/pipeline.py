#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from paths import MODEL_ROOT, TMP_ROOT  # noqa: E402
from methods.pp_structurev3.commands import add_runtime_args
from methods.pp_structurev3.runtime import scaled_config, prepare_runtime, observe_runtime
from methods.pp_structurev3.prefetch import (
    INPUT_PIPELINE, PREFETCH_BATCHES, PREFETCH_MAX_BYTES,
    PagePrefetch, RenderError, native_pipeline, prepared_batch, render_pages,
)

DEFAULT_CACHE_ROOT = MODEL_ROOT / "runtime_cache" / "paddle_ppstructurev3"

from common import (
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_image_boxes_payload,
    build_page_records,
    collect_pdf_paths,
    extract_markdown_image_markers,
    get_doc_output_dir,
    normalize_box_0_1000,
    parse_box_from_image_name,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP-StructureV3 with its official native PaddleOCR pipeline."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--lang", type=str)
    add_runtime_args(parser, prefix="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def coerce_markdown_document_text(merged_markdown: Any) -> str:
    if isinstance(merged_markdown, str):
        return merged_markdown.strip()
    if isinstance(merged_markdown, dict):
        return str(merged_markdown.get("markdown_texts", "")).strip()

    markdown_attr = getattr(merged_markdown, "markdown", None)
    if isinstance(markdown_attr, dict):
        return str(markdown_attr.get("markdown_texts", "")).strip()

    markdown_texts_attr = getattr(merged_markdown, "markdown_texts", None)
    if markdown_texts_attr is not None:
        return str(markdown_texts_attr).strip()

    text_attr = getattr(merged_markdown, "text", None)
    if text_attr is not None:
        return str(text_attr).strip()

    return str(merged_markdown).strip()


def load_result_json(result) -> Dict[str, Any]:
    if hasattr(result, "json"):
        raw = result.json
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
    runtime_root = TMP_ROOT / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=runtime_root) as tmp:
        temp_path = Path(tmp.name)
    try:
        result.save_to_json(temp_path)
        return json.loads(temp_path.read_text(encoding="utf-8"))
    finally:
        remove_file_if_exists(temp_path)


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
        "header_image",
        "footer_image",
    }
    return (
        normalized in explicit_labels
        or normalized.startswith("image_")
        or normalized.endswith("_image")
    )


def coerce_expression_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def build_expression_fields(
    *,
    primary_text: Any,
    fallback_text: Any = None,
) -> Dict[str, Any]:
    expression = coerce_expression_text(primary_text)
    if expression is None:
        expression = coerce_expression_text(fallback_text)
    preview = (expression or "")[:200]
    return {
        "page_marker_expression": expression,
        "document_marker_expressions": [expression] if expression else [],
        "block_content_preview": preview,
    }


def extract_image_boxes_from_ppstructure_page(
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

    res_payload = page_result_json.get("res", page_result_json)

    for block in res_payload.get("parsing_res_list", []):
        label = block.get("block_label")
        box = block.get("block_bbox")
        if not is_image_like_label(label) or not isinstance(box, list) or len(box) != 4:
            continue
        absolute_box = [float(value) for value in box]
        key = tuple(round(value, 3) for value in absolute_box)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        expression_fields = build_expression_fields(
            primary_text=block.get("block_content"),
            fallback_text=label,
        )
        entries.append(
            {
                "page_index": page_index,
                "marker_name": None,
                "marker_type": None,
                "page_marker_expression": expression_fields["page_marker_expression"],
                "page_marker_src": None,
                "document_marker_expressions": expression_fields[
                    "document_marker_expressions"
                ],
                "document_marker_sources": [],
                "absolute_box": absolute_box,
                "relative_box_0_1000": normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_size": [page_width, page_height],
                "box_source": "pp_structure_parsing_res_list",
                "block_label": label,
                "block_content_preview": expression_fields["block_content_preview"],
            }
        )

    layout_boxes = res_payload.get("layout_det_res", {}).get("boxes", [])
    for block in layout_boxes:
        label = block.get("label")
        box = block.get("coordinate")
        if not is_image_like_label(label) or not isinstance(box, list) or len(box) != 4:
            continue
        absolute_box = [float(value) for value in box]
        key = tuple(round(value, 3) for value in absolute_box)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        expression_fields = build_expression_fields(
            primary_text=block.get("block_content"),
            fallback_text=label,
        )
        entries.append(
            {
                "page_index": page_index,
                "marker_name": None,
                "marker_type": None,
                "page_marker_expression": expression_fields["page_marker_expression"],
                "page_marker_src": None,
                "document_marker_expressions": expression_fields[
                    "document_marker_expressions"
                ],
                "document_marker_sources": [],
                "absolute_box": absolute_box,
                "relative_box_0_1000": normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_size": [page_width, page_height],
                "box_source": "pp_structure_layout_det_res",
                "block_label": label,
                "block_content_preview": expression_fields["block_content_preview"],
                "score": block.get("score"),
            }
        )

    return entries


def read_page_result(result) -> Dict[str, Any]:
    # Canonical native dimensions share the block_bbox coordinate system.
    page_size = [result["width"], result["height"]]
    if any(type(value) is not int or value <= 0 for value in page_size):
        raise ValueError("Official page width and height must be positive integers")
    markdown = result.markdown
    # Native markdown_images contains PIL crops. Export only the text and the
    # continuation flags used by official concatenation, not all page images.
    return {
        "markdown": {key: markdown[key] for key in ("markdown_texts", "page_continuation_flags")},
        "json": load_result_json(result),
        "size": page_size,
    }


def export_pdf(args, pipeline, pdf_path, page_data, started) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    markdown_infos = [page["markdown"] for page in page_data]
    page_markdowns = [markdown["markdown_texts"].strip() for markdown in markdown_infos]
    page_result_jsons = [page["json"] for page in page_data]
    page_sizes = [page["size"] for page in page_data]
    with error_context("export"):
        document_markdown = coerce_markdown_document_text(
            pipeline.concatenate_markdown_pages(markdown_infos)
        )
        if document_markdown:
            document_markdown = document_markdown.rstrip() + "\n"
        save_markdown_bundle(
            doc_output_dir=doc_output_dir,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
        )

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
                extract_image_boxes_from_ppstructure_page(
                    page_index=page_index,
                    page_markdown=page_markdown,
                    document_markers_by_name=document_markers_by_name,
                    page_size=page_size,
                    page_result_json=page_result_json,
                )
            )

        doc_total_seconds = round(time.perf_counter() - started, 6)
        pages = build_page_records(page_markdowns, ["N/A"] * len(page_markdowns))
        for page_record, page_size in zip(pages, page_sizes):
            page_record["page_size"] = page_size
        metrics: Dict[str, Any] = {
            "method": "PP-StructureV3",
            "backend": "native_pipeline",
            "input_mode": "native_pdf",
            "execution_mode": "page_batch",
            "input_pipeline": INPUT_PIPELINE,
            "render_dpi": args.render_dpi,
            "pdf_render_scale": args.render_dpi / 72,
            "input_prefetch": {"queue_batches": PREFETCH_BATCHES, "max_queue_bytes": PREFETCH_MAX_BYTES},
            "pdf_path": str(pdf_path),
            "output_dir": str(doc_output_dir),
            "device": args.device,
            "lang": args.lang,
            "runtime_options": {
                "batch_scale": args.batch_scale,
            },
            "page_count": len(page_markdowns),
            "render_seconds": "N/A",
            "page_inference_seconds_sum": "N/A",
            "doc_total_seconds": doc_total_seconds,
            "document_markdown_path": "document.md",
            "pages": pages,
            "notes": [
                "PP-StructureV3 is run through its official native PaddleOCR pipeline and does not use a prompt.",
                "PP-StructureV3 is not a vLLM-native method; it is included here as an official document-parsing pipeline baseline.",
                "The official whole-document pipeline does not expose reliable page-level latency, so page latencies are recorded as 'N/A'.",
                "In page_batch mode, document wall time includes rendering and prefetch waiting and must not be summed as inference time.",
                "Coordinate boxes are normalized to a 0-1000 page-relative scale using each official page result's width and height, in the layout image coordinate system; no additional PDF rendering is performed.",
            ],
        }
        write_json(doc_output_dir / "metrics.json", metrics)
        write_json(
            doc_output_dir / "image_boxes.json",
            build_image_boxes_payload(
                method="PP-StructureV3",
                pdf_path=pdf_path,
                entries=image_box_entries,
                notes=[
                    "Image coordinates are extracted from official PP-StructureV3 structured results and markdown markers when available.",
                    "If a page image is present in the markdown but no parseable coordinate can be matched, the corresponding coordinate fields remain empty.",
                ],
            ),
        )
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")


def should_skip_pdf(args, pdf_path) -> bool:
    output = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if not args.skip_existing or not document_completed(output):
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics.get("execution_mode") != "page_batch" or metrics.get("render_dpi") != args.render_dpi:
        raise ValueError("Existing PP output uses a different or missing mode/DPI; use a new output directory")
    if metrics.get("input_pipeline") != INPUT_PIPELINE:
        raise ValueError("Existing PP output predates page prefetch; use a new output directory")
    if metrics["runtime_options"]["batch_scale"] != args.batch_scale:
        raise ValueError("Existing PP output uses a different batch scale; use a new output directory")
    print(f"[skip] {pdf_path}")
    return True


def process_prefetched_pdfs(args, runtime, pdf_paths, *, job_status=None) -> bool:
    """Render continuously, infer one native page batch, then scatter by PDF."""
    pipeline, native = runtime["pipeline"], runtime["native"]
    sampler = native.batch_sampler
    states = {}
    failed = set()
    had_failure = False
    status = job_status if job_status is not None else {}
    status["batch_errors"] = []
    selected = []
    for pdf_path in pdf_paths:
        try:
            if not should_skip_pdf(args, pdf_path):
                selected.append(pdf_path)
        except Exception:
            had_failure = True
            write_pdf_error(args, pdf_path)
            if args.fail_fast or runtime_unusable(sys.exc_info()[1]):
                raise
    if not selected:
        return had_failure
    input_paths = {pdf.resolve(): pdf for pdf in selected}
    if len(input_paths) != len(selected):
        raise ValueError("PP input contains duplicate PDF paths")

    with PagePrefetch(selected, lambda pdf: render_pages(sampler.pdf_reader, pdf),
                      sampler.batch_size) as supply:
        def fail(pdf, exc, *, extra=None, stop=True):
            nonlocal had_failure
            had_failure = True
            failed.add(pdf)
            states.pop(pdf, None)
            supply.skip(pdf)
            write_pdf_error(args, pdf, exception=exc, extra=extra)
            if stop and (args.fail_fast or runtime_unusable(exc)):
                raise exc

        exhausted = False
        while not exhausted:
            batch = []
            while len(batch) < sampler.batch_size:
                item = supply.get()
                if item is None:
                    exhausted = True
                    break
                if isinstance(item, RenderError):
                    fail(item.pdf, item.exception)
                    batch = [page for page in batch if page.pdf != item.pdf]
                    continue
                if item.pdf in failed:
                    continue
                states.setdefault(item.pdf, {"count": item.count, "pages": {}, "started": item.started})
                batch.append(item)
            if not batch:
                continue
            expected = {(page.pdf, page.index) for page in batch}
            group_paths = list(dict.fromkeys(page.pdf for page in batch))
            group_error = None
            try:
                with error_context("batch_inference"), prepared_batch(native, batch) as paths:
                    # Close the generator before restoring its original sampler,
                    # including fail-fast and invalid-result exits.
                    with closing(pipeline.predict_iter(paths)) as results:
                        for result in results:
                            source = result["input_path"]
                            if not isinstance(source, str) or not source:
                                raise ValueError("PP result must identify its input PDF")
                            pdf = input_paths.get(Path(source).resolve())
                            if pdf not in group_paths:
                                raise ValueError(f"PP result references an unrequested PDF: {source}")
                            if pdf in failed:
                                continue
                            state = states[pdf]
                            try:
                                with error_context("page_export"):
                                    index = result["page_index"]
                                    with error_context("page_export", index + 1 if type(index) is int else None):
                                        count = result["page_count"]
                                        if type(count) is not int or count != state["count"]:
                                            raise ValueError("PP result page_count differs from input PDF")
                                        if type(index) is not int or (pdf, index) not in expected:
                                            raise ValueError(f"Invalid PP page_index for current batch: {index!r}")
                                        if index in state["pages"]:
                                            raise ValueError(f"Duplicate PP page_index: {index}")
                                        state["pages"][index] = read_page_result(result)
                            except Exception as exc:
                                fail(pdf, exc)
            except Exception as exc:
                group_error = exc
                had_failure = True
                # A generator may fail after all valid pages were yielded. Keep
                # completed documents, but never lose the unassigned batch error.
                status["batch_errors"].append({
                    "stage": getattr(exc, "gdb_stage", "batch_inference"),
                    "error_type": type(exc).__name__, "message": str(exc),
                    "pages": [{"pdf_path": str(page.pdf), "page_index": page.index} for page in batch],
                })
                print("[pp-batch-error] " + json.dumps(status["batch_errors"][-1], sort_keys=True))
                traceback.print_exc()

            for pdf in group_paths:
                if pdf in failed:
                    continue
                state = states[pdf]
                if group_error is not None and len(state["pages"]) != state["count"]:
                    fail(pdf, group_error, stop=False, extra={
                        "failure_scope": "shared_page_batch",
                        "group_pdf_paths": [str(path) for path in group_paths],
                        "batch_page_indices": [page.index for page in batch if page.pdf == pdf],
                        "completed_page_indices": sorted(state["pages"]),
                        "page_number": None,
                        "notes": ["Native inference failed without locating the bad PDF; no retry. Remaining pages of failed PDFs are skipped."],
                    })
                    continue
                missing = sorted(index for path, index in expected if path == pdf and index not in state["pages"])
                if missing:
                    exc = ValueError(f"PP output missing pages from current batch: {missing}")
                    exc.gdb_stage = "output_page_coverage"
                    fail(pdf, exc)
                    continue
                if len(state["pages"]) == state["count"]:
                    try:
                        ordered = [state["pages"][index] for index in range(state["count"])]
                        export_pdf(args, pipeline, pdf, ordered, state["started"])
                        del states[pdf]
                    except Exception as exc:
                        fail(pdf, exc)
            if group_error is not None and (args.fail_fast or runtime_unusable(group_error)):
                raise group_error
            # Do not retain a consumed batch while waiting for the next one.
            batch.clear()
        for pdf in list(states):
            exc = ValueError("PP input stream ended with incomplete document output")
            exc.gdb_stage = "output_page_coverage"
            fail(pdf, exc)
    status["input_prefetch"] = supply.stats
    print("[pp-prefetch] " + json.dumps(supply.stats, sort_keys=True))
    return had_failure


def write_pdf_error(args: argparse.Namespace, pdf_path: Path, *, exception=None, extra=None) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    write_json(
        doc_output_dir / "error.json",
        build_error_payload(
            method_name="PP-StructureV3",
            pdf_path=pdf_path,
            exception=exception,
            extra={
                "backend": "native_pipeline",
                "device": args.device,
                "lang": args.lang,
                "execution_mode": "page_batch",
                "render_dpi": args.render_dpi,
                "runtime_options": {
                    "batch_scale": args.batch_scale,
                },
                **(extra or {}),
            },
        ),
    )
    print(f"[error] {pdf_path}")


def load_ppstructure_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    cache_root = DEFAULT_CACHE_ROOT
    paddlex_cache_home = cache_root / "paddlex_cache"
    xdg_cache_home = cache_root / "xdg_cache"
    hf_home = cache_root / "hf_home"
    modelscope_cache = cache_root / "modelscope"
    for path in (
        cache_root,
        paddlex_cache_home,
        xdg_cache_home,
        hf_home,
        modelscope_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddlex_cache_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("MODELSCOPE_CACHE", str(modelscope_cache))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for proxy_env in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        proxy_value = os.environ.get(proxy_env)
        if proxy_value and "127.0.0.1" in proxy_value:
            os.environ.pop(proxy_env, None)

    # PaddleX reads this flag at import; scope it to this worker, not the shell.
    os.environ["PADDLE_PDX_PDF_RENDER_SCALE"] = str(args.render_dpi / 72)
    from paddleocr import PPStructureV3

    init_kwargs: Dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "device": args.device,
    }
    if args.lang:
        init_kwargs["lang"] = args.lang
    if args.batch_scale != 1:
        from paddlex.inference import load_pipeline_config
        init_kwargs["paddlex_config"] = scaled_config(load_pipeline_config("PP-StructureV3"), args.batch_scale)
    pipeline = PPStructureV3(**init_kwargs)
    runtime = {"pipeline": pipeline, "native": native_pipeline(pipeline)}
    runtime["effective_batches"] = prepare_runtime(runtime["native"])
    print("[pp-ready] " + json.dumps({"batch_scale": args.batch_scale,
          "effective_batches": runtime["effective_batches"], "table_aux_preloaded": True}, sort_keys=True))
    return runtime


def process_pdf_paths(
    args: argparse.Namespace,
    pdf_paths: List[Path],
    *,
    runtime: Dict[str, Any],
    job_status: Dict[str, Any] | None = None,
) -> bool:
    with observe_runtime(runtime["native"]) as observations:
        try:
            return process_prefetched_pdfs(args, runtime, pdf_paths, job_status=job_status)
        finally:
            if job_status is not None:
                job_status["effective_batches"] = runtime["effective_batches"]
                job_status["runtime_observations"] = observations
            print("[pp-stages] " + json.dumps(observations, sort_keys=True))


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    runtime = load_ppstructure_runtime(args)

    pdf_paths = collect_pdf_paths(args.input_root, limit=args.limit)
    had_failure = process_pdf_paths(args, pdf_paths, runtime=runtime)
    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
