#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path, require_failure_patch  # noqa: E402

MONKEY_REPO_ROOT = method_repo_path("monkeyocr")
if str(MONKEY_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MONKEY_REPO_ROOT))

from common import (  # noqa: E402
    add_shard_args,
    apply_shard_to_pdf_paths,
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_image_boxes_payload,
    build_page_records,
    collect_pdf_paths,
    extract_markdown_image_markers,
    get_doc_output_dir,
    load_pdf_paths_from_file,
    normalize_box_0_1000,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)
from methods.monkeyocr.async_runtime import AsyncRecognitionLoop  # noqa: E402
from paths import TMP_ROOT
DEFAULT_CONFIG_PATH = MONKEY_REPO_ROOT / "model_configs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MonkeyOCR with its native multi-stage parser using the official vLLM backend."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-id", type=str, required=True)
    parser.add_argument("--method-display-name", type=str, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--region-concurrency", type=int, default=64,
                        help="vllm_async in-flight region limit; official default: 64.")
    parser.add_argument("--model-parallelism", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--chat-weight-path", type=str)
    parser.add_argument("--pred-abandon", action="store_true")
    parser.add_argument("--limit", type=int)
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def prepare_runtime_config(args: argparse.Namespace) -> Path:
    if args.region_concurrency < 1:
        raise ValueError("--region-concurrency must be positive")
    source_config = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    source_config.setdefault("chat_config", {})
    source_config["chat_config"]["backend"] = "vllm_async"
    source_config["chat_config"]["region_concurrency"] = args.region_concurrency
    if args.model_parallelism is not None:
        source_config["chat_config"]["model_parallelism"] = args.model_parallelism
    if args.device is not None:
        source_config["device"] = args.device
    if args.models_dir is not None:
        source_config["models_dir"] = str(args.models_dir)
        current_weight_path = str(
            source_config["chat_config"].get("weight_path", "") or ""
        ).strip()
        if args.chat_weight_path is None:
            if current_weight_path.startswith("model_weight/"):
                source_config["chat_config"]["weight_path"] = current_weight_path[
                    len("model_weight/") :
                ]
            elif not current_weight_path:
                source_config["chat_config"]["weight_path"] = "Recognition"
    if args.chat_weight_path is not None:
        source_config["chat_config"]["weight_path"] = args.chat_weight_path

    runtime_dir = TMP_ROOT / "runtime" / "monkeyocr"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=runtime_dir,
        suffix=".yaml",
        prefix="gdb_monkey_config_",
        delete=False,
    ) as tmp:
        temp_config_path = Path(tmp.name)
    temp_config_path.write_text(
        yaml.safe_dump(source_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return temp_config_path


def load_monkeyocr_runtime(runtime_config_path: Path) -> Dict[str, Any]:
    require_failure_patch("monkeyocr")
    chat_backend = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))["chat_config"]["backend"]
    if chat_backend != "vllm_async":
        raise ValueError("MonkeyOCR supports only the vllm_async runtime")
    require_failure_patch("monkeyocr", "region_concurrency.patch")
    require_failure_patch("monkeyocr", "tiny_image.patch")
    import transformers.modeling_utils as modeling_utils  # noqa: E402

    if not hasattr(modeling_utils, "init_empty_weights"):
        from accelerate import init_empty_weights  # noqa: E402

        modeling_utils.init_empty_weights = init_empty_weights

    from magic_pdf.model.custom_model import MonkeyOCR  # noqa: E402
    from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm  # noqa: E402

    monkey_model = MonkeyOCR(str(runtime_config_path))
    async_recognition = AsyncRecognitionLoop(monkey_model.chat_model)
    monkey_model.chat_model.batch_inference = async_recognition.batch_inference
    return {
        "monkey_model": monkey_model,
        "doc_analyze_proc": doc_analyze_llm,
        "async_recognition": async_recognition,
    }


def close_monkeyocr_runtime(runtime: Optional[Dict[str, Any]]) -> None:
    """Release the V1 engine and its loop after the entire document queue."""
    if runtime is not None:
        runtime["async_recognition"].close()


def iter_nested_dicts(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from iter_nested_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_nested_dicts(item)


def flatten_text_list(values: Any) -> List[str]:
    flattened: List[str] = []
    if isinstance(values, list):
        for item in values:
            flattened.extend(flatten_text_list(item))
    elif values is None:
        return flattened
    else:
        text = str(values).strip()
        if text:
            flattened.append(text)
    return flattened


def extract_monkey_image_boxes_for_page(
    *,
    page_index: int,
    page_markdown: str,
    document_markdown: str,
    page_size: List[int],
    page_content_list: Any,
    page_middle_json: Dict[str, Any],
) -> List[Dict[str, Any]]:
    page_width, page_height = page_size
    page_markers = extract_markdown_image_markers(page_markdown)
    page_markers_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for marker in page_markers:
        page_markers_by_name.setdefault(marker["name"], []).append(marker)

    document_markers = extract_markdown_image_markers(document_markdown)
    document_markers_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for marker in document_markers:
        document_markers_by_name.setdefault(marker["name"], []).append(marker)

    middle_candidates: Dict[str, List[Dict[str, Any]]] = {}
    for item in iter_nested_dicts(page_middle_json):
        image_path = item.get("image_path")
        bbox = item.get("bbox")
        if not isinstance(image_path, str) or not image_path:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        middle_candidates.setdefault(Path(image_path).name, []).append(item)

    entries: List[Dict[str, Any]] = []
    if not isinstance(page_content_list, list):
        return entries

    for item in page_content_list:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() != "image":
            continue
        img_path = str(item.get("img_path", "") or "").strip()
        marker_name = Path(img_path).name if img_path else None
        middle_match = None
        if marker_name and middle_candidates.get(marker_name):
            middle_match = middle_candidates[marker_name].pop(0)

        page_marker = None
        if marker_name and page_markers_by_name.get(marker_name):
            page_marker = page_markers_by_name[marker_name][0]

        absolute_box = None
        relative_box = None
        if middle_match is not None:
            absolute_box = [float(value) for value in middle_match["bbox"]]
            if page_width > 0 and page_height > 0:
                relative_box = normalize_box_0_1000(
                    absolute_box,
                    page_width=page_width,
                    page_height=page_height,
                )

        entries.append(
            {
                "page_index": page_index,
                "marker_name": marker_name,
                "marker_type": page_marker["marker_type"] if page_marker else None,
                "page_marker_expression": (
                    page_marker["match_text"]
                    if page_marker
                    else (f"![]({img_path})" if img_path else None)
                ),
                "page_marker_src": page_marker["src"] if page_marker else img_path,
                "document_marker_expressions": (
                    [
                        marker["match_text"]
                        for marker in document_markers_by_name.get(marker_name, [])
                    ]
                    if marker_name
                    else []
                ),
                "document_marker_sources": (
                    [marker["src"] for marker in document_markers_by_name.get(marker_name, [])]
                    if marker_name
                    else []
                ),
                "absolute_box": absolute_box,
                "relative_box_0_1000": relative_box,
                "page_size": [page_width, page_height],
                "box_source": (
                    "monkey_middle_json_bbox" if middle_match is not None else "unavailable"
                ),
                "image_caption_text": "\n".join(flatten_text_list(item.get("img_caption", []))),
                "image_footnote_text": "\n".join(flatten_text_list(item.get("img_footnote", []))),
            }
        )

    return entries


# Fixed supply window from the official README's -g 20 example, not a
# second tuning axis. An oversized PDF remains one group and is never split.
GROUP_PAGE_LIMIT = 20
REGION_IMAGE_NORMALIZATION = "qwen_fetch_image_if_short_side_lt_28_v1"


def analyze_group(runtime, paths, page_counts, pred_abandon):
    from magic_pdf.data.dataset import MultiFileDataset

    dataset = MultiFileDataset([pdf.read_bytes() for pdf in paths], [".pdf"] * len(paths))
    info = dataset.file_info
    if len(info) != len(paths):
        raise ValueError("Monkey dataset omitted an input file")
    offset = 0
    for i, (item, count) in enumerate(zip(info, page_counts)):
        if (item["file_index"] != i or item["file_type"] != "pdf"
                or item["page_count"] != count or item["start_page"] != offset
                or item["end_page"] != offset + count - 1):
            raise ValueError("Monkey dataset file identity/page range mismatch")
        offset += count
    results = dataset.apply(runtime["doc_analyze_proc"], MonkeyOCR_model=runtime["monkey_model"],
                            split_files=True, split_pages=False, pred_abandon=pred_abandon)
    if not isinstance(results, list) or len(results) != len(paths):
        raise ValueError("Monkey split_files did not return exactly one result per input")
    for result, count in zip(results, page_counts):
        records = result.get_infer_res()
        if [page["page_info"]["page_no"] for page in records] != list(range(count)):
            raise ValueError("Monkey returned missing, duplicate or reordered file-local pages")
    return results


def export_document(args, runtime, pdf, result, expected_pages, started, group_paths, group_page_count):
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.operators.pipes_llm import PipeResultLLM

    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    temp_root = TMP_ROOT / "runtime" / "monkeyocr"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="document_", dir=temp_root) as images:
        with error_context("postprocess"):
            # Official whole-file paragraph processing once, not once per page.
            processed = result.pipe_ocr_mode(imageWriter=FileBasedDataWriter(images),
                                             MonkeyOCR_model=runtime["monkey_model"])
            document = processed.get_markdown("images")
            middle = json.loads(processed.get_middle_json())
        pages = middle["pdf_info"]
        if [page["page_idx"] for page in pages] != list(range(expected_pages)):
            raise ValueError("Monkey completed document page identities do not match the input")
        markdowns, boxes, sizes = [], [], []
        for index, page in enumerate(pages, 1):
            with error_context("page_export", index):
                page_middle = {**middle, "pdf_info": [page]}
                # Locked SDK's result dataset reference; the view only formats
                # completed paragraphs and never invokes OCR or para_split again.
                view = PipeResultLLM(page_middle, result._dataset)
                markdown = view.get_markdown("images")
                size = page["page_size"]
                if len(size) != 2 or any(not isinstance(v, (int, float)) or v <= 0 for v in size):
                    raise ValueError("Monkey middle JSON page_size must be positive")
                markdowns.append(markdown)
                sizes.append(size)
                boxes.extend(extract_monkey_image_boxes_for_page(
                    page_index=index, page_markdown=markdown, document_markdown=document,
                    page_size=size, page_content_list=view.get_content_list("images"),
                    page_middle_json=page_middle))
    save_markdown_bundle(output, markdowns, document)
    records = build_page_records(markdowns, ["N/A"] * len(markdowns))
    for record, size in zip(records, sizes):
        record["page_size"] = size
    write_json(output / "metrics.json", {
        "method": args.method_display_name, "method_id": args.method_id,
        "execution_profile": args.method_id + "_vllm_async_multifile",
        "backend": "vllm_async", "region_concurrency": args.region_concurrency,
        "region_image_normalization": REGION_IMAGE_NORMALIZATION,
        "input_mode": "official_multi_file_group", "pdf_path": str(pdf), "output_dir": str(output),
        "group_page_limit": GROUP_PAGE_LIMIT, "group_page_count": group_page_count,
        "group_inputs": [str(p) for p in group_paths], "pred_abandon": args.pred_abandon,
        "split_files": True, "effective_split_pages": True,
        "config_path": str(args.config_path), "runtime_models_dir": str(args.models_dir),
        "page_count": len(markdowns), "render_seconds": "N/A",
        "page_inference_seconds_sum": "N/A", "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md", "pages": records,
        "notes": [
            "One resident async model analyzes bounded MultiFileDataset groups; P is shared within each group.",
            "20-page grouping follows the official README example, not an SDK default; oversized PDFs stay intact.",
            "Official split_files also enables split_pages and whole-page recognition when layout has no valid regions.",
            "Whole-file pipe_ocr_mode runs once; page views format its completed paragraphs without rerunning OCR.",
            "Document latency spans its group's analysis; no estimated per-page latency is invented.",
            "Temporary native images are cleaned; only text and image coordinates are exported.",
            "After the original max_size=1600 resize, regions below 28px use official HF vision normalization; original boxes are unchanged.",
        ]})
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=args.method_display_name, pdf_path=pdf, entries=boxes,
        notes=["Boxes use completed middle JSON page_size, in the same coordinate system as its bboxes.",
               "No output image assets are retained."]))
    remove_file_if_exists(output / "error.json")
    print(f"[done] {pdf}", flush=True)


def write_pdf_error(args, pdf_path, exc=None, *, group=None):
    exc = sys.exc_info()[1] if exc is None else exc
    extra = {"backend": "vllm_async", "region_concurrency": args.region_concurrency,
             "region_image_normalization": REGION_IMAGE_NORMALIZATION,
             "method_id": args.method_id, "execution_profile": args.method_id + "_vllm_async_multifile"}
    if group is not None:
        extra.update(stage="group_inference", page_number=None, page_numbers=[],
                     affected_inputs=[str(pdf) for pdf in group])
    write_json(get_doc_output_dir(args.input_root, args.output_root, pdf_path) / "error.json",
               build_error_payload(args.method_display_name, pdf_path, exception=exc, extra=extra))
    print(f"[error] {pdf_path}: {exc}", flush=True)


def process_pdf_paths(args, pdf_paths, *, runtime):
    from execution.trial import get_pdf_page_count

    had_failure = False
    pending, counts = [], []
    if len(set(pdf_paths)) != len(pdf_paths):
        raise ValueError("Duplicate Monkey input PDF")

    def process_group(paths, page_counts):
        failed = False
        started = time.perf_counter()
        try:
            results = analyze_group(runtime, paths, page_counts, args.pred_abandon)
        except Exception as exc:
            for pdf in paths:
                write_pdf_error(args, pdf, exc, group=paths)
            if args.fail_fast or runtime_unusable(exc):
                raise
            return True
        for pdf, result, count in zip(paths, results, page_counts):
            try:
                export_document(args, runtime, pdf, result, count, started, paths, sum(page_counts))
            except Exception as exc:
                failed = True
                write_pdf_error(args, pdf, exc)
                if args.fail_fast or runtime_unusable(exc):
                    raise
        return failed

    for pdf in pdf_paths:
        output = get_doc_output_dir(args.input_root, args.output_root, pdf)
        if args.skip_existing and document_completed(output):
            metrics = json.loads((output / "metrics.json").read_text())
            if metrics["execution_profile"] != args.method_id + "_vllm_async_multifile":
                raise ValueError(f"Cannot reuse another Monkey execution profile: {output}")
            if ("region_image_normalization" not in metrics
                    or metrics["region_image_normalization"] != REGION_IMAGE_NORMALIZATION):
                raise ValueError(f"Cannot reuse another Monkey image normalization policy: {output}")
            continue
        try:
            with error_context("inspect_pdf"):
                count = get_pdf_page_count(pdf)
                if count < 1:
                    raise ValueError("Monkey requires a nonempty input PDF")
        except Exception as exc:
            had_failure = True
            write_pdf_error(args, pdf, exc)
            if args.fail_fast or runtime_unusable(exc):
                raise
            continue
        if pending and sum(counts) + count > GROUP_PAGE_LIMIT:
            had_failure = process_group(pending, counts) or had_failure
            pending, counts = [], []
        pending.append(pdf)
        counts.append(count)
    if pending:
        had_failure = process_group(pending, counts) or had_failure
    return had_failure


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.config_path = args.config_path.resolve()
    if args.models_dir is not None:
        args.models_dir = args.models_dir.resolve()

    runtime_config_path = prepare_runtime_config(args)
    runtime = None
    try:
        if args.pdf_list_file is not None:
            args.pdf_list_file = args.pdf_list_file.resolve()
            pdf_paths = load_pdf_paths_from_file(args.pdf_list_file, limit=args.limit)
        else:
            pdf_paths = collect_pdf_paths(args.input_root, limit=args.limit)
        pdf_paths = apply_shard_to_pdf_paths(
            pdf_paths,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        runtime = load_monkeyocr_runtime(runtime_config_path)
        had_failure = process_pdf_paths(args, pdf_paths, runtime=runtime)
        if had_failure:
            raise SystemExit(1)
    finally:
        try:
            close_monkeyocr_runtime(runtime)
        finally:
            runtime_config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
