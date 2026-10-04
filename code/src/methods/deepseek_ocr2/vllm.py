#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import httpx
import json
import multiprocessing
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from common import (
    add_shard_args,
    append_error_payload,
    apply_shard_to_pdf_paths,
    build_error_payload,
    document_completed,
    error_context,
    build_image_boxes_payload,
    build_page_records,
    collect_pdf_paths,
    concat_document_markdown,
    get_doc_output_dir,
    load_pdf_paths_from_file,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)
from config.execution_profiles import resolve_execution_profile
from execution.page_pool import run_page_pool as run_shared_page_pool
from execution.workers import run_pdf_queue_worker
from methods.deepseek_ocr2 import render


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
REF_DET_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 through a vLLM OpenAI-compatible server.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-id")
    parser.add_argument("--allow-partial-success", action="store_true")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model-name", type=str, default="deepseek-ocr2")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--limit", type=int)
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--page-concurrency",
        type=int,
        default=1,
        help="Maximum in-flight page requests across PDFs.",
    )
    args = parser.parse_args()
    if bool(args.queue_dir) != bool(args.worker_id):
        parser.error("--queue-dir and --worker-id must be provided together")
    if args.page_concurrency < 1:
        parser.error("--page-concurrency must be positive")
    return args


def build_prompt(prompt: str) -> str:
    normalized = str(prompt or "").strip()
    if not normalized:
        return DEFAULT_PROMPT
    if normalized.startswith("<image>"):
        return normalized
    return "<image>\n" + normalized


def build_chat_completions_url(server_url: str) -> str:
    normalized = str(server_url or "").rstrip("/")
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def normalize_model_output(text: str) -> str:
    content = str(text or "")
    # DeepSeek-OCR2 may emit this token repeatedly if no EOS is hit.
    content = content.replace("<｜end▁of▁sentence｜>", "")
    return content.strip()


def is_image_like_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return normalized in {
        "image",
        "figure",
        "chart",
        "picture",
        "photo",
        "diagram",
        "graph",
    }


def _parse_float_box(values: Sequence[Any]) -> List[float] | None:
    if len(values) != 4:
        return None
    parsed: List[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            return None
    return parsed


def _iter_det_boxes(node: Any) -> Iterable[List[float]]:
    if isinstance(node, (list, tuple)):
        maybe_box = _parse_float_box(node)
        if maybe_box is not None:
            yield maybe_box
            return
        for item in node:
            yield from _iter_det_boxes(item)


def parse_ref_det_entries(raw_page_text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for match in REF_DET_PATTERN.finditer(raw_page_text):
        label = str(match.group(1) or "").strip()
        det_raw = str(match.group(2) or "").strip()
        try:
            det_obj = ast.literal_eval(det_raw)
        except (SyntaxError, ValueError):
            continue
        boxes = list(_iter_det_boxes(det_obj))
        if not boxes:
            continue
        entries.append(
            {
                "label": label,
                "boxes": boxes,
                "raw": match.group(0),
            }
        )
    return entries


def clean_page_markdown(raw_page_text: str) -> str:
    content = normalize_model_output(raw_page_text)
    content = REF_DET_PATTERN.sub("", content)
    content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:").replace("\r\n", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if not content:
        return ""
    return content + "\n"


def _clip_0_999(value: float) -> float:
    return max(0.0, min(999.0, value))


def box_999_to_1000(box: Sequence[float]) -> List[int]:
    x1, y1, x2, y2 = [_clip_0_999(float(value)) for value in box]
    return [
        int(round(x1 * 1000.0 / 999.0)),
        int(round(y1 * 1000.0 / 999.0)),
        int(round(x2 * 1000.0 / 999.0)),
        int(round(y2 * 1000.0 / 999.0)),
    ]


def box_999_to_absolute(box: Sequence[float], page_width: int, page_height: int) -> List[float]:
    x1, y1, x2, y2 = [_clip_0_999(float(value)) for value in box]
    return [
        x1 * float(page_width) / 999.0,
        y1 * float(page_height) / 999.0,
        x2 * float(page_width) / 999.0,
        y2 * float(page_height) / 999.0,
    ]


def extract_image_boxes(
    *,
    pdf_path: Path,
    raw_page_outputs: Sequence[str],
    page_sizes: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for page_index, (raw_page_text, page_size) in enumerate(
        zip(raw_page_outputs, page_sizes),
        start=1,
    ):
        page_width, page_height = int(page_size[0]), int(page_size[1])
        for ref_entry in parse_ref_det_entries(raw_page_text):
            label = str(ref_entry["label"])
            if not is_image_like_label(label):
                continue
            for box in ref_entry["boxes"]:
                entries.append(
                    {
                        "page_index": page_index,
                        "marker_name": None,
                        "marker_type": "deepseek_ref_det",
                        "page_marker_expression": ref_entry["raw"],
                        "page_marker_src": None,
                        "document_marker_expressions": [ref_entry["raw"]],
                        "document_marker_sources": [],
                        "absolute_box": box_999_to_absolute(
                            box,
                            page_width=page_width,
                            page_height=page_height,
                        ),
                        "relative_box_0_1000": box_999_to_1000(box),
                        "page_size": [page_width, page_height],
                        "box_source": "deepseek_ref_det",
                        "block_label": label,
                        "block_content_preview": ref_entry["raw"][:200],
                    }
                )

    return build_image_boxes_payload(
        method="DeepSeek-OCR-2",
        pdf_path=pdf_path,
        entries=entries,
        notes=[
            "Image boxes are extracted from DeepSeek-OCR-2 <|ref|>/<|det|> spans when available.",
            "<|det|> coordinates are interpreted in the model's 0-999 page-relative scale and converted to both absolute pixels and 0-1000 normalized coordinates.",
            "If the model output does not contain parseable <|ref|>/<|det|> entries, image_boxes.json may be empty.",
        ],
    )


def request_single_page(
    *, args, http_client, pdf_path, page_index, page_size, image_png, prompt,
):
    page_start = time.perf_counter()
    response = http_client.post(
        build_chat_completions_url(args.server_url),
        json={
            "model": args.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url":
                    "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "seed": args.seed,
        },
    )
    if response.status_code == 503:
        exc = httpx.HTTPStatusError("DeepSeek service unavailable", request=response.request, response=response)
        # Only our explicit engine-dead signal is fatal. Preserve the HTTP
        # error for generic 503 responses (including non-JSON reverse proxies).
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = None
        if isinstance(error_payload, dict):
            detail = error_payload.get("detail")
            if isinstance(detail, dict) and detail.get("error") in {"engine_dead", "preprocess_pool_broken"}:
                exc.gdb_runtime_unusable = True
        raise exc
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    finish_reason = choice["finish_reason"]
    raw_output = choice["message"]["content"]
    if not isinstance(raw_output, str):
        raise ValueError("DeepSeek page response content must be a string")
    if finish_reason not in {"stop", "length"}:
        raise ValueError(f"Unconfirmed DeepSeek page: finish_reason={finish_reason!r}")
    return {
        "pdf_path": str(pdf_path), "page_index": page_index, "raw_output": raw_output,
        "finish_reason": finish_reason,
        "page_seconds": round(time.perf_counter() - page_start, 6),
        "page_size": list(page_size),
    }


@error_context("export")
def write_pdf_outputs(
    *,
    args: argparse.Namespace,
    pdf_path: Path,
    prompt: str,
    page_results: Sequence[Dict[str, Any]],
    doc_started_at: float,
) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    page_markdowns: List[str] = []
    page_latencies: List[float] = []
    raw_page_outputs: List[str] = []
    page_sizes: List[List[int]] = []

    for page_result in sorted(page_results, key=lambda item: item["page_index"]):
        raw_output = page_result["raw_output"]
        page_seconds = page_result["page_seconds"]
        page_size = page_result["page_size"]
        page_sizes.append([int(page_size[0]), int(page_size[1])])
        raw_page_outputs.append(raw_output)
        page_markdowns.append(clean_page_markdown(raw_output))
        page_latencies.append(page_seconds)

    document_markdown = concat_document_markdown(page_markdowns)
    save_markdown_bundle(
        doc_output_dir=doc_output_dir,
        page_markdowns=page_markdowns,
        document_markdown=document_markdown,
    )

    metrics: Dict[str, Any] = {
        "method": "DeepSeek-OCR-2",
        "execution_profile": resolve_execution_profile("deepseek_ocr2").profile_id,
        "backend": "deepseek_chat_api",
        "input_mode": "page_pool",
        "pdf_path": str(pdf_path),
        "output_dir": str(doc_output_dir),
        "server_url": args.server_url,
        "model_name": args.model_name,
        "prompt": prompt,
        "dpi": args.dpi,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "scheduler": "page_pool",
        "page_concurrency": args.page_concurrency,
        "page_count": len(page_markdowns),
        "render_seconds": None,
        "preprocess_seconds": None,
        "generation_wall_seconds": None,
        "page_inference_seconds_sum": round(sum(page_latencies), 6),
        "service_metadata": {},
        "document_markdown_path": "document.md",
        "pages": build_page_records(page_markdowns, page_latencies),
        "notes": ["One bounded page pool across PDFs; original page order is restored.",
                  "Page render/request sums overlap and are not document or trial wall time."],
    }
    metrics["page_render_seconds_sum"] = sum(page["render_seconds"] for page in page_results)
    metrics["page_finish_reasons"] = [page["finish_reason"] for page in sorted(page_results, key=lambda item: item["page_index"])]
    write_json(
        doc_output_dir / "image_boxes.json",
        extract_image_boxes(
            pdf_path=pdf_path,
            raw_page_outputs=raw_page_outputs,
            page_sizes=page_sizes,
        ),
    )
    metrics["doc_total_seconds"] = round(time.perf_counter() - doc_started_at, 6)
    write_json(doc_output_dir / "metrics.json", metrics)
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")


def should_skip(args, pdf):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not document_completed(output):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Existing partial/failed output must be retained; use a new run: {output}")
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    same = metrics.get("execution_profile") == resolve_execution_profile(
        "deepseek_ocr2").profile_id
    same = same and metrics["prompt"] == build_prompt(args.prompt)
    same = same and all(metrics[key] == getattr(args, key) for key in (
        "model_name", "dpi", "max_tokens", "temperature", "top_p", "seed"))
    same = same and metrics["scheduler"] == "page_pool"
    same = same and metrics["page_concurrency"] == args.page_concurrency
    if not args.skip_existing or not same:
        raise FileExistsError(f"Existing output must not be overwritten or reused across configurations: {output}")
    return True


def record_error(args, pdf, exc):
    append_error_payload(get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
        build_error_payload("DeepSeek-OCR-2", pdf, exception=exc, extra={
            "backend": "deepseek_chat_api",
            "scheduler": "page_pool", "server_url": args.server_url, "model_name": args.model_name,
        }))
    print(f"[error] {pdf}: {exc}", flush=True)


def render_result(renderer, function, *args):
    try:
        return renderer.submit(function, *args).result()
    except BrokenProcessPool as exc:
        exc.gdb_runtime_unusable = True
        raise


def run_page(args, http_client, renderer, pdf, page):
    with error_context("render", page):
        png, size, render_seconds = render_result(renderer, render.render_page, pdf, page, args.dpi)
    with error_context("inference", page):
        result = request_single_page(args=args, http_client=http_client, pdf_path=pdf,
            page_index=page, page_size=size, image_png=png, prompt=build_prompt(args.prompt))
    result["render_seconds"] = render_seconds
    return result


def run_page_pool(args, pdf_paths, http_client, renderer):
    def export_document(pdf, results, started):
        write_pdf_outputs(
            args=args, pdf_path=pdf, prompt=build_prompt(args.prompt), page_results=results,
            doc_started_at=started,
        )
    return run_shared_page_pool(
        pdf_paths, concurrency=args.page_concurrency,
        page_count=lambda pdf: render_result(renderer, render.page_count, pdf),
        should_skip=lambda pdf: should_skip(args, pdf),
        run_page=lambda pdf, page: run_page(args, http_client, renderer, pdf, page),
        export_document=export_document, record_error=lambda pdf, exc: record_error(args, pdf, exc),
        fail_fast=args.fail_fast,
    )


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    with ExitStack() as stack:
        http_client = stack.enter_context(httpx.Client(timeout=args.timeout, trust_env=False))
        renderer = stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")))
        render_result(renderer, render.ready)  # Import/startup only; no PDF warmup.
        if args.queue_dir is not None:
            failed = run_pdf_queue_worker(
                queue_dir=args.queue_dir, worker_id=args.worker_id, fail_fast=args.fail_fast,
                input_root=args.input_root, output_root=args.output_root,
                allow_partial_success=args.allow_partial_success,
                process_job=lambda _path, pdfs: run_page_pool(args, pdfs, http_client, renderer),
            )
        else:
            pdf_paths = (load_pdf_paths_from_file(args.pdf_list_file.resolve(), limit=args.limit)
                         if args.pdf_list_file is not None else collect_pdf_paths(args.input_root, limit=args.limit))
            pdf_paths = apply_shard_to_pdf_paths(pdf_paths, num_shards=args.num_shards, shard_index=args.shard_index)
            failed = run_page_pool(args, pdf_paths, http_client, renderer)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
