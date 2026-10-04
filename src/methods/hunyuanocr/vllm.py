#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from common import (  # noqa: E402
    add_shard_args,
    append_error_payload,
    apply_shard_to_pdf_paths,
    build_image_boxes_payload,
    build_error_payload,
    document_completed,
    error_context,
    build_page_records,
    collect_pdf_paths,
    concat_document_markdown,
    get_doc_output_dir,
    load_pdf_paths_from_file,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)
from image_vllm_client import build_image_chat_messages, build_openai_client, chat_with_image  # noqa: E402
from method_paths import method_repo_path  # noqa: E402
from common import pdf_page_count, render_pdf_page  # noqa: E402
from config.execution_profiles import resolve_execution_profile  # noqa: E402
from execution.page_pool import run_page_pool  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from paths import TMP_ROOT  # noqa: E402


DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def postprocess_output(text: str, profile: str) -> tuple[str, Dict[str, Any]]:
    if profile != "hunyuanocr_15":
        return text, {}
    repo_root = method_repo_path("hunyuanocr")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from inference.utils.hunyuan_utils import (  # noqa: E402
        clean_repeated_substrings,
        process_one,
    )

    cleaned = clean_repeated_substrings(text)
    normalized, stats = process_one(cleaned)
    return normalized, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HunyuanOCR through a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", type=str, default="hunyuanocr")
    parser.add_argument("--method-id", default="hunyuanocr_10")
    parser.add_argument("--method-display-name", default="HunyuanOCR-1.0")
    parser.add_argument(
        "--prompt-profile",
        choices=["hunyuanocr_10", "hunyuanocr_15"],
        default="hunyuanocr_10",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--page-concurrency",
        type=int,
        default=max(1, int(os.getenv("HUNYUAN_PAGE_WORKERS", "1"))),
        help="Global bound on page render/request/postprocess tasks across input PDFs.",
    )
    add_shard_args(parser)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--render-cache-root", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-id", default="w00")
    parser.add_argument("--allow-partial-success", action="store_true")
    args = parser.parse_args()
    if args.page_concurrency < 1:
        parser.error("--page-concurrency must be positive")
    return args


def infer_page(args, client, image_path, page_number):
    page_start = time.perf_counter()
    early_stopped = False
    with error_context("inference", page_number):
        generation_kwargs = {
            "temperature": 0.0, "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": 1234,
            "extra_body": {"top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                           "skip_special_tokens": True},
        }
        prompt = DEFAULT_PROMPT
        if args.prompt_profile == "hunyuanocr_15":
            repo_root = method_repo_path("hunyuanocr")
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from inference.utils.hunyuan_utils import infer_stream
            text, early_stopped = infer_stream(client, {
                "model": args.model_name,
                "messages": build_image_chat_messages(image_path, prompt, system_prompt=""),
                **generation_kwargs,
            })
        else:
            text = chat_with_image(client, args.model_name, image_path, prompt,
                                   system_prompt="", **generation_kwargs)
    with error_context("postprocess", page_number):
        text, stats = postprocess_output(text, args.prompt_profile)
    return text, round(time.perf_counter() - page_start, 6), stats, early_stopped


def run_page(args, client, pdf, page):
    started = time.perf_counter()
    root = args.render_cache_root or TMP_ROOT / "runtime" / "hunyuanocr"
    with render_pdf_page(pdf, page, args.dpi, root) as image:
        render_seconds = round(time.perf_counter() - started, 6)
        return (*infer_page(args, client, image, page), render_seconds)


def export_document(args, pdf, results, started):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    pages = [r[0] for r in results]
    latencies = [r[1] for r in results]
    save_markdown_bundle(output, pages, concat_document_markdown(pages))
    metrics = {
        "method": args.method_display_name, "method_id": args.method_id, "backend": "vllm",
        "execution_profile": resolve_execution_profile(args.method_id).profile_id,
        "scheduler": "global_page_pool", "input_mode": "pdf_to_images",
        "pdf_path": str(pdf), "output_dir": str(output), "server_url": args.server_url,
        "model_name": args.model_name, "prompt": DEFAULT_PROMPT,
        "prompt_profile": args.prompt_profile, "max_tokens": args.max_tokens,
        "top_p": args.top_p, "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
        "dpi": args.dpi, "page_concurrency": args.page_concurrency, "page_count": len(pages),
        "page_render_seconds_sum": round(sum(r[4] for r in results), 6),
        "page_inference_seconds_sum": round(sum(latencies), 6),
        "doc_total_seconds": round(time.perf_counter() - started, 6),
        "document_markdown_path": "document.md", "pages": build_page_records(pages, latencies),
        "page_postprocess_stats": [r[2] for r in results],
        "notes": ["Independent page images; document markdown is merged in original page order.",
                  "Page render/inference sums overlap under concurrency and are not document wall time."],
    }
    if args.prompt_profile == "hunyuanocr_15":
        metrics["stream"] = True
        metrics["early_stopped_pages"] = sum(r[3] for r in results)
        for page_record, result in zip(metrics["pages"], results):
            page_record["early_stopped"] = result[3]
    write_json(output / "metrics.json", metrics)
    write_json(output / "image_boxes.json", build_image_boxes_payload(
        method=args.method_display_name, pdf_path=pdf, entries=[],
        notes=["No confirmed official image-coordinate output mode; image_boxes is intentionally empty."],
    ))
    remove_file_if_exists(output / "error.json")


def should_skip(args, pdf):
    output = get_doc_output_dir(args.input_root, args.output_root, pdf)
    if not document_completed(output):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Existing partial/failed output must be retained; use a new run: {output}")
        return False
    import json
    metrics = json.loads((output / "metrics.json").read_text())
    same = metrics.get("execution_profile") == resolve_execution_profile(args.method_id).profile_id
    same = same and all(metrics[key] == getattr(args, key) for key in (
        "method_id", "model_name", "prompt_profile", "dpi", "max_tokens", "top_p", "top_k",
        "repetition_penalty", "page_concurrency"))
    if not args.skip_existing or not same:
        raise FileExistsError(f"Existing result must not be overwritten or reused across configurations: {output}")
    return True


def record_error(args, pdf, exc):
    append_error_payload(get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json",
        build_error_payload(method_name=args.method_display_name, pdf_path=pdf, exception=exc,
            extra={"backend": "vllm", "scheduler": "global_page_pool", "server_url": args.server_url,
                   "model_name": args.model_name, "method_id": args.method_id}))
    print(f"[error] {pdf}: {exc}", flush=True)


def process_pdf_paths(args, client, pdf_paths):
    return run_page_pool(pdf_paths, concurrency=args.page_concurrency,
        page_count=pdf_page_count, should_skip=lambda pdf: should_skip(args, pdf),
        run_page=lambda pdf, page: run_page(args, client, pdf, page),
        export_document=lambda pdf, results, started: export_document(args, pdf, results, started),
        record_error=lambda pdf, exc: record_error(args, pdf, exc), fail_fast=args.fail_fast)


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()

    if args.prompt_profile == "hunyuanocr_15":
        repo_root = method_repo_path("hunyuanocr")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from inference.utils import hunyuan_utils  # noqa: F401

    client = build_openai_client(base_url=args.server_url, timeout=args.timeout)
    if args.queue_dir is not None:
        with client:
            failed = run_pdf_queue_worker(queue_dir=args.queue_dir, worker_id=args.worker_id,
                fail_fast=args.fail_fast, input_root=args.input_root, output_root=args.output_root,
                allow_partial_success=args.allow_partial_success,
                process_job=lambda _path, pdfs: process_pdf_paths(args, client, pdfs))
        if failed:
            raise SystemExit(1)
        return
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

    with client:
        had_failure = process_pdf_paths(args, client, pdf_paths)
    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
