#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Sequence
from urllib.parse import urljoin

import httpx

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path  # noqa: E402
from paths import TMP_ROOT  # noqa: E402

MINERU_REPO_ROOT = method_repo_path("mineru")
if str(MINERU_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MINERU_REPO_ROOT))

from common import (  # noqa: E402
    build_error_payload,
    document_completed,
    error_context,
    runtime_unusable,
    build_page_records,
    concat_document_markdown,
    get_doc_output_dir,
    remove_file_if_exists,
    save_markdown_bundle,
    write_json,
)
from execution.workers import run_pdf_queue_worker  # noqa: E402
from methods.mineru.output import (  # noqa: E402
    build_mineru_content_list,
    build_page_markdowns,
    build_page_size_map,
    extract_mineru_image_boxes,
    infer_mineru_lang,
    mineru_identity,
    resolve_backend_name,
)


class MinerUServiceUnavailable(RuntimeError):
    requeue_claimed_job = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent MinerU API worker. It consumes benchmark queue jobs, submits "
            "each PDF to the official /tasks endpoint, downloads the result ZIP, and "
            "adapts MinerU middle_json into GlobalDocBench's output schema."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Managed MinerU API base URL, for example http://127.0.0.1:9800.",
    )
    parser.add_argument("--backend", choices=("hybrid-auto-engine", "hybrid-engine", "hybrid-vllm-async-engine"), required=True)
    parser.add_argument("--method", type=str, choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--lang", type=str, default="auto")
    parser.add_argument("--disable-formula", action="store_true")
    parser.add_argument("--disable-table", action="store_true")
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=3600.0,
        help="Maximum wall-clock seconds to wait for one MinerU API task.",
    )
    parser.add_argument(
        "--task-poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval for /tasks/{task_id}.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        help=(
            "Explicit read-timeout override for all MinerU HTTP requests. "
            "By default, use the selected official version's per-stage timeouts."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def configure_http_timeouts(args: argparse.Namespace) -> None:
    """Read defaults from the selected checkout, not from a shared guessed limit."""
    from mineru.cli.api_client import build_http_timeout

    method_id = os.environ["GDB_METHOD_ID"]
    args.http_timeout = build_http_timeout()
    if method_id == "mineru_303_2509":
        args.result_http_timeout = build_http_timeout()
    elif method_id == "mineru_344_2605":
        from mineru.cli.api_client import build_result_download_timeout

        args.result_http_timeout = build_result_download_timeout()
    else:
        raise ValueError(f"Unsupported MinerU API method: {method_id}")
    if args.http_timeout_seconds is not None:
        args.http_timeout.read = args.http_timeout_seconds
        args.result_http_timeout.read = args.http_timeout_seconds


def load_mineru_render_runtime() -> Dict[str, Any]:
    from mineru.backend.vlm.vlm_middle_json_mkcontent import (  # noqa: E402
        union_make as vlm_union_make,
    )
    from mineru.utils.engine_utils import get_vlm_engine  # noqa: E402

    return {
        "vlm_union_make": vlm_union_make,
        "get_vlm_engine": get_vlm_engine,
    }


def normalize_base_url(server_url: str) -> str:
    return server_url.rstrip("/")


def request_api(method: str, url: str, **kwargs: Any) -> httpx.Response:
    # Preserve the previous connection-loss signal without treating a truncated
    # response body (or a timeout) as permission to resubmit the document.
    headers_received = False
    try:
        with httpx.stream(method, url, follow_redirects=True, **kwargs) as response:
            headers_received = True
            response.read()
            return response
    except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
        if not headers_received:
            raise MinerUServiceUnavailable(
                f"MinerU API connection failed at {url}; requeueing current job."
            ) from exc
        raise


def post_task(args: argparse.Namespace, pdf_path: Path, requested_lang: str, *,
              return_images: bool = False) -> Dict[str, Any]:
    data = {
        "lang_list": requested_lang,
        "backend": args.backend,
        "parse_method": args.method,
        "formula_enable": str(not args.disable_formula).lower(),
        "table_enable": str(not args.disable_table).lower(),
        "return_md": "true",
        "return_middle_json": "true",
        "return_model_output": "false",
        "return_content_list": "false",
        "return_images": str(return_images).lower(),
        "response_format_zip": "true",
        "return_original_file": "false",
        "start_page_id": "0",
        "end_page_id": "99999",
    }
    mime_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    with pdf_path.open("rb") as handle:
        response = request_api(
            "POST",
            urljoin(normalize_base_url(args.server_url) + "/", "tasks"),
            data=data,
            files={"files": (pdf_path.name, handle, mime_type)},
            timeout=args.http_timeout,
        )
    if response.status_code != 202:
        raise RuntimeError(f"MinerU task submit failed: {response.status_code} {response.text[:1000]}")
    payload = response.json()
    if not isinstance(payload.get("task_id"), str):
        raise RuntimeError(f"MinerU task submit returned invalid payload: {payload}")
    return payload


def wait_for_task(args: argparse.Namespace, task_payload: Dict[str, Any]) -> Dict[str, Any]:
    status_url = str(task_payload["status_url"])
    deadline = time.perf_counter() + float(args.task_timeout_seconds)
    last_payload: Dict[str, Any] = {}
    while time.perf_counter() < deadline:
        try:
            response = request_api("GET", status_url, timeout=args.http_timeout)
        except httpx.ReadTimeout:
            # MinerU can block status responses during cold start or heavy
            # local parsing. The official api_client treats this as retryable.
            time.sleep(float(args.task_poll_seconds))
            continue
        if response.status_code != 200:
            raise RuntimeError(f"MinerU task status failed: {response.status_code} {response.text[:1000]}")
        payload = response.json()
        last_payload = payload
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(f"MinerU task failed: {json.dumps(payload, ensure_ascii=False)[:2000]}")
        time.sleep(float(args.task_poll_seconds))
    raise TimeoutError(
        "Timed out waiting for MinerU task "
        f"{task_payload.get('task_id')} after {args.task_timeout_seconds}s. "
        f"Last status: {json.dumps(last_payload, ensure_ascii=False)[:1000]}"
    )


def safe_extract_zip_bytes(zip_bytes: bytes, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tempfile.NamedTemporaryFile(suffix=".zip", prefix="gdb_mineru_api_", delete=False) as tmp:
        zip_path = Path(tmp.name)
        tmp.write(zip_bytes)
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            for member in zip_file.infolist():
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"Refusing unsafe ZIP entry: {member.filename}")
                target_path = (output_root / Path(*member_path.parts)).resolve()
                if target_path != output_root and output_root not in target_path.parents:
                    raise RuntimeError(f"Refusing unsafe ZIP entry: {member.filename}")
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member, "r") as source, target_path.open("wb") as handle:
                    handle.write(source.read())
    finally:
        zip_path.unlink(missing_ok=True)


def download_and_extract_result(
    args: argparse.Namespace,
    task_payload: Dict[str, Any],
    extract_dir: Path,
) -> None:
    result_url = str(task_payload["result_url"])
    response = request_api("GET", result_url, timeout=args.result_http_timeout)
    if response.status_code != 200:
        raise RuntimeError(f"MinerU result download failed: {response.status_code} {response.text[:1000]}")
    content_type = response.headers.get("content-type", "")
    if "application/zip" not in content_type:
        raise RuntimeError(f"Expected MinerU ZIP result, got content-type={content_type or 'unknown'}")
    safe_extract_zip_bytes(response.content, extract_dir)


def find_middle_json_path(extract_dir: Path, pdf_stem: str) -> Path:
    matches = sorted(extract_dir.rglob(f"{pdf_stem}_middle.json"))
    if matches:
        return matches[0]
    all_middle_json = sorted(extract_dir.rglob("*_middle.json"))
    if len(all_middle_json) == 1:
        return all_middle_json[0]
    raise FileNotFoundError(f"MinerU API result did not include {pdf_stem}_middle.json under {extract_dir}")


def process_api_pdf(
    args: argparse.Namespace,
    *,
    pdf_path: Path,
    render_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    if args.skip_existing and document_completed(doc_output_dir):
        print(f"[skip] {pdf_path}")
        return {"skipped": True}

    requested_lang = infer_mineru_lang(args.input_root, pdf_path, args.lang)
    backend_resolved = resolve_backend_name(
        args.backend, render_runtime["get_vlm_engine"], is_async=True
    )

    doc_total_start = time.perf_counter()
    with error_context("submit"):
        task_payload = post_task(args, pdf_path, requested_lang)
    with error_context("inference"):
        wait_for_task(args, task_payload)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gdb_mineru_api_result_", dir=TMP_ROOT) as temp_dir_str:
        extract_dir = Path(temp_dir_str)
        with error_context("download"):
            download_and_extract_result(args, task_payload, extract_dir)
        middle_json_path = find_middle_json_path(extract_dir, pdf_path.stem)
        middle_json = json.loads(middle_json_path.read_text(encoding="utf-8"))

    doc_total_seconds = round(time.perf_counter() - doc_total_start, 6)
    pdf_info = middle_json.get("pdf_info", [])
    if not isinstance(pdf_info, list):
        raise TypeError("MinerU middle_json['pdf_info'] should be a list.")

    union_make = render_runtime["vlm_union_make"]
    page_markdowns = build_page_markdowns(pdf_info, union_make)
    document_markdown = concat_document_markdown(page_markdowns)
    content_list = build_mineru_content_list(pdf_info, union_make)
    page_sizes = build_page_size_map(pdf_info)
    save_markdown_bundle(
        doc_output_dir=doc_output_dir,
        page_markdowns=page_markdowns,
        document_markdown=document_markdown,
    )

    page_latencies = ["N/A"] * len(page_markdowns)
    metrics: Dict[str, Any] = {
        "method": "MinerU",
        **mineru_identity(),
        "execution_mode": "mineru_api_tasks",
        "backend_requested": args.backend,
        "backend_resolved": backend_resolved,
        "input_mode": "native_pdf_via_official_tasks_api",
        "pdf_path": str(pdf_path),
        "output_dir": str(doc_output_dir),
        "parse_method": args.method,
        "language_hint": requested_lang,
        "api_server_url": args.server_url,
        "parse_server_url": None,
        "http_timeouts": {
            "source": "official_defaults" if args.http_timeout_seconds is None else "project_read_override",
            "request": args.http_timeout.as_dict(),
            "result_download": args.result_http_timeout.as_dict(),
            "task_wait_seconds": args.task_timeout_seconds,
        },
        "formula_enable": not args.disable_formula,
        "table_enable": not args.disable_table,
        "page_count": len(page_markdowns),
        "render_seconds": "N/A",
        "page_inference_seconds_sum": "N/A",
        "doc_total_seconds": doc_total_seconds,
        "document_markdown_path": "document.md",
        "pages": build_page_records(page_markdowns, page_latencies),
        "task": {
            "task_id": task_payload.get("task_id"),
            "queued_ahead": task_payload.get("queued_ahead"),
            "file_names": task_payload.get("file_names"),
        },
        "notes": [
            "MinerU is run through its official persistent mineru-api /tasks interface.",
            "The API service owns parsing concurrency via MINERU_API_MAX_CONCURRENT_REQUESTS.",
            "backend_resolved uses the same pinned environment as the managed hybrid API service.",
            "document.md is produced by converting the official middle_json into page-level markdown with MinerU's renderer.",
            "The official API path does not expose stable per-page latency, so page latencies are recorded as 'N/A'.",
        ],
    }
    write_json(doc_output_dir / "metrics.json", metrics)
    write_json(
        doc_output_dir / "image_boxes.json",
        extract_mineru_image_boxes(
            pdf_path=pdf_path,
            backend_requested=args.backend,
            backend_resolved=backend_resolved,
            page_markdowns=page_markdowns,
            document_markdown=document_markdown,
            content_list=content_list,
            page_sizes=page_sizes,
        ),
    )
    remove_file_if_exists(doc_output_dir / "error.json")
    print(f"[done] {pdf_path}")
    return {
        "task_id": task_payload.get("task_id"),
        "doc_total_seconds": doc_total_seconds,
        "page_count": len(page_markdowns),
    }


def write_pdf_error(args: argparse.Namespace, pdf_path: Path, exc: BaseException | None = None) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    extra: Dict[str, Any] = {
        **mineru_identity(),
        "execution_mode": "mineru_api_tasks",
        "backend_requested": args.backend,
        "parse_method": args.method,
        "language_hint": infer_mineru_lang(args.input_root, pdf_path, args.lang),
        "api_server_url": args.server_url,
        "parse_server_url": None,
        "formula_enable": not args.disable_formula,
        "table_enable": not args.disable_table,
    }
    if exc is not None:
        extra.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    write_json(
        doc_output_dir / "error.json",
        build_error_payload(
            method_name="MinerU",
            pdf_path=pdf_path,
            extra=extra,
        ),
    )
    print(f"[error] {pdf_path}")
    if exc is not None:
        print(f"[error-detail] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()


def process_pdf_paths(
    args: argparse.Namespace,
    pdf_paths: Sequence[Path],
    *,
    render_runtime: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    had_failure = False
    processed: list[dict[str, Any]] = []
    failed_docs: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        try:
            processed.append(process_api_pdf(args, pdf_path=pdf_path, render_runtime=render_runtime))
        except MinerUServiceUnavailable:
            raise
        except Exception as exc:
            had_failure = True
            write_pdf_error(args, pdf_path, exc)
            failed_docs.append(
                {
                    "pdf_path": str(pdf_path),
                    "error_json": str(get_doc_output_dir(args.input_root, args.output_root, pdf_path) / "error.json"),
                    "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )
            if args.fail_fast or runtime_unusable(sys.exc_info()[1]):
                raise
    return had_failure, {
        "processed_docs": processed,
        "failed_doc_count": len(failed_docs),
        "failed_docs": failed_docs,
    }


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    args.server_url = normalize_base_url(args.server_url)
    configure_http_timeouts(args)
    render_runtime = load_mineru_render_runtime()

    # The managed service has preloaded before accepting HTTP. This shared queue
    # barrier reports client readiness after imports, before submitting any PDF;
    # its done timestamp includes result download/export, not process teardown.
    should_exit_failed = run_pdf_queue_worker(
        queue_dir=args.queue_dir,
        worker_id=args.worker_id,
        input_root=args.input_root,
        output_root=args.output_root,
        fail_fast=args.fail_fast,
        process_job=lambda _claimed_path, pdf_paths: process_pdf_paths(args, pdf_paths, render_runtime=render_runtime),
    )
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
