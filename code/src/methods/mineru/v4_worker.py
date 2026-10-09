"""Persistent CPU clients of the official MinerU 4 V1 parsing service."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import (  # noqa: E402
    build_error_payload, build_page_records, document_completed, error_context,
    get_doc_output_dir, runtime_unusable, save_markdown_bundle, write_json,
)
from execution.workers import run_pdf_queue_worker  # noqa: E402
from methods.mineru.v4_output import convert_result  # noqa: E402
from methods.mineru.versions.v4011_standard import METHOD_ID, VERSION, SOURCE_REVISION  # noqa: E402


def process_documents(args, pdf_paths, parser):
    import httpx
    import pypdfium2

    processed, failed = [], []
    for pdf_path in pdf_paths:
        output = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
        if args.skip_existing and document_completed(output):
            processed.append({"pdf_path": str(pdf_path), "skipped_existing": True})
            continue
        stage = "read_input"
        try:
            start = time.perf_counter()
            with error_context(stage):
                with pypdfium2.PdfDocument(pdf_path) as pdf:
                    page_count = len(pdf)
            status = None

            def observe_status(value):
                nonlocal status
                status = value

            stage = "official_api_parse"
            with error_context(stage):
                result = parser.parse(pdf_path, page_range="all", status_callback=observe_status)
                if status != "completed":
                    raise RuntimeError(f"MinerU V1 job ended with {status!r}, expected completed")
            stage = "convert_output"
            with error_context(stage):
                document, pages, boxes = convert_result(result, input_page_count=page_count, pdf_path=pdf_path)
            stage = "write_output"
            with error_context(stage):
                save_markdown_bundle(output, pages, document)
                write_json(output / "image_boxes.json", boxes)
                write_json(output / "metrics.json", {
                    "method": METHOD_ID, "method_id": METHOD_ID, "pdf_path": str(pdf_path),
                    "mineru_version": VERSION, "source_binding": "pypi_wheel",
                    "reference_source_revision": SOURCE_REVISION,
                    "execution_profile": "mineru_4011_standard_v1_api",
                    "tier": "standard", "small_backend": "torch", "vlm_engine": "vllm",
                    "checkpoint": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
                    "small_model": "OpenDataLab/MinerU-4_models_torch",
                    "input_mode": "native_pdf_local_via_official_v1_sdk", "page_range": "all",
                    "ocr_mode": "auto", "page_count": page_count, "input_page_count": page_count,
                    "recognized_page_count": len(result.pages), "request_completed": True,
                    "job_status": status, "document_markdown_path": "document.md",
                    "pages": build_page_records(pages, []),
                    "doc_total_seconds": round(time.perf_counter() - start, 6),
                    "notes": ["Whole-document official rendering preserves cross-page merges; page views are separate projections.",
                              "Official SDK transport retries and polling defaults; no document retry."],
                })
            processed.append({"pdf_path": str(pdf_path), "page_count": page_count})
        except Exception as exc:
            # Document failures are skipped. A dead listener or another
            # recognized unusable runtime stops this client without retrying.
            cause = exc
            seen = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, httpx.ConnectError):
                    exc.gdb_runtime_unusable = True
                cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
            if getattr(exc, "code", None) == "server_shutting_down":
                exc.gdb_runtime_unusable = True
            write_json(output / "error.json", build_error_payload(
                METHOD_ID, pdf_path, exception=exc, stage=stage,
                extra={"official_error_code": getattr(exc, "code", None)},
            ))
            failed.append({"pdf_path": str(pdf_path), "reason": str(exc),
                           "error_json": str(output / "error.json")})
            if runtime_unusable(exc):
                return True, {"processed_docs": processed, "failed_docs": failed, "stop_worker": True}
    return bool(failed), {"processed_docs": processed, "failed_docs": failed}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ("input-root", "output-root", "queue-dir"):
        cli.add_argument("--" + flag, type=Path, required=True)
    cli.add_argument("--worker-id", required=True)
    cli.add_argument("--server-url", required=True)
    cli.add_argument("--skip-existing", action="store_true")
    args = cli.parse_args()
    if version("mineru") != VERSION:
        raise RuntimeError(f"This adapter requires MinerU {VERSION}, got {version('mineru')}")
    from mineru.parser import MinerUApiParser

    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    parser = MinerUApiParser(api_url=args.server_url, api_key="", tier="standard", ocr_mode="auto")
    # Resolve official source capability discovery before the timing barrier.
    # Only a managed same-host service with --allow-local-source is supported.
    if not parser._supports_local_source():
        raise RuntimeError("Managed MinerU 4 service must advertise local PDF sources")
    failed = run_pdf_queue_worker(
        queue_dir=args.queue_dir, worker_id=args.worker_id, fail_fast=False,
        input_root=args.input_root, output_root=args.output_root,
        process_job=lambda _job, paths: process_documents(args, paths, parser),
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
