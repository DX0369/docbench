#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path  # noqa: E402

GLM_REPO_ROOT = method_repo_path("glmocr")
if str(GLM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GLM_REPO_ROOT))

from common import build_error_payload, document_completed, get_doc_output_dir, write_json, runtime_unusable  # noqa: E402
from execution.workers import run_pdf_queue_worker  # noqa: E402
from methods.glmocr.pipeline import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    prepare_runtime_config,
    export_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent GLM-OCR SDK worker that consumes PDF jobs from a queue."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ocr-api-host", default="127.0.0.1")
    parser.add_argument("--ocr-api-port", type=int, required=True)
    parser.add_argument("--ocr-api-model-name", default="glm-ocr")
    parser.add_argument("--layout-device", type=str)
    parser.add_argument("--layout-model-dir", type=str)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--ocr-connection-pool-size", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-partial-success", action="store_true")
    return parser.parse_args()


def write_pdf_error(args: argparse.Namespace, pdf_path: Path, runtime_config_path: Path, exc: BaseException, *, batch=None) -> None:
    doc_output_dir = get_doc_output_dir(args.input_root, args.output_root, pdf_path)
    batch_fields = {}
    if batch is not None:
        # SDK page numbers span the entire batch, not this PDF. Never label
        # an arbitrary document with another document's page number.
        batch_fields = {"stage": "batch_inference", "page_number": None, "page_numbers": [],
                        "affected_inputs": [str(path) for path in batch]}
    write_json(
        doc_output_dir / "error.json",
        build_error_payload(
            method_name="GLM-OCR",
            exception=exc,
            pdf_path=pdf_path,
            extra={
                "method_id": "glmocr",
                "execution_profile": "glmocr_multi_pdf_regions",
                "backend": "vllm",
                "config_path": str(args.config_path),
                "runtime_config_path": str(runtime_config_path),
                "ocr_api_host": args.ocr_api_host,
                "ocr_api_port": args.ocr_api_port,
                "ocr_api_model_name": args.ocr_api_model_name,
                "layout_device": args.layout_device,
                "layout_model_dir": args.layout_model_dir,
                "error_type": type(exc).__name__,
                "error_summary": str(exc),
                **batch_fields,
            },
        ),
    )


def process_pdf_paths(
    *,
    args: argparse.Namespace,
    parser: Any,
    runtime_config_path: Path,
    pdf_paths: list[Path],
) -> tuple[bool, dict[str, object]]:
    paths = {str(path.resolve()): path for path in pdf_paths}
    if len(paths) != len(pdf_paths):
        raise ValueError("Duplicate input PDF in GLM batch")
    for key, path in list(paths.items()):
        output = get_doc_output_dir(args.input_root, args.output_root, path)
        if args.skip_existing and document_completed(output):
            if json.loads((output / "metrics.json").read_text())["execution_profile"] != "glmocr_multi_pdf_regions":
                raise ValueError(f"Cannot reuse a different GLM execution profile: {output}")
            del paths[key]
    failed_docs, finished = [], set()

    def record_failure(pdf, exc, *, batch=None):
        write_pdf_error(args, pdf, runtime_config_path, exc, batch=batch)
        failed_docs.append({"pdf_path": str(pdf), "reason": type(exc).__name__,
                            "error_json": str(get_doc_output_dir(args.input_root, args.output_root, pdf) / "error.json")})
        print(f"[error] {pdf}: {type(exc).__name__}: {exc}", flush=True)

    batch_error = None
    started = time.perf_counter()
    try:
        results = parser.parse(list(paths), stream=True, preserve_order=True,
                               save_layout_visualization=False) if paths else ()
        for result in results:
            originals = result.original_images
            if not isinstance(originals, list) or len(originals) != 1:
                raise ValueError("GLM document result must identify exactly one original input")
            key = str(Path(originals[0]).resolve())
            if key not in paths:
                raise ValueError(f"GLM returned an unknown original input: {key}")
            if key in finished:
                exc = ValueError(f"GLM returned a duplicate document: {key}")
                record_failure(paths[key], exc)
                raise exc
            try:
                export_document(args, result, paths[key], started)
            except Exception as exc:
                record_failure(paths[key], exc)
                if args.fail_fast or runtime_unusable(exc):
                    raise
            finally:
                finished.add(key)
        if len(finished) != len(paths):
            raise RuntimeError("GLM batch ended before every input produced a document result")
    except Exception as exc:
        batch_error = {"error_type": type(exc).__name__, "error_summary": str(exc)}
        remaining = [paths[key] for key in paths if key not in finished]
        for pdf in remaining:
            record_failure(pdf, exc, batch=remaining)
        if args.fail_fast or runtime_unusable(exc):
            raise
    return bool(failed_docs) or batch_error is not None, {
        "failed_doc_count": len(failed_docs),
        "failed_docs": failed_docs,
        **({"batch_error": batch_error} if batch_error is not None else {}),
    }


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()
    args.config_path = args.config_path.resolve()

    from glmocr import GlmOcr  # noqa: E402

    runtime_config_path = prepare_runtime_config(args)
    try:
        with GlmOcr(
            config_path=str(runtime_config_path),
            mode="selfhosted",
            ocr_api_host=args.ocr_api_host,
            ocr_api_port=args.ocr_api_port,
            layout_device=args.layout_device,
        ) as parser:

            def process_job(_claimed_path: Path, pdf_paths: list[Path]) -> tuple[bool, dict[str, object]]:
                return process_pdf_paths(
                    args=args,
                    parser=parser,
                    runtime_config_path=runtime_config_path,
                    pdf_paths=pdf_paths,
                )

            should_exit_failed = run_pdf_queue_worker(
                queue_dir=args.queue_dir,
                worker_id=args.worker_id,
                input_root=args.input_root,
                output_root=args.output_root,
                fail_fast=args.fail_fast,
                process_job=process_job,
                allow_partial_success=args.allow_partial_success,
            )
    finally:
        runtime_config_path.unlink(missing_ok=True)
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
