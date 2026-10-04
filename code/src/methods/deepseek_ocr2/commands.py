from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python
from paths import SRC_ROOT


DEEPSEEK_OCR2 = get_method_version("deepseek_ocr2")


def build_runner(
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path,
) -> list[str]:
    assert port is not None
    return conda_python("gdb_deepseek_ocr2") + [
        str(SRC_ROOT / "methods" / "deepseek_ocr2" / "vllm.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / "deepseek_ocr2"),
        "--server-url",
        f"http://127.0.0.1:{port}/v1",
        "--model-name",
        "deepseek-ocr2",
        "--max-tokens",
        str(args.deepseek_max_tokens),
        "--dpi",
        "200",
        "--page-concurrency",
        str(args.deepseek_page_concurrency),
        "--pdf-list-file",
        str(pdf_list_file),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]


def build_worker(args, gpu_id, queue_dir, worker_id):
    cmd = build_runner(args, gpu_id, args.service_port, queue_dir)
    index = cmd.index("--pdf-list-file")
    del cmd[index:index + 2]
    cmd.extend(["--queue-dir", str(queue_dir), "--worker-id", worker_id])
    if getattr(args, "allow_partial_success", False):
        cmd.append("--allow-partial-success")
    return cmd


def build_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    return conda_python(args.deepseek_service_env) + [
        str(SRC_ROOT / "methods" / "deepseek_ocr2" / "api_service.py"),
        "--model-path",
        str(DEEPSEEK_OCR2.model_path),
        "--served-model-name",
        "deepseek-ocr2",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--max-model-len",
        str(args.deepseek_max_model_len),
        "--gpu-memory-utilization",
        str(args.deepseek_gpu_memory_utilization),
        "--preprocess-workers",
        str(args.deepseek_preprocess_workers),
        "--engine-iteration-timeout-seconds",
        str(args.deepseek_engine_iteration_timeout_seconds),
    ]
