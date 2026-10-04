from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


GLMOCR = get_method_version("glmocr")


def _append_pipeline_options(cmd: list[str], args: argparse.Namespace) -> list[str]:
    if args.glm_layout_model_dir:
        cmd.extend(["--layout-model-dir", args.glm_layout_model_dir])
    if args.glm_max_workers is not None:
        cmd.extend(["--max-workers", str(args.glm_max_workers)])
    if args.glm_ocr_connection_pool_size is not None:
        cmd.extend(["--ocr-connection-pool-size", str(args.glm_ocr_connection_pool_size)])
    return cmd


def build_worker(
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    if getattr(args, "service_port", None) is None:
        raise ValueError("GLM-OCR persistent worker requires a managed service port.")
    cmd = conda_python("gdb_glmocr") + [
        str(SRC_ROOT / "methods" / "glmocr" / "worker.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / "glmocr"),
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--ocr-api-host",
        "127.0.0.1",
        "--ocr-api-port",
        str(args.service_port),
        "--ocr-api-model-name",
        "glm-ocr",
    ]
    _append_pipeline_options(cmd, args)
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    return [
        env_vllm("gdb_glmocr"),
        "serve",
        str(GLMOCR.model_path),
        "--served-model-name",
        "glm-ocr",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--allowed-local-media-path",
        "/",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":3}',
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        str(args.glm_vllm_gpu_memory_utilization),
    ]
