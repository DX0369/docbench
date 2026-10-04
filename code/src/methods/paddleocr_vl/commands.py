from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import MethodVersion, get_method_version
from execution.commands import conda_python
from paths import SRC_ROOT


PADDLEOCR_VL_15 = get_method_version("paddleocr_vl_15")
PADDLEOCR_VL_16 = get_method_version("paddleocr_vl_16")


def build_profile_runner(
    version: MethodVersion,
    pipeline_version: str,
    args: argparse.Namespace,
    gpu_id: int,
    port: int | None,
    pdf_list_file: Path | None,
) -> list[str]:
    assert port is not None
    cmd = conda_python("gdb_paddle") + [
        str(SRC_ROOT / "methods" / "paddleocr_vl" / "pipeline.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / version.method_id),
        "--method-id",
        version.method_id,
        "--method-display-name",
        version.display_name,
        "--pipeline-version",
        pipeline_version,
        "--server-url",
        f"http://127.0.0.1:{port}/v1",
        "--api-model-name",
        version.official_checkpoint,
        "--render-dpi",
        "200",
    ]
    if pdf_list_file is not None:
        cmd.extend(["--pdf-list-file", str(pdf_list_file)])
    if args.paddle_vl_rec_max_concurrency is not None:
        cmd.extend(["--vl-rec-max-concurrency", str(args.paddle_vl_rec_max_concurrency)])
    if args.paddle_use_queues is not None:
        cmd.append("--use-queues" if args.paddle_use_queues else "--no-use-queues")
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_profile_service(
    version: MethodVersion,
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    cmd = conda_python(args.paddle_service_env)
    # A resident service is terminated by the launcher. Stream its diagnostics
    # directly so conda does not discard buffered startup/compilation logs.
    cmd.insert(2, "--no-capture-output")
    cmd += [
        str(SRC_ROOT / "methods" / "paddleocr_vl" / "serve.py"),
        "--model",
        str(version.model_path),
        "--served-model-name",
        version.official_checkpoint,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.paddle_vllm_gpu_memory_utilization),
    ]
    if args.paddle_vllm_max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.paddle_vllm_max_model_len)])
    if args.paddle_vllm_enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def build_15_runner(
    args: argparse.Namespace, gpu_id: int, port: int | None, pdf_list_file: Path
) -> list[str]:
    return build_profile_runner(PADDLEOCR_VL_15, "v1.5", args, gpu_id, port, pdf_list_file)


def build_15_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    return build_profile_service(PADDLEOCR_VL_15, args, gpu_id, port)


def build_15_worker(args, gpu_id, queue_dir, worker_id):
    return build_profile_runner(PADDLEOCR_VL_15, "v1.5", args, gpu_id, args.service_port, None) + [
        "--queue-dir", str(queue_dir), "--worker-id", worker_id]


def build_16_runner(
    args: argparse.Namespace, gpu_id: int, port: int | None, pdf_list_file: Path
) -> list[str]:
    return build_profile_runner(PADDLEOCR_VL_16, "v1.6", args, gpu_id, port, pdf_list_file)


def build_16_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    return build_profile_service(PADDLEOCR_VL_16, args, gpu_id, port)


def build_16_worker(args, gpu_id, queue_dir, worker_id):
    return build_profile_runner(PADDLEOCR_VL_16, "v1.6", args, gpu_id, args.service_port, None) + [
        "--queue-dir", str(queue_dir), "--worker-id", worker_id]
