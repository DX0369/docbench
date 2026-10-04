from __future__ import annotations

import argparse
from pathlib import Path

from execution.commands import conda_python
from methods.mineru.version_profile import MinerUVersionProfile
from methods.mineru.versions.v303_2509 import PROFILE as PROFILE_303_2509
from methods.mineru.versions.v344_2605 import PROFILE as PROFILE_344_2605
from paths import SRC_ROOT


def append_parse_options(
    cmd: list[str],
    profile: MinerUVersionProfile,
    args: argparse.Namespace,
) -> list[str]:
    cmd.extend(
        [
            "--backend",
            profile.api_backend,
            "--method",
            args.mineru_parse_method,
        ]
    )
    if args.mineru_lang:
        cmd.extend(["--lang", args.mineru_lang])
    if args.mineru_disable_formula:
        cmd.append("--disable-formula")
    if args.mineru_disable_table:
        cmd.append("--disable-table")
    return cmd


def build_profile_worker(
    profile: MinerUVersionProfile,
    args: argparse.Namespace,
    gpu_id: int,
    queue_dir: Path,
    worker_id: str,
) -> list[str]:
    cmd = conda_python(profile.conda_env) + [
        str(SRC_ROOT / "methods" / "mineru" / "api_worker.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / profile.output_name),
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
        "--server-url",
        f"http://127.0.0.1:{args.service_port}",
        "--task-timeout-seconds",
        str(args.mineru_task_timeout_seconds),
        "--task-poll-seconds",
        str(args.mineru_task_poll_seconds),
    ]
    if args.mineru_http_timeout_seconds is not None:
        cmd.extend(["--http-timeout-seconds", str(args.mineru_http_timeout_seconds)])
    append_parse_options(cmd, profile, args)
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_profile_service(
    profile: MinerUVersionProfile,
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    cmd = conda_python(profile.conda_env)
    # A persistent service is terminated after clients finish. Conda's default
    # capture would otherwise discard its buffered startup/inference logs.
    cmd.insert(-1, "--no-capture-output")
    cmd += [
        str(SRC_ROOT / "methods" / "mineru" / "api_service.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-output-root",
        str(Path(args.output_root_base) / f"_{profile.output_name}_api_service_output"),
    ]
    cmd.extend(["--preload-models", "--method-id", profile.method_id])
    append_parse_options(cmd, profile, args)
    optional_args = [
        ("--gpu-memory-utilization", args.mineru_vllm_gpu_memory_utilization),
        ("--api-max-concurrent-requests", args.mineru_api_max_concurrent_requests),
        ("--api-processing-window-size", args.mineru_api_processing_window_size),
        ("--api-task-retention-seconds", args.mineru_api_task_retention_seconds),
        ("--hybrid-batch-ratio", args.mineru_hybrid_batch_ratio),
        ("--pdf-render-threads", args.mineru_pdf_render_threads),
        ("--intra-op-num-threads", args.mineru_intra_op_num_threads),
        ("--inter-op-num-threads", args.mineru_inter_op_num_threads),
    ]
    for flag, value in optional_args:
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def build_303_2509_worker(
    args: argparse.Namespace, gpu_id: int, queue_dir: Path, worker_id: str
) -> list[str]:
    return build_profile_worker(PROFILE_303_2509, args, gpu_id, queue_dir, worker_id)


def build_303_2509_service(
    args: argparse.Namespace, gpu_id: int, port: int
) -> list[str]:
    return build_profile_service(PROFILE_303_2509, args, gpu_id, port)


def build_344_2605_worker(
    args: argparse.Namespace, gpu_id: int, queue_dir: Path, worker_id: str
) -> list[str]:
    return build_profile_worker(PROFILE_344_2605, args, gpu_id, queue_dir, worker_id)


def build_344_2605_service(
    args: argparse.Namespace, gpu_id: int, port: int
) -> list[str]:
    return build_profile_service(PROFILE_344_2605, args, gpu_id, port)


def healthcheck_url(port: int) -> str:
    # Uvicorn accepts HTTP only after our wrapped lifespan finishes preloading.
    return f"http://127.0.0.1:{port}/health"
