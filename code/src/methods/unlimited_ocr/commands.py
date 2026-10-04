from __future__ import annotations

import argparse
import os
from pathlib import Path

from config.method_versions import get_method_version
from paths import CONDA_ENV_ROOT, PROJECT_ROOT, SRC_ROOT


UNLIMITED_OCR_SGLANG = get_method_version("unlimited_ocr_sglang")
SERVED_MODEL_NAME = "Unlimited-OCR"
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
RUNTIME_KEYS = (
    "unlimited_ocr_sglang_env",
    "unlimited_ocr_cuda_home",
    "unlimited_ocr_flashinfer_workspace",
    "unlimited_ocr_mem_fraction_static",
    "unlimited_ocr_context_length",
    "unlimited_ocr_request_timeout_seconds",
    "unlimited_ocr_request_mode",
)


def add_runtime_args(parser: argparse.ArgumentParser, *, include_pressure: bool = True) -> None:
    parser.add_argument("--unlimited-ocr-sglang-env", default="gdb_unlimited_ocr_sglang")
    parser.add_argument("--unlimited-ocr-cuda-home", default="/data/wangsq/software/cuda-envs/cuda-12.8")
    parser.add_argument("--unlimited-ocr-flashinfer-workspace",
                        default=str(PROJECT_ROOT / ".cache" / "unlimited_ocr_sglang"))
    parser.add_argument("--unlimited-ocr-mem-fraction-static", type=float, default=0.9)
    parser.add_argument("--unlimited-ocr-context-length", type=int, default=32768)
    parser.add_argument("--unlimited-ocr-request-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--unlimited-ocr-request-mode", choices=("joint", "page_pool"), default="joint")
    if include_pressure:
        parser.add_argument("--unlimited-ocr-page-concurrency", type=int, default=1)


def runtime_cli_args(args: argparse.Namespace) -> list[str]:
    return [value for key in RUNTIME_KEYS
            for value in ("--" + key.replace("_", "-"), str(getattr(args, key)))]


def _clean_env_prefix() -> list[str]:
    prefix = ["/usr/bin/env"]
    for variable in PROXY_VARIABLES:
        prefix.extend(["-u", variable])
    return prefix


def build_worker(args: argparse.Namespace, gpu_id: int, queue_dir: Path, worker_id: str) -> list[str]:
    env_bin = CONDA_ENV_ROOT / args.unlimited_ocr_sglang_env / "bin"
    return _clean_env_prefix() + [
        str(env_bin / "python"),
        str(SRC_ROOT / "methods" / "unlimited_ocr" / "worker.py"),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(Path(args.output_root_base) / UNLIMITED_OCR_SGLANG.method_id),
        "--server-url",
        f"http://127.0.0.1:{args.service_port}",
        "--model-name",
        SERVED_MODEL_NAME,
        "--dpi",
        "200",
        "--request-timeout-seconds",
        str(args.unlimited_ocr_request_timeout_seconds),
        "--no-repeat-ngram-size",
        "35",
        "--ngram-window",
        "1024",
        "--request-mode", args.unlimited_ocr_request_mode,
        *(["--page-concurrency", str(args.unlimited_ocr_page_concurrency)]
          if args.unlimited_ocr_request_mode == "page_pool" else []),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
        "--queue-dir", str(queue_dir), "--worker-id", worker_id,
    ]


def build_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    env_bin = CONDA_ENV_ROOT / args.unlimited_ocr_sglang_env / "bin"
    cuda_home = Path(args.unlimited_ocr_cuda_home).expanduser().resolve()
    workspace = Path(args.unlimited_ocr_flashinfer_workspace).expanduser().resolve()
    service_path = f"{env_bin}:{cuda_home / 'bin'}"
    existing_path = os.environ.get("PATH")
    if existing_path:
        service_path = f"{service_path}:{existing_path}"
    return _clean_env_prefix() + [
        f"PATH={service_path}",
        f"CUDA_HOME={cuda_home}",
        f"CUDACXX={cuda_home / 'bin' / 'nvcc'}",
        f"FLASHINFER_WORKSPACE_BASE={workspace}",
        f"TVM_FFI_CACHE_DIR={workspace / 'tvm-ffi'}",
        f"TMPDIR={PROJECT_ROOT / 'tmp'}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        str(env_bin / "python"),
        "-m",
        "sglang.launch_server",
        "--model",
        str(UNLIMITED_OCR_SGLANG.model_path),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--attention-backend",
        "fa3",
        "--page-size",
        "1",
        "--mem-fraction-static",
        str(args.unlimited_ocr_mem_fraction_static),
        "--context-length",
        str(args.unlimited_ocr_context_length),
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"
