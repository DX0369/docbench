from __future__ import annotations

import argparse
from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


QIANFAN_OCR = get_method_version("qianfan_ocr")


def _worker(args, queue_dir, worker_id, enable_think):
    assert args.service_port is not None
    method_id = "qianfan_ocr_think" if enable_think else "qianfan_ocr"
    cmd = conda_python("gdb_qianfan") + [
        str(SRC_ROOT / "methods" / "qianfan_ocr" / "vllm.py"),
        "--input-root",
        args.input_root,
        "--output-root",
        str(Path(args.output_root_base) / method_id),
        "--server-url",
        f"http://127.0.0.1:{args.service_port}/v1",
        "--model-name",
        "qianfan-ocr",
        "--max-tokens",
        "16384",
        "--dpi",
        "200",
        "--request-mode",
        args.qianfan_request_mode,
        *(["--page-concurrency", str(args.qianfan_page_concurrency)] if args.qianfan_request_mode == "page_pool" else []),
        "--queue-dir",
        str(queue_dir),
        "--worker-id",
        worker_id,
    ]
    if enable_think:
        cmd.append("--enable-think")
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def build_worker(args, gpu_id, queue_dir, worker_id):
    return _worker(args, queue_dir, worker_id, False)


def build_think_worker(args, gpu_id, queue_dir, worker_id):
    return _worker(args, queue_dir, worker_id, True)


def build_service(
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
) -> list[str]:
    cmd = [
        env_vllm(args.qianfan_service_env),
        "serve",
        str(QIANFAN_OCR.model_path),
        "--served-model-name",
        "qianfan-ocr",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--hf-overrides",
        '{"architectures":["InternVLChatModel"],"model_type":"internvl_chat"}',
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.90",
    ]
    if args.qianfan_execution_mode == "eager":
        cmd.append("--enforce-eager")
    return cmd
