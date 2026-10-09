"""MinerU 4.0.11 wheel runtime; its release checkout is reference source only."""
from __future__ import annotations

import argparse
from pathlib import Path

from execution.commands import conda_python
from paths import CONDA_ENV_ROOT, MODEL_ROOT, SRC_ROOT

METHOD_ID = "mineru_4011_standard"
VERSION = "4.0.11"
CONDA_ENV = "gdb_mineru_4"
SOURCE_REVISION = "f232373fb319d63e82aefd30be40a2ccc6005d73"


def environment_overrides() -> dict[str, str]:
    return {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MINERU_MODEL_BASE_DIR": str(MODEL_ROOT / "mineru"),
        "MINERU_MODEL_SOURCE": "local",
        "MINERU_MODEL_SMALL_BACKEND": "torch",
        "MINERU_MODEL_VLM_ENGINE": "vllm",
        # No remote engine or global proxy is involved in this local profile.
        "MINERU_MODEL_VLM_SERVER_URL": "",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
    }


def build_service(args: argparse.Namespace, gpu_id: int, port: int) -> list[str]:
    cmd = conda_python(CONDA_ENV)
    cmd[-1] = str(CONDA_ENV_ROOT / CONDA_ENV / "bin/mineru-kit")
    cmd.insert(-1, "--no-capture-output")
    return cmd + [
        "api-server",
        "--host", "127.0.0.1", "--port", str(port),
        "--tier", "standard", "--concurrency", str(args.mineru4_concurrency),
        "--preload-models", "--allow-local-source",
        "--upload-dir", str(Path(args.output_root_base) / "_mineru4_service"),
    ]


def build_worker(args: argparse.Namespace, gpu_id: int, queue_dir: Path, worker_id: str) -> list[str]:
    cmd = conda_python(CONDA_ENV) + [
        str(SRC_ROOT / "methods/mineru/v4_worker.py"),
        "--input-root", args.input_root,
        "--output-root", str(Path(args.output_root_base) / METHOD_ID),
        "--queue-dir", str(queue_dir), "--worker-id", worker_id,
        "--server-url", f"http://127.0.0.1:{args.service_port}",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/health"
