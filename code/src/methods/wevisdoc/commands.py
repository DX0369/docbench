from __future__ import annotations

from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


WEVISDOC = get_method_version("wevisdoc_2b")


def _model_path(args) -> Path:
    return (
        Path(args.wevisdoc_model_path).expanduser().resolve()
        if args.wevisdoc_model_path
        else WEVISDOC.model_path
    )


def build_worker(args, gpu_id, queue_dir, worker_id):
    return conda_python(args.wevisdoc_env) + [
        str(SRC_ROOT / "methods" / "wevisdoc" / "vllm.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / WEVISDOC.method_id),
        "--queue-dir", str(queue_dir),
        "--worker-id", str(worker_id),
        "--server-url", f"http://127.0.0.1:{args.service_port}/v1",
        "--model-name", "wevisdoc",
        "--dpi", str(args.wevisdoc_dpi),
        "--page-concurrency", str(args.wevisdoc_page_concurrency),
        "--timeout", "600",
        *( ["--skip-existing"] if args.skip_existing else []),
        *( ["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]


def build_service(args, gpu_id: int, port: int) -> list[str]:
    return [
        env_vllm(args.wevisdoc_env),
        "serve",
        str(_model_path(args)),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", "wevisdoc",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "8",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--mm-encoder-tp-mode", "data",
        # gdb_glmocr's Torch 2.11/vLLM 0.20 stack cannot use the default
        # AOT path on this RTX 3090; eager mode is the existing local
        # compatibility route and does not alter request semantics.
        "--enforce-eager",
        "--async-scheduling",
    ]


def healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/models"
