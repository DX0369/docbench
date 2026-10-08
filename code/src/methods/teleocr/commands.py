from __future__ import annotations

from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python
from paths import SRC_ROOT


def build_worker(args, gpu_id, queue_dir, worker_id):
    version = get_method_version("teleocr_vllm_async")
    if type(args.teleocr_max_concurrency) is not int or args.teleocr_max_concurrency < 1:
        raise ValueError("TeleOCR max_concurrency must be a positive integer")
    model = Path(args.teleocr_model_path).expanduser().resolve() if args.teleocr_model_path else version.model_path
    return conda_python(args.teleocr_env) + [
        str(SRC_ROOT / "methods" / "teleocr" / "worker.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / version.method_id),
        "--queue-dir", str(queue_dir), "--worker-id", worker_id,
        "--model-path", str(model),
        "--backend", args.teleocr_backend,
        "--layout-mode", args.teleocr_layout_mode,
        "--max-model-len", str(args.teleocr_max_model_len),
        "--gpu-memory-utilization", str(args.teleocr_gpu_memory_utilization),
        "--pdf-tools", args.teleocr_pdf_tools,
        "--pdf-tools-workers", str(args.teleocr_pdf_tools_workers),
        "--max-concurrency", str(args.teleocr_max_concurrency),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]
