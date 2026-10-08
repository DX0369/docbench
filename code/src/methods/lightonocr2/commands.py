from __future__ import annotations

from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


LIGHTONOCR2 = get_method_version("lightonocr2_1b")


def _model_path(args) -> Path:
    return Path(args.lightonocr2_model_path).expanduser().resolve() if args.lightonocr2_model_path else LIGHTONOCR2.model_path


def build_worker(args, gpu_id, queue_dir, worker_id):
    return conda_python(args.lightonocr2_env) + [
        str(SRC_ROOT / "methods" / "lightonocr2" / "vllm.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / LIGHTONOCR2.method_id),
        "--queue-dir", str(queue_dir),
        "--worker-id", str(worker_id),
        "--server-url", f"http://127.0.0.1:{args.service_port}/v1",
        "--model-name", "lightonocr2",
        "--dpi", str(args.lightonocr2_dpi),
        "--max-image-size", str(args.lightonocr2_max_image_size),
        "--max-tokens", str(args.lightonocr2_max_tokens),
        "--temperature", str(args.lightonocr2_temperature),
        "--top-p", str(args.lightonocr2_top_p),
        "--page-concurrency", str(args.lightonocr2_page_concurrency),
        *( ["--skip-existing"] if args.skip_existing else []),
        *( ["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]


def build_service(args, gpu_id: int, port: int) -> list[str]:
    return [
        env_vllm(args.lightonocr2_env),
        "serve",
        str(_model_path(args)),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--served-model-name", "lightonocr2",
        # These are the official LightOnOCR-2 vLLM serving settings.  In
        # particular, one request contains exactly one page image.
        "--limit-mm-per-prompt", '{"image":1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ]


def healthcheck_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1/models"
