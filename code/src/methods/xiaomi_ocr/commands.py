from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python, env_vllm
from paths import SRC_ROOT


XIAOMI = get_method_version("xiaomi_ocr_0")


def build_worker(args, gpu_id, queue_dir, worker_id):
    return conda_python(args.xiaomi_env) + [
        str(SRC_ROOT / "methods" / "xiaomi_ocr" / "vllm.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / XIAOMI.method_id),
        "--queue-dir", str(queue_dir),
        "--worker-id", str(worker_id),
        "--server-url", f"http://127.0.0.1:{args.service_port}/v1",
        "--dpi", str(args.xiaomi_dpi),
        "--max-tokens", str(args.xiaomi_max_tokens),
        "--page-concurrency", str(args.xiaomi_page_concurrency),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]


def build_service(args, gpu_id, port):
    model = Path(args.xiaomi_model_path).expanduser().resolve() if args.xiaomi_model_path else XIAOMI.model_path
    return [
        env_vllm(args.xiaomi_env), "serve", str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", "xiaomi_ocr_0",
        "--trust-remote-code",
        "--max-model-len", str(args.xiaomi_max_model_len),
        "--limit-mm-per-prompt", '{"image":1}',
    ]


def healthcheck_url(port):
    return f"http://127.0.0.1:{port}/v1/models"
