from pathlib import Path

from config.method_versions import get_method_version
from execution.commands import conda_python
from method_paths import method_repo_path
from paths import SRC_ROOT

VERSION = get_method_version("monkeyocrv2_b_parsing")


def build_service(args, gpu_id, port):
    python = conda_python(VERSION.conda_environment)
    python.insert(-1, "--no-capture-output")
    return ["env", "VLLM_USE_V2_MODEL_RUNNER=1", "TORCHDYNAMO_DISABLE=0"] + python + [
        str(method_repo_path("monkeyocrv2") / "parsing/serve.py"),
        "--model-path", str(VERSION.model_path),
        "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", "MonkeyOCRv2",
    ]


def build_worker(args, gpu_id, queue_dir, worker_id):
    return conda_python(VERSION.conda_environment) + [
        str(SRC_ROOT / "methods/monkeyocrv2/pipeline.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / VERSION.method_id),
        "--queue-dir", str(queue_dir), "--worker-id", str(worker_id),
        "--server-url", f"http://127.0.0.1:{args.service_port}",
        "--page-max-inflight", str(args.monkeyocrv2_page_max_inflight),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]


def healthcheck_url(port):
    return f"http://127.0.0.1:{port}/v1/models"
