from pathlib import Path
import os

from config.method_versions import get_method_version
from execution.commands import conda_python
from paths import SRC_ROOT, PROJECT_ROOT


def build_worker(args, gpu_id, queue_dir, worker_id):
    version = get_method_version("jina_ocr_v1")
    model = Path(args.jina_model_path).expanduser().resolve() if args.jina_model_path else version.model_path
    cuda = Path(args.jina_cuda_home).expanduser().resolve()
    workspace = Path(args.jina_flashinfer_workspace).expanduser().resolve() if args.jina_flashinfer_workspace else PROJECT_ROOT / ".cache/flashinfer/jina_ocr_v1"
    return ["env", f"CUDA_HOME={cuda}",
            f"PATH={cuda / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
            f"FLASHINFER_WORKSPACE_BASE={workspace}"] + conda_python(args.jina_env) + [
        str(SRC_ROOT / "methods/jina_ocr/worker.py"),
        "--input-root", str(args.input_root),
        "--output-root", str(Path(args.output_root_base) / version.method_id),
        "--queue-dir", str(queue_dir), "--worker-id", str(worker_id),
        "--model-path", str(model), "--dpi", str(args.jina_dpi),
        "--max-tokens", str(args.jina_max_tokens),
        "--max-model-len", str(args.jina_max_model_len),
        "--page-concurrency", str(args.jina_page_concurrency),
        *(["--skip-existing"] if args.skip_existing else []),
        *(["--fail-fast"] if getattr(args, "fail_fast", False) else []),
    ]
