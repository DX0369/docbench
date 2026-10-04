#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from method_paths import method_repo_path  # noqa: E402
from methods.mineru.preload import (  # noqa: E402
    API_PRELOAD_BACKENDS, prepare_preload_environment, preload_hybrid_models,
    supports_api_preload,
)

MINERU_REPO_ROOT = method_repo_path("mineru")
if str(MINERU_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MINERU_REPO_ROOT))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Start MinerU's official FastAPI service after applying benchmark "
            "runtime environment options. Environment variables must be set before "
            "mineru.cli.fast_api is imported because the app-level semaphore is "
            "created during module import."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload-models", action="store_true", required=True,
                        help="Initialize the pinned hybrid runtime before accepting PDFs; no warmup PDF.")
    parser.add_argument("--method-id", choices=tuple(API_PRELOAD_BACKENDS), required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--method", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--lang", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--disable-formula", action="store_true")
    parser.add_argument("--disable-table", action="store_true")
    parser.add_argument("--api-output-root", type=Path)
    parser.add_argument("--api-max-concurrent-requests", type=int)
    parser.add_argument("--api-processing-window-size", type=int)
    parser.add_argument("--api-task-retention-seconds", type=int)
    parser.add_argument("--hybrid-batch-ratio", type=int)
    parser.add_argument("--pdf-render-threads", type=int)
    parser.add_argument("--intra-op-num-threads", type=int)
    parser.add_argument("--inter-op-num-threads", type=int)
    parser.add_argument(
        "--enable-access-log",
        action="store_true",
        help="Keep MinerU API access logs enabled. Disabled by default for throughput runs.",
    )
    parser.add_argument(
        "--shutdown-on-stdin-eof",
        action="store_true",
        help="Ask MinerU API to exit when stdin closes.",
    )
    args, remaining = parser.parse_known_args()
    if not supports_api_preload(args.method_id, args.backend):
        parser.error("Only the pinned local hybrid API backend is supported")
    if "--reload" in remaining:
        parser.error("--reload would bypass the model preload lifespan")
    return args, remaining


def install_preload_lifespan(app, args: argparse.Namespace) -> None:
    """Keep official startup/cleanup and load in the actual API process/loop."""
    from fastapi.responses import JSONResponse

    official_lifespan = app.router.lifespan_context
    app.state.gdb_model_readiness = None

    @asynccontextmanager
    async def lifespan(application):
        application.state.gdb_model_readiness = None
        prepare_preload_environment(table_enable=not args.disable_table)
        try:
            async with official_lifespan(application) as state:
                application.state.gdb_model_readiness = preload_hybrid_models(
                    method_id=args.method_id, backend=args.backend,
                    parse_method=args.method, language=args.lang,
                    formula_enable=not args.disable_formula,
                    table_enable=not args.disable_table,
                    model_kwargs=application.state.config,
                )
                print(f"[mineru-models-ready] {application.state.gdb_model_readiness}", flush=True)
                yield state
        finally:
            application.state.gdb_model_readiness = None

    app.router.lifespan_context = lifespan

    @app.get("/ready")
    async def model_ready():
        loaded = app.state.gdb_model_readiness
        manager = getattr(app.state, "task_manager", None)
        if loaded is None or manager is None or not manager.is_healthy():
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", **loaded}


def set_env_if_present(name: str, value: object | None) -> None:
    if value is not None:
        os.environ[name] = str(value)


def main() -> None:
    args, remaining = parse_args()
    # Restore official compilation even if the launcher shell inherited an old override.
    os.environ.pop("TORCHDYNAMO_DISABLE", None)
    set_env_if_present("MINERU_API_OUTPUT_ROOT", args.api_output_root.resolve() if args.api_output_root else None)
    set_env_if_present("MINERU_API_MAX_CONCURRENT_REQUESTS", args.api_max_concurrent_requests)
    set_env_if_present("MINERU_PROCESSING_WINDOW_SIZE", args.api_processing_window_size)
    set_env_if_present("MINERU_API_TASK_RETENTION_SECONDS", args.api_task_retention_seconds)
    set_env_if_present("MINERU_HYBRID_BATCH_RATIO", args.hybrid_batch_ratio)
    set_env_if_present("MINERU_PDF_RENDER_THREADS", args.pdf_render_threads)
    set_env_if_present("MINERU_INTRA_OP_NUM_THREADS", args.intra_op_num_threads)
    set_env_if_present("MINERU_INTER_OP_NUM_THREADS", args.inter_op_num_threads)
    if not args.enable_access_log:
        os.environ["MINERU_API_DISABLE_ACCESS_LOG"] = "1"
    if args.shutdown_on_stdin_eof:
        os.environ["MINERU_API_SHUTDOWN_ON_STDIN_EOF"] = "1"
    if os.environ.get("GDB_MINERU_DISABLE_CUDNN") == "1":
        import torch  # noqa: E402

        torch.backends.cudnn.enabled = False

    sys.argv = [
        "mineru.cli.fast_api",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        *remaining,
    ]
    prepare_preload_environment(table_enable=not args.disable_table)
    from mineru.cli import fast_api

    install_preload_lifespan(fast_api.app, args)
    fast_api.main()


if __name__ == "__main__":
    main()
