#!/usr/bin/env python3
"""Batch runner for vLLM-backed methods on multiple GPUs.

This script schedules the validated vLLM-capable methods across a GPU pool.
For methods that require an external vLLM OpenAI-compatible service, it starts
the service, waits for readiness, runs the benchmark runner, then shuts the
service down. If a method fails, the failure is recorded and the worker moves
on to the next method.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import (  # noqa: E402
    CONDA_ENV_ROOT,
    EXPERIMENT_ROOT,
    TMP_ROOT,
    PDF_ROOT,
)
from execution.tasks import SERVICE_TASKS, ServiceTask, MINERU_PROFILES, uses_local_managed_service, uses_persistent_worker  # noqa: E402
from execution.processes import assert_service_port_available, http_service_ready, run_command, terminate_process, wait_for_http_or_process_exit  # noqa: E402
from execution.tasks import managed_service_identity  # noqa: E402
from execution.queue import (  # noqa: E402
    chunk_pdf_paths,
    collect_job_failures,
    collect_done_statuses,
    collect_failed_docs,
    collect_unprocessed_docs,
    collect_pdf_paths,
    write_pending_job_list,
    write_slot_job_list,
)
from config.execution_profiles import resolve_execution_profile  # noqa: E402
from methods.hunyuanocr.dflash import add_dflash_args, resolve_dflash_args, validate_dflash_paths  # noqa: E402
from config.method_versions import get_method_version  # noqa: E402
from methods.mineru.version_profile import write_runtime_config  # noqa: E402


GLMOCR = get_method_version("glmocr")



DEFAULT_METHODS = [
    "hunyuanocr_15",
    "qianfan_ocr",
    "paddleocr_vl_15",
    "glmocr",
    "monkeyocr_pro_1_2b",
    "qianfan_ocr_think",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default=str(PDF_ROOT),
        help="PDF root directory.",
    )
    parser.add_argument(
        "--output-root-base",
        default=None,
        help="Base output directory. Each method writes under a subdirectory.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Base log directory for services, runners, queues, and batch summaries.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids used as worker slots.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method ids. Defaults to all validated vLLM methods.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    from config.execution_profiles import RUNTIME_ARGUMENTS
    for key, typ, default in RUNTIME_ARGUMENTS["teleocr"]:
        choices = {"teleocr_backend": ("vllm-async-engine",), "teleocr_layout_mode": ("Detection",)}
        parser.add_argument("--" + key.replace("_", "-"), type=typ, default=default,
                            **({"choices": choices[key]} if key in choices else {}))
    parser.add_argument("--teleocr-max-concurrency", type=int, default=100,
                        help="Official default 100; fixed-P and sweep instead use --pressure.")
    parser.add_argument("--monkey-region-concurrency", type=int, default=64,
                        help="vllm_async in-flight region limit; pressure sweeps override this value.")
    parser.add_argument("--hunyuan-runner-env", default="gdb_hunyuan")
    parser.add_argument("--hunyuan-service-env", default="gdb_hunyuan")
    add_dflash_args(parser)
    parser.add_argument(
        "--hunyuan-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Override Hunyuan AR service memory fraction; default 0.90 (project setting for 1.0, official for 1.5). DFlash uses its own profile.",
    )
    parser.add_argument(
        "--hunyuan-vllm-enforce-eager",
        action="store_true",
        help=(
            "Pass --enforce-eager to the HunyuanOCR vLLM service. Use this for "
            "older/patched environments that cannot run vLLM compile/CUDA graph mode."
        ),
    )
    parser.add_argument(
        "--hunyuan-vllm-max-model-len",
        type=int,
        help=(
            "Optional smoke override for HunyuanOCR context length. The official "
            "1.5 value remains 131072 when omitted."
        ),
    )
    parser.add_argument(
        "--hunyuan-vllm-max-num-batched-tokens",
        type=int,
        help=(
            "Optional smoke override for HunyuanOCR batched-token budget. "
            "When omitted, 1.5 follows its official context length."
        ),
    )
    parser.add_argument(
        "--hunyuan-page-concurrency",
        type=int,
        default=1,
        help="Global bound on Hunyuan page render/request tasks across input PDFs.",
    )
    parser.add_argument("--qianfan-service-env", default="gdb_qianfan")
    parser.add_argument("--qianfan-page-concurrency", type=int, default=1)
    parser.add_argument("--qianfan-execution-mode", choices=("eager", "auto"), default="auto",
                        help="Qianfan: let vLLM choose compilation/CUDA graphs (default), or explicitly force eager.")
    parser.add_argument("--deepseek-service-env", default="gdb_deepseek_ocr2")
    parser.add_argument("--paddle-service-env", default="gdb_paddle_vllm")
    from methods.unlimited_ocr.commands import add_runtime_args
    add_runtime_args(parser)
    parser.add_argument(
        "--monkey-models-dir",
        help=(
            "Override the model root selected by a MonkeyOCR method id. Only valid "
            "when one MonkeyOCR variant is selected."
        ),
    )
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--startup-poll-seconds", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paddle-vl-rec-max-concurrency",
        type=int,
        help=(
            "PaddleOCR-VL internal VLM request concurrency. Passed through to "
            "PaddleOCRVL(vl_rec_max_concurrency=...)."
        ),
    )
    paddle_queue_group = parser.add_mutually_exclusive_group()
    paddle_queue_group.add_argument(
        "--paddle-use-queues",
        dest="paddle_use_queues",
        action="store_true",
        help="Enable PaddleOCR-VL pipeline queues.",
    )
    paddle_queue_group.add_argument(
        "--paddle-no-use-queues",
        dest="paddle_use_queues",
        action="store_false",
        help="Disable PaddleOCR-VL pipeline queues.",
    )
    parser.set_defaults(paddle_use_queues=None)
    parser.add_argument(
        "--paddle-vllm-gpu-memory-utilization",
        type=float,
        default=0.55,
        help="GPU memory utilization for the PaddleOCR-VL vLLM service.",
    )
    parser.add_argument(
        "--paddle-vllm-max-model-len",
        type=int,
        default=16384,
        help="PaddleOCR-VL service context limit; defaults to the official deployment preset.",
    )
    parser.add_argument(
        "--paddle-vllm-enforce-eager",
        action="store_true",
        help="Pass --enforce-eager to the PaddleOCR-VL vLLM service.",
    )
    parser.add_argument(
        "--deepseek-max-model-len",
        type=int,
        default=8192,
        help="DeepSeek-OCR-2 vLLM max model length.",
    )
    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=8192,
        help=(
            "DeepSeek-OCR-2 per-page generation limit. The official vLLM example "
            "uses 8192; larger values can cause long repeated outputs on some pages."
        ),
    )
    parser.add_argument(
        "--deepseek-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization for the DeepSeek-OCR-2 service.",
    )
    parser.add_argument(
        "--deepseek-preprocess-workers",
        type=int,
        default=64,
        help="DeepSeek-OCR-2 CPU preprocessing processes (independent of page concurrency).",
    )
    parser.add_argument(
        "--deepseek-engine-iteration-timeout-seconds",
        type=int,
        default=600,
        help="DeepSeek-OCR-2 vLLM engine iteration timeout; forwarded through VLLM_ENGINE_ITERATION_TIMEOUT_S.",
    )
    parser.add_argument(
        "--deepseek-page-concurrency",
        type=int,
        default=1,
        help="Maximum in-flight DeepSeek-OCR-2 page requests across PDFs.",
    )
    parser.add_argument(
        "--glm-max-workers",
        type=int,
        default=None,
        help=(
            "Optional GLM-OCR internal pipeline.max_workers override. If omitted, "
            "the SDK config default is used."
        ),
    )
    parser.add_argument(
        "--glm-ocr-connection-pool-size",
        type=int,
        default=128,
        help=(
            "GLM-OCR HTTP connection reuse pool; default 128. "
            "Use --glm-max-workers to control region concurrency independently."
        ),
    )
    parser.add_argument(
        "--glm-vllm-gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization for the GLM-OCR vLLM service.",
    )
    parser.add_argument(
        "--glm-layout-model-dir",
        default=str(GLMOCR.additional_model_paths[0]),
        help=(
            "GLM-OCR layout model path or HuggingFace id. For reproducible "
            "throughput runs, prefer a local path to a pre-downloaded "
            "PP-DocLayoutV3_safetensors directory."
        ),
    )
    parser.add_argument("--mineru-parse-method", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--mineru-vllm-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--mineru-lang", default="auto")
    parser.add_argument("--mineru-disable-formula", action="store_true")
    parser.add_argument("--mineru-disable-table", action="store_true")
    parser.add_argument("--mineru-api-max-concurrent-requests", type=int, default=3)
    parser.add_argument("--mineru-api-processing-window-size", type=int)
    parser.add_argument("--mineru-api-task-retention-seconds", type=int, default=3600)
    parser.add_argument("--mineru-task-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--mineru-task-poll-seconds", type=float, default=1.0)
    parser.add_argument("--mineru-http-timeout-seconds", type=float,
                        help="Override MinerU HTTP read timeouts; default: official per-version settings.")
    parser.add_argument("--mineru-hybrid-batch-ratio", type=int)
    parser.add_argument("--mineru-pdf-render-threads", type=int)
    parser.add_argument("--mineru-intra-op-num-threads", type=int)
    parser.add_argument("--mineru-inter-op-num-threads", type=int)
    parser.add_argument("--run-id", default=time.strftime("service_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    run_root = (TMP_ROOT / "smoke" if args.dry_run else EXPERIMENT_ROOT / "benchmark") / args.run_id
    args.output_root_base = args.output_root_base or str(run_root / "outputs")
    args.log_root = args.log_root or str(Path(args.output_root_base).parent / "logs")
    return args


def run_method_on_gpu(
    *,
    config: ServiceTask,
    method_id: str,
    gpu_id: int,
    port: int | None,
    args: argparse.Namespace,
    jobs: deque[list[Path]],
    method_results: list[dict[str, object]],
    lock: threading.Lock,
    job_dir: Path,
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_log_dir = Path(args.log_root) / method_id
    service_log = method_log_dir / f"{timestamp}_service_gpu{gpu_id}.log"
    service_proc: subprocess.Popen[str] | None = None
    active_batch: list[Path] = []
    status: dict[str, object] = {
        "method": method_id,
        "gpu_id": gpu_id,
        "port": port,
        "status": "pending",
        "service_log": str(service_log),
        "runner_logs": [],
        "jobs_completed": 0,
        "failed_docs": [],
        "unprocessed_docs": [],
    }
    try:
        if uses_local_managed_service(config, args):
            assert port is not None
            service_cmd = config.service_builder(args, gpu_id, port)
            if args.dry_run:
                print(f"[dry-run][{method_id}][gpu {gpu_id}] service:", " ".join(service_cmd))
            else:
                if method_id == "hunyuanocr_15_dflash":
                    validate_dflash_paths(args, create_workspace=True)
                service_env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
                if config.env_overrides:
                    service_env.update(config.env_overrides)
                assert_service_port_available(port)
                identity = managed_service_identity(method_id, service_cmd, port)
                service_proc = run_command(
                    service_cmd,
                    service_log,
                    env_overrides=service_env,
                    wait=False,
                )
                if config.healthcheck_builder is not None:
                    wait_for_http_or_process_exit(
                        config.healthcheck_builder(port),
                        proc=service_proc,
                        timeout_seconds=args.startup_timeout,
                        poll_seconds=args.startup_poll_seconds,
                        **identity,
                    )

        job_index = 0
        while True:
            with lock:
                if not jobs:
                    break
                pdf_batch = jobs.popleft()
                active_batch = pdf_batch
            runner_log = method_log_dir / f"{timestamp}_runner_gpu{gpu_id}_job{job_index:04d}.log"
            pdf_list_file = write_slot_job_list(job_dir, f"gpu{gpu_id}", job_index, pdf_batch)
            status["runner_logs"].append(str(runner_log))
            if config.runner_builder is None:
                raise ValueError(f"Method {method_id} requires its persistent worker.")
            runner_cmd = config.runner_builder(args, gpu_id, port, pdf_list_file)
            if args.dry_run:
                print(f"[dry-run][{method_id}][gpu {gpu_id}] runner:", " ".join(runner_cmd))
                status["status"] = "dry_run"
                status["jobs_completed"] = int(status["jobs_completed"]) + 1
            else:
                runner_env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
                if config.env_overrides:
                    runner_env.update(config.env_overrides)
                runner_proc = run_command(
                    runner_cmd,
                    runner_log,
                    env_overrides=runner_env,
                    wait=False,
                )
                return_code = runner_proc.wait()
                status["jobs_completed"] = int(status["jobs_completed"]) + 1
                outcomes = collect_job_failures(
                    input_root=Path(args.input_root),
                    output_root=Path(args.output_root_base) / method_id,
                    pdf_paths=pdf_batch,
                )
                failed = [item for item in outcomes if item["status"] == "failed"]
                unprocessed = [item for item in outcomes if item["status"] == "unprocessed"]
                status["failed_docs"].extend(failed)
                status["unprocessed_docs"].extend(unprocessed)
                if return_code != 0 or outcomes:
                    status["status"] = "failed"
                    status["return_code"] = return_code
                service_dead = service_proc is not None and service_proc.poll() is not None
                fatal = any(item.get("runtime_unusable") for item in failed)
                # Recorded document errors may continue; crashes/incomplete batches must stop.
                unhealthy = False
                if outcomes and port is not None and config.healthcheck_builder is not None:
                    unhealthy = not http_service_ready(config.healthcheck_builder(port))
                active_batch = []
                if service_dead or unhealthy or fatal or unprocessed or return_code not in (0, 1) or any(item.get("record_error") for item in failed) or (return_code != 0 and not failed):
                    status["status"] = "failed"
                    status["stop_reason"] = "service_or_worker_unavailable"
                    break
            job_index += 1
        else:
            pass
        if status["status"] == "pending":
            status["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["status"] = "failed"
        status["error"] = repr(exc)
        if active_batch:
            outcomes = collect_job_failures(input_root=Path(args.input_root),
                output_root=Path(args.output_root_base) / method_id, pdf_paths=active_batch)
            status["failed_docs"].extend(item for item in outcomes if item["status"] == "failed")
            status["unprocessed_docs"].extend(item for item in outcomes if item["status"] == "unprocessed")
    finally:
        terminate_process(service_proc)
        with lock:
            method_results.append(status)


def run_method_across_gpus(
    method_id: str,
    config: ServiceTask,
    gpus: list[int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if method_id == "teleocr_vllm_async" and len(gpus) != 1:
        raise ValueError("TeleOCR requires one resident worker on one GPU per invocation")
    if method_id == "deepseek_ocr2" and len(gpus) != 1:
        raise ValueError("DeepSeek page_pool requires one GPU per invocation/service")
    if method_id == "unlimited_ocr_sglang" and len(gpus) != 1:
        raise ValueError("Unlimited-OCR requires one GPU per invocation/service")
    if (method_id.startswith("hunyuanocr_") or method_id in {"qianfan_ocr", "qianfan_ocr_think"}) and len(gpus) != 1:
        raise ValueError("Hunyuan/Qianfan persistent routes require one GPU per invocation/service")
    use_persistent_workers = uses_persistent_worker(config, args)
    if use_persistent_workers:
        return run_method_with_persistent_workers(method_id, config, gpus, args)

    pdf_paths = collect_pdf_paths(Path(args.input_root))
    jobs = chunk_pdf_paths(pdf_paths, config.docs_per_job)
    if not jobs:
        return []
    method_results: list[dict[str, object]] = []
    lock = threading.Lock()
    method_log_dir = Path(args.log_root) / method_id
    job_dir = method_log_dir / "jobs"
    ports = [
        None if not uses_local_managed_service(config, args) else 9000 + (100 * list(SERVICE_TASKS).index(method_id)) + idx
        for idx in range(len(gpus))
    ]
    threads = [
        threading.Thread(
            target=run_method_on_gpu,
            kwargs={
                "config": config,
                "method_id": method_id,
                "gpu_id": gpu_id,
                "port": ports[worker_index],
                "args": args,
                "jobs": jobs,
                "method_results": method_results,
                "lock": lock,
                "job_dir": job_dir,
            },
            daemon=False,
        )
        for worker_index, gpu_id in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if jobs and method_results:
        method_results[0]["unprocessed_docs"].extend(
            {"pdf_path": str(path), "status": "unprocessed", "reason": "worker_stopped", "worker_id": None}
            for batch in jobs for path in batch
        )
    return sorted(method_results, key=lambda item: int(item["gpu_id"]))


def run_method_with_persistent_workers(
    method_id: str,
    config: ServiceTask,
    gpus: list[int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    pdf_paths = collect_pdf_paths(Path(args.input_root))
    page_pool = method_id.startswith("hunyuanocr_") or method_id in {"qianfan_ocr", "qianfan_ocr_think"}
    page_pool = page_pool or method_id == "deepseek_ocr2"
    page_pool = page_pool or (method_id == "unlimited_ocr_sglang" and args.unlimited_ocr_request_mode == "page_pool")
    jobs = [pdf_paths] if page_pool and pdf_paths else chunk_pdf_paths(pdf_paths, config.docs_per_job)
    if not jobs:
        return []

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_log_dir = Path(args.log_root) / method_id
    queue_dir = method_log_dir / f"queue_{timestamp}"
    pending_dir = queue_dir / "pending"
    service_port = 9000 + (100 * list(SERVICE_TASKS).index(method_id)) if uses_local_managed_service(config, args) else None
    args.service_port = service_port
    for job_index, pdf_batch in enumerate(jobs):
        write_pending_job_list(pending_dir, job_index, pdf_batch)

    if args.dry_run and uses_local_managed_service(config, args):
        assert service_port is not None
        service_cmd = config.service_builder(args, gpus[0], service_port)
        print(f"[dry-run][{method_id}] service:", " ".join(service_cmd))

    if args.dry_run:
        results: list[dict[str, object]] = []
        for gpu_id in gpus:
            worker_id = f"gpu{gpu_id}"
            worker_log = method_log_dir / f"{timestamp}_worker_gpu{gpu_id}.log"
            worker_cmd = config.worker_builder(args, gpu_id, queue_dir, worker_id)
            print(f"[dry-run][{method_id}][gpu {gpu_id}] worker:", " ".join(worker_cmd))
            results.append(
                {
                    "method": method_id,
                    "gpu_id": gpu_id,
                    "status": "dry_run",
                    "worker_log": str(worker_log),
                    "queue_dir": str(queue_dir),
                }
            )
        return results

    service_proc: subprocess.Popen[str] | None = None
    worker_runs: list[tuple[int, Path, subprocess.Popen[str]]] = []
    try:
        if uses_local_managed_service(config, args):
            assert service_port is not None
            service_log = method_log_dir / f"{timestamp}_service.log"
            if method_id == "hunyuanocr_15_dflash":
                validate_dflash_paths(args, create_workspace=True)
            service_cmd = config.service_builder(args, gpus[0], service_port)
            service_env = {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpus)}
            if config.env_overrides:
                service_env.update(config.env_overrides)
            if method_id.startswith("mineru"):
                service_env["HF_ENDPOINT"] = "https://hf-mirror.com"
            assert_service_port_available(service_port)
            identity = managed_service_identity(method_id, service_cmd, service_port)
            service_proc = run_command(service_cmd, service_log, env_overrides=service_env, wait=False)
            if config.healthcheck_builder is not None:
                wait_for_http_or_process_exit(config.healthcheck_builder(service_port), proc=service_proc,
                    timeout_seconds=args.startup_timeout, poll_seconds=args.startup_poll_seconds,
                    **identity)

        for gpu_id in gpus:
            worker_id = f"gpu{gpu_id}"
            worker_log = method_log_dir / f"{timestamp}_worker_gpu{gpu_id}.log"
            worker_cmd = config.worker_builder(args, gpu_id, queue_dir, worker_id)
            worker_env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
            if config.env_overrides:
                worker_env.update(config.env_overrides)
            if method_id == "glmocr":
                # conda run does not expose pip-installed nvidia-* libraries.
                cu13_lib = (CONDA_ENV_ROOT / config.runner_env / "lib" / "python3.10"
                            / "site-packages" / "nvidia" / "cu13" / "lib")
                existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                worker_env["LD_LIBRARY_PATH"] = str(cu13_lib) + (":" + existing_ld if existing_ld else "")
                worker_env["HF_HUB_OFFLINE"] = "1"
                worker_env["TRANSFORMERS_OFFLINE"] = "1"
            if method_id.startswith("mineru"):
                worker_env["HF_ENDPOINT"] = "https://hf-mirror.com"
            proc = run_command(worker_cmd, worker_log, env_overrides=worker_env, wait=False)
            worker_runs.append((gpu_id, worker_log, proc))

        done_dir = queue_dir / "done"
        results: list[dict[str, object]] = []
        for gpu_id, worker_log, proc in worker_runs:
            return_code = proc.wait()
            jobs_completed = 0
            jobs_failed = 0
            for status_path in done_dir.glob(f"*__gpu{gpu_id}.json"):
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                jobs_completed += 1
                if bool(payload.get("failed")):
                    jobs_failed += 1
            results.append(
                {
                    "method": method_id,
                    "gpu_id": gpu_id,
                    "status": "ok" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "worker_log": str(worker_log),
                    "jobs_completed": jobs_completed,
                    "jobs_failed": jobs_failed,
                    "queue_dir": str(queue_dir),
                    "service_port": service_port,
                }
            )

        statuses = collect_done_statuses(done_dir)
        for item in results:
            own = [job for job in statuses if job["worker_id"] == f"gpu{item['gpu_id']}"]
            item["failed_docs"] = collect_failed_docs(own)
            item["unprocessed_docs"] = [doc for job in own for doc in job.get("unprocessed_docs", [])]
        if results:
            results[0]["unclaimed_docs"] = collect_unprocessed_docs([], queue_dir)
        total_done = len(statuses)
        if total_done != len(jobs):
            for item in results:
                item["status"] = "failed"
                item["queue_incomplete"] = True
                item["jobs_expected"] = len(jobs)
                item["jobs_recorded"] = total_done

        return sorted(results, key=lambda item: int(item["gpu_id"]))
    finally:
        for _gpu_id, _log, proc in worker_runs:
            terminate_process(proc)
        terminate_process(service_proc)


def main() -> None:
    args = parse_args()
    resolve_dflash_args(args)
    # Workers may change their working directory while writing logs. Resolve
    # data paths once so generated job lists remain valid in every worker.
    args.input_root = str(Path(args.input_root).resolve())
    args.output_root_base = str(Path(args.output_root_base).resolve())
    args.log_root = Path(args.log_root).resolve()
    selected = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in selected if item not in SERVICE_TASKS]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    selected_monkey_variants = [item for item in selected if item.startswith("monkeyocr_")]
    if args.monkey_models_dir and len(selected_monkey_variants) > 1:
        raise SystemExit(
            "--monkey-models-dir cannot override multiple MonkeyOCR variants in one run."
        )
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    results: list[dict[str, object]] = []
    args.log_root.mkdir(parents=True, exist_ok=True)

    for method_id in selected:
        config = SERVICE_TASKS[method_id]
        if method_id in MINERU_PROFILES:
            runtime_config = write_runtime_config(MINERU_PROFILES[method_id])
            print(
                f"[config] method={method_id} env={config.runner_env} "
                f"repo={MINERU_PROFILES[method_id].repository_path} "
                f"runtime_config={runtime_config}"
            )
        profile = resolve_execution_profile(
            method_id,
            unlimited_request_mode=args.unlimited_ocr_request_mode,
            execution_mode=(
                "api" if method_id.startswith("mineru") else None
            ),
        )
        print(
            f"[start] method={method_id} profile={profile.profile_id} "
            f"class={profile.execution_class} gpus={','.join(str(g) for g in gpus)}"
        )
        method_results = run_method_across_gpus(method_id, config, gpus, args)
        results.append(
            {
                "method": method_id,
                "execution_profile": asdict(profile),
                "mode": "method_serial_gpu_parallel",
                "workers": method_results,
                "status": ("ok" if all(item["status"] in {"ok", "dry_run"} for item in method_results) else "failed"),
            }
        )
        print(f"[done] method={method_id} status={results[-1]['status']}")

    summary_path = args.log_root / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(
        json.dumps(
            {
                "methods": selected,
                "gpus": gpus,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failed = [item for item in results if item["status"] == "failed"]
    failed_methods = [str(item["method"]) for item in failed]
    failed_methods_path = args.log_root / "failed_methods.txt"
    failed_methods_path.write_text(
        "".join(f"{method}\n" for method in failed_methods),
        encoding="utf-8",
    )
    print(f"[done] summary: {summary_path}")
    print(f"[done] failed methods file: {failed_methods_path}")
    if failed:
        failed_methods_csv = ",".join(failed_methods)
        print(f"[done] failed methods: {failed_methods_csv}")
        print(
            "[done] rerun example: "
            f"python {SRC_ROOT / 'benchmark' / 'run_service_batch.py'} "
            f"--methods {failed_methods_csv} --gpus {','.join(str(g) for g in gpus)} "
            f"--input-root {args.input_root} --output-root-base {args.output_root_base} "
            f"--log-root {args.log_root} "
            f"{'--skip-existing' if args.skip_existing else ''}".strip()
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
