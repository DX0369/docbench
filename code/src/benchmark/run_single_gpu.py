#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from execution.tasks import NATIVE_TASKS, SERVICE_TASKS, MINERU_PROFILES, uses_local_managed_service  # noqa: E402
from methods.mineru.version_profile import write_runtime_config  # noqa: E402
from config.execution_profiles import runtime_options
from config.throughput_manifest import DEFAULT_THROUGHPUT_MANIFEST, load_throughput_manifest
from methods.hunyuanocr.dflash import validate_dflash_paths  # noqa: E402
from methods.hunyuanocr.commands import AR_GPU_MEMORY_DEFAULTS, resolve_ar_gpu_memory_utilization  # noqa: E402
from methods.mineru_popo import commands as popo_commands
from paths import (  # noqa: E402
    CONDA_ENV_ROOT,
    EXPERIMENT_ROOT,
    TMP_ROOT,
    PDF_ROOT,
)
from common import collect_pdf_paths, load_pdf_paths_from_file, get_doc_output_dir, write_json  # noqa: E402
from execution.processes import assert_service_port_available, run_command, terminate_process, wait_for_http_or_process_exit  # noqa: E402
from execution.tasks import managed_service_identity  # noqa: E402
from execution.workers import start_queue_processing, queue_processing_seconds, write_timing_state
from execution.queue import (  # noqa: E402
    chunk_pdf_paths,
    collect_done_statuses,
    collect_failed_docs,
    collect_unprocessed_docs,
    write_pending_job_list,
)
from execution.trial import (  # noqa: E402
    CpuMonitor,
    GpuMonitor,
    get_pdf_page_count,
)


@dataclass(frozen=True)
class MethodSpec:
    kind: str
    task_config: Any
    env_overrides: dict[str, str]


def parse_args() -> argparse.Namespace:
    from config.execution_profiles import (add_runtime_arguments, load_selected_profile,
        profile_for_args, resolve_runtime_defaults, runtime_keys)
    from config.throughput_pressure import DEFAULT_PRESSURE_METHODS, latency_mapping, pressure_mapping
    parser = argparse.ArgumentParser(description="Run exactly one method at one fixed pressure P; never search.", allow_abbrev=False)
    parser.add_argument("--method", required=True, choices=DEFAULT_PRESSURE_METHODS)
    parser.add_argument("--gpu-id", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pressure", type=int)
    selection.add_argument("--profile-file", type=Path)
    parser.add_argument("--execution-profile", help="Exact route key when loading a selected profile.")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--pdf-list-file", type=Path)
    parser.add_argument("--output-root-base", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--service-port", type=int)
    parser.add_argument("--monitor-interval-seconds", type=float, default=2.)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--startup-poll-seconds", type=float, default=3.)
    parser.add_argument("--allow-partial-success", action="store_true", help="Diagnostic only; never selected by the tuner.")
    parser.add_argument(
        "--latency-mode",
        action="store_true",
        help="Process the existing PDF list sequentially in memory with one document per callback.",
    )
    parser.add_argument("--keep-trial-outputs", action="store_true", default=True)
    parser.add_argument("--discard-successful-outputs", dest="keep_trial_outputs", action="store_false",
                        help="Tuner only: discard successful temporary document outputs.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.latency_mode and args.pressure is not None and args.pressure != 1:
        parser.error("--latency-mode requires --pressure 1")
    if args.latency_mode and args.profile_file is not None:
        parser.error("--latency-mode cannot use a saved throughput profile")
    if args.profile_file is not None:
        runtime_flags = {"--" + key.replace("_", "-") for key in runtime_keys(args.method)}
        if any(token.split("=")[0] in runtime_flags for token in sys.argv[1:]):
            parser.error("With --profile-file runtime options come from the saved profile; select --execution-profile instead.")
        profile = args.execution_profile or profile_for_args(args.method, args)
        entry = load_selected_profile(args.profile_file, args.method, profile)
        args.pressure = entry["pressure"]
        for key, value in entry["runtime_options"].items():
            setattr(args, key, value)
    elif args.execution_profile is not None and args.execution_profile != profile_for_args(args.method, args):
        parser.error("--execution-profile does not match the selected request route")
    resolve_runtime_defaults(args, args.method)
    if args.latency_mode:
        outer, overrides, mapping, docs = latency_mapping(
            args.method,
            args.unlimited_ocr_request_mode,
        )
    else:
        outer, overrides, mapping, docs = pressure_mapping(
            args.method,
            args.pressure,
            unlimited_request_mode=args.unlimited_ocr_request_mode,
        )
    if args.latency_mode and outer != 1:
        parser.error("--latency-mode requires pressure 1 to map to one worker")
    for key, value in overrides.items():
        setattr(args, key, value)
    if args.latency_mode and args.method == "marker" and args.marker_workers != 1:
        parser.error("--latency-mode requires one Marker document worker")
    args.outer_concurrency, args.docs_per_job, args.pressure_mapping = outer, docs, mapping
    if args.input_root is None:
        args.input_root = PDF_ROOT
        if args.pdf_list_file is None:
            args.pdf_list_file = DEFAULT_THROUGHPUT_MANIFEST
    return args


def build_method_spec(method: str, args: argparse.Namespace) -> MethodSpec:
    if method in SERVICE_TASKS:
        config = SERVICE_TASKS[method]
        if method in MINERU_PROFILES and not args.dry_run:
            write_runtime_config(MINERU_PROFILES[method])
        env_overrides = dict(config.env_overrides or {})
        return MethodSpec(kind="vllm", task_config=config, env_overrides=env_overrides)
    if method in NATIVE_TASKS:
        config = NATIVE_TASKS[method]
        env_overrides = dict(config.env_overrides or {})
        return MethodSpec(kind="native", task_config=config, env_overrides=env_overrides)
    raise KeyError(f"Unknown throughput method: {method}")


def load_pdf_paths(args: argparse.Namespace) -> List[Path]:
    if args.pdf_list_file is not None:
        args.pdf_list_file = args.pdf_list_file.resolve()
        loader = load_throughput_manifest if args.pdf_list_file.suffix == ".json" else load_pdf_paths_from_file
        return loader(args.pdf_list_file)
    return collect_pdf_paths(args.input_root)


def build_trial_args(base_args: argparse.Namespace, *, trial_output_root: Path) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.input_root = str(base_args.input_root)
    args.output_root_base = str(trial_output_root)
    return args


def _build_worker_env_overrides(
    *,
    method: str,
    config: Any,
    base_args: argparse.Namespace,
    method_spec: MethodSpec,
    worker_id: str,
) -> dict[str, str]:
    """Build env_overrides for a worker process, including method-specific fixes."""
    env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": str(base_args.gpu_id)}
    env.update(method_spec.env_overrides)

    if method.startswith("mineru"):
        env["HF_ENDPOINT"] = "https://hf-mirror.com"

    if method == "glmocr":
        # GLM-OCR layout detection needs libnvrtc-builtins.so.13.0 from
        # nvidia/cu13/lib inside its conda env.  conda run does not expose
        # pip-installed nvidia-* libraries on LD_LIBRARY_PATH automatically.
        cu13_lib = (
            CONDA_ENV_ROOT / config.runner_env / "lib" / "python3.10"
            / "site-packages" / "nvidia" / "cu13" / "lib"
        )
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(cu13_lib) + (":" + existing_ld if existing_ld else "")
        )
        # HuggingFace model downloads require network which is blocked in
        # this environment; force offline mode so cached models are used.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    if getattr(base_args, "latency_mode", False):
        env["GDB_LATENCY_MODE"] = "1"
        env["GDB_LATENCY_METHOD"] = method
        env["GDB_LATENCY_PDF_PATHS"] = json.dumps(
            [str(path) for path in base_args.latency_pdf_paths],
            ensure_ascii=False,
        )
        env["GDB_LATENCY_RECORD_PATH"] = str(
            base_args.latency_record_root / f"{worker_id}.json"
        )
        env["GDB_LATENCY_TIMING_ROOT"] = str(base_args.latency_timing_root)

    return env


def start_vllm_service(
    *,
    method: str,
    config: Any,
    args: argparse.Namespace,
    gpu_id: int,
    port: int,
    trial_log_dir: Path,
    method_spec: MethodSpec,
) -> tuple[subprocess.Popen[str] | None, float]:
    if config.service_builder is None or not uses_local_managed_service(config, args):
        return None, 0.0
    service_log = trial_log_dir / f"{method}_service_gpu{gpu_id}.log"
    if method == "hunyuanocr_15_dflash" and not args.dry_run:
        validate_dflash_paths(args, create_workspace=True)
    service_cmd = config.service_builder(args, gpu_id, port)
    if args.dry_run:
        print(f"[dry-run][{method}] service:", " ".join(service_cmd))
        return None, 0.0

    assert_service_port_available(port)
    identity = managed_service_identity(method, service_cmd, port)
    startup_start = time.perf_counter()
    env_overrides = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }
    env_overrides.update(method_spec.env_overrides)
    disable_dynamo = (args.hunyuan_vllm_enforce_eager if method in AR_GPU_MEMORY_DEFAULTS
                      else method != "hunyuanocr_15_dflash")
    if method in {"deepseek_ocr2", "glmocr", "lightonocr2_1b", "xiaomi_ocr_0"} or method in MINERU_PROFILES:
        # Preserve the official model/vLLM compilation defaults.
        disable_dynamo = False
    if method in {"paddleocr_vl_15", "paddleocr_vl_16"}:
        disable_dynamo = args.paddle_vllm_enforce_eager
    if method in {"qianfan_ocr", "qianfan_ocr_think"}:
        disable_dynamo = args.qianfan_execution_mode == "eager"
    if disable_dynamo:
        env_overrides["TORCHDYNAMO_DISABLE"] = "1"
    if method.startswith("mineru"):
        env_overrides["HF_ENDPOINT"] = "https://hf-mirror.com"
    if method == "glmocr":
        # vLLM serve for GLM-OCR may need the same offline HF settings when
        # loading model configs from the local model directory.
        env_overrides["HF_HUB_OFFLINE"] = "1"
        env_overrides["TRANSFORMERS_OFFLINE"] = "1"

    proc = run_command(
        service_cmd,
        service_log,
        env_overrides=env_overrides,
        wait=False,
    )
    try:
        if config.healthcheck_builder is not None:
            wait_for_http_or_process_exit(
                config.healthcheck_builder(port), proc=proc,
                timeout_seconds=args.startup_timeout, poll_seconds=args.startup_poll_seconds,
                **identity)
    except BaseException:
        terminate_process(proc)
        raise
    startup_seconds = round(time.perf_counter() - startup_start, 6)
    return proc, startup_seconds


def uses_model_ready_timing(method: str, args: argparse.Namespace) -> bool:
    if method in {"teleocr_vllm_async", "lightonocr2_1b", "xiaomi_ocr_0"}:
        return True
    if method == "mineru_344_2605_popo":
        return True  # The combination coordinator owns its inner ready/start/done.
    if method == "unlimited_ocr_sglang":
        return True
    if method == "youtu":
        return True
    if method == "deepseek_ocr2":
        return True
    if method.startswith("hunyuanocr_"):
        return True
    if method in {"qianfan_ocr", "qianfan_ocr_think"}:
        return True
    if method in {"marker", "pp_structurev3", "glmocr", "paddleocr_vl_15", "paddleocr_vl_16"} or method.startswith("monkeyocr_"):
        return True
    if method in MINERU_PROFILES:
        return True
    return False


def successful_inputs(pdf_paths, job_statuses, failed_docs, unprocessed_docs):
    recorded = {path for job in job_statuses for path in job["job_pdf_paths"]}
    excluded = {item["pdf_path"] for item in (*failed_docs, *unprocessed_docs)}
    return [path for path in pdf_paths if str(path) in recorded and str(path) not in excluded]


def load_latency_job_statuses(record_root: Path, pdf_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert the worker's in-memory latency records to queue-compatible status rows."""
    record_paths = sorted(record_root.glob("*.json")) if record_root.exists() else []
    records: list[dict[str, Any]] = []
    for path in record_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Latency record must be a list: {path}")
        records.extend(payload)
    by_path = {str(item.get("pdf_path")): item for item in records}
    if len(by_path) != len(records):
        raise ValueError("Latency records contain duplicate PDF paths")
    statuses: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        item = by_path.get(str(pdf_path))
        if item is None:
            continue
        failed = item.get("status") != "ok"
        status: dict[str, Any] = {
            "job_pdf_paths": [str(pdf_path)],
            "failed": failed,
            "processing_seconds": item.get("processing_seconds"),
        }
        if failed:
            failure = item.get("failure") or {}
            status["failed_docs"] = [{
                "pdf_path": str(pdf_path),
                "reason": str(failure.get("reason") or item.get("error") or "latency document failed"),
                "error_json": failure.get("error_json"),
            }]
        elif item.get("input_page_count") is not None:
            status["successful_input_page_counts"] = {
                str(pdf_path): item["input_page_count"]
            }
        statuses.append(status)
    return statuses, records




def run_trial(
    *,
    method: str,
    method_spec: MethodSpec,
    base_args: argparse.Namespace,
    pdf_paths: List[Path],
    total_pages: int,
    concurrency: int,
    run_name: str,
) -> Dict[str, Any]:
    if base_args.latency_mode and method == "mineru_344_2605_popo":
        raise ValueError("MinerU-Popo latency mode is not supported until the colocated services fit the target GPU")
    if method == "teleocr_vllm_async" and (concurrency != 1 or base_args.teleocr_max_concurrency < 1):
        raise ValueError("TeleOCR requires one resident worker and positive request concurrency")
    if method == "lightonocr2_1b" and (concurrency != 1 or base_args.lightonocr2_page_concurrency < 1):
        raise ValueError("LightOnOCR-2 requires one resident worker and positive page concurrency")
    if method == "xiaomi_ocr_0" and (concurrency != 1 or base_args.xiaomi_page_concurrency < 1):
        raise ValueError("Xiaomi requires one resident worker and positive page concurrency")
    if method == "mineru_344_2605_popo":
        if concurrency != 1:
            raise ValueError("Popo requires one coordinator; tune --popo-concurrency")
        popo_commands.validate_options(base_args)
    if method in {"paddleocr_vl_15", "paddleocr_vl_16"}:
        if concurrency != 1 or base_args.paddle_use_queues is not True:
            raise ValueError("Paddle throughput requires one pipeline and enabled queues; tune --pressure")
        if base_args.paddle_vl_rec_max_concurrency is not None and base_args.paddle_vl_rec_max_concurrency < 1:
            raise ValueError("--paddle-vl-rec-max-concurrency must be positive")
    if method == "pp_structurev3" and concurrency != 1:
        raise ValueError("PP page_batch requires one pipeline per GPU")
    if method == "marker" and (concurrency != 1 or base_args.marker_workers < 1):
        raise ValueError("Marker requires one outer supervisor and positive --marker-workers")
    if method.startswith("monkeyocr_"):
        if concurrency != 1 or base_args.monkey_region_concurrency < 1:
            raise ValueError("Monkey async requires one outer worker and positive --monkey-region-concurrency")
    if method == "youtu" and (concurrency != 1 or base_args.youtu_ocr_batch_size < 1):
        raise ValueError("Youtu requires one resident worker and positive --youtu-ocr-batch-size")
    if method in AR_GPU_MEMORY_DEFAULTS:
        base_args = argparse.Namespace(**vars(base_args))
        base_args.hunyuan_vllm_gpu_memory_utilization = resolve_ar_gpu_memory_utilization(
            method, base_args.hunyuan_vllm_gpu_memory_utilization)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_id = f"p{base_args.pressure:03d}_{timestamp}"
    trial_output_root = base_args.output_root_base / run_name / trial_id
    trial_log_dir = base_args.log_root / trial_id
    internal_dir = trial_log_dir / ".internal"
    queue_dir = internal_dir if base_args.latency_mode else trial_log_dir / "queue"
    pending_dir = queue_dir / "pending"
    done_dir = queue_dir / "done"
    worker_log_dir = trial_log_dir / "workers"
    worker_log_dir.mkdir(parents=True, exist_ok=True)

    if base_args.latency_mode:
        base_args.latency_pdf_paths = list(pdf_paths)
        base_args.latency_record_root = internal_dir / "latency"
        base_args.latency_timing_root = internal_dir
        base_args.latency_record_root.mkdir(parents=True, exist_ok=True)

    config = method_spec.task_config
    docs_per_job = base_args.docs_per_job or int(config.docs_per_job)
    if method == "mineru_344_2605_popo":
        docs_per_job = 1
    if method in {"pp_structurev3", "paddleocr_vl_15", "paddleocr_vl_16", "glmocr", "monkeyocr_pro_1_2b", "monkeyocr_pro_3b"}:
        docs_per_job = len(pdf_paths)
    if method == "unlimited_ocr_sglang" and base_args.unlimited_ocr_request_mode == "page_pool":
        if concurrency != 1:
            raise ValueError("Unlimited page_pool requires one client")
        docs_per_job = len(pdf_paths)
    if method == "deepseek_ocr2":
        if concurrency != 1:
            raise ValueError("DeepSeek page pool requires one outer worker; use --pressure")
        docs_per_job = len(pdf_paths)
    if method.startswith("hunyuanocr_"):
        if concurrency != 1:
            raise ValueError("Hunyuan page pool requires one outer worker; use --pressure")
        docs_per_job = len(pdf_paths)
    if method in {"lightonocr2_1b", "xiaomi_ocr_0"}:
        if concurrency != 1:
            raise ValueError(f"{method} page pool requires one outer worker; use --pressure")
        docs_per_job = len(pdf_paths)
    if method in {"qianfan_ocr", "qianfan_ocr_think"}:
        if concurrency != 1:
            raise ValueError("Qianfan page_pool requires one outer worker; use --pressure")
        docs_per_job = len(pdf_paths)
    if base_args.latency_mode:
        jobs = chunk_pdf_paths(pdf_paths, 1)
    else:
        jobs = chunk_pdf_paths(pdf_paths, docs_per_job)
        for job_index, batch in enumerate(jobs):
            write_pending_job_list(pending_dir, job_index, batch)

    trial_args = build_trial_args(base_args, trial_output_root=trial_output_root)
    monitor = GpuMonitor(base_args.gpu_id, base_args.monitor_interval_seconds)
    cpu_monitor = CpuMonitor(base_args.monitor_interval_seconds)
    service_proc: subprocess.Popen[str] | None = None
    startup_seconds = 0.0
    worker_procs: list[tuple[str, Path, subprocess.Popen[str]]] = []
    model_ready_timing = uses_model_ready_timing(method, base_args)
    coordinator_timing = method == "mineru_344_2605_popo"
    coordinator_startup_seconds = None
    processing_start = None
    processing_seconds = None
    trial_error = None
    worker_results: List[Dict[str, Any]] = []
    if model_ready_timing and not coordinator_timing and not base_args.dry_run:
        timing_dir = queue_dir / "timing"
        timing_dir.mkdir(parents=True, exist_ok=False)
        write_timing_state(timing_dir / "control.json", {"parent_pid": os.getpid(), "method": method})

    trial_start = time.perf_counter()
    if not base_args.dry_run:
        monitor.start()
        cpu_monitor.start()
    try:
        if method_spec.kind == "vllm":
            service_port = base_args.service_port
            if service_port is None and uses_local_managed_service(config, trial_args):
                service_port = 9800
            trial_args.service_port = service_port
            service_proc, startup_seconds = start_vllm_service(
                method=method,
                config=config,
                args=trial_args,
                gpu_id=base_args.gpu_id,
                port=int(service_port) if service_port is not None else 0,
                trial_log_dir=trial_log_dir,
                method_spec=method_spec,
            )

            def launch_vllm_worker(worker_index: int) -> None:
                worker_id = f"w{worker_index:02d}"
                worker_log = worker_log_dir / f"{worker_id}.log"
                if config.worker_builder is None:
                    raise ValueError(f"Method {method} has no registered persistent worker")
                worker_cmd = config.worker_builder(
                    trial_args,
                    base_args.gpu_id,
                    queue_dir,
                    worker_id,
                )
                if base_args.dry_run:
                    print(f"[dry-run][{method}][c={concurrency}] worker:", " ".join(worker_cmd))
                    return
                worker_env = _build_worker_env_overrides(
                    method=method,
                    config=config,
                    base_args=base_args,
                    method_spec=method_spec,
                    worker_id=worker_id,
                )
                proc = run_command(
                    worker_cmd,
                    worker_log,
                    env_overrides=worker_env,
                    wait=False,
                )
                worker_procs.append((worker_id, worker_log, proc))

            for worker_index in range(concurrency):
                launch_vllm_worker(worker_index)
        else:
            def launch_native_worker(worker_index: int) -> None:
                worker_id = f"w{worker_index:02d}"
                worker_log = worker_log_dir / f"{worker_id}.log"
                worker_cmd = config.worker_builder(
                    trial_args,
                    base_args.gpu_id,
                    queue_dir,
                    worker_id,
                )
                env_overrides = _build_worker_env_overrides(
                    method=method,
                    config=config,
                    base_args=base_args,
                    method_spec=method_spec,
                    worker_id=worker_id,
                )
                if base_args.dry_run:
                    print(f"[dry-run][{method}][c={concurrency}] worker:", " ".join(worker_cmd))
                    return
                proc = run_command(
                    worker_cmd,
                    worker_log,
                    env_overrides=env_overrides,
                    wait=False,
                )
                worker_procs.append((worker_id, worker_log, proc))

            for worker_index in range(concurrency):
                launch_native_worker(worker_index)

        if model_ready_timing and not coordinator_timing and not base_args.dry_run:
            processing_start = start_queue_processing(queue_dir, worker_procs,
                timeout=base_args.startup_timeout, service=service_proc)
        for worker_id, worker_log, proc in worker_procs:
            return_code = popo_commands.wait_for_coordinator(proc) if coordinator_timing else proc.wait()
            worker_results.append(
                {
                    "worker_id": worker_id,
                    "worker_log": str(worker_log),
                    "return_code": return_code,
                    "status": "ok" if return_code == 0 else "failed",
                }
            )
        if coordinator_timing and not base_args.dry_run:
            state = json.loads((popo_commands.runtime_root(queue_dir, "w00") / "run_status.json").read_text())
            coordinator_startup_seconds = state["startup_seconds"]
            if "processing_seconds" in state:
                processing_seconds = state["processing_seconds"]
            if state["status"] != "completed":
                trial_error = "Popo coordinator did not complete successfully; see its run_status.json"
        elif model_ready_timing and not base_args.dry_run:
            processing_seconds = round(queue_processing_seconds(
                queue_dir, [worker_id for worker_id, _, _ in worker_procs], processing_start), 6)
    except Exception as exc:
        trial_error = f"{type(exc).__name__}: {exc}"
        print(f"[trial-error] {trial_error}", flush=True)
    finally:
        for _, _, proc in worker_procs:
            terminate_process(proc)
        terminate_process(service_proc)
        if not base_args.dry_run:
            monitor.stop()
            cpu_monitor.stop()

    # t2 is the worker output/status completion event, never teardown or join.
    total_wall_seconds = None
    if processing_start is not None and processing_seconds is not None:
        total_wall_seconds = round(processing_start - trial_start + processing_seconds, 6)
    elif coordinator_timing and coordinator_startup_seconds is not None and processing_seconds is not None:
        total_wall_seconds = round(coordinator_startup_seconds + processing_seconds, 6)
    if not worker_results:
        worker_results = [{"worker_id": worker_id, "worker_log": str(log), "return_code": proc.poll(),
                           "status": "ok" if proc.poll() == 0 else "failed"}
                          for worker_id, log, proc in worker_procs]
    latency_records: list[dict[str, Any]] = []
    if base_args.latency_mode:
        try:
            job_statuses, latency_records = load_latency_job_statuses(
                base_args.latency_record_root, pdf_paths
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            trial_error = f"{type(exc).__name__}: {exc}"
            job_statuses = []
    else:
        job_statuses = collect_done_statuses(done_dir)
    failed_jobs = [payload for payload in job_statuses if bool(payload.get("failed"))]
    failed_docs = collect_failed_docs(job_statuses)
    unprocessed_docs = (
        [] if base_args.latency_mode
        else collect_unprocessed_docs(job_statuses, queue_dir)
    )
    completed_inputs = successful_inputs(pdf_paths, job_statuses, failed_docs, unprocessed_docs)
    if model_ready_timing and not coordinator_timing and not base_args.dry_run:
        try:
            page_counts = {
                Path(path): count
                for job in job_statuses
                for path, count in job.get("successful_input_page_counts", {}).items()
            }
            if set(page_counts) != set(completed_inputs):
                raise ValueError("Worker success page counts do not match completed input documents")
        except (KeyError, ValueError) as exc:
            trial_error = f"{type(exc).__name__}: {exc}"
            page_counts, completed_inputs = {}, []
    else:
        page_counts = {path: get_pdf_page_count(path) for path in completed_inputs}
    queue_complete = len(job_statuses) == len(jobs)
    workers_ok = all(item["status"] == "ok" for item in worker_results)
    queue_success = queue_complete and not failed_jobs and not failed_docs and not unprocessed_docs and workers_ok and trial_error is None
    worker_exit_failures_ignored = False
    partial_success_accepted = bool(
        base_args.allow_partial_success and not base_args.dry_run and queue_complete and failed_jobs and trial_error is None
    )
    trial_status = (
        "dry_run"
        if base_args.dry_run and trial_error is None
        else (
            "ok"
            if queue_success
            else "partial_ok" if partial_success_accepted else "failed"
        )
    )

    successful_pages = sum(page_counts.values())
    successful_docs = len(completed_inputs)
    if base_args.latency_mode and not base_args.dry_run:
        latency_output = trial_output_root / method / "latency_summary.json"
        latency_output.parent.mkdir(parents=True, exist_ok=True)
        latency_payload = {
            "method": method,
            "measurement_mode": "latency",
            "latency_profile": base_args.pressure_mapping["profile"],
            "pressure_mapping": base_args.pressure_mapping,
            "pressure": base_args.pressure,
            "startup_seconds": round(processing_start - trial_start, 6) if processing_start is not None else None,
            "processing_seconds": processing_seconds,
            "total_wall_seconds": total_wall_seconds,
            "documents": latency_records,
        }
        latency_output.write_text(
            json.dumps(latency_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    quality_summary = {}
    if method == "unlimited_ocr_sglang" and base_args.unlimited_ocr_request_mode == "joint":
        warning_docs = []
        for path in completed_inputs:
            output = get_doc_output_dir(base_args.input_root, trial_output_root / method, path)
            metrics = json.loads((output / "metrics.json").read_text())
            if metrics["quality_warnings"]:
                warning_docs.append({"pdf_path": str(path), "input_page_count": metrics["input_page_count"],
                    "output_segment_count": metrics["output_segment_count"], "finish_reason": metrics["finish_reason"],
                    "quality_warnings": metrics["quality_warnings"]})
        quality_summary = {
            "throughput_page_basis": "completed_request_input",
            "quality_warning_doc_count": len(warning_docs),
            "truncated_doc_count": sum("generation_length" in doc["quality_warnings"] for doc in warning_docs),
            "page_mismatch_doc_count": sum("page_count_mismatch" in doc["quality_warnings"] for doc in warning_docs),
            "quality_warning_docs": warning_docs,
        }
    elif method in {"qianfan_ocr", "qianfan_ocr_think"}:
        warning_docs = []
        for path in completed_inputs:
            output = get_doc_output_dir(base_args.input_root, trial_output_root / method, path)
            metrics = json.loads((output / "metrics.json").read_text())
            if metrics["quality_warnings"]:
                warning_docs.append({"pdf_path": str(path),
                    "invalid_image_box_count": metrics["invalid_image_box_count"],
                    "quality_warnings": metrics["quality_warnings"]})
        quality_summary = {"quality_warning_doc_count": len(warning_docs),
                           "quality_warning_docs": warning_docs}
    valid_processing = processing_seconds is not None and processing_seconds > 0
    summary: Dict[str, Any] = {
        **quality_summary,
        "method": method,
        "measurement_mode": "latency" if base_args.latency_mode else "throughput",
        "kind": method_spec.kind,
        "gpu_id": base_args.gpu_id,
        "concurrency": concurrency,
        "docs_per_job": docs_per_job,
        "doc_count": len(pdf_paths),
        "page_count": total_pages,
        "status": trial_status,
        "run_name": run_name,
        "trial_id": trial_id,
        "trial_output_root": str(trial_output_root),
        "trial_log_dir": str(trial_log_dir),
        "service_startup_seconds": startup_seconds if startup_seconds else None,
        "startup_seconds": (coordinator_startup_seconds if coordinator_timing else
                            round(processing_start - trial_start, 6) if processing_start is not None else None),
        "timing_scope": "models_ready_to_outputs_complete_v1" if model_ready_timing else "legacy_service_startup_subtracted_includes_worker_startup_and_cleanup",
        "trial_error": trial_error,
        "successful_doc_count": successful_docs,
        "successful_page_count": successful_pages,
        "total_wall_seconds": total_wall_seconds,
        "processing_seconds": processing_seconds,
        "docs_per_second": (round(successful_docs / total_wall_seconds, 6) if total_wall_seconds is not None and total_wall_seconds > 0 and not base_args.dry_run else None),
        "pages_per_second": (
            round(successful_pages / total_wall_seconds, 6) if total_wall_seconds is not None and total_wall_seconds > 0 and not base_args.dry_run else None
        ),
        "processing_docs_per_second": (
            round(successful_docs / processing_seconds, 6) if valid_processing else None
        ),
        "processing_pages_per_second": (
            round(successful_pages / processing_seconds, 6) if valid_processing else None
        ),
        "queue_complete": queue_complete,
        "jobs_expected": len(jobs),
        "jobs_recorded": len(job_statuses),
        "failed_job_count": len(failed_jobs),
        "unprocessed_doc_count": len(unprocessed_docs),
        "unprocessed_docs": unprocessed_docs,
        "failed_doc_count": len(failed_docs),
        "failed_docs": failed_docs,
        "partial_success_accepted": partial_success_accepted,
        "worker_exit_failures_ignored": worker_exit_failures_ignored,
        "worker_results": worker_results,
        "job_statuses": job_statuses,
        "pressure": base_args.pressure,
        "pressure_mapping": base_args.pressure_mapping,
        "runtime_options": runtime_options(base_args, method) if not coordinator_timing else popo_commands.runtime_options(base_args),
        "gpu_monitor": monitor.summary(),
        "cpu_monitor": cpu_monitor.summary(),
    }

    trial_log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_log_dir / "summary.json"
    from config.execution_profiles import resolve_execution_profile
    summary["execution_profile"] = resolve_execution_profile(
        method,
        unlimited_request_mode=base_args.unlimited_ocr_request_mode,
    ).profile_id
    if base_args.latency_mode:
        summary["latency_profile"] = base_args.pressure_mapping.get("profile", "official_defaults")
        summary["document_concurrency"] = concurrency
        summary["internal_pressure_semantics"] = base_args.pressure_mapping.get(
            "internal_pressure_semantics", "method_default"
        )
        summary["latency_summary_path"] = str(
            trial_output_root / method / "latency_summary.json"
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] method={method} concurrency={concurrency} "
        f"status={trial_status} total_wall_seconds={total_wall_seconds}"
    )
    print(f"[done] trial summary: {summary_path}")

    if (not base_args.keep_trial_outputs and not base_args.dry_run and trial_status == "ok"
            and not quality_summary.get("quality_warning_doc_count")):
        shutil.rmtree(trial_output_root, ignore_errors=True)
        summary["trial_output_removed"] = True

    if base_args.latency_mode and trial_status == "ok":
        # The final latency summary is part of outputs; queue/timing transport
        # state is not a result artifact and should not accumulate.
        shutil.rmtree(internal_dir, ignore_errors=True)

    return summary


def main() -> None:
    args = parse_args()
    from config.execution_profiles import profile_for_args
    args.run_name = args.run_name or f"{args.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    root = TMP_ROOT / "smoke" if args.dry_run else EXPERIMENT_ROOT / "benchmark"
    if args.output_root_base is None:
        args.output_root_base = root
    if args.log_root is None:
        args.log_root = args.output_root_base / args.run_name / "logs"
    args.input_root = args.input_root.resolve()
    args.output_root_base = args.output_root_base.resolve()
    args.log_root = args.log_root.resolve()
    if args.pdf_list_file is not None:
        args.pdf_list_file = args.pdf_list_file.resolve()
    aggregate_dir = args.log_root
    if aggregate_dir.exists() or (args.output_root_base / args.run_name).exists():
        raise FileExistsError(f"Run output already exists; choose a new --run-name: {args.run_name}")
    metadata = args.output_root_base / args.run_name / "metadata"
    metadata.mkdir(parents=True, exist_ok=False)
    import shlex
    (metadata / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    status_path = metadata / "run_status.json"
    write_json(status_path, {"status": "running", "method": args.method, "pressure": args.pressure})
    try:
        pdf_paths = load_pdf_paths(args)
        if not pdf_paths or len(set(pdf_paths)) != len(pdf_paths):
            raise ValueError("Fixed-P run requires nonempty unique input PDFs")
        if args.docs_per_job is None:
            args.docs_per_job = len(pdf_paths)
        total_pages = sum(get_pdf_page_count(path) for path in pdf_paths)
        spec = build_method_spec(args.method, args)
        trial = run_trial(method=args.method, method_spec=spec, base_args=args, pdf_paths=pdf_paths,
                          total_pages=total_pages, concurrency=args.outer_concurrency, run_name=args.run_name)
        aggregate = {
            "method": args.method, "execution_profile": profile_for_args(args.method, args),
            "pressure": args.pressure, "gpu_id": args.gpu_id, "run_name": args.run_name,
            "input_root": str(args.input_root), "pdf_list_file": str(args.pdf_list_file) if args.pdf_list_file else None,
            "input_pdfs": [str(path) for path in pdf_paths], "doc_count": len(pdf_paths), "page_count": total_pages,
            "status": trial["status"], "results": [trial],
            "best_stable_config": trial if trial["status"] == "ok" else None,
            "failed_doc_count": trial["failed_doc_count"], "failed_docs": trial["failed_docs"],
        }
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        write_json(aggregate_dir / "aggregate_summary.json", aggregate)
        print(f"[done] fixed-P result: {aggregate_dir / 'aggregate_summary.json'}")
        if trial["status"] not in ("ok", "dry_run", "partial_ok"):
            raise SystemExit(1)
    except BaseException as exc:
        write_json(status_path, {"status": "failed", "method": args.method, "pressure": args.pressure,
                                "error": f"{type(exc).__name__}: {exc}"})
        raise
    else:
        write_json(status_path, {"status": "dry_run" if args.dry_run else "completed" if trial["status"] == "ok" else "partial",
                                "method": args.method, "pressure": args.pressure,
                                "result": str(aggregate_dir / "aggregate_summary.json")})


if __name__ == "__main__":
    main()
