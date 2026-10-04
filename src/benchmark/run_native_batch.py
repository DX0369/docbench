#!/usr/bin/env python3
"""Batch runner for non-vLLM native/pipeline methods on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    TMP_ROOT,
    PDF_ROOT,
)
from execution.processes import run_command  # noqa: E402
from execution.queue import (  # noqa: E402
    chunk_pdf_paths,
    collect_pdf_paths,
    collect_done_statuses,
    collect_failed_docs,
    collect_unprocessed_docs,
    write_pending_job_list,
)
from config.execution_profiles import resolve_execution_profile  # noqa: E402
from execution.tasks import NATIVE_TASKS, NativeTask  # noqa: E402
from methods.pp_structurev3.commands import add_runtime_args
from methods.mineru_popo.commands import add_arguments as add_popo_arguments, runtime_root as popo_runtime_root
from methods.mineru_popo.commands import wait_for_coordinator


DEFAULT_METHODS = ["youtu", "pp_structurev3", "marker"]


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
        help="Base output directory.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Directory for worker logs, job queues, and summaries.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids used as worker slots.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method ids.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    add_popo_arguments(parser)
    parser.add_argument("--youtu-env", default="gdb_youtu")
    parser.add_argument("--youtu-model-path")
    parser.add_argument("--youtu-ocr-batch-size", type=int, default=5)
    parser.add_argument("--pp-device", default="gpu", choices=["gpu", "cpu"])
    add_runtime_args(parser)
    parser.add_argument("--marker-workers", type=int, default=1,
                        help="Marker shared-model document workers per GPU.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=time.strftime("native_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    run_root = (TMP_ROOT / "smoke" if args.dry_run else EXPERIMENT_ROOT / "benchmark") / args.run_id
    args.output_root_base = args.output_root_base or str(run_root / "outputs")
    args.log_root = args.log_root or str(Path(args.output_root_base).parent / "logs")
    return args


def run_method(
    method_id: str,
    config: NativeTask,
    gpus: list[int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    pdf_paths = collect_pdf_paths(Path(args.input_root))
    docs_per_job = config.docs_per_job
    if method_id == "pp_structurev3":
        docs_per_job = max(1, (len(pdf_paths) + len(gpus) - 1) // len(gpus))
        if len(set(gpus)) != len(gpus):
            raise ValueError("PP requires distinct GPU slots, not duplicated GPU IDs")
    jobs = chunk_pdf_paths(pdf_paths, docs_per_job)
    if not jobs:
        return []

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    method_log_dir = Path(args.log_root) / method_id
    queue_dir = method_log_dir / f"queue_{timestamp}"
    pending_dir = queue_dir / "pending"
    done_dir = queue_dir / "done"
    for job_index, pdf_batch in enumerate(jobs):
        write_pending_job_list(pending_dir, job_index, pdf_batch)

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

    worker_runs: list[tuple[int, Path, subprocess.Popen[str]]] = []
    for gpu_id in gpus:
        worker_id = f"gpu{gpu_id}"
        worker_log = method_log_dir / f"{timestamp}_worker_gpu{gpu_id}.log"
        worker_cmd = config.worker_builder(args, gpu_id, queue_dir, worker_id)
        env_overrides = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        if config.env_overrides:
            env_overrides.update(config.env_overrides)
        proc = run_command(
            worker_cmd,
            worker_log,
            env_overrides=env_overrides,
            wait=False,
        )
        worker_runs.append((gpu_id, worker_log, proc))

    results: list[dict[str, object]] = []
    for gpu_id, worker_log, proc in worker_runs:
        return_code = wait_for_coordinator(proc) if method_id == "mineru_344_2605_popo" else proc.wait()
        jobs_completed = 0
        jobs_failed = 0
        suffix = ".marker*" if method_id == "marker" else ".popo*" if method_id == "mineru_344_2605_popo" else ""
        pattern = f"*__gpu{gpu_id}{suffix}.json"
        for status_path in done_dir.glob(pattern):
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
            }
        )

    statuses = collect_done_statuses(done_dir)
    for item in results:
        owner = f"gpu{item['gpu_id']}"
        own = [job for job in statuses if job["worker_id"] == owner or (
            method_id == "marker" and job["worker_id"].startswith(owner + ".marker")) or (
            method_id == "mineru_344_2605_popo" and job["worker_id"].startswith(owner + ".popo"))]
        item["failed_docs"] = collect_failed_docs(own)
        item["unprocessed_docs"] = [doc for job in own for doc in job.get("unprocessed_docs", [])]
        if any(job["failed"] for job in own):
            item["status"] = "failed"
        if method_id == "mineru_344_2605_popo":
            state_path = popo_runtime_root(queue_dir, owner) / "run_status.json"
            item["runtime_status"] = str(state_path)
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                for field in ("startup_seconds", "processing_seconds", "concurrency"):
                    if field in state:
                        item[field] = state[field]
                if state["status"] != "completed":
                    item["status"] = "failed"
            else:
                item["status"] = "failed"
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


def main() -> None:
    args = parse_args()
    # Workers write logs from method-specific directories; keep data paths
    # stable when callers provide paths relative to the project root.
    args.input_root = str(Path(args.input_root).resolve())
    args.output_root_base = str(Path(args.output_root_base).resolve())
    args.log_root = Path(args.log_root).resolve()
    selected = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in selected if item not in NATIVE_TASKS]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")
    if "youtu" in selected and len(gpus) != len(set(gpus)):
        raise SystemExit("Youtu requires one worker per GPU; duplicate GPU IDs are not allowed.")
    if "mineru_344_2605_popo" in selected and len(gpus) != 1:
        raise SystemExit("MinerU + Popo currently requires exactly one GPU worker.")

    results: list[dict[str, object]] = []
    args.log_root.mkdir(parents=True, exist_ok=True)

    for method_id in selected:
        config = NATIVE_TASKS[method_id]
        profile = resolve_execution_profile(
            method_id,
        )
        print(
            f"[start] method={method_id} profile={profile.profile_id} "
            f"class={profile.execution_class} gpus={','.join(str(g) for g in gpus)}"
        )
        method_results = run_method(method_id, config, gpus, args)
        results.append(
            {
                "method": method_id,
                "execution_profile": asdict(profile),
                "mode": "method_serial_gpu_parallel_native",
                "workers": method_results,
                "status": (
                    "ok"
                    if all(item["status"] in {"ok", "dry_run"} for item in method_results)
                    else "failed"
                ),
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
        sys.exit(1)


if __name__ == "__main__":
    main()
