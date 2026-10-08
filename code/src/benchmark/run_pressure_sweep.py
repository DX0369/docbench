#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import collect_pdf_paths, load_pdf_paths_from_file  # noqa: E402
from config.throughput_manifest import DEFAULT_THROUGHPUT_MANIFEST, load_throughput_manifest
from config.throughput_pressure import DEFAULT_PRESSURE_METHODS, PRESSURE_SPECS, pressure_mapping  # noqa: E402
from execution.processes import run_command  # noqa: E402
from execution.trial import get_pdf_page_count  # noqa: E402
from paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    PDF_ROOT,
    PROJECT_ROOT,
)


SINGLE_GPU_RUNNER = SRC_ROOT / "benchmark" / "run_single_gpu.py"


@dataclass(frozen=True)
class PressureTrial:
    method: str
    pressure: int
    gpu_id: int
    service_port: int
    run_name: str
    command: list[str]
    mapping: dict[str, Any]
    aggregate_path: Path
    log_path: Path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    items = [int(item) for item in parse_csv(value)]
    if any(item <= 0 for item in items):
        raise ValueError("pressure/gpu values must be positive integers except GPU id 0 is allowed.")
    return items


def parse_gpu_csv(value: str) -> list[int]:
    items = [int(item) for item in parse_csv(value)]
    if any(item < 0 for item in items):
        raise ValueError("GPU ids must be non-negative.")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified pressure-based throughput runner. Humans sweep only pressure; "
            "the script maps pressure to each method's official high-throughput knob."
        )
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_PRESSURE_METHODS))
    parser.add_argument("--pressures", help="Increasing positive pressures; default 1,2,4,8,16,32,64,128,256 (GLM stops at 128).")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--execution-policy", choices=("gpu-pool", "serial"), default="gpu-pool",
                        help="Same-card trials are exclusive; serial also disables cross-card method concurrency.")
    parser.add_argument("--input-root", type=Path, help="Directory input, or relative-output path root when a list is supplied; default input is the fixed 20-document manifest.")
    parser.add_argument("--pdf-list-file", type=Path, help="Runtime TXT list or canonical throughput JSON manifest.")
    parser.add_argument("--run-name", default=f"pressure_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--output-root-base", type=Path, default=PROJECT_ROOT / "tmp" / "tuning")
    parser.add_argument("--log-root", type=Path, help="Optional diagnostic root; defaults to this run's logs directory")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT / "tuning")
    parser.add_argument("--report-root", type=Path, help="Generate an optional Markdown report only when requested")
    parser.add_argument("--service-port-base", type=int, default=9900)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--startup-poll-seconds", type=float, default=3.0)
    parser.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-trial-outputs", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-partial-success", action="store_true")
    parser.add_argument("--max-initial-gpu-memory-mb", type=int, default=1024,
                        help="Refuse occupied cards; negative disables this preflight.")
    parser.add_argument("--save-profile", type=Path,
                        help="Explicitly save selected successful configurations to this schema-v2 JSON.")
    from config.execution_profiles import add_runtime_arguments
    add_runtime_arguments(parser)
    args = parser.parse_args()
    if args.input_root is None:
        args.input_root = PDF_ROOT
        if args.pdf_list_file is None:
            args.pdf_list_file = DEFAULT_THROUGHPUT_MANIFEST
    return args


def build_subset(args: argparse.Namespace, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    if args.pdf_list_file is not None:
        loader = load_throughput_manifest if args.pdf_list_file.suffix == ".json" else load_pdf_paths_from_file
        pdf_paths = loader(args.pdf_list_file.resolve())
        source = str(args.pdf_list_file.resolve())
    else:
        pdf_paths = collect_pdf_paths(args.input_root.resolve())
        source = str(args.input_root.resolve())
    if not pdf_paths:
        raise RuntimeError("No PDFs selected for pressure benchmark.")

    run_dir.mkdir(parents=True, exist_ok=True)
    subset_path = run_dir / "subset_pdfs.txt"
    subset_path.write_text("\n".join(str(path.resolve()) for path in pdf_paths) + "\n", encoding="utf-8")
    meta = {
        "source": source,
        "sample_docs": len(pdf_paths),
        "page_count": sum(get_pdf_page_count(path) for path in pdf_paths),
        "pdfs": [str(path.resolve()) for path in pdf_paths],
    }
    return subset_path, meta


def initial_gpu_memory_mb(gpu_id: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        index, used = [item.strip() for item in line.split(",", maxsplit=1)]
        if int(index) == gpu_id:
            return int(used)
    raise RuntimeError(f"GPU {gpu_id} not found in nvidia-smi output")


def resolve_pressures(args: argparse.Namespace, method: str) -> list[int]:
    if args.pressures is not None:
        values = parse_int_csv(args.pressures)
        if not values or values != sorted(set(values)):
            raise ValueError("Pressures must be nonempty, unique and increasing")
    else:
        values = list(PRESSURE_SPECS[method].default_pressures)
    maximum = PRESSURE_SPECS[method].max_pressure
    values = [value for value in values if maximum is None or value <= maximum]
    if not values:
        raise ValueError(f"No pressures within {method}'s limit of {maximum}")
    return values




def build_trial(*, args, method, pressure, gpu_id, trial_index, subset_path, subset_meta) -> PressureTrial:
    from config.execution_profiles import resolve_runtime_defaults, runtime_cli_args
    effective = argparse.Namespace(**vars(args))
    resolve_runtime_defaults(effective, method)
    outer, _, mapping, docs = pressure_mapping(
        method, pressure, unlimited_request_mode=args.unlimited_ocr_request_mode
    )
    if docs is None:
        docs = int(subset_meta["sample_docs"])
    run_name = f"{args.run_name}_{method}_p{pressure:03d}"
    service_port = args.service_port_base + trial_index * 2
    command = [
        sys.executable, str(SINGLE_GPU_RUNNER), "--method", method, "--gpu-id", str(gpu_id),
        "--pressure", str(pressure), "--input-root", str(args.input_root),
        "--pdf-list-file", str(subset_path), "--output-root-base", str(args.output_root_base / args.run_name),
        "--log-root", str(args.log_root / run_name), "--run-name", run_name,
        "--service-port", str(service_port), "--startup-timeout", str(args.startup_timeout),
        "--startup-poll-seconds", str(args.startup_poll_seconds),
        "--monitor-interval-seconds", str(args.monitor_interval_seconds),
        *runtime_cli_args(effective, method),
    ]
    for enabled, flag in ((args.dry_run, "--dry-run"), (args.skip_existing, "--skip-existing"),
                          (args.allow_partial_success, "--allow-partial-success")):
        if enabled:
            command.append(flag)
    if not args.keep_trial_outputs:
        command.append("--discard-successful-outputs")
    mapping.update(method=method, pressure=pressure, gpu_id=gpu_id, service_port=service_port,
                   run_name=run_name, docs_per_job=docs, outer_concurrency=outer)
    return PressureTrial(method=method, pressure=pressure, gpu_id=gpu_id, service_port=service_port,
        run_name=run_name, command=command, mapping=mapping,
        aggregate_path=args.log_root / run_name / "aggregate_summary.json",
        log_path=args.log_root / f"{run_name}.log")


def worker_loop(
    trial_queue: queue.Queue[PressureTrial],
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    *,
    dry_run: bool,
    stop_on_failure: bool,
) -> None:
    while True:
        try:
            trial = trial_queue.get_nowait()
        except queue.Empty:
            return
        trial.log_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(
                f"[dry-run][gpu {trial.gpu_id}][{trial.method}][p={trial.pressure}] "
                + " ".join(trial.command)
            )
            payload = {
                "method": trial.method,
                "pressure": trial.pressure,
                "gpu_id": trial.gpu_id,
                "status": "dry_run",
                "mapping": trial.mapping,
                "command": trial.command,
            }
        else:
            print(f"[start][gpu {trial.gpu_id}] method={trial.method} pressure={trial.pressure}")
            proc = run_command(
                trial.command,
                trial.log_path,
                env_overrides={"CUDA_VISIBLE_DEVICES": str(trial.gpu_id)},
                wait=False,
            )
            return_code = proc.wait()
            aggregate: dict[str, Any] | None = None
            runner_error = None
            try:
                aggregate = json.loads(trial.aggregate_path.read_text(encoding="utf-8"))
                if not isinstance(aggregate, dict):
                    raise TypeError("Trial result must be a JSON object")
                if aggregate["status"] not in {"ok", "partial_ok", "failed"}:
                    raise ValueError("Trial result has an unknown status")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                runner_error = f"{type(exc).__name__}: {exc}"
            status = ("ok" if return_code == 0 and runner_error is None
                      and aggregate["status"] == "ok" else "failed")
            payload = {
                "method": trial.method,
                "pressure": trial.pressure,
                "gpu_id": trial.gpu_id,
                "status": status,
                "return_code": return_code,
                "mapping": trial.mapping,
                "aggregate_path": str(trial.aggregate_path),
                "log_path": str(trial.log_path),
                "aggregate": aggregate,
                "runner_error": runner_error,
            }
            print(f"[done][gpu {trial.gpu_id}] method={trial.method} pressure={trial.pressure} status={status}")
            if stop_on_failure and status not in {"ok", "partial_ok", "dry_run"}:
                with trial_queue.mutex:
                    trial_queue.queue.clear()
        with results_lock:
            results.append(payload)
        trial_queue.task_done()


def run_method_sweep(trials: list[PressureTrial], *, dry_run: bool) -> dict[str, Any]:
    """One method stays on one GPU; stop before launching any higher pressure."""
    rows = []
    selected = None
    previous_rate = None
    stop_reason = "exhausted_pressure_grid"
    for trial in trials:
        pending = queue.Queue()
        pending.put(trial)
        current = []
        try:
            worker_loop(pending, current, threading.Lock(), dry_run=dry_run, stop_on_failure=True)
        except Exception as exc:
            current = [{"method": trial.method, "pressure": trial.pressure, "gpu_id": trial.gpu_id,
                        "status": "failed", "log_path": str(trial.log_path), "mapping": trial.mapping,
                        "runner_error": f"{type(exc).__name__}: {exc}", "aggregate": None}]
        row = current[0]
        rows.append(row)
        if dry_run:
            continue
        if row["status"] != "ok":
            stop_reason = "execution_failure"
            break
        try:
            summary = row["aggregate"]["best_stable_config"]
            rate = summary["processing_pages_per_second"]
            if (summary["status"] != "ok" or summary["failed_doc_count"] != 0
                    or summary["unprocessed_doc_count"] != 0
                    or type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0):
                raise ValueError("Stable trial requires complete execution and a positive measured processing rate")
        except (KeyError, TypeError, ValueError) as exc:
            row.update(status="failed", runner_error=f"{type(exc).__name__}: {exc}")
            stop_reason = "invalid_trial_summary"
            break
        if previous_rate is not None:
            gain = (rate - previous_rate) / previous_rate
            row["gain_vs_previous_accepted"] = gain
            if gain < 0.05 and not math.isclose(gain, 0.05, rel_tol=1e-12, abs_tol=1e-12):
                stop_reason = "throughput_improvement_below_5_percent"
                break
        selected, previous_rate = row, rate
    if not dry_run and stop_reason == "exhausted_pressure_grid":
        maximum = PRESSURE_SPECS[trials[0].method].max_pressure
        if maximum is not None and rows[-1]["pressure"] == maximum:
            stop_reason = "method_limit"
    return {"method": trials[0].method, "gpu_id": trials[0].gpu_id, "results": rows,
            "selected": selected, "stop_reason": "dry_run" if dry_run else stop_reason,
            "status": "failed" if any(r["status"] == "failed" for r in rows)
                      else "dry_run" if dry_run else "completed"}


def method_worker_loop(pending, outcomes, lock, *, dry_run):
    while True:
        try:
            trials = pending.get_nowait()
        except queue.Empty:
            return
        outcome = run_method_sweep(trials, dry_run=dry_run)
        with lock:
            outcomes.append(outcome)
        pending.task_done()


def write_report(report_dir: Path, run_name: str, payload: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pressure_benchmark_summary.md"
    lines = [
        f"# Pressure Benchmark Summary: {run_name}",
        "",
        f"- dry run: `{payload['dry_run']}`",
        f"- methods: `{', '.join(payload['methods'])}`",
        f"- pressures: `{', '.join(str(item) for item in payload['pressures'])}`",
        f"- gpus: `{', '.join(str(item) for item in payload['gpus'])}`",
        f"- subset: `{payload['subset_path']}`",
        "",
        "| Method | Pressure | GPU | Status | Processing pages/s | Mapping |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in sorted(payload["results"], key=lambda item: (item["method"], int(item["pressure"]))):
        aggregate = row.get("aggregate") if isinstance(row.get("aggregate"), dict) else {}
        best = aggregate.get("best_stable_config") if isinstance(aggregate, dict) else None
        pages_per_second = ""
        if isinstance(best, dict):
            value = best.get("processing_pages_per_second")
            if isinstance(value, (int, float)):
                pages_per_second = f"{value:.6f}"
        lines.append(
            "| {method} | {pressure} | {gpu_id} | {status} | {pps} | `{mapping}` |".format(
                method=row["method"],
                pressure=row["pressure"],
                gpu_id=row["gpu_id"],
                status=row["status"],
                pps=pages_per_second,
                mapping=json.dumps(row["mapping"], ensure_ascii=False, sort_keys=True),
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    methods = parse_csv(args.methods)
    if not methods or len(methods) != len(set(methods)):
        raise SystemExit("Methods must be nonempty and unique.")
    unknown = [method for method in methods if method not in DEFAULT_PRESSURE_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    selected_monkey_variants = [method for method in methods if method.startswith("monkeyocr_")]
    if args.monkey_models_dir and len(selected_monkey_variants) > 1:
        raise SystemExit(
            "--monkey-models-dir cannot override multiple MonkeyOCR variants in one run."
        )
    pressures_by_method = {method: resolve_pressures(args, method) for method in methods}
    pressures = sorted({value for values in pressures_by_method.values() for value in values})
    gpus = parse_gpu_csv(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided.")
    if len(gpus) != len(set(gpus)):
        raise SystemExit("GPU ids must be unique; same-card trials cannot overlap.")
    if not args.dry_run and args.max_initial_gpu_memory_mb >= 0:
        observed = {gpu_id: initial_gpu_memory_mb(gpu_id) for gpu_id in gpus}
        busy = {
            gpu_id: used
            for gpu_id, used in observed.items()
            if used > args.max_initial_gpu_memory_mb
        }
        if busy:
            details = ", ".join(f"gpu {gpu}: {used} MiB" for gpu, used in busy.items())
            raise SystemExit(
                "Selected GPU is already occupied; no trial was started: " + details
            )

    if args.dry_run:
        if args.experiment_root.is_relative_to(EXPERIMENT_ROOT):
            args.experiment_root = PROJECT_ROOT / "tmp" / args.experiment_root.relative_to(EXPERIMENT_ROOT)
    run_dir = args.experiment_root.resolve() / args.run_name
    args.log_root = args.log_root.resolve() if args.log_root is not None else run_dir / "logs"
    if (run_dir / "results.json").exists() or (run_dir / "metadata" / "run_status.json").exists():
        raise SystemExit(f"Run already exists; choose a new --run-name: {run_dir}")
    subset_path, subset_meta = build_subset(args, PROJECT_ROOT / "tmp" / "tuning" / args.run_name)

    trials: list[PressureTrial] = []
    trial_index = 0
    for method_index, method in enumerate(methods):
        for pressure in pressures_by_method[method]:
            gpu_id = gpus[method_index % len(gpus)]
            trials.append(
                build_trial(
                    args=args,
                    method=method,
                    pressure=pressure,
                    gpu_id=gpu_id,
                    trial_index=trial_index,
                    subset_path=subset_path,
                    subset_meta=subset_meta,
                )
            )
            trial_index += 1

    plan = {
        "run_name": args.run_name,
        "dry_run": args.dry_run,
        "methods": methods,
        "pressures": pressures,
        "pressures_by_method": pressures_by_method,
        "gpus": gpus,
        "subset_path": str(subset_path),
        "subset": subset_meta,
        "trials": [
            {
                "method": trial.method,
                "pressure": trial.pressure,
                "gpu_id": trial.gpu_id,
                "service_port": trial.service_port,
                "run_name": trial.run_name,
                "mapping": trial.mapping,
                "command": trial.command,
                "aggregate_path": str(trial.aggregate_path),
                "log_path": str(trial.log_path),
            }
            for trial in trials
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    status_path = metadata / "run_status.json"
    status_path.write_text(json.dumps({"status": "running", "dry_run": args.dry_run}), encoding="utf-8")

    trial_queues = {gpu_id: queue.Queue() for gpu_id in gpus}
    for method in methods:
        method_trials = [trial for trial in trials if trial.method == method]
        trial_queues[method_trials[0].gpu_id].put(method_trials)

    outcomes: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=method_worker_loop,
            args=(trial_queues[gpu_id], outcomes, results_lock),
            kwargs={"dry_run": args.dry_run},
            daemon=False,
        )
        for gpu_id in gpus
    ]
    for thread in threads:
        thread.start()
        if args.execution_policy == "serial":
            thread.join()
    for thread in threads:
        thread.join()

    results = [row for outcome in outcomes for row in outcome["results"]]
    payload = {
        **plan,
        "execution_policy": args.execution_policy,
        "method_outcomes": [
            {"method": outcome["method"], "gpu_id": outcome["gpu_id"],
             "status": outcome["status"], "stop_reason": outcome["stop_reason"],
             "selected_pressure": outcome["selected"]["pressure"] if outcome["selected"] else None}
            for outcome in outcomes
        ],
        "results": sorted(results, key=lambda item: (item["method"], int(item["pressure"]))),
    }
    summary_path = run_dir / "results.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report_root is not None:
        write_report(args.report_root.resolve() / args.run_name, args.run_name, payload)
    print(f"[done] pressure summary: {summary_path}")

    failed = [item for item in results if item["status"] not in {"ok", "partial_ok", "dry_run"}]
    incomplete = len(outcomes) != len(methods)
    status = "failed" if failed or incomplete else "dry_run" if args.dry_run else "partial" if any(item["status"] == "partial_ok" for item in results) else "completed"
    status_path.write_text(json.dumps({"status": status, "dry_run": args.dry_run}), encoding="utf-8")
    if args.save_profile is not None and not args.dry_run:
        from config.execution_profiles import save_selected_profile
        try:
            for outcome in outcomes:
                if outcome["selected"] is not None:
                    save_selected_profile(args.save_profile, outcome["method"],
                        outcome["selected"]["aggregate"]["best_stable_config"])
        except Exception as exc:
            status_path.write_text(json.dumps({"status": "failed", "measurement_status": status,
                "profile_save_error": f"{type(exc).__name__}: {exc}",
                "result": str(summary_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
    if failed or incomplete:
        raise SystemExit(1)
    quality_outputs_retained = any(
        row["method"] == "unlimited_ocr_sglang" and row.get("aggregate") is not None
        and any(trial.get("quality_warning_doc_count", 0) for trial in row["aggregate"]["results"])
        for row in results
    )
    if not args.dry_run and not args.keep_trial_outputs and not quality_outputs_retained:
        temporary_run = PROJECT_ROOT / "tmp" / "tuning" / args.run_name
        if not run_dir.is_relative_to(temporary_run):
            shutil.rmtree(temporary_run, ignore_errors=True)


if __name__ == "__main__":
    main()
