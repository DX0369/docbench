#!/usr/bin/env python3
from __future__ import annotations

import traceback
from contextlib import contextmanager
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from common import load_pdf_paths_from_file
from execution.queue import claim_next_job, collect_job_failures, ensure_queue_layout, write_job_status

QueueHandler = Callable[[Path, list[Path]], bool | tuple[bool, dict[str, Any]]]


def write_timing_state(path: Path, payload: dict[str, Any]) -> None:
    """Single-writer atomic state, under the existing per-run queue directory."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.replace(path)


@contextmanager
def queue_processing_timing(queue_dir: Path, worker_id: str):
    control = queue_dir / "timing" / "control.json"
    if not control.is_file():
        yield
        return
    parent_pid = json.loads(control.read_text())["parent_pid"]
    state_path = control.parent / f"{worker_id}.json"
    state = {"status": "ready", "pid": os.getpid(), "ready_at": time.perf_counter()}
    write_timing_state(state_path, state)
    # Called only AFTER the caller's model initialization, BEFORE claiming work.
    while not (control.parent / "start.json").is_file():
        os.kill(parent_pid, 0)  # Do not wait forever if the owning launcher exits.
        time.sleep(0.05)
    try:
        yield
        state["status"] = "done"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_at"] = time.perf_counter()
        write_timing_state(state_path, state)


def start_queue_processing(queue_dir: Path, workers, *, timeout: float, service=None, services=()) -> float:
    """Wait for every initialized worker, then release the single start barrier."""
    deadline = time.monotonic() + timeout
    root = queue_dir / "timing"
    watched_services = tuple(services) + ((service,) if service is not None else ())
    while True:
        if any(proc.poll() is not None for proc in watched_services):
            raise RuntimeError("Service exited before workers were ready")
        ready = True
        for worker_id, _log, process in workers:
            if process.poll() is not None:
                raise RuntimeError(f"Worker {worker_id} exited before processing started")
            state = root / f"{worker_id}.json"
            ready = ready and state.is_file() and json.loads(state.read_text())["status"] == "ready"
        if ready:
            started = time.perf_counter()
            write_timing_state(root / "start.json", {"started_at": started})
            return started
        if time.monotonic() >= deadline:
            raise TimeoutError("Workers did not finish model initialization before startup timeout")
        time.sleep(0.05)


def queue_processing_seconds(queue_dir: Path, worker_ids: list[str], started: float) -> float:
    states = [json.loads((queue_dir / "timing" / f"{worker_id}.json").read_text()) for worker_id in worker_ids]
    if any(state["status"] not in ("done", "failed") for state in states):
        raise ValueError("Missing worker processing end; cannot report model-excluded throughput")
    return max(state["finished_at"] for state in states) - started


def _requeue_claimed_job(claimed_path: Path, pending_dir: Path) -> None:
    original_stem = claimed_path.stem.split("__", 1)[0]
    target_path = pending_dir / f"{original_stem}.txt"
    suffix = 1
    while target_path.exists():
        target_path = pending_dir / f"{original_stem}_retry{suffix}.txt"
        suffix += 1
    claimed_path.replace(target_path)


def run_pdf_queue_worker(
    *,
    queue_dir: Path,
    worker_id: str,
    fail_fast: bool,
    process_job: QueueHandler,
    allow_partial_success: bool = False,
    input_root: Path | None = None,
    output_root: Path | None = None,
) -> bool:
    with queue_processing_timing(queue_dir, worker_id):
        return _consume_pdf_queue(queue_dir=queue_dir, worker_id=worker_id, fail_fast=fail_fast,
                                  process_job=process_job, allow_partial_success=allow_partial_success,
                                  input_root=input_root, output_root=output_root)


def _consume_pdf_queue(
    *, queue_dir, worker_id, fail_fast, process_job, allow_partial_success,
    input_root, output_root,
) -> bool:
    pending_dir, running_dir, done_dir = ensure_queue_layout(queue_dir)
    control = queue_dir / "timing" / "control.json"
    # Coverage validation is enabled by the throughput controller's optional
    # method field. Other queue owners (including Popo) retain their protocol.
    method = json.loads(control.read_text()).get("method") if control.is_file() else None

    had_failure = False
    while True:
        claimed_path = claim_next_job(pending_dir, running_dir, worker_id)
        if claimed_path is None:
            break
        job_failed = False
        pdf_paths: list[Path] = []
        error_type: str | None = None
        error_summary: str | None = None
        extra_status: dict[str, Any] = {}
        requeued_job = False
        try:
            pdf_paths = load_pdf_paths_from_file(claimed_path)
            job_result = process_job(claimed_path, pdf_paths)
            if isinstance(job_result, tuple):
                job_failed = bool(job_result[0])
                extra_status.update(job_result[1])
            else:
                job_failed = bool(job_result)
            had_failure = had_failure or job_failed
        except BaseException as exc:  # noqa: BLE001
            job_failed = True
            had_failure = True
            error_type = type(exc).__name__
            error_summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if bool(getattr(exc, "requeue_claimed_job", False)):
                requeued_job = True
            else:
                raise
        finally:
            if input_root is not None and output_root is not None:
                if method is not None:
                    from validation.validate_outputs import validate_completed_page_counts
                    counts, _ = validate_completed_page_counts(method, input_root, output_root, pdf_paths)
                    extra_status["successful_input_page_counts"] = {str(path): count for path, count in counts.items()}
                outcomes = collect_job_failures(input_root=input_root, output_root=output_root, pdf_paths=pdf_paths)
                extra_status["failed_docs"] = [item for item in outcomes if item["status"] == "failed"]
                extra_status["unprocessed_docs"] = [item for item in outcomes if item["status"] == "unprocessed"]
                extra_status["failed_doc_count"] = len(extra_status["failed_docs"])
                extra_status["unprocessed_doc_count"] = len(extra_status["unprocessed_docs"])
                extra_status["successful_doc_count"] = len(pdf_paths) - len(outcomes)
                job_failed = job_failed or bool(outcomes)
                had_failure = had_failure or job_failed
                extra_status["stop_worker"] = bool(extra_status.get("stop_worker")) or any(
                    item.get("runtime_unusable") or item.get("record_error") or item["status"] == "unprocessed"
                    for item in outcomes
                )
            if claimed_path.exists():
                if requeued_job:
                    extra_status["requeued"] = bool(extra_status["unprocessed_docs"]) if input_root is not None and output_root is not None else True
                if error_type is not None:
                    extra_status["error_type"] = error_type
                if error_summary is not None:
                    extra_status["error_summary"] = error_summary
                write_job_status(
                    done_dir,
                    claimed_path=claimed_path,
                    worker_id=worker_id,
                    failed=job_failed,
                    job_pdf_paths=[str(path) for path in pdf_paths],
                    extra=extra_status,
                )
                if requeued_job and extra_status["requeued"]:
                    if input_root is not None and output_root is not None:
                        claimed_path.write_text("".join(item["pdf_path"] + "\n" for item in extra_status["unprocessed_docs"]), encoding="utf-8")
                    _requeue_claimed_job(claimed_path, pending_dir)
                else:
                    claimed_path.unlink(missing_ok=True)
        if requeued_job:
            break
        if extra_status.get("stop_worker"):
            break
        if fail_fast and job_failed:
            break

    return bool(had_failure and not allow_partial_success)
