#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_ROOT = THIS_DIR.parents[1]

if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from execution.workers import queue_processing_timing, run_pdf_queue_worker, write_timing_state
from methods.marker.pipeline import configure_marker_runtime, load_marker_runtime, process_pdf_paths

WORKER_EXIT_TIMEOUT = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marker shared-model supervisor with persistent document workers."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=str, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Document processes sharing the parent models.")
    parser.add_argument("--torch-device", type=str)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def create_shared_models(args):
    # The native CLI sets this before importing its model stack.
    os.environ["IN_STREAMLIT"] = "true"
    configure_marker_runtime(args)
    from marker.models import create_model_dict

    models = create_model_dict()
    for predictor in models.values():
        predictor.model.share_memory()
    return models


def _document_worker(args, models, worker_id, start, connection):
    try:
        runtime = load_marker_runtime(args, artifact_dict=models)
        connection.send(("ready", None))
        start.wait()
        failed = run_pdf_queue_worker(
            queue_dir=args.queue_dir, worker_id=worker_id,
            input_root=args.input_root, output_root=args.output_root,
            fail_fast=args.fail_fast,
            process_job=lambda _path, pdfs: process_pdf_paths(args, pdfs, runtime=runtime),
        )
        # Report before process/model teardown; ordinary document errors are not crashes.
        connection.send(("done", {"failed": failed, "finished_at": time.perf_counter()}))
    except BaseException as exc:
        connection.send(("failed", f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        connection.close()


def _wait_for_phase(children, phase):
    pending = {name: (process, connection) for name, process, connection in children}
    results = []
    while pending:
        for name, (process, connection) in list(pending.items()):
            if connection.poll():
                try:
                    received, payload = connection.recv()
                except EOFError as exc:
                    raise RuntimeError(f"Marker worker {name} closed before {phase}") from exc
                if received != phase:
                    raise RuntimeError(f"Marker worker {name}: {received}: {payload}")
                results.append(payload)
                del pending[name]
            elif process.exitcode is not None:
                raise RuntimeError(f"Marker worker {name} exited {process.exitcode} before {phase}")
        if phase == "ready":
            for name, process, _ in children:
                if process.exitcode is not None:
                    raise RuntimeError(f"Marker worker {name} exited during initialization")
        if pending:
            time.sleep(0.02)
    return results


def run_shared_workers(args, models):
    from torch.multiprocessing import get_context

    if type(args.workers) is not int or args.workers < 1:
        raise ValueError("Marker workers must be a positive integer")
    context = get_context("spawn")
    start = context.Event()
    children = []
    try:
        for index in range(args.workers):
            name = f"{args.worker_id}.marker{index:03d}"
            reader, writer = context.Pipe(duplex=False)
            process = context.Process(target=_document_worker, args=(args, models, name, start, writer))
            try:
                process.start()
            except BaseException:
                reader.close()
                raise
            finally:
                writer.close()
            children.append((name, process, reader))
        _wait_for_phase(children, "ready")
        with queue_processing_timing(args.queue_dir, args.worker_id):
            start.set()
            results = _wait_for_phase(children, "done")
        timing_path = args.queue_dir / "timing" / f"{args.worker_id}.json"
        if timing_path.is_file():
            state = json.loads(timing_path.read_text())
            state["finished_at"] = max(item["finished_at"] for item in results)
            write_timing_state(timing_path, state)
        for name, process, _ in children:
            process.join(timeout=WORKER_EXIT_TIMEOUT)
            if process.is_alive():
                raise RuntimeError(f"Marker worker {name} timed out during teardown")
            if process.exitcode != 0:
                raise RuntimeError(f"Marker worker {name} exited {process.exitcode} during teardown")
        return any(item["failed"] for item in results)
    except BaseException as exc:
        timing_path = args.queue_dir / "timing" / f"{args.worker_id}.json"
        if timing_path.is_file():
            state = json.loads(timing_path.read_text())
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_timing_state(timing_path, state)
        raise
    finally:
        # Keep the parent models alive until every CUDA IPC consumer is gone.
        for _, process, _ in children:
            if process.is_alive():
                process.terminate()
        for _, process, connection in children:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.queue_dir = args.queue_dir.resolve()

    models = create_shared_models(args)
    should_exit_failed = run_shared_workers(args, models)
    if should_exit_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
