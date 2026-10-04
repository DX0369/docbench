#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence


from common import document_completed, get_doc_output_dir, load_pdf_paths_from_file


def collect_pdf_paths(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob("*.pdf") if path.is_file())


def chunk_pdf_paths(pdf_paths: Sequence[Path], docs_per_job: int) -> deque[list[Path]]:
    if docs_per_job <= 0:
        raise ValueError("docs_per_job must be positive")
    jobs: deque[list[Path]] = deque()
    for index in range(0, len(pdf_paths), docs_per_job):
        jobs.append(list(pdf_paths[index : index + docs_per_job]))
    return jobs


def ensure_queue_layout(queue_dir: Path) -> tuple[Path, Path, Path]:
    pending_dir = queue_dir / "pending"
    running_dir = queue_dir / "running"
    done_dir = queue_dir / "done"
    pending_dir.mkdir(parents=True, exist_ok=True)
    running_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    return pending_dir, running_dir, done_dir


def write_pending_job_list(job_dir: Path, job_index: int, pdf_paths: Sequence[Path]) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / f"job_{job_index:06d}.txt"
    job_path.write_text("".join(f"{path}\n" for path in pdf_paths), encoding="utf-8")
    return job_path


def write_slot_job_list(job_dir: Path, slot_id: str, job_index: int, pdf_paths: Sequence[Path]) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / f"{slot_id}_job{job_index:04d}.txt"
    job_path.write_text("".join(f"{path}\n" for path in pdf_paths), encoding="utf-8")
    return job_path


def claim_next_job(
    pending_dir: Path,
    running_dir: Path,
    worker_id: str,
) -> Path | None:
    for pending_path in sorted(pending_dir.glob("job_*.txt")):
        claimed_path = running_dir / f"{pending_path.stem}__{worker_id}.txt"
        try:
            os.replace(pending_path, claimed_path)
            return claimed_path
        except FileNotFoundError:
            continue
    return None


def collect_done_statuses(done_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for status_path in sorted(done_dir.glob("*.json")):
        payloads.append(json.loads(status_path.read_text(encoding="utf-8")))
    return payloads


def collect_failed_docs(job_statuses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for payload in job_statuses:
        if not bool(payload.get("failed")):
            continue
        failed_docs = payload.get("failed_docs")
        if isinstance(failed_docs, list) and failed_docs:
            for item in failed_docs:
                if not isinstance(item, dict):
                    continue
                pdf_path = str(item.get("pdf_path", "")).strip()
                if not pdf_path:
                    continue
                error_json = item.get("error_json")
                reason = item.get("reason")
                key = (
                    pdf_path,
                    str(error_json) if error_json else None,
                    str(reason) if reason else None,
                )
                if key in seen:
                    continue
                seen.add(key)
                collected.append(
                    {
                        "pdf_path": pdf_path,
                        "error_json": str(error_json) if error_json else None,
                        "reason": str(reason) if reason else None,
                    }
                )
            continue

        # Process failure alone does not identify failed input documents.
    return collected


def write_job_status(
    done_dir: Path,
    *,
    claimed_path: Path,
    worker_id: str,
    failed: bool,
    job_pdf_paths: Iterable[str],
    extra: dict[str, Any] | None = None,
) -> None:
    done_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "worker_id": worker_id,
        "job_file": str(claimed_path),
        "failed": failed,
        "job_pdf_paths": list(job_pdf_paths),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    done_path = done_dir / f"{claimed_path.stem}.json"
    done_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_job_failures(
    *,
    input_root: Path,
    output_root: Path,
    pdf_paths: list[Path],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        doc_output_dir = get_doc_output_dir(input_root, output_root, pdf_path)
        document_md = doc_output_dir / "document.md"
        error_json = doc_output_dir / "error.json"
        if error_json.exists():
            record_error = None
            try:
                error = json.loads(error_json.read_text(encoding="utf-8"))
                if not isinstance(error, dict):
                    raise ValueError("error.json must contain an object")
            except (OSError, ValueError) as exc:
                error = {}
                record_error = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "pdf_path": str(pdf_path),
                    "reason": "invalid_error_json" if record_error else "error_json_present",
                    "record_error": record_error,
                    "status": "failed",
                    "runtime_unusable": bool(error.get("runtime_unusable", False)),
                    "error_json": str(error_json),
                }
            )
        elif not document_completed(doc_output_dir):
            failures.append(
                {
                    "pdf_path": str(pdf_path),
                    "status": "unprocessed",
                    "reason": "incomplete_output" if document_md.exists() else "missing_document_md",
                }
            )
    return failures


def collect_unprocessed_docs(job_statuses: Sequence[dict[str, Any]], queue_dir: Path) -> list[dict[str, Any]]:
    """Unresolved work is separate from documents with explicit error records."""
    items = [item for job in job_statuses for item in job.get("unprocessed_docs", [])]
    for directory in ("pending", "running"):
        for job_file in sorted((queue_dir / directory).glob("*.txt")):
            items.extend(
                {"pdf_path": str(path), "status": "unprocessed", "reason": directory,
                 "job_file": str(job_file), "worker_id": None}
                for path in load_pdf_paths_from_file(job_file)
            )
    seen = set()
    result = []
    for item in items:
        if item["pdf_path"] not in seen:
            seen.add(item["pdf_path"])
            result.append(item)
    return result
