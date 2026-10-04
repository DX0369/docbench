"""Bounded page tasks across PDFs; methods own inference and document export."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time

from common import error_context, runtime_unusable


def run_page_pool(pdf_paths, *, concurrency, page_count, should_skip, run_page,
                  export_document, record_error, fail_fast=False):
    """Record every observed error; callbacks must retain earlier failures.

    Error callbacks run serially in the collecting thread. A failed document
    gets no new pages, but its already submitted tasks are still collected.
    Fatal errors and fail_fast retain their immediate-abort behavior.
    """
    if concurrency < 1:
        raise ValueError("page concurrency must be positive")
    states = {}
    failed = False
    for pdf in pdf_paths:
        if should_skip(pdf):
            continue
        try:
            with error_context("inspect_pdf"):
                count = page_count(pdf)
                if count < 1:
                    raise ValueError(f"Invalid page count: {pdf}: {count}")
            states[pdf] = {"count": count, "results": {}, "failed": False, "started": None}
        except Exception as exc:
            record_error(pdf, exc)
            failed = True
            if fail_fast or runtime_unusable(exc):
                raise

    def jobs():
        for pdf, state in states.items():
            for page in range(1, state["count"] + 1):
                if state["failed"]:
                    break
                yield pdf, page

    iterator = iter(jobs())
    pending = {}
    exhausted = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        while pending or not exhausted:
            while len(pending) < concurrency and not exhausted:
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                pdf, page = item
                if states[pdf]["started"] is None:
                    states[pdf]["started"] = time.perf_counter()
                pending[pool.submit(run_page, pdf, page)] = (pdf, page)
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pdf, page = pending.pop(future)
                state = states[pdf]
                try:
                    value = future.result()
                    if not state["failed"]:
                        state["results"][page] = value
                except Exception as exc:
                    record_error(pdf, exc)
                    state["failed"] = True
                    state["results"].clear()
                    failed = True
                    if fail_fast or runtime_unusable(exc):
                        for other in pending:
                            other.cancel()
                        raise
                if not state["failed"] and len(state["results"]) == state["count"]:
                    try:
                        with error_context("export"):
                            ordered = [state["results"][p] for p in range(1, state["count"] + 1)]
                            export_document(pdf, ordered, state["started"])
                        state["results"].clear()
                        print(f"[done] {pdf}", flush=True)
                    except Exception as exc:
                        record_error(pdf, exc)
                        state["failed"] = True
                        failed = True
                        if fail_fast or runtime_unusable(exc):
                            raise
    return failed
