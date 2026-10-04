"""Bounded CPU-only PDF rendering; all model calls remain on the consumer."""
from __future__ import annotations

from collections import deque
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
import time
from typing import Any

from common import error_context
from execution.trial import get_pdf_page_count

PREFETCH_BATCHES = 2
PREFETCH_MAX_BYTES = 512 * 1024 * 1024
INPUT_PIPELINE = "bounded_page_prefetch_v1"


@dataclass
class Page:
    pdf: Path
    index: int
    count: int
    image: Any
    started: float


@dataclass
class RenderError:
    pdf: Path
    exception: Exception


def render_pages(reader, pdf: Path):
    """Use the existing official reader, including its DPI and pixel protection."""
    started = time.perf_counter()
    with error_context("inspect_pdf"):
        expected = get_pdf_page_count(pdf)
        if expected <= 0:
            raise ValueError(f"PDF has no pages: {pdf}")
    with error_context("render_pdf"):
        doc = reader.load(str(pdf))
        try:
            count = len(doc)
            if count != expected:
                raise ValueError(f"Native PDF page count {count} differs from pdfinfo {expected}")
            # read(path) nests PDFium's non-reentrant lock in this SDK version.
            # Match the official sampler: load(path), then read(document).
            with closing(reader.read(doc)) as images:
                for index in range(count):
                    with error_context("render_pdf", index + 1):
                        try:
                            image = next(images)
                        except StopIteration as exc:
                            raise ValueError("Official PDF renderer ended before the last page") from exc
                    yield Page(pdf, index, count, image, started)
                if next(images, None) is not None:
                    raise ValueError("Official PDF renderer returned more pages than the PDF")
        finally:
            doc.close()


class PagePrefetch:
    """One producer, two batches of queued pages, and byte-budget backpressure.

    An oversized page may occupy an otherwise image-empty queue by itself.
    The current consumer batch and one producer-side page are outside the queue
    budget; this is not a total process-memory limit.
    """

    def __init__(self, pdfs, source, batch_size, *, max_bytes=PREFETCH_MAX_BYTES):
        if batch_size <= 0 or max_bytes <= 0:
            raise ValueError("Prefetch batch size and memory budget must be positive")
        self.pdfs, self.source = pdfs, source
        self.max_items, self.max_bytes = PREFETCH_BATCHES * batch_size, max_bytes
        self.items = deque()
        self.condition = Condition()
        self.cancelled = self.finished = False
        self.crash = None
        self.skipped = set()
        self.queued_bytes = 0
        self.stats = {"rendered_pages": 0, "peak_queue_pages": 0, "peak_queue_bytes": 0,
                      "consumer_wait_seconds": 0.0}
        self.thread = Thread(target=self._produce, name="pp-pdf-prefetch", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        with self.condition:
            self.cancelled = True
            self.items.clear()
            self.queued_bytes = 0
            self.condition.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            exc = RuntimeError("PP PDF producer did not stop; this worker must exit")
            exc.gdb_runtime_unusable = True
            raise exc

    def skip(self, pdf):
        with self.condition:
            self.skipped.add(pdf)
            self.items = deque(item for item in self.items if item.pdf != pdf)
            self.queued_bytes = sum(item.image.nbytes for item in self.items if isinstance(item, Page))
            self.condition.notify_all()

    def _put(self, item):
        size = item.image.nbytes if isinstance(item, Page) else 0
        with self.condition:
            while not self.cancelled and item.pdf not in self.skipped:
                full = len(self.items) >= self.max_items
                over_budget = size and self.queued_bytes and self.queued_bytes + size > self.max_bytes
                if not full and not over_budget:
                    self.items.append(item)
                    self.queued_bytes += size
                    self.stats["peak_queue_pages"] = max(self.stats["peak_queue_pages"],
                        sum(isinstance(value, Page) for value in self.items))
                    self.stats["peak_queue_bytes"] = max(self.stats["peak_queue_bytes"], self.queued_bytes)
                    self.condition.notify_all()
                    return True
                self.condition.wait()
            return False

    def _produce(self):
        try:
            for pdf in self.pdfs:
                with self.condition:
                    if self.cancelled:
                        break
                    if pdf in self.skipped:
                        continue
                try:
                    with closing(self.source(pdf)) as pages:
                        for page in pages:
                            with self.condition:
                                self.stats["rendered_pages"] += 1
                            if not self._put(page):
                                break
                except Exception as exc:
                    if not self._put(RenderError(pdf, exc)):
                        continue
        except BaseException as exc:
            self.crash = exc
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def get(self):
        with self.condition:
            while not self.items and not self.finished:
                start = time.perf_counter()
                self.condition.wait()
                self.stats["consumer_wait_seconds"] += time.perf_counter() - start
            if self.crash is not None:
                raise self.crash
            if not self.items:
                return None
            item = self.items.popleft()
            if isinstance(item, Page):
                self.queued_bytes -= item.image.nbytes
            self.condition.notify_all()
            return item


def native_pipeline(pipeline):
    outer = pipeline.paddlex_pipeline
    if outer.multi_device_inference:
        raise ValueError("PP page prefetch requires one device and one native pipeline")
    return outer._pipeline


@contextmanager
def prepared_batch(native, pages):
    """Feed a single official ImgBatch; preserve native source/page metadata."""
    from paddlex.inference.common.batch_sampler.image_batch_sampler import ImgBatch

    batch = ImgBatch()
    for page in pages:
        batch.append(page.image, str(page.pdf), page.index, page.count)
    paths = list(dict.fromkeys(batch.input_paths))
    original = native.batch_sampler

    def sample(inputs):
        if inputs != paths:
            raise ValueError("Prepared PP batch was requested with different PDF paths")
        yield batch

    native.batch_sampler = sample
    try:
        yield paths
    finally:
        native.batch_sampler = original
