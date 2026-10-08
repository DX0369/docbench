"""TeleOCR's resident official vLLM async runtime.

The upstream TeleOCR functions are used unchanged.  This module only owns the
runtime lifetime and supplies the predictor to ``aio_doc_analyze`` so that a
single AsyncLLM and event loop are reused for every queued document.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class TeleOCRRuntime:
    def __init__(self, args: Any) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.args = args
        self.predictor = self._load_predictor()

    def _resident_tasks(self) -> set[asyncio.Task[Any]]:
        """Return vLLM tasks that must survive a document-level exception."""
        engine = getattr(getattr(self.predictor, "client", None), "vllm_async_llm", None)
        objects = [getattr(engine, "output_handler", None)]
        resources = getattr(getattr(engine, "engine_core", None), "resources", None)
        objects.extend([
            getattr(resources, "output_queue_task", None),
            getattr(resources, "stats_update_task", None),
        ])
        return {task for task in objects if isinstance(task, asyncio.Task)}

    def _cancel_document_tasks(self, baseline: set[asyncio.Task[Any]]) -> None:
        pending = asyncio.all_tasks(self.loop)
        transient = pending - baseline - self._resident_tasks()
        transient.discard(asyncio.current_task(self.loop))
        if not transient:
            return
        for task in transient:
            task.cancel()
        self.loop.run_until_complete(
            asyncio.gather(*transient, return_exceptions=True)
        )

    def _load_predictor(self):
        import TeleOCR.config as config

        config.model_path = str(self.args.model_path)
        config.BACKEND = self.args.backend
        config.LAYOUT_MODE = self.args.layout_mode
        config.MAX_MODEL_LEN = self.args.max_model_len
        config.GPU_MEMORY_UTILIZATION = self.args.gpu_memory_utilization
        config.PDF_TOOLS = self.args.pdf_tools
        config.PDF_TOOLS_WORKER_MAX_NUM = self.args.pdf_tools_workers

        if config.BACKEND != "vllm-async-engine":
            raise ValueError("TeleOCR benchmark adapter requires vllm-async-engine")

        # Importing the plugin is explicit so a missing editable plugin fails at
        # readiness instead of at the first generation request.
        import TeleOCR_vllm  # noqa: F401
        from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE

        predictor = TeleOCRMODEL_SERVICE.get_model(
            config.BACKEND,
            config.model_path,
            None,
            gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
            max_model_len=config.MAX_MODEL_LEN,
        )
        # TeleOCR's upstream factory uses 100 as its default and does not expose
        # this value as a factory argument.  Both semaphore layers are updated:
        # the outer one is used by TeleOCR's two-step pipeline and the inner one
        # is the fallback used by direct client calls.
        predictor.max_concurrency = self.args.max_concurrency
        predictor.client.max_concurrency = self.args.max_concurrency
        # Let AsyncLLM create its output and IPC tasks while this persistent
        # loop is current, before queue_processing_timing signals readiness.
        self.loop.run_until_complete(asyncio.sleep(0))
        return predictor

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        from TeleOCR.src.vlm_analyze import aio_doc_analyze
        from TeleOCR.tools.pdf_image_tools import convert_pdf_bytes_to_bytes
        from TeleOCR.tools.read_file import read_fn

        # Match the official aio_do_parse route's PDF normalization while
        # bypassing its image/layout-PDF writer.
        pdf_bytes = convert_pdf_bytes_to_bytes(read_fn(pdf_path), None)
        baseline = asyncio.all_tasks(self.loop)
        try:
            middle_json = self.loop.run_until_complete(
                aio_doc_analyze(
                    pdf_bytes,
                    predictor=self.predictor,
                    image_writer=None,
                )
            )
            engine = getattr(getattr(self.predictor, "client", None), "vllm_async_llm", None)
            if engine is not None and bool(getattr(engine, "errored", False)):
                error = RuntimeError("TeleOCR AsyncLLM engine became unhealthy")
                error.gdb_runtime_unusable = True
                raise error
            return middle_json
        except BaseException:
            self._cancel_document_tasks(baseline)
            raise

    def close(self) -> list[str]:
        """Shutdown the vLLM engine after all outputs have been written."""
        warnings: list[str] = []
        if self.loop.is_closed():
            return warnings
        try:
            engine = getattr(getattr(self.predictor, "client", None), "vllm_async_llm", None)
            if engine is not None:
                try:
                    engine.shutdown()
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"AsyncLLM shutdown: {type(exc).__name__}: {exc}")
            try:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"event loop shutdown: {type(exc).__name__}: {exc}")
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)
        return warnings


def load_runtime(args: Any) -> TeleOCRRuntime:
    return TeleOCRRuntime(args)
