"""One event loop for the lifetime of a synchronous MonkeyOCR pipeline."""
from __future__ import annotations

import asyncio
from typing import Any

from common import runtime_unusable

# Placeholder for a failed region. It survives the official text sanitizers, so
# the owning input PDF can be identified after the group's official analysis.
REGION_FAILED_PREFIX = "\x00GDB_REGION_FAILED:"


class AsyncRecognitionLoop:
    """Keep vLLM's output tasks alive between documents in one worker thread."""

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model
        self.loop = asyncio.new_event_loop()
        # Patched official switch: return per-region exceptions instead of raising the first one.
        chat_model.gdb_return_region_exceptions = True
        self.region_failures: list[str] = []

    def batch_inference(self, images: list, questions: list) -> list[str]:
        # Use the official async implementation unchanged, not its per-call
        # asyncio.run() bridge, which cancels the engine's background tasks.
        results = self.loop.run_until_complete(
            self.chat_model.async_batch_inference(images, questions)
        )
        for result in results:
            if isinstance(result, BaseException) and (
                    not isinstance(result, Exception) or runtime_unusable(result)):
                raise result
        out = []
        for result in results:
            if isinstance(result, Exception):
                self.region_failures.append(f"{type(result).__name__}: {result}")
                out.append(f"{REGION_FAILED_PREFIX}{len(self.region_failures) - 1}\x00")
            else:
                out.append(result)
        return out

    async def _shutdown(self) -> None:
        try:
            self.chat_model.engine.shutdown()
        finally:
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            results = await asyncio.gather(*pending, return_exceptions=True)
            await self.loop.shutdown_asyncgens()
            await self.loop.shutdown_default_executor()
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise result

    def close(self) -> None:
        if self.loop.is_closed():
            return
        try:
            self.loop.run_until_complete(self._shutdown())
        finally:
            self.loop.close()
