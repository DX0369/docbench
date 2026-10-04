"""One event loop for the lifetime of a synchronous MonkeyOCR pipeline."""
from __future__ import annotations

import asyncio
from typing import Any


class AsyncRecognitionLoop:
    """Keep vLLM's output tasks alive between documents in one worker thread."""

    def __init__(self, chat_model: Any) -> None:
        self.chat_model = chat_model
        self.loop = asyncio.new_event_loop()

    def batch_inference(self, images: list, questions: list) -> list[str]:
        # Use the official async implementation unchanged, not its per-call
        # asyncio.run() bridge, which cancels the engine's background tasks.
        return self.loop.run_until_complete(
            self.chat_model.async_batch_inference(images, questions)
        )

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
