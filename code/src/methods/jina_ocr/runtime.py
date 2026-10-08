"""One resident AsyncLLM; official model helpers own vision and FastMTP."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path


STRICT_PROMPT = """Just return the plain text representation of this document as if you were reading it naturally.
Turn equations and math symbols into a LaTeX representation, make sure to use $ and $ as a delimiter for inline math, and $$ and $$ for block math. Do NOT use ascii or unicode math symbols such as ∈ ∉ ⊂ ⊃ ⊆ ⊇ ∅ ∪ ∩ ∀ ∃ ¬, just use LaTeX syntax, ex $ \\in $ $ \\notin $ etc. If you were going to surround a math expression in \\( \\) or \\[ \\] delimiters, surround it with $ $ or $$ $$ instead.
Convert tables into HTML format. Keep the syntax simple, but use <th> for header rows, and use rowspan and colspans appropriately. Don't use <br> inside of table cells, just split that into new rows as needed. Do NOT use LaTeX or Markdown table syntax.
Ignore all graphical content in the image document. Do not attempt to describe or convert the images.
Remove the headers and footers, but keep references and footnotes.
Read any natural handwriting.
This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.
If there is no text at all that you think you should read, you can output null."""


class JinaRuntime:
    def __init__(self, args):
        model = str(args.model_path.resolve())
        sys.path.insert(0, model)
        src = str(Path(__file__).resolve().parents[2])
        os.environ["PYTHONPATH"] = os.pathsep.join((model, src, os.environ.get("PYTHONPATH", "")))
        from deepseek_ocr_mtp import register, prepare_vllm_input, vllm_llm_kwargs, vllm_sampling_params
        register()
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.prepare = prepare_vllm_input
        self.sampling = vllm_sampling_params(max_tokens=args.max_tokens)
        # Fail rather than silently omitting an official sampling setting.
        if getattr(self.sampling, "repetition_detection", None) is None:
            raise RuntimeError("This vLLM lacks the official repetition_detection setting")
        self.loop = asyncio.new_event_loop()
        kwargs = vllm_llm_kwargs(
            model, max_model_len=args.max_model_len,
            worker_extension_cls="methods.jina_ocr.vllm_worker.JinaRegistration",
        )
        async def initialize():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
        try:
            self.engine = self.loop.run_until_complete(initialize())
        except BaseException:
            self.loop.close()
            raise
        self.thread = threading.Thread(target=self.loop.run_forever, name="jina-async-engine", daemon=True)
        self.thread.start()

    async def _generate(self, image):
        request_id = uuid.uuid4().hex
        try:
            prompt = self.prepare(self.tokenizer, image, prompt=STRICT_PROMPT)
            final = None
            async for output in self.engine.generate(prompt, self.sampling, request_id):
                final = output
            if final is None or len(final.outputs) != 1:
                raise RuntimeError("Jina engine did not return one page completion")
            completion = final.outputs[0]
            if completion.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"Jina generation failed: {completion.finish_reason}")
            return completion.text, completion.finish_reason
        except Exception as exc:
            if self.engine.errored:
                exc.gdb_runtime_unusable = True
            raise

    def generate(self, image):
        return asyncio.run_coroutine_threadsafe(self._generate(image), self.loop).result()

    def close(self):
        async def shutdown():
            self.engine.shutdown()
            await self.loop.shutdown_asyncgens()
        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
