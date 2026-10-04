#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import multiprocessing
import os
import sys
import time
import types
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Iterable


def _sanitize_positive_int_env(name: str, default: int) -> None:
    raw_value = os.environ.get(name)
    try:
        value = int(raw_value) if raw_value is not None else None
    except ValueError:
        value = None
    if value is None or value <= 0:
        os.environ[name] = str(default)


_sanitize_positive_int_env("OMP_NUM_THREADS", 4)
_sanitize_positive_int_env("MKL_NUM_THREADS", 4)

THIS_DIR = Path(__file__).resolve().parent
# Direct script execution prepends this adapter directory to sys.path. It also
# contains vllm.py, which must not shadow the installed vLLM package imported
# by the service below.
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != THIS_DIR
]

SRC_ROOT = THIS_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
import uvicorn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from method_paths import method_repo_path  # noqa: E402

DEEPSEEK_REPO_ROOT = method_repo_path("deepseek_ocr2")
DEEPSEEK_VLLM_ROOT = DEEPSEEK_REPO_ROOT / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm"

if str(DEEPSEEK_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPSEEK_VLLM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve DeepSeek-OCR-2 through a minimal OpenAI-compatible API.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model-name", type=str, default="deepseek-ocr2")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--logits-ngram-size", type=int, default=20)
    parser.add_argument("--logits-window-size", type=int, default=50)
    parser.add_argument("--preprocess-workers", type=int, default=64)
    parser.add_argument(
        "--engine-iteration-timeout-seconds",
        type=int,
        default=600,
        help=(
            "Forwarded to vLLM through VLLM_ENGINE_ITERATION_TIMEOUT_S before "
            "AsyncLLMEngine is imported. DeepSeek OCR2 multimodal p64 can exceed "
            "vLLM's 60s default engine-step timeout."
        ),
    )
    return parser.parse_args()


_PROCESS_PREPROCESSOR: Any | None = None
_PROCESS_IMAGE_PROCESS_MODULE: Any | None = None
_PROCESS_CONFIG_MODULE: Any | None = None


def _init_process_preprocessor(model_path: str, deepseek_vllm_root: str, startup_barrier=None) -> None:
    global _PROCESS_PREPROCESSOR
    global _PROCESS_IMAGE_PROCESS_MODULE
    global _PROCESS_CONFIG_MODULE

    _sanitize_positive_int_env("OMP_NUM_THREADS", 4)
    _sanitize_positive_int_env("MKL_NUM_THREADS", 4)

    deepseek_vllm_root = Path(deepseek_vllm_root)
    if str(deepseek_vllm_root) not in sys.path:
        sys.path.insert(0, str(deepseek_vllm_root))

    config_module = types.ModuleType("config")
    config_module.BASE_SIZE = 1024
    config_module.IMAGE_SIZE = 768
    config_module.CROP_MODE = True
    config_module.MIN_CROPS = 2
    config_module.MAX_CROPS = 6
    config_module.MAX_CONCURRENCY = 1
    config_module.NUM_WORKERS = 1
    config_module.PRINT_NUM_VIS_TOKENS = False
    config_module.SKIP_REPEAT = True
    config_module.MODEL_PATH = str(model_path)
    config_module.INPUT_PATH = ""
    config_module.OUTPUT_PATH = ""
    config_module.PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    config_module.TOKENIZER = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    sys.modules["config"] = config_module

    import process.image_process as image_process_module  # type: ignore  # noqa: WPS433
    from process.image_process import DeepseekOCR2Processor  # type: ignore  # noqa: WPS433

    _PROCESS_CONFIG_MODULE = config_module
    _PROCESS_IMAGE_PROCESS_MODULE = image_process_module
    _PROCESS_PREPROCESSOR = DeepseekOCR2Processor()
    if startup_barrier is not None:
        startup_barrier.wait(timeout=600)


def preprocessor_ready():
    """Initialization probe only: no image preprocessing or token generation."""
    return os.getpid()


def preprocess_chat_process(messages):
    prompt, images = extract_prompt_and_images(messages)
    _PROCESS_CONFIG_MODULE.PROMPT = prompt
    _PROCESS_IMAGE_PROCESS_MODULE.PROMPT = prompt
    try:
        return {"prompt": prompt, "multi_modal_data": {"image":
            _PROCESS_PREPROCESSOR.tokenize_with_images(images=images, bos=True, eos=True, cropping=True)}}
    finally:
        for image in images:
            image.close()


def decode_data_url_image(url: str) -> Image.Image:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("Only data URL images are supported.")
    try:
        _, encoded = url.split(",", 1)
        raw = base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to decode data URL image: {exc}") from exc
    image = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(image).convert("RGB")


def extract_prompt_and_images(messages: Iterable[dict[str, Any]]) -> tuple[str, list[Image.Image]]:
    prompt_parts: list[str] = []
    images: list[Image.Image] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                prompt_parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url") or {}
                images.append(decode_data_url_image(str(image_url.get("url") or "")))
    prompt = "".join(prompt_parts).strip()
    if not prompt:
        raise ValueError("Request does not contain prompt text.")
    if not images:
        raise ValueError("Request does not contain any image.")
    return prompt, images


def build_sampling_params_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    whitelist_token_ids = {128821, 128822}
    kwargs: dict[str, Any] = {
        "temperature": float(body.get("temperature", 0.0)),
        "max_tokens": int(body.get("max_tokens", 8192)),
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "_logits_ngram_size": int(body.pop("_logits_ngram_size", 20)),
        "_logits_window_size": int(body.pop("_logits_window_size", 50)),
        "_whitelist_token_ids": whitelist_token_ids,
    }
    top_p = body.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    seed = body.get("seed")
    if seed is not None:
        kwargs["seed"] = int(seed)
    return kwargs


def build_app(args: argparse.Namespace) -> FastAPI:
    os.environ.setdefault("VLLM_USE_V1", "0")
    if int(args.engine_iteration_timeout_seconds) > 0:
        os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = str(
            int(args.engine_iteration_timeout_seconds)
        )
    if hasattr(os, "environ") and os.environ.get("TRITON_PTXAS_PATH") is None:
        maybe_ptxas = Path("/usr/local/cuda-11.8/bin/ptxas")
        if maybe_ptxas.exists():
            os.environ["TRITON_PTXAS_PATH"] = str(maybe_ptxas)

    config_module = types.ModuleType("config")
    config_module.BASE_SIZE = 1024
    config_module.IMAGE_SIZE = 768
    config_module.CROP_MODE = True
    config_module.MIN_CROPS = 2
    config_module.MAX_CROPS = 6
    config_module.NUM_WORKERS = 64
    config_module.PRINT_NUM_VIS_TOKENS = False
    config_module.SKIP_REPEAT = True
    config_module.MODEL_PATH = str(args.model_path)
    config_module.INPUT_PATH = ""
    config_module.OUTPUT_PATH = ""
    config_module.PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    config_module.TOKENIZER = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
    )
    sys.modules["config"] = config_module

    from vllm import AsyncLLMEngine, SamplingParams  # noqa: WPS433
    from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: WPS433
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: WPS433

    from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # type: ignore  # noqa: WPS433,E402
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # type: ignore  # noqa: WPS433,E402

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

    engine_args = AsyncEngineArgs(
        model=str(args.model_path),
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=256,
        swap_space=0,
        disable_mm_preprocessor_cache=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    max_num_seqs = engine.engine.scheduler_config.max_num_seqs
    context = multiprocessing.get_context("spawn")
    worker_count = max(1, int(args.preprocess_workers))
    preprocess_executor = ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_process_preprocessor,
        initargs=(str(args.model_path), str(DEEPSEEK_VLLM_ROOT), context.Barrier(worker_count)),
        mp_context=context,
    )
    app = FastAPI(title="DeepSeek-OCR-2 API", version="0.1.0")
    preprocess_failure = None

    @app.on_event("startup")
    async def initialize_preprocessors() -> None:
        # Force CPU workers/tokenizers to initialize before health is reachable.
        # No warmup PDF or model generation is performed here.
        loop = asyncio.get_running_loop()
        try:
            await asyncio.gather(*(loop.run_in_executor(preprocess_executor, preprocessor_ready)
                                  for _ in range(max(1, int(args.preprocess_workers)))))
        except BaseException:
            await asyncio.to_thread(preprocess_executor.shutdown, wait=True, cancel_futures=True)
            raise

    @app.on_event("shutdown")
    async def shutdown_preprocessors() -> None:
        await asyncio.to_thread(preprocess_executor.shutdown, wait=True, cancel_futures=True)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if engine.is_stopped:
            raise HTTPException(status_code=503, detail=str(engine.dead_error))
        if preprocess_failure is not None:
            raise HTTPException(status_code=503, detail={"error": "preprocess_pool_broken", "message": preprocess_failure})
        return {
            "status": "ok",
            "max_num_seqs": str(max_num_seqs),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "deepseek",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        nonlocal preprocess_failure
        try:
            body = await request.json()
            payload = await asyncio.get_running_loop().run_in_executor(
                preprocess_executor,
                preprocess_chat_process,
                body["messages"],
            )
            sampling_kwargs = build_sampling_params_kwargs(
                {
                    **body,
                    "_logits_ngram_size": args.logits_ngram_size,
                    "_logits_window_size": args.logits_window_size,
                }
            )
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=int(sampling_kwargs.pop("_logits_ngram_size")),
                    window_size=int(sampling_kwargs.pop("_logits_window_size")),
                    whitelist_token_ids=set(sampling_kwargs.pop("_whitelist_token_ids")),
                )
            ]
            sampling_params = SamplingParams(
                logits_processors=logits_processors,
                **sampling_kwargs,
            )
            request_id = f"deepseek-{uuid.uuid4().hex}"
            final_output = None
            finish_reason = None
            async for request_output in engine.generate(payload, sampling_params, request_id):
                if request_output.outputs:
                    final_output = request_output.outputs[0].text
                    finish_reason = request_output.outputs[0].finish_reason
            if not isinstance(final_output, str) or finish_reason is None:
                raise RuntimeError("DeepSeek engine returned no completed output")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, BrokenProcessPool):
                preprocess_failure = str(exc)
                raise HTTPException(status_code=503, detail={"error": "preprocess_pool_broken", "message": str(exc)}) from exc
            if engine.is_stopped:
                raise HTTPException(status_code=503, detail={"error": "engine_dead", "message": str(engine.dead_error)}) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        content = str(final_output).replace("<｜end▁of▁sentence｜>", "").strip()
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }

    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
