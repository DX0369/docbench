#!/usr/bin/env python3
from __future__ import annotations

import base64
import mimetypes
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import OpenAI


_HTTP_ATTEMPTS: ContextVar[list[int] | None] = ContextVar("gdb_http_attempts", default=None)


def _count_http_attempt(_request: httpx.Request) -> None:
    attempts = _HTTP_ATTEMPTS.get()
    if attempts is not None:
        attempts[0] += 1


def build_openai_client(base_url: str, timeout: float) -> OpenAI:
    # The SDK keeps its default transport retries; the request hook only counts attempts.
    return OpenAI(
        api_key="EMPTY",
        base_url=base_url.rstrip("/") + "/",
        timeout=timeout,
        http_client=httpx.Client(
            timeout=timeout,
            trust_env=False,
            event_hooks={"request": [_count_http_attempt]},
        ),
    )


@contextmanager
def count_http_attempts():
    """Yield a one-item list holding the HTTP attempts sent by this thread inside the block."""
    attempts = [0]
    token = _HTTP_ATTEMPTS.set(attempts)
    try:
        yield attempts
    finally:
        _HTTP_ATTEMPTS.reset(token)


def sdk_retry_warning(page_number: int, attempts: int) -> list[dict[str, Any]]:
    """One logical request sends one HTTP attempt; extra attempts are SDK transport retries."""
    if attempts > 1:
        return [{"page_number": page_number, "reason": "sdk_transport_retry", "count": attempts - 1}]
    return []


def propagate_chat_completion_errors(client: Any, operation: Callable[..., Any]) -> Callable[..., Any]:
    """Surface SDK failures swallowed by a synchronous official parser.

    Install once per client/operation. The SDK still owns retries; only errors
    escaping its create() are observed. Each operation gets an isolated scope,
    so an empty successful response is not confused with a failed request.
    """
    failures: ContextVar[list[Exception] | None] = ContextVar("chat_completion_failures", default=None)
    create = client.chat.completions.create

    @wraps(create)
    def observed_create(*args: Any, **kwargs: Any) -> Any:
        try:
            return create(*args, **kwargs)
        except Exception as exc:
            current = failures.get()
            if current is not None and not current:
                current.append(exc)
            raise

    @wraps(operation)
    def checked_operation(*args: Any, **kwargs: Any) -> Any:
        current: list[Exception] = []
        token = failures.set(current)
        try:
            result = operation(*args, **kwargs)
            if current:
                raise current[0]
            return result
        finally:
            failures.reset(token)

    client.chat.completions.create = observed_create
    return checked_operation


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_image_chat_messages(
    image_path: Path,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path)},
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def chat_with_image(
    client: OpenAI,
    model_name: str,
    image_path: Path,
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    create_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": build_image_chat_messages(
            image_path=image_path,
            prompt=prompt,
            system_prompt=system_prompt,
        ),
        "temperature": temperature,
        "stream": False,
    }
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if seed is not None:
        create_kwargs["seed"] = seed
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**create_kwargs)
    choice = response.choices[0]
    return content_to_text(choice.message.content).strip(), choice.finish_reason
