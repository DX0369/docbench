"""Persistent CPU client; keep the checkpoint sampling protocol, not API demo defaults."""
from __future__ import annotations
import json

GENERATION = {"max_tokens": 8192, "temperature": 0.7, "top_p": 0.8,
              "extra_body": {"top_k": 20}}
SERVED_MODEL = "Popo"


class PopoServiceUnavailable(RuntimeError):
    gdb_runtime_unusable = True


class PopoClient:
    def __init__(self, base_url):
        from openai import OpenAI

        # One attempt; no SDK retry nested inside the upstream five-try loop.
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0)

    def close(self):
        self.client.close()

    def generate(self, prompt, base64_image):
        from openai import APIConnectionError, APITimeoutError

        content = []
        if base64_image:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"}})
        content.append({"type": "text", "text": prompt[:100000]})
        try:
            response = self.client.chat.completions.create(
                model=SERVED_MODEL, messages=[{"role": "user", "content": content}], **GENERATION)
        except APITimeoutError:
            raise
        except APIConnectionError as exc:
            raise PopoServiceUnavailable(str(exc)) from exc
        if len(response.choices) != 1:
            raise ValueError("Popo response must contain exactly one choice")
        choice = response.choices[0]
        result = choice.message.content
        if choice.finish_reason != "stop" or not isinstance(result, str) or not result.strip():
            # Keep the actual transport response, including null vs "" and stop
            # metadata, instead of diagnosing both as a generic empty failure.
            print("[popo-response] " + json.dumps(response.model_dump(mode="json"), ensure_ascii=False), flush=True)
        if choice.finish_reason != "stop":
            raise ValueError(f"Popo generation did not finish normally: {choice.finish_reason}")
        # Official contd/image training emits "" for an empty relation list.
        # Title assignments and table-cell decisions must not inherit this rule.
        if result == "" and prompt.startswith(("<image>\nTruncation Detection: ",
                                               "<image>\nImage-Text Correlation Analysis: ")):
            return result
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Popo returned missing/empty text")
        return result
