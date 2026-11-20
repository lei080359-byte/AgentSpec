"""OpenAI-compatible API backend (stdlib-only, no SDK required).

Works with any service exposing ``POST /v1/chat/completions`` - OpenAI,
DeepSeek, Moonshot, vLLM, Ollama (``/v1``), LM Studio, etc.

Credentials are read from the environment by default:

* ``OPENAI_API_KEY``  - API key
* ``OPENAI_BASE_URL`` - e.g. ``https://api.deepseek.com/v1``
"""

from __future__ import annotations

import json
import os
import urllib.request

from agentspec.llm.base import ChatLLM, GenerationConfig


class OpenAICompatibleChatLLM(ChatLLM):
    """Minimal chat-completions client built on ``urllib``."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key missing: pass api_key=... or set the OPENAI_API_KEY env var"
            )
        self.timeout = timeout

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        }
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
