"""Agent memory: short-term conversation buffer + long-term retrieval store."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Sliding-window short-term memory.

    Keeps the most recent ``window_size`` messages; an optional system
    prompt is always preserved at the front.
    """

    def __init__(self, window_size: int = 20, system_prompt: str | None = None):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.system_prompt = system_prompt
        self._messages: list[Message] = []

    def add(self, role: str, content: str, **metadata: Any) -> None:
        self._messages.append(Message(role=role, content=content, metadata=metadata))
        if len(self._messages) > self.window_size:
            self._messages = self._messages[-self.window_size:]

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def as_chat(self) -> list[dict[str, str]]:
        """Render as an OpenAI-style message list."""
        chat = []
        if self.system_prompt:
            chat.append({"role": "system", "content": self.system_prompt})
        chat.extend({"role": m.role, "content": m.content} for m in self._messages)
        return chat

    def clear(self) -> None:
        self._messages.clear()


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LongTermMemory:
    """Embedding-free long-term memory with IDF-weighted keyword retrieval.

    Deliberately dependency-free: documents are scored by the IDF-weighted
    overlap with the query tokens. Swap in a vector store later without
    changing the agent-facing API.
    """

    def __init__(self):
        self._docs: list[tuple[str, set[str], dict[str, Any]]] = []
        self._df: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, text: str, **metadata: Any) -> None:
        tokens = set(_tokenize(text))
        self._docs.append((text, tokens, metadata))
        for tok in tokens:
            self._df[tok] = self._df.get(tok, 0) + 1

    def _idf(self, token: str) -> float:
        n = len(self._docs)
        return math.log((1 + n) / (1 + self._df.get(token, 0))) + 1.0

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str, dict[str, Any]]]:
        """Return the ``top_k`` most relevant ``(score, text, metadata)`` triples."""
        query_tokens = set(_tokenize(query))
        scored = []
        for text, tokens, metadata in self._docs:
            score = sum(self._idf(tok) for tok in query_tokens & tokens)
            if score > 0:
                scored.append((score, text, metadata))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._docs.clear()
        self._df.clear()
