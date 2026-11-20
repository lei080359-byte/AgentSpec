"""LLM backend abstractions.

Two levels of abstraction are provided:

* :class:`ChatLLM` - text-in / text-out chat model, used by the agent loops.
* :class:`TokenModel` - token-level causal LM interface (encode/decode plus
  next-token probabilities), required by the speculative decoding module.

The core library is intentionally dependency-free (pure Python + stdlib).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class GenerationConfig:
    """Sampling configuration for chat-level generation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)


class ChatLLM(ABC):
    """Text-in / text-out chat model used by agent loops."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Generate a completion for ``prompt``."""

    def __call__(self, prompt: str, config: GenerationConfig | None = None) -> str:
        return self.generate(prompt, config)


@runtime_checkable
class TokenModel(Protocol):
    """Token-level causal LM interface required by speculative decoding.

    Implementations only need to expose a *next-token probability*
    distribution given a prefix. The HuggingFace backend serves these calls
    with a KV-cache so repeated prefix extensions are cheap.
    """

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def next_token_probs(self, prefix_ids: Sequence[int]) -> Sequence[float]:
        """Return the probability distribution over the next token.

        The returned sequence must have length ``vocab_size`` and sum to 1.
        """
        ...
