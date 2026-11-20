"""Deterministic mock models.

These let the test-suite (and the demo) exercise the full agent loop and
the speculative decoder without downloading any weights or touching a GPU.
Pure stdlib - no numerical dependencies.
"""

from __future__ import annotations

import math
import random
import re
from typing import Sequence

from agentspec.llm.base import ChatLLM, GenerationConfig


class MockChatLLM(ChatLLM):
    """A scripted chat model: each ``generate`` call pops one response.

    Useful for deterministic tests and demos of the agent loops. When the
    script is exhausted a canned ``Final Answer`` is returned so agent loops
    always terminate.
    """

    def __init__(self, responses: Sequence[str] | None = None):
        self._responses = list(responses or [])
        self.prompts: list[str] = []

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            return "Final Answer: (script exhausted)"
        return self._responses.pop(0)


class MockTokenModel:
    """A small synthetic causal LM with a controllable draft/target correlation.

    Tokens are integers ``0..vocab_size-1`` rendered as ``"<t{i}>"``. The
    next-token logits are a deterministic function of the last token:

    * the *base* component depends only on ``shared_seed`` - two models built
      with the same ``shared_seed`` agree on it;
    * the *noise* component depends on ``noise_seed`` and is scaled by
      ``noise_scale``.

    A draft model that shares ``shared_seed`` with its target but has a
    different ``noise_seed`` behaves like a real, imperfectly-correlated
    draft: mostly agreeing, occasionally diverging.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        shared_seed: int = 0,
        noise_seed: int = 0,
        noise_scale: float = 0.0,
    ):
        self._vocab_size = vocab_size
        self._shared_seed = shared_seed
        self._noise_seed = noise_seed
        self._noise_scale = noise_scale

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return [int(m.group(1)) for m in re.finditer(r"<t(\d+)>", text)]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"<t{i}>" for i in ids)

    def _logits(self, last_token: int) -> list[float]:
        base_rng = random.Random(f"base:{self._shared_seed}:{last_token}")
        logits = [base_rng.gauss(0.0, 1.0) for _ in range(self._vocab_size)]
        if self._noise_scale > 0:
            noise_rng = random.Random(f"noise:{self._noise_seed}:{last_token}")
            logits = [
                x + self._noise_scale * noise_rng.gauss(0.0, 1.0) for x in logits
            ]
        return logits

    def next_token_probs(self, prefix_ids: Sequence[int]) -> list[float]:
        last = int(prefix_ids[-1]) if prefix_ids else -1
        logits = self._logits(last)
        peak = max(logits)
        exp = [math.exp(x - peak) for x in logits]
        total = sum(exp)
        return [x / total for x in exp]
