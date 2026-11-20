"""HuggingFace Transformers backend (optional dependency).

Requires the ``hf`` extra:  ``pip install agentspec[hf]``.

:class:`HFCausalLM` implements the :class:`~agentspec.llm.base.TokenModel`
protocol with an internal KV-cache: when ``next_token_probs`` is called with
a prefix that extends the previous one, only the *new* tokens are forwarded.
That is exactly the access pattern of :func:`speculative_decode`, so draft
proposal rounds and target verification rounds both run incrementally
instead of recomputing the full prefix every time.
"""

from __future__ import annotations

import math
from typing import Sequence

from agentspec.llm.base import ChatLLM, GenerationConfig


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    exp = [math.exp(x - peak) for x in logits]
    total = sum(exp)
    return [x / total for x in exp]


class HFCausalLM:
    """Token-level wrapper around a ``transformers`` causal LM."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "HFCausalLM requires the 'hf' extra: pip install agentspec[hf]"
            ) from exc

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()

        self.reset_cache()

    # -- TokenModel protocol ------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def reset_cache(self) -> None:
        """Drop the KV-cache (call between independent generations)."""
        self._past = None
        self._cached_prefix: list[int] = []

    def next_token_probs(self, prefix_ids: Sequence[int]) -> list[float]:
        ids = [int(i) for i in prefix_ids]
        if not ids:
            raise ValueError("prefix_ids must be non-empty")

        torch = self._torch
        with torch.no_grad():
            if (
                self._past is not None
                and len(ids) > len(self._cached_prefix)
                and ids[: len(self._cached_prefix)] == self._cached_prefix
            ):
                # Incremental step: only forward the new suffix.
                new_ids = ids[len(self._cached_prefix):]
                inputs = torch.tensor([new_ids], device=self.device)
                out = self.model(
                    input_ids=inputs, past_key_values=self._past, use_cache=True
                )
            else:
                inputs = torch.tensor([ids], device=self.device)
                out = self.model(input_ids=inputs, use_cache=True)

        self._past = out.past_key_values
        self._cached_prefix = ids
        logits = out.logits[0, -1].float().tolist()
        return _softmax(logits)


class HFChatLLM(ChatLLM):
    """Chat-level wrapper so HF instruct models can drive the agent loops."""

    def __init__(self, model_name_or_path: str, **kwargs):
        self.lm = HFCausalLM(model_name_or_path, **kwargs)

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        tokenizer = self.lm.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = prompt
        else:
            text = prompt

        self.lm.reset_cache()
        ids = self.lm.encode(text)
        if not ids:
            ids = [tokenizer.bos_token_id or 0]
        eos_id = tokenizer.eos_token_id

        out: list[int] = []
        import random

        rng = random.Random()
        for _ in range(config.max_new_tokens):
            probs = self.lm.next_token_probs(ids + out)
            if config.temperature <= 0:
                nxt = max(range(len(probs)), key=probs.__getitem__)
            else:
                scaled = [
                    math.pow(max(p, 1e-12), 1.0 / config.temperature) for p in probs
                ]
                total = sum(scaled)
                nxt = rng.choices(range(len(probs)), weights=[s / total for s in scaled])[0]
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)

        text_out = self.lm.decode(out)
        for stop in config.stop_sequences:
            if stop in text_out:
                text_out = text_out.split(stop)[0]
        return text_out
