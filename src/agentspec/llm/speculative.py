"""Speculative decoding (a.k.a. speculative sampling).

Implements the draft-then-verify algorithm from:

    Leviathan, Kalman & Matias (2023). "Fast Inference from Transformers
    via Speculative Decoding." ICML 2023.

Algorithm sketch (per round):

1. The small *draft* model autoregressively proposes ``gamma`` tokens.
2. The large *target* model scores every proposed position (with the
   HuggingFace backend these calls share a KV-cache, so the whole round
   costs roughly one parallel target forward pass).
3. Position ``i`` is accepted with probability ``min(1, p(x_i) / q(x_i))``
   where ``p``/``q`` are the target/draft distributions. On the first
   rejection a token is resampled from the normalized residual
   ``norm(max(0, p - q))`` and the round ends; if all tokens are accepted
   a free "bonus" token is sampled from the target distribution.

The guarantee of the algorithm is that the output distribution is
*identical* to sampling from the target model directly, while each round
produces ``>= 1`` tokens for a single (parallelizable) target scoring pass.

This module is pure Python (stdlib only) so it can be unit-tested and
studied without any numerical dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from agentspec.llm.base import TokenModel


@dataclass
class SpeculativeStats:
    """Book-keeping for a speculative generation run."""

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by the target model."""
        return self.accepted / max(1, self.proposed)

    @property
    def tokens_per_round(self) -> float:
        """Average number of tokens committed per target scoring round.

        This is the quantity that translates (roughly linearly) into wall
        clock speedup when the draft model is much cheaper than the target.
        """
        return self.accepted / max(1, self.rounds) + 1.0


@dataclass
class SpeculativeResult:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stats: SpeculativeStats = field(default_factory=SpeculativeStats)


def _with_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    """Apply temperature scaling to a probability distribution."""
    probs = [max(float(p), 1e-12) for p in probs]
    if temperature <= 0:
        best = max(range(len(probs)), key=probs.__getitem__)
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    if temperature == 1.0:
        total = sum(probs)
        return [p / total for p in probs]
    scaled = [math.pow(p, 1.0 / temperature) for p in probs]
    total = sum(scaled)
    return [p / total for p in scaled]


def _draw(probs: Sequence[float], temperature: float, rng: random.Random) -> int:
    """Draw a token id; deterministic argmax when ``temperature <= 0``."""
    if temperature <= 0:
        return max(range(len(probs)), key=probs.__getitem__)
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]


def residual_distribution(p: Sequence[float], q: Sequence[float]) -> list[float]:
    """Normalized positive part of ``p - q`` used for rejection resampling.

    Falls back to ``p`` when the residual mass is zero (e.g. ``p == q``),
    which keeps the sampler well-defined while preserving exactness.
    """
    residual = [max(float(a) - float(b), 0.0) for a, b in zip(p, q)]
    mass = sum(residual)
    if mass <= 0:
        total = sum(p)
        return [float(x) / total for x in p]
    return [x / mass for x in residual]


def speculative_decode(
    target: TokenModel,
    draft: TokenModel,
    prompt: str | Sequence[int],
    *,
    max_new_tokens: int = 128,
    gamma: int = 4,
    temperature: float = 1.0,
    seed: int | None = None,
    eos_token_id: int | None = None,
) -> SpeculativeResult:
    """Generate tokens from ``target`` accelerated by ``draft`` proposals.

    Args:
        target: The large, high-quality model whose distribution we sample.
        draft: A small, fast model that proposes candidate continuations.
        prompt: Raw text (encoded with ``target.encode``) or a token id list.
        max_new_tokens: Hard budget of tokens to generate.
        gamma: Number of tokens the draft proposes per round (lookahead).
        temperature: Sampling temperature; ``<= 0`` means greedy decoding.
        seed: Optional RNG seed for reproducibility.
        eos_token_id: If given, generation stops once this token is produced.

    Returns:
        A :class:`SpeculativeResult` with the generated ids, decoded text and
        acceptance statistics.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if target.vocab_size != draft.vocab_size:
        raise ValueError("target and draft must share the same vocabulary")

    rng = random.Random(seed)
    prompt_ids = list(target.encode(prompt)) if isinstance(prompt, str) else list(prompt)
    ids = list(prompt_ids)
    stats = SpeculativeStats()
    budget = max_new_tokens

    while budget > 0:
        stats.rounds += 1
        block = min(gamma, budget)

        # ---- 1. Draft proposes `block` tokens autoregressively ----------
        draft_ids: list[int] = []
        draft_dists: list[list[float]] = []
        prefix = list(ids)
        for _ in range(block):
            q = _with_temperature(draft.next_token_probs(prefix), temperature)
            token = _draw(q, temperature, rng) if temperature > 0 else _draw(q, 0.0, rng)
            draft_ids.append(token)
            draft_dists.append(q)
            prefix.append(token)
        stats.proposed += len(draft_ids)

        # ---- 2. Target verifies the proposal ----------------------------
        rejected = False
        for i, token in enumerate(draft_ids):
            p = _with_temperature(target.next_token_probs(ids + draft_ids[:i]), temperature)
            if temperature <= 0:
                accept = _draw(p, 0.0, rng) == token
            else:
                accept = rng.random() < min(1.0, p[token] / max(draft_dists[i][token], 1e-12))
            if accept:
                ids.append(token)
                stats.accepted += 1
                budget -= 1
            else:
                if temperature <= 0:
                    ids.append(_draw(p, 0.0, rng))
                else:
                    ids.append(_draw(residual_distribution(p, draft_dists[i]), 1.0, rng))
                budget -= 1
                rejected = True
                break
            if budget <= 0:
                break

        # ---- 3. All accepted -> free bonus token from the target --------
        if not rejected and budget > 0:
            p = _with_temperature(target.next_token_probs(ids), temperature)
            ids.append(_draw(p, temperature, rng))
            budget -= 1

        if eos_token_id is not None and eos_token_id in ids[len(prompt_ids):]:
            ids = ids[: ids.index(eos_token_id, len(prompt_ids)) + 1]
            break

    new_ids = ids[len(prompt_ids):]
    return SpeculativeResult(token_ids=new_ids, text=target.decode(new_ids), stats=stats)
