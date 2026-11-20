"""HuggingFace Transformers backend (optional dependency).

Requires the ``hf`` extra:  ``pip install agentspec[hf]``.

:class:`HFCausalLM` implements the :class:`~agentspec.llm.base.TokenModel`
protocol with an internal KV-cache. On top of that it exposes the two
primitives needed by *efficient* speculative decoding:

* :meth:`HFCausalLM.logits_tensor` - incremental single-token scoring for the
  cheap draft model;
* :meth:`HFCausalLM.verify_block_logits` - **one** forward pass that scores an
  entire proposed block and returns ``gamma + 1`` next-token distributions.

:func:`speculative_decode_hf` implements the draft-then-verify algorithm with
batched verification and GPU-side sampling (no Python <-> GPU transfers of a
150k-wide distribution per token). The pure-Python implementation in
``llm/speculative.py`` remains the reference algorithm and is used by the
unit tests.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from agentspec.llm.base import ChatLLM, GenerationConfig
from agentspec.llm.speculative import SpeculativeResult, SpeculativeStats


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

    # -- cache management ---------------------------------------------------

    def reset_cache(self) -> None:
        """Drop the KV-cache (call between independent generations)."""
        self._past = None
        self._cached_prefix: list[int] = []

    def truncate_cache(self, n: int) -> None:
        """Keep only the first ``n`` cached positions.

        After a speculative round the cached KV entries for rejected draft
        positions are invalid (the committed prefix diverged there), so the
        cache is cropped back to the valid prefix at the start of each round.
        """
        if self._past is None or n < 0:
            return
        if isinstance(self._past, tuple):  # legacy tuple-of-tuples cache
            self._past = tuple(
                (k[..., :n, :], v[..., :n, :]) for k, v in self._past
            )
        elif hasattr(self._past, "crop"):  # transformers >= 4.45 Cache objects
            self._past.crop(n)
        else:  # unknown cache type - safest to drop it
            self.reset_cache()
            return
        self._cached_prefix = self._cached_prefix[:n]

    # -- TokenModel protocol ------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def logits_tensor(self, prefix_ids: Sequence[int]):
        """Return raw next-token logits (GPU tensor), updating the KV-cache.

        Only the *new* suffix is forwarded when ``prefix_ids`` extends the
        previously cached prefix.
        """
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
                new_ids = ids[len(self._cached_prefix):]
                out = self.model(
                    input_ids=torch.tensor([new_ids], device=self.device),
                    past_key_values=self._past,
                    use_cache=True,
                )
            else:
                out = self.model(
                    input_ids=torch.tensor([ids], device=self.device), use_cache=True
                )
        self._past = out.past_key_values
        self._cached_prefix = ids
        return out.logits[0, -1].float()

    def next_token_probs(self, prefix_ids: Sequence[int]) -> list[float]:
        """Protocol-compliant distribution (used by the reference algorithm)."""
        torch = self._torch
        return torch.softmax(self.logits_tensor(prefix_ids), dim=-1).tolist()

    def verify_block_logits(self, prefix_ids: Sequence[int], proposed_ids: Sequence[int]):
        """Score a whole draft block in ONE forward pass.

        With the cache covering ``prefix_ids[:-1]``, the input is
        ``[prefix_ids[-1]] + proposed_ids``. Position ``i`` of the returned
        list is the distribution used to accept ``proposed_ids[i]``; the final
        element is the bonus-token distribution after the full block.
        """
        ids = [int(i) for i in prefix_ids]
        props = [int(i) for i in proposed_ids]
        torch = self._torch
        if not ids or not props:
            raise ValueError("prefix_ids and proposed_ids must be non-empty")

        if self._past is None or self._cached_prefix != ids[:-1]:
            # Rebuild the cache for the committed prefix (usually only on the
            # first round; subsequent rounds just crop it).
            self.reset_cache()
            if len(ids) > 1:
                with torch.no_grad():
                    out = self.model(
                        input_ids=torch.tensor([ids[:-1]], device=self.device),
                        use_cache=True,
                    )
                self._past = out.past_key_values
                self._cached_prefix = list(ids[:-1])

        input_ids = [ids[-1]] + props
        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([input_ids], device=self.device),
                past_key_values=self._past,
                use_cache=True,
            )
        self._past = out.past_key_values
        self._cached_prefix = ids + props
        return [out.logits[0, i].float() for i in range(out.logits.shape[1])]


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
        rng = random.Random()
        for _ in range(config.max_new_tokens):
            probs = self.lm.next_token_probs(ids + out)
            if config.temperature <= 0:
                nxt = max(range(len(probs)), key=probs.__getitem__)
            else:
                scaled = [math.pow(max(p, 1e-12), 1.0 / config.temperature) for p in probs]
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


def speculative_decode_hf(
    target: HFCausalLM,
    draft: HFCausalLM,
    prompt: str | Sequence[int],
    *,
    max_new_tokens: int = 128,
    gamma: int = 4,
    temperature: float = 0.0,
    seed: int | None = None,
    eos_token_id: int | None = None,
) -> SpeculativeResult:
    """GPU-efficient speculative decoding with batched target verification.

    One round = ``gamma`` cheap draft steps + **one** target forward pass over
    the whole proposed block. All acceptance / rejection / residual sampling
    happens on the GPU, so no 150k-wide distributions ever cross into Python.

    Greedy mode (``temperature <= 0``) is bit-identical to decoding with the
    target model alone; sampling mode preserves the target distribution.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    # Qwen2.5 small models use vocab 151936 while 7B/14B use 152064: the
    # draft's support is a strict subset of the target's, which is valid for
    # speculative sampling (extra target tokens can only enter through the
    # residual distribution). Pad q to the target width below.
    if draft.vocab_size > target.vocab_size:
        raise ValueError("draft vocabulary must be a subset of the target vocabulary")

    torch = target._torch
    if seed is not None:
        torch.manual_seed(seed)

    prompt_ids = list(target.encode(prompt)) if isinstance(prompt, str) else list(prompt)
    ids = list(prompt_ids)
    stats = SpeculativeStats()
    budget = max_new_tokens
    device = target.device

    while budget > 0:
        stats.rounds += 1
        block = min(gamma, budget)

        # Crop caches back to the last known-valid prefix. After a rejection
        # the cached KV entries of the diverged draft positions are invalid.
        n_keep = len(ids) - 1
        if n_keep > 0:
            target.truncate_cache(n_keep)
            draft.truncate_cache(n_keep)
        else:
            target.reset_cache()
            draft.reset_cache()

        # ---- 1. Draft proposes `block` tokens (one cheap step each) ------
        draft_ids: list[int] = []
        draft_dists: list = []
        prefix = list(ids)
        for _ in range(block):
            q_logits = draft.logits_tensor(prefix)
            if temperature <= 0:
                token = int(q_logits.argmax().item())
                draft_dists.append(None)
            else:
                q = torch.softmax(q_logits / temperature, dim=-1)
                token = int(torch.multinomial(q, 1).item())
                draft_dists.append(q)
            draft_ids.append(token)
            prefix.append(token)
        stats.proposed += len(draft_ids)

        # ---- 2. Target verifies the full block in ONE forward pass -------
        p_logits = target.verify_block_logits(ids, draft_ids)

        # ---- 3. Accept / reject ------------------------------------------
        rejected = False
        for i, token in enumerate(draft_ids):
            if temperature <= 0:
                accept = int(p_logits[i].argmax().item()) == token
            else:
                p = torch.softmax(p_logits[i] / temperature, dim=-1)
                q = draft_dists[i]
                if q.numel() < p.numel():
                    q = torch.nn.functional.pad(q, (0, p.numel() - q.numel()), value=0.0)
                qx = max(float(q[token].item()), 1e-12)
                px = float(p[token].item())
                accept = float(torch.rand((), device=device).item()) < min(1.0, px / qx)
            if accept:
                ids.append(token)
                stats.accepted += 1
                budget -= 1
            else:
                if temperature <= 0:
                    nxt = int(p_logits[i].argmax().item())
                else:
                    q = draft_dists[i]
                    if q.numel() < p.numel():
                        q = torch.nn.functional.pad(q, (0, p.numel() - q.numel()), value=0.0)
                    residual = (p - q).clamp(min=0)
                    mass = residual.sum()
                    dist = residual / mass if float(mass.item()) > 0 else p
                    nxt = int(torch.multinomial(dist, 1).item())
                ids.append(nxt)
                budget -= 1
                rejected = True
                break
            if budget <= 0:
                break

        # ---- 4. All accepted -> bonus token from the last distribution ---
        if not rejected and budget > 0:
            last = p_logits[block]
            if temperature <= 0:
                bonus = int(last.argmax().item())
            else:
                bonus = int(torch.multinomial(torch.softmax(last / temperature, -1), 1).item())
            ids.append(bonus)
            budget -= 1

        if eos_token_id is not None and eos_token_id in ids[len(prompt_ids):]:
            ids = ids[: ids.index(eos_token_id, len(prompt_ids)) + 1]
            break

    new_ids = ids[len(prompt_ids):]
    return SpeculativeResult(token_ids=new_ids, text=target.decode(new_ids), stats=stats)
