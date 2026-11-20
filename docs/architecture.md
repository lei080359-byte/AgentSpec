# Architecture & Design Notes

## Why two layers in one repo

Most agent frameworks (LangChain, AutoGen, ...) treat the LLM as a remote
black box, and most inference-optimization repos (vLLM, TensorRT-LLM, ...)
stop at the decoding loop. AgentSpec sits at the intersection:

* the **agent layer** produces long, structured, highly-predictable text
  (formats like `Thought:/Action:/Action Input:`) - exactly the regime where
  speculative decoding shines, because a small draft model learns the
  scaffold trivially and acceptance rates are high;
* the **inference layer** is exposed through a minimal `TokenModel`
  protocol (`encode` / `decode` / `next_token_probs`), so the speculative
  decoder can be tested with deterministic mock models and reused with any
  real backend.

## Speculative decoding: correctness & cost model

The implementation in `llm/speculative.py` follows Leviathan et al. (2023):

1. Draft proposes `gamma` tokens autoregressively.
2. Target scores each position. Token `i` is accepted with probability
   `min(1, p(x_i)/q(x_i))`; on the first rejection we resample from
   `norm(max(0, p - q))` and stop; if all are accepted we get a free bonus
   token from `p`.
3. The output distribution is provably identical to sampling from the
   target model. The greedy special case (`temperature <= 0`) reduces to
   "accept iff the target's argmax equals the proposal", and the test-suite
   asserts the output is then *bit-identical* to plain autoregressive
   decoding.

**Two implementations.** `llm/speculative.py` is the pure-Python reference
algorithm (stdlib only, unit-tested with deterministic mock models).
`speculative_decode_hf` in `llm/hf_backend.py` is the GPU-optimized path: one
target forward scores the entire proposed block, and all sampling stays on
the GPU. Both share the same accept/reject semantics; the greedy output of
each is bit-identical to decoding with the target alone.

**Batched verification.** `verify_block_logits(prefix, proposals)` keeps the
KV-cache for `prefix[:-1]`, feeds `[prefix[-1]] + proposals` in a single
forward, and reads `gamma + 1` logits rows: row `i` verifies proposal `i`,
the last row is the bonus-token distribution. A rejection invalidates the
cached KV of the diverged draft positions, so `truncate_cache(n)` crops the
cache back to the last valid prefix at the start of every round (this also
handles the legacy tuple cache format). The draft model's cache is cropped
the same way. Verified numerically: batched and sequential logits differ by
exactly `0.0`.

**Cost model (and when it actually wins).** If a target forward costs `C`, a
draft forward costs `c`, and a round commits `k` tokens
(`1 <= k <= gamma + 1`), cost per token goes from `C` to roughly
`(gamma * c + C_block) / k` where `C_block` is the (slightly more expensive)
multi-token target forward. This is why *acceptance rate is everything*:
structured, predictable text (ReAct trajectories, JSON arrays, repeated
status lines) yields acceptance near 100% and `k ≈ gamma + 1`; free-form
prose with a weak draft yields 20-40% and `k ≈ 2`, where the extra draft
work can outweigh the saving. On a very fast GPU the fixed Python/kernel
launch overhead per round also matters. See the Benchmark section in the
README for measured numbers and the resulting analysis.

## Agent loops

### ReAct

* System prompt enumerates tools as `name(arg: type): description`,
  generated from the registry's JSON schemas.
* Responses are parsed defensively: code fences are stripped, the first
  balanced `{...}` block is extracted via a small brace matcher, and
  anything after a hallucinated `Observation:` is cut before parsing.
* Tool errors (unknown tool, bad JSON, exceptions) are returned as
  `Observation: Error: ...` so the model can self-correct instead of
  crashing the loop.

### Plan-and-Execute

One planning call produces a numbered list; each step is executed by a
*fresh* `ReActAgent` (bounded context, no cross-step contamination); a
final synthesis call merges the step results. `step_agent_factory` is
injectable for tests and for heterogeneous executors.

### Memory

* `ConversationMemory` - sliding window, system prompt pinned.
* `LongTermMemory` - embedding-free IDF keyword retrieval; the interface is
  deliberately vector-store-shaped so a real embedding backend can be
  dropped in without touching the agents.

## Safety

The built-in calculator is an AST interpreter with a whitelist of
operators, math functions and constants. No `eval`, no attribute access, no
name resolution beyond the whitelist. Tests assert hostile inputs
(`__import__`, `open`, attribute tricks) are rejected.
