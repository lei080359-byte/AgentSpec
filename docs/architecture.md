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

**Cost model.** If a target forward costs `C`, a draft forward costs
`c << C`, and the average number of committed tokens per round is `k`
(`1 <= k <= gamma + 1`), then cost per token drops from `C` to roughly
`(gamma * c + C) / k`. With `gamma=4`, a 10x cheaper draft, and a 70%
acceptance rate, `k ≈ 2.6` and the theoretical saving is about 2x.

The HuggingFace backend (`llm/hf_backend.py`) keeps an internal KV-cache
and only forwards *new* tokens when the prefix grows, which matches the
access pattern of the speculative loop. A further optimization (roadmap) is
to score all `gamma` positions in a single batched forward pass.

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
