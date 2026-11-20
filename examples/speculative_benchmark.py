"""Benchmark speculative decoding vs. plain autoregressive decoding.

Requires the ``hf`` extra and two models that share a tokenizer, e.g. a
Qwen2.5 pair (0.5B draft + 1.5B/3B target fits on a 6 GB GPU):

    python examples/speculative_benchmark.py \
        --target Qwen/Qwen2.5-1.5B-Instruct \
        --draft  Qwen/Qwen2.5-0.5B-Instruct \
        --gamma 4 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

PROMPTS = [
    "Explain what a KV-cache is in transformer inference, in two paragraphs.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the idea of speculative decoding in three sentences.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from agentspec.llm.hf_backend import HFCausalLM
    from agentspec.llm.speculative import speculative_decode

    print(f"Loading target: {args.target}")
    target = HFCausalLM(args.target)
    print(f"Loading draft:  {args.draft}")
    draft = HFCausalLM(args.draft)

    total_spec_time = total_auto_time = 0.0
    for prompt in PROMPTS:
        ids = target.encode(prompt)

        # --- speculative ---
        target.reset_cache(); draft.reset_cache()
        t0 = time.perf_counter()
        result = speculative_decode(
            target, draft, ids,
            max_new_tokens=args.max_new_tokens, gamma=args.gamma,
            temperature=0.0,
        )
        spec_time = time.perf_counter() - t0

        # --- plain autoregressive greedy ---
        target.reset_cache()
        t0 = time.perf_counter()
        out = list(ids)
        for _ in range(args.max_new_tokens):
            probs = target.next_token_probs(out)
            out.append(max(range(len(probs)), key=probs.__getitem__))
        auto_time = time.perf_counter() - t0

        n = args.max_new_tokens
        total_spec_time += spec_time
        total_auto_time += auto_time
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  speculative: {spec_time:6.2f}s  ({n / spec_time:5.1f} tok/s)  "
              f"acceptance={result.stats.acceptance_rate:.0%}  "
              f"tokens/round={result.stats.tokens_per_round:.2f}")
        print(f"  autoregressive: {auto_time:6.2f}s  ({n / auto_time:5.1f} tok/s)")

    print("\n=========== summary ===========")
    print(f"overall speedup: x{total_auto_time / total_spec_time:.2f}")


if __name__ == "__main__":
    main()
