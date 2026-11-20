"""Benchmark speculative decoding vs. plain autoregressive decoding.

Requires the ``hf`` extra and two models that share a tokenizer, e.g. a
Qwen2.5 pair (0.5B draft + 1.5B/7B target). Both paths are measured with
GPU-side argmax, so the comparison is apples-to-apples:

    python examples/speculative_benchmark.py \
        --target Qwen/Qwen2.5-7B-Instruct \
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
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from agentspec.llm.hf_backend import HFCausalLM, speculative_decode_hf

    print(f"Loading target: {args.target}", flush=True)
    target = HFCausalLM(args.target)
    print(f"Loading draft:  {args.draft}", flush=True)
    draft = HFCausalLM(args.draft)

    total_spec_time = total_auto_time = 0.0
    for prompt in PROMPTS:
        ids = target.encode(prompt)

        # --- speculative (batched target verification) ---
        target.reset_cache(); draft.reset_cache()
        t0 = time.perf_counter()
        result = speculative_decode_hf(
            target, draft, ids,
            max_new_tokens=args.max_new_tokens, gamma=args.gamma,
            temperature=args.temperature,
        )
        spec_time = time.perf_counter() - t0

        # --- plain autoregressive greedy (same GPU-side argmax) ---
        target.reset_cache()
        t0 = time.perf_counter()
        out = list(ids)
        for _ in range(args.max_new_tokens):
            nxt = int(target.logits_tensor(out).argmax().item())
            out.append(nxt)
        auto_time = time.perf_counter() - t0

        n = args.max_new_tokens
        total_spec_time += spec_time
        total_auto_time += auto_time
        print(f"\nPrompt: {prompt[:60]}...", flush=True)
        print(f"  speculative:    {spec_time:6.2f}s  ({n / spec_time:5.1f} tok/s)  "
              f"acceptance={result.stats.acceptance_rate:.0%}  "
              f"tokens/round={result.stats.tokens_per_round:.2f}", flush=True)
        print(f"  autoregressive: {auto_time:6.2f}s  ({n / auto_time:5.1f} tok/s)", flush=True)

    print("\n=========== summary ===========", flush=True)
    print(f"overall speedup: x{total_auto_time / total_spec_time:.2f}", flush=True)


if __name__ == "__main__":
    main()
