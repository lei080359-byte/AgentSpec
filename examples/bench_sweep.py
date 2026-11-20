"""Analyze where speculative decoding actually pays off.

Sweeps ``gamma`` over two prompt sets - free-form general text and highly
structured agent trajectories - and reports per-prompt acceptance, tokens per
round, and wall-clock speedup. Includes an untimed warmup pass so CUDA kernel
compilation does not pollute the first measurement.

    python examples/bench_sweep.py \
        --target /path/to/Qwen2.5-7B-Instruct \
        --draft  /path/to/Qwen2.5-0.5B-Instruct \
        --gammas 2,3,4,6 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

GENERAL_PROMPTS = [
    "Explain what a KV-cache is in transformer inference, in two paragraphs.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Summarize the idea of speculative decoding in three sentences.",
]

AGENT_PROMPTS = [
    "You are a ReAct agent. Continue this trajectory in exactly the same format:\n"
    "Thought: I need to compute 2+2.\nAction: calculator\n"
    'Action Input: {"expression": "2+2"}\nObservation: 4\n'
    "Thought: I need to compute 3+3.\nAction: calculator\n"
    'Action Input: {"expression": "3+3"}\nObservation: 6\n',
    "Output a JSON array of items with the exact same shape as below:\n"
    '[{"id": 1, "name": "item_1", "status": "ok"},\n'
    '{"id": 2, "name": "item_2", "status": "ok"},\n',
    "Repeat the following status line verbatim on every line:\n"
    "Step complete: tool call succeeded, observation recorded.\n"
    "Step complete: tool call succeeded, observation recorded.\n",
]


def measure(target, draft, prompts, gamma, n):
    from agentspec.llm.hf_backend import speculative_decode_hf

    rows = []
    spec_total = auto_total = 0.0
    for label, prompt in prompts:
        ids = target.encode(prompt)

        target.reset_cache(); draft.reset_cache()
        t0 = time.perf_counter()
        result = speculative_decode_hf(target, draft, ids, max_new_tokens=n, gamma=gamma)
        st = time.perf_counter() - t0

        target.reset_cache()
        t0 = time.perf_counter()
        out = list(ids)
        for _ in range(n):
            out.append(int(target.logits_tensor(out).argmax().item()))
        at = time.perf_counter() - t0

        spec_total += st
        auto_total += at
        rows.append((label, st, at, result.stats.acceptance_rate, result.stats.tokens_per_round))
    return rows, spec_total, auto_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--gammas", default="2,3,4,6")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-set", choices=["general", "agent", "all"], default="all")
    args = parser.parse_args()

    from agentspec.llm.hf_backend import HFCausalLM

    target = HFCausalLM(args.target)
    draft = HFCausalLM(args.draft)
    general = [("general-1", GENERAL_PROMPTS[0]), ("general-2", GENERAL_PROMPTS[1]),
               ("general-3", GENERAL_PROMPTS[2])]
    agent = [("agent-1", AGENT_PROMPTS[0]), ("agent-2", AGENT_PROMPTS[1]),
             ("agent-3", AGENT_PROMPTS[2])]
    sets = {"general": [("general", general)], "agent": [("agent", agent)],
            "all": [("general", general), ("agent", agent)]}[args.prompt_set]

    # Warmup: kernel compilation / allocator stabilization, not timed.
    from agentspec.llm.hf_backend import speculative_decode_hf
    warm = target.encode("Warmup: ")
    target.reset_cache(); draft.reset_cache()
    speculative_decode_hf(target, draft, warm, max_new_tokens=8, gamma=4)
    target.reset_cache()
    w = list(warm)
    for _ in range(8):
        w.append(int(target.logits_tensor(w).argmax().item()))
    print("warmup done", flush=True)

    for gamma in [int(g) for g in args.gammas.split(",")]:
        for set_name, prompts in sets:
            rows, st, at = measure(target, draft, prompts, gamma, args.max_new_tokens)
            print(f"\ngamma={gamma} set={set_name}", flush=True)
            for label, s, a, acc, tpr in rows:
                print(f"  {label:10s} spec={s:6.2f}s auto={a:6.2f}s "
                      f"accept={acc:5.0%} tpr={tpr:4.2f} speedup=x{a/s:4.2f}", flush=True)
            print(f"  OVERALL speedup=x{at/st:.2f}", flush=True)


if __name__ == "__main__":
    main()
