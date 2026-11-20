"""Run a ReAct agent against a task.

Backends:

* ``mock`` (default) - a scripted model, no GPU / no API key needed.
* ``api``            - any OpenAI-compatible endpoint (OPENAI_API_KEY / OPENAI_BASE_URL).
* ``hf``             - a local HuggingFace instruct model (needs the ``hf`` extra).

Examples:
    python examples/react_demo.py --backend mock
    python examples/react_demo.py --backend api --model deepseek-chat \
        --task "What is (128 * 46) + sqrt(81)?"
    python examples/react_demo.py --backend hf --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentspec import ReActAgent, build_default_registry


def build_llm(args):
    if args.backend == "mock":
        from agentspec.llm.mock import MockChatLLM

        return MockChatLLM([
            "Thought: I need the product of 128 and 46 first.\n"
            "Action: calculator\n"
            'Action Input: {"expression": "128 * 46"}',
            "Thought: Now add the square root of 81.\n"
            "Action: calculator\n"
            'Action Input: {"expression": "5888 + sqrt(81)"}',
            "Thought: The computation is complete.\nFinal Answer: 5897",
        ])
    if args.backend == "api":
        from agentspec.llm.api_backend import OpenAICompatibleChatLLM

        return OpenAICompatibleChatLLM(model=args.model)
    if args.backend == "hf":
        from agentspec.llm.hf_backend import HFChatLLM

        return HFChatLLM(args.model)
    raise ValueError(args.backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "api", "hf"], default="mock")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--task", default="What is (128 * 46) + sqrt(81)?")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    llm = build_llm(args)

    def log(step):
        print(f"\n--- step ---\nThought: {step.thought}")
        print(f"Action: {step.action}  Input: {step.action_input}")
        print(f"Observation: {step.observation}")

    agent = ReActAgent(llm, build_default_registry(), max_steps=args.max_steps, on_step=log)
    result = agent.run(args.task)

    print("\n===================")
    print(f"Task:   {args.task}")
    print(f"Answer: {result.answer}")
    print(f"Finish: {result.finish_reason} ({len(result.steps)} steps)")


if __name__ == "__main__":
    main()
