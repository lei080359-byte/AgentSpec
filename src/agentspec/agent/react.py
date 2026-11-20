"""ReAct agent loop (Yao et al., 2022) with robust output parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agentspec.llm.base import ChatLLM, GenerationConfig
from agentspec.tools.registry import ToolRegistry

REACT_SYSTEM_PROMPT = """You are a careful reasoning agent. Solve the task step by step.

You may use the following tools:
{tools}

Respond strictly in this format:

Thought: <your reasoning about what to do next>
Action: <tool name, must be one of the tools above>
Action Input: <a JSON object with the tool arguments>

After each Action you will receive an Observation with the tool result.
When you know the final answer, respond with:

Thought: <your reasoning>
Final Answer: <the answer>"""

_FINAL_RE = re.compile(r"Final Answer\s*:\s*(?P<answer>.*)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*:\s*`?(?P<name>[A-Za-z_][\w\-]*)`?", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input\s*:\s*(?P<raw>.*)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(
    r"Thought\s*:\s*(?P<thought>.*?)(?=\n\s*(?:Action|Final Answer)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences and trailing prose."""
    raw = raw.strip()
    fence = _FENCE_RE.match(raw)
    if fence:
        raw = fence.group("body").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"input": obj}
    except json.JSONDecodeError:
        pass
    # Brace-matching fallback: grab the first balanced {...} block.
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"Action Input is not a JSON object: {raw!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : i + 1])
    raise ValueError(f"Unbalanced JSON in Action Input: {raw!r}")


@dataclass
class AgentStep:
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    raw_response: str = ""


@dataclass
class AgentResult:
    answer: str | None
    steps: list[AgentStep] = field(default_factory=list)
    finish_reason: str = "final_answer"  # or "max_steps"
    trace: str = ""


class ReActAgent:
    """Reason-then-act loop: Thought -> Action -> Observation -> ... -> Answer."""

    def __init__(
        self,
        llm: ChatLLM,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
        generation_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.2, stop_sequences=["\nObservation:"]
        )
        self._system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.on_step = on_step

    def _build_prompt(self, task: str, scratchpad: str) -> str:
        system = self._system_prompt.format(tools=self.tools.prompt_description())
        prompt = f"{system}\n\nTask: {task}\n"
        if scratchpad:
            prompt += scratchpad
        return prompt

    def run(self, task: str) -> AgentResult:
        steps: list[AgentStep] = []
        scratchpad = ""

        for _ in range(self.max_steps):
            response = self.llm.generate(self._build_prompt(task, scratchpad), self.generation_config)
            # Guard against the model hallucinating its own observations.
            response = response.split("Observation:")[0].strip()
            scratchpad += ("\n" if scratchpad else "") + response

            final = _FINAL_RE.search(response)
            if final:
                return AgentResult(
                    answer=final.group("answer").strip(),
                    steps=steps,
                    finish_reason="final_answer",
                    trace=scratchpad,
                )

            thought_m = _THOUGHT_RE.search(response)
            action_m = _ACTION_RE.search(response)
            input_m = _INPUT_RE.search(response)
            step = AgentStep(
                thought=thought_m.group("thought").strip() if thought_m else "",
                raw_response=response,
            )

            if not action_m or not input_m:
                observation = (
                    "Error: could not parse your response. Follow the format exactly: "
                    "'Thought: ...', then 'Action: <name>' and 'Action Input: <json>', "
                    "or 'Final Answer: ...'."
                )
            else:
                step.action = action_m.group("name")
                try:
                    step.action_input = _extract_json_object(input_m.group("raw"))
                    observation = self.tools.execute(step.action, step.action_input)
                except (ValueError, KeyError, TypeError) as exc:
                    observation = f"Error: {exc}"

            step.observation = observation
            steps.append(step)
            if self.on_step:
                self.on_step(step)
            scratchpad += f"\nObservation: {observation}\n"

        return AgentResult(answer=None, steps=steps, finish_reason="max_steps", trace=scratchpad)
