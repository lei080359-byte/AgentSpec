"""Plan-and-Execute agent: decompose first, then execute step by step."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from agentspec.agent.react import AgentResult, ReActAgent
from agentspec.llm.base import ChatLLM, GenerationConfig
from agentspec.tools.registry import ToolRegistry

PLAN_PROMPT = """You are a planning agent. Break the task below into at most {max_steps} concrete,
self-contained steps that an executor agent can carry out one by one.

Output format - numbered lines only, no commentary:
1. <step>
2. <step>
...

Task: {task}"""

SYNTH_PROMPT = """You planned and executed a task. Combine the step results into one concise
final answer.

Task: {task}

Plan and results:
{report}

Final Answer:"""

_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s*(?P<step>.+?)\s*$", re.MULTILINE)


@dataclass
class PlanResult:
    answer: str | None
    plan: list[str] = field(default_factory=list)
    step_results: list[AgentResult] = field(default_factory=list)


class PlanExecuteAgent:
    """Plan-and-Execute: one planning call, then a fresh ReAct agent per step.

    ``step_agent_factory`` lets callers (and tests) inject a custom executor;
    by default each step gets a fresh :class:`ReActAgent` sharing the planner's
    LLM and tools.
    """

    def __init__(
        self,
        llm: ChatLLM,
        tools: ToolRegistry | None = None,
        max_plan_steps: int = 6,
        step_agent_factory: Callable[[], ReActAgent] | None = None,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_plan_steps = max_plan_steps
        if step_agent_factory is not None:
            self._step_agent_factory = step_agent_factory
        else:
            self._step_agent_factory = lambda: ReActAgent(
                self.llm, self.tools, max_steps=4
            )

    def run(self, task: str) -> PlanResult:
        plan_text = self.llm.generate(
            PLAN_PROMPT.format(max_steps=self.max_plan_steps, task=task),
            GenerationConfig(temperature=0.2),
        )
        steps = _STEP_RE.findall(plan_text)[: self.max_plan_steps]
        if not steps:  # the model refused to plan - fall back to direct ReAct
            result = self._step_agent_factory().run(task)
            return PlanResult(answer=result.answer, plan=[task], step_results=[result])

        results: list[AgentResult] = []
        for step in steps:
            results.append(self._step_agent_factory().run(step))

        report = "\n".join(
            f"{i}. {step}\n   -> {res.answer or '(no answer)'}"
            for i, (step, res) in enumerate(zip(steps, results), start=1)
        )
        answer = self.llm.generate(
            SYNTH_PROMPT.format(task=task, report=report),
            GenerationConfig(temperature=0.2),
        ).strip()
        answer = re.sub(r"^Final Answer\s*:\s*", "", answer, flags=re.IGNORECASE)
        return PlanResult(answer=answer, plan=steps, step_results=results)
