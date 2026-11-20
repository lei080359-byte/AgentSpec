import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentspec.agent.planner import PlanExecuteAgent
from agentspec.agent.react import AgentResult, ReActAgent
from agentspec.llm.mock import MockChatLLM
from agentspec.tools.builtin import build_default_registry


class TestReActAgent(unittest.TestCase):
    def test_two_step_tool_use_then_answer(self):
        llm = MockChatLLM([
            'Thought: I should compute the product.\n'
            'Action: calculator\n'
            'Action Input: {"expression": "6*7"}',
            'Thought: I have the result.\nFinal Answer: 42',
        ])
        agent = ReActAgent(llm, build_default_registry())
        result = agent.run("What is 6 times 7?")
        self.assertEqual(result.answer, "42")
        self.assertEqual(result.finish_reason, "final_answer")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].action, "calculator")
        self.assertEqual(result.steps[0].observation, "42")

    def test_tool_error_is_fed_back_as_observation(self):
        llm = MockChatLLM([
            'Action: nope\nAction Input: {}',
            'Final Answer: recovered',
        ])
        agent = ReActAgent(llm, build_default_registry())
        result = agent.run("anything")
        self.assertIn("Unknown tool", result.steps[0].observation)
        self.assertEqual(result.answer, "recovered")

    def test_json_in_code_fence_is_parsed(self):
        llm = MockChatLLM([
            'Action: calculator\nAction Input: ```json\n{"expression": "1+1"}\n```',
            'Final Answer: 2',
        ])
        agent = ReActAgent(llm, build_default_registry())
        result = agent.run("1+1?")
        self.assertEqual(result.steps[0].observation, "2")

    def test_max_steps_fallback(self):
        llm = MockChatLLM(["I have no idea what format to use."] * 10)
        agent = ReActAgent(llm, build_default_registry(), max_steps=3)
        result = agent.run("anything")
        self.assertIsNone(result.answer)
        self.assertEqual(result.finish_reason, "max_steps")
        self.assertEqual(len(result.steps), 3)


class TestPlanExecuteAgent(unittest.TestCase):
    def test_plan_then_execute_then_synthesize(self):
        llm = MockChatLLM([
            # 1) planning call
            "1. Compute 6*7\n2. Compute 10+5",
            # 2) final synthesis call
            "Final Answer: 42 and 15",
        ])

        def fake_executor():
            class _Fake:
                def __init__(self):
                    self.calls = 0

                def run(self, step):
                    self.calls += 1
                    value = "42" if "6*7" in step else "15"
                    return AgentResult(answer=value)

            return _Fake()

        agent = PlanExecuteAgent(llm, build_default_registry(), step_agent_factory=fake_executor)
        result = agent.run("compute two things")
        self.assertEqual(result.plan, ["Compute 6*7", "Compute 10+5"])
        self.assertEqual(len(result.step_results), 2)
        self.assertEqual(result.answer, "42 and 15")


if __name__ == "__main__":
    unittest.main()
