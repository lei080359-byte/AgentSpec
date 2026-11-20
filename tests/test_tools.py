import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentspec.tools.builtin import build_default_registry, calculator
from agentspec.tools.registry import ToolRegistry


class TestToolRegistry(unittest.TestCase):
    def test_schema_generation(self):
        registry = ToolRegistry()

        @registry.register
        def add(a: int, b: int = 1) -> int:
            """Add two integers."""
            return a + b

        schema = registry.schemas()[0]["function"]
        self.assertEqual(schema["name"], "add")
        self.assertEqual(schema["description"], "Add two integers.")
        self.assertEqual(schema["parameters"]["properties"]["a"]["type"], "integer")
        self.assertEqual(schema["parameters"]["required"], ["a"])

    def test_execute_with_dict_and_json_string(self):
        registry = build_default_registry()
        self.assertEqual(registry.execute("calculator", {"expression": "2+3*4"}), "14")
        self.assertEqual(registry.execute("calculator", '{"expression": "2+3*4"}'), "14")

    def test_unknown_tool_raises_keyerror(self):
        registry = build_default_registry()
        with self.assertRaises(KeyError):
            registry.execute("does_not_exist", {})

    def test_prompt_description_lists_tools(self):
        registry = build_default_registry()
        desc = registry.prompt_description()
        self.assertIn("calculator", desc)
        self.assertIn("word_count", desc)


class TestCalculator(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculator("2 + 3 * 4"), "14")
        self.assertEqual(calculator("(1 + 2) ** 3 / 9"), "3")
        self.assertEqual(calculator("sqrt(16)"), "4")
        self.assertEqual(calculator("pi"), str(__import__("math").pi))

    def test_rejects_dangerous_input(self):
        for evil in ("__import__('os')", "open('x')", "x = 1", "1; 2", "foo(1)"):
            with self.assertRaises((ValueError, SyntaxError)):
                calculator(evil)


if __name__ == "__main__":
    unittest.main()
