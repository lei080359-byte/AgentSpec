import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agentspec.agent.memory import ConversationMemory, LongTermMemory


class TestConversationMemory(unittest.TestCase):
    def test_sliding_window(self):
        mem = ConversationMemory(window_size=3, system_prompt="sys")
        for i in range(5):
            mem.add("user", f"msg{i}")
        self.assertEqual([m.content for m in mem.messages], ["msg2", "msg3", "msg4"])

    def test_system_prompt_preserved_in_chat_view(self):
        mem = ConversationMemory(window_size=2, system_prompt="sys")
        mem.add("user", "hello")
        chat = mem.as_chat()
        self.assertEqual(chat[0], {"role": "system", "content": "sys"})
        self.assertEqual(chat[1]["content"], "hello")


class TestLongTermMemory(unittest.TestCase):
    def test_retrieve_ranks_by_relevance(self):
        mem = LongTermMemory()
        mem.add("speculative decoding accelerates LLM inference with a draft model")
        mem.add("the weather forecast predicts rain tomorrow")
        mem.add("draft models in speculative decoding must share the tokenizer")
        hits = mem.retrieve("speculative decoding draft model", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("speculative decoding", hits[0][1])

    def test_no_match_returns_empty(self):
        mem = LongTermMemory()
        mem.add("completely unrelated content")
        self.assertEqual(mem.retrieve("zzzz qqqq"), [])


if __name__ == "__main__":
    unittest.main()
