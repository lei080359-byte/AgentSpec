"""AgentSpec - a lightweight LLM agent framework with speculative-decoding acceleration."""

from agentspec.agent.memory import ConversationMemory, LongTermMemory
from agentspec.agent.planner import PlanExecuteAgent
from agentspec.agent.react import AgentResult, AgentStep, ReActAgent
from agentspec.llm.base import ChatLLM, GenerationConfig, TokenModel
from agentspec.llm.mock import MockChatLLM, MockTokenModel
from agentspec.llm.speculative import SpeculativeResult, SpeculativeStats, speculative_decode
from agentspec.tools.builtin import build_default_registry
from agentspec.tools.registry import ToolRegistry, tool

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "AgentStep",
    "ChatLLM",
    "ConversationMemory",
    "GenerationConfig",
    "LongTermMemory",
    "MockChatLLM",
    "MockTokenModel",
    "PlanExecuteAgent",
    "ReActAgent",
    "SpeculativeResult",
    "SpeculativeStats",
    "TokenModel",
    "ToolRegistry",
    "build_default_registry",
    "speculative_decode",
    "tool",
]
