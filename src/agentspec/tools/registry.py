"""Tool registration and dispatch.

A tool is any Python callable. The registry derives a JSON-schema-ish
signature from type hints and the docstring, renders a prompt description
for the agent loops, and dispatches calls with robust argument parsing.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]


def _schema_for(func: Callable[..., Any]) -> dict[str, Any]:
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        ann = hints.get(name, str)
        properties[name] = {"type": _TYPE_MAP.get(ann, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class ToolRegistry:
    """A named collection of tools an agent may call."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Register a callable as a tool. Usable as a decorator."""

        def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or f.__name__
            if description is not None:
                desc = description
            else:
                doc = inspect.getdoc(f) or ""
                desc = doc.splitlines()[0] if doc else f"Tool {tool_name}"
            self._tools[tool_name] = ToolSpec(
                name=tool_name,
                description=desc,
                parameters=_schema_for(f),
                func=f,
            )
            return f

        return _wrap(func) if func is not None else _wrap

    # Allow ``registry.tool(...)`` as an alias decorator.
    tool = register

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-function-calling-style schemas for all tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def prompt_description(self) -> str:
        """Human-readable tool list for ReAct-style system prompts."""
        lines = []
        for spec in self._tools.values():
            args = ", ".join(
                f"{k}: {v['type']}" for k, v in spec.parameters["properties"].items()
            )
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict[str, Any] | str) -> str:
        """Run a tool by name and return its result as a string.

        ``arguments`` may be a dict or a JSON string. Any exception raised by
        the tool is converted into an ``"Error: ..."`` observation so the
        agent loop can recover instead of crashing.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {self.names}")
        if isinstance(arguments, str):
            arguments = arguments.strip()
            arguments = json.loads(arguments) if arguments else {}
        result = self._tools[name].func(**arguments)
        return result if isinstance(result, str) else str(result)


def tool(registry: ToolRegistry, func: Callable[..., Any] | None = None, **kwargs):
    """Convenience decorator: ``@tool(registry)`` registers a function."""
    return registry.register(func, **kwargs)
