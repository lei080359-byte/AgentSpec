"""Built-in tools shipped with AgentSpec.

The calculator is evaluated through a restricted ``ast`` interpreter - no
``eval`` - so hostile or malformed expressions cannot touch the process.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone, timedelta

from agentspec.tools.registry import ToolRegistry

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("Only whitelisted math functions may be called")
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. "(3 + 4) * sqrt(16) / 2"."""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def get_current_time(utc_offset_hours: int = 0) -> str:
    """Return the current time, optionally shifted by a UTC offset in hours."""
    tz = timezone(timedelta(hours=utc_offset_hours))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z")


def word_count(text: str) -> str:
    """Count words (whitespace-separated) and characters in a text."""
    return f"words={len(text.split())} chars={len(text)}"


def build_default_registry() -> ToolRegistry:
    """A registry pre-loaded with the safe built-in tools."""
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(get_current_time)
    registry.register(word_count)
    return registry
