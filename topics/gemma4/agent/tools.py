"""
Tool implementations for the Gemma 4 agent demo.

Each tool has:
  - a JSON-schema description (fed to the model via `ollama.chat(tools=...)`)
  - a Python callable that takes the model's arguments dict and returns a
    string result (what we paste back into the conversation as tool output).
"""

from __future__ import annotations

import ast
import datetime as dt
import operator
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Safe calculator — AST-based, no eval(), no builtins, math ops only.
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def calculator(args: dict[str, Any]) -> str:
    expr = str(args.get("expr", "")).strip()
    if not expr:
        return "error: empty expression"
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"error: {e}"


# ---------------------------------------------------------------------------
# Current datetime.
# ---------------------------------------------------------------------------

def current_datetime(_args: dict[str, Any]) -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo).
# ---------------------------------------------------------------------------

def web_search(args: dict[str, Any]) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        return "error: empty query"
    try:
        from ddgs import DDGS
    except ImportError:
        return "error: ddgs package not installed"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"error: {e}"
    if not results:
        return "no results"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File read, sandboxed to ./sandbox/ next to this script.
# ---------------------------------------------------------------------------

SANDBOX = Path(__file__).parent / "sandbox"


def read_file(args: dict[str, Any]) -> str:
    name = str(args.get("path", "")).strip()
    if not name:
        return "error: empty path"
    SANDBOX.mkdir(exist_ok=True)
    # Resolve and assert the path stays inside SANDBOX.
    target = (SANDBOX / name).resolve()
    try:
        target.relative_to(SANDBOX.resolve())
    except ValueError:
        return "error: path escapes sandbox"
    if not target.exists():
        available = sorted(p.name for p in SANDBOX.iterdir()) if SANDBOX.exists() else []
        return f"error: not found. Available files: {available}"
    try:
        text = target.read_text(errors="replace")
    except Exception as e:
        return f"error: {e}"
    # Cap output so the model's context doesn't explode on a huge file.
    if len(text) > 20_000:
        text = text[:20_000] + "\n...[truncated]"
    return text


def list_files(_args: dict[str, Any]) -> str:
    SANDBOX.mkdir(exist_ok=True)
    files = sorted(p.name for p in SANDBOX.iterdir() if p.is_file())
    return "\n".join(files) if files else "(sandbox is empty)"


# ---------------------------------------------------------------------------
# Registry — names, schemas, and callables wired together.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+, -, *, /, //, %, **). Numbers only — no variables or functions.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "e.g. '847 * 293' or '(1+2)**10'"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Return the current date and time in ISO 8601 format.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo. Returns up to 5 results with title, snippet, URL.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files available in the sandbox directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the sandbox directory. Use list_files first to discover what's there.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Filename relative to the sandbox directory."}},
                "required": ["path"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {
    "calculator": calculator,
    "current_datetime": current_datetime,
    "web_search": web_search,
    "list_files": list_files,
    "read_file": read_file,
}
