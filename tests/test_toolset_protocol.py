# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``ToolSet`` protocol + ``Backend.chat_with_tools`` surface."""

from __future__ import annotations

import pytest

from assay import (
    Backend,
    ChatToolsBackend,
    Invocation,
    InvocationResult,
    Tool,
    ToolSet,
    Usage,
)


# ----------------------------------------------------------------------
# ToolSet protocol — runtime conformance
# ----------------------------------------------------------------------


class _SourceToolSet:
    """Reference ToolSet for protocol-conformance tests."""

    def __init__(self):
        self.dispatched: list[tuple[str, dict]] = []

    def to_anthropic_tools(self) -> list[dict]:
        return [
            {
                "name": "read_source",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    def to_openai_tools(self) -> list[dict]:
        return [{"type": "function", "function": self.to_anthropic_tools()[0]}]

    def dispatch(self, name: str, arguments: dict) -> str:
        self.dispatched.append((name, arguments))
        if name == "read_source":
            return f"<contents of {arguments['path']}>"
        raise ValueError(f"unknown tool {name!r}")

    def mcp_config(self):
        return None


def test_toolset_runtime_conformance():
    ts = _SourceToolSet()
    assert isinstance(ts, ToolSet)


def test_toolset_emits_anthropic_shape():
    ts = _SourceToolSet()
    out = ts.to_anthropic_tools()
    assert isinstance(out, list)
    assert out[0]["name"] == "read_source"
    assert "input_schema" in out[0]


def test_toolset_emits_openai_shape():
    ts = _SourceToolSet()
    out = ts.to_openai_tools()
    assert out[0]["type"] == "function"


def test_toolset_dispatch_executes():
    ts = _SourceToolSet()
    result = ts.dispatch("read_source", {"path": "x.py"})
    assert result == "<contents of x.py>"
    assert ts.dispatched == [("read_source", {"path": "x.py"})]


def test_toolset_dispatch_rejects_unknown_name():
    ts = _SourceToolSet()
    with pytest.raises(ValueError, match="unknown tool"):
        ts.dispatch("nonexistent", {})


def test_toolset_mcp_config_optional():
    ts = _SourceToolSet()
    assert ts.mcp_config() is None


# ----------------------------------------------------------------------
# Backend vs ChatToolsBackend protocol narrowing
# ----------------------------------------------------------------------


class _MinimalBackend:
    """Implements only ``invoke`` — Backend conformant, NOT ChatToolsBackend."""

    name = "minimal"
    model = "minimal-model"

    def invoke(self, invocation: Invocation) -> InvocationResult:
        return InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )


class _ToolsBackend(_MinimalBackend):
    """Implements both surfaces."""

    name = "tools"
    model = "tools-model"

    def chat_with_tools(
        self, invocation, toolset, *, cache=None, max_rounds=25,
    ) -> InvocationResult:
        toolset.dispatch("read_source", {"path": "noop.py"})
        return InvocationResult(
            final_message="ok", transcript=(), usage=Usage(), stop_reason="end_turn",
        )


def test_minimal_backend_is_backend_but_not_chat_tools():
    """A backend with only invoke() should satisfy Backend, not ChatToolsBackend."""
    b = _MinimalBackend()
    assert isinstance(b, Backend)
    assert not isinstance(b, ChatToolsBackend)


def test_tools_backend_satisfies_both_protocols():
    b = _ToolsBackend()
    assert isinstance(b, Backend)
    assert isinstance(b, ChatToolsBackend)


def test_chat_tools_backend_drives_dispatch():
    """A ChatToolsBackend's chat_with_tools should be callable with a ToolSet."""
    ts = _SourceToolSet()
    b = _ToolsBackend()
    result = b.chat_with_tools(
        Invocation(system="s", user="u", tools=None, max_turns=5, budget_usd=None),
        ts,
    )
    assert result.final_message == "ok"
    assert ts.dispatched == [("read_source", {"path": "noop.py"})]


# ----------------------------------------------------------------------
# Tool dataclass still works as a separate single-descriptor type
# ----------------------------------------------------------------------


def test_tool_dataclass_unchanged():
    t = Tool(name="x", description="d", input_schema={"type": "object"})
    assert t.name == "x"
    # frozen
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "y"  # type: ignore[misc]
