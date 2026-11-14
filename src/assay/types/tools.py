# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tool descriptors and the dual-vendor tool-set protocol for Assay.

``Tool`` is a single descriptor (name + description + JSON-schema for
arguments). ``ToolSet`` is the protocol an aggregate of tools must
implement so a Backend can drive a tool-use chat loop:

- ``to_anthropic_tools()`` — emit the tool list in Anthropic's
  Messages-API tool-use shape.
- ``to_openai_tools()``    — emit in OpenAI's Chat Completions
  function-calling shape.
- ``dispatch(name, args)`` — execute one tool call in-process and
  return the stringified result that goes back into the conversation
  as a tool-use result block.
- ``mcp_config()``         — required method returning optional bridge config.
  When non-``None``, a backend may attach the consumer-owned MCP server so
  model-side tools resolve through the same tool set.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .frozen import freeze_dict


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", freeze_dict(self.input_schema))


@runtime_checkable
class ToolSet(Protocol):
    """Aggregate of tools the model may call within a chat-with-tools loop.

    Implementations live in consumer packages. The protocol explicitly
    requires both Anthropic- and OpenAI-shaped descriptor renderers so either
    adapter can choose its format without Assay owning a vendor SDK.
    """

    def to_anthropic_tools(self) -> list[dict]:
        """Return the tool list in Anthropic's Messages-API shape."""

    def to_openai_tools(self) -> list[dict]:
        """Return the tool list in OpenAI's Chat Completions shape."""

    def dispatch(self, name: str, arguments: dict) -> str:
        """Execute one tool call in-process. Return the result string.

        Implementations should:
        - validate ``name`` against the declared tool set
        - validate ``arguments`` against the tool's input_schema (or
          tolerate loose input — caller's choice)
        - raise on unknown names or invalid arguments
        - return a string the backend can wrap in a tool_use_result
          block (no further serialization)
        """

    def mcp_config(self) -> Any | None:
        """Optional MCP-server bridge config; ``None`` if not provided."""
