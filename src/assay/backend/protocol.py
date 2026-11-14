# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Structural protocols implemented by model-backend adapters.

Two surfaces, split so adapters can opt into the more advanced one
without breaking minimal callers:

- ``Backend``           — single agent call via ``invoke(invocation)``.
                          The vendor SDK may run its own autonomous
                          tool-use loop internally; assay sees only
                          the final result.
- ``ChatToolsBackend``  — tool-capable call with an explicitly supplied
                          ``ToolSet`` via
                          ``chat_with_tools(invocation, toolset, *,
                          cache=None, max_rounds=25)``. Adapters that
                          want this capability implement both.
                          Callers narrow via
                          ``isinstance(backend, ChatToolsBackend)``.

Assay intentionally ships no concrete adapter; consumers provide them. Because
these are ``runtime_checkable`` structural protocols, runtime checks verify the
presence of members, not their signatures or behavioral correctness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..types.invocation import Invocation, InvocationResult

if TYPE_CHECKING:
    from ..primitives.tool_cache import ToolCallCache
    from ..types.tools import ToolSet


@runtime_checkable
class Backend(Protocol):
    name: str
    model: str

    def invoke(self, invocation: Invocation) -> InvocationResult: ...


@runtime_checkable
class ChatToolsBackend(Backend, Protocol):
    """Backend accepting an explicit ToolSet for a tool-capable invocation.

    Concrete adapters may drive a direct API loop or delegate to an autonomous
    SDK/CLI loop. ``cache`` and ``max_rounds`` are portable request controls,
    but this structural protocol cannot prove a concrete enforcement level.
    An adapter must either enforce a supplied control according to its declared
    feature profile or reject it before transport; silently ignoring a supplied
    control is non-conforming. Callers needing hard limits must preflight an
    adapter whose documented policy provides those guarantees.

    The method returns the same ``InvocationResult`` shape as ``invoke``. The
    adapter owns vendor conversion, dispatch transport, transcript capture,
    and error mapping.
    """

    def chat_with_tools(
        self,
        invocation: Invocation,
        toolset: ToolSet,
        *,
        cache: ToolCallCache | None = None,
        max_rounds: int = 25,
    ) -> InvocationResult: ...
