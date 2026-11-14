# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Normalized completeness and trace evidence for backend invocations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Literal


BackendEventKind = Literal[
    "request",
    "retry",
    "tool_call",
    "tool_result",
    "usage",
    "limit",
    "cancellation",
    "child_start",
    "child_terminal",
    "terminal",
]

BACKEND_EXECUTION_TRACE_VERSION = "assay.backend-execution-trace.v1"
_SHA256_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ExecutionCompleteness:
    """Whether the evidence needed to trust a terminal result is complete.

    Every field defaults to ``False``.  Existing third-party adapters
    therefore remain source compatible while being treated conservatively
    until they explicitly populate the new evidence.
    """

    token_usage_complete: bool = False
    cost_complete: bool = False
    terminal_complete: bool = False
    trace_complete: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "token_usage_complete",
            "cost_complete",
            "terminal_complete",
            "trace_complete",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(
                    f"ExecutionCompleteness.{field_name} must be bool"
                )

    @property
    def complete(self) -> bool:
        return (
            self.token_usage_complete
            and self.cost_complete
            and self.terminal_complete
            and self.trace_complete
        )


@dataclass(frozen=True)
class BackendExecutionEvent:
    """One redacted, normalized event in an invocation trace."""

    sequence: int
    kind: BackendEventKind
    name: str | None = None
    parent_id: str | None = None
    child_id: str | None = None

    _KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "request",
            "retry",
            "tool_call",
            "tool_result",
            "usage",
            "limit",
            "cancellation",
            "child_start",
            "child_terminal",
            "terminal",
        }
    )

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError(
                "BackendExecutionEvent.sequence must be an exact integer >= 0"
            )
        if self.kind not in self._KINDS:
            raise ValueError(
                f"unknown backend event kind {self.kind!r}; expected one of "
                f"{sorted(self._KINDS)}"
            )
        for field_name in ("name", "parent_id", "child_id"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"BackendExecutionEvent.{field_name} must be a non-empty "
                    "string when set"
                )
        if self.kind in {
            "tool_call",
            "tool_result",
            "limit",
            "cancellation",
        } and self.name is None:
            raise ValueError(
                f"BackendExecutionEvent kind {self.kind!r} requires name"
            )
        if self.kind in {"child_start", "child_terminal"} and self.child_id is None:
            raise ValueError(
                f"BackendExecutionEvent kind {self.kind!r} requires child_id"
            )


@dataclass(frozen=True)
class BackendExecutionTrace:
    """Bounded provider-neutral identity and lifecycle evidence.

    Raw prompts, responses, reasoning, tool arguments, and credentials do not
    belong in this record.  Provider adapters may retain those only in their
    separately governed, redacted debug sinks.
    """

    backend: str
    requested_model: str
    served_model: str | None = None
    requested_effort: str | None = None
    served_effort: str | None = None
    endpoint: str | None = None
    provider_request_id: str | None = None
    feature_profile_digest: str | None = None
    tool_authority_digest: str | None = None
    events: tuple[BackendExecutionEvent, ...] = ()
    schema_version: str = BACKEND_EXECUTION_TRACE_VERSION
    # Set only when a separately trusted provider-native instruction, agent,
    # or skill surface is deliberately enabled. Confined invocations leave it
    # unset rather than hashing ambient host state.
    trusted_context_digest: str | None = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("BackendExecutionTrace.backend must be non-empty")
        if (
            not isinstance(self.requested_model, str)
            or not self.requested_model.strip()
        ):
            raise ValueError(
                "BackendExecutionTrace.requested_model must be non-empty"
            )
        for field_name in (
            "served_model",
            "requested_effort",
            "served_effort",
            "endpoint",
            "provider_request_id",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"BackendExecutionTrace.{field_name} must be a non-empty "
                    "string when set"
                )
        for field_name in (
            "feature_profile_digest",
            "tool_authority_digest",
            "trusted_context_digest",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str)
                or _SHA256_DIGEST_RE.fullmatch(value) is None
            ):
                raise ValueError(
                    f"BackendExecutionTrace.{field_name} must be 'sha256:' "
                    "followed by 64 lowercase hexadecimal digits"
                )
        if self.schema_version != BACKEND_EXECUTION_TRACE_VERSION:
            raise ValueError(
                "BackendExecutionTrace.schema_version must be "
                f"{BACKEND_EXECUTION_TRACE_VERSION!r}"
            )
        if not isinstance(self.events, tuple):
            raise TypeError("BackendExecutionTrace.events must be a tuple")
        if any(type(event) is not BackendExecutionEvent for event in self.events):
            raise TypeError(
                "BackendExecutionTrace.events entries must be "
                "BackendExecutionEvent values"
            )
        sequences = tuple(event.sequence for event in self.events)
        if sequences != tuple(range(len(self.events))):
            raise ValueError(
                "BackendExecutionTrace event sequences must be contiguous "
                "and start at zero"
            )

    @property
    def terminal_event_count(self) -> int:
        return sum(event.kind == "terminal" for event in self.events)


__all__ = [
    "BACKEND_EXECUTION_TRACE_VERSION",
    "BackendEventKind",
    "BackendExecutionEvent",
    "BackendExecutionTrace",
    "ExecutionCompleteness",
]
