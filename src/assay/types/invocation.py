# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Backend invocation primitives for assay.

An ``Invocation`` is the work-unit description handed to a Backend;
an ``InvocationResult`` is what comes back. Adapters serialize
``Invocation`` into vendor-native formats and parse responses back
into ``InvocationResult``. Concrete error records provide the standard
failure taxonomy. Adapter-specific failures can satisfy the structural
``BackendError`` protocol when they expose ``kind``, ``message``, and
``retriable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .policy import BackendPolicy
from .tools import Tool
from .backend_execution import BackendExecutionTrace, ExecutionCompleteness
from .cost_bounds import UsageCostBounds

@dataclass(frozen=True)
class Usage:
    """Aggregate billable usage for one logical invocation.

    ``input_tokens`` is the total input-token count. When cache telemetry is
    available, ``cache_creation_tokens`` and ``cache_read_tokens`` are
    non-overlapping subsets of that total rather than additional tokens.
    Cache fields are additive and canonically omitted at zero so legacy
    records retain their original bytes and content hashes.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_creation_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_default": True},
    )
    cache_read_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_default": True},
    )

    cost_bounds: UsageCostBounds | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    def __post_init__(self) -> None:
        if self.cost_bounds is not None and type(self.cost_bounds) is not UsageCostBounds:
            raise TypeError("Usage.cost_bounds must be UsageCostBounds or None")

    @property
    def budget_cost_usd(self) -> float:
        """Conservative debit, not an assertion of the actual charge."""

        if self.cost_bounds is not None and self.cost_bounds.covers(self):
            return self.cost_bounds.upper_bound_usd
        return self.cost_usd


@dataclass(frozen=True)
class UsagePricingSegment:
    """Exact pricing inputs and result for one ordered provider segment.

    A segment normally represents one physical provider response. The two
    cache counters are non-overlapping subsets of the inclusive
    ``input_tokens`` total. Keeping segments separate lets a consumer replay
    pricing when one logical invocation crosses a provider pricing tier.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"UsagePricingSegment.{field_name} must be an exact "
                    "non-negative integer"
                )
        if (
            self.cache_creation_tokens + self.cache_read_tokens
            > self.input_tokens
        ):
            raise ValueError(
                "UsagePricingSegment cache-token subsets must not exceed "
                "input_tokens"
            )
        if (
            type(self.cost_usd) not in (int, float)
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError(
                "UsagePricingSegment.cost_usd must be a finite "
                "non-negative number"
            )


@dataclass(frozen=True)
class UsagePricingEvidence:
    """Ordered, prompt-free pricing evidence for one logical invocation."""

    segments: tuple[UsagePricingSegment, ...]
    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TypeError("UsagePricingEvidence.segments must be a tuple")
        if any(
            type(segment) is not UsagePricingSegment
            for segment in self.segments
        ):
            raise TypeError(
                "UsagePricingEvidence.segments entries must be "
                "UsagePricingSegment values"
            )
        if type(self.complete) is not bool:
            raise TypeError("UsagePricingEvidence.complete must be a bool")

    @property
    def usage(self) -> Usage:
        """Return the exact aggregate represented by the ordered segments."""

        return Usage(
            input_tokens=sum(segment.input_tokens for segment in self.segments),
            output_tokens=sum(segment.output_tokens for segment in self.segments),
            cost_usd=math.fsum(segment.cost_usd for segment in self.segments),
            cache_creation_tokens=sum(
                segment.cache_creation_tokens for segment in self.segments
            ),
            cache_read_tokens=sum(
                segment.cache_read_tokens for segment in self.segments
            ),
        )

    def reconciles(self, usage: Usage) -> bool:
        """Return whether the segments exactly explain aggregate ``usage``."""

        aggregate = self.usage
        return (
            aggregate.input_tokens == usage.input_tokens
            and aggregate.output_tokens == usage.output_tokens
            and aggregate.cache_creation_tokens == usage.cache_creation_tokens
            and aggregate.cache_read_tokens == usage.cache_read_tokens
            and math.isclose(
                aggregate.cost_usd,
                usage.cost_usd,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )


@dataclass(frozen=True)
class Message:
    role: str
    # Prompt / completion text is repr-excluded: message content
    # can carry credential-shaped text, and a bare repr() in a traceback
    # or SdkDebug dump would leak it. Redaction of *persisted* blobs is a
    # workflow responsibility (assay._safety.redact_secrets); this just
    # closes the incidental repr/log leak.
    content: str = field(repr=False)


@dataclass(frozen=True)
class Invocation:
    # system/user prompts are repr-excluded for the same reason as
    # Message.content — they routinely interpolate captured
    # blobs, env-derived context, and occasionally secrets.
    system: str = field(repr=False)
    user: str = field(repr=False)
    tools: tuple[Tool, ...] | None
    max_turns: int
    budget_usd: float | None
    policy: BackendPolicy | None = None
    # Optional vendor-neutral JSON Schema for the final assistant message.
    # Backends with native constrained decoding should pass it through; other
    # backends may ignore it because the caller still validates the response.
    # Keeping the schema on the invocation makes structured output an explicit
    # capability instead of burying the only contract in prompt prose.
    response_schema: Mapping[str, Any] | None = None
    # Optional identity- and operation-bound provider continuity. The opaque
    # identifier is persistence material, never an authorization credential.
    conversation: "ConversationTransportRequest | None" = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 1
        ):
            raise ValueError("Invocation.max_turns must be an integer >= 1")
        budget = self.budget_usd
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget < 0
        ):
            raise ValueError(
                "Invocation.budget_usd must be None or a finite "
                "non-negative number"
            )
        if self.response_schema is not None and not isinstance(
            self.response_schema, Mapping
        ):
            raise TypeError("Invocation.response_schema must be a mapping")
        if self.conversation is not None:
            from .conversation import ConversationTransportRequest

            if not isinstance(
                self.conversation,
                ConversationTransportRequest,
            ):
                raise TypeError(
                    "Invocation.conversation must be a "
                    "ConversationTransportRequest or None"
                )


@runtime_checkable
class BackendError(Protocol):
    """Structural error contract carried by ``InvocationResult.error``."""

    kind: str
    message: str
    retriable: bool


# Discriminator field ``kind`` comes last so all preceding fields can
# carry defaults without ordering issues. Each subtype's defaults
# match the most common shape callers construct.
@dataclass(frozen=True)
class TransportError:
    http_status: int | None = None
    message: str = ""
    retriable: bool = True
    kind: str = "transport"


@dataclass(frozen=True)
class BudgetExhausted:
    spent_usd: float = 0.0
    limit_usd: float = 0.0
    message: str = ""
    retriable: bool = False
    kind: str = "budget_exhausted"


@dataclass(frozen=True)
class MaxTurnsExceeded:
    turns_used: int = 0
    turns_limit: int = 0
    message: str = ""
    retriable: bool = False
    kind: str = "max_turns"


@dataclass(frozen=True)
class ContentRejected:
    message: str = ""
    retriable: bool = False
    kind: str = "content_rejected"


@dataclass(frozen=True)
class ExternalCancellation:
    cause: str = "unknown"
    message: str = ""
    retriable: bool = True
    kind: str = "cancellation"


@dataclass(frozen=True)
class IncompleteEvidence:
    """A nominally successful result lacks required promotion evidence."""

    missing: tuple[str, ...] = ()
    message: str = ""
    retriable: bool = False
    kind: str = "incomplete_evidence"

    def __post_init__(self) -> None:
        if not self.missing:
            raise ValueError("IncompleteEvidence.missing must not be empty")
        if any(
            not isinstance(field_name, str) or not field_name.strip()
            for field_name in self.missing
        ):
            raise ValueError(
                "IncompleteEvidence.missing entries must be non-empty strings"
            )


@dataclass(frozen=True)
class StaleConversation:
    """The provider rejected an explicitly requested conversation handle."""

    message: str = ""
    retriable: bool = False
    kind: str = "stale_conversation"


@dataclass(frozen=True)
class InvocationResult:
    """Terminal backend result, including typed failure and usage evidence."""

    final_message: str
    transcript: tuple[Message, ...]
    usage: Usage
    stop_reason: str
    error: "BackendError | None" = None
    # Additive execution evidence. Older adapters remain source-compatible,
    # but their default is deliberately incomplete rather than silently
    # treating absent provider usage/terminal data as trustworthy zeroes.
    completeness: ExecutionCompleteness = field(
        default_factory=ExecutionCompleteness,
        metadata={
            "persist_optional": True,
        },
    )
    execution_trace: BackendExecutionTrace | None = field(
        default=None,
        metadata={
            "persist_optional": True,
        },
    )
    conversation: "ConversationTransportOutcome | None" = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )
    # Ordered physical-response pricing inputs. ``None`` preserves legacy
    # adapters; an explicit incomplete record can retain partial segments
    # without claiming that they reconcile aggregate Usage.
    pricing_evidence: UsagePricingEvidence | None = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )

    @property
    def cost_bounds(self) -> UsageCostBounds | None:
        return self.usage.cost_bounds

    def __post_init__(self) -> None:
        if type(self.usage) is not Usage:
            raise TypeError("InvocationResult.usage must be an assay Usage")
        if type(self.completeness) is not ExecutionCompleteness:
            raise TypeError(
                "InvocationResult.completeness must be an "
                "ExecutionCompleteness"
            )
        if self.cost_bounds is not None and any(
            segment.cache_read_tokens is None for segment in self.cost_bounds.segments
        ) and (self.completeness.token_usage_complete or self.completeness.cost_complete):
            raise ValueError("unknown cache reads cannot have complete actual accounting")
        if self.execution_trace is not None and (
            type(self.execution_trace) is not BackendExecutionTrace
        ):
            raise TypeError(
                "InvocationResult.execution_trace must be a "
                "BackendExecutionTrace or None"
            )
        if not isinstance(self.transcript, tuple):
            raise TypeError("InvocationResult.transcript must be a tuple")
        if any(type(message) is not Message for message in self.transcript):
            raise TypeError(
                "InvocationResult.transcript entries must be Message values"
            )
        if self.conversation is not None:
            from .conversation import ConversationTransportOutcome

            if type(self.conversation) is not ConversationTransportOutcome:
                raise TypeError(
                    "InvocationResult.conversation must be a "
                    "ConversationTransportOutcome or None"
                )
        if self.pricing_evidence is not None and (
            type(self.pricing_evidence) is not UsagePricingEvidence
        ):
            raise TypeError(
                "InvocationResult.pricing_evidence must be a "
                "UsagePricingEvidence or None"
            )
        if isinstance(self.error, StaleConversation):
            if (
                self.conversation is None
                or self.conversation.state != "stale"
            ):
                raise ValueError(
                    "StaleConversation requires a stale conversation outcome"
                )
        if self.completeness.token_usage_complete:
            for field_name in (
                "input_tokens",
                "output_tokens",
                "cache_creation_tokens",
                "cache_read_tokens",
            ):
                value = getattr(self.usage, field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"InvocationResult.usage.{field_name} must be an "
                        "exact non-negative integer when token usage is "
                        "complete"
                    )
            if (
                self.usage.cache_creation_tokens
                + self.usage.cache_read_tokens
                > self.usage.input_tokens
            ):
                raise ValueError(
                    "InvocationResult.usage cache-token subsets must not "
                    "exceed input_tokens when token usage is complete"
                )
        if self.completeness.cost_complete:
            value = self.usage.cost_usd
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "InvocationResult.usage.cost_usd must be a finite "
                    "non-negative number when cost is complete"
                )
        pricing_evidence = self.pricing_evidence
        if pricing_evidence is not None and pricing_evidence.complete:
            if not (
                self.completeness.token_usage_complete
                and self.completeness.cost_complete
            ):
                raise ValueError(
                    "complete InvocationResult.pricing_evidence requires "
                    "complete token usage and cost evidence"
                )
            if not pricing_evidence.reconciles(self.usage):
                raise ValueError(
                    "complete InvocationResult.pricing_evidence must "
                    "reconcile aggregate Usage"
                )
        if not self.completeness.trace_complete:
            return
        if self.execution_trace is None:
            raise ValueError(
                "InvocationResult.execution_trace is required when "
                "completeness.trace_complete is True"
            )
        if self.execution_trace.terminal_event_count != 1:
            raise ValueError(
                "InvocationResult.execution_trace must contain exactly one "
                "terminal event when completeness.trace_complete is True"
            )
        if self.execution_trace.events[-1].kind != "terminal":
            raise ValueError(
                "InvocationResult.execution_trace terminal event must be "
                "last when completeness.trace_complete is True"
            )


# Import after defining Usage/InvocationResult so conversation.py can consume
# Usage while this module is initializing. Keeping the concrete names in this
# module's globals makes public annotations resolvable with get_type_hints().
from .conversation import (  # noqa: E402
    ConversationTransportOutcome,
    ConversationTransportRequest,
)
