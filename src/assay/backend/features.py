# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Closed, vendor-neutral backend feature declarations.

The minimal :class:`~assay.backend.protocol.Backend` protocol intentionally
stays small.  Consumers that need to make policy decisions before transport
can additionally narrow a backend to :class:`FeatureAwareBackend`.

Feature values describe observed enforcement, not provider marketing.  The
defaults are deliberately conservative so an unknown adapter never acquires a
capability merely by omitting a declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Literal, Protocol, runtime_checkable

from .protocol import Backend


StructuredOutput = Literal["native", "prompt_only", "unsupported"]
ToolLoop = Literal["sdk_owned", "adapter_owned", "none"]
SessionContinuity = Literal["none", "retry_only", "workflow_bound"]
EventDetail = Literal["terminal_only", "usage", "tool_calls", "agent_graph"]
LimitEnforcement = Literal[
    "hard", "provider_stop", "round_boundary", "post_hoc", "unsupported"
]
DeadlineEnforcement = Literal[
    "hard_invocation", "request_only", "post_hoc", "unsupported"
]
RetryScope = Literal["invocation", "per_completion", "unsupported"]
CacheControl = Literal["adapter_owned", "provider_owned", "unsupported"]
Cancellation = Literal[
    "interruptible", "boundary_only", "process_kill", "unsupported"
]
ProjectInstructions = Literal[
    "not_applicable", "suppressible", "unsuppressible"
]
NativeDelegation = Literal["none", "read_only", "writer"]


BACKEND_FEATURE_SCHEMA_VERSION = "assay.backend-features.v1"
BACKEND_CONFORMANCE_VERSION = "assay.backend-conformance.v2"


@dataclass(frozen=True)
class BackendFeatureSet:
    """Observed feature and enforcement profile for one concrete route."""

    structured_output: StructuredOutput = "unsupported"
    tool_loop: ToolLoop = "none"
    session_continuity: SessionContinuity = "none"
    event_detail: EventDetail = "terminal_only"
    turn_limit: LimitEnforcement = "unsupported"
    response_token_limit: LimitEnforcement = "unsupported"
    context_byte_limit: LimitEnforcement = "unsupported"
    usd_limit: LimitEnforcement = "unsupported"
    deadline: DeadlineEnforcement = "unsupported"
    retry_scope: RetryScope = "unsupported"
    cache_control: CacheControl = "unsupported"
    cancellation: Cancellation = "unsupported"
    project_instructions: ProjectInstructions = "unsuppressible"
    native_delegation: NativeDelegation = "none"
    schema_version: str = BACKEND_FEATURE_SCHEMA_VERSION

    _VALUES: ClassVar[dict[str, frozenset[str]]] = {
        "structured_output": frozenset(
            {"native", "prompt_only", "unsupported"}
        ),
        "tool_loop": frozenset({"sdk_owned", "adapter_owned", "none"}),
        "session_continuity": frozenset(
            {"none", "retry_only", "workflow_bound"}
        ),
        "event_detail": frozenset(
            {"terminal_only", "usage", "tool_calls", "agent_graph"}
        ),
        "turn_limit": frozenset(
            {
                "hard",
                "provider_stop",
                "round_boundary",
                "post_hoc",
                "unsupported",
            }
        ),
        "response_token_limit": frozenset(
            {
                "hard",
                "provider_stop",
                "round_boundary",
                "post_hoc",
                "unsupported",
            }
        ),
        "context_byte_limit": frozenset(
            {
                "hard",
                "provider_stop",
                "round_boundary",
                "post_hoc",
                "unsupported",
            }
        ),
        "usd_limit": frozenset(
            {
                "hard",
                "provider_stop",
                "round_boundary",
                "post_hoc",
                "unsupported",
            }
        ),
        "deadline": frozenset(
            {"hard_invocation", "request_only", "post_hoc", "unsupported"}
        ),
        "retry_scope": frozenset(
            {"invocation", "per_completion", "unsupported"}
        ),
        "cache_control": frozenset(
            {"adapter_owned", "provider_owned", "unsupported"}
        ),
        "cancellation": frozenset(
            {"interruptible", "boundary_only", "process_kill", "unsupported"}
        ),
        "project_instructions": frozenset(
            {"not_applicable", "suppressible", "unsuppressible"}
        ),
        "native_delegation": frozenset({"none", "read_only", "writer"}),
    }

    @classmethod
    def allowed_values(cls, field_name: str) -> frozenset[str]:
        """Return the closed vocabulary for one declared feature field."""

        try:
            return cls._VALUES[field_name]
        except KeyError as exc:
            raise ValueError(
                f"unknown BackendFeatureSet field {field_name!r}"
            ) from exc

    def __post_init__(self) -> None:
        if self.schema_version != BACKEND_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "BackendFeatureSet.schema_version must be "
                f"{BACKEND_FEATURE_SCHEMA_VERSION!r}"
            )
        for item in fields(self):
            if item.name == "schema_version":
                continue
            value = getattr(self, item.name)
            allowed = self._VALUES[item.name]
            if value not in allowed:
                raise ValueError(
                    f"BackendFeatureSet.{item.name} is {value!r}; "
                    f"expected one of {sorted(allowed)}"
                )


CONSERVATIVE_BACKEND_FEATURES = BackendFeatureSet()


@runtime_checkable
class FeatureAwareBackend(Backend, Protocol):
    """Optional protocol for adapters with an observed feature profile."""

    @property
    def features(self) -> BackendFeatureSet: ...


__all__ = [
    "BACKEND_CONFORMANCE_VERSION",
    "BACKEND_FEATURE_SCHEMA_VERSION",
    "BackendFeatureSet",
    "CONSERVATIVE_BACKEND_FEATURES",
    "FeatureAwareBackend",
]
