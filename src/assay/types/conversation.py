# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fingerprint-bound conversation continuity records and decisions.

The records in this module deliberately contain no prompt text, response
schema, route configuration, policy body, runtime configuration, trusted
context, or tool catalogue.  Those semantic inputs are represented only by
their SHA-256 digests.  A provider-issued ``opaque_id`` may be retained for
continuity, but it must be a non-authorizing identifier rather than a
credential.

Handle schema v3 adds durable, attributable recovery resolutions to the v2
crash-window ``pending_operation`` marker. Older records are deliberately not
auto-migrated: absence of either marker cannot prove that no provider
operation was in flight or that a prior recovery decision was auditable.
Durable stores must reject older versions and require explicit reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from typing import Literal

from .invocation import Usage


ConversationMode = Literal["none", "retry_only", "workflow_bound"]
ConversationState = Literal["active", "stale", "closed"]
ConversationResumeScope = Literal["transport_retry", "workflow_correction"]
ConversationResumeAction = Literal["fresh", "reuse", "reject"]
ConversationTransportAction = Literal["fresh", "reuse"]
PendingConversationOperationState = Literal["pending", "indeterminate"]
ConversationRecoveryDisposition = Literal["not_started"]

CONVERSATION_HANDLE_SCHEMA_VERSION = "assay.conversation-handle.v3"
_MODES = frozenset({"none", "retry_only", "workflow_bound"})
_STATES = frozenset({"active", "stale", "closed"})
_SCOPES = frozenset({"transport_retry", "workflow_correction"})
_TRANSPORT_ACTIONS = frozenset({"fresh", "reuse"})
_PENDING_STATES = frozenset({"pending", "indeterminate"})
_RECOVERY_DISPOSITIONS = frozenset({"not_started"})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _require_digest(field_name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 'sha256:' followed by 64 lowercase "
            "hexadecimal digits"
        )


def _require_opaque_id(
    field_name: str,
    value: str | None,
    *,
    required: bool = False,
) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} must be a non-empty string")
        return
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a non-empty, control-free UTF-8 string "
            "no larger than 4096 bytes"
        )


@dataclass(frozen=True)
class ConversationTransportRequest:
    """One identity-bound, non-authorizing provider-continuity request."""

    mode: ConversationMode
    action: ConversationTransportAction
    opaque_id: str | None
    identity_fingerprint: str
    operation_digest: str

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                "ConversationTransportRequest.mode must be one of "
                f"{sorted(_MODES)}"
            )
        if self.action not in _TRANSPORT_ACTIONS:
            raise ValueError(
                "ConversationTransportRequest.action must be fresh or reuse"
            )
        _require_digest(
            "ConversationTransportRequest.identity_fingerprint",
            self.identity_fingerprint,
        )
        _require_digest(
            "ConversationTransportRequest.operation_digest",
            self.operation_digest,
        )
        _require_opaque_id(
            "ConversationTransportRequest.opaque_id",
            self.opaque_id,
            required=self.action == "reuse",
        )
        if self.action == "fresh" and self.opaque_id is not None:
            raise ValueError(
                "ConversationTransportRequest.opaque_id must be None for "
                "a fresh action"
            )
        if self.mode == "none" and self.action != "fresh":
            raise ValueError(
                "ConversationTransportRequest.mode='none' requires a fresh "
                "action"
            )


@dataclass(frozen=True)
class ConversationTransportOutcome:
    """Provider continuity returned from one exact logical operation.

    ``opaque_id`` is persistence material only. It must never be a credential
    or authorize provider access on its own.
    """

    state: ConversationState
    opaque_id: str | None
    identity_fingerprint: str
    operation_digest: str

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise ValueError(
                "ConversationTransportOutcome.state must be one of "
                f"{sorted(_STATES)}"
            )
        _require_digest(
            "ConversationTransportOutcome.identity_fingerprint",
            self.identity_fingerprint,
        )
        _require_digest(
            "ConversationTransportOutcome.operation_digest",
            self.operation_digest,
        )
        _require_opaque_id(
            "ConversationTransportOutcome.opaque_id",
            self.opaque_id,
            required=self.state == "active",
        )
        if self.state == "stale" and self.opaque_id is not None:
            raise ValueError(
                "ConversationTransportOutcome.opaque_id must be None when "
                "state is stale"
            )


@dataclass(frozen=True)
class PendingConversationOperation:
    """Durable begin/commit marker for an uncertain provider operation."""

    operation_digest: str
    identity_fingerprint: str
    started_at: str
    state: PendingConversationOperationState = "pending"

    def __post_init__(self) -> None:
        _require_digest(
            "PendingConversationOperation.operation_digest",
            self.operation_digest,
        )
        _require_digest(
            "PendingConversationOperation.identity_fingerprint",
            self.identity_fingerprint,
        )
        if (
            not isinstance(self.started_at, str)
            or not self.started_at.strip()
            or len(self.started_at.encode("utf-8")) > 128
            or any(ord(character) < 0x20 for character in self.started_at)
        ):
            raise ValueError(
                "PendingConversationOperation.started_at must be a non-empty, "
                "control-free UTF-8 string no larger than 128 bytes"
            )
        if self.state not in _PENDING_STATES:
            raise ValueError(
                "PendingConversationOperation.state must be pending or "
                "indeterminate"
            )


@dataclass(frozen=True)
class ConversationRecoveryResolution:
    """Auditable proof used to clear one uncertain provider operation."""

    operation_digest: str
    identity_fingerprint: str
    evidence_digest: str
    resolved_at: str
    actor: str
    disposition: ConversationRecoveryDisposition = "not_started"

    def __post_init__(self) -> None:
        for field_name in (
            "operation_digest",
            "identity_fingerprint",
            "evidence_digest",
        ):
            _require_digest(
                f"ConversationRecoveryResolution.{field_name}",
                getattr(self, field_name),
            )
        for field_name, max_bytes in (("resolved_at", 128), ("actor", 1024)):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > max_bytes
                or any(ord(character) < 0x20 for character in value)
            ):
                raise ValueError(
                    f"ConversationRecoveryResolution.{field_name} must be a "
                    "non-empty, control-free UTF-8 string no larger than "
                    f"{max_bytes} bytes"
                )
        if self.disposition not in _RECOVERY_DISPOSITIONS:
            raise ValueError(
                "ConversationRecoveryResolution.disposition must be "
                "'not_started'"
            )


@dataclass(frozen=True)
class ConversationIdentity:
    """Every semantic input that must match before a handle can be reused."""

    role: str
    backend: str
    model: str
    prompt_abi: str
    conversation_mode: ConversationMode
    route_digest: str
    prompt_digest: str
    schema_digest: str
    tool_authority_digest: str
    feature_profile_digest: str
    policy_digest: str
    runtime_digest: str
    trusted_context_digest: str

    def __post_init__(self) -> None:
        for field_name in ("role", "backend", "model", "prompt_abi"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ConversationIdentity.{field_name} must be non-empty"
                )
        if self.conversation_mode not in _MODES:
            raise ValueError(
                "ConversationIdentity.conversation_mode must be one of "
                f"{sorted(_MODES)}"
            )
        for field_name in (
            "route_digest",
            "prompt_digest",
            "schema_digest",
            "tool_authority_digest",
            "feature_profile_digest",
            "policy_digest",
            "runtime_digest",
            "trusted_context_digest",
        ):
            _require_digest(
                f"ConversationIdentity.{field_name}",
                getattr(self, field_name),
            )

    @property
    def fingerprint(self) -> str:
        payload = {
            "backend": self.backend,
            "conversation_mode": self.conversation_mode,
            "feature_profile_digest": self.feature_profile_digest,
            "model": self.model,
            "policy_digest": self.policy_digest,
            "prompt_abi": self.prompt_abi,
            "prompt_digest": self.prompt_digest,
            "role": self.role,
            "route_digest": self.route_digest,
            "runtime_digest": self.runtime_digest,
            "schema_digest": self.schema_digest,
            "tool_authority_digest": self.tool_authority_digest,
            "trusted_context_digest": self.trusted_context_digest,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConversationHandle:
    """Opaque provider continuity plus cumulative logical-invocation ledger."""

    mode: ConversationMode
    identity_fingerprint: str
    opaque_id: str | None
    revision: int = 0
    turns_used: int = 0
    context_bytes: int = 0
    usage: Usage = Usage()
    state: ConversationState = "active"
    operation_digests: tuple[str, ...] = ()
    pending_operation: PendingConversationOperation | None = None
    recovery_resolutions: tuple[ConversationRecoveryResolution, ...] = ()
    schema_version: str = CONVERSATION_HANDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(
                f"ConversationHandle.mode must be one of {sorted(_MODES)}"
            )
        _require_digest(
            "ConversationHandle.identity_fingerprint",
            self.identity_fingerprint,
        )
        if self.mode == "none" and self.opaque_id is not None:
            raise ValueError(
                "ConversationHandle.opaque_id must be None in none mode"
            )
        _require_opaque_id(
            "ConversationHandle.opaque_id",
            self.opaque_id,
        )
        for field_name in ("revision", "turns_used", "context_bytes"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"ConversationHandle.{field_name} must be an exact "
                    "integer >= 0"
                )
        if type(self.usage) is not Usage:
            raise TypeError("ConversationHandle.usage must be an assay Usage")
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        ):
            value = getattr(self.usage, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"ConversationHandle.usage.{field_name} must be an exact "
                    "integer >= 0"
                )
        if (
            self.usage.cache_creation_tokens + self.usage.cache_read_tokens
            > self.usage.input_tokens
        ):
            raise ValueError(
                "ConversationHandle.usage cache-token subsets must not "
                "exceed input_tokens"
            )
        cost = self.usage.cost_usd
        if (
            type(cost) not in (int, float)
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError(
                "ConversationHandle.usage.cost_usd must be a finite "
                "non-negative number"
            )
        if self.state not in _STATES:
            raise ValueError(
                f"ConversationHandle.state must be one of {sorted(_STATES)}"
            )
        if not isinstance(self.operation_digests, tuple):
            raise TypeError(
                "ConversationHandle.operation_digests must be a tuple"
            )
        if len(self.operation_digests) != len(set(self.operation_digests)):
            raise ValueError(
                "ConversationHandle.operation_digests must not contain "
                "duplicates"
            )
        for digest in self.operation_digests:
            _require_digest(
                "ConversationHandle.operation_digests entry",
                digest,
            )
        if self.pending_operation is not None:
            if not isinstance(
                self.pending_operation,
                PendingConversationOperation,
            ):
                raise TypeError(
                    "ConversationHandle.pending_operation must be a "
                    "PendingConversationOperation or None"
                )
            if (
                self.pending_operation.identity_fingerprint
                != self.identity_fingerprint
            ):
                raise ValueError(
                    "ConversationHandle.pending_operation identity "
                    "fingerprint must match the handle"
                )
            if (
                self.pending_operation.operation_digest
                in self.operation_digests
            ):
                raise ValueError(
                    "ConversationHandle.pending_operation must not already "
                    "be committed"
                )
        if not isinstance(self.recovery_resolutions, tuple):
            raise TypeError(
                "ConversationHandle.recovery_resolutions must be a tuple"
            )
        resolved_operations: set[str] = set()
        for resolution in self.recovery_resolutions:
            if not isinstance(resolution, ConversationRecoveryResolution):
                raise TypeError(
                    "ConversationHandle recovery resolution must be a "
                    "ConversationRecoveryResolution"
                )
            if resolution.identity_fingerprint != self.identity_fingerprint:
                raise ValueError(
                    "ConversationHandle recovery resolution identity "
                    "fingerprint must match the handle"
                )
            if resolution.operation_digest in resolved_operations:
                raise ValueError(
                    "ConversationHandle recovery resolutions must contain "
                    "at most one record per operation"
                )
            resolved_operations.add(resolution.operation_digest)
        if self.schema_version != CONVERSATION_HANDLE_SCHEMA_VERSION:
            raise ValueError(
                "ConversationHandle.schema_version must be "
                f"{CONVERSATION_HANDLE_SCHEMA_VERSION!r}"
            )

    @property
    def last_operation_digest(self) -> str | None:
        """Return the newest applied operation for compatibility/debugging."""

        if not self.operation_digests:
            return None
        return self.operation_digests[-1]


@dataclass(frozen=True)
class ConversationResumeDecision:
    action: ConversationResumeAction
    reason: str
    handle: ConversationHandle

    def __post_init__(self) -> None:
        if self.action not in {"fresh", "reuse", "reject"}:
            raise ValueError(
                "ConversationResumeDecision.action must be fresh, reuse, "
                "or reject"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "ConversationResumeDecision.reason must be non-empty"
            )


def new_conversation_handle(
    identity: ConversationIdentity,
    *,
    mode: ConversationMode,
    opaque_id: str | None,
) -> ConversationHandle:
    """Create a zero-ledger handle bound to one exact semantic identity."""

    if identity.conversation_mode != mode:
        raise ValueError(
            "conversation identity mode does not match the requested handle mode"
        )
    return ConversationHandle(
        mode=mode,
        identity_fingerprint=identity.fingerprint,
        opaque_id=opaque_id,
    )


def decide_conversation_resume(
    handle: ConversationHandle,
    identity: ConversationIdentity,
    *,
    expected_mode: ConversationMode,
    scope: ConversationResumeScope,
    provider_reports_stale: bool = False,
    allow_stale_fallback: bool = False,
) -> ConversationResumeDecision:
    """Return a fail-closed reuse/fresh/reject decision.

    A stale-provider fallback changes only the opaque transport continuity.
    The returned fresh decision retains the same handle and cumulative ledger;
    callers must replace the provider ID without resetting turns, bytes,
    tokens, cost, or semantic identity.
    """

    if scope not in _SCOPES:
        raise ValueError(
            f"unknown conversation resume scope {scope!r}; expected one of "
            f"{sorted(_SCOPES)}"
        )
    if expected_mode not in _MODES:
        raise ValueError(
            "expected conversation mode must be one of "
            f"{sorted(_MODES)}"
        )
    if identity.conversation_mode != expected_mode:
        return ConversationResumeDecision(
            "reject",
            "conversation identity mode does not match the expected mode",
            handle,
        )
    if handle.identity_fingerprint != identity.fingerprint:
        return ConversationResumeDecision(
            "reject",
            "conversation identity fingerprint mismatch",
            handle,
        )
    if handle.mode != expected_mode:
        return ConversationResumeDecision(
            "reject",
            "conversation mode does not match the expected continuity mode",
            handle,
        )
    if handle.pending_operation is not None:
        return ConversationResumeDecision(
            "reject",
            "conversation has an unresolved pending provider operation",
            handle,
        )
    if handle.state == "closed":
        return ConversationResumeDecision(
            "reject",
            "conversation handle is closed",
            handle,
        )
    stale = provider_reports_stale or handle.state == "stale"
    if handle.mode == "none":
        return ConversationResumeDecision(
            "fresh",
            "route declares no conversation continuity",
            handle,
        )
    if (
        handle.mode == "retry_only"
        and scope != "transport_retry"
    ):
        return ConversationResumeDecision(
            "reject",
            "retry_only handle cannot cross a workflow correction boundary",
            handle,
        )
    if stale:
        if not allow_stale_fallback:
            return ConversationResumeDecision(
                "reject",
                "provider conversation is stale and fallback is disabled",
                handle,
            )
        return ConversationResumeDecision(
            "fresh",
            "provider conversation is stale; start fresh while retaining the "
            "logical cumulative ledger",
            (
                handle
                if handle.opaque_id is None and handle.state == "stale"
                else replace(
                    handle,
                    opaque_id=None,
                    state="stale",
                    revision=handle.revision + 1,
                )
            ),
        )
    if handle.opaque_id is None:
        return ConversationResumeDecision(
            "fresh",
            "no provider conversation is assigned",
            handle,
        )
    return ConversationResumeDecision(
        "reuse",
        "conversation identity and continuity scope match",
        handle,
    )


def advance_conversation_handle(
    handle: ConversationHandle,
    identity: ConversationIdentity,
    *,
    operation_digest: str,
    opaque_id: str | None = None,
    turns: int = 0,
    context_bytes: int = 0,
    usage: Usage = Usage(),
) -> ConversationHandle:
    """Advance cumulative counters once per idempotent operation digest."""

    if handle.identity_fingerprint != identity.fingerprint:
        raise ValueError("conversation identity fingerprint mismatch")
    if handle.mode != identity.conversation_mode:
        raise ValueError("conversation handle mode does not match identity")
    if handle.state == "closed":
        raise ValueError("cannot advance a closed conversation handle")
    _require_digest("operation_digest", operation_digest)
    if operation_digest in handle.operation_digests:
        return handle
    for field_name, value in (
        ("turns", turns),
        ("context_bytes", context_bytes),
        ("input_tokens", usage.input_tokens),
        ("output_tokens", usage.output_tokens),
        ("cache_creation_tokens", usage.cache_creation_tokens),
        ("cache_read_tokens", usage.cache_read_tokens),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{field_name} delta must be an exact integer >= 0")
    if (
        usage.cache_creation_tokens + usage.cache_read_tokens
        > usage.input_tokens
    ):
        raise ValueError(
            "cache-token delta subsets must not exceed input_tokens"
        )
    if (
        type(usage.cost_usd) not in (int, float)
        or not math.isfinite(usage.cost_usd)
        or usage.cost_usd < 0
    ):
        raise ValueError("cost delta must be finite and non-negative")
    next_opaque_id = handle.opaque_id if opaque_id is None else opaque_id
    return replace(
        handle,
        opaque_id=next_opaque_id,
        revision=handle.revision + 1,
        turns_used=handle.turns_used + turns,
        context_bytes=handle.context_bytes + context_bytes,
        usage=Usage(
            input_tokens=handle.usage.input_tokens + usage.input_tokens,
            output_tokens=handle.usage.output_tokens + usage.output_tokens,
            cost_usd=handle.usage.cost_usd + usage.cost_usd,
            cache_creation_tokens=(
                handle.usage.cache_creation_tokens
                + usage.cache_creation_tokens
            ),
            cache_read_tokens=(
                handle.usage.cache_read_tokens + usage.cache_read_tokens
            ),
        ),
        state="active",
        operation_digests=(*handle.operation_digests, operation_digest),
    )


def mark_conversation_state(
    handle: ConversationHandle,
    *,
    state: Literal["stale", "closed"],
) -> ConversationHandle:
    """Persist an explicit provider-stale or terminally closed state."""

    if handle.state == state:
        return handle
    return replace(
        handle,
        state=state,
        opaque_id=None if state == "stale" else handle.opaque_id,
        revision=handle.revision + 1,
    )


__all__ = [
    "CONVERSATION_HANDLE_SCHEMA_VERSION",
    "ConversationHandle",
    "ConversationIdentity",
    "ConversationMode",
    "ConversationRecoveryDisposition",
    "ConversationRecoveryResolution",
    "ConversationTransportAction",
    "ConversationTransportOutcome",
    "ConversationTransportRequest",
    "ConversationResumeAction",
    "ConversationResumeDecision",
    "ConversationResumeScope",
    "ConversationState",
    "PendingConversationOperation",
    "PendingConversationOperationState",
    "advance_conversation_handle",
    "decide_conversation_resume",
    "mark_conversation_state",
    "new_conversation_handle",
]
