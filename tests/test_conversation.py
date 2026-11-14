# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from assay import (
    ConversationHandle,
    ConversationIdentity,
    ConversationRecoveryResolution,
    ConversationTransportOutcome,
    ConversationTransportRequest,
    PendingConversationOperation,
    Usage,
    advance_conversation_handle,
    decide_conversation_resume,
    mark_conversation_state,
    new_conversation_handle,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _identity(**overrides: str) -> ConversationIdentity:
    values = {
        "role": "implementer",
        "backend": "codex",
        "model": "gpt-5.6-sol",
        "prompt_abi": "assay.prompt.v2",
        "conversation_mode": "workflow_bound",
        "route_digest": _digest("1"),
        "prompt_digest": _digest("2"),
        "schema_digest": _digest("3"),
        "tool_authority_digest": _digest("4"),
        "feature_profile_digest": _digest("5"),
        "policy_digest": _digest("6"),
        "runtime_digest": _digest("7"),
        "trusted_context_digest": _digest("8"),
    }
    values.update(overrides)
    return ConversationIdentity(**values)


def test_identity_fingerprint_binds_every_semantic_input() -> None:
    original = _identity()
    for field_name in (
        "role",
        "backend",
        "model",
        "prompt_abi",
        "conversation_mode",
        "route_digest",
        "prompt_digest",
        "schema_digest",
        "tool_authority_digest",
        "feature_profile_digest",
        "policy_digest",
        "runtime_digest",
        "trusted_context_digest",
    ):
        if field_name == "conversation_mode":
            replacement = "retry_only"
        elif field_name in {"role", "backend", "model", "prompt_abi"}:
            replacement = f"{getattr(original, field_name)}-changed"
        else:
            replacement = _digest("9")
        assert replace(original, **{field_name: replacement}).fingerprint != (
            original.fingerprint
        )


def test_resume_modes_are_fail_closed() -> None:
    none_identity = _identity(conversation_mode="none")
    none_handle = new_conversation_handle(
        none_identity,
        mode="none",
        opaque_id=None,
    )
    assert (
        decide_conversation_resume(
            none_handle,
            none_identity,
            expected_mode="none",
            scope="workflow_correction",
        ).action
        == "fresh"
    )

    retry_identity = _identity(conversation_mode="retry_only")
    retry_handle = new_conversation_handle(
        retry_identity,
        mode="retry_only",
        opaque_id="thread-1",
    )
    assert (
        decide_conversation_resume(
            retry_handle,
            retry_identity,
            expected_mode="retry_only",
            scope="transport_retry",
        ).action
        == "reuse"
    )
    assert (
        decide_conversation_resume(
            retry_handle,
            retry_identity,
            expected_mode="retry_only",
            scope="workflow_correction",
        ).action
        == "reject"
    )

    workflow_identity = _identity()
    workflow_handle = new_conversation_handle(
        workflow_identity,
        mode="workflow_bound",
        opaque_id="thread-2",
    )
    assert (
        decide_conversation_resume(
            workflow_handle,
            workflow_identity,
            expected_mode="workflow_bound",
            scope="workflow_correction",
        ).action
        == "reuse"
    )


def test_cross_fingerprint_resume_is_rejected() -> None:
    identity = _identity()
    handle = new_conversation_handle(
        identity,
        mode="workflow_bound",
        opaque_id="thread-1",
    )
    decision = decide_conversation_resume(
        handle,
        _identity(tool_authority_digest=_digest("9")),
        expected_mode="workflow_bound",
        scope="workflow_correction",
    )
    assert decision.action == "reject"
    assert "fingerprint mismatch" in decision.reason


def test_resume_rejects_mode_tampering_and_unresolved_operation() -> None:
    retry_identity = _identity(conversation_mode="retry_only")
    retry_handle = new_conversation_handle(
        retry_identity,
        mode="retry_only",
        opaque_id="thread-1",
    )
    workflow_identity = replace(
        retry_identity,
        conversation_mode="workflow_bound",
    )
    tampered = replace(retry_handle, mode="workflow_bound")

    decision = decide_conversation_resume(
        tampered,
        workflow_identity,
        expected_mode="workflow_bound",
        scope="workflow_correction",
    )
    assert decision.action == "reject"
    assert retry_identity.fingerprint != workflow_identity.fingerprint

    pending = PendingConversationOperation(
        operation_digest=_digest("a"),
        identity_fingerprint=workflow_identity.fingerprint,
        started_at="2026-07-28T17:00:00+00:00",
    )
    for state in ("pending", "indeterminate"):
        handle = ConversationHandle(
            mode="workflow_bound",
            identity_fingerprint=workflow_identity.fingerprint,
            opaque_id="thread-2",
            pending_operation=replace(pending, state=state),
        )
        blocked = decide_conversation_resume(
            handle,
            workflow_identity,
            expected_mode="workflow_bound",
            scope="workflow_correction",
        )
        assert blocked.action == "reject"
        assert "unresolved pending" in blocked.reason


def test_stale_fallback_clears_provider_id_without_resetting_ledger() -> None:
    identity = _identity()
    handle = new_conversation_handle(
        identity,
        mode="workflow_bound",
        opaque_id="thread-old",
    )
    handle = advance_conversation_handle(
        handle,
        identity,
        operation_digest=_digest("a"),
        turns=3,
        context_bytes=200,
        usage=Usage(input_tokens=20, output_tokens=10, cost_usd=1.25),
    )

    decision = decide_conversation_resume(
        handle,
        identity,
        expected_mode="workflow_bound",
        scope="workflow_correction",
        provider_reports_stale=True,
        allow_stale_fallback=True,
    )

    assert decision.action == "fresh"
    assert decision.handle.opaque_id is None
    assert decision.handle.state == "stale"
    assert decision.handle.turns_used == 3
    assert decision.handle.context_bytes == 200
    assert decision.handle.usage == Usage(
        input_tokens=20,
        output_tokens=10,
        cost_usd=1.25,
    )


def test_operation_idempotency_covers_non_adjacent_replays() -> None:
    identity = _identity()
    handle = new_conversation_handle(
        identity,
        mode="workflow_bound",
        opaque_id=None,
    )
    first = advance_conversation_handle(
        handle,
        identity,
        operation_digest=_digest("a"),
        opaque_id="thread-new",
        turns=1,
        context_bytes=10,
        usage=Usage(input_tokens=2, output_tokens=3, cost_usd=0.5),
    )
    second = advance_conversation_handle(
        first,
        identity,
        operation_digest=_digest("b"),
        turns=2,
        context_bytes=20,
        usage=Usage(input_tokens=4, output_tokens=5, cost_usd=1.0),
    )
    replay = advance_conversation_handle(
        second,
        identity,
        operation_digest=_digest("a"),
        turns=100,
        context_bytes=100,
        usage=Usage(input_tokens=100, output_tokens=100, cost_usd=100),
    )

    assert replay is second
    assert replay.turns_used == 3
    assert replay.operation_digests == (_digest("a"), _digest("b"))


def test_conversation_ledger_aggregates_cache_token_subsets() -> None:
    identity = _identity()
    handle = new_conversation_handle(
        identity,
        mode="workflow_bound",
        opaque_id="thread-1",
    )

    advanced = advance_conversation_handle(
        handle,
        identity,
        operation_digest=_digest("c"),
        usage=Usage(
            input_tokens=20,
            output_tokens=4,
            cost_usd=0.1,
            cache_creation_tokens=5,
            cache_read_tokens=10,
        ),
    )

    assert advanced.usage == Usage(
        input_tokens=20,
        output_tokens=4,
        cost_usd=0.1,
        cache_creation_tokens=5,
        cache_read_tokens=10,
    )


def test_closed_and_invalid_opaque_handles_fail_closed() -> None:
    identity = _identity(conversation_mode="retry_only")
    handle = new_conversation_handle(
        identity,
        mode="retry_only",
        opaque_id="thread-1",
    )
    closed = mark_conversation_state(handle, state="closed")
    assert (
        decide_conversation_resume(
            closed,
            identity,
            expected_mode="retry_only",
            scope="transport_retry",
        ).action
        == "reject"
    )
    with pytest.raises(ValueError, match="control-free"):
        ConversationHandle(
            mode="retry_only",
            identity_fingerprint=identity.fingerprint,
            opaque_id="bad\nhandle",
        )
    with pytest.raises(TypeError, match="usage"):
        ConversationHandle(
            mode="retry_only",
            identity_fingerprint=identity.fingerprint,
            opaque_id="thread-1",
            usage=SimpleNamespace(  # type: ignore[arg-type]
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            ),
        )


def test_transport_request_binds_identity_operation_and_action() -> None:
    fresh = ConversationTransportRequest(
        mode="workflow_bound",
        action="fresh",
        opaque_id=None,
        identity_fingerprint=_digest("1"),
        operation_digest=_digest("2"),
    )
    reuse = replace(fresh, action="reuse", opaque_id="provider-thread")

    assert fresh.opaque_id is None
    assert reuse.opaque_id == "provider-thread"
    with pytest.raises(ValueError, match="non-empty"):
        replace(reuse, opaque_id=None)
    with pytest.raises(ValueError, match="must be None"):
        replace(fresh, opaque_id="unexpected")
    with pytest.raises(ValueError, match="control-free"):
        replace(reuse, opaque_id="bad\nthread")


def test_transport_outcome_requires_active_id_and_clears_stale_id() -> None:
    active = ConversationTransportOutcome(
        state="active",
        opaque_id="provider-thread",
        identity_fingerprint=_digest("1"),
        operation_digest=_digest("2"),
    )

    assert active.opaque_id == "provider-thread"
    with pytest.raises(ValueError, match="non-empty"):
        replace(active, opaque_id=None)
    with pytest.raises(ValueError, match="must be None"):
        replace(active, state="stale")

    stale = replace(active, state="stale", opaque_id=None)
    assert stale.opaque_id is None


def test_pending_operation_is_identity_bound_and_not_already_committed() -> None:
    pending = PendingConversationOperation(
        operation_digest=_digest("a"),
        identity_fingerprint=_digest("b"),
        started_at="2026-07-28T17:00:00+00:00",
    )
    handle = ConversationHandle(
        mode="workflow_bound",
        identity_fingerprint=_digest("b"),
        opaque_id="provider-thread",
        pending_operation=pending,
    )

    assert handle.pending_operation is pending
    with pytest.raises(ValueError, match="must match"):
        replace(
            handle,
            pending_operation=replace(
                pending,
                identity_fingerprint=_digest("c"),
            ),
        )
    with pytest.raises(ValueError, match="must not already be committed"):
        replace(handle, operation_digests=(_digest("a"),))
    assert replace(pending, state="indeterminate").state == "indeterminate"


def test_recovery_resolution_is_typed_identity_bound_and_unique() -> None:
    resolution = ConversationRecoveryResolution(
        operation_digest=_digest("a"),
        identity_fingerprint=_digest("b"),
        evidence_digest=_digest("c"),
        resolved_at="2026-07-28T17:01:00+00:00",
        actor="operator:alice",
    )
    handle = ConversationHandle(
        mode="workflow_bound",
        identity_fingerprint=_digest("b"),
        opaque_id=None,
        recovery_resolutions=(resolution,),
    )

    assert handle.recovery_resolutions == (resolution,)
    assert resolution.disposition == "not_started"
    with pytest.raises(ValueError, match="must match"):
        replace(
            handle,
            recovery_resolutions=(
                replace(
                    resolution,
                    identity_fingerprint=_digest("d"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="at most one"):
        replace(
            handle,
            recovery_resolutions=(resolution, resolution),
        )
    with pytest.raises(ValueError, match="actor"):
        replace(resolution, actor="operator\nforged")
