# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Backend protocol and BackendError discriminated union.

The Backend protocol is runtime-checkable; any class with the right
shape (``name: str`` + ``model: str`` +
``invoke(invocation) -> InvocationResult``)
satisfies it.

The concrete BackendError subtypes each carry the expected
``kind`` discriminator and satisfy the BackendError Protocol via
runtime check.

AuthError is deliberately omitted — importing it must fail.
"""

from __future__ import annotations

import dataclasses
import typing

import pytest

from assay import (
    BACKEND_CONFORMANCE_VERSION,
    BACKEND_EXECUTION_TRACE_VERSION,
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendFeatureSet,
    BackendPolicy,
    Backend,
    BackendError,
    BudgetExhausted,
    ContentRejected,
    ConversationTransportOutcome,
    ConversationTransportRequest,
    ExecutionCompleteness,
    ExternalCancellation,
    FeatureAwareBackend,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    StaleConversation,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
    admit_result_evidence,
    missing_execution_evidence,
)
from assay.backend.protocol import ChatToolsBackend
from assay.canonical import parse, serialize
from assay.corpus.codec import decode_dataclass
from assay.primitives.tool_cache import ToolCallCache
from assay.types.tools import ToolSet


def test_invocation_is_frozen():
    inv = Invocation(
        system="sys", user="u", tools=None, max_turns=5, budget_usd=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.max_turns = 10  # type: ignore[misc]


@pytest.mark.parametrize("max_turns", [0, -1, True, 1.5])
def test_invocation_rejects_invalid_turn_limits(max_turns):
    with pytest.raises(ValueError, match="max_turns"):
        Invocation(
            system="sys",
            user="u",
            tools=None,
            max_turns=max_turns,
            budget_usd=None,
        )


@pytest.mark.parametrize(
    "budget_usd",
    [-1, True, "1", float("nan"), float("inf")],
)
def test_invocation_rejects_invalid_budget_limits(budget_usd):
    with pytest.raises(ValueError, match="budget_usd"):
        Invocation(
            system="sys",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=budget_usd,
        )


def test_invocation_accepts_zero_budget_and_rejects_non_mapping_schema():
    invocation = Invocation(
        system="sys",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=0.0,
    )
    assert invocation.budget_usd == 0.0

    with pytest.raises(TypeError, match="response_schema"):
        Invocation(
            system="sys",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
            response_schema=[],
        )


def test_invocation_accepts_only_typed_conversation_request():
    request = ConversationTransportRequest(
        mode="workflow_bound",
        action="reuse",
        opaque_id="provider-thread",
        identity_fingerprint="sha256:" + "a" * 64,
        operation_digest="sha256:" + "b" * 64,
    )

    invocation = Invocation(
        system="sys",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        conversation=request,
    )

    assert invocation.conversation is request
    with pytest.raises(TypeError, match="ConversationTransportRequest"):
        Invocation(
            system="sys",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
            conversation="provider-thread",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("retry_budget", 0),
        ("retry_budget", True),
        ("retry_budget", 1.5),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", True),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("on_cancellation", "ignore"),
        ("on_max_turns", "ignore"),
    ],
)
def test_backend_policy_rejects_invalid_controls(field_name, value):
    values = {
        "retry_budget": 1,
        "timeout_seconds": None,
        "on_cancellation": "raise",
        "on_max_turns": "raise",
    }
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        BackendPolicy(**values)


def test_invocation_result_is_frozen():
    r = InvocationResult(
        final_message="ok", transcript=(), usage=Usage(), stop_reason="end_turn"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.stop_reason = "other"  # type: ignore[misc]


def test_invocation_result_defaults_to_incomplete_execution_evidence():
    r = InvocationResult(
        final_message="ok", transcript=(), usage=Usage(), stop_reason="end_turn"
    )
    assert r.completeness == ExecutionCompleteness()
    assert r.completeness.complete is False
    assert r.execution_trace is None
    assert r.pricing_evidence is None


def test_usage_cache_fields_preserve_legacy_canonical_shape_and_codec():
    usage = Usage(input_tokens=12, output_tokens=3, cost_usd=0.25)
    payload = parse(serialize(usage))

    assert payload == {
        "cost_usd": 0.25,
        "input_tokens": 12,
        "output_tokens": 3,
    }
    assert decode_dataclass(payload, Usage) == usage

    cached = dataclasses.replace(
        usage,
        cache_creation_tokens=4,
        cache_read_tokens=5,
    )
    cached_payload = parse(serialize(cached))
    assert cached_payload["cache_creation_tokens"] == 4
    assert cached_payload["cache_read_tokens"] == 5
    assert decode_dataclass(cached_payload, Usage) == cached


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"input_tokens": True}, "exact non-negative integer"),
        ({"output_tokens": -1}, "exact non-negative integer"),
        ({"cache_creation_tokens": 6}, "must not exceed"),
        ({"cache_read_tokens": 6}, "must not exceed"),
        ({"cost_usd": float("inf")}, "finite non-negative"),
    ],
)
def test_usage_pricing_segment_is_strict(overrides, error):
    values = {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_creation_tokens": 5,
        "cache_read_tokens": 5,
        "cost_usd": 0.2,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=error):
        UsagePricingSegment(**values)


def test_complete_pricing_evidence_round_trips_and_reconciles_usage():
    evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(
                input_tokens=100,
                output_tokens=10,
                cache_creation_tokens=20,
                cache_read_tokens=30,
                cost_usd=0.4,
            ),
            UsagePricingSegment(
                input_tokens=300,
                output_tokens=20,
                cache_creation_tokens=0,
                cache_read_tokens=250,
                cost_usd=0.6,
            ),
        ),
        complete=True,
    )
    result = InvocationResult(
        final_message="ok",
        transcript=(),
        usage=Usage(
            input_tokens=400,
            output_tokens=30,
            cost_usd=1.0,
            cache_creation_tokens=20,
            cache_read_tokens=280,
        ),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
        ),
        pricing_evidence=evidence,
    )

    assert evidence.usage == result.usage
    payload = parse(serialize(result))
    assert payload["pricing_evidence"]["complete"] is True
    assert len(payload["pricing_evidence"]["segments"]) == 2
    assert decode_dataclass(payload, InvocationResult) == result


def test_complete_pricing_evidence_requires_complete_aggregate_evidence():
    evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(10, 2, 1, 3, 0.2),
        ),
        complete=True,
    )

    with pytest.raises(ValueError, match="complete token usage and cost"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=evidence.usage,
            stop_reason="end_turn",
            pricing_evidence=evidence,
        )


def test_complete_pricing_evidence_rejects_aggregate_mismatch():
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(10, 2, 1, 3, 0.2),),
        complete=True,
    )

    with pytest.raises(ValueError, match="reconcile aggregate Usage"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=dataclasses.replace(evidence.usage, input_tokens=11),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
            ),
            pricing_evidence=evidence,
        )


def test_incomplete_pricing_evidence_can_retain_partial_segments():
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(10, 2, 1, 3, 0.2),),
        complete=False,
    )

    result = InvocationResult(
        final_message="partial",
        transcript=(),
        usage=Usage(),
        stop_reason="error",
        pricing_evidence=evidence,
    )

    assert result.pricing_evidence is evidence


def test_stale_error_requires_matching_typed_conversation_outcome():
    stale = ConversationTransportOutcome(
        state="stale",
        opaque_id=None,
        identity_fingerprint="sha256:" + "a" * 64,
        operation_digest="sha256:" + "b" * 64,
    )
    result = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason="stale_conversation",
        error=StaleConversation(message="provider rejected conversation"),
        conversation=stale,
    )

    assert result.error is not None
    assert result.error.kind == "stale_conversation"
    assert result.error.retriable is False
    assert result.conversation is stale
    with pytest.raises(ValueError, match="requires a stale"):
        InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(),
            stop_reason="stale_conversation",
            error=StaleConversation(),
        )


def test_result_evidence_admission_rejects_incomplete_success():
    result = InvocationResult(
        final_message="nominal",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )

    admitted = admit_result_evidence(result, mode="enforce")

    assert isinstance(admitted.error, IncompleteEvidence)
    assert admitted.error.missing == (
        "token_usage_complete",
        "cost_complete",
        "terminal_complete",
        "trace_complete",
    )
    assert missing_execution_evidence(admitted) == admitted.error.missing
    assert admitted.final_message == "nominal"


def test_result_evidence_admission_explicit_diagnostic_preserves_success():
    result = InvocationResult(
        final_message="diagnostic",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )

    assert admit_result_evidence(result, mode="diagnostic") is result


def test_result_evidence_admission_preserves_existing_backend_failure():
    result = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason="error",
        error=TransportError(message="provider unavailable"),
    )

    assert admit_result_evidence(result, mode="enforce") is result


def test_result_evidence_admission_accepts_complete_success():
    trace = BackendExecutionTrace(
        backend="fake",
        requested_model="fake-model",
        events=(
            BackendExecutionEvent(
                sequence=0,
                kind="terminal",
                name="end_turn",
            ),
        ),
    )
    result = InvocationResult(
        final_message="complete",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=trace,
    )

    assert admit_result_evidence(result, mode="enforce") is result


def test_semantic_admission_preserves_terminal_success_with_accounting_debt():
    trace = BackendExecutionTrace(
        backend="fake",
        requested_model="fake-model",
        events=(
            BackendExecutionEvent(
                sequence=0,
                kind="terminal",
                name="end_turn",
            ),
        ),
    )
    result = InvocationResult(
        final_message="valid resumed response",
        transcript=(),
        usage=Usage(cost_usd=0.4),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=trace,
    )

    assert admit_result_evidence(result, mode="semantic") is result
    assert missing_execution_evidence(result) == (
        "token_usage_complete",
        "cost_complete",
    )
    assert isinstance(
        admit_result_evidence(result, mode="enforce").error,
        IncompleteEvidence,
    )


@pytest.mark.parametrize("missing_field", ["terminal_complete", "trace_complete"])
def test_semantic_admission_rejects_missing_control_evidence(missing_field):
    completeness = {
        "token_usage_complete": True,
        "cost_complete": True,
        "terminal_complete": True,
        "trace_complete": True,
    }
    completeness[missing_field] = False
    result = InvocationResult(
        final_message="untrusted terminal",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(**completeness),
        execution_trace=(
            BackendExecutionTrace(
                backend="fake",
                requested_model="fake-model",
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            )
            if completeness["trace_complete"]
            else None
        ),
    )

    admitted = admit_result_evidence(result, mode="semantic")

    assert isinstance(admitted.error, IncompleteEvidence)
    assert missing_field in admitted.error.missing


def test_result_evidence_admission_rejects_unknown_mode():
    result = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )
    with pytest.raises(ValueError, match="unknown result evidence"):
        admit_result_evidence(result, mode="shadow")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    [
        "token_usage_complete",
        "cost_complete",
        "terminal_complete",
        "trace_complete",
    ],
)
def test_execution_completeness_requires_exact_bools(field_name):
    with pytest.raises(TypeError, match=rf"{field_name} must be bool"):
        ExecutionCompleteness(**{field_name: 1})


def test_trace_complete_requires_execution_trace():
    with pytest.raises(ValueError, match="execution_trace is required"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(trace_complete=True),
        )


@pytest.mark.parametrize("terminal_event_count", [0, 2])
def test_trace_complete_requires_exactly_one_terminal_event(
    terminal_event_count,
):
    events = tuple(
        BackendExecutionEvent(sequence=index, kind="terminal")
        for index in range(terminal_event_count)
    )
    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="requested",
        events=events,
    )

    with pytest.raises(ValueError, match="exactly one terminal event"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(trace_complete=True),
            execution_trace=trace,
        )


def test_trace_complete_accepts_exactly_one_terminal_event():
    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="requested",
        events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
    )

    result = InvocationResult(
        final_message="ok",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(trace_complete=True),
        execution_trace=trace,
    )

    assert result.execution_trace is trace


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (-1, 0),
        (True, 0),
        (1.0, 0),
        (0, -1),
        (0, False),
        (0, 1.0),
    ],
)
def test_complete_token_usage_requires_exact_nonnegative_integers(
    input_tokens,
    output_tokens,
):
    with pytest.raises(ValueError, match="exact non-negative integer"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
            ),
        )


@pytest.mark.parametrize(
    "cost_usd",
    [-1.0, float("nan"), float("inf"), True, "0"],
)
def test_complete_cost_requires_finite_nonnegative_number(cost_usd):
    with pytest.raises(ValueError, match="finite non-negative number"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(cost_usd=cost_usd),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(cost_complete=True),
        )


def test_trace_complete_requires_terminal_event_to_be_last():
    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="requested",
        events=(
            BackendExecutionEvent(sequence=0, kind="terminal"),
            BackendExecutionEvent(sequence=1, kind="request"),
        ),
    )

    with pytest.raises(ValueError, match="terminal event must be last"):
        InvocationResult(
            final_message="ok",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(trace_complete=True),
            execution_trace=trace,
        )


def test_execution_trace_requires_contiguous_events_and_tracks_terminal():
    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="requested",
        served_model="served",
        tool_authority_digest="sha256:" + "a" * 64,
        events=(
            BackendExecutionEvent(sequence=0, kind="request"),
            BackendExecutionEvent(sequence=1, kind="usage"),
            BackendExecutionEvent(sequence=2, kind="terminal"),
        ),
    )
    assert trace.schema_version == BACKEND_EXECUTION_TRACE_VERSION
    assert trace.terminal_event_count == 1

    with pytest.raises(ValueError, match="contiguous"):
        BackendExecutionTrace(
            backend="stub",
            requested_model="requested",
            events=(BackendExecutionEvent(sequence=1, kind="terminal"),),
        )


def test_execution_trace_preserves_positional_abi_and_freezes_event_shape():
    events = (BackendExecutionEvent(sequence=0, kind="terminal"),)
    trace = BackendExecutionTrace(
        "stub",
        "requested",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        events,
        BACKEND_EXECUTION_TRACE_VERSION,
    )

    assert trace.events is events
    assert trace.trusted_context_digest is None
    with pytest.raises(TypeError, match="events must be a tuple"):
        BackendExecutionTrace(
            backend="stub",
            requested_model="requested",
            events=list(events),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="entries"):
        BackendExecutionTrace(
            backend="stub",
            requested_model="requested",
            events=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("usage", object(), "usage"),
        ("completeness", object(), "completeness"),
        ("execution_trace", object(), "execution_trace"),
        ("transcript", [], "transcript"),
    ],
)
def test_invocation_result_requires_immutable_typed_evidence(
    field_name,
    value,
    message,
):
    kwargs = {
        "final_message": "ok",
        "transcript": (),
        "usage": Usage(),
        "stop_reason": "end_turn",
    }
    kwargs[field_name] = value
    with pytest.raises(TypeError, match=message):
        InvocationResult(**kwargs)


def test_additive_backend_fields_preserve_legacy_canonical_shapes():
    invocation = Invocation(
        system="sys",
        user="task",
        tools=None,
        max_turns=1,
        budget_usd=None,
    )
    invocation_payload = parse(serialize(invocation))
    assert "conversation" not in invocation_payload
    assert decode_dataclass(invocation_payload, Invocation) == invocation

    result = InvocationResult(
        final_message="ok",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )
    result_payload = parse(serialize(result))
    # These fields predate the additive conversation contract and must retain
    # their canonical wire shape for records written by the immediate parent.
    assert result_payload["completeness"] == {
        "cost_complete": False,
        "terminal_complete": False,
        "token_usage_complete": False,
        "trace_complete": False,
    }
    assert result_payload["execution_trace"] is None
    assert "conversation" not in result_payload
    assert "pricing_evidence" not in result_payload
    assert result_payload["usage"] == {
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    assert decode_dataclass(result_payload, InvocationResult) == result

    pre_execution_evidence_payload = dict(result_payload)
    del pre_execution_evidence_payload["completeness"]
    del pre_execution_evidence_payload["execution_trace"]
    assert (
        decode_dataclass(pre_execution_evidence_payload, InvocationResult)
        == result
    )

    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="requested",
    )
    trace_payload = parse(serialize(trace))
    assert "trusted_context_digest" not in trace_payload
    assert decode_dataclass(trace_payload, BackendExecutionTrace) == trace


@pytest.mark.parametrize("error_kind", ["incomplete", "stale"])
def test_new_backend_errors_round_trip_through_strict_codec(error_kind):
    conversation = None
    if error_kind == "incomplete":
        error = IncompleteEvidence(missing=("trace_complete",))
        stop_reason = "incomplete_evidence"
    else:
        error = StaleConversation(message="provider rejected thread")
        stop_reason = "stale_conversation"
        conversation = ConversationTransportOutcome(
            state="stale",
            opaque_id=None,
            identity_fingerprint="sha256:" + "a" * 64,
            operation_digest="sha256:" + "b" * 64,
        )
    result = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason=stop_reason,
        error=error,
        conversation=conversation,
    )

    restored = decode_dataclass(
        parse(serialize(result)),
        InvocationResult,
    )

    assert restored == result
    assert restored.error is not None
    assert restored.error.kind == result.error.kind


def test_public_backend_annotations_resolve_at_runtime():
    invocation_hints = typing.get_type_hints(Invocation)
    result_hints = typing.get_type_hints(InvocationResult)
    chat_hints = typing.get_type_hints(ChatToolsBackend.chat_with_tools)

    assert "ConversationTransportRequest" in str(
        invocation_hints["conversation"]
    )
    assert "ConversationTransportOutcome" in str(
        result_hints["conversation"]
    )
    assert chat_hints["toolset"] is ToolSet
    assert ToolCallCache in typing.get_args(chat_hints["cache"])


@pytest.mark.parametrize(
    ("field_name", "digest"),
    [
        (field_name, digest)
        for field_name in (
            "feature_profile_digest",
            "tool_authority_digest",
            "trusted_context_digest",
        )
        for digest in (
            "",
            "a" * 64,
            "sha256:" + "A" * 64,
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
        )
    ],
)
def test_execution_trace_rejects_invalid_provenance_digest(
    field_name,
    digest,
):
    with pytest.raises(ValueError, match=field_name):
        BackendExecutionTrace(
            backend="stub",
            requested_model="requested",
            **{field_name: digest},
        )


@pytest.mark.parametrize("sequence", [-1, True, 1.5])
def test_execution_event_requires_exact_nonnegative_sequence(sequence):
    with pytest.raises(ValueError, match="sequence"):
        BackendExecutionEvent(sequence=sequence, kind="request")


@pytest.mark.parametrize(
    "kind",
    ["tool_call", "tool_result", "limit", "cancellation"],
)
def test_execution_event_identity_bearing_kinds_require_name(kind):
    with pytest.raises(ValueError, match="requires name"):
        BackendExecutionEvent(sequence=0, kind=kind)


@pytest.mark.parametrize("kind", ["child_start", "child_terminal"])
def test_child_execution_events_require_child_id(kind):
    with pytest.raises(ValueError, match="requires child_id"):
        BackendExecutionEvent(sequence=0, kind=kind)


def test_backend_feature_set_is_closed_and_conservative_by_default():
    features = BackendFeatureSet()
    assert features.structured_output == "unsupported"
    assert features.project_instructions == "unsuppressible"
    assert features.native_delegation == "none"
    assert BackendFeatureSet.allowed_values("tool_loop") == frozenset(
        {"sdk_owned", "adapter_owned", "none"}
    )
    assert BACKEND_CONFORMANCE_VERSION == "assay.backend-conformance.v2"

    with pytest.raises(ValueError, match="structured_output"):
        BackendFeatureSet(structured_output="best_effort")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown BackendFeatureSet field"):
        BackendFeatureSet.allowed_values("unknown")


def test_feature_aware_backend_is_additive_to_minimal_protocol():
    class MinimalBackend:
        name = "minimal"
        model = "minimal-model"

        def invoke(self, invocation: Invocation) -> InvocationResult:
            raise NotImplementedError

    class DeclaredBackend(MinimalBackend):
        @property
        def features(self) -> BackendFeatureSet:
            return BackendFeatureSet(structured_output="prompt_only")

    assert isinstance(MinimalBackend(), Backend)
    assert not isinstance(MinimalBackend(), FeatureAwareBackend)
    assert isinstance(DeclaredBackend(), FeatureAwareBackend)


def test_backend_protocol_accepts_well_shaped_class():
    class StubBackend:
        name = "stub"
        model = "stub-model"

        def invoke(self, invocation: Invocation) -> InvocationResult:
            return InvocationResult(
                final_message="ok",
                transcript=(),
                usage=Usage(),
                stop_reason="end_turn",
            )

    assert isinstance(StubBackend(), Backend)


@pytest.mark.parametrize(
    "cls,expected_kind",
    [
        (TransportError, "transport"),
        (BudgetExhausted, "budget_exhausted"),
        (MaxTurnsExceeded, "max_turns"),
        (ContentRejected, "content_rejected"),
        (ExternalCancellation, "cancellation"),
    ],
)
def test_backend_error_subtypes_have_expected_kind(cls, expected_kind):
    inst = cls()
    assert inst.kind == expected_kind
    assert isinstance(inst, BackendError)


def test_transport_error_defaults():
    t = TransportError()
    assert t.retriable is True
    assert t.http_status is None


def test_budget_exhausted_not_retriable():
    b = BudgetExhausted(spent_usd=100.0, limit_usd=50.0)
    assert b.retriable is False
    assert b.spent_usd == 100.0


def test_auth_error_deliberately_omitted():
    # Authentication failures use non-retriable TransportError values;
    # there is no second error shape to branch on.
    with pytest.raises(ImportError):
        from assay.types.invocation import AuthError  # noqa: F401


def test_backend_protocol_contract_surface():
    """ARC-9b: pin the Backend / ChatToolsBackend contract so an assay
    change (or a non-conforming crucible adapter) is caught structurally.
    name + model + invoke ⇒ Backend; chat_with_tools additionally ⇒ ChatToolsBackend;
    missing any required member ⇒ not a Backend."""
    from assay.backend.protocol import Backend, ChatToolsBackend

    class MinimalBackend:
        name = "x"
        model = "x-model"
        def invoke(self, invocation): ...

    class MissingInvoke:
        name = "x"
        model = "x-model"

    class MissingName:
        model = "x-model"
        def invoke(self, invocation): ...

    class MissingModel:
        name = "x"
        def invoke(self, invocation): ...

    class ChatBackend:
        name = "x"
        model = "x-model"
        def invoke(self, invocation): ...
        def chat_with_tools(self, invocation, toolset, *, cache=None, max_rounds=25): ...

    assert isinstance(MinimalBackend(), Backend)
    assert not isinstance(MinimalBackend(), ChatToolsBackend)
    assert not isinstance(MissingInvoke(), Backend)
    assert not isinstance(MissingName(), Backend)
    assert not isinstance(MissingModel(), Backend)
    assert isinstance(ChatBackend(), Backend)
    assert isinstance(ChatBackend(), ChatToolsBackend)


def test_invocation_and_message_keep_prompts_out_of_repr():
    """ABK-1: prompt-bearing fields are repr-excluded, so a credential in
    a prompt can't leak through a traceback / SdkDebug repr dump."""
    from assay import Invocation, Message

    secret = "sk-ant-SUPERSECRET-TOKEN"
    msg = Message(role="user", content=secret)
    assert secret not in repr(msg)
    assert "user" in repr(msg)  # non-sensitive field still visible

    inv = Invocation(system=secret, user=secret, tools=None,
                     max_turns=1, budget_usd=None)
    assert secret not in repr(inv)
