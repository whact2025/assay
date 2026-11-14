# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict review_v2 contract tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Message,
    Rubric,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.primitives.review import (
    _CHUNK_SYSTEM_SUFFIX,
    _MAX_ARTIFACT_CHARS,
    _MAX_JSON_SCALAR_BIND_CANDIDATES,
    _MAX_MAPPER_CHUNKS,
    _MAX_MAPPER_ENVELOPE_CHARS,
    _SourceChunk,
    _bounded_source_chunks,
    _canonicalize_review_response,
    _chunk_user_message,
    _elide_artifact,
    _extract_json_object,
    _normalize_severity,
    _parse_findings,
    _passes,
    _resolve_artifact_text,
    _resolve_json_container,
    _resolve_json_scalar,
    _review_sources,
    _source_anchors,
    _source_chunks,
    _text_hash,
    ReviewResponseRepairPolicy,
    review,
)


def _make_rubric(**overrides) -> Rubric:
    values = {
        "name": "test_rubric",
        "system_prompt": "Review under review_v2.",
        "user_template": "Review:\n{artifact}",
    }
    values.update(overrides)
    return Rubric(**values)


def _finding(**overrides) -> dict:
    value = {
        "id": "f1",
        "criterion_id": "scope.in_bounds",
        "severity": "High",
        "confidence": 0.9,
        "disposition": "confirmed",
        "category": "scope",
        "targets": [],
        "evidence": [
            {
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "line 1",
                "excerpt": "needle",
            }
        ],
        "proposal": {"action": "remove it", "replacement": None},
        "attack_spec": None,
    }
    value.update(overrides)
    return value


def _response(**overrides) -> dict:
    value = {
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The complete artifact satisfies the rubric.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
    }
    value.update(overrides)
    return value


def _criterion_result(**overrides) -> dict:
    value = {
        "criterion_id": "criterion.one",
        "outcome": "satisfied",
        "reasoning": "The cited source supports the result.",
        "evidence": [{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "line 1",
            "excerpt": "needle",
        }],
        "headline": True,
    }
    value.update(overrides)
    return value


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_CHUNK_TRACE = BackendExecutionTrace(
    backend="chunking-stub",
    requested_model="chunking-stub-model",
    events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
)


class StubBackend:
    name = "stub"
    model = "stub-model"

    def __init__(
        self,
        final: str,
        error=None,
        *,
        completeness: ExecutionCompleteness | None = None,
        execution_trace: BackendExecutionTrace | None = None,
    ):
        self.final = final
        self.error = error
        self.completeness = completeness
        self.execution_trace = execution_trace
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        result_kwargs = {}
        if self.completeness is not None:
            result_kwargs["completeness"] = self.completeness
        if self.execution_trace is not None:
            result_kwargs["execution_trace"] = self.execution_trace
        return InvocationResult(
            final_message=self.final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=self.final),
            ),
            usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.005),
            stop_reason="end_turn",
            error=self.error,
            **result_kwargs,
        )


class ChunkingBackend:
    """Protocol-aware backend for deterministic chunk/reducer tests."""

    name = "chunking-stub"
    model = "chunking-stub-model"

    def __init__(self, *, reducer_findings: list[dict] | None = None):
        self.invocations: list[Invocation] = []
        self.reducer_findings = reducer_findings or []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        request = json.loads(invocation.user)
        if request["schema_version"] == "assay.review.chunk.v1":
            payload = {
                key: request[key]
                for key in (
                    "schema_version",
                    "source_id",
                    "source_sha256",
                    "source_chars",
                    "chunk_index",
                    "chunk_count",
                    "char_start",
                    "char_end",
                    "chunk_sha256",
                    "evidence_addressable",
                )
            }
            payload.update({"summary": "No local issue observed.", "findings": []})
        else:
            assert request["schema_version"] == "assay.review.reduce.v1"
            payload = _response(
                coverage={
                    "complete": True,
                    "sources_reviewed": request["required_sources"],
                    "omissions": [],
                },
                findings=self.reducer_findings,
            )
        final = json.dumps(payload)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.005),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_CHUNK_TRACE,
        )


def _run(tmp_path: Path, payload: dict | str, **kwargs):
    final = payload if isinstance(payload, str) else json.dumps(payload)
    return review(
        b"artifact with needle in it",
        _make_rubric(**kwargs.pop("rubric", {})),
        backend=StubBackend(final),
        corpus=Corpus(tmp_path),
        **kwargs,
    )


# Rubric and artifact sources -------------------------------------------------


def test_rubric_defaults_to_review_v2():
    rubric = Rubric.from_yaml_text(
        'name: r\nsystem_prompt: strict\nuser_template: "Review {artifact}"\n'
    )
    assert rubric.response_format == "review_v2"
    assert rubric.allowed_decisions == (
        "pass", "revise", "reject", "inconclusive"
    )


def test_rubric_rejects_non_v2_response_format():
    with pytest.raises(ValueError, match="review_v2"):
        _make_rubric(response_format="json_v1")


def test_rubric_validates_allowed_decisions_and_template():
    assert _make_rubric(allowed_decisions=("pass", "revise", "pass")).allowed_decisions == (
        "pass", "revise"
    )
    with pytest.raises(ValueError, match="unknown decision"):
        _make_rubric(allowed_decisions=("approve",))
    with pytest.raises(ValueError, match="artifact"):
        _make_rubric(user_template="no placeholder")


def test_rubric_normalizes_and_validates_floor():
    assert _make_rubric(severity_floor="critical ").severity_floor == "Critical"
    with pytest.raises(ValueError, match="severity_floor"):
        _make_rubric(severity_floor="Blocker")


def test_resolve_artifact_sources(tmp_path: Path):
    corpus = Corpus(tmp_path / "corpus")
    assert _resolve_artifact_text(b"bytes", corpus)[1] == "bytes"
    path = tmp_path / "artifact.txt"
    path.write_text("path", encoding="utf-8")
    assert _resolve_artifact_text(path, corpus)[1] == "path"
    ref = corpus.put_artifact(b"stored", kind="inputs")
    assert _resolve_artifact_text(ref, corpus)[1] == "stored"
    with pytest.raises(TypeError, match="sha256"):
        _resolve_artifact_text("bare string", corpus)


# Exact one-object parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        '{"schema_version":"assay.review.v2"}',
        '```json\n{"schema_version":"assay.review.v2"}\n```',
        '```\n{"schema_version":"assay.review.v2"}\n```',
    ],
)
def test_extract_accepts_one_complete_object(message: str):
    assert _extract_json_object(message) == {
        "schema_version": "assay.review.v2"
    }


@pytest.mark.parametrize(
    "message",
    [
        'prose {"decision":"pass"}',
        '{"decision":"pass"} trailing',
        '{"decision":"pass"}{"decision":"reject"}',
        '[{"decision":"pass"}]',
        '{"decision":"pass","decision":"reject"}',
        '```json\n{"decision":"pass"}\n``` trailing',
        "not json",
    ],
)
def test_extract_rejects_ambiguous_or_non_object_output(message: str):
    assert _extract_json_object(message) is None


# Strict schema and evidence -------------------------------------------------


def test_review_pass_is_derived_from_decision(tmp_path: Path):
    verdict = _run(tmp_path, _response())
    assert verdict.decision == "pass"
    assert verdict.passed is True
    assert verdict.coverage.complete is True
    assert verdict.coverage.sources_reviewed == ("artifact",)
    assert verdict.input_tokens == 1
    assert verdict.output_tokens == 2


def test_model_passed_field_is_forbidden(tmp_path: Path):
    payload = _response(passed=True)
    verdict = _run(tmp_path, payload)
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"
    assert "unexpected=['passed']" in verdict.reasoning


@pytest.mark.parametrize(
    "field",
    ["schema_version", "decision", "reasoning", "coverage", "findings"],
)
def test_every_root_field_is_required(tmp_path: Path, field: str):
    payload = _response()
    del payload[field]
    verdict = _run(tmp_path, payload)
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"


def test_reasoning_must_be_nonempty(tmp_path: Path):
    verdict = _run(tmp_path, _response(reasoning="  "))
    assert verdict.decision == "inconclusive"


def test_revise_with_fully_typed_grounded_finding(tmp_path: Path):
    payload = _response(
        decision="revise",
        reasoning="A blocking scope issue is grounded in the artifact.",
        findings=[_finding()],
    )
    verdict = _run(tmp_path, payload)
    assert verdict.decision == "revise"
    assert verdict.passed is False
    finding = verdict.findings[0]
    assert finding.criterion_id == "scope.in_bounds"
    assert finding.confidence == 0.9
    assert finding.evidence[0].source_id == "artifact"
    assert finding.proposal is not None
    assert finding.suggested_action == "remove it"


def test_response_repair_policy_addition_preserves_positional_signature():
    policy = ReviewResponseRepairPolicy(
        False,
        True,
        False,
        True,
        True,
        False,
        256,
    )

    assert policy.repair_unique_exact_excerpt_char_spans is False
    assert policy.materialize_source_anchors is True
    assert policy.materialize_json_paths is False
    assert policy.require_source_anchors is True
    assert policy.derive_decision_from_criterion_results is True
    assert policy.recover_fully_satisfied_revise_to_pass is False
    assert policy.source_anchor_chars == 256
    assert (
        policy.repair_unresolved_source_anchor_from_unique_exact_excerpt
        is False
    )
    assert policy.repair_unique_whitespace_normalized_excerpt_char_spans is False
    assert policy.repair_unique_json_quote_escaped_excerpt_char_spans is False


def test_response_repair_policy_is_opt_in_and_repairs_only_unique_exact_span(
    tmp_path: Path,
):
    payload = _response(
        decision="revise",
        reasoning="A blocking issue is grounded in the artifact.",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "line 1 chars:0-6",
            "excerpt": "needle",
        }])],
    )
    without_policy = _run(tmp_path / "disabled", payload)
    assert without_policy.decision == "inconclusive"
    assert "review_response_repair" not in without_policy.artifacts

    corpus = Corpus(tmp_path / "enabled")
    final = json.dumps(payload)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=StubBackend(final),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(),
    )

    assert verdict.decision == "revise"
    assert verdict.findings[0].evidence[0].location == "line 1 chars:14-20"
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["schema_version"] == "assay.review-response-repair.v1"
    assert repair["actions"] == [{
        "operation": "repair_unique_exact_excerpt_char_span",
        "path": "findings[0].evidence[0]",
        "source_id": "artifact",
        "source_sha256": repair["actions"][0]["source_sha256"],
        "excerpt_sha256": repair["actions"][0]["excerpt_sha256"],
        "before_location": "line 1 chars:0-6",
        "after_location": "line 1 chars:14-20",
    }]
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript[-1] == {"role": "assistant", "content": final}
    raw_ref = verdict.artifacts["review_raw_response"]
    assert corpus.get_artifact(raw_ref) == final.encode()
    assert repair["raw_response_ref"] == raw_ref
    assert repair["raw_response_sha256"] == raw_ref


def test_response_repair_preserves_meaningful_outer_whitespace(tmp_path: Path):
    excerpt = " needle "
    payload = _response(
        decision="revise",
        reasoning="A blocking issue is grounded in the artifact.",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "line 1 chars:0-8",
            "excerpt": excerpt,
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.excerpt == excerpt
    assert evidence.location == "line 1 chars:13-21"
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["actions"] == [{
        "operation": "repair_unique_exact_excerpt_char_span",
        "path": "findings[0].evidence[0]",
        "source_id": "artifact",
        "source_sha256": repair["actions"][0]["source_sha256"],
        "excerpt_sha256": repair["actions"][0]["excerpt_sha256"],
        "before_location": "line 1 chars:0-8",
        "after_location": "line 1 chars:13-21",
    }]


def test_response_repair_rejects_all_whitespace_excerpt(tmp_path: Path):
    source = "alpha \tbeta"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:0-2",
            "excerpt": " \t",
        }])],
    )
    policy = ReviewResponseRepairPolicy()
    canonicalized = _canonicalize_review_response(
        payload,
        raw_response=json.dumps(payload),
        rubric=_make_rubric(),
        trusted_sources={"artifact": source},
        anchors=(),
        policy=policy,
    )

    assert canonicalized.payload == payload
    assert canonicalized.report is None

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=policy,
    )
    assert verdict.decision == "inconclusive"
    assert "expected non-empty string" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def _run_json_quote_escape_review(
    tmp_path: Path,
    source: str,
    excerpt: str,
    *,
    location: object = "descriptive location",
    require_source_anchors: bool = False,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": location,
            "excerpt": excerpt,
        }])],
    )
    corpus = Corpus(tmp_path)
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=require_source_anchors,
        require_source_anchors=require_source_anchors,
        repair_unique_json_quote_escaped_excerpt_char_spans=True,
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=policy,
    )
    return verdict, corpus


def test_response_repair_restores_unique_json_quote_escaped_excerpt(
    tmp_path: Path,
):
    source = json.dumps(
        {
            "phase": "post_fix",
            "records": [{"path": r"C:\tmp\result"}],
        },
        separators=(",", ":"),
    )
    exact_excerpt = source[1:-1]
    supplied = exact_excerpt.replace('"', r'\"')
    assert r"C:\\tmp\\result" in supplied

    verdict, corpus = _run_json_quote_escape_review(
        tmp_path,
        source,
        supplied,
        location="post-fix record",
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    start = source.index(exact_excerpt)
    assert evidence.excerpt == exact_excerpt
    assert evidence.location == f"chars:{start}-{start + len(exact_excerpt)}"
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["actions"] == [{
        "operation": "repair_unique_json_quote_escaped_excerpt_char_span",
        "path": "findings[0].evidence[0]",
        "source_id": "artifact",
        "source_sha256": _text_hash(source),
        "before": {
            "location": "post-fix record",
            "excerpt_sha256": _text_hash(supplied),
        },
        "after": {
            "location": evidence.location,
            "excerpt_sha256": _text_hash(exact_excerpt),
        },
    }]
    assert repair["policy"][
        "repair_unique_json_quote_escaped_excerpt_char_spans"
    ] is True


def test_json_quote_escape_repair_removes_exactly_one_escape_level(
    tmp_path: Path,
):
    source = json.dumps(
        {"message": 'say "hello"'},
        separators=(",", ":"),
    )
    exact_excerpt = source[1:-1]
    supplied = exact_excerpt.replace('"', r'\"')
    assert r'\\"hello\\"' in supplied

    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        source,
        supplied,
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.excerpt == exact_excerpt
    assert r'\"hello\"' in evidence.excerpt


def test_json_quote_escape_repair_does_not_rewrite_exact_escaped_quote(
    tmp_path: Path,
):
    source = json.dumps(
        {"message": 'say "hello"'},
        separators=(",", ":"),
    )
    exact_excerpt = r'\"hello\"'

    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        source,
        exact_excerpt,
    )

    assert verdict.decision == "revise"
    assert verdict.findings[0].evidence[0].excerpt == exact_excerpt
    assert verdict.findings[0].evidence[0].location == "descriptive location"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    "source",
    (
        'prefix "phase":"post_fix" suffix',
        '{"phase":"post_fix","phase":"duplicate"}',
        '{"phase":"post_fix"}{"second":true}',
        '{"phase":"post_fix","constant":NaN}',
    ),
)
def test_json_quote_escape_repair_requires_one_complete_strict_json_document(
    tmp_path: Path,
    source: str,
):
    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        source,
        r'\"phase\":\"post_fix\"',
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_json_quote_escape_repair_rejects_ambiguous_exact_candidate(
    tmp_path: Path,
):
    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        '[{"phase":"post_fix"},{"phase":"post_fix"}]',
        r'\"phase\":\"post_fix\"',
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    ("value", "source_escape", "reviewer_drift"),
    (
        ("alpha\nbeta", r"\n", "\n"),
        ("alpha\tbeta", r"\t", "\t"),
        ("alpha", r"\u0061", "a"),
        (r"C:\tmp", r"\\", "\\"),
    ),
)
def test_json_quote_escape_repair_does_not_decode_other_json_escapes(
    tmp_path: Path,
    value: str,
    source_escape: str,
    reviewer_drift: str,
):
    source = json.dumps(
        {"message": value},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if source_escape == r"\u0061":
        source = source.replace("alpha", r"\u0061lpha")
    exact_excerpt = source[1:-1]
    assert source_escape in exact_excerpt
    supplied = exact_excerpt.replace('"', r'\"').replace(
        source_escape,
        reviewer_drift,
    )

    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        source,
        supplied,
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    "location",
    (
        "anchor:sha256:" + "0" * 64,
        "json:$.phase",
        "chars:0-5",
        "chars:not-a-span",
        "line 1 chars:0-5",
        "line(chars:0-5)",
        "(chars:999-1000)",
        None,
        17,
        "",
        "  ",
    ),
)
def test_json_quote_escape_repair_requires_descriptive_location(
    tmp_path: Path,
    location: object,
):
    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        '{"phase":"post_fix"}',
        r'\"phase\":\"post_fix\"',
        location=location,
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_required_anchor_policy_disables_json_quote_escape_repair(
    tmp_path: Path,
):
    verdict, _corpus = _run_json_quote_escape_review(
        tmp_path,
        '{"phase":"post_fix"}',
        r'\"phase\":\"post_fix\"',
        require_source_anchors=True,
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_response_repair_can_restore_one_unique_whitespace_normalized_quote(
    tmp_path: Path,
):
    source = (
        "prefix this\nis directory materialization for that exact "
        "creates_path, not an expanded\npath scope suffix"
    )
    supplied = (
        "this is directory materialization for that exact creates_path, not an expanded"
    )
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "T3 path-classification correction",
            "excerpt": supplied,
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    expected = supplied.replace("this is", "this\nis")
    assert evidence.excerpt == expected
    start = source.index(expected)
    assert evidence.location == f"chars:{start}-{start + len(expected)}"
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["actions"] == [{
        "operation": "repair_unique_whitespace_normalized_excerpt_char_span",
        "path": "findings[0].evidence[0]",
        "source_id": "artifact",
        "source_sha256": repair["actions"][0]["source_sha256"],
        "before": {
            "location": "T3 path-classification correction",
            "excerpt_sha256": repair["actions"][0]["before"]["excerpt_sha256"],
        },
        "after": {
            "location": evidence.location,
            "excerpt_sha256": repair["actions"][0]["after"]["excerpt_sha256"],
        },
    }]


@pytest.mark.parametrize(
    ("separator", "escaped_separator"),
    (("\n", r"\n"), ("\r", r"\r"), ("\t", r"\t")),
)
def test_response_repair_restores_json_escaped_whitespace_quote(
    tmp_path: Path,
    separator: str,
    escaped_separator: str,
):
    source = json.dumps([f"this{separator}is directory materialization"])
    supplied = "this is directory materialization"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": supplied,
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    exact_excerpt = f"this{escaped_separator}is directory materialization"
    start = source.index(exact_excerpt)
    evidence = verdict.findings[0].evidence[0]
    assert evidence.location == f"chars:{start}-{start + len(exact_excerpt)}"
    assert evidence.excerpt == exact_excerpt


def test_response_repair_does_not_decode_literal_json_backslash_n(
    tmp_path: Path,
):
    source = json.dumps([r"this\nis lexical content"])
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": "this is lexical content",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    ("source", "excerpt"),
    (
        (
            'prefix "alpha beta gamma" suffix',
            '"alpha\\nbeta gamma"',
        ),
        (
            'prefix ["alpha beta gamma"] suffix',
            '["alpha\\nbeta gamma"]',
        ),
    ),
)
def test_response_repair_does_not_let_quote_enable_json_escape_decoding(
    tmp_path: Path,
    source: str,
    excerpt: str,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": excerpt,
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    ("escape", "constant"),
    (("n", "NaN"), ("t", "Infinity"), ("r", "-Infinity")),
)
def test_response_repair_does_not_decode_nonstandard_json_document(
    tmp_path: Path,
    escape: str,
    constant: str,
):
    source = f'{{"x":"alpha\\{escape}beta","bad":{constant}}}'
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": "alpha beta",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    "location",
    (
        "anchor:sha256:" + "0" * 64,
        "json:$.missing",
        None,
        17,
        "",
        "  ",
    ),
)
def test_whitespace_repair_does_not_replace_machine_or_invalid_location(
    tmp_path: Path,
    location: object,
):
    source = "alpha\nbeta"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": location,
            "excerpt": "alpha beta",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            materialize_json_paths=True,
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_required_anchor_policy_disables_whitespace_quote_repair(
    tmp_path: Path,
):
    source = "alpha\nbeta"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": "alpha beta",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            require_source_anchors=True,
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


@pytest.mark.parametrize(
    ("source", "force_recursion_error"),
    (
        pytest.param(
            '{"x":"alpha\\nbeta","x":"duplicate"}', False, id="duplicate-keys",
        ),
        pytest.param('["alpha\\nbeta"]', True, id="recursion-error"),
    ),
)
def test_whitespace_repair_json_probe_failures_are_contained(
    tmp_path: Path,
    source: str,
    force_recursion_error: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    probe_failures = []
    if force_recursion_error:
        real_loads = json.loads

        def loads_with_probe_failure(candidate, *args, **kwargs):
            if candidate == source and "parse_constant" in kwargs:
                # Exercise probe containment independently of Python's stack limit.
                probe_failures.append(candidate)
                raise RecursionError("simulated JSON probe recursion limit")
            return real_loads(candidate, *args, **kwargs)

        monkeypatch.setattr(json, "loads", loads_with_probe_failure)

    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": "alpha beta",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts
    if force_recursion_error:
        assert probe_failures


@pytest.mark.parametrize(
    ("source", "excerpt"),
    [
        ("alpha\nbeta and alpha\tbeta", "alpha beta"),
        ("alpha\nbeta", "alpha gamma"),
        (" alpha\nbeta ", " alpha beta "),
        ("alpha\nbeta", "   "),
    ],
)
def test_whitespace_normalized_quote_repair_fails_closed(
    tmp_path: Path,
    source: str,
    excerpt: str,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "semantic location",
            "excerpt": excerpt,
        }])],
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_repaired_raw_response_is_reachable_when_transcript_omits_final(
    tmp_path: Path,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:0-6",
            "excerpt": "needle",
        }])],
    )
    final = json.dumps(payload)

    class TranscriptlessBackend(StubBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            return dataclasses.replace(
                result,
                transcript=(Message(role="assistant", content="different"),),
            )

    corpus = Corpus(tmp_path)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=TranscriptlessBackend(final),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(),
    )
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript == [{"role": "assistant", "content": "different"}]
    raw_ref = verdict.artifacts["review_raw_response"]
    assert corpus.get_artifact(raw_ref) == final.encode()
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["raw_response_ref"] == raw_ref
    assert report["raw_response_sha256"] == raw_ref


def test_span_repair_rejects_ambiguous_excerpt(tmp_path: Path):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:99-105",
            "excerpt": "needle",
        }])],
    )
    verdict = review(
        b"needle and needle",
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(),
    )
    assert verdict.decision == "inconclusive"
    assert "review_response_repair" not in verdict.artifacts


def test_evidence_provider_cannot_override_retained_artifact(tmp_path: Path):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:0-6",
            "excerpt": "needle",
        }])],
    )
    with pytest.raises(ValueError, match="conflicting retained text"):
        review(
            b"artifact with needle in it",
            _make_rubric(),
            backend=StubBackend(json.dumps(payload)),
            corpus=Corpus(tmp_path),
            evidence_source_provider=lambda: {"artifact": "needle redirected"},
            response_repair_policy=ReviewResponseRepairPolicy(),
        )


def test_response_canonicalization_is_idempotent_without_second_report():
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:0-6",
            "excerpt": "needle",
        }])],
    )
    policy = ReviewResponseRepairPolicy()
    source = "artifact with needle in it"
    first = _canonicalize_review_response(
        payload,
        raw_response=json.dumps(payload),
        rubric=_make_rubric(),
        trusted_sources={"artifact": source},
        anchors=(),
        policy=policy,
    )
    second = _canonicalize_review_response(
        first.payload,
        raw_response=json.dumps(first.payload),
        rubric=_make_rubric(),
        trusted_sources={"artifact": source},
        anchors=(),
        policy=policy,
    )
    assert first.report is not None
    assert second.payload == first.payload
    assert second.report is None


def test_source_anchor_materializes_exact_source_fields(tmp_path: Path):
    source = "artifact with needle in it"
    anchor = _source_anchors("artifact", source, max_chars=512)[0]
    payload = _response(
        decision="revise",
        reasoning="The cited anchor contains the blocking issue.",
        findings=[_finding(evidence=[{
            "source_id": "invented",
            "kind": "source_quote",
            "location": f"anchor:{anchor.anchor_id}",
            "excerpt": "",
        }])],
    )
    backend = StubBackend(json.dumps(payload))
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "artifact"
    assert evidence.location == f"chars:0-{len(source)}"
    assert evidence.excerpt == source
    assert anchor.anchor_id in backend.invocations[0].user
    assert "DETERMINISTIC EVIDENCE REFERENCES" in backend.invocations[0].system
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][0]["operation"] == "materialize_source_anchor"
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    assert rubric_artifact["system_prompt"] == backend.invocations[0].system
    assert rubric_artifact["declared_system_prompt"] == (
        _make_rubric().system_prompt
    )


def test_source_anchor_preserves_boundary_whitespace_in_exact_span(
    tmp_path: Path,
):
    source = "\n  exact anchored evidence  \n"
    anchor = _source_anchors("artifact", source, max_chars=512)[0]
    payload = _response(
        decision="revise",
        reasoning="The cited anchor contains the blocking issue.",
        findings=[_finding(evidence=[{
            "source_id": "invented",
            "kind": "source_quote",
            "location": f"anchor:{anchor.anchor_id}",
            "excerpt": "",
        }])],
    )

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "artifact"
    assert evidence.location == f"chars:0-{len(source)}"
    assert evidence.excerpt == source


def test_unknown_anchor_is_reserved_and_fails_closed(tmp_path: Path):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "anchor:sha256:" + "0" * 64,
            "excerpt": "needle",
        }])],
    )
    verdict = _run(
        tmp_path,
        payload,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
        ),
    )
    assert verdict.decision == "inconclusive"
    assert "unresolved deterministic source anchor" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def test_unknown_anchor_with_unique_exact_excerpt_is_repaired_when_enabled(
    tmp_path: Path,
):
    source = "prefix exact retained evidence suffix"
    excerpt = "exact retained evidence"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "anchor:sha256:" + "0" * 64,
            "excerpt": excerpt,
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            repair_unresolved_source_anchor_from_unique_exact_excerpt=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    start = source.index(excerpt)
    assert evidence.source_id == "artifact"
    assert evidence.location == f"chars:{start}-{start + len(excerpt)}"
    assert evidence.excerpt == excerpt
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    action = report["actions"][0]
    assert action["operation"] == (
        "repair_unresolved_source_anchor_from_unique_exact_excerpt"
    )
    assert action["before"]["anchor_id"] == "sha256:" + "0" * 64
    assert action["before"]["excerpt_sha256"] == action["after"][
        "excerpt_sha256"
    ]


@pytest.mark.parametrize(
    ("source_id", "source", "excerpt"),
    [
        ("artifact", "needle and needle", "needle"),
        ("artifact", "aaa", "aa"),
        ("missing", "one exact needle", "needle"),
        ("artifact", "one exact needle", ""),
        ("artifact", "one exact needle", "not retained"),
    ],
    ids=[
        "ambiguous",
        "overlapping-ambiguous",
        "wrong-source",
        "blank",
        "absent",
    ],
)
def test_unknown_anchor_exact_excerpt_repair_remains_fail_closed(
    tmp_path: Path,
    source_id: str,
    source: str,
    excerpt: str,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": source_id,
            "kind": "source_quote",
            "location": "anchor:sha256:" + "0" * 64,
            "excerpt": excerpt,
        }])],
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            repair_unresolved_source_anchor_from_unique_exact_excerpt=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "unresolved deterministic source anchor" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def test_unknown_anchor_exact_excerpt_repair_does_not_relax_required_catalog(
    tmp_path: Path,
):
    source = "prefix exact retained evidence suffix"
    excerpt = "exact retained evidence"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "anchor:sha256:" + "0" * 64,
            "excerpt": excerpt,
        }])],
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            require_source_anchors=True,
            repair_unresolved_source_anchor_from_unique_exact_excerpt=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "unresolved deterministic source anchor" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def test_unknown_anchor_exact_excerpt_repair_requires_anchor_materialization(
    tmp_path: Path,
):
    source = "prefix exact retained evidence suffix"
    excerpt = "exact retained evidence"
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "anchor:sha256:" + "0" * 64,
            "excerpt": excerpt,
        }])],
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unresolved_source_anchor_from_unique_exact_excerpt=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "unresolved deterministic evidence reference" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def test_distinct_retained_source_is_appended_as_anchor_catalog(tmp_path: Path):
    retained = "retained evidence source"
    anchor = _source_anchors("artifact", retained, max_chars=512)[0]
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": f"anchor:{anchor.anchor_id}",
            "excerpt": "",
        }])],
    )
    backend = StubBackend(json.dumps(payload))
    verdict = review(
        b"raw artifact bytes",
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
        prompt_artifact_text="prompt envelope",
        evidence_source_texts={"artifact": retained},
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
        ),
    )
    assert verdict.decision == "revise"
    assert verdict.findings[0].evidence[0].excerpt == retained
    assert "assay.untrusted-source.anchored.v1" not in backend.invocations[0].user
    assert "assay.evidence-anchor-catalog.v1" in backend.invocations[0].user
    assert anchor.anchor_id in backend.invocations[0].user


def test_json_path_materializes_unique_exact_scalar_token(tmp_path: Path):
    source = '{"tier":{"name":"needle"}}'
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "json:$.tier.name",
            "excerpt": "",
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.excerpt == '"needle"'
    start = source.index('"needle"')
    assert evidence.location == f"chars:{start}-{start + len(evidence.excerpt)}"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][0]["operation"] == (
        "materialize_json_scalar_path"
    )


def test_json_path_materializes_pretty_container_span(tmp_path: Path):
    source = '''{
  "tiers": [
    {"depends_on": []},
    {"depends_on": [
      "t0"
    ]}
  ]
}'''
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "tier_dependency",
            "location": "json:$.tiers[1].depends_on",
            "excerpt": "[t0]",
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    expected = '[\n      "t0"\n    ]'
    assert evidence.excerpt == expected
    start = source.index(expected)
    assert evidence.location == f"chars:{start}-{start + len(expected)}"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][0]["operation"] == (
        "materialize_json_container_path"
    )


def test_json_path_binds_repeated_identical_container_to_addressed_path():
    source = '{"tiers":[{"depends_on":["t0"]},{"depends_on":["t0"]}]}'

    resolved = _resolve_json_container(
        source,
        "json:$.tiers[1].depends_on",
    )

    assert resolved is not None
    token, start, end = resolved
    assert token == '["t0"]'
    assert start == source.rindex(token)
    assert end == start + len(token)


@pytest.mark.parametrize(
    "source",
    (
        '{"tier":{"depends_on":[]},"tier":{"depends_on":[]}}',
        '{"tier":{"depends_on":[]}',
        '{"tier":{"depends_on":[NaN]}}',
    ),
)
def test_json_container_path_rejects_non_strict_document(source: str):
    assert _resolve_json_container(
        source,
        "json:$.tier.depends_on",
    ) is None


def test_json_path_materializes_addressed_duplicate_object_value(
    tmp_path: Path,
):
    source = '{"id":"slice-1","slice_id":"slice-1"}'
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "json:$.slice_id",
            "excerpt": "",
        }])],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "revise"
    evidence = verdict.findings[0].evidence[0]
    assert evidence.excerpt == '"slice-1"'
    start = source.rindex(evidence.excerpt)
    assert evidence.location == f"chars:{start}-{start + len(evidence.excerpt)}"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][0]["operation"] == (
        "materialize_json_scalar_path"
    )


@pytest.mark.parametrize(
    ("source", "location", "expected_start"),
    (
        ('{"items":["same","same"]}', "json:$.items[1]", 17),
        ('{"same":"same"}', "json:$.same", 8),
    ),
)
def test_json_scalar_path_binds_duplicate_array_values_and_object_keys(
    source: str,
    location: str,
    expected_start: int,
):
    assert _resolve_json_scalar(source, location) == (
        '"same"',
        expected_start,
        expected_start + len('"same"'),
    )


def test_json_scalar_path_returns_addressed_raw_spelling():
    source = '{"needle":"decoy","value":"\\u006e\\u0065\\u0065\\u0064\\u006c\\u0065"}'
    token = '"\\u006e\\u0065\\u0065\\u0064\\u006c\\u0065"'
    start = source.index(token)

    assert _resolve_json_scalar(source, "json:$.value") == (
        token,
        start,
        start + len(token),
    )


def test_json_scalar_path_binds_duplicate_numeric_raw_spellings():
    source = '{"decimal":1.0,"exponent":1e0}'
    token = "1e0"
    start = source.index(token)

    assert _resolve_json_scalar(source, "json:$.exponent") == (
        token,
        start,
        start + len(token),
    )


def test_json_scalar_path_numeric_prefixes_do_not_consume_candidate_budget():
    decoys = ",".join(
        "10" for _ in range(_MAX_JSON_SCALAR_BIND_CANDIDATES + 1)
    )
    source = f'{{"decoys":[{decoys}],"target":0}}'
    start = source.rindex("0")

    assert _resolve_json_scalar(source, "json:$.target") == (
        "0",
        start,
        start + 1,
    )


def test_json_scalar_path_accepts_exact_candidate_limit():
    source = json.dumps({
        "target": "same",
        "decoys": [
            "same"
            for _ in range(_MAX_JSON_SCALAR_BIND_CANDIDATES - 1)
        ],
    })
    token = '"same"'
    start = source.index(token)

    assert _resolve_json_scalar(source, "json:$.target") == (
        token,
        start,
        start + len(token),
    )


def test_json_scalar_path_rejects_oversized_candidate_set_before_reparse(
    monkeypatch: pytest.MonkeyPatch,
):
    source = json.dumps({
        "target": "same",
        "decoys": [
            "same" for _ in range(_MAX_JSON_SCALAR_BIND_CANDIDATES)
        ],
    })
    decode_calls = 0
    original_decode = json.JSONDecoder.decode

    def counted_decode(decoder, text, *args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(decoder, text, *args, **kwargs)

    monkeypatch.setattr(json.JSONDecoder, "decode", counted_decode)

    assert _resolve_json_scalar(source, "json:$.target") is None
    assert decode_calls == 1


def test_json_path_over_candidate_limit_is_inconclusive_without_repair(
    tmp_path: Path,
):
    source = json.dumps({
        "target": "same",
        "decoys": [
            "same" for _ in range(_MAX_JSON_SCALAR_BIND_CANDIDATES)
        ],
    })
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "json:$.target",
            "excerpt": "",
        }])],
    )
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert "unresolved deterministic JSON path" in verdict.reasoning
    assert "review_response_repair" not in verdict.artifacts


def test_json_scalar_path_rejects_escaped_equivalent_duplicate_keys():
    source = '{"id":"first","\\u0069\\u0064":"second"}'
    assert _resolve_json_scalar(source, "json:$.id") is None


@pytest.mark.parametrize(
    "source",
    (
        '{"id":"first","id":"second"}',
        '{"id":"unterminated}',
        '{"id":NaN}',
    ),
)
def test_json_scalar_path_rejects_non_strict_document(source: str):
    assert _resolve_json_scalar(source, "json:$.id") is None


@pytest.mark.parametrize(
    "location",
    ["line1", "module.path", "tiers[0]", "$.tier.name"],
)
def test_bare_legacy_location_is_never_interpreted_as_json_path(
    tmp_path: Path,
    location: str,
):
    payload = _response(
        decision="revise",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": location,
            "excerpt": "needle",
        }])],
    )
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            materialize_json_paths=True,
            derive_decision_from_criterion_results=True,
        ),
    )
    assert verdict.decision == "revise"
    assert verdict.findings[0].evidence[0].location == location
    assert "review_response_repair" not in verdict.artifacts


def test_oversized_anchor_render_degrades_policy_without_protocol_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", 5_000)
    source = "A" * 3_000
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )
    backend = StubBackend(json.dumps(_response()))
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        response_repair_policy=policy,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 1
    assert backend.invocations[0].user.startswith("Review:\n")
    assert "assay.review.chunk.v1" not in backend.invocations[0].user
    assert "assay.untrusted-source.anchored.v1" not in backend.invocations[0].user
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    effective = rubric_artifact["response_repair_policy"]
    assert effective["materialize_source_anchors"] is False
    assert effective["materialize_json_paths"] is True
    assert rubric_artifact["response_repair_capabilities"] == {
        "source_anchor_catalog_available": False,
        "source_anchor_count": 0,
    }
    degradation = rubric_artifact["response_repair_degradation"]
    assert degradation["reason"] == "anchor_render_limit_exceeded"
    assert degradation["requested_anchor_render_chars"] > 5_000
    assert degradation["effective_rendered_chars"] <= 5_000
    assert degradation["anchor_render_limit_chars"] == 5_000
    assert degradation["requested_response_repair_policy"] == {
        "repair_unique_exact_excerpt_char_spans": True,
        "repair_unresolved_source_anchor_from_unique_exact_excerpt": False,
        "repair_unique_whitespace_normalized_excerpt_char_spans": False,
        "repair_unique_json_quote_escaped_excerpt_char_spans": False,
        "materialize_source_anchors": True,
        "materialize_json_paths": True,
        "require_source_anchors": False,
        "derive_decision_from_criterion_results": False,
        "recover_fully_satisfied_revise_to_pass": False,
        "source_anchor_chars": 512,
    }
    assert degradation["effective_response_repair_policy"] == effective


def test_oversized_required_anchor_only_render_fails_before_backend_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", 5_000)
    source = "A" * 3_000
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=False,
        require_source_anchors=True,
    )
    backend = StubBackend(json.dumps(_response()))
    corpus = Corpus(tmp_path)

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        response_repair_policy=policy,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "review_prompt_envelope_failure"
    assert backend.invocations == []
    assert verdict.cost_usd == 0.0
    assert verdict.input_tokens == 0
    assert verdict.output_tokens == 0
    assert verdict.cache_creation_tokens == 0
    assert verdict.cache_read_tokens == 0
    assert verdict.execution_completeness is None
    assert json.loads(corpus.get_artifact(verdict.transcript_ref)) == []
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    effective = rubric_artifact["response_repair_policy"]
    assert effective["materialize_source_anchors"] is True
    assert effective["materialize_json_paths"] is False
    capabilities = rubric_artifact["response_repair_capabilities"]
    assert capabilities["source_anchor_catalog_available"] is True
    assert capabilities["source_anchor_count"] > 0
    degradation = rubric_artifact["response_repair_degradation"]
    assert degradation["reason"] == "required_anchor_render_limit_exceeded"
    assert degradation["requested_anchor_render_chars"] > 5_000
    assert degradation["effective_rendered_chars"] > 5_000
    assert degradation["anchor_render_limit_chars"] == 5_000
    assert degradation["requested_response_repair_policy"] == effective
    assert degradation["effective_response_repair_policy"] == effective
    rendered = json.loads(corpus.get_artifact(verdict.rendered_prompt_ref))
    assert rendered["response_repair_degradation"] == degradation
    assert "anchor:sha256:" in rendered["user"]
    assert rendered["response_repair_capabilities"] == capabilities


def test_required_anchor_catalog_must_be_nonempty_before_backend_invocation(
    tmp_path: Path,
):
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        require_source_anchors=True,
    )
    backend = StubBackend(json.dumps(_response()))
    corpus = Corpus(tmp_path)

    verdict = review(
        b"",
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        response_repair_policy=policy,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "review_anchor_catalog_failure"
    assert backend.invocations == []
    assert verdict.cache_creation_tokens == 0
    assert verdict.cache_read_tokens == 0
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    assert rubric_artifact["response_repair_capabilities"] == {
        "source_anchor_catalog_available": False,
        "source_anchor_count": 0,
    }


def test_required_anchors_reject_chunk_protocol_before_backend_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", 5_000)
    monkeypatch.setitem(review.__globals__, "_MAX_MAPPER_ENVELOPE_CHARS", 5_000)
    backend = StubBackend(json.dumps(_response()))

    with pytest.raises(
        ValueError,
        match="required source anchors are unavailable under the chunk",
    ):
        review(
            b"A" * 6_000,
            _make_rubric(),
            backend=backend,
            corpus=Corpus(tmp_path),
            response_repair_policy=ReviewResponseRepairPolicy(
                materialize_source_anchors=True,
                require_source_anchors=True,
            ),
        )

    assert backend.invocations == []


def test_effective_prompt_overflow_uses_chunk_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", 5_000)
    monkeypatch.setitem(review.__globals__, "_MAX_MAPPER_ENVELOPE_CHARS", 5_000)
    source = '"' * 3_000
    backend = ChunkingBackend()
    provider_calls: list[bool] = []
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        evidence_source_provider=lambda: provider_calls.append(True) or {},
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "pass"
    map_invocations = backend.invocations[:-1]
    assert len(map_invocations) >= 2
    assert all(
        len(invocation.system) + len(invocation.user) <= 5_000
        for invocation in map_invocations
    )
    assert "".join(
        json.loads(invocation.user)["chunk_text"]
        for invocation in map_invocations
    ) == source
    assert provider_calls == [True]
    assert verdict.cost_usd == 0.005 * len(backend.invocations)
    assert verdict.input_tokens == len(backend.invocations)
    assert verdict.output_tokens == 2 * len(backend.invocations)
    assert corpus.get_artifact(verdict.artifact_ref) == source.encode()
    rendered = json.loads(corpus.get_artifact(verdict.rendered_prompt_ref))
    assert rendered["protocol"] == "bounded_chunk_map_terminal_reduce"
    degradation = rendered["response_repair_degradation"]
    assert degradation["reason"] == "rendered_prompt_requires_chunk_protocol"
    assert degradation["effective_rendered_chars"] > 5_000
    assert degradation["anchor_render_limit_chars"] == 5_000
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    effective = rubric_artifact["response_repair_policy"]
    assert effective["materialize_source_anchors"] is False
    assert effective["materialize_json_paths"] is False
    assert rubric_artifact["response_repair_degradation"] == degradation
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["complete"] is True
    assert manifest["sources"][0]["source_chars"] == len(source)
    assert all(
        item["status"] == "reviewed"
        for item in manifest["sources"][0]["chunks"]
    )


def test_mapper_rendered_envelope_accepts_cap_and_rejects_cap_plus_one():
    sources = _review_sources(
        artifact_text="x",
        context=None,
        evidence_source_texts=None,
        required_source_ids=("artifact",),
    )
    rubric = _make_rubric(system_prompt="")
    seed = _source_chunks(sources, max_chars=1)[0]
    user_chars = len(_chunk_user_message(seed, sources, rubric))
    system_at_cap = "S" * (_MAX_MAPPER_ENVELOPE_CHARS - user_chars)

    chunks = _bounded_source_chunks(
        sources,
        rubric,
        system_message=system_at_cap,
        max_chunk_chars=1,
        envelope_chars=_MAX_MAPPER_ENVELOPE_CHARS,
    )
    assert len(chunks) == 1
    assert (
        len(system_at_cap) + len(_chunk_user_message(chunks[0], sources, rubric))
        == _MAX_MAPPER_ENVELOPE_CHARS
    )

    with pytest.raises(
        ValueError,
        match="mapper prompt envelope is irreducible",
    ):
        _bounded_source_chunks(
            sources,
            rubric,
            system_message=system_at_cap + "S",
            max_chunk_chars=1,
            envelope_chars=_MAX_MAPPER_ENVELOPE_CHARS,
        )


def test_irreducible_mapper_overhead_fails_before_backend_invocation(
    tmp_path: Path,
):
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        b"x",
        _make_rubric(system_prompt="R" * _MAX_ARTIFACT_CHARS),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "mapper_prompt_envelope_failure"
    assert verdict.findings[0].category == "review_prompt_envelope"
    assert backend.invocations == []
    assert verdict.cost_usd == 0.0
    assert verdict.input_tokens == 0
    assert verdict.output_tokens == 0
    assert verdict.execution_completeness is None
    assert verdict.coverage.sources_reviewed == ()
    assert json.loads(corpus.get_artifact(verdict.transcript_ref)) == []


def test_near_irreducible_mapper_overhead_over_chunk_cap_fails_before_call(
    tmp_path: Path,
):
    # Make exactly one retained character fit each mapper envelope.  The
    # source requires chunking by raw size, but partition construction reaches
    # the fixed mapper ceiling before it can retain the 257th range.
    artifact = "\x00" * (_MAX_ARTIFACT_CHARS + 1)
    sources = _review_sources(
        artifact_text=artifact,
        context=None,
        evidence_source_texts=None,
        required_source_ids=("artifact",),
    )
    source = sources[0]
    conservative_one_char = _SourceChunk(
        source_id=source.source_id,
        source_sha256=source.sha256,
        source_chars=len(source.text),
        index=len(source.text) - 1,
        count=len(source.text),
        char_start=_MAX_MAPPER_CHUNKS,
        char_end=_MAX_MAPPER_CHUNKS + 1,
        text=artifact[:1],
        sha256=_text_hash(artifact[:1]),
        evidence_addressable=source.evidence_addressable,
    )
    empty_rubric = _make_rubric(system_prompt="")
    fixed_mapper_chars = len(_CHUNK_SYSTEM_SUFFIX) + len(
        _chunk_user_message(conservative_one_char, sources, empty_rubric)
    )
    rubric = _make_rubric(
        system_prompt="R" * (
            _MAX_MAPPER_ENVELOPE_CHARS - fixed_mapper_chars
        )
    )
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        artifact.encode(),
        rubric,
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "mapper_chunk_limit_failure"
    assert verdict.findings[0].category == "review_resource_limit"
    assert f"more than {_MAX_MAPPER_CHUNKS} chunks" in verdict.reasoning
    assert "char_start=256" in verdict.reasoning
    assert backend.invocations == []
    assert verdict.cost_usd == 0.0
    assert verdict.input_tokens == 0
    assert verdict.output_tokens == 0
    assert verdict.execution_completeness is None
    assert verdict.coverage.sources_reviewed == ()
    assert json.loads(corpus.get_artifact(verdict.transcript_ref)) == []


def test_production_size_render_overflow_uses_two_bounded_maps(
    tmp_path: Path,
):
    source = "A" * 189_029
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        source.encode(),
        _make_rubric(
            system_prompt="R" * 10_768,
            user_template="Review:\n{artifact}",
        ),
        backend=backend,
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_json_paths=True,
        ),
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 3  # two maps and one terminal reducer
    map_requests = [
        json.loads(invocation.user) for invocation in backend.invocations[:2]
    ]
    assert [request["schema_version"] for request in map_requests] == [
        "assay.review.chunk.v1",
        "assay.review.chunk.v1",
    ]
    assert [len(request["chunk_text"]) for request in map_requests] == [
        160_000,
        29_029,
    ]
    assert "".join(request["chunk_text"] for request in map_requests) == source
    assert all(
        len(invocation.system) + len(invocation.user)
        <= _MAX_MAPPER_ENVELOPE_CHARS
        for invocation in backend.invocations[:2]
    )
    rendered = json.loads(corpus.get_artifact(verdict.rendered_prompt_ref))
    degradation = rendered["response_repair_degradation"]
    assert degradation["reason"] == "rendered_prompt_requires_chunk_protocol"
    assert degradation["effective_rendered_chars"] > _MAX_ARTIFACT_CHARS


def test_decision_derivation_repairs_conflict_then_runs_unchanged_validators(
    tmp_path: Path,
):
    payload = _response(
        decision="pass",
        reasoning="The criterion is violated by the cited blocker.",
        findings=[_finding(criterion_id="criterion.one")],
        criterion_results=[_criterion_result(outcome="violated")],
    )
    corpus = Corpus(tmp_path / "valid")
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(required_criteria=("criterion.one",)),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            derive_decision_from_criterion_results=True,
        ),
    )
    assert verdict.decision == "revise"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][-1] == {
        "operation": "derive_decision_from_criterion_results",
        "path": "decision",
        "before": "pass",
        "after": "revise",
        "basis": "violated_criterion",
    }

    conservative = _response(
        decision="inconclusive",
        criterion_results=[_criterion_result(outcome="satisfied")],
    )
    conservative_corpus = Corpus(tmp_path / "conservative")
    failed = review(
        b"artifact with needle in it",
        _make_rubric(required_criteria=("criterion.one",)),
        backend=StubBackend(json.dumps(conservative)),
        corpus=conservative_corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            derive_decision_from_criterion_results=True,
        ),
    )
    assert failed.decision == "inconclusive"
    assert failed.findings[0].id == "schema_failure"
    assert "inconclusive decision requires an inconclusive result" in (
        failed.reasoning
    )
    assert "review_response_repair" not in failed.artifacts
    assert "review_raw_response" not in failed.artifacts


def test_noop_decision_derivation_produces_no_report(tmp_path: Path):
    payload = _response(
        decision="revise",
        findings=[_finding(criterion_id="criterion.one")],
        criterion_results=[_criterion_result(outcome="violated")],
    )
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(required_criteria=("criterion.one",)),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            derive_decision_from_criterion_results=True,
        ),
    )
    assert verdict.decision == "revise"
    assert "review_response_repair" not in verdict.artifacts


def test_bounded_fully_satisfied_revise_to_pass_is_separately_audited(
    tmp_path: Path,
):
    payload = _response(
        decision="revise",
        reasoning="All declared criteria and current findings are closed.",
        findings=[_finding(
            criterion_id="criterion.one",
            disposition="addressed",
        )],
        criterion_results=[_criterion_result(outcome="satisfied")],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(required_criteria=("criterion.one",)),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            derive_decision_from_criterion_results=True,
            recover_fully_satisfied_revise_to_pass=True,
        ),
    )

    assert verdict.decision == "pass"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["decision_repair_mode"] == (
        "bounded_fully_satisfied_revise_to_pass"
    )
    assert report["actions"][-1] == {
        "operation": "recover_fully_satisfied_revise_to_pass",
        "path": "decision",
        "before": "revise",
        "after": "pass",
        "basis": "criteria_satisfied",
    }


@pytest.mark.parametrize(
    ("coverage_complete", "omissions", "disposition", "expected_error"),
    (
        (
            False,
            [],
            "addressed",
            "complete=false requires at least one omission",
        ),
        (
            True,
            ["missing input"],
            "addressed",
            "complete=true is incompatible with non-empty omissions",
        ),
        (True, [], "confirmed", "revise requires a violated result"),
    ),
)
def test_bounded_revise_to_pass_refuses_unproven_closure(
    tmp_path: Path,
    coverage_complete: bool,
    omissions: list[str],
    disposition: str,
    expected_error: str,
):
    payload = _response(
        decision="revise",
        findings=[_finding(
            criterion_id="criterion.one",
            disposition=disposition,
        )],
        criterion_results=[_criterion_result(outcome="satisfied")],
    )
    payload["coverage"]["complete"] = coverage_complete
    payload["coverage"]["omissions"] = omissions
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(required_criteria=("criterion.one",)),
        backend=StubBackend(json.dumps(payload)),
        corpus=Corpus(tmp_path),
        response_repair_policy=ReviewResponseRepairPolicy(
            derive_decision_from_criterion_results=True,
            recover_fully_satisfied_revise_to_pass=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"
    assert expected_error in verdict.reasoning


def test_schema_failure_preserves_independently_valid_actionable_finding(
    tmp_path: Path,
):
    corpus = Corpus(tmp_path)
    payload = _response(
        decision="revise",
        reasoning="The grounded issue requires revision.",
        findings=[
            _finding(id="grounded-blocker"),
            _finding(
                id="unrelated-bad-citation",
                severity="Medium",
                evidence=[{
                    "source_id": "artifact",
                    "kind": "source_quote",
                    "location": "line 99",
                    "excerpt": "fabricated unrelated text",
                }],
            ),
        ],
    )

    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "schema_failure",
        "grounded-blocker",
    ]
    assert verdict.findings[1].source == "llm"
    partial = json.loads(corpus.get_artifact(
        verdict.artifacts["review_partial_response"]
    ))
    assert partial["schema_version"] == "assay.review-partial-response.v1"
    assert partial["claimed_decision"] == "revise"
    assert partial["valid_findings"] == [{
        "index": 0,
        "finding_id": "grounded-blocker",
    }]
    assert partial["invalid_findings"][0]["index"] == 1
    assert partial["invalid_findings"][0]["finding_id"] == (
        "unrelated-bad-citation"
    )
    assert "not a verbatim substring" in partial["invalid_findings"][0]["error"]
    assert verdict.input_tokens == 1
    assert verdict.output_tokens == 2


def test_partial_recovery_fails_closed_without_valid_actionable_finding(
    tmp_path: Path,
):
    payload = _response(
        decision="revise",
        reasoning="An ungrounded issue was attempted.",
        findings=[_finding(evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "line 99",
            "excerpt": "fabricated unrelated text",
        }])],
    )

    verdict = _run(tmp_path, payload)

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == ["schema_failure"]
    assert "review_partial_response" not in verdict.artifacts


@pytest.mark.parametrize(
    "field",
    [
        "id", "criterion_id", "severity", "confidence", "disposition",
        "category", "targets", "evidence", "proposal", "attack_spec",
    ],
)
def test_every_finding_field_is_required(tmp_path: Path, field: str):
    finding = _finding()
    del finding[field]
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[finding]),
    )
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"


@pytest.mark.parametrize("severity", ["critical", "Blocker", None, 2, []])
def test_finding_severity_is_exact_enum(tmp_path: Path, severity):
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(severity=severity)]),
    )
    assert verdict.decision == "inconclusive"


def test_non_scalar_decision_fails_closed(tmp_path: Path):
    verdict = _run(tmp_path, _response(decision=[]))
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"


@pytest.mark.parametrize("confidence", [-0.01, 1.01, True, "0.9", None])
def test_confidence_is_numeric_unit_interval(tmp_path: Path, confidence):
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(confidence=confidence)]),
    )
    assert verdict.decision == "inconclusive"


def test_duplicate_finding_ids_fail_schema(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(), _finding()]),
    )
    assert verdict.decision == "inconclusive"
    assert "duplicate" in verdict.reasoning


def test_duplicate_finding_targets_fail_schema(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(
            decision="revise",
            findings=[_finding(targets=["prior:f1", "prior:f1"])],
        ),
    )
    assert verdict.decision == "inconclusive"
    assert "findings[0].targets[1]: duplicate" in verdict.reasoning


def test_duplicate_coverage_sources_fail_schema(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(
            coverage={
                "complete": True,
                "sources_reviewed": ["artifact", "artifact"],
                "omissions": [],
            }
        ),
    )
    assert verdict.decision == "inconclusive"
    assert "coverage.sources_reviewed[1]: duplicate" in verdict.reasoning


def test_duplicate_coverage_omissions_fail_schema(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(
            coverage={
                "complete": False,
                "sources_reviewed": ["artifact"],
                "omissions": ["middle unavailable", "middle unavailable"],
            }
        ),
    )
    assert verdict.decision == "inconclusive"
    assert "coverage.omissions[1]: duplicate" in verdict.reasoning


def test_finding_requires_nonempty_evidence(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(evidence=[])]),
    )
    assert verdict.decision == "inconclusive"
    assert "requires at least one" in verdict.reasoning


def test_artifact_evidence_must_be_verbatim(tmp_path: Path):
    evidence = [{
        "source_id": "artifact", "kind": "source_quote",
        "location": "line 1", "excerpt": "hallucinated",
    }]
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(evidence=evidence)]),
    )
    assert verdict.decision == "inconclusive"
    assert "not a verbatim substring" in verdict.reasoning


def test_context_evidence_is_grounded(tmp_path: Path):
    evidence = [{
        "source_id": "context:spec_intent", "kind": "intent_quote",
        "location": "intent", "excerpt": "must stay stable",
    }]
    payload = _response(
        decision="revise",
        coverage={
            "complete": True,
            "sources_reviewed": ["artifact", "context:spec_intent"],
            "omissions": [],
        },
        findings=[_finding(evidence=evidence)],
    )
    verdict = _run(
        tmp_path,
        payload,
        context={"spec_intent": "The API must stay stable."},
    )
    assert verdict.decision == "revise"


def test_unknown_or_undeclared_evidence_source_fails(tmp_path: Path):
    evidence = [{
        "source_id": "web:random", "kind": "quote",
        "location": "page", "excerpt": "needle",
    }]
    payload = _response(
        decision="revise",
        coverage={
            "complete": True,
            "sources_reviewed": ["artifact", "web:random"],
            "omissions": [],
        },
        findings=[_finding(evidence=evidence)],
    )
    assert _run(tmp_path, payload).decision == "inconclusive"


def test_generated_text_cannot_masquerade_as_evidence(tmp_path: Path):
    evidence = [{
        "source_id": "artifact", "kind": "test_code",
        "location": "test", "excerpt": "needle",
    }]
    verdict = _run(
        tmp_path,
        _response(decision="revise", findings=[_finding(evidence=evidence)]),
    )
    assert verdict.decision == "inconclusive"
    assert "generated material" in verdict.reasoning


def test_attack_spec_is_typed_declarative_data(tmp_path: Path):
    attack = {
        "setup": "Create an empty input.",
        "stimulus": "Call the public operation.",
        "oracle": "It returns the documented sentinel.",
    }
    verdict = _run(
        tmp_path,
        _response(
            decision="revise",
            findings=[_finding(attack_spec=attack, proposal=None)],
        ),
    )
    assert verdict.decision == "revise"
    assert verdict.findings[0].attack_spec.oracle.startswith("It returns")


# Decision and coverage invariants -------------------------------------------


def test_pass_cannot_hide_blocking_active_finding(tmp_path: Path):
    verdict = _run(tmp_path, _response(findings=[_finding()]))
    assert verdict.decision == "inconclusive"
    assert "pass is incompatible" in verdict.reasoning


def test_pass_can_retain_addressed_high_finding(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(findings=[_finding(disposition="addressed")]),
    )
    assert verdict.decision == "pass"


def test_revise_requires_blocking_finding(tmp_path: Path):
    verdict = _run(tmp_path, _response(decision="revise"))
    assert verdict.decision == "inconclusive"
    assert "requires an active finding" in verdict.reasoning


def test_reject_requires_confirmed_or_unfixable_blocker(tmp_path: Path):
    suspected = _finding(disposition="suspected")
    verdict = _run(
        tmp_path,
        _response(decision="reject", findings=[suspected]),
    )
    assert verdict.decision == "inconclusive"
    assert "confirmed or unfixable" in verdict.reasoning


def test_disallowed_rubric_decision_fails_schema(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(decision="inconclusive"),
        rubric={"allowed_decisions": ("pass", "revise")},
    )
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"


def test_incomplete_coverage_cannot_pass(tmp_path: Path):
    payload = _response(
        coverage={
            "complete": False,
            "sources_reviewed": ["artifact"],
            "omissions": ["middle section unavailable"],
        }
    )
    verdict = _run(tmp_path, payload)
    assert verdict.decision == "inconclusive"
    assert verdict.passed is False


def test_complete_coverage_cannot_claim_omissions(tmp_path: Path):
    payload = _response(
        coverage={
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": ["contradiction"],
        }
    )
    assert _run(tmp_path, payload).decision == "inconclusive"


def test_oversized_artifact_is_fully_reviewed_by_chunk_protocol(tmp_path: Path):
    artifact = ("A" * (_MAX_ARTIFACT_CHARS + 1)).encode()
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        artifact,
        _make_rubric(),
        backend=backend,
        corpus=corpus,
    )
    assert verdict.decision == "pass"
    assert verdict.coverage.complete is True
    assert verdict.coverage.artifact_chars_total == _MAX_ARTIFACT_CHARS + 1
    assert verdict.coverage.artifact_chars_reviewed == _MAX_ARTIFACT_CHARS + 1
    assert len(backend.invocations) == 3  # two maps and one terminal reducer
    assert verdict.input_tokens == 3
    assert verdict.output_tokens == 6
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["chunk_chars"] == 160_000
    assert manifest["complete"] is True
    assert manifest["sources"][0]["source_chars"] == len(artifact)
    assert [item["status"] for item in manifest["sources"][0]["chunks"]] == [
        "reviewed",
        "reviewed",
    ]


def test_oversized_required_context_is_fully_chunked_and_counted(
    tmp_path: Path,
):
    context = "context needle" + "X" * _MAX_ARTIFACT_CHARS
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(user_template="{artifact}\n{spec}"),
        backend=backend,
        corpus=corpus,
        context={"spec": context},
        required_source_ids=("artifact", "context:spec"),
    )
    assert verdict.decision == "pass"
    assert verdict.coverage.complete is True
    assert verdict.coverage.sources_reviewed == ("artifact", "context:spec")
    assert len(backend.invocations) == 4  # artifact + two context maps + reducer
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    by_id = {item["source_id"]: item for item in manifest["sources"]}
    assert by_id["artifact"]["source_chars"] == len("artifact with needle in it")
    assert by_id["context:spec"]["source_chars"] == len(context)
    assert len(by_id["context:spec"]["chunks"]) == 2


def test_single_prompt_required_sources_can_be_narrower_than_terminal_set(
    tmp_path: Path,
):
    backend = StubBackend(json.dumps(_response()))
    corpus = Corpus(tmp_path)

    verdict = review(
        b"artifact with needle in it",
        _make_rubric(user_template="{artifact}\n{spec}"),
        backend=backend,
        corpus=corpus,
        context={"spec": "required context"},
        required_source_ids=("artifact", "context:spec"),
        single_prompt_required_source_ids=("artifact",),
    )

    assert verdict.decision == "pass"
    assert verdict.coverage.sources_reviewed == ("artifact",)
    assert len(backend.invocations) == 1
    assert "review_source_manifest" not in verdict.artifacts


def test_render_overflow_keeps_full_terminal_required_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", 5_000)
    monkeypatch.setitem(
        review.__globals__, "_MAX_MAPPER_ENVELOPE_CHARS", 5_000
    )
    artifact = '"' * 1_500
    context = '"' * 1_200
    assert len(artifact) + len(context) < 5_000
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        artifact.encode(),
        _make_rubric(user_template="{artifact}\n{spec}"),
        backend=backend,
        corpus=corpus,
        context={"spec": context},
        required_source_ids=("artifact", "context:spec"),
        single_prompt_required_source_ids=("artifact",),
    )

    assert verdict.decision == "pass"
    assert verdict.coverage.sources_reviewed == ("artifact", "context:spec")
    terminal_request = json.loads(backend.invocations[-1].user)
    assert terminal_request["schema_version"] == "assay.review.reduce.v1"
    assert terminal_request["required_sources"] == [
        "artifact",
        "context:spec",
    ]
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert {
        source["source_id"]: source["terminal_required"]
        for source in manifest["sources"]
    } == {"artifact": True, "context:spec": True}


@pytest.mark.parametrize(
    ("single_prompt_required_source_ids", "error_type", "message"),
    [
        (["artifact"], TypeError, "must be a tuple"),
        ((), ValueError, "must not be empty"),
        (("context:spec",), ValueError, "must contain 'artifact'"),
        (("artifact", "artifact"), ValueError, "must be unique"),
        (
            ("artifact", "context:other"),
            ValueError,
            "must be a subset of required_source_ids",
        ),
    ],
)
def test_invalid_single_prompt_required_sources_fail_before_transport(
    tmp_path: Path,
    single_prompt_required_source_ids,
    error_type: type[Exception],
    message: str,
):
    backend = StubBackend(json.dumps(_response()))

    with pytest.raises(error_type, match=message):
        review(
            b"artifact with needle in it",
            _make_rubric(user_template="{artifact}\n{spec}"),
            backend=backend,
            corpus=Corpus(tmp_path),
            context={"spec": "required context"},
            required_source_ids=("artifact", "context:spec"),
            single_prompt_required_source_ids=(
                single_prompt_required_source_ids
            ),
        )

    assert backend.invocations == []


def test_adaptive_mapper_preserves_multiple_escaping_heavy_sources(
    tmp_path: Path,
):
    artifact = '"\x00' * 30_000
    context = "\\\n" * 30_000
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        artifact.encode(),
        _make_rubric(user_template="{artifact}\n{spec}"),
        backend=backend,
        corpus=corpus,
        context={"spec": context},
        required_source_ids=("artifact", "context:spec"),
    )

    assert verdict.decision == "pass"
    map_requests = [
        json.loads(invocation.user)
        for invocation in backend.invocations
        if json.loads(invocation.user)["schema_version"]
        == "assay.review.chunk.v1"
    ]
    assert len(map_requests) >= 2
    assert all(
        len(invocation.system) + len(invocation.user)
        <= _MAX_MAPPER_ENVELOPE_CHARS
        for invocation in backend.invocations[:-1]
    )
    expected = {"artifact": artifact, "context:spec": context}
    reconstructed = {
        source_id: "".join(
            request["chunk_text"]
            for request in map_requests
            if request["source_id"] == source_id
        )
        for source_id in expected
    }
    assert reconstructed == expected
    for source_id in expected:
        source_requests = [
            request
            for request in map_requests
            if request["source_id"] == source_id
        ]
        assert [request["chunk_index"] for request in source_requests] == list(
            range(len(source_requests))
        )
        assert all(
            request["chunk_count"] == len(source_requests)
            for request in source_requests
        )
        assert [
            (request["char_start"], request["char_end"])
            for request in source_requests
        ] == [
            (
                sum(len(item["chunk_text"]) for item in source_requests[:index]),
                sum(
                    len(item["chunk_text"])
                    for item in source_requests[: index + 1]
                ),
            )
            for index in range(len(source_requests))
        ]
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["mapper_envelope_chars"] == _MAX_MAPPER_ENVELOPE_CHARS
    assert manifest["mapper_chunk_limit"] == _MAX_MAPPER_CHUNKS
    assert {
        item["source_id"]: item["source_chars"] for item in manifest["sources"]
    } == {source_id: len(text) for source_id, text in expected.items()}


def test_required_source_must_be_declared_in_coverage(tmp_path: Path):
    verdict = _run(
        tmp_path,
        _response(),
        required_source_ids=("artifact", "context:spec"),
        context={"spec": "required context"},
        rubric={"user_template": "{artifact}\n{spec}"},
    )
    assert verdict.decision == "inconclusive"
    assert "missing required sources" in verdict.reasoning


def test_prior_evidence_requires_exact_retained_source_text(tmp_path: Path):
    finding = _finding(
        severity="Low",
        disposition="addressed",
        evidence=[{
            "source_id": "prior:red:f1",
            "kind": "source_quote",
            "location": "prior finding",
            "excerpt": "retained observation",
        }],
    )
    payload = _response(
        coverage={
            "complete": True,
            "sources_reviewed": ["artifact", "prior:red:f1"],
            "omissions": [],
        },
        findings=[finding],
    )
    assert _run(
        tmp_path / "valid",
        payload,
        evidence_source_texts={"prior:red:f1": "retained observation"},
        required_source_ids=("artifact", "prior:red:f1"),
    ).decision == "pass"
    forged = _run(
        tmp_path / "forged",
        payload,
        evidence_source_texts={"prior:red:f1": "different retained text"},
        required_source_ids=("artifact", "prior:red:f1"),
    )
    assert forged.decision == "inconclusive"
    assert "not a verbatim substring" in forged.reasoning


def test_tool_evidence_requires_post_invocation_registry_entry(tmp_path: Path):
    finding = _finding(
        severity="Low",
        disposition="addressed",
        evidence=[{
            "source_id": "tool:trace-0001",
            "kind": "tool_observation",
            "location": "call 1",
            "excerpt": "measured result",
        }],
    )
    payload = _response(
        coverage={
            "complete": True,
            "sources_reviewed": ["artifact", "tool:trace-0001"],
            "omissions": [],
        },
        findings=[finding],
    )
    valid = _run(
        tmp_path / "valid",
        payload,
        evidence_source_provider=lambda: {
            "tool:trace-0001": "measured result",
        },
    )
    assert valid.decision == "pass"
    forged = _run(
        tmp_path / "forged",
        payload,
        evidence_source_provider=lambda: {},
    )
    assert forged.decision == "inconclusive"
    assert "no retained source text" in forged.reasoning


def test_elision_marker_is_visible_and_tiny_cap_does_not_leak():
    out = _elide_artifact("ABCDEFGHIJ" * 10, max_chars=1)
    assert "coverage is incomplete" in out
    assert "ABCDEFGHIJ" not in out
    assert out.startswith("A") and out.endswith("J")


def test_passes_ignores_resolved_findings():
    payload = _response(findings=[_finding(disposition="dismissed")])
    findings = _parse_findings(
        payload,
        artifact_text="needle",
        context=None,
        sources_reviewed=("artifact",),
    )
    assert _passes(findings, "High") is True
    findings[0] = _parse_findings(
        _response(findings=[_finding(disposition="confirmed")]),
        artifact_text="needle",
        context=None,
        sources_reviewed=("artifact",),
    )[0]
    assert _passes(findings, "High") is False


# Failure persistence and metadata -------------------------------------------


def test_parse_failure_is_inconclusive(tmp_path: Path):
    verdict = _run(tmp_path, "prose before {\"decision\": \"pass\"}")
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "parse_failure"
    assert verdict.coverage.complete is False


def test_backend_error_is_inconclusive(tmp_path: Path):
    corpus = Corpus(tmp_path)
    verdict = review(
        b"needle",
        _make_rubric(),
        backend=StubBackend(
            "", error=TransportError(http_status=500, message="boom")
        ),
        corpus=corpus,
    )
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "backend_transport"
    assert "backend error" in verdict.reasoning


def test_review_persists_transcript_artifact_and_commit(tmp_path: Path):
    corpus = Corpus(tmp_path)
    verdict = review(
        b"needle",
        _make_rubric(),
        backend=StubBackend(json.dumps(_response())),
        corpus=corpus,
        artifact_commit="deadbeef",
    )
    assert corpus.get_artifact(verdict.artifact_ref) == b"needle"
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript[-1]["role"] == "assistant"
    assert verdict.artifact_commit == "deadbeef"


def test_review_persists_semantic_success_accounting_debt(tmp_path: Path):
    completeness = ExecutionCompleteness(
        token_usage_complete=False,
        cost_complete=False,
        terminal_complete=True,
        trace_complete=True,
    )
    trace = BackendExecutionTrace(
        backend="stub",
        requested_model="stub-model",
        events=(
            BackendExecutionEvent(
                sequence=0,
                kind="terminal",
                name="end_turn",
            ),
        ),
    )
    corpus = Corpus(tmp_path)

    verdict = review(
        b"needle",
        _make_rubric(),
        backend=StubBackend(
            json.dumps(_response()),
            completeness=completeness,
            execution_trace=trace,
        ),
        corpus=corpus,
    )

    assert verdict.decision == "pass"
    assert verdict.execution_completeness == completeness
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    ref = corpus.put_record(verdict, kind="verdicts")
    from assay import Verdict

    restored = corpus.get_record(ref, as_type=Verdict)
    assert restored.execution_completeness == completeness
    assert restored.accounting_debt == verdict.accounting_debt


def test_single_review_propagates_pricing_evidence(tmp_path: Path):
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(1, 2, 0, 1, 0.005),),
        complete=True,
    )

    class _PricingBackend(StubBackend):
        def invoke(self, invocation):
            result = super().invoke(invocation)
            return dataclasses.replace(
                result,
                usage=Usage(
                    input_tokens=1,
                    output_tokens=2,
                    cost_usd=0.005,
                    cache_read_tokens=1,
                ),
                completeness=ExecutionCompleteness(
                    token_usage_complete=True,
                    cost_complete=True,
                ),
                pricing_evidence=evidence,
            )

    verdict = review(
        b"needle",
        _make_rubric(),
        backend=_PricingBackend(json.dumps(_response())),
        corpus=Corpus(tmp_path),
    )

    assert verdict.decision == "pass"
    assert verdict.cache_creation_tokens == 0
    assert verdict.cache_read_tokens == 1
    assert verdict.pricing_evidence == evidence


def test_review_v2_verdict_record_round_trips_nested_types(tmp_path: Path):
    corpus = Corpus(tmp_path)
    payload = _response(
        decision="revise",
        findings=[_finding(attack_spec={
            "setup": "Prepare the edge case.",
            "stimulus": "Invoke the public API.",
            "oracle": "Observe the documented result.",
        })],
    )
    verdict = review(
        b"needle",
        _make_rubric(),
        backend=StubBackend(json.dumps(payload)),
        corpus=corpus,
    )
    ref = corpus.put_record(verdict, kind="verdicts")
    from assay import Evidence, Finding, ReviewCoverage, Verdict

    restored = corpus.get_record(ref, as_type=Verdict)
    assert isinstance(restored.coverage, ReviewCoverage)
    assert isinstance(restored.findings[0], Finding)
    assert isinstance(restored.findings[0].evidence[0], Evidence)
    assert restored.findings[0].proposal.action == "remove it"
    assert restored.findings[0].attack_spec.oracle.startswith("Observe")


def test_review_rejects_unknown_template_placeholder(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown placeholder"):
        review(
            b"needle",
            _make_rubric(user_template="{artifact} {missing}"),
            backend=StubBackend(json.dumps(_response())),
            corpus=Corpus(tmp_path),
        )


@pytest.mark.parametrize(
    "raw, expected",
    [("high", "High"), ("NOPE", "Critical"), (None, "Critical")],
)
def test_internal_severity_normalizer_remains_fail_closed(raw, expected):
    assert _normalize_severity(raw) == expected
