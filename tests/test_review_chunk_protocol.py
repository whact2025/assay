# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed properties of bounded source map/reduce review."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
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
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.primitives.review import (
    _MAX_ARTIFACT_CHARS,
    _MAX_REVIEW_CHUNK_CHARS,
    _chunk_response_schema,
    _intermediate_response_schema,
    _review_response_schema,
    ReviewResponseRepairPolicy,
    review,
)

from .test_review import ChunkingBackend, _finding, _make_rubric, _response

review_module = importlib.import_module("assay.primitives.review")

_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)


class _ObservationBackend:
    name = "observation-stub"
    model = "observation-stub-model"

    def __init__(self, severity: str):
        self.severity = severity
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        request = json.loads(invocation.user)
        if request["schema_version"] == "assay.review.chunk.v1":
            findings = []
            if request["source_id"] == "artifact" and request["chunk_index"] == 0:
                findings = [
                    _finding(
                        id="local_observation",
                        severity=self.severity,
                        disposition="confirmed",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "A",
                        }],
                    )
                ]
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
            payload.update({"summary": "Mapped exact range.", "findings": findings})
        else:
            targets = request["required_targets"]
            payload = _response(
                coverage={
                    "complete": True,
                    "sources_reviewed": request["required_sources"],
                    "omissions": [],
                },
                findings=[
                    _finding(
                        id="terminal_headline",
                        severity="Low",
                        disposition="addressed",
                        targets=targets,
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "mapped source",
                            "excerpt": "A",
                        }],
                    )
                ],
            )
        final = json.dumps(payload)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(cost_usd=0.001),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
            ),
        )


class _HierarchicalBackend:
    name = "hierarchical-stub"
    model = "hierarchical-stub-model"

    def __init__(self, *, corrupt_first_intermediate: bool = False):
        self.corrupt_first_intermediate = corrupt_first_intermediate
        self.corrupted = False
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        request = json.loads(invocation.user)
        schema = request["schema_version"]
        if schema == "assay.review.chunk.v1":
            finding = _finding(
                id=f"local_{request['chunk_index']}",
                severity="Low",
                disposition="confirmed",
                evidence=[{
                    "source_id": "artifact",
                    "kind": "source_quote",
                    "location": f"chunk {request['chunk_index']}",
                    "excerpt": "A",
                }],
            )
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
            payload.update({"summary": "This range contains A.", "findings": [finding]})
        elif schema == "assay.review.reduce-map.v1":
            targets = request["required_targets"]
            findings = []
            if targets:
                findings = [
                    _finding(
                        id=f"digest_{request['depth']}",
                        severity="Low",
                        disposition="confirmed",
                        targets=targets,
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "child reports",
                            "excerpt": "A",
                        }],
                    )
                ]
            payload = {
                "schema_version": "assay.review.reduce-map.v1",
                "batch_id": request["batch_id"],
                "depth": request["depth"],
                "input_report_ids": request["input_report_ids"],
                "input_reports_sha256": request["input_reports_sha256"],
                "summary": "All child reports retain the observed A ranges.",
                "findings": findings,
            }
            if self.corrupt_first_intermediate and not self.corrupted:
                payload["input_reports_sha256"] = "sha256:" + "0" * 64
                self.corrupted = True
        else:
            assert schema == "assay.review.reduce.v1"
            payload = _response(
                coverage={
                    "complete": True,
                    "sources_reviewed": request["required_sources"],
                    "omissions": [],
                },
                findings=[
                    _finding(
                        id="terminal_headline",
                        severity="Low",
                        disposition="addressed",
                        targets=request["required_targets"],
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "reduced observations",
                            "excerpt": "A",
                        }],
                    )
                ],
            )
        final = json.dumps(payload)
        return InvocationResult(
            final_message=final,
            transcript=(Message(role="assistant", content=final),),
            usage=Usage(cost_usd=0.00001),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
            ),
        )


class _SchemaConstrainedObservationBackend(_ObservationBackend):
    """Stub that derives map targets from the advertised response schema."""

    def invoke(self, invocation: Invocation) -> InvocationResult:
        result = super().invoke(invocation)
        request = json.loads(invocation.user)
        if request["schema_version"] != "assay.review.chunk.v1":
            return result
        assert invocation.response_schema is not None
        target_schema = invocation.response_schema["properties"]["findings"][
            "items"
        ]["properties"]["targets"]
        disposition_schema = invocation.response_schema["properties"]["findings"][
            "items"
        ]["properties"]["disposition"]
        payload = json.loads(result.final_message)
        for finding in payload["findings"]:
            finding["targets"] = (
                [] if target_schema.get("maxItems") == 0 else ["schema:allowed"]
            )
            finding["disposition"] = (
                "confirmed"
                if disposition_schema["enum"]
                == ["confirmed", "suspected", "unfixable"]
                else "dismissed"
            )
        final = json.dumps(payload)
        return dataclasses.replace(result, final_message=final)


def _oversized_artifact() -> bytes:
    return ("A" * (_MAX_ARTIFACT_CHARS + 1)).encode()


def _map_context_record(
    source_id: str,
    text: str,
    *,
    include_text: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "source_id": source_id,
        "source_sha256": (
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        ),
        "source_chars": len(text),
    }
    if include_text:
        record["text"] = text
    return record


def test_only_chunk_finding_schema_specializes_map_contract():
    chunk_finding = _chunk_response_schema()["properties"]["findings"]["items"]
    review_finding = _review_response_schema(_make_rubric())["properties"][
        "findings"
    ]["items"]
    intermediate_finding = _intermediate_response_schema()["properties"][
        "findings"
    ]["items"]

    chunk_targets = chunk_finding["properties"]["targets"]
    review_targets = review_finding["properties"]["targets"]
    intermediate_targets = intermediate_finding["properties"]["targets"]

    assert chunk_targets["maxItems"] == 0
    assert chunk_targets["uniqueItems"] is True
    assert chunk_finding["properties"]["disposition"]["enum"] == [
        "confirmed",
        "suspected",
        "unfixable",
    ]
    assert "maxItems" not in review_targets
    assert review_targets["uniqueItems"] is True
    assert "dismissed" in review_finding["properties"]["disposition"]["enum"]
    assert "addressed" in review_finding["properties"]["disposition"]["enum"]
    assert "maxItems" not in intermediate_targets
    assert intermediate_targets["uniqueItems"] is True
    assert "dismissed" in intermediate_finding["properties"]["disposition"]["enum"]
    assert "addressed" in intermediate_finding["properties"]["disposition"]["enum"]


def test_schema_constrained_compliant_mapper_passes(tmp_path: Path):
    backend = _SchemaConstrainedObservationBackend("Low")

    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    assert verdict.decision == "pass"
    assert verdict.findings[0].targets[0].startswith("map:")


class _AliasedPriorBackend(ChunkingBackend):
    """Emit the compact prior target advertised by terminal reduction."""

    def __init__(self, *, duplicate_full_target: bool = False):
        super().__init__()
        self.duplicate_full_target = duplicate_full_target
        self.emitted_targets: list[str] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        request = json.loads(invocation.user)
        if request["schema_version"] != "assay.review.reduce.v1":
            return super().invoke(invocation)
        self.invocations.append(invocation)
        (target,) = request["required_prior_targets"]
        (alias_record,) = request["prior_target_aliases"]
        (source_record,) = request["prior_target_sources"]
        assert alias_record["finding_ref"] == target
        assert source_record["finding_ref"] == target
        targets = [alias_record["alias"]]
        if self.duplicate_full_target:
            targets.append(target)
        self.emitted_targets = targets
        payload = _response(
            coverage={
                "complete": True,
                "sources_reviewed": request["required_sources"],
                "omissions": [],
            },
            findings=[_finding(
                id="prior_adjudication",
                severity="Low",
                disposition="addressed",
                targets=targets,
                evidence=[{
                    "source_id": source_record["source_id"],
                    "kind": "source_quote",
                    "location": "retained history",
                    "excerpt": "retained prior observation",
                }],
            )],
        )
        final = json.dumps(payload)
        return InvocationResult(
            final_message=final,
            transcript=(Message(role="assistant", content=final),),
            usage=Usage(cost_usd=0.00001),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
            ),
        )


def test_chunk_reducer_expands_alias_bound_to_distinct_prior_source(
    tmp_path: Path,
):
    target = "prior:history:0:sha256-deadbeef:finding-1"
    source_id = "context:history"
    alias = "harness-obligation:prior-deadbeef"
    backend = _AliasedPriorBackend()

    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
        evidence_source_texts={source_id: "retained prior observation"},
        required_source_ids=("artifact", source_id),
        prior_target_source_ids={target: source_id},
        prior_target_aliases={alias: target},
    )

    assert verdict.decision == "pass"
    assert backend.emitted_targets == [alias]
    assert verdict.findings[0].targets == (target,)
    assert verdict.findings[0].evidence[0].source_id == source_id


def test_chunk_reducer_rejects_alias_and_full_target_double_adjudication(
    tmp_path: Path,
):
    target = "prior:history:0:sha256-deadbeef:finding-1"
    source_id = "context:history"
    alias = "harness-obligation:prior-deadbeef"

    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=_AliasedPriorBackend(duplicate_full_target=True),
        corpus=Corpus(tmp_path),
        evidence_source_texts={source_id: "retained prior observation"},
        required_source_ids=("artifact", source_id),
        prior_target_source_ids={target: source_id},
        prior_target_aliases={alias: target},
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reducer_schema_failure"
    assert "exactly once" in verdict.reasoning


@pytest.mark.parametrize(
    ("target_sources", "aliases", "match"),
    [
        (
            {"not-a-prior": "context:history"},
            None,
            "keys must use the 'prior:' namespace",
        ),
        (
            {"prior:f1": "context:missing"},
            None,
            "must bind a required evidence source",
        ),
        (
            {"prior:f1": "context:history"},
            {"short": "prior:unknown"},
            "must bind a required prior target",
        ),
        (
            {
                "prior:f1": "context:history",
                "prior:f2": "context:history",
            },
            {"prior:f2": "prior:f1"},
            "collides with retained authority",
        ),
        (
            {"prior:f1": "context:history"},
            {"short-1": "prior:f1", "short-2": "prior:f1"},
            "values must be unique",
        ),
    ],
)
def test_prior_target_bindings_fail_before_transport(
    tmp_path: Path,
    target_sources,
    aliases,
    match: str,
):
    backend = ChunkingBackend()

    with pytest.raises(ValueError, match=match):
        review(
            _oversized_artifact(),
            _make_rubric(),
            backend=backend,
            corpus=Corpus(tmp_path),
            evidence_source_texts={
                "context:history": "retained prior observation"
            },
            required_source_ids=("artifact", "context:history"),
            prior_target_source_ids=target_sources,
            prior_target_aliases=aliases,
        )

    assert backend.invocations == []


def test_review_chunk_chars_controls_ranges_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_chars=4,
    )

    assert verdict.decision == "pass"
    map_requests = [
        request
        for invocation in backend.invocations
        if (request := json.loads(invocation.user))["schema_version"]
        == "assay.review.chunk.v1"
    ]
    assert [
        (request["char_start"], request["char_end"], request["chunk_text"])
        for request in map_requests
    ] == [
        (0, 4, "ABCD"),
        (4, 8, "EFGH"),
        (8, 11, "IJK"),
    ]
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["chunk_chars"] == 4
    assert [
        (chunk["char_start"], chunk["char_end"])
        for chunk in manifest["sources"][0]["chunks"]
    ] == [(0, 4), (4, 8), (8, 11)]


def test_artifact_maps_receive_only_selected_required_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    directive = "formatting is an attributed integration change"
    constraints = "retain all source-local evidence"
    unselected = "mapped independently but not copied beside the artifact"
    selected_ids = (
        "context:directive_intent",
        "context:constraints",
    )
    selected_with_text = [
        _map_context_record(
            "context:directive_intent",
            directive,
            include_text=True,
        ),
        _map_context_record(
            "context:constraints",
            constraints,
            include_text=True,
        ),
    ]
    selected_metadata = [
        {key: value for key, value in record.items() if key != "text"}
        for record in selected_with_text
    ]
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        context={
            "constraints": constraints,
            "directive_intent": directive,
            "unselected": unselected,
        },
        required_source_ids=(
            "artifact",
            "context:constraints",
            "context:directive_intent",
            "context:unselected",
        ),
        map_context_source_ids=selected_ids,
        chunk_chars=4,
    )

    assert verdict.decision == "pass"
    map_requests = [
        request
        for invocation in backend.invocations
        if (request := json.loads(invocation.user))["schema_version"]
        == "assay.review.chunk.v1"
    ]
    artifact_requests = [
        request for request in map_requests if request["source_id"] == "artifact"
    ]
    context_requests = [
        request for request in map_requests if request["source_id"] != "artifact"
    ]
    assert artifact_requests
    assert all(
        request["map_context_sources"] == selected_with_text
        for request in artifact_requests
    )
    artifact_invocations = [
        invocation
        for invocation in backend.invocations
        if (request := json.loads(invocation.user))["schema_version"]
        == "assay.review.chunk.v1"
        and request["source_id"] == "artifact"
    ]
    assert all(
        "interpretation-only companions" in invocation.system
        and "not mapped coverage or mapper evidence" in invocation.system
        and "Never cite a companion" in invocation.system
        and "as instructions." in invocation.system
        for invocation in artifact_invocations
    )
    assert all("map_context_sources" not in request for request in context_requests)
    assert {request["source_id"] for request in context_requests} == {
        "context:constraints",
        "context:directive_intent",
        "context:unselected",
    }

    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_context_sources"] == selected_metadata
    assert all("text" not in record for record in manifest["map_context_sources"])
    artifact_attempts = [
        item
        for item in manifest["map_attempt_ledger"]
        if item["source_id"] == "artifact"
    ]
    context_attempts = [
        item
        for item in manifest["map_attempt_ledger"]
        if item["source_id"] != "artifact"
    ]
    assert artifact_attempts
    assert all(
        item["map_context_sources"] == selected_metadata
        for item in artifact_attempts
    )
    assert all("map_context_sources" not in item for item in context_attempts)
    assert all(
        "text" not in record
        for item in artifact_attempts
        for record in item["map_context_sources"]
    )


def test_empty_map_context_opt_in_preserves_chunk_prompt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    default_backend = ChunkingBackend()
    default_corpus = Corpus(tmp_path / "default")
    default_verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=default_backend,
        corpus=default_corpus,
    )
    explicit_backend = ChunkingBackend()
    explicit_corpus = Corpus(tmp_path / "explicit")
    explicit_verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=explicit_backend,
        corpus=explicit_corpus,
        map_context_source_ids=(),
    )

    assert default_verdict.decision == explicit_verdict.decision == "pass"
    assert [
        (invocation.system, invocation.user)
        for invocation in default_backend.invocations
    ] == [
        (invocation.system, invocation.user)
        for invocation in explicit_backend.invocations
    ]
    assert all(
        "map_context_sources" not in json.loads(invocation.user)
        for invocation in default_backend.invocations
    )
    assert all(
        "map_context_sources" not in invocation.system
        for invocation in default_backend.invocations
    )
    assert default_corpus.get_artifact(
        default_verdict.artifacts["review_source_manifest"]
    ) == explicit_corpus.get_artifact(
        explicit_verdict.artifacts["review_source_manifest"]
    )


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (["context:directive"], TypeError, "must be a tuple"),
        (("",), TypeError, "non-empty strings"),
        ((1,), TypeError, "non-empty strings"),
        (
            ("context:directive", "context:directive"),
            ValueError,
            "must be unique",
        ),
        (("artifact",), ValueError, "only retained context"),
        (("prior:finding",), ValueError, "only retained context"),
        (("context:",), ValueError, "only retained context"),
        (("context:missing",), ValueError, "unavailable retained source"),
        (
            ("context:optional",),
            ValueError,
            "only terminal-required retained context",
        ),
    ],
)
def test_map_context_selection_fails_before_transport(
    tmp_path: Path,
    value,
    error: type[Exception],
    match: str,
):
    backend = ChunkingBackend()

    with pytest.raises(error, match=match):
        review(
            b"artifact",
            _make_rubric(),
            backend=backend,
            corpus=Corpus(tmp_path),
            context={
                "directive": "required",
                "optional": "retained but not required",
            },
            required_source_ids=("artifact", "context:directive"),
            map_context_source_ids=value,
        )

    assert backend.invocations == []


def test_chunk_coordinate_mismatch_prevents_reduction(tmp_path: Path):
    class CorruptingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                return result
            payload = json.loads(result.final_message)
            payload["chunk_sha256"] = "sha256:" + "0" * 64
            final = json.dumps(payload)
            return dataclasses.replace(result, final_message=final)

    backend = CorruptingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_schema_failure"
    assert "chunk_sha256" in verdict.reasoning
    assert len(backend.invocations) == 1
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["complete"] is False
    assert all(
        chunk["status"] == "not_reviewed"
        for chunk in manifest["sources"][0]["chunks"]
    )


def test_artifact_mapper_cannot_cite_visible_context_companion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class ContextCitingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                request["schema_version"] != "assay.review.chunk.v1"
                or request["source_id"] != "artifact"
            ):
                return result
            payload = json.loads(result.final_message)
            payload["findings"] = [
                _finding(
                    disposition="confirmed",
                    targets=[],
                    evidence=[{
                        "source_id": "context:directive",
                        "kind": "source_quote",
                        "location": "visible companion",
                        "excerpt": "interpret this artifact",
                    }],
                )
            ]
            return dataclasses.replace(
                result,
                final_message=json.dumps(payload),
            )

    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    backend = ContextCitingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        context={"directive": "interpret this artifact"},
        required_source_ids=("artifact", "context:directive"),
        map_context_source_ids=("context:directive",),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_schema_failure"
    assert "evidence" in verdict.reasoning
    assert len(backend.invocations) == 1
    request = json.loads(backend.invocations[0].user)
    assert request["map_context_sources"][0]["text"] == (
        "interpret this artifact"
    )
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["complete"] is False
    assert manifest["map_attempt_ledger"][0]["status"] == "schema_rejected"


def test_chunk_schema_failure_retries_only_current_chunk_and_converges(
    tmp_path: Path,
):
    class RepairingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                request["schema_version"] == "assay.review.chunk.v1"
                and request["chunk_index"] == 0
                and "response_repair" not in request
            ):
                payload = json.loads(result.final_message)
                payload["findings"] = [
                    _finding(
                        evidence=[
                            {
                                "source_id": "artifact",
                                "kind": "source_quote",
                                "location": "chunk 0",
                                "excerpt": "not present in the immutable chunk",
                            }
                        ]
                    )
                ]
                result = dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            return dataclasses.replace(
                result,
                pricing_evidence=UsagePricingEvidence(
                    segments=(UsagePricingSegment(1, 2, 0, 0, 0.005),),
                    complete=True,
                ),
            )

    backend = RepairingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    requests = [json.loads(item.user) for item in backend.invocations]
    assert [item["schema_version"] for item in requests] == [
        "assay.review.chunk.v1",
        "assay.review.chunk.v1",
        "assay.review.chunk.v1",
        "assay.review.reduce.v1",
    ]
    first, repaired = requests[:2]
    assert first["chunk_sha256"] == repaired["chunk_sha256"]
    assert first["chunk_text"] == repaired["chunk_text"]
    assert "BOUNDED-SOURCE MAP RESPONSE REPAIR" not in (
        backend.invocations[0].system
    )
    assert "BOUNDED-SOURCE MAP RESPONSE REPAIR" in backend.invocations[1].system
    assert "BOUNDED-SOURCE MAP RESPONSE REPAIR" not in (
        backend.invocations[2].system
    )
    assert repaired["response_repair"]["attempt"] == 1
    assert repaired["response_repair"]["validation"] == {
        "code": "schema_failure",
        "path": "findings[0].evidence[0].excerpt",
    }
    assert "not present in the immutable chunk" not in backend.invocations[1].user
    assert "not present in the immutable chunk" not in backend.invocations[1].system
    rendered = json.loads(corpus.get_artifact(verdict.rendered_prompt_ref))
    assert [item["phase"] for item in rendered["invocations"]] == [
        "map",
        "map_repair",
        "map",
        "reduce",
    ]
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["schema_version"] == "assay.review-source-manifest.v2"
    assert manifest["map_retry_policy"] == {
        "scope": "review",
        "authorized_retries": 1,
        "consumed_retries": 1,
    }
    assert manifest["budget_usd"] is None
    ledger = manifest["map_attempt_ledger"]
    assert [item["status"] for item in ledger] == [
        "schema_rejected",
        "accepted",
        "accepted",
    ]
    assert ledger[0]["validation_code"] == "schema_failure"
    assert ledger[0]["validation_path"] == "findings[0].evidence[0].excerpt"
    assert ledger[0]["pricing_evidence_call_index"] == 0
    assert ledger[0]["pricing_segment_start"] == 0
    assert ledger[0]["pricing_segment_end"] == 1
    assert [
        (item["pricing_segment_start"], item["pricing_segment_end"])
        for item in ledger
    ] == [(0, 1), (1, 2), (2, 3)]
    assert ledger[0]["response_ref"] == ledger[0]["response_sha256"]
    for item, invocation in zip(ledger, backend.invocations[:3], strict=True):
        assert item["request_budget_usd"] == invocation.budget_usd
        assert item["request_max_turns"] == invocation.max_turns
        assert item["response_schema_sha256"].startswith("sha256:")
        assert item["invocation_sha256"].startswith("sha256:")
    rejected = corpus.get_artifact(ledger[0]["response_ref"]).decode()
    assert "not present in the immutable chunk" in rejected
    # The stub's transcript retained its original valid response while its
    # final_message was replaced above. The harness ledger must bind the exact
    # rejected final_message independently of adapter transcript behavior.
    assert rejected != backend.invocations[0].user


def test_map_context_is_immutable_across_bounded_map_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class RepairingContextBackend(ChunkingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.invalidated = False

        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                not self.invalidated
                and request["schema_version"] == "assay.review.chunk.v1"
                and request["source_id"] == "artifact"
                and "response_repair" not in request
            ):
                self.invalidated = True
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            return result

    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    directive = "retain this exact companion across retries"
    context_with_text = [
        _map_context_record(
            "context:directive",
            directive,
            include_text=True,
        )
    ]
    context_metadata = [
        _map_context_record(
            "context:directive",
            directive,
            include_text=False,
        )
    ]
    backend = RepairingContextBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        b"ABCDEFGHIJK",
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        context={"directive": directive},
        required_source_ids=("artifact", "context:directive"),
        map_context_source_ids=("context:directive",),
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    requests = [json.loads(invocation.user) for invocation in backend.invocations]
    first, repaired = requests[:2]
    assert first["source_id"] == repaired["source_id"] == "artifact"
    assert first["chunk_sha256"] == repaired["chunk_sha256"]
    assert first["chunk_text"] == repaired["chunk_text"]
    assert first["map_context_sources"] == context_with_text
    assert repaired["map_context_sources"] == context_with_text
    assert repaired["response_repair"]["attempt"] == 1
    assert all(
        len(invocation.system) + len(invocation.user)
        <= review_module._MAX_MAPPER_ENVELOPE_CHARS
        for invocation in backend.invocations[:-1]
    )

    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    artifact_attempts = [
        item
        for item in manifest["map_attempt_ledger"]
        if item["source_id"] == "artifact"
    ]
    assert [item["status"] for item in artifact_attempts] == [
        "schema_rejected",
        "accepted",
    ]
    assert all(
        item["map_context_sources"] == context_metadata
        for item in artifact_attempts
    )
    assert all(
        "map_context_sources" not in item
        for item in manifest["map_attempt_ledger"]
        if item["source_id"] != "artifact"
    )


def test_first_adaptive_map_reserves_complete_repair_envelope(
    tmp_path: Path,
):
    class RepairFirstAdaptiveMap(ChunkingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.invalidated = False

        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                not self.invalidated
                and request["schema_version"] == "assay.review.chunk.v1"
                and request["chunk_index"] == 0
                and "response_repair" not in request
            ):
                self.invalidated = True
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            return result

    source = "\x00" * 40_000
    backend = RepairFirstAdaptiveMap()
    corpus = Corpus(tmp_path)

    verdict = review(
        source.encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    requests = [json.loads(item.user) for item in backend.invocations]
    assert [item["schema_version"] for item in requests] == [
        "assay.review.chunk.v1",
        "assay.review.chunk.v1",
        "assay.review.chunk.v1",
        "assay.review.reduce.v1",
    ]
    first, repaired, second = requests[:3]
    assert [item["chunk_index"] for item in (first, repaired, second)] == [
        0,
        0,
        1,
    ]
    immutable_fields = (
        "source_id",
        "source_sha256",
        "source_chars",
        "chunk_index",
        "chunk_count",
        "char_start",
        "char_end",
        "chunk_sha256",
        "chunk_text",
    )
    assert {
        field: first[field] for field in immutable_fields
    } == {
        field: repaired[field] for field in immutable_fields
    }
    assert repaired["response_repair"]["attempt"] == 1
    assert first["char_start"] == 0
    assert second["char_start"] == first["char_end"]
    assert second["char_end"] == len(source)
    assert "".join((first["chunk_text"], second["chunk_text"])) == source
    assert all(
        len(invocation.system) + len(invocation.user)
        <= review_module._MAX_MAPPER_ENVELOPE_CHARS
        for invocation in backend.invocations[:3]
    )
    assert verdict.cost_usd == pytest.approx(0.020)
    assert verdict.input_tokens == 4
    assert verdict.output_tokens == 8
    assert verdict.accounting_debt == ()

    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"] == {
        "scope": "review",
        "authorized_retries": 1,
        "consumed_retries": 1,
    }
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "accepted",
        "accepted",
    ]


def test_chunk_schema_repair_retains_valid_sibling_findings(
    tmp_path: Path,
):
    class PartiallyInvalidBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] == "assay.review.chunk.v1":
                if request["chunk_index"] != 0:
                    return result
                payload = json.loads(result.final_message)
                retained_high = _finding(
                    id="retained_high",
                    severity="High",
                    disposition="confirmed",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "A",
                    }],
                )
                if "response_repair" in request:
                    payload["findings"] = [retained_high]
                    return dataclasses.replace(
                        result,
                        final_message=json.dumps(payload),
                    )
                payload["findings"] = [
                    retained_high,
                    _finding(
                        id="omitted_high",
                        severity="High",
                        disposition="confirmed",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "A",
                        }],
                    ),
                    _finding(
                        id="invalid_sibling",
                        severity="Low",
                        disposition="suspected",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "not present in the immutable chunk",
                        }],
                    ),
                ]
                return dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            targets = request["required_targets"]
            payload = _response(
                decision="revise",
                reasoning="The retained High finding requires remediation.",
                findings=[
                    _finding(
                        id="terminal_headline",
                        severity="High",
                        disposition="confirmed",
                        targets=targets,
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "mapped source",
                            "excerpt": "A",
                        }],
                    )
                ]
            )
            return dataclasses.replace(
                result,
                final_message=json.dumps(payload),
            )

    backend = PartiallyInvalidBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    repair_request = backend.invocations[1].user
    assert "retained_high" not in repair_request
    assert "omitted_high" not in repair_request
    assert "invalid_sibling" not in repair_request
    terminal_request = json.loads(backend.invocations[-1].user)
    assert len(terminal_request["required_targets"]) == 1
    terminal_findings = [
        finding
        for report in terminal_request["reduction_reports"]
        for finding in report["findings"]
    ]
    assert len(terminal_findings) == 2
    terminal_ids = {finding["id"] for finding in terminal_findings}
    assert any(item.endswith(":retained_high") for item in terminal_ids)
    assert any(item.endswith(":omitted_high") for item in terminal_ids)

    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    first_attempt, accepted_repair = manifest["map_attempt_ledger"][:2]
    assert first_attempt["newly_retained_finding_ids"] == sorted(terminal_ids)
    assert accepted_repair["applied_retained_finding_ids"] == sorted(
        terminal_ids
    )
    first_chunk = manifest["sources"][0]["chunks"][0]
    assert first_chunk["supporting_response_sha256s"] == [
        first_attempt["response_sha256"]
    ]
    assert first_chunk["response_sha256"] == accepted_repair["response_sha256"]


def test_chunk_schema_repair_does_not_salvage_duplicate_identities(
    tmp_path: Path,
):
    class DuplicateFindingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                payload = _response(
                    decision="revise",
                    reasoning="The unique retained finding requires remediation.",
                    findings=[
                        _finding(
                            id="terminal_headline",
                            severity="Critical",
                            disposition="confirmed",
                            targets=request["required_targets"],
                            evidence=[{
                                "source_id": "artifact",
                                "kind": "source_quote",
                                "location": "mapped source",
                                "excerpt": "A",
                            }],
                        )
                    ],
                )
                return dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            if (
                request["chunk_index"] != 0
                or "response_repair" in request
            ):
                return result
            payload = json.loads(result.final_message)
            payload["findings"] = [
                _finding(
                    id="same",
                    severity="Low",
                    disposition="suspected",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "A",
                    }],
                ),
                _finding(
                    id=" same ",
                    severity="Critical",
                    disposition="confirmed",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "A",
                    }],
                ),
                _finding(
                    id="unique_critical",
                    severity="Critical",
                    disposition="confirmed",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "A",
                    }],
                ),
            ]
            return dataclasses.replace(result, final_message=json.dumps(payload))

    backend = DuplicateFindingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    first_attempt, accepted_repair = manifest["map_attempt_ledger"][:2]
    assert first_attempt["status"] == "schema_rejected"
    retained_ids = first_attempt["newly_retained_finding_ids"]
    assert len(retained_ids) == 1
    assert retained_ids[0].endswith(":unique_critical")
    assert accepted_repair["applied_retained_finding_ids"] == retained_ids


def test_chunk_schema_repair_rejects_changed_retained_identity(
    tmp_path: Path,
):
    class ConflictingRepairBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                raise AssertionError("retained identity conflict must skip reduction")
            if request["chunk_index"] != 0:
                return result
            payload = json.loads(result.final_message)
            if "response_repair" not in request:
                payload["findings"] = [
                    _finding(
                        id="same",
                        severity="Low",
                        disposition="suspected",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "A",
                        }],
                    ),
                    _finding(
                        id="invalid_sibling",
                        severity="Low",
                        disposition="suspected",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "not present in the immutable chunk",
                        }],
                    ),
                ]
            else:
                payload["findings"] = [
                    _finding(
                        id="same",
                        severity="Critical",
                        disposition="confirmed",
                        evidence=[{
                            "source_id": "artifact",
                            "kind": "source_quote",
                            "location": "chunk 0",
                            "excerpt": "A",
                        }],
                    )
                ]
            return dataclasses.replace(result, final_message=json.dumps(payload))

    backend = ConflictingRepairBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_retained_finding_conflict"
    assert len(backend.invocations) == 2
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "retention_rejected",
    ]
    assert manifest["map_attempt_ledger"][1]["validation_code"] == (
        "retained_finding_conflict"
    )


def test_chunk_schema_repair_rejects_retention_overflow_across_retries(
    tmp_path: Path,
):
    class OverflowingRepairBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                raise AssertionError("retention overflow must skip reduction")
            if request["chunk_index"] != 0:
                return result
            payload = json.loads(result.final_message)
            start = 0 if "response_repair" not in request else 63
            count = 63 if start == 0 else 2
            payload["findings"] = [
                _finding(
                    id=f"retained_{index}",
                    severity="Low",
                    disposition="suspected",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "A",
                    }],
                )
                for index in range(start, start + count)
            ]
            payload["findings"].append(
                _finding(
                    id=f"invalid_{start}",
                    severity="Low",
                    disposition="suspected",
                    evidence=[{
                        "source_id": "artifact",
                        "kind": "source_quote",
                        "location": "chunk 0",
                        "excerpt": "not present in the immutable chunk",
                    }],
                )
            )
            return dataclasses.replace(result, final_message=json.dumps(payload))

    backend = OverflowingRepairBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_retained_findings_overflow"
    assert len(backend.invocations) == 2
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "retention_rejected",
    ]
    assert manifest["map_attempt_ledger"][1]["validation_code"] == (
        "retained_findings_overflow"
    )


def test_chunk_retry_allowance_is_review_wide_not_per_chunk(tmp_path: Path):
    class CorruptEachInitialMap(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                request["schema_version"] != "assay.review.chunk.v1"
                or "response_repair" in request
            ):
                return result
            payload = json.loads(result.final_message)
            payload["chunk_sha256"] = "sha256:" + "0" * 64
            return dataclasses.replace(result, final_message=json.dumps(payload))

    backend = CorruptEachInitialMap()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 3
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"]["consumed_retries"] == 1
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "accepted",
        "schema_rejected",
    ]


def test_chunk_parse_failure_retries_only_current_chunk(tmp_path: Path):
    class RepairingParseBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                request["schema_version"] == "assay.review.chunk.v1"
                and request["chunk_index"] == 0
                and "response_repair" not in request
            ):
                return dataclasses.replace(result, final_message="not JSON")
            return result

    backend = RepairingParseBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    repaired = json.loads(backend.invocations[1].user)["response_repair"]
    assert repaired["validation"] == {
        "code": "parse_failure",
        "path": "response",
    }
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "parse_rejected",
        "accepted",
        "accepted",
    ]


def test_chunk_budget_exhaustion_does_not_dispatch_authorized_repair(
    tmp_path: Path,
):
    class CorruptingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] == "assay.review.chunk.v1":
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(
                    result,
                    final_message=json.dumps(payload),
                )
            return result

    backend = CorruptingBackend()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        budget_usd=0.005,
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_review_budget_exhausted"
    assert len(backend.invocations) == 1
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"] == {
        "scope": "review",
        "authorized_retries": 1,
        "consumed_retries": 0,
    }
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected"
    ]


def test_chunk_manifest_binds_aggregate_and_remaining_call_budget(tmp_path: Path):
    manifests = []
    manifest_refs = []
    backends = []
    for index, budget in enumerate((0.1, 1.0)):
        corpus = Corpus(tmp_path / str(index))
        backend = ChunkingBackend()
        backends.append(backend)
        verdict = review(
            _oversized_artifact(),
            _make_rubric(),
            backend=backend,
            corpus=corpus,
            budget_usd=budget,
        )
        manifests.append(json.loads(
            corpus.get_artifact(verdict.artifacts["review_source_manifest"])
        ))
        manifest_refs.append(verdict.artifacts["review_source_manifest"])

    assert manifest_refs[0] != manifest_refs[1]
    assert [item["budget_usd"] for item in manifests] == [0.1, 1.0]
    for manifest, backend, budget in zip(
        manifests, backends, (0.1, 1.0), strict=True
    ):
        spent = 0.0
        ledger = manifest["map_attempt_ledger"]
        for item, invocation in zip(
            ledger, backend.invocations[: len(ledger)], strict=True
        ):
            assert item["request_budget_usd"] == pytest.approx(budget - spent)
            assert item["request_budget_usd"] == invocation.budget_usd
            assert item["request_max_turns"] == invocation.max_turns
            canonical_schema = json.dumps(
                invocation.response_schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            response_schema_sha256 = (
                "sha256:"
                + hashlib.sha256(canonical_schema.encode()).hexdigest()
            )
            assert item["response_schema_sha256"] == response_schema_sha256
            invocation_payload = {
                "system_sha256": (
                    "sha256:"
                    + hashlib.sha256(invocation.system.encode()).hexdigest()
                ),
                "user_sha256": (
                    "sha256:"
                    + hashlib.sha256(invocation.user.encode()).hexdigest()
                ),
                "tools": None,
                "max_turns": invocation.max_turns,
                "budget_usd": invocation.budget_usd,
                "response_schema_sha256": response_schema_sha256,
                "conversation": None,
            }
            canonical_invocation = json.dumps(
                invocation_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            assert item["invocation_sha256"] == (
                "sha256:"
                + hashlib.sha256(canonical_invocation.encode()).hexdigest()
            )
            spent += item["usage"]["cost_usd"]


def test_chunk_repair_backend_error_is_terminal_and_audited(tmp_path: Path):
    class RepairBackendError(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                return result
            if "response_repair" not in request:
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(result, final_message=json.dumps(payload))
            return dataclasses.replace(
                result,
                error=TransportError(message="repair transport"),
            )

    backend = RepairBackendError()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=2,
    )

    assert verdict.findings[0].id == "chunk_backend_transport"
    assert len(backend.invocations) == 2
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"]["consumed_retries"] == 1
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "backend_error",
    ]


def test_chunk_repair_accounting_debt_is_terminal_and_audited(tmp_path: Path):
    class RepairAccountingDebt(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                return result
            if "response_repair" not in request:
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(result, final_message=json.dumps(payload))
            return dataclasses.replace(
                result,
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
            )

    backend = RepairAccountingDebt()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=2,
    )

    assert verdict.findings[0].id == "chunk_review_accounting_incomplete"
    assert len(backend.invocations) == 2
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"]["consumed_retries"] == 1
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "accounting_incomplete",
    ]


def test_chunk_repair_budget_overrun_is_terminal_and_audited(tmp_path: Path):
    class RepairOverrun(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                return result
            if "response_repair" not in request:
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                return dataclasses.replace(result, final_message=json.dumps(payload))
            return dataclasses.replace(result, usage=Usage(cost_usd=0.006))

    backend = RepairOverrun()
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        budget_usd=0.01,
        chunk_response_retry_budget=2,
    )

    assert verdict.findings[0].id == "chunk_review_budget_overrun"
    assert len(backend.invocations) == 2
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"]["consumed_retries"] == 1
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "schema_rejected",
        "budget_overrun",
    ]


def test_default_zero_keeps_initial_map_prompt_free_of_repair_protocol(
    tmp_path: Path,
):
    backend = ChunkingBackend()
    review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    map_invocations = [
        item
        for item in backend.invocations
        if json.loads(item.user)["schema_version"] == "assay.review.chunk.v1"
    ]
    assert map_invocations
    assert all(
        "BOUNDED-SOURCE MAP RESPONSE REPAIR" not in item.system
        for item in map_invocations
    )


def test_chunk_schema_retry_exhaustion_fails_before_later_chunks(
    tmp_path: Path,
):
    class AlwaysCorruptingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.chunk.v1":
                return result
            payload = json.loads(result.final_message)
            payload["chunk_sha256"] = "sha256:" + "0" * 64
            return dataclasses.replace(result, final_message=json.dumps(payload))

    backend = AlwaysCorruptingBackend()
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
        chunk_response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_schema_failure"
    assert len(backend.invocations) == 2
    assert "response_repair" not in json.loads(backend.invocations[0].user)
    assert json.loads(backend.invocations[1].user)["response_repair"][
        "attempt"
    ] == 1


@pytest.mark.parametrize(
    "value",
    [-1, True, 1.5, review_module.MAX_CHUNK_RESPONSE_RETRY_BUDGET + 1],
)
def test_chunk_response_retry_budget_must_be_bounded_integer(
    tmp_path: Path,
    value,
):
    with pytest.raises(ValueError, match="chunk_response_retry_budget"):
        review(
            _oversized_artifact(),
            _make_rubric(),
            backend=ChunkingBackend(),
            corpus=Corpus(tmp_path),
            chunk_response_retry_budget=value,
        )


def test_max_chunk_retry_budget_reserves_bounded_adaptive_maps(
    tmp_path: Path,
):
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        ("\x00" * 40_000).encode(),
        _make_rubric(),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=(
            review_module.MAX_CHUNK_RESPONSE_RETRY_BUDGET
        ),
    )

    assert verdict.decision == "pass"
    map_invocations = backend.invocations[:-1]
    assert len(map_invocations) == 2
    assert all(
        len(invocation.system) + len(invocation.user)
        <= review_module._MAX_MAPPER_ENVELOPE_CHARS
        for invocation in map_invocations
    )
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"] == {
        "scope": "review",
        "authorized_retries": (
            review_module.MAX_CHUNK_RESPONSE_RETRY_BUDGET
        ),
        "consumed_retries": 0,
    }


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, _MAX_REVIEW_CHUNK_CHARS + 1],
)
def test_review_chunk_chars_must_be_a_bounded_positive_integer(
    tmp_path: Path,
    value,
):
    backend = ChunkingBackend()

    with pytest.raises(ValueError, match="chunk_chars"):
        review(
            b"artifact",
            _make_rubric(),
            backend=backend,
            corpus=Corpus(tmp_path),
            chunk_chars=value,
        )

    assert backend.invocations == []


def test_terminal_reducer_repairs_span_and_attaches_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    finding = _finding(
        severity="Low",
        disposition="addressed",
        evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "chars:99-105",
            "excerpt": "needle",
        }],
    )
    corpus = Corpus(tmp_path)
    verdict = review(
        b"artifact with needle in it",
        _make_rubric(),
        backend=ChunkingBackend(reducer_findings=[finding]),
        corpus=corpus,
        response_repair_policy=ReviewResponseRepairPolicy(
            materialize_source_anchors=True,
            materialize_json_paths=True,
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
            repair_unique_json_quote_escaped_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "pass"
    assert verdict.findings[0].evidence[0].location == "chars:14-20"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["actions"][0]["operation"] == (
        "repair_unique_exact_excerpt_char_span"
    )
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    assert rubric_artifact["response_repair_policy"] == {
        "repair_unique_exact_excerpt_char_spans": True,
        "repair_unresolved_source_anchor_from_unique_exact_excerpt": False,
        "repair_unique_whitespace_normalized_excerpt_char_spans": False,
        "repair_unique_json_quote_escaped_excerpt_char_spans": False,
        "materialize_source_anchors": False,
        "materialize_json_paths": False,
        "require_source_anchors": False,
        "derive_decision_from_criterion_results": False,
        "recover_fully_satisfied_revise_to_pass": False,
        "source_anchor_chars": 512,
    }
    degradation = rubric_artifact["response_repair_degradation"]
    assert degradation["reason"] == (
        "chunk_protocol_disables_single_source_references"
    )
    assert degradation["requested_response_repair_policy"] == {
        **rubric_artifact["response_repair_policy"],
        "materialize_source_anchors": True,
        "materialize_json_paths": True,
        "repair_unique_whitespace_normalized_excerpt_char_spans": True,
        "repair_unique_json_quote_escaped_excerpt_char_spans": True,
    }
    assert degradation["effective_response_repair_policy"] == (
        rubric_artifact["response_repair_policy"]
    )
    assert verdict.artifacts["review_raw_response"] == report[
        "raw_response_ref"
    ]


def test_dynamic_chunk_switch_disables_whitespace_quote_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 1_000)
    finding = _finding(
        severity="Low",
        disposition="addressed",
        evidence=[{
            "source_id": "artifact",
            "kind": "source_quote",
            "location": "descriptive location",
            "excerpt": "this is directory materialization",
        }],
    )
    corpus = Corpus(tmp_path)
    backend = ChunkingBackend(reducer_findings=[finding])
    verdict = review(
        b"this\nis directory materialization",
        _make_rubric(
            user_template="Review {artifact}\nConstraints {spec_constraints}",
        ),
        backend=backend,
        corpus=corpus,
        context={"spec_constraints": "\x00" * 300},
        required_source_ids=("artifact", "context:spec_constraints"),
        map_context_source_ids=("context:spec_constraints",),
        response_repair_policy=ReviewResponseRepairPolicy(
            repair_unique_whitespace_normalized_excerpt_char_spans=True,
            repair_unique_json_quote_escaped_excerpt_char_spans=True,
        ),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reducer_schema_failure"
    rubric_artifact = json.loads(corpus.get_artifact(verdict.rubric_ref))
    assert rubric_artifact["response_repair_degradation"]["reason"] == (
        "rendered_prompt_requires_chunk_protocol"
    )
    requested = rubric_artifact["response_repair_degradation"][
        "requested_response_repair_policy"
    ]
    effective = rubric_artifact["response_repair_policy"]
    assert requested[
        "repair_unique_whitespace_normalized_excerpt_char_spans"
    ] is True
    assert effective[
        "repair_unique_whitespace_normalized_excerpt_char_spans"
    ] is False
    assert requested[
        "repair_unique_json_quote_escaped_excerpt_char_spans"
    ] is True
    assert effective[
        "repair_unique_json_quote_escaped_excerpt_char_spans"
    ] is False
    map_requests = [
        request
        for invocation in backend.invocations
        if (request := json.loads(invocation.user))["schema_version"]
        == "assay.review.chunk.v1"
    ]
    artifact_requests = [
        request for request in map_requests if request["source_id"] == "artifact"
    ]
    context_requests = [
        request for request in map_requests if request["source_id"] != "artifact"
    ]
    assert artifact_requests
    assert all(
        request["map_context_sources"][0]["text"] == "\x00" * 300
        for request in artifact_requests
    )
    assert context_requests
    assert all("map_context_sources" not in request for request in context_requests)


def test_escape_expanded_map_context_fails_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 50_000)
    context_text = "\x00" * 40_000
    backend = ChunkingBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        b"A",
        _make_rubric(
            user_template="Review {artifact}\nDirective {directive}",
        ),
        backend=backend,
        corpus=corpus,
        context={"directive": context_text},
        required_source_ids=("artifact", "context:directive"),
        map_context_source_ids=("context:directive",),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "mapper_prompt_envelope_failure"
    assert "irreducible" in verdict.reasoning
    assert backend.invocations == []
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["complete"] is False
    assert manifest["map_attempt_ledger"] == []
    assert manifest["map_context_sources"] == [
        _map_context_record(
            "context:directive",
            context_text,
            include_text=False,
        )
    ]


def test_map_observations_are_reduced_to_one_terminal_finding(tmp_path: Path):
    backend = _ObservationBackend("Low")
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    assert verdict.decision == "pass"
    assert [finding.id for finding in verdict.findings] == ["terminal_headline"]
    assert len(verdict.findings[0].targets) == 1
    assert verdict.findings[0].targets[0].startswith("map:")


def test_blocking_map_observation_vetoes_reducer_pass(tmp_path: Path):
    backend = _ObservationBackend("High")
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reducer_schema_failure"
    assert "pass conflicts with blocking mapped findings" in verdict.reasoning


def test_visible_map_context_does_not_clear_actual_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_ARTIFACT_CHARS", 10)
    backend = _ObservationBackend("High")
    verdict = review(
        b"AAAAAAAAAAA",
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
        context={"directive": "formatting is attributed, defects are not"},
        required_source_ids=("artifact", "context:directive"),
        map_context_source_ids=("context:directive",),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reducer_schema_failure"
    assert "pass conflicts with blocking mapped findings" in verdict.reasoning
    artifact_request = next(
        request
        for invocation in backend.invocations
        if (request := json.loads(invocation.user)).get("source_id")
        == "artifact"
    )
    assert artifact_request["map_context_sources"][0]["text"] == (
        "formatting is attributed, defects are not"
    )


@pytest.mark.parametrize("budget", [0.001, 0.002])
def test_chunk_protocol_enforces_one_aggregate_budget(tmp_path: Path, budget: float):
    backend = ChunkingBackend()
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
        budget_usd=budget,
    )

    assert verdict.decision == "inconclusive"
    assert "budget" in verdict.findings[0].id
    assert verdict.cost_usd <= budget or "overrun" in verdict.findings[0].id


def test_chunk_review_concatenates_pricing_segments_in_call_order(
    tmp_path: Path,
):
    class _PricingChunkingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            sequence = len(self.invocations)
            usage = Usage(
                input_tokens=sequence,
                output_tokens=2,
                cost_usd=0.005,
                cache_read_tokens=sequence,
            )
            return dataclasses.replace(
                result,
                usage=usage,
                pricing_evidence=UsagePricingEvidence(
                    segments=(
                        UsagePricingSegment(
                            sequence,
                            2,
                            0,
                            sequence,
                            0.005,
                        ),
                    ),
                    complete=True,
                ),
            )

    backend = _PricingChunkingBackend()
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    assert verdict.pricing_evidence is not None
    assert verdict.pricing_evidence.complete is True
    assert verdict.cache_creation_tokens == 0
    assert verdict.cache_read_tokens == sum(
        range(1, len(backend.invocations) + 1)
    )
    assert [
        segment.input_tokens
        for segment in verdict.pricing_evidence.segments
    ] == list(range(1, len(backend.invocations) + 1))


def test_chunk_review_marks_pricing_incomplete_when_one_call_omits_it(
    tmp_path: Path,
):
    class _PartialPricingBackend(ChunkingBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            sequence = len(self.invocations)
            if sequence == 2:
                return result
            return dataclasses.replace(
                result,
                pricing_evidence=UsagePricingEvidence(
                    segments=(UsagePricingSegment(1, 2, 0, 0, 0.005),),
                    complete=True,
                ),
            )

    backend = _PartialPricingBackend()
    verdict = review(
        _oversized_artifact(),
        _make_rubric(),
        backend=backend,
        corpus=Corpus(tmp_path),
    )

    assert verdict.pricing_evidence is not None
    assert verdict.pricing_evidence.complete is False
    assert len(verdict.pricing_evidence.segments) == len(backend.invocations) - 1


def test_chunk_protocol_stops_before_next_call_on_accounting_debt(
    tmp_path: Path,
):
    class _AccountingDebtBackend(_ObservationBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            assert len(self.invocations) == 1, (
                "accounting debt must stop before another paid invocation"
            )
            return dataclasses.replace(
                result,
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
                execution_trace=BackendExecutionTrace(
                    backend=self.name,
                    requested_model=self.model,
                    events=(
                        BackendExecutionEvent(
                            sequence=0,
                            kind="terminal",
                            name="end_turn",
                        ),
                    ),
                ),
            )

    backend = _AccountingDebtBackend("Low")
    corpus = Corpus(tmp_path)
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=corpus,
        chunk_response_retry_budget=2,
    )

    assert len(backend.invocations) == 1
    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "chunk_review_accounting_incomplete"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"]["consumed_retries"] == 0
    assert [item["status"] for item in manifest["map_attempt_ledger"]] == [
        "accounting_incomplete"
    ]


def test_hierarchical_reducer_stops_after_accounting_debt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_REDUCER_ENVELOPE_CHARS", 6_000)

    class _ReducerDebtBackend(_HierarchicalBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.reduce-map.v1":
                return result
            assert sum(
                json.loads(item.user)["schema_version"]
                == "assay.review.reduce-map.v1"
                for item in self.invocations
            ) == 1, "no later reducer call is authorized after accounting debt"
            return dataclasses.replace(
                result,
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
                execution_trace=BackendExecutionTrace(
                    backend=self.name,
                    requested_model=self.model,
                    events=(
                        BackendExecutionEvent(
                            sequence=0,
                            kind="terminal",
                            name="end_turn",
                        ),
                    ),
                ),
            )

    backend = _ReducerDebtBackend()
    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
        chunk_chars=2_000,
    )

    schemas = [
        json.loads(item.user)["schema_version"]
        for item in backend.invocations
    ]
    assert schemas.count("assay.review.reduce-map.v1") == 1
    assert "assay.review.reduce.v1" not in schemas
    assert verdict.findings[0].id == "reducer_accounting_incomplete"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )


def test_hierarchical_reduction_uses_bounded_multi_level_fan_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_REDUCER_ENVELOPE_CHARS", 6_000)
    backend = _HierarchicalBackend()
    corpus = Corpus(tmp_path)

    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=corpus,
        chunk_chars=2_000,
    )

    assert verdict.decision == "pass"
    assert [finding.id for finding in verdict.findings] == ["terminal_headline"]
    map_targets = [
        target for target in verdict.findings[0].targets if target.startswith("map:")
    ]
    assert len(map_targets) == 101
    reduce_requests = [
        json.loads(invocation.user)
        for invocation in backend.invocations
        if json.loads(invocation.user)["schema_version"]
        == "assay.review.reduce-map.v1"
    ]
    assert len({request["depth"] for request in reduce_requests}) >= 2
    for invocation in backend.invocations:
        request = json.loads(invocation.user)
        if request["schema_version"].startswith("assay.review.reduce"):
            assert len(invocation.system) + len(invocation.user) <= 6_000
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    completed_depths = {
        item["depth"]
        for item in manifest["reduction_ledger"]
        if item["phase"] == "reduce_map" and item["status"] == "complete"
    }
    assert len(completed_depths) >= 2
    assert manifest["reduction_ledger"][-1]["phase"] == "terminal_reduce"
    assert manifest["reduction_ledger"][-1]["status"] == "complete"
    terminal = manifest["reduction_ledger"][-1]
    assert len(terminal["reachable_leaf_report_ids"]) == 101
    assert len(set(terminal["reachable_leaf_report_ids"])) == 101
    assert terminal["frontier_report_sha256"].startswith("sha256:")


def test_malformed_intermediate_reduction_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_REDUCER_ENVELOPE_CHARS", 6_000)
    backend = _HierarchicalBackend(corrupt_first_intermediate=True)
    corpus = Corpus(tmp_path)

    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=corpus,
        chunk_chars=2_000,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reduce_map_schema_failure"
    assert "input_reports_sha256" in verdict.reasoning
    assert not any(
        json.loads(invocation.user)["schema_version"] == "assay.review.reduce.v1"
        for invocation in backend.invocations
    )
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["complete"] is False
    assert manifest["reduction_ledger"][-1]["status"] == "schema_failure"


def test_duplicate_intermediate_input_report_ids_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_REDUCER_ENVELOPE_CHARS", 6_000)

    class DuplicateInputReportBackend(_HierarchicalBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if request["schema_version"] != "assay.review.reduce-map.v1":
                return result
            payload = json.loads(result.final_message)
            payload["input_report_ids"].append(payload["input_report_ids"][0])
            return dataclasses.replace(result, final_message=json.dumps(payload))

    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=DuplicateInputReportBackend(),
        corpus=Corpus(tmp_path),
        chunk_chars=2_000,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reduce_map_schema_failure"
    assert "input_report_ids" in verdict.reasoning
    assert "duplicate" in verdict.reasoning


def test_aggregate_budget_applies_across_hierarchical_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(review_module, "_MAX_REDUCER_ENVELOPE_CHARS", 6_000)
    backend = _HierarchicalBackend()

    verdict = review(
        _oversized_artifact(),
        _make_rubric(severity_floor="High"),
        backend=backend,
        corpus=Corpus(tmp_path),
        chunk_chars=2_000,
        # 101 map calls consume 0.00101. This permits one complete reduce-map
        # call, then the next call deterministically crosses the shared cap.
        budget_usd=0.001025,
    )

    assert verdict.decision == "inconclusive"
    assert "budget" in verdict.findings[0].id
    schemas = [json.loads(item.user)["schema_version"] for item in backend.invocations]
    assert "assay.review.reduce-map.v1" in schemas
    assert "assay.review.reduce.v1" not in schemas
