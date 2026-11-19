# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.plan``.

Drives the Critic/Judge debate loop with a programmable stub backend
that returns scripted responses per call. No live LLM is invoked.
"""

from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from assay import (
    AssertionCriterion,
    BlastRadiusConstraint,
    BudgetExhausted,
    ContentRejected,
    Evidence,
    ExecutionCompleteness,
    ExternalCancellation,
    Finding,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    LifecycleEvent,
    MaxTurnsExceeded,
    Message,
    PlanPolicy,
    ProjectContext,
    ReadOnlySourceReference,
    Rubric,
    ReviewCoverage,
    ReviewDecision,
    Spec,
    TierDAG,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
    Verdict,
)
from assay.corpus.store import Corpus
from assay.canonical import hash_record, parse, serialize
from assay.corpus.codec import decode_dataclass
from assay.primitives.plan import (
    _PLANNER_SYSTEM_PROMPT,
    _active_finding_frontier,
    _active_planner_feedback,
    _canonicalize_scope_path,
    _candidate_repair_changed_fields,
    _candidate_repair_delta_finding,
    _compute_weighted_severity,
    _empty_scope_findings,
    _finding_semantic_key,
    _parse_tier_dag,
    _persistence_veto,
    _planner_user_message,
    _requirement_ref_source_findings,
    _root_requirement_sources,
    _serialize_tier_dag,
    _tier_dag_response_schema,
    _update_persistence,
    plan,
)
from assay.primitives.review import review as assay_review


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_spec(intent: str = "do the thing") -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent=intent,
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _rubric(name: str, *, severity_floor: str = "Medium") -> Rubric:
    return Rubric(
        name=name,
        system_prompt="Be terse.",
        user_template="Review: {artifact}",
        severity_floor=severity_floor,
    )


_DAG_OK = {
    "tiers": [
        {
            "tier_id": "t0",
            "depends_on": [],
            "slices": [
                {
                    "id": "s0",
                    "directive": {
                        "slice_id": "s0",
                        "intent": "implement the thing",
                        "requirement_refs": ["do the thing"],
                        "files_in_scope": ["src/thing.py"],
                        "creates_paths": [],
                        "verify_mode": "tests",
                        "blast_radius": {
                            "max_files": 1,
                            "languages": [],
                        },
                    },
                }
            ],
        }
    ]
}


def _review_response(decision: str, findings: list[dict] | None = None) -> str:
    body = {
        "schema_version": "assay.review.v2",
        "decision": decision,
        "reasoning": "ok" if decision == "pass" else "needs work",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": findings or [],
    }
    return f"```json\n{json.dumps(body)}\n```"


def _review_finding(
    fid: str,
    *,
    severity: str = "High",
    disposition: str = "confirmed",
    location: str | None = None,
    targets: tuple[str, ...] = (),
) -> dict:
    return {
        "id": fid,
        "criterion_id": "plan.clarity",
        "severity": severity,
        "confidence": 0.9,
        "disposition": disposition,
        "category": "clarity",
        "targets": list(targets),
        "evidence": [{
            "source_id": "artifact",
            "kind": "dag_excerpt",
            "location": location or fid,
            "excerpt": "tiers",
        }],
        "proposal": {"action": "be more specific", "replacement": None},
        "attack_spec": None,
    }


_VERDICT_PASS = _review_response("pass")
_VERDICT_FAIL = _review_response("revise", [_review_finding("f1")])
_INVALID_REVIEW = "not a review_v2 JSON object"


def _blast_radius_directive_path_finding(path: str) -> Finding:
    return Finding(
        id=f"out_of_blast_radius_s0_{path}",
        severity="High",
        category="blast_radius",
        evidence=(
            Evidence(
                kind="directive_path",
                location="tier_dag[t0].slices[s0]",
                excerpt=path,
            ),
            Evidence(
                kind="constraint",
                location="spec.constraints.blast_radius",
                excerpt="('src',)",
            ),
        ),
        suggested_action=(
            f"remove {path!r} from the slice's scope because the root "
            "constraint is immutable"
        ),
        source="deterministic",
    )


def _deletion_intent_finding() -> Finding:
    return Finding(
        id="delete_outside_radius_s0_outside/generated.py",
        severity="Critical",
        category="deletion",
        evidence=(
            Evidence(
                kind="intent_excerpt",
                location="tier_dag[t0].slices[s0].directive.intent",
                excerpt="delete outside/generated.py",
            ),
            Evidence(
                kind="constraint",
                location="spec.constraints.blast_radius",
                excerpt="('src',)",
            ),
        ),
        suggested_action=(
            "remove the deletion from intent because the root blast radius "
            "constraint is immutable"
        ),
        source="deterministic",
    )


def _parallel_tier_shared_path_finding(path: str) -> Finding:
    return Finding(
        id=f"parallel_conflict_t0_{path}",
        severity="High",
        category="parallel_tier_conflict",
        evidence=(
            Evidence(
                kind="shared_path",
                location="tier_dag[t0]",
                excerpt=f"{path} claimed by slices ['s0', 's1']",
            ),
        ),
        suggested_action=(
            "move conflicting slices to different tiers via depends_on, or "
            "narrow their declared write scopes"
        ),
        source="deterministic",
    )


def _verdict_fail_with_ids(*ids: str) -> str:
    """Build a JSON-fenced critic verdict whose findings carry the given ids."""
    return _review_response("revise", [_review_finding(fid) for fid in ids])


class ScriptedBackend:
    """Backend stub that returns canned responses in order."""

    name = "stub"
    model = "stub-model-v1"

    def __init__(self, responses: list[str | dict]):
        # Each entry: a final_message string, or a dict with
        # {"final": ..., "error": ...}
        self._responses = list(responses)
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        if not self._responses:
            raise AssertionError(
                "ScriptedBackend exhausted: more invocations than scripted "
                f"responses (call {len(self.invocations)})"
            )
        entry = self._responses.pop(0)
        if isinstance(entry, str):
            final, error = entry, None
            usage = Usage(input_tokens=1, output_tokens=2, cost_usd=0.01)
            completeness = ExecutionCompleteness()
            pricing_evidence = None
        else:
            final = entry.get("final", "")
            error = entry.get("error")
            cost = float(entry.get("cost", 0.01))
            usage = entry.get(
                "usage",
                Usage(input_tokens=1, output_tokens=2, cost_usd=cost),
            )
            completeness = entry.get(
                "completeness",
                ExecutionCompleteness(),
            )
            pricing_evidence = entry.get("pricing_evidence")
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=usage,
            stop_reason="end_turn",
            error=error,
            completeness=completeness,
            pricing_evidence=pricing_evidence,
        )


_COMPLETE_ACCOUNTING = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=False,
)


def _priced_entry(
    final: str,
    *segments: UsagePricingSegment,
) -> dict[str, object]:
    evidence = UsagePricingEvidence(segments=segments, complete=True)
    return {
        "final": final,
        "usage": evidence.usage,
        "completeness": _COMPLETE_ACCOUNTING,
        "pricing_evidence": evidence,
    }


def _plan_with_scripts(
    spec: Spec,
    corpus: Corpus,
    *,
    planner: list[str | dict],
    critic: list[str | dict],
    judge: list[str | dict] | None = None,
    policy: PlanPolicy,
    candidate_transform=None,
    candidate_validator=None,
    candidate_context_provider=None,
    max_candidate_validation_repairs: int = 0,
    budget_usd: float | None = None,
    review_fn=assay_review,
):
    planner_backend = ScriptedBackend(planner)
    critic_backend = ScriptedBackend(critic)
    judge_backend = ScriptedBackend(judge or []) if judge is not None else None
    outcome = plan(
        spec,
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=judge_backend,
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge") if judge is not None else None,
        policy=policy,
        candidate_transform=candidate_transform,
        candidate_validator=candidate_validator,
        candidate_context_provider=candidate_context_provider,
        max_candidate_validation_repairs=max_candidate_validation_repairs,
        budget_usd=budget_usd,
        review_fn=review_fn,
    )
    return outcome, planner_backend, critic_backend, judge_backend


def _scripted_typed_review(
    script: list[tuple[ReviewDecision, tuple[Finding, ...]]],
):
    calls: list[str] = []

    def review_fn(artifact, rubric, *, corpus, kind, **kwargs):
        del kwargs
        if not script:
            raise AssertionError("typed review script exhausted")
        decision, findings = script.pop(0)
        calls.append(kind)
        artifact_ref = corpus.put_artifact(
            artifact,
            kind="reviewed_artifacts",
        )
        return Verdict(
            rubric=rubric.name,
            decision=decision,
            findings=findings,
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="scripted typed review",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="scripted",
            transcript_ref=None,
            cost_usd=0.01,
            execution=None,
            kind=kind,
        )

    return calls, review_fn


def _fixed_candidate() -> TierDAG:
    candidate = _parse_tier_dag(_DAG_OK)
    assert candidate is not None
    return candidate


def test_fixed_candidate_accepts_without_planner_and_forces_judge(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    spec = _make_spec()
    calls, review_fn = _scripted_typed_review(
        [
            ("pass", ()),
            ("pass", ()),
        ]
    )
    critic_backend = object()
    judge_backend = object()

    outcome = plan(
        spec,
        planner_backend=None,
        candidate_dag=_fixed_candidate(),
        critic_backend=critic_backend,
        judge_backend=judge_backend,
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        policy=PlanPolicy(max_critique_rounds=9, judge_required=False),
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.accepted
    assert outcome.rounds_used == 1
    assert outcome.final_critic_decision == "pass"
    assert outcome.final_judge_decision == "pass"
    assert outcome.spec.tier_dag == outcome.candidate_dag
    assert outcome.candidate_ref is not None


def test_fixed_candidate_revision_is_one_shot_and_non_executable(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    calls, review_fn = _scripted_typed_review(
        [
            (
                "revise",
                (
                    Finding(
                        id="candidate_needs_work",
                        severity="High",
                        category="clarity",
                        evidence=(),
                        suggested_action="submit corrected candidate bytes",
                        source="llm",
                    ),
                ),
            ),
            ("pass", ()),
        ]
    )

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        policy=PlanPolicy(max_critique_rounds=20),
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.decision == "rejected"
    assert outcome.status == "candidate_revision_required"
    assert outcome.rounds_used == 1
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert outcome.candidate_ref is not None


def test_fixed_candidate_nonacceptance_preserves_prior_executable_dag(tmp_path):
    prior_payload = copy.deepcopy(_DAG_OK)
    prior_payload["tiers"][0]["slices"][0]["directive"]["files_in_scope"] = [
        "src/prior.py"
    ]
    prior_dag = _parse_tier_dag(prior_payload)
    assert prior_dag is not None
    candidate = _fixed_candidate()
    assert candidate != prior_dag
    calls, review_fn = _scripted_typed_review(
        [
            (
                "revise",
                (
                    Finding(
                        id="candidate_needs_work",
                        severity="High",
                        category="clarity",
                        evidence=(),
                        suggested_action="submit corrected candidate bytes",
                        source="llm",
                    ),
                ),
            ),
            ("pass", ()),
        ]
    )

    outcome = plan(
        dataclasses.replace(_make_spec(), tier_dag=prior_dag),
        candidate_dag=candidate,
        critic_backend=object(),
        judge_backend=object(),
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag == prior_dag
    assert outcome.spec.tier_dag != candidate
    assert outcome.candidate_dag == candidate


@pytest.mark.parametrize(
    "decisions",
    [
        ("inconclusive", "pass"),
        ("pass", "inconclusive"),
    ],
)
def test_fixed_candidate_runs_both_roles_before_failure_breakers(
    tmp_path,
    decisions,
):
    corpus = Corpus(tmp_path / "corpus")
    calls, review_fn = _scripted_typed_review(
        [
            (decisions[0], ()),
            (decisions[1], ()),
        ]
    )

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        policy=PlanPolicy(
            max_critic_failures=1,
            max_judge_failures=1,
        ),
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.decision == "inconclusive"
    assert outcome.status == "candidate_review_inconclusive"
    assert outcome.spec.tier_dag is None


@pytest.mark.parametrize(
    "decisions",
    [
        ("reject", "inconclusive"),
        ("inconclusive", "reject"),
    ],
)
def test_fixed_candidate_reject_dominates_inconclusive_role_order(
    tmp_path,
    decisions,
):
    calls, review_fn = _scripted_typed_review(
        [
            (decisions[0], ()),
            (decisions[1], ()),
        ]
    )

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.decision == "rejected"
    assert outcome.status == "candidate_rejected"
    assert outcome.spec.tier_dag is None


def test_fixed_candidate_deterministic_rejection_precedes_review(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    calls, review_fn = _scripted_typed_review([])
    finding = Finding(
        id="fixed_candidate_blocker",
        severity="High",
        category="scope",
        evidence=(),
        suggested_action="correct the candidate",
        source="deterministic",
    )

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        candidate_validator=lambda _candidate: (finding,),
        review_fn=review_fn,
    )

    assert calls == []
    assert outcome.decision == "rejected"
    assert outcome.status == "candidate_validation_rejected"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert outcome.candidate_ref is not None


def test_fixed_candidate_builtin_gates_are_terminal_before_review(tmp_path):
    oversized = _parse_tier_dag(_dag_with_slice_count(17))
    conflicting = _parse_tier_dag(_DAG_CONFLICT)
    empty_payload = copy.deepcopy(_DAG_OK)
    empty_payload["tiers"][0]["slices"][0]["directive"]["files_in_scope"] = []
    empty_scope = _parse_tier_dag(empty_payload)
    bad_ref_payload = copy.deepcopy(_DAG_OK)
    bad_ref_payload["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = ["not copied from root authority"]
    bad_requirement_ref = _parse_tier_dag(bad_ref_payload)

    cases = (
        (oversized, PlanPolicy(max_total_slices=16)),
        (conflicting, PlanPolicy()),
        (
            empty_scope,
            PlanPolicy(max_verification_only_empty_scope_slices=0),
        ),
        (bad_requirement_ref, PlanPolicy()),
    )
    for index, (candidate, policy) in enumerate(cases):
        assert candidate is not None
        calls, review_fn = _scripted_typed_review([])
        outcome = plan(
            _make_spec(),
            candidate_dag=candidate,
            critic_backend=object(),
            judge_backend=object(),
            corpus=Corpus(tmp_path / f"corpus-{index}"),
            critic_rubric=_rubric("critic"),
            judge_rubric=_rubric("judge"),
            policy=policy,
            review_fn=review_fn,
        )

        assert calls == []
        assert outcome.status == "candidate_validation_rejected"
        assert outcome.spec.tier_dag is None


def test_fixed_candidate_transform_is_reviewed_and_returned(tmp_path):
    transformed_payload = copy.deepcopy(_DAG_OK)
    transformed_payload["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/transformed.py"]
    transformed = _parse_tier_dag(transformed_payload)
    assert transformed is not None
    calls, review_fn = _scripted_typed_review([
        ("pass", ()),
        ("pass", ()),
    ])

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        candidate_transform=lambda _candidate: transformed,
        review_fn=review_fn,
    )

    assert calls == ["critic", "judge"]
    assert outcome.accepted
    assert outcome.candidate_dag == transformed
    assert outcome.spec.tier_dag == transformed


def test_fixed_candidate_transform_failure_precedes_review(tmp_path):
    calls, review_fn = _scripted_typed_review([])

    def _fail_transform(_candidate):
        raise RuntimeError("transform failed")

    outcome = plan(
        _make_spec(),
        candidate_dag=_fixed_candidate(),
        critic_backend=object(),
        judge_backend=object(),
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        candidate_transform=_fail_transform,
        review_fn=review_fn,
    )

    assert calls == []
    assert outcome.status == "candidate_transform_failed"
    assert outcome.spec.tier_dag is None


def test_fixed_candidate_requires_exactly_one_candidate_source(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    spec = _make_spec()

    with pytest.raises(
        ValueError,
        match="exactly one of planner_backend and candidate_dag",
    ):
        plan(
            spec,
            critic_backend=object(),
            judge_backend=object(),
            corpus=corpus,
            critic_rubric=_rubric("critic"),
            judge_rubric=_rubric("judge"),
        )

    with pytest.raises(
        ValueError,
        match="exactly one of planner_backend and candidate_dag",
    ):
        plan(
            spec,
            planner_backend=object(),
            candidate_dag=_fixed_candidate(),
            critic_backend=object(),
            judge_backend=object(),
            corpus=corpus,
            critic_rubric=_rubric("critic"),
            judge_rubric=_rubric("judge"),
        )


def test_fixed_candidate_rejects_planner_repair_and_tool_authority(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    kwargs = {
        "spec": _make_spec(),
        "candidate_dag": _fixed_candidate(),
        "critic_backend": object(),
        "judge_backend": object(),
        "corpus": corpus,
        "critic_rubric": _rubric("critic"),
        "judge_rubric": _rubric("judge"),
    }

    with pytest.raises(ValueError, match="does not permit candidate validation"):
        plan(**kwargs, max_candidate_validation_repairs=1)
    with pytest.raises(ValueError, match="does not permit Planner tools"):
        plan(**kwargs, allowed_tools=("Read",))
    with pytest.raises(ValueError, match="does not permit a Planner ToolSet"):
        plan(**kwargs, prompt_toolset=object())


def test_fixed_candidate_requires_independent_judge_even_when_policy_disables_it(
    tmp_path,
):
    with pytest.raises(ValueError, match="independently routed judge_backend"):
        plan(
            _make_spec(),
            candidate_dag=_fixed_candidate(),
            critic_backend=object(),
            judge_backend=None,
            corpus=Corpus(tmp_path / "corpus"),
            critic_rubric=_rubric("critic"),
            judge_rubric=None,
            policy=PlanPolicy(judge_required=False),
        )


# ----------------------------------------------------------------------
# _parse_tier_dag
# ----------------------------------------------------------------------


def test_parse_tier_dag_happy():
    dag = _parse_tier_dag(_DAG_OK)
    assert dag is not None
    assert len(dag.tiers) == 1
    assert dag.tiers[0].slices[0].directive.intent == "implement the thing"


def test_parse_tier_dag_rejects_non_list_tiers():
    assert _parse_tier_dag({"tiers": "oops"}) is None


def test_parse_tier_dag_returns_none_on_missing_tier_id():
    bad = {
        "tiers": [{"slices": [{"id": "s0", "directive": {"verify_mode": "tests"}}]}]
    }
    assert _parse_tier_dag(bad) is None


def test_parse_tier_dag_handles_minimal_slice():
    minimal = {
        "tiers": [
            {
                "tier_id": "t0",
                "depends_on": [],
                "slices": [{"id": "s0", "directive": {}}],
            }
        ]
    }
    dag = _parse_tier_dag(minimal)
    assert dag is not None
    assert dag.tiers[0].slices[0].directive.verify_mode == "tests"  # default


_WRONG_SHAPE_MUTATIONS = (
    "payload_is_a_list",
    "slices_is_a_string",
    "slices_is_an_object",
    "slice_is_a_string",
    "directive_is_prose",
    "directive_is_a_list",
    "blast_radius_is_a_string",
    "blast_radius_is_a_list",
    "blast_radius_is_a_pair_list",
    "files_in_scope_is_a_string",
    "files_in_scope_is_an_object",
    "creates_paths_is_a_string",
    "requirement_refs_is_a_string",
    "depends_on_is_a_string",
    "depends_on_is_an_object",
    "files_in_scope_member_is_a_list",
    "creates_paths_member_is_an_object",
    "requirement_refs_member_is_an_object",
    "depends_on_member_is_a_list",
    "string_member_is_not_utf8_encodable",
    "blast_radius_is_nested_beyond_the_recursion_limit",
)


def _wrong_shape_payload(mutation: str):
    """Return ``_DAG_OK`` with one parseable but structurally wrong value."""

    payload = copy.deepcopy(_DAG_OK)
    tier = payload["tiers"][0]
    slice_raw = tier["slices"][0]
    directive = slice_raw["directive"]
    if mutation == "payload_is_a_list":
        return [payload]
    if mutation == "slices_is_a_string":
        tier["slices"] = "s0"
    elif mutation == "slices_is_an_object":
        tier["slices"] = {"id": "s0"}
    elif mutation == "slice_is_a_string":
        tier["slices"] = ["oops"]
    elif mutation == "directive_is_prose":
        slice_raw["directive"] = "do it"
    elif mutation == "directive_is_a_list":
        slice_raw["directive"] = ["x"]
    elif mutation == "blast_radius_is_a_string":
        directive["blast_radius"] = "small"
    elif mutation == "blast_radius_is_a_list":
        directive["blast_radius"] = ["src"]
    elif mutation == "blast_radius_is_a_pair_list":
        directive["blast_radius"] = [["max_files", 1]]
    elif mutation == "files_in_scope_is_a_string":
        directive["files_in_scope"] = "src/a.py"
    elif mutation == "files_in_scope_is_an_object":
        directive["files_in_scope"] = {"a": 1}
    elif mutation == "creates_paths_is_a_string":
        directive["creates_paths"] = "src/new.py"
    elif mutation == "requirement_refs_is_a_string":
        directive["requirement_refs"] = "do"
    elif mutation == "depends_on_is_a_string":
        tier["depends_on"] = "t0"
    elif mutation == "depends_on_is_an_object":
        tier["depends_on"] = {"t0": 1}
    elif mutation == "files_in_scope_member_is_a_list":
        directive["files_in_scope"] = [["src/thing.py"]]
    elif mutation == "creates_paths_member_is_an_object":
        directive["creates_paths"] = [{"path": "src/new.py"}]
    elif mutation == "requirement_refs_member_is_an_object":
        directive["requirement_refs"] = [{"text": "do the thing"}]
    elif mutation == "depends_on_member_is_a_list":
        tier["depends_on"] = [["t0"]]
    elif mutation == "string_member_is_not_utf8_encodable":
        # What ``json.loads`` yields for the pure-ASCII escape ``\ud800``.
        directive["intent"] = "lone surrogate \ud800"
    elif mutation == "blast_radius_is_nested_beyond_the_recursion_limit":
        nested: dict = {"leaf": 1}
        for _ in range(sys.getrecursionlimit() * 5):
            nested = {"n": nested}
        directive["blast_radius"] = {"max_files": 1, "deep": nested}
    else:  # pragma: no cover - guards the parametrisation itself
        raise AssertionError(mutation)
    return payload


@pytest.mark.parametrize("mutation", _WRONG_SHAPE_MUTATIONS)
def test_planner_wrong_shape_parse_is_closed(mutation):
    """Parseable JSON of the wrong shape is no candidate and never raises.

    A bare string must not be split into characters, an object must not be
    read as its keys, a container must not be stringified into a path or id,
    and a value that cannot be persisted as UTF-8 must not reach review.
    """

    assert _parse_tier_dag(_wrong_shape_payload(mutation)) is None


def test_parse_tier_dag_keeps_absent_and_null_lists_empty():
    """The lenient defaults for absent, null and empty list fields stay."""

    payload = copy.deepcopy(_DAG_OK)
    tier = payload["tiers"][0]
    directive = tier["slices"][0]["directive"]
    tier["depends_on"] = None
    directive["files_in_scope"] = None
    directive["creates_paths"] = None
    directive["requirement_refs"] = None
    directive["blast_radius"] = None

    dag = _parse_tier_dag(payload)

    assert dag is not None
    assert dag.tiers[0].depends_on == ()
    parsed = dag.tiers[0].slices[0].directive
    assert parsed.files_in_scope == ()
    assert parsed.creates_paths == ()
    assert parsed.requirement_refs == ()
    assert parsed.blast_radius == {}
    del tier["depends_on"]
    del directive["files_in_scope"]
    assert _parse_tier_dag(payload) == dag


def test_parse_tier_dag_still_coerces_scalar_members_to_text():
    """Scalar ids and list members keep their ``str()`` coercion."""

    payload = copy.deepcopy(_DAG_OK)
    tier = payload["tiers"][0]
    tier["tier_id"] = 0
    payload["tiers"].append({"tier_id": 1, "depends_on": [0], "slices": []})

    dag = _parse_tier_dag(payload)

    assert dag is not None
    assert [parsed.tier_id for parsed in dag.tiers] == ["0", "1"]
    assert dag.tiers[1].depends_on == ("0",)


_SINGLE_ROUND_RETRY_POLICY = PlanPolicy(
    max_critique_rounds=1,
    judge_required=False,
    breaker_required=False,
    persistence_veto_threshold=10,
)


@pytest.mark.parametrize(
    "mutation",
    ["slice_is_a_string", "string_member_is_not_utf8_encodable"],
)
def test_plan_feeds_wrong_shape_planner_reply_back_as_parse_failure(
    tmp_path: Path,
    mutation: str,
):
    """A wrong-shape reply costs one attempt; it never escapes ``plan()``."""

    bad_reply = json.dumps(_wrong_shape_payload(mutation))
    assert bad_reply.isascii()
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[bad_reply, json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=_SINGLE_ROUND_RETRY_POLICY,
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag == _parse_tier_dag(_DAG_OK)
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1
    feedback = planner_backend.invocations[1].user
    assert "planner_parse_failure" in feedback
    assert "did not match the TierDAG schema" in feedback
    assert outcome.cost_usd == pytest.approx(0.03)


@pytest.mark.parametrize(
    "case",
    [
        "nesting_beyond_the_recursion_limit",
        "blast_radius_nested_beyond_the_recursion_limit",
        "integer_beyond_the_digit_limit",
    ],
)
def test_plan_feeds_unloadable_planner_json_back_as_parse_failure(
    tmp_path: Path,
    case: str,
):
    """``json.loads`` failures that are not ``JSONDecodeError`` stay contained."""

    depth = sys.getrecursionlimit() * 100
    if case == "nesting_beyond_the_recursion_limit":
        bad_reply = '{"tiers": ' + "[" * depth + "]" * depth + "}"
    elif case == "blast_radius_nested_beyond_the_recursion_limit":
        marker = '"blast_radius": {'
        bad_reply = json.dumps(_DAG_OK)
        assert marker in bad_reply
        bad_reply = bad_reply.replace(
            marker,
            marker + '"deep": ' + '{"n": ' * depth + "1" + "}" * depth + ", ",
        )
    else:
        digit_limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        if not digit_limit:
            pytest.skip("this interpreter does not limit integer literals")
        bad_reply = json.dumps(_DAG_OK).replace(
            '"tier_id": "t0"', '"tier_id": ' + "1" * (digit_limit + 1)
        )
        assert bad_reply != json.dumps(_DAG_OK)
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[bad_reply, json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=_SINGLE_ROUND_RETRY_POLICY,
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag == _parse_tier_dag(_DAG_OK)
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1
    assert "planner_parse_failure" in planner_backend.invocations[1].user


def test_plan_never_reviews_a_bare_string_scope_split_into_characters(
    tmp_path: Path,
):
    bare_string_scope = copy.deepcopy(_DAG_OK)
    bare_string_scope["tiers"][0]["slices"][0]["directive"]["files_in_scope"] = (
        "src/thing.py"
    )
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(bare_string_scope), json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=_SINGLE_ROUND_RETRY_POLICY,
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag is not None
    directive = outcome.spec.tier_dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == ("src/thing.py",)
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1


# ----------------------------------------------------------------------
# _canonicalize_scope_path — strip a workspace/project-name prefix so the
# planner's paths are project-root-relative (the cwd the build runs from).
# ----------------------------------------------------------------------


def test_canonicalize_strips_leading_project_prefix():
    # The planner sometimes prefixes with the project/submodule name; strip it.
    assert (
        _canonicalize_scope_path("fabric-usd/crates/foo.rs", "fabric-usd")
        == "crates/foo.rs"
    )


def test_canonicalize_noop_when_already_relative():
    assert _canonicalize_scope_path("crates/foo.rs", "fabric-usd") == "crates/foo.rs"


def test_canonicalize_noop_on_empty_prefix():
    assert _canonicalize_scope_path("crates/foo.rs", "") == "crates/foo.rs"


def test_canonicalize_keeps_real_nested_dir_prefix(tmp_path):
    # Doubled kit/kit layout: the submodule root `kit/` holds an inner `kit/`.
    # `kit/source/foo.cpp` is a REAL path (resolves to <base>/kit/source/foo.cpp);
    # stripping the `kit/` prefix would point downstream rules at a nonexistent
    # `source/foo.cpp`. With base_dir given and the inner `kit/` on disk, keep it.
    base = tmp_path / "kit"  # project root = <workspace>/<submodule>
    (base / "kit" / "source").mkdir(parents=True)
    (base / "kit" / "source" / "foo.cpp").write_text("x")
    assert (
        _canonicalize_scope_path("kit/source/foo.cpp", "kit", str(base))
        == "kit/source/foo.cpp"
    )


@pytest.mark.parametrize(
    "path",
    [
        "kit/source/new.cpp",  # a new file beside an existing one
        "kit/fresh/new.cpp",  # a new file in a new directory
        "kit/rendering/bar.cpp",  # exists only as <base>/rendering/bar.cpp
    ],
)
def test_canonicalize_keeps_every_path_under_real_nested_dir(tmp_path, path):
    # The same-named directory is real, so the segment is part of a
    # project-root-relative path whether or not the file exists yet. A path
    # that only repeats the project name is not guessed at and rewritten here.
    base = tmp_path / "kit"
    (base / "kit" / "source").mkdir(parents=True)
    (base / "rendering").mkdir()
    (base / "rendering" / "bar.cpp").write_text("x")
    assert _canonicalize_scope_path(path, "kit", str(base)) == path


def test_canonicalize_strips_when_base_has_no_same_named_dir(tmp_path):
    # fabric-usd case: <base>/fabric-usd does not exist (only crates/ does),
    # so the leading prefix IS a redundant project-name prefix — strip it.
    base = tmp_path / "fabric-usd"
    (base / "crates").mkdir(parents=True)
    assert (
        _canonicalize_scope_path("fabric-usd/crates/foo.rs", "fabric-usd", str(base))
        == "crates/foo.rs"
    )


def test_canonicalize_strips_when_same_named_entry_is_a_regular_file(tmp_path):
    # Nothing can live below a regular file, so the prefix repeats the name.
    base = tmp_path / "kit"
    base.mkdir()
    (base / "kit").write_text("x")
    assert (
        _canonicalize_scope_path("kit/source/foo.cpp", "kit", str(base))
        == "source/foo.cpp"
    )


def test_canonicalize_safe_when_prefix_is_namespaced():
    # swebench sets project="swebench/<repo>"; a real source path like
    # "django/db/models.py" must NOT be stripped (it doesn't start with the
    # full namespaced prefix) — guards the project-name-collision footgun.
    assert (
        _canonicalize_scope_path("django/db/models.py", "swebench/django/django")
        == "django/db/models.py"
    )


def test_canonicalize_only_strips_full_prefix_segment():
    # A prefix that is a substring but not a full leading path segment is left.
    assert _canonicalize_scope_path("fabric-usd-extra/x.rs", "fabric-usd") == (
        "fabric-usd-extra/x.rs"
    )


@pytest.mark.parametrize("path", ["/etc/passwd", r"C:\Windows\system.ini"])
def test_canonicalize_preserves_absolute_scope_for_consumer_rejection(path):
    assert _canonicalize_scope_path(path, "project") == path.replace("\\", "/")


def test_parse_tier_dag_canonicalizes_scope_paths():
    dag = _parse_tier_dag(
        {
            "tiers": [
                {
                    "tier_id": "t0",
                    "depends_on": [],
                    "slices": [
                        {
                            "id": "s0",
                            "directive": {
                                "files_in_scope": ["fabric-usd/crates/a.rs"],
                                "creates_paths": ["fabric-usd/crates/b.rs"],
                            },
                        }
                    ],
                }
            ]
        },
        strip_prefix="fabric-usd",
    )
    assert dag is not None
    d = dag.tiers[0].slices[0].directive
    assert d.files_in_scope == ("crates/a.rs",)
    assert d.creates_paths == ("crates/b.rs",)


def _same_named_nested_project(tmp_path: Path) -> tuple[Spec, Path, Path]:
    """Return ``(spec, workspace, project_root)`` for a doubled ``kit/kit`` tree.

    The project root is ``<workspace>/kit``. It holds a real nested ``kit``
    directory, so every ``kit/...`` path is project-relative as written.
    ``rendering/bar.cpp`` exists only directly below the project root.
    """

    workspace = tmp_path / "ws"
    project_root = workspace / "kit"
    (project_root / "kit" / "source").mkdir(parents=True)
    (project_root / "kit" / "source" / "foo.cpp").write_text("x")
    (project_root / "rendering").mkdir()
    (project_root / "rendering" / "bar.cpp").write_text("x")
    base = _make_spec()
    spec = dataclasses.replace(
        base,
        project_context=dataclasses.replace(
            base.project_context, project="kit", submodule_path="kit"
        ),
    )
    return spec, workspace, project_root


def _plan_scope_paths(
    spec: Spec,
    corpus_root: Path,
    *,
    files_in_scope: list[str],
    creates_paths: list[str] | None = None,
    **plan_kwargs,
):
    """Plan one accepted round whose planner reply carries the given scopes."""

    payload = copy.deepcopy(_DAG_OK)
    directive = payload["tiers"][0]["slices"][0]["directive"]
    directive["files_in_scope"] = files_in_scope
    directive["creates_paths"] = creates_paths or []
    planner_backend = ScriptedBackend([json.dumps(payload)])
    critic_backend = ScriptedBackend([_VERDICT_PASS])
    outcome = plan(
        spec,
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=None,
        corpus=Corpus(corpus_root),
        critic_rubric=_rubric("critic"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        **plan_kwargs,
    )
    return outcome, planner_backend, critic_backend


@pytest.mark.parametrize("root_type", [str, Path])
def test_plan_canonicalizes_scope_against_project_root_from_any_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_type,
):
    """A real same-named directory keeps the segment on every scope path.

    The existing file, the new file beside it, and the new file in a new
    directory stay together below ``kit/``. ``kit/rendering/bar.cpp`` names
    nothing (the file is ``rendering/bar.cpp``); it is kept as written so the
    consumer's deterministic scope gate reports it, never silently rewritten.
    """

    spec, _, project_root = _same_named_nested_project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    outcome, _, _ = _plan_scope_paths(
        spec,
        tmp_path / "corpus",
        files_in_scope=["kit/source/foo.cpp", "kit/rendering/bar.cpp"],
        creates_paths=["kit/source/new.cpp", "kit/fresh/new.cpp"],
        project_root=root_type(project_root),
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag is not None
    directive = outcome.spec.tier_dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == (
        "kit/source/foo.cpp",
        "kit/rendering/bar.cpp",
    )
    assert directive.creates_paths == ("kit/source/new.cpp", "kit/fresh/new.cpp")


def test_plan_strips_repeated_project_name_when_project_root_has_no_such_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without a same-named directory the segment can only repeat the name."""

    spec, workspace, _ = _same_named_nested_project(tmp_path)
    flat_root = tmp_path / "flat" / "kit"
    (flat_root / "rendering").mkdir(parents=True)
    (flat_root / "rendering" / "bar.cpp").write_text("x")
    # The process cwd holds ``kit/kit``; only the explicit root decides.
    monkeypatch.chdir(workspace)

    outcome, _, _ = _plan_scope_paths(
        spec,
        tmp_path / "corpus",
        files_in_scope=["kit/rendering/bar.cpp"],
        creates_paths=["kit/rendering/new.cpp"],
        project_root=flat_root,
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag is not None
    directive = outcome.spec.tier_dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == ("rendering/bar.cpp",)
    assert directive.creates_paths == ("rendering/new.cpp",)


@pytest.mark.parametrize("process_cwd", ["workspace", "project_root"])
def test_plan_without_project_root_never_probes_the_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_cwd: str,
):
    """Without a root there is no proof, whatever the process cwd holds.

    Each natural cwd fallback would keep the segment from one of these
    directories: a ``<cwd>/<prefix>`` base finds ``kit/kit`` from the
    workspace, a plain ``os.getcwd()`` base finds it from the project root.
    """

    spec, workspace, project_root = _same_named_nested_project(tmp_path)
    monkeypatch.chdir(
        {"workspace": workspace, "project_root": project_root}[process_cwd]
    )

    outcome, _, _ = _plan_scope_paths(
        spec,
        tmp_path / "corpus",
        files_in_scope=["kit/source/foo.cpp"],
        creates_paths=["kit/source/new.cpp"],
    )

    assert outcome.decision == "accepted"
    assert outcome.spec.tier_dag is not None
    directive = outcome.spec.tier_dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == ("source/foo.cpp",)
    assert directive.creates_paths == ("source/new.cpp",)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("relative", ValueError),
        ("empty", ValueError),
        ("regular_file", ValueError),
        ("missing", ValueError),
        ("not_a_path", TypeError),
        ("bytes_path", TypeError),
    ],
)
def test_plan_rejects_invalid_project_root_before_any_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: type[Exception],
):
    spec, workspace, project_root = _same_named_nested_project(tmp_path)
    # A relative root would silently resolve against this directory.
    monkeypatch.chdir(workspace)
    root: object = {
        "relative": "kit",
        "empty": "",
        "regular_file": str(project_root / "rendering" / "bar.cpp"),
        "missing": str(tmp_path / "missing"),
        "not_a_path": 3,
        "bytes_path": bytes(project_root),
    }[case]
    planner_backend = ScriptedBackend([json.dumps(_DAG_OK)])
    critic_backend = ScriptedBackend([_VERDICT_PASS])

    with pytest.raises(error, match="project_root "):
        plan(
            spec,
            planner_backend=planner_backend,
            critic_backend=critic_backend,
            judge_backend=None,
            corpus=Corpus(tmp_path / "corpus"),
            critic_rubric=_rubric("critic"),
            policy=PlanPolicy(max_critique_rounds=0, judge_required=False),
            project_root=root,
        )

    assert planner_backend.invocations == []
    assert critic_backend.invocations == []


# ----------------------------------------------------------------------
# _planner_user_message
# ----------------------------------------------------------------------


def test_planner_user_message_includes_intent():
    msg = _planner_user_message(_make_spec("custom intent here"))
    assert "custom intent here" in msg
    assert "Produce the initial" in msg


def test_planner_user_message_renders_exact_requirement_source_catalog():
    spec = dataclasses.replace(
        _make_spec('preserve "quoted" intent\nwithout a wrapper'),
        constraints=(
            BlastRadiusConstraint(no_edits_outside=("alias_resolution.py",)),
        ),
        done_criteria=(
            AssertionCriterion(test_name="test_alias_resolution", must_pass=True),
        ),
    )

    msg = _planner_user_message(spec)
    catalog_line = next(
        line for line in msg.splitlines() if line.startswith('{"intent":')
    )

    assert json.loads(catalog_line) == dict(_root_requirement_sources(spec))
    assert "source-id to exact-citable-text catalog" in msg
    assert "Copy only an exact contiguous excerpt from a JSON string value" in msg
    assert "Never copy a source id, JSON syntax, Markdown label" in msg
    assert "- `blast_radius`: BlastRadiusConstraint" not in msg


def test_requirement_source_catalog_has_bounded_linear_rendering():
    intent = "x" * 10_000

    msg = _planner_user_message(_make_spec(intent))

    assert msg.count(intent) == 1
    assert len(msg) < len(intent) + 2_000


def test_planner_system_prompt_requires_root_semantic_fidelity():
    prompt = " ".join(_PLANNER_SYSTEM_PROMPT.split())

    assert "Every observable behavior claimed anywhere in the plan" in prompt
    assert "MUST be entailed" in prompt
    for preserved_dimension in (
        "ordering and precedence",
        "quantifiers",
        "modality",
        "scope and conditions",
        "replacement-versus-combination",
    ):
        assert preserved_dimension in prompt
    assert "keep that ambiguity explicit in the directive" in prompt
    assert "Do not silently choose one interpretation" in prompt
    assert "proposed test or documentation expectations" in prompt
    assert '"requirement_refs"' in prompt
    assert prompt.count("## Required output contract") == 1
    assert "serial within one Tier checkout" in prompt
    assert "execute in parallel" not in prompt
    assert "immutable source linkage" in prompt


def test_planner_user_message_includes_critic_feedback_on_revision():
    from assay import Finding
    f = Finding(
        id="missing_tests",
        severity="High",
        category="completeness",
        evidence=(),
        suggested_action="add tests for slice X",
        source="llm",
    )
    msg = _planner_user_message(_make_spec(), prior_critic_findings=(f,))
    assert "missing_tests" in msg
    assert "add tests for slice X" in msg
    assert "Revise" in msg


def test_planner_user_message_includes_previous_canonical_dag_and_bounds():
    dag = _parse_tier_dag(_DAG_OK)
    assert dag is not None

    msg = _planner_user_message(
        _make_spec(),
        previous_dag=dag,
        slice_cap=8,
        tier_cap=6,
        compaction_required=True,
        planner_guidance="Use independently verifiable increments.",
    )

    assert "Previous canonical TierDAG" in msg
    assert '"files_in_scope": [' in msg
    assert '"src/thing.py"' in msg
    assert "at most 8 total slices" in msg
    assert "at most 6 total tiers" in msg
    assert "single bounded compaction repair" in msg
    assert "independently verifiable increments" in msg


def test_active_planner_feedback_excludes_terminal_and_collapses_lineage():
    def finding(
        fid: str,
        *,
        disposition: str = "confirmed",
        targets: tuple[str, ...] = (),
        location: str = "t0:s0",
    ) -> Finding:
        return Finding(
            id=fid,
            severity="High",
            category="clarity",
            criterion_id="plan.clarity",
            evidence=(
                Evidence(
                    source_id="artifact",
                    kind="dag_excerpt",
                    location=location,
                    excerpt="same grounded defect",
                ),
            ),
            suggested_action="split the independently verifiable increments",
            source="llm",
            disposition=disposition,
            targets=targets,
        )

    renamed = finding("renamed-defect")
    addressed = finding(
        "addressed-history", disposition="addressed", location="t1:s1"
    )
    suppressed = dataclasses.replace(
        finding("suppressed", location="t2:s2"), suppressed=True
    )
    critic, judge = _active_planner_feedback(
        (finding("original-defect"), renamed, addressed, suppressed), ()
    )

    assert critic == (renamed,)
    assert judge == ()


def test_judge_target_supersedes_critic_feedback_lineage():
    critic_finding = Finding(
        id="too-broad",
        severity="High",
        category="clarity",
        evidence=(),
        suggested_action="split it",
        source="llm",
    )
    judge_closure = Finding(
        id="closure",
        severity="High",
        category="clarity",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="addressed",
        targets=(
            "prior:critic:0:sha256-"
            + "a" * 64
            + ":too-broad",
        ),
    )

    assert _active_planner_feedback(
        (critic_finding,), (judge_closure,)
    ) == ((), ())


def test_semantic_lineage_ignores_mutable_spans_but_keeps_distinct_actions():
    original = Finding(
        id="old-name",
        severity="High",
        category="clarity",
        criterion_id="plan.clarity",
        evidence=(Evidence(
            source_id="artifact",
            kind="dag_excerpt",
            location="chars:10-20",
            excerpt="old candidate slice",
        ),),
        suggested_action="split the independently verifiable increments",
        source="llm",
    )
    revised_span = dataclasses.replace(
        original,
        id="new-name",
        evidence=(Evidence(
            source_id="artifact",
            kind="dag_excerpt",
            location="chars:80-104",
            excerpt="rewritten candidate slice",
        ),),
    )
    separate_defect = dataclasses.replace(
        revised_span,
        suggested_action="add the missing dependency edge",
    )

    assert _finding_semantic_key(original) == _finding_semantic_key(revised_span)
    assert _finding_semantic_key(original) != _finding_semantic_key(separate_defect)


def test_cross_role_bare_id_reuse_does_not_close_critic_lineage():
    critic_finding = Finding(
        id="generic-defect",
        severity="High",
        category="clarity",
        evidence=(Evidence(
            source_id="artifact",
            kind="dag_excerpt",
            location="t0:s0",
            excerpt="critic defect",
        ),),
        suggested_action="split it",
        source="llm",
    )
    unrelated_judge_history = Finding(
        id="generic-defect",
        severity="High",
        category="correctness",
        evidence=(Evidence(
            source_id="artifact",
            kind="dag_excerpt",
            location="t1:s1",
            excerpt="different judge observation",
        ),),
        suggested_action=None,
        source="llm",
        disposition="addressed",
    )

    assert _active_planner_feedback(
        (critic_finding,), (unrelated_judge_history,)
    ) == ((critic_finding,), ())

    active_same_shape = dataclasses.replace(
        critic_finding,
        id="judge-observation",
    )
    assert _active_planner_feedback(
        (critic_finding,), (active_same_shape,)
    ) == ((critic_finding,), (active_same_shape,))


def test_malformed_or_noncritic_target_cannot_close_critic_lineage():
    critic_finding = Finding(
        id="too-broad",
        severity="High",
        category="clarity",
        evidence=(),
        suggested_action="split it",
        source="llm",
    )
    for target in (
        "prior:critic:0:not-a-digest:too-broad",
        "prior:judge:0:sha256-" + "a" * 64 + ":too-broad",
    ):
        judge_closure = Finding(
            id="closure",
            severity="High",
            category="clarity",
            evidence=(),
            suggested_action=None,
            source="llm",
            disposition="addressed",
            targets=(target,),
        )

        assert _active_planner_feedback(
            (critic_finding,), (judge_closure,)
        ) == ((critic_finding,), ())


def test_active_targeted_descendant_owns_one_persistence_lineage():
    critic_finding = Finding(
        id="too-broad",
        severity="High",
        category="clarity",
        evidence=(),
        suggested_action="split it",
        source="llm",
    )
    judge_descendant = Finding(
        id="still-broad",
        severity="High",
        category="clarity",
        evidence=(),
        suggested_action="separate the increments",
        source="llm",
        disposition="confirmed",
        targets=(
            "prior:critic:0:sha256-" + "a" * 64 + ":too-broad",
        ),
    )

    frontier = _active_finding_frontier(
        (critic_finding,), (judge_descendant,)
    )

    assert frontier == ((
        "judge",
        judge_descendant,
        _finding_semantic_key(critic_finding),
    ),)
    counts: dict[object, int] = {}
    _update_persistence(
        counts,
        (judge_descendant,),
        semantic_keys=(frontier[0][2],),
    )
    assert counts == {_finding_semantic_key(critic_finding): 1}


# ----------------------------------------------------------------------
# _persistence_veto
# ----------------------------------------------------------------------


def test_persistence_veto_returns_first_threshold_hit():
    counts = {"a": 1, "b": 2, "c": 3}
    veto = _persistence_veto(counts, threshold=2)
    assert veto is not None
    assert veto[1] >= 2


def test_persistence_veto_returns_none_when_below():
    assert _persistence_veto({"a": 1}, threshold=2) is None


def test_persistence_veto_returns_none_on_zero_threshold():
    assert _persistence_veto({"a": 5}, threshold=0) is None


def test_persistence_counts_active_ids_once_and_resets_inactive_state():
    active = Finding(
        id="f1",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="confirmed",
    )
    duplicate_from_other_role = Finding(
        id="f1",
        severity="High",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="suspected",
    )
    counts: dict[object, int] = {}

    _update_persistence(counts, (active, duplicate_from_other_role))
    assert counts == {_finding_semantic_key(active): 1}

    addressed = Finding(
        id="f1",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="addressed",
    )
    _update_persistence(counts, (addressed,))
    assert counts == {}

    suppressed = Finding(
        id="f1",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="confirmed",
        suppressed=True,
    )
    _update_persistence(counts, (suppressed,))
    assert counts == {}


def test_weighted_severity_ignores_resolved_and_suppressed_findings():
    weights = (("Critical", 10.0),)
    active = Finding(
        id="active",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
    )
    addressed = Finding(
        id="addressed",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="addressed",
    )
    dismissed = Finding(
        id="dismissed",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        disposition="dismissed",
    )
    suppressed = Finding(
        id="suppressed",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action=None,
        source="llm",
        suppressed=True,
    )

    assert _compute_weighted_severity(
        (addressed, dismissed, suppressed), weights
    ) == 0.0
    assert _compute_weighted_severity(
        (active, addressed, dismissed, suppressed), weights
    ) == 10.0


# ----------------------------------------------------------------------
# plan() — end-to-end happy path
# ----------------------------------------------------------------------


def test_plan_first_round_passes(tmp_path: Path):
    """Critic passes immediately; no revision needed; spec carries TierDAG."""
    corpus = Corpus(tmp_path)
    spec = _make_spec()
    outcome, *_ = _plan_with_scripts(
        spec, corpus,
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )
    new_spec = outcome.spec
    assert outcome.decision == "accepted"
    assert new_spec.tier_dag is not None
    assert len(new_spec.tier_dag.tiers) == 1
    assert new_spec.version > spec.version
    assert len(new_spec.verdict_refs) == 1  # critic only


def test_plan_aggregates_ordered_pricing_evidence_across_semantic_rounds(
    tmp_path: Path,
):
    """Logical retries and role calls remain ordered in the outcome ledger."""

    planner_retry_0 = UsagePricingSegment(11, 3, 2, 1, 0.11)
    planner_retry_1 = UsagePricingSegment(13, 5, 0, 4, 0.12)
    critic_round_0 = UsagePricingSegment(17, 7, 3, 2, 0.21)
    judge_round_0 = UsagePricingSegment(19, 11, 4, 1, 0.31)
    planner_round_1 = UsagePricingSegment(23, 13, 1, 5, 0.41)
    critic_round_1 = UsagePricingSegment(29, 17, 2, 6, 0.51)
    judge_round_1 = UsagePricingSegment(31, 19, 7, 3, 0.61)

    outcome, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            _priced_entry(
                json.dumps(_DAG_OK),
                planner_retry_0,
                planner_retry_1,
            ),
            _priced_entry(json.dumps(_DAG_OK), planner_round_1),
        ],
        critic=[
            _priced_entry(_VERDICT_FAIL, critic_round_0),
            _priced_entry(_VERDICT_PASS, critic_round_1),
        ],
        judge=[
            _priced_entry(_VERDICT_PASS, judge_round_0),
            _priced_entry(_VERDICT_PASS, judge_round_1),
        ],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    expected_segments = (
        planner_retry_0,
        planner_retry_1,
        critic_round_0,
        judge_round_0,
        planner_round_1,
        critic_round_1,
        judge_round_1,
    )
    expected_evidence = UsagePricingEvidence(
        segments=expected_segments,
        complete=True,
    )
    assert outcome.accepted
    assert outcome.rounds_used == 2
    assert outcome.usage == expected_evidence.usage
    assert outcome.cost_usd == expected_evidence.usage.cost_usd
    assert outcome.execution_completeness == _COMPLETE_ACCOUNTING
    assert outcome.pricing_evidence == expected_evidence
    assert outcome.pricing_evidence.reconciles(outcome.usage)


def test_plan_does_not_complete_pricing_evidence_when_one_call_omits_it(
    tmp_path: Path,
):
    planner_segment = UsagePricingSegment(11, 3, 2, 1, 0.11)
    critic_usage = Usage(input_tokens=17, output_tokens=7, cost_usd=0.21)

    outcome, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[_priced_entry(json.dumps(_DAG_OK), planner_segment)],
        critic=[
            {
                "final": _VERDICT_PASS,
                "usage": critic_usage,
                "completeness": _COMPLETE_ACCOUNTING,
            }
        ],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.accepted
    assert outcome.usage == Usage(
        input_tokens=28,
        output_tokens=10,
        cost_usd=0.32,
        cache_creation_tokens=2,
        cache_read_tokens=1,
    )
    assert outcome.execution_completeness == _COMPLETE_ACCOUNTING
    assert outcome.pricing_evidence == UsagePricingEvidence(
        segments=(planner_segment,),
        complete=False,
    )
    assert not outcome.pricing_evidence.reconciles(outcome.usage)


def test_plan_uses_verdict_cache_totals_when_incomplete_segments_repeat(
    tmp_path: Path,
):
    """Cumulative reviewer segments must not replace exact review totals."""

    planner_segment = UsagePricingSegment(11, 3, 2, 1, 0.11)
    cumulative_snapshot = UsagePricingSegment(17, 1, 15, 0, 0.10)
    repeated_evidence = UsagePricingEvidence(
        segments=(cumulative_snapshot, cumulative_snapshot),
        complete=False,
    )
    critic_usage = Usage(
        input_tokens=17,
        output_tokens=7,
        cost_usd=0.21,
        cache_creation_tokens=15,
    )

    outcome, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[_priced_entry(json.dumps(_DAG_OK), planner_segment)],
        critic=[{
            "final": _VERDICT_PASS,
            "usage": critic_usage,
            "completeness": _COMPLETE_ACCOUNTING,
            "pricing_evidence": repeated_evidence,
        }],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.accepted
    assert outcome.usage == Usage(
        input_tokens=28,
        output_tokens=10,
        cost_usd=0.32,
        cache_creation_tokens=17,
        cache_read_tokens=1,
    )
    assert outcome.execution_completeness == _COMPLETE_ACCOUNTING
    assert outcome.pricing_evidence == UsagePricingEvidence(
        segments=(
            planner_segment,
            cumulative_snapshot,
            cumulative_snapshot,
        ),
        complete=False,
    )
    assert outcome.pricing_evidence.usage.cache_creation_tokens > (
        outcome.usage.input_tokens
    )


def test_plan_recovers_legacy_cache_totals_only_from_complete_evidence(
    tmp_path: Path,
):
    planner_segment = UsagePricingSegment(11, 3, 2, 1, 0.11)
    legacy_segment = UsagePricingSegment(17, 7, 15, 0, 0.21)
    legacy_evidence = UsagePricingEvidence(
        segments=(legacy_segment,),
        complete=True,
    )

    def legacy_review(artifact, rubric, *, corpus, kind, **kwargs):
        del kwargs
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=corpus.put_artifact(
                artifact,
                kind="reviewed_artifacts",
            ),
            artifact_commit=None,
            reasoning="legacy complete verdict",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name="legacy",
            transcript_ref=None,
            cost_usd=0.21,
            execution=None,
            input_tokens=17,
            output_tokens=7,
            kind=kind,
            execution_completeness=_COMPLETE_ACCOUNTING,
            pricing_evidence=legacy_evidence,
        )

    outcome = plan(
        _make_spec(),
        planner_backend=ScriptedBackend([
            _priced_entry(json.dumps(_DAG_OK), planner_segment),
        ]),
        critic_backend=ScriptedBackend([]),
        judge_backend=None,
        corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        review_fn=legacy_review,
    )

    assert outcome.accepted
    assert outcome.usage == Usage(
        input_tokens=28,
        output_tokens=10,
        cost_usd=0.32,
        cache_creation_tokens=17,
        cache_read_tokens=1,
    )
    assert outcome.execution_completeness == _COMPLETE_ACCOUNTING
    assert outcome.pricing_evidence == UsagePricingEvidence(
        segments=(planner_segment, legacy_segment),
        complete=True,
    )


def test_plan_marks_legacy_cache_totals_unknown_for_incomplete_evidence(
    tmp_path: Path,
):
    planner_segment = UsagePricingSegment(11, 3, 2, 1, 0.11)
    repeated_snapshot = UsagePricingSegment(17, 1, 15, 0, 0.10)
    incomplete_evidence = UsagePricingEvidence(
        segments=(repeated_snapshot, repeated_snapshot),
        complete=False,
    )

    def legacy_review(artifact, rubric, *, corpus, kind, **kwargs):
        del kwargs
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=corpus.put_artifact(
                artifact,
                kind="reviewed_artifacts",
            ),
            artifact_commit=None,
            reasoning="legacy incomplete verdict",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name="legacy",
            transcript_ref=None,
            cost_usd=0.21,
            execution=None,
            input_tokens=17,
            output_tokens=7,
            kind=kind,
            execution_completeness=_COMPLETE_ACCOUNTING,
            pricing_evidence=incomplete_evidence,
        )

    outcome = plan(
        _make_spec(),
        planner_backend=ScriptedBackend([
            _priced_entry(json.dumps(_DAG_OK), planner_segment),
        ]),
        critic_backend=ScriptedBackend([]),
        judge_backend=None,
        corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        review_fn=legacy_review,
    )

    assert outcome.accepted
    assert outcome.usage == Usage(
        input_tokens=28,
        output_tokens=10,
        cost_usd=0.32,
        cache_creation_tokens=2,
        cache_read_tokens=1,
    )
    assert outcome.execution_completeness == dataclasses.replace(
        _COMPLETE_ACCOUNTING,
        token_usage_complete=False,
    )
    assert outcome.accounting_debt == ("token_usage_complete",)
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is False
    assert outcome.pricing_evidence.usage.cache_creation_tokens > (
        outcome.usage.input_tokens
    )


def test_plan_consensus_precedes_advisory_persistence_veto(tmp_path: Path):
    """A pass-allowed advisory cannot become a threshold-one hard veto."""

    advisory = Finding(
        id="intent_coverage.advisory_ambiguity",
        severity="Medium",
        category="ungrounded_intent",
        evidence=(Evidence(
            source_id="artifact",
            kind="directive_field",
            location="tiers[0].slices[0].directive.intent",
            excerpt="preserve behavior outside the required domain",
        ),),
        suggested_action="keep the ambiguity explicit",
        source="llm",
        criterion_id="intent_coverage",
        confidence=0.55,
        disposition="suspected",
    )
    calls, review_fn = _scripted_typed_review([
        ("pass", (advisory,)),
        ("pass", ()),
    ])
    planner_backend = ScriptedBackend([json.dumps(_DAG_OK)])
    corpus = Corpus(tmp_path)
    outcome = plan(
        _make_spec(),
        planner_backend=planner_backend,
        critic_backend=ScriptedBackend([]),
        judge_backend=ScriptedBackend([]),
        corpus=corpus,
        critic_rubric=_rubric("critic", severity_floor="High"),
        judge_rubric=_rubric("judge", severity_floor="High"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=1,
        ),
        review_fn=review_fn,
    )

    assert outcome.accepted
    assert outcome.status == "accepted"
    assert outcome.final_critic_decision == "pass"
    assert outcome.final_judge_decision == "pass"
    assert outcome.rounds_used == 1
    assert outcome.cost_usd == pytest.approx(0.03)
    assert outcome.spec.tier_dag is not None
    assert outcome.candidate_ref is not None
    assert len(outcome.spec.verdict_refs) == 2
    assert len(outcome.spec.lifecycle_event_refs) == 1
    event = corpus.get_record(
        outcome.spec.lifecycle_event_refs[0],
        as_type=LifecycleEvent,
    )
    assert isinstance(event, LifecycleEvent)
    assert event.kind == "planned"
    assert "critic + judge both passed" in event.reason
    assert len(planner_backend.invocations) == 1
    assert calls == ["critic", "judge"]


def test_plan_consensus_still_runs_weighted_judge_backstop(tmp_path: Path):
    blocker = Finding(
        id="judge.weighted_blocker",
        severity="Medium",
        category="correctness",
        evidence=(Evidence(
            source_id="artifact",
            kind="directive_field",
            location="tiers[0].slices[0].directive.intent",
            excerpt="blocking contradiction",
        ),),
        suggested_action="remove the contradiction",
        source="llm",
        criterion_id="contradiction",
        confidence=1.0,
        disposition="confirmed",
    )
    calls, review_fn = _scripted_typed_review([
        ("pass", ()),
        ("pass", (blocker,)),
    ])
    corpus = Corpus(tmp_path)
    outcome = plan(
        _make_spec(),
        planner_backend=ScriptedBackend([json.dumps(_DAG_OK)]),
        critic_backend=ScriptedBackend([]),
        judge_backend=ScriptedBackend([]),
        corpus=corpus,
        critic_rubric=_rubric("critic", severity_floor="High"),
        judge_rubric=_rubric("judge", severity_floor="High"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=1,
            judge_veto_weighted_threshold=1.0,
        ),
        review_fn=review_fn,
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "weighted_severity_veto"
    assert outcome.rounds_used == 1
    assert outcome.cost_usd == pytest.approx(0.03)
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert outcome.candidate_ref is not None
    assert len(outcome.spec.verdict_refs) == 2
    assert len(outcome.spec.lifecycle_event_refs) == 1
    event = corpus.get_record(
        outcome.spec.lifecycle_event_refs[0],
        as_type=LifecycleEvent,
    )
    assert isinstance(event, LifecycleEvent)
    assert event.kind == "plan_rejected"
    assert "weighted-severity Judge veto" in event.reason
    assert calls == ["critic", "judge"]


def test_planner_invocation_has_strict_tier_dag_response_schema(tmp_path: Path):
    outcome, planner_backend, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )

    assert outcome.accepted
    schema = planner_backend.invocations[0].response_schema
    assert schema == _tier_dag_response_schema()
    assert schema is not None
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["tiers"]
    tier_schema = schema["properties"]["tiers"]["items"]
    assert tier_schema["additionalProperties"] is False
    assert tier_schema["required"] == ["tier_id", "depends_on", "slices"]
    directive_schema = tier_schema["properties"]["slices"]["items"]["properties"][
        "directive"
    ]
    assert directive_schema["additionalProperties"] is False
    assert "requirement_refs" in directive_schema["required"]
    assert directive_schema["properties"]["requirement_refs"] == {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
    }
    assert directive_schema["properties"]["verify_mode"]["enum"] == [
        "skip",
        "build_only",
        "tests",
        "bench",
    ]
    assert directive_schema["properties"]["blast_radius"] == {
        "type": "object",
        "properties": {
            "max_files": {"type": "integer", "minimum": 0},
            "languages": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
        "required": ["max_files", "languages"],
        "additionalProperties": False,
    }


def test_requirement_refs_parse_serialize_round_trip():
    dag = _parse_tier_dag(_DAG_OK)
    assert dag is not None
    assert dag.tiers[0].slices[0].directive.requirement_refs == (
        "do the thing",
    )

    serialized = json.loads(_serialize_tier_dag(dag))
    directive = serialized["tiers"][0]["slices"][0]["directive"]
    assert directive["requirement_refs"] == ["do the thing"]
    reparsed = _parse_tier_dag(serialized)
    assert reparsed == dag


def test_plan_retries_missing_and_non_root_requirement_refs_before_review(
    tmp_path: Path,
):
    missing_ref = json.loads(json.dumps(_DAG_OK))
    del missing_ref["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ]
    non_root_ref = json.loads(json.dumps(_DAG_OK))
    non_root_ref["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = ["generated implementation claim"]

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(missing_ref),
            json.dumps(non_root_ref),
            json.dumps(_DAG_OK),
        ],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.accepted
    assert len(planner_backend.invocations) == 3
    assert len(critic_backend.invocations) == 1
    assert "missing_requirement_ref_round_0_0_0" in (
        planner_backend.invocations[1].user
    )
    assert "non_root_requirement_ref_round_1_0_0_0" in (
        planner_backend.invocations[2].user
    )
    assert "generated implementation claim" in (
        planner_backend.invocations[2].user
    )


def _spec_with_blast_radius_requirement() -> Spec:
    return dataclasses.replace(
        _make_spec(),
        constraints=(
            BlastRadiusConstraint(no_edits_outside=("alias_resolution.py",)),
        ),
    )


@pytest.mark.parametrize("list_prefix", ["", "- "])
def test_plan_repairs_exact_legacy_requirement_display_wrappers(
    tmp_path: Path,
    list_prefix: str,
):
    spec = _spec_with_blast_radius_requirement()
    canonical = str(spec.constraints[0])
    wrapped = f"{list_prefix}`blast_radius`: {canonical}"
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = [wrapped]
    repaired = json.loads(json.dumps(invalid))
    repaired["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = [canonical]

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        spec,
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(repaired)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=lambda _candidate: (),
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1
    feedback = planner_backend.invocations[1].user
    replacement = json.dumps(
        {
            "source_id": "constraints[0]",
            "exact_citable_text": canonical,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    handoff = json.loads(next(
        line
        for line in feedback.splitlines()
        if line.startswith('{"id": "non_root_requirement_ref_')
    ))
    assert replacement in handoff["evidence"][0]["excerpt"]
    assert replacement in handoff["suggested_action"]
    assert "non-citable prompt display wrapper" in handoff["evidence"][0][
        "excerpt"
    ]
    assert "with only the exact_citable_text" in handoff["suggested_action"]


def test_requirement_ref_feedback_stays_generic_for_nonmatching_text():
    spec = _spec_with_blast_radius_requirement()
    canonical = str(spec.constraints[0])
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = [f"blast_radius: {canonical}"]
    dag = _parse_tier_dag(invalid)
    assert dag is not None

    findings = _requirement_ref_source_findings(
        dag,
        root_spec=spec,
        round_index=0,
    )

    assert len(findings) == 1
    assert "canonical_replacement" not in findings[0].evidence[0].excerpt
    assert findings[0].suggested_action == (
        "replace this value with one exact contiguous excerpt from the root "
        "intent, constraints, or done criteria; do not cite generated plan "
        "text, repository text, or review feedback"
    )


def test_requirement_ref_feedback_stays_generic_for_ambiguous_wrapper():
    constraint = BlastRadiusConstraint(
        no_edits_outside=("alias_resolution.py",),
    )
    spec = dataclasses.replace(
        _make_spec(),
        constraints=(constraint, constraint),
    )
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = [f"`blast_radius`: {constraint}"]
    dag = _parse_tier_dag(invalid)
    assert dag is not None

    findings = _requirement_ref_source_findings(
        dag,
        root_spec=spec,
        round_index=0,
    )

    assert len(findings) == 1
    assert "canonical_replacement" not in findings[0].evidence[0].excerpt
    assert "exact_citable_text" not in (findings[0].suggested_action or "")


def test_requirement_ref_validator_keeps_canonical_refs_unchanged():
    spec = _spec_with_blast_radius_requirement()
    canonical = str(spec.constraints[0])
    valid = json.loads(json.dumps(_DAG_OK))
    valid["tiers"][0]["slices"][0]["directive"][
        "requirement_refs"
    ] = [canonical, "alias_resolution.py"]
    dag = _parse_tier_dag(valid)
    assert dag is not None

    assert _requirement_ref_source_findings(
        dag,
        root_spec=spec,
        round_index=0,
    ) == ()


_KIMI_QUOTE_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "plan_kimi_missing_backticks.json")
    .read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case", _KIMI_QUOTE_FIXTURE["cases"], ids=lambda case: case["id"],
)
def test_captured_kimi_backtick_refs_offer_exact_hints_without_acceptance(case):
    spec = _make_spec(_KIMI_QUOTE_FIXTURE["root_intent"])
    payload = copy.deepcopy(_DAG_OK)
    payload["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = [
        case["invalid_requirement_ref"],
    ]
    dag = _parse_tier_dag(payload)
    assert dag is not None
    findings = _requirement_ref_source_findings(dag, root_spec=spec, round_index=0)

    assert len(findings) == 1
    assert findings[0].severity == "High"
    assert dag.tiers[0].slices[0].directive.requirement_refs == (
        case["invalid_requirement_ref"],
    )
    hint = json.loads(findings[0].evidence[0].excerpt.split("canonical_replacement=", 1)[1])
    assert hint["source_id"] == "intent"
    assert hint["exact_citable_text"] == case["exact_citable_text"]
    span = hint["normalized_source_span"]
    source = dict(_root_requirement_sources(spec))[hint["source_id"]]
    assert source[span["start"]:span["end"]] == hint["exact_citable_text"]
    assert json.dumps(hint, ensure_ascii=False, separators=(",", ":")) in findings[0].suggested_action

    payload["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = [
        hint["exact_citable_text"],
    ]
    corrected = _parse_tier_dag(payload)
    assert corrected is not None
    assert _requirement_ref_source_findings(corrected, root_spec=spec, round_index=1) == ()


def test_captured_kimi_quotes_repair_once_before_critic_and_judge(tmp_path):
    spec = _make_spec(_KIMI_QUOTE_FIXTURE["root_intent"])
    invalid = copy.deepcopy(_DAG_OK)
    invalid["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = [
        case["invalid_requirement_ref"] for case in _KIMI_QUOTE_FIXTURE["cases"]
    ]
    repaired = copy.deepcopy(invalid)
    repaired["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = [
        case["exact_citable_text"] for case in _KIMI_QUOTE_FIXTURE["cases"]
    ]
    outcome, planner, critic, judge = _plan_with_scripts(
        spec, Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(repaired)],
        critic=[_VERDICT_PASS], judge=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0, judge_required=True,
            breaker_required=False, persistence_veto_threshold=2,
        ),
        candidate_validator=lambda _candidate: (),
        max_candidate_validation_repairs=1,
    )
    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner.invocations) == 2
    assert len(critic.invocations) == len(judge.invocations) == 1
    hints = [
        json.loads(line)
        for line in planner.invocations[1].user.splitlines()
        if line.startswith('{"id": "non_root_requirement_ref_')
    ]
    assert len(hints) == 4
    for hint, case in zip(hints, _KIMI_QUOTE_FIXTURE["cases"]):
        replacement = json.loads(hint["evidence"][0]["excerpt"].split("canonical_replacement=", 1)[1])
        assert replacement["exact_citable_text"] == case["exact_citable_text"]
    assert outcome.candidate_dag == _parse_tier_dag(repaired)
    assert outcome.spec.intent == spec.intent
    assert outcome.spec.constraints == spec.constraints
    assert outcome.spec.done_criteria == spec.done_criteria


def test_plan_candidate_transform_identity_preserves_candidate(tmp_path: Path):
    seen = []

    def identity(candidate):
        seen.append(candidate)
        return candidate

    outcome, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_transform=identity,
    )

    assert outcome.accepted
    assert len(seen) == 1
    assert outcome.candidate_dag is seen[0]


def test_plan_uses_transformed_candidate_for_review_and_outcome(tmp_path: Path):
    def replace_scope(candidate):
        tier = candidate.tiers[0]
        candidate_slice = tier.slices[0]
        directive = candidate_slice.directive
        transformed_directive = dataclasses.replace(
            directive,
            files_in_scope=("src/transformed.py",),
        )
        transformed_slice = dataclasses.replace(
            candidate_slice,
            directive=transformed_directive,
        )
        transformed_tier = dataclasses.replace(tier, slices=(transformed_slice,))
        return dataclasses.replace(candidate, tiers=(transformed_tier,))

    outcome, _, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_transform=replace_scope,
    )

    assert outcome.accepted
    assert outcome.candidate_dag is not None
    directive = outcome.candidate_dag.tiers[0].slices[0].directive
    assert directive.files_in_scope == ("src/transformed.py",)
    assert "src/transformed.py" in critic_backend.invocations[0].user
    assert "src/thing.py" not in critic_backend.invocations[0].user


@pytest.mark.parametrize("mode", ["raise", "wrong_type"])
def test_plan_candidate_transform_failure_is_fail_closed(
    tmp_path: Path,
    mode: str,
):
    def failing_transform(candidate):
        if mode == "raise":
            raise RuntimeError("normalization failed")
        return {"tiers": candidate.tiers}

    outcome, _, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_transform=failing_transform,
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "candidate_transform_failed"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(critic_backend.invocations) == 0


def test_plan_repairs_deterministic_candidate_before_paid_review(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]
    seen: list[str] = []

    def validate_candidate(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        seen.append(path)
        if path == "src/phantom.py":
            return (
                Finding(
                    id="missing_files_in_scope__s0__src/phantom.py",
                    severity="Critical",
                    category="phantom_reference",
                    evidence=(
                        Evidence(
                            kind="filesystem_check",
                            location="t0.s0.files_in_scope",
                            excerpt="src/phantom.py does not exist",
                        ),
                    ),
                    suggested_action="use the grounded path",
                    source="deterministic",
                ),
            )
        return ()

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    assert seen == ["src/phantom.py", "src/thing.py"]
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1
    assert "missing_files_in_scope__s0__src/phantom.py" in (
        planner_backend.invocations[1].user
    )


def test_plan_allows_missing_file_to_move_into_creates_paths(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    directive = invalid["tiers"][0]["slices"][0]["directive"]
    directive["files_in_scope"] = ["src/new_contract.py"]

    repaired = json.loads(json.dumps(invalid))
    repaired_directive = repaired["tiers"][0]["slices"][0]["directive"]
    repaired_directive["files_in_scope"] = []
    repaired_directive["creates_paths"] = ["src/new_contract.py"]

    def validate_candidate(candidate):
        candidate_directive = candidate.tiers[0].slices[0].directive
        if "src/new_contract.py" not in candidate_directive.files_in_scope:
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/new_contract.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/new_contract.py does not exist",
                    ),
                ),
                suggested_action=(
                    "move the intended new output from files_in_scope to "
                    "creates_paths"
                ),
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(repaired)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    repaired_directive = outcome.spec.tier_dag.tiers[0].slices[0].directive
    assert repaired_directive.files_in_scope == ()
    assert repaired_directive.creates_paths == ("src/new_contract.py",)
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1


def test_plan_rejects_scope_addition_smuggled_with_path_reclassification(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    directive = invalid["tiers"][0]["slices"][0]["directive"]
    directive["files_in_scope"] = ["src/new_contract.py"]

    overbroad_repair = json.loads(json.dumps(invalid))
    repaired_directive = overbroad_repair["tiers"][0]["slices"][0][
        "directive"
    ]
    repaired_directive["files_in_scope"] = []
    repaired_directive["creates_paths"] = [
        "src/new_contract.py",
        "src/unrelated.py",
    ]

    def validate_candidate(candidate):
        candidate_directive = candidate.tiers[0].slices[0].directive
        if "src/new_contract.py" not in candidate_directive.files_in_scope:
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/new_contract.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/new_contract.py does not exist",
                    ),
                ),
                suggested_action="correct files_in_scope",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(overbroad_repair)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "planner_contract_failure"
    assert outcome.spec.tier_dag is None
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0


def test_plan_allows_blast_radius_finding_to_repair_scope_fields(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid_path = "outside/thing.py"
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = [invalid_path]

    def validate_candidate(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        return (
            (_blast_radius_directive_path_finding(path),)
            if path == invalid_path
            else ()
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1


def test_plan_rejects_blast_radius_smuggled_with_scope_repair(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid_path = "outside/thing.py"
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = [invalid_path]
    overbroad_repair = json.loads(json.dumps(_DAG_OK))
    overbroad_repair["tiers"][0]["slices"][0]["directive"][
        "blast_radius"
    ] = {"max_files": 99}

    def validate_candidate(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        return (
            (_blast_radius_directive_path_finding(path),)
            if path == invalid_path
            else ()
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(overbroad_repair)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "planner_contract_failure"
    assert outcome.spec.tier_dag is None
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0


def test_plan_allows_parallel_finding_to_repair_creates_paths(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    first_slice = invalid["tiers"][0]["slices"][0]
    first_slice["directive"]["files_in_scope"] = []
    first_slice["directive"]["creates_paths"] = ["src/shared.py"]
    second_slice = json.loads(json.dumps(first_slice))
    second_slice["id"] = "s1"
    second_slice["directive"]["slice_id"] = "s1"
    invalid["tiers"][0]["slices"].append(second_slice)

    repaired = json.loads(json.dumps(invalid))
    repaired["tiers"][0]["slices"][1]["directive"][
        "creates_paths"
    ] = ["src/independent.py"]

    def validate_candidate(candidate):
        creates = tuple(
            candidate_slice.directive.creates_paths
            for candidate_slice in candidate.tiers[0].slices
        )
        if creates == (("src/shared.py",), ("src/shared.py",)):
            return (
                _parallel_tier_shared_path_finding("src/shared.py"),
            )
        return ()

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(repaired)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1


@pytest.mark.parametrize("smuggle_blast_radius", [False, True])
def test_plan_intent_repair_cannot_smuggle_blast_radius(
    tmp_path: Path,
    smuggle_blast_radius: bool,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "intent"
    ] = "delete outside/generated.py"
    repaired = json.loads(json.dumps(_DAG_OK))
    if smuggle_blast_radius:
        repaired["tiers"][0]["slices"][0]["directive"][
            "blast_radius"
        ] = {"max_files": 99}

    def validate_candidate(candidate):
        intent = candidate.tiers[0].slices[0].directive.intent
        return (
            (_deletion_intent_finding(),)
            if intent == "delete outside/generated.py"
            else ()
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(repaired)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=validate_candidate,
        max_candidate_validation_repairs=1,
    )

    assert outcome.accepted is not smuggle_blast_radius
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == (
        0 if smuggle_blast_radius else 1
    )
    if smuggle_blast_radius:
        assert outcome.status == "planner_contract_failure"
        assert outcome.spec.tier_dag is None


def test_plan_repeated_deterministic_candidate_failure_skips_all_reviews(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]

    def reject_candidate(_candidate):
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="use a grounded path",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, judge_backend = (
        _plan_with_scripts(
            _make_spec(),
            Corpus(tmp_path),
            planner=[json.dumps(invalid), json.dumps(invalid)],
            critic=[_VERDICT_PASS],
            judge=[_VERDICT_PASS],
            policy=PlanPolicy(
                max_critique_rounds=0,
                judge_required=True,
                breaker_required=False,
                persistence_veto_threshold=2,
            ),
            candidate_validator=reject_candidate,
            max_candidate_validation_repairs=1,
        )
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "planner_contract_failure"
    assert outcome.candidate_validation_repairs_used == 1
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(outcome.spec.verdict_refs) == 2
    assert outcome.cost_usd == pytest.approx(0.02)
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 0


def test_candidate_repair_metric_requires_an_actual_planner_attempt(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]

    def reject_candidate(_candidate):
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="correct files_in_scope",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=reject_candidate,
        max_candidate_validation_repairs=1,
        budget_usd=0.01,
    )

    assert outcome.status == "budget_exhausted"
    assert outcome.candidate_validation_repairs_used == 0
    assert len(planner_backend.invocations) == 1
    assert len(critic_backend.invocations) == 0


def test_round_exhaustion_names_candidate_validation_correction_truthfully(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]

    def reject_phantom(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        if path != "src/phantom.py":
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="correct files_in_scope",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        candidate_validator=reject_phantom,
        max_candidate_validation_repairs=1,
    )

    assert outcome.status == "rounds_exhausted"
    assert outcome.rounds_used == 2
    assert outcome.candidate_validation_repairs_used == 1
    assert "1 bounded candidate-validation correction attempt(s)" in outcome.reason
    assert "compaction" not in outcome.reason
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1


def test_plan_rejects_unrelated_field_change_in_candidate_repair(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]
    overbroad_repair = json.loads(json.dumps(_DAG_OK))
    overbroad_repair["tiers"][0]["slices"][0]["directive"][
        "intent"
    ] = "unrelated semantic rewrite"

    def reject_phantom(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        if path != "src/phantom.py":
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="correct files_in_scope only",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(invalid), json.dumps(overbroad_repair)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=reject_phantom,
        max_candidate_validation_repairs=1,
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "planner_contract_failure"
    assert outcome.spec.tier_dag is None
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0


def test_plan_parse_and_candidate_gates_share_one_repair_allowance(
    tmp_path: Path,
):
    invalid = json.loads(json.dumps(_DAG_OK))
    invalid["tiers"][0]["slices"][0]["directive"][
        "files_in_scope"
    ] = ["src/phantom.py"]

    def reject_phantom(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        if path != "src/phantom.py":
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="use a grounded path",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, judge_backend = (
        _plan_with_scripts(
            _make_spec(),
            Corpus(tmp_path),
            planner=["not JSON", json.dumps(invalid)],
            critic=[_VERDICT_PASS],
            judge=[_VERDICT_PASS],
            policy=PlanPolicy(
                max_critique_rounds=0,
                judge_required=True,
                breaker_required=False,
                persistence_veto_threshold=2,
            ),
            candidate_validator=reject_phantom,
            max_candidate_validation_repairs=1,
        )
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "planner_contract_failure"
    assert outcome.spec.tier_dag is None
    assert len(outcome.spec.verdict_refs) == 2
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 0


@pytest.mark.parametrize("mode", ["raise", "wrong_type"])
def test_plan_candidate_validator_failure_is_fail_closed(
    tmp_path: Path,
    mode: str,
):
    def failing_validator(_candidate):
        if mode == "raise":
            raise RuntimeError("validator unavailable")
        return ["not-a-finding"]

    outcome, _, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=failing_validator,
        max_candidate_validation_repairs=1,
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "candidate_validation_failed"
    assert outcome.spec.tier_dag is None
    assert len(critic_backend.invocations) == 0


def test_plan_revises_once_then_passes(tmp_path: Path):
    """Round 0 critic fails; round 1 critic passes; final DAG persisted."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,  # high — don't fire veto
        ),
    )
    new_spec = outcome.spec
    assert outcome.decision == "accepted"
    assert new_spec.tier_dag is not None
    assert len(new_spec.verdict_refs) == 2  # two critic verdicts


def test_plan_two_revisions_then_passes_both_independent_reviewers(
    tmp_path: Path,
):
    """Every authorizable candidate retains both Critic and Judge review."""

    outcome, planner_backend, critic_backend, judge_backend = (
        _plan_with_scripts(
            _make_spec(),
            Corpus(tmp_path),
            planner=[json.dumps(_DAG_OK)] * 3,
            critic=[_VERDICT_FAIL, _VERDICT_FAIL, _VERDICT_PASS],
            judge=[_VERDICT_PASS] * 3,
            policy=PlanPolicy(
                max_critique_rounds=2,
                judge_required=True,
                breaker_required=False,
                persistence_veto_threshold=10,
            ),
        )
    )

    assert outcome.accepted
    assert outcome.rounds_used == 3
    assert len(planner_backend.invocations) == 3
    assert len(critic_backend.invocations) == 3
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 3
    assert len(outcome.spec.verdict_refs) == 6


def test_plan_judge_runs_when_critic_findings_present(tmp_path: Path):
    """Judge runs when critic has findings and judge_required=True."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL, _VERDICT_PASS],
        judge=[_VERDICT_PASS, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    # A required independent Judge runs on every candidate, including when
    # the Critic passes with no findings.
    assert len(outcome.spec.verdict_refs) == 4


def test_plan_review_prompt_compacts_findings_but_preserves_history_indices(
    tmp_path: Path,
):
    corpus = Corpus(tmp_path)
    planner_backend = ScriptedBackend([
        json.dumps(_DAG_OK),
        json.dumps(_DAG_OK),
    ])
    critic_backend = ScriptedBackend([
        _verdict_fail_with_ids("critic-round-zero"),
        _VERDICT_PASS,
    ])
    judge_backend = ScriptedBackend([
        _verdict_fail_with_ids("judge-round-zero"),
        _VERDICT_PASS,
    ])
    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    outcome = plan(
        _make_spec(),
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=judge_backend,
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        review_fn=tracking_review,
    )

    assert outcome.accepted
    assert [kind for kind, _ in calls] == ["critic", "judge", "critic", "judge"]
    expected_context_keys = {
        "spec_intent",
        "spec_constraints",
        "spec_done_criteria",
        "prior_critic_findings",
        "prior_judge_findings",
    }
    assert all(set(context) == expected_context_keys for _, context in calls)
    assert all("prior_findings" not in context for _, context in calls)
    assert json.loads(calls[0][1]["spec_constraints"]) == {
        "schema_version": "assay.plan-review-context.v1",
        "root_constraints": [],
        "effective_caps": {
            # The first candidate anchors the relative inflation bound before
            # review, so the reviewer sees the same effective cap enforced by
            # the post-parse gate rather than the pre-candidate planner hint.
            "hard_max_total_slices": 8,
            "hard_max_total_tiers": None,
            "hard_max_verification_only_empty_scope_slices": None,
        },
    }
    first_judge_history = json.loads(calls[1][1]["prior_critic_findings"])
    assert first_judge_history[-1]["round"] == 0
    assert first_judge_history[-1]["findings"][0]["id"] == "critic-round-zero"
    second_critic_judges = json.loads(calls[2][1]["prior_judge_findings"])
    assert second_critic_judges[-1]["findings"][0]["id"] == "judge-round-zero"
    final_judge_critics = json.loads(calls[3][1]["prior_critic_findings"])
    assert [entry["round"] for entry in final_judge_critics] == [0, 1]
    assert [entry["findings"] for entry in final_judge_critics] == [[], []]
    final_judge_history = json.loads(calls[3][1]["prior_judge_findings"])
    assert final_judge_history[0]["findings"][0]["id"] == "judge-round-zero"
    # All four complete Verdicts remain independently persisted even though
    # resolved finding bodies are absent from later prompt contexts.
    assert len(outcome.spec.verdict_refs) == 4
    assert "critic-round-zero" in planner_backend.invocations[1].user
    assert "judge-round-zero" in planner_backend.invocations[1].user


def test_plan_fresh_judge_reassessment_omits_only_its_own_frontier(
    tmp_path: Path,
):
    """A passing Critic can trigger a fresh, still-mandatory Judge audit."""

    corpus = Corpus(tmp_path)
    planner_backend = ScriptedBackend([
        json.dumps(_DAG_OK),
        json.dumps(_DAG_OK),
    ])
    critic_backend = ScriptedBackend([
        _verdict_fail_with_ids("critic-round-zero"),
        _VERDICT_PASS,
    ])
    judge_backend = ScriptedBackend([
        _verdict_fail_with_ids("judge-round-zero"),
        _VERDICT_PASS,
    ])
    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    policy = PlanPolicy(
        max_critique_rounds=1,
        judge_required=True,
        fresh_judge_reassessment_min_prior_rounds=1,
        breaker_required=False,
        persistence_veto_threshold=10,
        max_total_slices=1,
        max_total_tiers=1,
        max_verification_only_empty_scope_slices=0,
        planner_guidance="PRIVATE PRODUCT GUIDANCE",
    )
    outcome = plan(
        dataclasses.replace(_make_spec(), plan_policy=policy),
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=judge_backend,
        corpus=corpus,
        critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"),
        review_fn=tracking_review,
    )

    assert outcome.accepted
    assert [kind for kind, _ in calls] == ["critic", "judge", "critic", "judge"]
    # The prior Judge finding was delivered to the Planner before any prompt
    # softening, so the revised candidate is accountable to that feedback.
    assert "critic-round-zero" in planner_backend.invocations[1].user
    assert "judge-round-zero" in planner_backend.invocations[1].user
    final_judge_context = calls[-1][1]
    assert json.loads(final_judge_context["prior_judge_findings"]) == []
    projected_constraints = json.loads(
        final_judge_context["spec_constraints"]
    )
    assert projected_constraints == {
        "schema_version": "assay.plan-review-context.v1",
        "root_constraints": [],
        "effective_caps": {
            "hard_max_total_slices": 1,
            "hard_max_total_tiers": 1,
            "hard_max_verification_only_empty_scope_slices": 0,
        },
    }
    assert "PRIVATE PRODUCT GUIDANCE" not in final_judge_context[
        "spec_constraints"
    ]
    final_critic_history = json.loads(
        final_judge_context["prior_critic_findings"]
    )
    assert [entry["round"] for entry in final_critic_history] == [0, 1]
    assert [entry["findings"] for entry in final_critic_history] == [[], []]
    # Nothing is erased from durable debate evidence.
    assert len(outcome.spec.verdict_refs) == 4


def test_plan_fresh_judge_reassessment_still_fails_closed_on_revise(
    tmp_path: Path,
):
    """A delayed blind re-audit cannot turn a new Judge blocker into pass."""

    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    outcome, planner_backend, critic_backend, judge_backend = (
        _plan_with_scripts(
            _make_spec(),
            Corpus(tmp_path),
            planner=[json.dumps(_DAG_OK)] * 4,
            critic=[
                _verdict_fail_with_ids("critic-stable"),
                _verdict_fail_with_ids("critic-stable"),
                _verdict_fail_with_ids("critic-stable"),
                _review_response(
                    "pass",
                    [_review_finding("critic-stable", severity="Low")],
                ),
            ],
            judge=[
                _verdict_fail_with_ids("judge-stable"),
                _verdict_fail_with_ids("judge-stable"),
                _verdict_fail_with_ids("judge-stable"),
                _verdict_fail_with_ids("judge-fresh-untargeted"),
            ],
            policy=PlanPolicy(
                max_critique_rounds=3,
                judge_required=True,
                fresh_judge_reassessment_min_prior_rounds=3,
                breaker_required=False,
                persistence_veto_threshold=10,
            ),
            review_fn=tracking_review,
        )
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "rounds_exhausted"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 4
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 4
    assert len(outcome.spec.verdict_refs) == 8
    assert "judge-stable" in planner_backend.invocations[3].user
    # Prior Judge feedback remains visible until the configured threshold and
    # Critic-pass conjunction is reached.
    assert json.loads(calls[3][1]["prior_judge_findings"])
    assert json.loads(calls[5][1]["prior_judge_findings"])
    assert json.loads(calls[-1][1]["prior_judge_findings"]) == []
    final_critic_history = json.loads(
        calls[-1][1]["prior_critic_findings"]
    )
    assert final_critic_history[-1]["findings"][0]["id"] == "critic-stable"
    assert final_critic_history[-1]["findings"][0]["severity"] == "Low"


def test_plan_fresh_judge_reassessment_counts_only_conclusive_rounds(
    tmp_path: Path,
):
    """Inconclusive Judge attempts neither unlock nor consume the escape hatch."""

    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    outcome, _, _, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 5,
        critic=[
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _VERDICT_PASS,
        ],
        judge=[
            _INVALID_REVIEW,
            _verdict_fail_with_ids("judge-stable"),
            _INVALID_REVIEW,
            _verdict_fail_with_ids("judge-stable"),
            _VERDICT_PASS,
        ],
        policy=PlanPolicy(
            max_critique_rounds=4,
            judge_required=True,
            fresh_judge_reassessment_min_prior_rounds=2,
            breaker_required=False,
            persistence_veto_threshold=10,
            max_judge_failures=3,
        ),
        review_fn=tracking_review,
    )

    assert outcome.accepted
    assert len(outcome.spec.verdict_refs) == 10
    # Before the second conclusive Judge verdict, two inconclusive attempts
    # plus one revise must not satisfy the threshold.
    assert json.loads(calls[7][1]["prior_judge_findings"])
    # After the second revise is delivered to the Planner and the Critic
    # passes its revision, the one-shot fresh reassessment becomes eligible.
    assert json.loads(calls[9][1]["prior_judge_findings"]) == []


@pytest.mark.parametrize(
    ("final_judge", "expected_decision", "expected_status"),
    [
        (_VERDICT_PASS, "accepted", "accepted"),
        (
            _verdict_fail_with_ids("judge-stable"),
            "rejected",
            "persistence_veto",
        ),
    ],
)
def test_plan_fresh_judge_reassessment_precedes_product_persistence_veto(
    tmp_path: Path,
    final_judge: str,
    expected_decision: str,
    expected_status: str,
):
    """DeepSWE's 2/3 thresholds permit one fresh but still fail-closed audit."""

    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    outcome, _, _, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 3,
        critic=[
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _VERDICT_PASS,
        ],
        judge=[
            _verdict_fail_with_ids("judge-stable"),
            _verdict_fail_with_ids("judge-stable"),
            final_judge,
        ],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=True,
            fresh_judge_reassessment_min_prior_rounds=2,
            breaker_required=False,
            persistence_veto_threshold=3,
        ),
        review_fn=tracking_review,
    )

    assert outcome.decision == expected_decision
    assert outcome.status == expected_status
    assert json.loads(calls[-1][1]["prior_judge_findings"]) == []
    assert len(outcome.spec.verdict_refs) == 6


def test_plan_fresh_judge_pass_still_runs_weighted_backstop(tmp_path: Path):
    """Blind re-audit changes context, never current-verdict veto policy."""

    stable = Finding(
        id="stable",
        severity="High",
        category="correctness",
        evidence=(),
        suggested_action="repair the candidate",
        source="llm",
        disposition="confirmed",
    )
    current_critical = Finding(
        id="fresh-critical",
        severity="Critical",
        category="correctness",
        evidence=(),
        suggested_action="do not execute this candidate",
        source="llm",
        disposition="confirmed",
    )
    scripted_calls, scripted_review = _scripted_typed_review([
        ("revise", (stable,)),
        ("revise", (stable,)),
        ("pass", ()),
        ("pass", (current_critical,)),
    ])
    contexts: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        contexts.append((kwargs["kind"], dict(kwargs["context"])))
        return scripted_review(*args, **kwargs)

    outcome, _, _, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 2,
        critic=[],
        judge=[],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=True,
            fresh_judge_reassessment_min_prior_rounds=1,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        review_fn=tracking_review,
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "weighted_severity_veto"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert scripted_calls == ["critic", "judge", "critic", "judge"]
    assert json.loads(contexts[-1][1]["prior_judge_findings"]) == []


def test_plan_fresh_judge_reassessment_is_one_shot(tmp_path: Path):
    """A fresh blocker restores ordinary Judge history on the next round."""

    calls: list[tuple[str, dict]] = []

    def tracking_review(*args, **kwargs):
        calls.append((kwargs["kind"], dict(kwargs["context"])))
        return assay_review(*args, **kwargs)

    outcome, planner_backend, _, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 5,
        critic=[
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _verdict_fail_with_ids("critic-stable"),
            _VERDICT_PASS,
            _VERDICT_PASS,
        ],
        judge=[
            _verdict_fail_with_ids("judge-stable"),
            _verdict_fail_with_ids("judge-stable"),
            _verdict_fail_with_ids("judge-stable"),
            _verdict_fail_with_ids("judge-fresh-blocker"),
            _verdict_fail_with_ids("judge-still-blocking"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=4,
            judge_required=True,
            fresh_judge_reassessment_min_prior_rounds=3,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        review_fn=tracking_review,
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "rounds_exhausted"
    blind_judge_context = calls[7][1]
    later_judge_context = calls[9][1]
    assert json.loads(blind_judge_context["prior_judge_findings"]) == []
    restored_frontier = json.loads(
        later_judge_context["prior_judge_findings"]
    )
    assert restored_frontier[-1]["findings"][0]["id"] == (
        "judge-fresh-blocker"
    )
    assert "judge-fresh-blocker" in planner_backend.invocations[4].user
    assert len(outcome.spec.verdict_refs) == 10


def test_plan_does_not_start_critic_after_planner_consumes_budget(tmp_path: Path):
    planner_backend = ScriptedBackend([{
        "final": json.dumps(_DAG_OK),
        "cost": 0.01,
    }])
    critic_backend = ScriptedBackend([_VERDICT_PASS])
    outcome = plan(
        _make_spec(),
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=None,
        corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"),
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        budget_usd=0.01,
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "budget_exhausted"
    assert len(critic_backend.invocations) == 0


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            BudgetExhausted(spent_usd=0.01, limit_usd=0.01),
            "budget_exhausted",
        ),
        (ExternalCancellation(message="cancelled"), "cancelled"),
        (ContentRejected(message="denied"), "hard_fail_content_rejected"),
        (MaxTurnsExceeded(message="turn cap"), "hard_fail_max_turns"),
        (
            IncompleteEvidence(
                missing=("terminal_complete",),
                message="terminal evidence missing",
            ),
            "planner_protocol_failure",
        ),
    ],
)
def test_plan_never_retries_terminal_backend_stop(
    tmp_path: Path,
    error,
    expected_status: str,
):
    planner_backend = ScriptedBackend([
        {"error": error, "cost": 0.01},
        json.dumps(_DAG_OK),
    ])
    critic_backend = ScriptedBackend([_VERDICT_PASS])

    outcome = plan(
        _make_spec(),
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=None,
        corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"),
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.status == expected_status
    assert outcome.decision == "inconclusive"
    assert outcome.cost_usd == pytest.approx(0.01)
    assert len(planner_backend.invocations) == 1
    assert len(critic_backend.invocations) == 0


def test_plan_requires_independent_backends_and_mandatory_judge(tmp_path: Path):
    corpus = Corpus(tmp_path)
    shared = ScriptedBackend([])
    policy = PlanPolicy(
        max_critique_rounds=0,
        judge_required=True,
        breaker_required=False,
        persistence_veto_threshold=2,
    )
    with pytest.raises(ValueError, match="judge_rubric"):
        plan(
            _make_spec(),
            planner_backend=ScriptedBackend([]),
            critic_backend=ScriptedBackend([]),
            judge_backend=None,
            corpus=corpus,
            critic_rubric=_rubric("critic"),
            policy=policy,
        )
    with pytest.raises(ValueError, match="independent"):
        plan(
            _make_spec(),
            planner_backend=shared,
            critic_backend=shared,
            judge_backend=ScriptedBackend([]),
            corpus=corpus,
            critic_rubric=_rubric("critic"),
            judge_rubric=_rubric("judge"),
            policy=policy,
        )


def test_plan_judge_reject_never_attaches_candidate(tmp_path: Path):
    reject = _review_response("reject", [_review_finding("fundamental")])
    outcome, *_ = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        judge=[reject],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None


def test_plan_persistence_veto_short_circuits(tmp_path: Path):
    """A finding surviving threshold rounds triggers an early veto exit."""
    corpus = Corpus(tmp_path)
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL, _VERDICT_FAIL],
        policy=PlanPolicy(
            max_critique_rounds=5,  # plenty of headroom
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )
    # Loop exited after the second critic round.
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None


def test_plan_persistence_uses_judge_target_closure_frontier(tmp_path: Path):
    critic_blocker = _review_response(
        "revise", [_review_finding("too-broad")]
    )
    judge_closure = _review_response(
        "pass",
        [_review_finding(
            "closure",
            disposition="addressed",
            targets=(
                "prior:critic:0:sha256-" + "a" * 64 + ":too-broad",
            ),
        )],
    )
    outcome, planner_backend, critic_backend, judge_backend = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[critic_blocker, _VERDICT_PASS],
        judge=[judge_closure, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=True,
            persistence_veto_threshold=1,
        ),
    )

    assert outcome.accepted
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 2


def test_plan_persistence_veto_survives_finding_id_rename(tmp_path: Path):
    first = _review_response(
        "revise", [_review_finding("broad-old", location="t0:s0")]
    )
    renamed = _review_response(
        "revise", [_review_finding("broad-renamed", location="t0:s0")]
    )
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[first, renamed],
        policy=PlanPolicy(
            max_critique_rounds=5,
            judge_required=False,
            persistence_veto_threshold=2,
        ),
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "persistence_veto"
    assert "broad-renamed" in outcome.reason
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2


def test_plan_persistence_does_not_merge_same_id_different_defects(
    tmp_path: Path,
):
    first = _review_response(
        "revise", [_review_finding("reused-id", location="t0:s0")]
    )
    distinct = _review_response(
        "revise", [_review_finding("reused-id", location="t1:s1")]
    )
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[first, distinct],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            persistence_veto_threshold=2,
        ),
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "rounds_exhausted"
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2


def test_plan_resolved_finding_resets_persistence_before_acceptance(
    tmp_path: Path,
):
    """An addressed prior defect is audit history, not a surviving defect."""

    addressed = _review_response(
        "pass",
        [_review_finding("f1", severity="Critical", disposition="addressed")],
    )
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL, addressed],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )

    assert outcome.accepted
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2


def test_plan_13033_style_addressed_critical_judge_findings_do_not_veto(
    tmp_path: Path,
):
    """Resolved Critical history cannot trip persistence or weighted vetoes."""

    resolved_critical_findings = [
        _review_finding(
            "prior-critical-1",
            severity="Critical",
            disposition="addressed",
        ),
        _review_finding(
            "prior-critical-2",
            severity="Critical",
            disposition="addressed",
        ),
    ]
    judge_pass = _review_response("pass", resolved_critical_findings)
    outcome, _, _, judge_backend = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        judge=[judge_pass],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=True,
            breaker_required=False,
            # Either old bug would reject this round immediately.
            persistence_veto_threshold=1,
            judge_veto_weighted_threshold=10.0,
        ),
    )

    assert outcome.accepted
    assert outcome.status == "accepted"
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 1


def test_plan_returns_unchanged_on_unparseable_planner(tmp_path: Path):
    corpus = Corpus(tmp_path)
    # max_critique_rounds=1 means the planner runs round 0 + one
    # retry-with-rebuke round on parse failure. Script both so the
    # loop exhausts cleanly and the primitive bails to "no DAG".
    spec = _make_spec()
    outcome, *_ = _plan_with_scripts(
        spec, corpus,
        planner=["not json at all", "still not json after the rebuke"],
        critic=[],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )
    new_spec = outcome.spec
    # Planner ran the full retry budget and bailed without a parseable
    # DAG. It stamps a lifecycle event recording the failed attempt
    # (audit trail), so the returned Spec is a fresh version with no
    # tier_dag — not identity-equal to the input.
    assert new_spec.tier_dag is None
    assert outcome.decision == "inconclusive"
    assert any(
        ref for ref in new_spec.lifecycle_event_refs
    )  # at least one lifecycle event recording the attempt


def test_plan_max_rounds_exhausted(tmp_path: Path):
    """Exhaustion preserves a diagnostic candidate but never executes it."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_VERDICT_FAIL, _VERDICT_FAIL],
        policy=PlanPolicy(
            max_critique_rounds=1,  # rounds = 1+1 = 2
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    assert outcome.decision == "inconclusive"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(outcome.spec.verdict_refs) == 2


def test_plan_critic_failure_limit_stops_consecutive_inconclusive_reviews(
    tmp_path: Path,
):
    """Critic parse/backend/schema failures cannot consume every plan round."""
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 2,
        critic=[_INVALID_REVIEW] * 2,
        policy=PlanPolicy(
            max_critique_rounds=5,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=99,
            max_critic_failures=2,
        ),
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "critic_failures_exhausted"
    assert outcome.rounds_used == 2
    assert outcome.final_critic_decision == "inconclusive"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2


def test_plan_conclusive_critic_decision_resets_failure_count(tmp_path: Path):
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 4,
        critic=[
            _INVALID_REVIEW,
            _VERDICT_FAIL,
            _INVALID_REVIEW,
            _VERDICT_PASS,
        ],
        policy=PlanPolicy(
            max_critique_rounds=3,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=99,
            max_critic_failures=2,
        ),
    )

    assert outcome.decision == "accepted"
    assert outcome.status == "accepted"
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 4


def test_plan_judge_failure_limit_stops_consecutive_inconclusive_reviews(
    tmp_path: Path,
):
    outcome, planner_backend, critic_backend, judge_backend = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 2,
        critic=[_VERDICT_PASS] * 2,
        judge=[_INVALID_REVIEW] * 2,
        policy=PlanPolicy(
            max_critique_rounds=5,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=99,
            max_judge_failures=2,
        ),
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "judge_failures_exhausted"
    assert outcome.rounds_used == 2
    assert outcome.final_critic_decision == "pass"
    assert outcome.final_judge_decision == "inconclusive"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 2
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 2


def test_plan_conclusive_judge_decision_resets_failure_count(tmp_path: Path):
    outcome, planner_backend, critic_backend, judge_backend = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK)] * 4,
        critic=[_VERDICT_PASS] * 4,
        judge=[
            _INVALID_REVIEW,
            _VERDICT_FAIL,
            _INVALID_REVIEW,
            _VERDICT_PASS,
        ],
        policy=PlanPolicy(
            max_critique_rounds=3,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=99,
            max_judge_failures=2,
        ),
    )

    assert outcome.decision == "accepted"
    assert outcome.status == "accepted"
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 4
    assert judge_backend is not None
    assert len(judge_backend.invocations) == 4


def test_plan_no_inflation_when_round_zero_only(tmp_path: Path):
    """First-round critic findings can never be inflation — guard stays silent."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK)],
        critic=[_verdict_fail_with_ids("f1", "f2")],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    # Only the critic verdict — no synthetic inflation verdict appended.
    assert len(outcome.spec.verdict_refs) == 1


def _staged_verdicts(corpus: Corpus, outcome) -> list[Verdict]:
    return [
        corpus.get_record(ref, as_type=Verdict)
        for ref in outcome.spec.verdict_refs
    ]


def test_plan_no_inflation_when_first_critic_review_follows_failed_planner_round(
    tmp_path: Path,
):
    """The attempt index also counts attempts that never reached the Critic."""
    corpus = Corpus(tmp_path)
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=["not json", json.dumps(_DAG_OK)],
        critic=[_verdict_fail_with_ids("f1", "f2")],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 1
    assert [v.kind for v in _staged_verdicts(corpus, outcome)] == ["critic"]


def test_plan_inflation_counts_only_findings_added_after_first_critic_review(
    tmp_path: Path,
):
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=["not json", json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[
            _verdict_fail_with_ids("f1"),
            _verdict_fail_with_ids("f1", "f2"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    verdicts = _staged_verdicts(corpus, outcome)
    assert [v.kind for v in verdicts] == ["critic", "critic", "inflation"]
    assert [f.evidence[0].excerpt for f in verdicts[-1].findings] == ["f2"]


def test_plan_no_inflation_when_first_conclusive_critic_review_follows_inconclusive(
    tmp_path: Path,
):
    """A Critic reply without a verdict is not a prior review of a candidate."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[_INVALID_REVIEW, _verdict_fail_with_ids("f1", "f2")],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
            max_critic_failures=3,
        ),
    )

    verdicts = _staged_verdicts(corpus, outcome)
    assert [(v.kind, v.decision) for v in verdicts] == [
        ("critic", "inconclusive"),
        ("critic", "revise"),
    ]


def test_plan_no_inflation_when_findings_persist(tmp_path: Path):
    """Same finding ids round-over-round → no inflation; only persistence."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[
            _verdict_fail_with_ids("f1"),
            _verdict_fail_with_ids("f1"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    # Two critic verdicts, no inflation Verdict.
    assert len(outcome.spec.verdict_refs) == 2


def test_plan_no_inflation_when_same_semantic_finding_is_renamed(tmp_path: Path):
    """Model-generated ids may change without creating a new defect lineage."""

    first = _review_response(
        "revise", [_review_finding("old-name", location="t0:s0")]
    )
    renamed = _review_response(
        "revise", [_review_finding("new-name", location="t0:s0")]
    )
    outcome, *_ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[first, renamed],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert len(outcome.spec.verdict_refs) == 2


def test_plan_inflation_when_revision_adds_new_id(tmp_path: Path):
    """Round 1 introduces a finding id absent in round 0 → inflation Verdict."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[
            _verdict_fail_with_ids("f1"),
            _verdict_fail_with_ids("f1", "f2"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    # 2 critic + 1 synthetic inflation Verdict = 3 refs.
    assert len(outcome.spec.verdict_refs) == 3


def test_plan_inflation_collects_across_rounds(tmp_path: Path):
    """Multiple revision rounds each adding new ids → all accumulated."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK)] * 3,
        critic=[
            _verdict_fail_with_ids("a"),
            _verdict_fail_with_ids("a", "b"),
            _verdict_fail_with_ids("a", "b", "c"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=2,  # rounds = 3
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )
    # 3 critic + 1 synthetic inflation Verdict = 4 refs.
    assert len(outcome.spec.verdict_refs) == 4


def test_plan_inflation_stamped_on_persistence_veto_exit(tmp_path: Path):
    """Inflation Verdict still stamped when persistence_veto fires early."""
    corpus = Corpus(tmp_path)
    outcome, *_ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[json.dumps(_DAG_OK), json.dumps(_DAG_OK)],
        critic=[
            _verdict_fail_with_ids("a"),
            _verdict_fail_with_ids("a", "b"),
        ],
        policy=PlanPolicy(
            max_critique_rounds=5,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,  # 'a' surviving 2 rounds → veto
        ),
    )
    # 2 critic + 1 inflation = 3 refs (loop exited on persistence_veto).
    assert outcome.decision == "rejected"
    assert len(outcome.spec.verdict_refs) == 3


def test_plan_falls_back_to_spec_plan_policy(tmp_path: Path):
    """When policy arg is None, use spec.plan_policy."""
    corpus = Corpus(tmp_path)
    spec = _make_spec()
    from dataclasses import replace
    spec_with_policy = replace(
        spec,
        plan_policy=PlanPolicy(
            max_critique_rounds=0,  # only the initial round
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
    )
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        spec_with_policy, corpus,
        planner=[json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=spec_with_policy.plan_policy,
    )
    assert outcome.spec.tier_dag is not None
    # max_critique_rounds=0 → rounds=max(1, 0+1)=1 → exactly 1 planner+critic call
    assert len(planner_backend.invocations) == 1
    assert len(critic_backend.invocations) == 1


_DAG_CONFLICT = {
    "tiers": [
        {
            "tier_id": "t0",
            "depends_on": [],
            "slices": [
                {"id": "s0", "directive": {
                    "slice_id": "s0", "intent": "a", "files_in_scope": ["src/x.py"],
                    "creates_paths": [], "verify_mode": "tests",
                    "requirement_refs": ["do the thing"],
                    "blast_radius": {"max_files": 1}}},
                {"id": "s1", "directive": {
                    "slice_id": "s1", "intent": "b", "files_in_scope": ["src/x.py"],
                    "creates_paths": [], "verify_mode": "tests",
                    "requirement_refs": ["do the thing"],
                    "blast_radius": {"max_files": 1}}},
            ],
        }
    ]
}


def _dag_with_slice_count(count: int, *, first_pair_conflicts: bool = False) -> dict:
    slices = []
    for index in range(count):
        path_index = 0 if first_pair_conflicts and index == 1 else index
        slices.append({
            "id": f"s{index}",
            "directive": {
                "slice_id": f"s{index}",
                "intent": f"implement independently verifiable increment {index}",
                "requirement_refs": ["do the thing"],
                "files_in_scope": [f"src/part_{path_index}.py"],
                "creates_paths": [],
                "verify_mode": "tests",
                "blast_radius": {"max_files": 1},
            },
        })
    return {
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": slices,
        }],
    }


def test_codex_style_1_8_17_plan_gets_one_compaction_repair(tmp_path: Path):
    """A first over-limit revision is repairable, not an immediate veto."""

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(_dag_with_slice_count(1)),
            json.dumps(_dag_with_slice_count(8)),
            json.dumps(_dag_with_slice_count(17)),
            json.dumps(_dag_with_slice_count(8)),
        ],
        critic=[_VERDICT_FAIL, _VERDICT_FAIL, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=False,
            persistence_veto_threshold=99,
            max_total_slices=16,
            max_total_tiers=12,
            inflation_compaction_retries=1,
        ),
    )

    assert outcome.accepted
    assert outcome.rounds_used == 4
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 3
    assert outcome.candidate_dag is not None
    assert len(outcome.candidate_dag.all_slices()) == 8
    compaction_prompt = planner_backend.invocations[3].user
    assert "single bounded compaction repair" in compaction_prompt
    assert "at most 8 total slices" in compaction_prompt
    assert '"id": "s16"' in compaction_prompt


def test_early_compaction_does_not_consume_a_configured_review_round(
    tmp_path: Path,
):
    """A repair call is additional to, not a replacement for, review rounds."""

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(_dag_with_slice_count(17)),
            json.dumps(_dag_with_slice_count(8)),
            json.dumps(_dag_with_slice_count(8)),
            json.dumps(_dag_with_slice_count(8)),
        ],
        critic=[_VERDICT_FAIL, _VERDICT_FAIL, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=False,
            persistence_veto_threshold=99,
            max_total_slices=16,
            inflation_compaction_retries=1,
        ),
    )

    assert outcome.accepted
    assert outcome.rounds_used == 4
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 3
    assert "single bounded compaction repair" in planner_backend.invocations[1].user


@pytest.mark.parametrize(
    ("compacted_slice_count", "expected_status", "expected_reviews"),
    [(8, "accepted", 2), (17, "inflation_veto", 1)],
)
def test_candidate_validation_and_compaction_repairs_are_independently_reachable(
    tmp_path: Path,
    compacted_slice_count: int,
    expected_status: str,
    expected_reviews: int,
):
    """Both conditional slots are reachable; repeated excess fails closed."""

    invalid = _dag_with_slice_count(1)
    invalid["tiers"][0]["slices"][0]["directive"]["files_in_scope"] = ["src/phantom.py"]

    def reject_phantom(candidate):
        path = candidate.tiers[0].slices[0].directive.files_in_scope[0]
        if path != "src/phantom.py":
            return ()
        return (
            Finding(
                id="missing_files_in_scope__s0__src/phantom.py",
                severity="Critical",
                category="phantom_reference",
                evidence=(
                    Evidence(
                        kind="filesystem_check",
                        location="t0.s0.files_in_scope",
                        excerpt="src/phantom.py does not exist",
                    ),
                ),
                suggested_action="use a grounded path",
                source="deterministic",
            ),
        )

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(invalid),
            json.dumps(_dag_with_slice_count(1)),
            json.dumps(_dag_with_slice_count(17)),
            json.dumps(_dag_with_slice_count(compacted_slice_count)),
        ],
        critic=[_VERDICT_FAIL, _VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            persistence_veto_threshold=99,
            max_total_slices=16,
            max_total_tiers=12,
            inflation_compaction_retries=1,
        ),
        candidate_validator=reject_phantom,
        max_candidate_validation_repairs=1,
    )

    assert outcome.status == expected_status
    assert outcome.rounds_used == 4
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner_backend.invocations) == 4
    # The invalid candidate and every oversized candidate remain pre-review.
    assert len(critic_backend.invocations) == expected_reviews
    assert (
        "missing_files_in_scope__s0__src/phantom.py"
        in planner_backend.invocations[1].user
    )
    assert "single bounded compaction repair" in planner_backend.invocations[3].user
    assert outcome.candidate_dag is not None
    assert len(outcome.candidate_dag.all_slices()) == (
        compacted_slice_count if expected_status == "accepted" else 1
    )


def test_repeated_excess_veto_retains_last_bounded_candidate(tmp_path: Path):
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(_dag_with_slice_count(1)),
            json.dumps(_dag_with_slice_count(8)),
            json.dumps(_dag_with_slice_count(17)),
            json.dumps(_dag_with_slice_count(17)),
        ],
        critic=[_VERDICT_FAIL, _VERDICT_FAIL],
        policy=PlanPolicy(
            max_critique_rounds=2,
            judge_required=False,
            persistence_veto_threshold=99,
            max_total_slices=16,
            inflation_compaction_retries=1,
        ),
    )

    assert outcome.decision == "rejected"
    assert outcome.status == "inflation_veto"
    assert len(planner_backend.invocations) == 4
    assert len(critic_backend.invocations) == 2
    assert outcome.candidate_dag is not None
    assert len(outcome.candidate_dag.all_slices()) == 8


def test_glm_style_conflicting_anchor_then_53_slices_compacts(tmp_path: Path):
    """The first transformed DAG anchors even when its scopes conflict."""

    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(),
        Corpus(tmp_path),
        planner=[
            json.dumps(_dag_with_slice_count(5, first_pair_conflicts=True)),
            json.dumps(_dag_with_slice_count(53)),
            json.dumps(_dag_with_slice_count(10)),
        ],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            persistence_veto_threshold=99,
            max_total_slices=16,
            max_total_tiers=12,
            inflation_compaction_retries=1,
        ),
    )

    assert outcome.accepted
    assert len(planner_backend.invocations) == 3
    assert len(critic_backend.invocations) == 1
    # floor=8, int(anchor 5 * ratio 2.5)=12; the conflicting first DAG,
    # rather than the pathological 53-slice revision, owns the anchor.
    assert "at most 12 total slices" in planner_backend.invocations[1].user
    assert "single bounded compaction repair" in planner_backend.invocations[2].user
    assert outcome.candidate_dag is not None
    assert len(outcome.candidate_dag.all_slices()) == 10


def test_empty_scope_gate_allows_one_verification_only_slice():
    payload = _dag_with_slice_count(1)
    directive = payload["tiers"][0]["slices"][0]["directive"]
    directive["intent"] = "run tests to verify the completed increments"
    directive["files_in_scope"] = []
    dag = _parse_tier_dag(payload)
    assert dag is not None

    assert _empty_scope_findings(
        dag, allowed_verification_only=1, round_index=0
    ) == ()


def test_empty_scope_gate_rejects_mutation_shaped_filler():
    payload = _dag_with_slice_count(1)
    directive = payload["tiers"][0]["slices"][0]["directive"]
    directive["files_in_scope"] = []
    dag = _parse_tier_dag(payload)
    assert dag is not None

    findings = _empty_scope_findings(
        dag, allowed_verification_only=1, round_index=2
    )
    assert len(findings) == 1
    assert findings[0].id == "empty_scope_filler_2_s0"


def _lost_planner_attempt() -> dict:
    """A planner call whose backend returned a retriable error and no reply."""
    return {
        "error": TransportError(
            http_status=529, message="overloaded", retriable=True
        ),
    }


def test_plan_backend_error_is_not_treated_as_parse_failure(tmp_path: Path, caplog):
    """APLAN-1: a backend error each round must be logged as a backend
    error (and fed back as such) — NOT as a 'your JSON was invalid'
    rebuke. The loop retries; with every attempt lost the outcome names the
    planner backend instead of a planner that produced no candidate."""
    import logging

    corpus = Corpus(tmp_path)
    with caplog.at_level(logging.WARNING, logger="assay.primitives.plan"):
        outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
            _make_spec(), corpus,
            planner=[_lost_planner_attempt(), _lost_planner_attempt()],
            critic=[],
            policy=PlanPolicy(max_critique_rounds=1, judge_required=False,
                              breaker_required=False, persistence_veto_threshold=99),
        )
    assert outcome.spec.tier_dag is None
    assert "backend error" in caplog.text
    assert "not valid" not in caplog.text  # NOT the parse-failure rebuke path
    assert outcome.decision == "inconclusive"
    assert outcome.status == "planner_infrastructure_failure"
    assert "final 2 planning attempt(s)" in outcome.reason
    assert "transport" in outcome.reason
    assert "overloaded" in outcome.reason
    assert outcome.candidate_dag is None
    assert outcome.rounds_used == 2
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == 0


@pytest.mark.parametrize("case", ["reviewed_then_lost", "unparseable_then_lost"])
def test_plan_reports_planner_infrastructure_failure_when_final_attempts_are_lost(
    tmp_path: Path,
    case: str,
):
    """A round budget that ends on a lost planner call did not fail to agree."""
    corpus = Corpus(tmp_path)
    reviewed = case == "reviewed_then_lost"
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), corpus,
        planner=[
            json.dumps(_DAG_OK) if reviewed else "not json",
            _lost_planner_attempt(),
        ],
        critic=[_VERDICT_FAIL] if reviewed else [],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "planner_infrastructure_failure"
    assert "final 1 planning attempt(s)" in outcome.reason
    assert "transport: overloaded" in outcome.reason
    assert outcome.rounds_used == 2
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag == (
        _parse_tier_dag(_DAG_OK) if reviewed else None
    )
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == (1 if reviewed else 0)
    event = corpus.get_record(
        outcome.spec.lifecycle_event_refs[-1], as_type=LifecycleEvent
    )
    assert event.kind == "plan_inconclusive"


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_reason"),
    [
        (
            "lost_then_unparseable",
            "no_candidate",
            "planner produced no safe bounded tier_dag in any round",
        ),
        (
            "lost_then_reviewed",
            "rounds_exhausted",
            "max rounds (2) exhausted without consensus",
        ),
    ],
    ids=["lost_then_unparseable", "lost_then_reviewed"],
)
def test_plan_backend_loss_before_a_final_planner_reply_keeps_existing_status(
    tmp_path: Path,
    case: str,
    expected_status: str,
    expected_reason: str,
):
    """Current behaviour: only a budget that ENDS on a lost call changes status.

    Whether a loss earlier in the budget should also be non-permanent for a
    consumer is an open owner decision; until then the status is unchanged
    and the reason names the lost attempts.
    """
    reviewed = case == "lost_then_reviewed"
    outcome, planner_backend, critic_backend, _ = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[
            _lost_planner_attempt(),
            json.dumps(_DAG_OK) if reviewed else "not json",
        ],
        critic=[_VERDICT_FAIL] if reviewed else [],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == expected_status
    assert outcome.reason == (
        f"{expected_reason}; 1 planning attempt(s) were lost to retriable "
        "planner backend errors (last: transport)"
    )
    assert len(planner_backend.invocations) == 2
    assert len(critic_backend.invocations) == (1 if reviewed else 0)


def test_plan_accepts_after_a_lost_planner_attempt_without_naming_it(
    tmp_path: Path,
):
    outcome, planner_backend, _, _ = _plan_with_scripts(
        _make_spec(), Corpus(tmp_path),
        planner=[_lost_planner_attempt(), json.dumps(_DAG_OK)],
        critic=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
    )

    assert outcome.decision == "accepted"
    assert outcome.status == "accepted"
    assert outcome.reason == "critic + judge both passed"
    assert len(planner_backend.invocations) == 2


def test_plan_rejects_parallel_tier_conflict_and_replans(tmp_path: Path, caplog):
    """APLAN-2: a DAG with two same-tier slices writing the same file is a
    parallel conflict — fed back for re-plan, never adopted."""
    import logging

    corpus = Corpus(tmp_path)
    with caplog.at_level(logging.WARNING, logger="assay.primitives.plan"):
        outcome, *_ = _plan_with_scripts(
            _make_spec(), corpus,
            planner=[json.dumps(_DAG_CONFLICT), json.dumps(_DAG_CONFLICT)],
            critic=[],
            policy=PlanPolicy(max_critique_rounds=1, judge_required=False,
                              breaker_required=False, persistence_veto_threshold=99),
        )
    assert outcome.spec.tier_dag is None  # conflicting DAG never adopted
    assert "parallel-tier file conflict" in caplog.text


def _dag_with_read_only_reference():
    payload = copy.deepcopy(_DAG_OK)
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"] = [{
        "reference_id": "test-existing",
        "path": "tests/test_existing.py",
        "sha256": "sha256:" + "a" * 64,
        "start_line": 11,
        "end_line": 19,
    }]
    return payload


def test_read_only_references_preserve_historical_hashes_and_round_trip(tmp_path):
    # Recorded from untouched Assay 6548947: both canonical and review artifact.
    old = _parse_tier_dag(_DAG_OK)
    assert hash_record(old) == (
        "sha256:f35ddadd871c8b81facf71fdeba25f7997b1187d3d70eccb6b369489e32c265a"
    )
    assert hashlib.sha256(_serialize_tier_dag(old)).hexdigest() == (
        "aaaeefd820505faee8f7d7aa8967d61ed1c6e4e41236f83af85c801203e1879d"
    )
    assert decode_dataclass(parse(serialize(old)), TierDAG) == old
    explicit_empty = copy.deepcopy(_DAG_OK)
    explicit_empty["tiers"][0]["slices"][0]["directive"]["read_only_references"] = []
    assert _serialize_tier_dag(_parse_tier_dag(explicit_empty)) == _serialize_tier_dag(old)
    dag = _parse_tier_dag(_dag_with_read_only_reference())
    reference = dag.tiers[0].slices[0].directive.read_only_references[0]
    assert type(reference) is ReadOnlySourceReference
    assert _parse_tier_dag(json.loads(_serialize_tier_dag(dag))) == dag
    assert hash_record(dag) != hash_record(old)
    corpus = Corpus(tmp_path)
    key = corpus.put_artifact(serialize(dag), kind="plan_candidates")
    assert decode_dataclass(parse(corpus.get_artifact(key)), TierDAG) == dag
    schema = _tier_dag_response_schema()["properties"]["tiers"]["items"][
        "properties"
    ]["slices"]["items"]["properties"]["directive"]
    assert "read_only_references" in schema["required"]
    ref_schema = schema["properties"]["read_only_references"]["items"]
    assert ref_schema["additionalProperties"] is False
    assert set(ref_schema["required"]) == {
        "reference_id", "path", "sha256", "start_line", "end_line",
    }


@pytest.mark.parametrize("mutation", [
    "null", "object", "unknown_key", "missing_key", "boolean_line",
    "string_line", "bad_hash", "unnormalized_path", "duplicate_id",
])
def test_planner_read_only_reference_parse_is_closed(mutation):
    payload = _dag_with_read_only_reference()
    directive = payload["tiers"][0]["slices"][0]["directive"]
    refs = directive["read_only_references"]
    if mutation == "null":
        directive["read_only_references"] = None
    elif mutation == "object":
        directive["read_only_references"] = refs[0]
    elif mutation == "unknown_key":
        refs[0]["write"] = True
    elif mutation == "missing_key":
        del refs[0]["sha256"]
    elif mutation == "boolean_line":
        refs[0]["start_line"] = True
    elif mutation == "string_line":
        refs[0]["start_line"] = "11"
    elif mutation == "bad_hash":
        refs[0]["sha256"] = "a" * 64
    elif mutation == "unnormalized_path":
        refs[0]["path"] = "./tests/test_existing.py"
    elif mutation == "duplicate_id":
        refs.append(dict(refs[0]))
    assert _parse_tier_dag(payload) is None


def test_read_only_references_do_not_normalize_away_authored_source_identity():
    payload = _dag_with_read_only_reference()
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"][0][
        "path"
    ] = "demo/tests/test_existing.py"
    dag = _parse_tier_dag(payload, strip_prefix="demo")
    assert dag.tiers[0].slices[0].directive.read_only_references[0].path == (
        "demo/tests/test_existing.py"
    )


def test_read_only_reference_delta_is_not_hidden_inside_requirement_repair():
    before = _parse_tier_dag(_DAG_OK)
    after = _parse_tier_dag(_dag_with_read_only_reference())
    assert _candidate_repair_changed_fields(before, after) == {"read_only_references"}
    finding = _candidate_repair_delta_finding(
        before=before, after=after, source="requirement_ref_source_gate", findings=(),
    )
    assert finding is not None
    assert "read_only_references" in str(finding)


def test_shared_read_only_references_do_not_expand_parallel_mutation_scope():
    dag = _parse_tier_dag(_dag_with_read_only_reference())
    first = dag.tiers[0].slices[0]
    second = dataclasses.replace(
        first, id="s1", directive=dataclasses.replace(
            first.directive, slice_id="s1", files_in_scope=("different.py",),
        ),
    )
    parallel = dataclasses.replace(dag, tiers=(dataclasses.replace(
        dag.tiers[0], slices=(first, second),
    ),))
    assert parallel.parallel_conflicts() == []
    assert first.directive.files_in_scope == ("src/thing.py",)
    assert first.directive.creates_paths == ()
    assert first.directive.blast_radius == {"max_files": 1, "languages": ()}


@pytest.mark.parametrize("failure", [
    "raise", "not_mapping", "non_string_key", "empty_key", "non_string_value",
    "invalid_utf8_key", "invalid_utf8_value",
    "artifact", "spec_intent", "spec_constraints", "spec_done_criteria",
    "prior_critic_findings", "prior_judge_findings",
])
@pytest.mark.parametrize("fixed", [False, True])
def test_candidate_context_failure_retains_candidate_and_usage_before_reviews(
    tmp_path, failure, fixed,
):
    def capture(candidate):
        assert candidate == _parse_tier_dag(_DAG_OK)
        if failure == "raise":
            raise RuntimeError("source changed")
        if failure == "not_mapping":
            return [("sources", "text")]
        if failure == "non_string_key":
            return {1: "text"}
        if failure == "empty_key":
            return {" ": "text"}
        if failure == "non_string_value":
            return {"sources": {"text": "mutable"}}
        if failure == "invalid_utf8_key":
            return {"\ud800": "source"}
        if failure == "invalid_utf8_value":
            return {"sources": "\ud800"}
        return {failure: "do the thing"}

    critic, judge = ScriptedBackend([]), ScriptedBackend([])
    planner = ScriptedBackend([json.dumps(_DAG_OK)])
    outcome = plan(
        _make_spec(),
        candidate_dag=_parse_tier_dag(_DAG_OK) if fixed else None,
        planner_backend=None if fixed else planner,
        critic_backend=critic, judge_backend=judge,
        corpus=Corpus(tmp_path), critic_rubric=_rubric("critic"),
        judge_rubric=_rubric("judge"), candidate_context_provider=capture,
    )
    assert outcome.decision == "inconclusive"
    assert outcome.status == "candidate_context_failed"
    assert outcome.candidate_dag == _parse_tier_dag(_DAG_OK)
    assert outcome.candidate_ref is not None
    assert outcome.spec.tier_dag is None
    assert outcome.cost_usd == (0.0 if fixed else 0.01)
    assert outcome.usage.cost_usd == (0.0 if fixed else 0.01)
    assert outcome.usage.input_tokens == (0 if fixed else 1)
    assert outcome.usage.output_tokens == (0 if fixed else 2)
    assert len(planner.invocations) == (0 if fixed else 1)
    assert critic.invocations == judge.invocations == []


def test_candidate_context_is_captured_once_and_immutable_between_roles(tmp_path):
    source = {"read_only_sources": "exact verified bytes"}
    captured = []
    calls, delegate = _scripted_typed_review([("pass", ()), ("pass", ())])
    seen_contexts = []

    def capture(candidate):
        captured.append(candidate)
        return source

    def review(artifact, rubric, **kwargs):
        context = kwargs["context"]
        seen_contexts.append(context)
        assert context["read_only_sources"] == "exact verified bytes"
        with pytest.raises(TypeError):
            context["read_only_sources"] = "reviewer mutation"
        source["read_only_sources"] = "external mutation after capture"
        return delegate(artifact, rubric, **kwargs)

    candidate = _parse_tier_dag(_dag_with_read_only_reference())
    outcome = plan(
        _make_spec(), candidate_dag=candidate,
        critic_backend=object(), judge_backend=object(), corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"), judge_rubric=_rubric("judge"),
        candidate_context_provider=capture, review_fn=review,
    )
    assert outcome.accepted
    assert captured == [candidate]
    assert calls == ["critic", "judge"]
    assert seen_contexts[0]["spec_intent"] == seen_contexts[1]["spec_intent"]
    assert seen_contexts[0]["read_only_sources"] == seen_contexts[1]["read_only_sources"]


def test_candidate_context_only_captures_after_deterministic_admission(tmp_path):
    capture_calls = []
    def forbidden(_candidate):
        capture_calls.append(_candidate)
        raise AssertionError("invalid candidate must not acquire source context")

    invalid = copy.deepcopy(_DAG_OK)
    invalid["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = ["invented"]
    outcome = plan(
        _make_spec(), candidate_dag=_parse_tier_dag(invalid),
        critic_backend=object(), judge_backend=object(), corpus=Corpus(tmp_path),
        critic_rubric=_rubric("critic"), judge_rubric=_rubric("judge"),
        candidate_context_provider=forbidden,
    )
    assert outcome.status == "candidate_validation_rejected"
    assert capture_calls == []


def test_candidate_context_is_rendered_and_hash_bound_by_real_review(tmp_path):
    source_text = "def verify_existing_contract():\n    return True\n"
    context_calls = []
    corpus = Corpus(tmp_path)
    critic, judge = ScriptedBackend([_VERDICT_PASS]), ScriptedBackend([_VERDICT_PASS])

    def capture(candidate):
        context_calls.append(candidate)
        return {"read_only_sources": source_text}

    rubric = dataclasses.replace(
        _rubric("critic"), user_template="Review: {artifact}\nSources: {read_only_sources}",
    )
    outcome = plan(
        _make_spec(), candidate_dag=_parse_tier_dag(_dag_with_read_only_reference()),
        critic_backend=critic, judge_backend=judge, corpus=corpus,
        critic_rubric=rubric, judge_rubric=dataclasses.replace(rubric, name="judge"),
        candidate_context_provider=capture,
    )
    assert outcome.accepted
    assert len(context_calls) == 1
    expected_digest = "sha256:" + hashlib.sha256(source_text.encode()).hexdigest()
    for backend in (critic, judge):
        assert len(backend.invocations) == 1
        envelope = json.loads(backend.invocations[0].user.split("\nSources: ", 1)[1])
        assert envelope["source_id"] == "context:read_only_sources"
        assert envelope["text"] == source_text
        assert envelope["sha256"] == expected_digest


def test_noncallable_context_provider_rejects_before_planner_transport(tmp_path):
    planner, critic = ScriptedBackend([]), ScriptedBackend([])
    with pytest.raises(TypeError, match="candidate_context_provider"):
        plan(
            _make_spec(), planner_backend=planner, critic_backend=critic,
            judge_backend=None, corpus=Corpus(tmp_path), critic_rubric=_rubric("critic"),
            candidate_context_provider={"read_only_sources": "not a callback"},
        )
    assert planner.invocations == critic.invocations == []
